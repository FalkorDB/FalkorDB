//! The tag kind: exact, case-sensitive string values, the native replacement for the TAG half of
//! a RediSearch Range field.
//!
//! # Keyed on the string itself
//!
//! The numeric trees key on a `u64`, and a string does not fit in one. Every scheme that squeezes
//! it in — an 8-byte prefix, a hash — answers `n.name = 'Bob'` with a *superset*, and that is not
//! allowed here: `utilize_index` drops the filter for a plain string equality (the plan is a bare
//! `Node By Index Scan`), so nothing downstream would reject the extra rows.
//!
//! So the tag trees key on the string itself: a [`StrTree`] of `(string, doc)` tuples, ordered by
//! the string's bytes. Equality is one point lookup and exact; a string range is one cursor over
//! one run of the tree, in string order. There is no dictionary, so nothing interns a string or
//! hands out an id, and a string whose last tuple goes leaves nothing behind.
//!
//! # Versions
//!
//! The trees are copy-on-write like the numeric ones: `Clone` is a root clone, a version's fork
//! copies only the path to what it changes, and a reader reads its own version with no lock.
//! Tuples moving between versions — a unit of work's staged changes, an indexer batch read on one
//! version and added to another — carry the string ([`ColumnKey::Tag`],
//! [`EncodedTuples::tags`]), the same key every version stores.

use std::cmp::{self, Ordering};
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use super::data_structures::cow_btree::{StrBounds, StrRanges, StrTree};
use super::encode::{ColumnKey, KeyKind};
use super::numeric::{DocIter, EncodedTuples};
use super::range::RangeKind;
use crate::runtime::value::Value;

/// Which strings a tag predicate selects: a set of them, or the strings between two bounds.
///
/// A built column reads it as key ranges of its trees ([`ranges`](Self::ranges)); a column still
/// building answers with a scan, which matches it against each value ([`contains`](Self::contains)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrSel {
    /// These strings, sorted and distinct.
    Set(Vec<Arc<String>>),
    /// Every string between these bounds, in byte (codepoint) order.
    Window(StrBounds),
    /// Every string any member selects: a union with a window in it (`v = 'a' OR v > 'x'`). The
    /// members are never themselves unions.
    Any(Vec<StrSel>),
}

impl StrSel {
    /// The strings `strings`, sorted and distinct.
    #[must_use]
    pub fn set(mut strings: Vec<Arc<String>>) -> Self {
        strings.sort_unstable();
        strings.dedup();
        Self::Set(strings)
    }

    /// Selects no string.
    #[must_use]
    pub const fn none() -> Self {
        Self::Set(Vec::new())
    }

    /// The strings from `min` to `max`, each end included or not; `None` leaves that side open.
    #[must_use]
    pub fn window(
        min: Option<Arc<String>>,
        max: Option<Arc<String>>,
        include_min: bool,
        include_max: bool,
    ) -> Self {
        let bound = |v: Option<Arc<String>>, inclusive: bool| match v {
            None => Bound::Unbounded,
            Some(v) if inclusive => Bound::Included(v),
            Some(v) => Bound::Excluded(v),
        };
        Self::Window((bound(min, include_min), bound(max, include_max)))
    }

    /// Whether this selects no string by construction: an empty set, or a window no string lies
    /// in.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Set(strings) => strings.is_empty(),
            Self::Window(window) => is_empty_window(window),
            Self::Any(members) => members.iter().all(Self::is_empty),
        }
    }

    /// The strings either selects: a union of string predicates on one attribute. Two sets stay a
    /// set; anything with a window in it is an [`Any`](Self::Any), flattened.
    #[must_use]
    pub fn or(
        self,
        other: Self,
    ) -> Self {
        match (self, other) {
            (a, b) if b.is_empty() => a,
            (a, b) if a.is_empty() => b,
            (Self::Set(mut a), Self::Set(b)) => {
                a.extend(b);
                Self::set(a)
            }
            (a, b) => {
                let mut members = Vec::new();
                for side in [a, b] {
                    match side {
                        Self::Any(inner) => members.extend(inner),
                        other => members.push(other),
                    }
                }
                Self::Any(members)
            }
        }
    }

    /// The key ranges of the strings this selects, in order and disjoint: a point per string of a
    /// set, the window itself, or none for a window no string lies in.
    #[must_use]
    pub fn ranges(&self) -> Vec<StrBounds> {
        match self {
            Self::Set(strings) => strings
                .iter()
                .map(|s| (Bound::Included(s.clone()), Bound::Included(s.clone())))
                .collect(),
            // Members may overlap (`v = 'a' OR v >= 'a'`): their ranges are merged, so no doc is
            // read twice.
            Self::Any(members) => merge_ranges(members.iter().flat_map(Self::ranges).collect()),
            Self::Window(window) if is_empty_window(window) => Vec::new(),
            Self::Window(window) => vec![window.clone()],
        }
    }

    /// Whether `s` is selected.
    #[must_use]
    pub fn contains(
        &self,
        s: &Arc<String>,
    ) -> bool {
        match self {
            Self::Set(strings) => strings.binary_search(s).is_ok(),
            Self::Window(window) => window.contains(s),
            Self::Any(members) => members.iter().any(|member| member.contains(s)),
        }
    }

    /// The strings both select: a conjunction of string predicates on one attribute.
    #[must_use]
    pub fn intersect(
        self,
        other: Self,
    ) -> Self {
        match (self, other) {
            (Self::Any(members), b) | (b, Self::Any(members)) => Self::Any(
                members
                    .into_iter()
                    .map(|member| member.intersect(b.clone()))
                    .collect(),
            ),
            (Self::Set(a), b) | (b, Self::Set(a)) => {
                Self::Set(a.into_iter().filter(|s| b.contains(s)).collect())
            }
            (Self::Window((alo, ahi)), Self::Window((blo, bhi))) => {
                Self::Window((cmp::max_by(alo, blo, cmp_lo), cmp::min_by(ahi, bhi, cmp_hi)))
            }
        }
    }
}

/// Whether no string lies in `window`. An inverted window is not a valid range for a cursor, so
/// it is answered before one is built.
fn is_empty_window((lo, hi): &StrBounds) -> bool {
    match (lo, hi) {
        (Bound::Included(lo), Bound::Included(hi)) => lo > hi,
        (Bound::Included(lo) | Bound::Excluded(lo), Bound::Included(hi) | Bound::Excluded(hi)) => {
            lo >= hi
        }
        _ => false,
    }
}

/// Lower ends in the order they start: from the very start, then at a key, then just past it.
fn cmp_lo(
    a: &Bound<Arc<String>>,
    b: &Bound<Arc<String>>,
) -> Ordering {
    match (a, b) {
        (Bound::Unbounded, Bound::Unbounded) => Ordering::Equal,
        (Bound::Unbounded, _) => Ordering::Less,
        (_, Bound::Unbounded) => Ordering::Greater,
        (Bound::Included(x) | Bound::Excluded(x), Bound::Included(y) | Bound::Excluded(y)) => x
            .cmp(y)
            .then(matches!(a, Bound::Excluded(_)).cmp(&matches!(b, Bound::Excluded(_)))),
    }
}

/// Upper ends in the order they end: just before a key, then at it, then at the very end.
fn cmp_hi(
    a: &Bound<Arc<String>>,
    b: &Bound<Arc<String>>,
) -> Ordering {
    match (a, b) {
        (Bound::Unbounded, Bound::Unbounded) => Ordering::Equal,
        (Bound::Unbounded, _) => Ordering::Greater,
        (_, Bound::Unbounded) => Ordering::Less,
        (Bound::Included(x) | Bound::Excluded(x), Bound::Included(y) | Bound::Excluded(y)) => x
            .cmp(y)
            .then(matches!(a, Bound::Included(_)).cmp(&matches!(b, Bound::Included(_)))),
    }
}

/// `ranges` as disjoint ranges in key order, each key in exactly the ranges it was in before.
/// Ranges that overlap or meet with no key between them are joined; two that both exclude the key
/// between them (`< 'x'` and `> 'x'`) are not, since `x` is in neither.
fn merge_ranges(mut ranges: Vec<StrBounds>) -> Vec<StrBounds> {
    // Whether a range ending at `hi` and one starting at `lo` leave no key between them.
    let joins = |hi: &Bound<Arc<String>>, lo: &Bound<Arc<String>>| match (hi, lo) {
        (Bound::Unbounded, _) | (_, Bound::Unbounded) => true,
        (Bound::Excluded(x), Bound::Excluded(y)) => y < x,
        (Bound::Included(x) | Bound::Excluded(x), Bound::Included(y) | Bound::Excluded(y)) => {
            y <= x
        }
    };
    ranges.sort_by(|a, b| cmp_lo(&a.0, &b.0));
    let mut out: Vec<StrBounds> = Vec::with_capacity(ranges.len());
    for (lo, hi) in ranges {
        match out.last_mut() {
            Some((_, last_hi)) if joins(last_hi, &lo) => {
                if cmp_hi(&hi, last_hi) == Ordering::Greater {
                    *last_hi = hi;
                }
            }
            _ => out.push((lo, hi)),
        }
    }
    out
}

/// The tag kind of one `(label, attribute)` column: entity ids keyed by their string value.
///
/// `Clone` is `O(1)`: a root clone per tree.
#[derive(Clone, Default)]
pub struct TagIndex {
    /// String scalar values: one tuple per indexed entity.
    tree: StrTree,
    /// String elements of **list**-valued properties: one tuple per distinct element.
    ///
    /// A separate key space for the same reason the numeric kind keeps one: sharing would make
    /// `WHERE n.v = 'a'` match a node whose `v` is `['a', 'b']`, which is false in Cypher, and
    /// `Equal` has no post-filter to catch it. RediSearch keeps the same split with its
    /// `string:arr` sub-field.
    array_tree: StrTree,
}

impl TagIndex {
    /// The kinds of key this kind's trees hold, one tree each.
    const TREES: [KeyKind; 2] = [KeyKind::Tag, KeyKind::TagArray];

    /// The tree holding keys of `kind`.
    fn tree(
        &self,
        kind: KeyKind,
    ) -> &StrTree {
        match kind {
            KeyKind::Tag => &self.tree,
            KeyKind::TagArray => &self.array_tree,
            KeyKind::Scalar | KeyKind::Array | KeyKind::Geo => {
                unreachable!("the tag kind holds only tag trees")
            }
        }
    }

    /// The tree holding keys of `kind`, to change.
    fn tree_mut(
        &mut self,
        kind: KeyKind,
    ) -> &mut StrTree {
        match kind {
            KeyKind::Tag => &mut self.tree,
            KeyKind::TagArray => &mut self.array_tree,
            KeyKind::Scalar | KeyKind::Array | KeyKind::Geo => {
                unreachable!("the tag kind holds only tag trees")
            }
        }
    }

    /// The ids stored under the string `s` in the tree of `kind`.
    #[must_use]
    pub fn rows_at(
        &self,
        kind: KeyKind,
        s: &Arc<String>,
    ) -> DocIter {
        DocIter::Strings(Box::new(StrRanges::new(
            self.tree(kind),
            vec![(Bound::Included(s.clone()), Bound::Included(s.clone()))],
        )))
    }

    /// Every `(kind, string, doc)` tuple the kind holds, the scalar tree's then the array tree's.
    /// For tests that compare a column with what it must hold.
    #[cfg(test)]
    #[must_use]
    pub fn tuples(&self) -> Vec<(KeyKind, String, u64)> {
        Self::TREES
            .into_iter()
            .flat_map(|kind| {
                self.tree(kind).tuples().into_iter().map(move |(key, doc)| {
                    (kind, String::from_utf8(key).expect("keys are strings"), doc)
                })
            })
            .collect()
    }
}

/// `tuples` sorted and distinct, as the tree takes them. The strings are shared, not copied.
fn sorted_keys(tuples: &[(Arc<String>, u64)]) -> Vec<(Arc<String>, u64)> {
    let mut keys = tuples.to_vec();
    keys.sort_unstable();
    keys.dedup();
    keys
}

impl RangeKind for TagIndex {
    type Keys = StrSel;
    type Element = Arc<String>;

    /// A string goes in the tag tree, and a list's string elements in the tag array tree, each
    /// keyed on the string itself. Not deduplicated, as
    /// [`for_each_stored_key`](super::encode::for_each_stored_key) is not.
    fn for_each_stored_key(
        &self,
        value: &Value,
        mut emit: impl FnMut(KeyKind, ColumnKey),
    ) {
        match value {
            Value::String(s) => emit(KeyKind::Tag, ColumnKey::Tag(s.clone())),
            Value::List(items) => {
                for item in items.iter() {
                    if let Value::String(s) = item {
                        emit(KeyKind::TagArray, ColumnKey::Tag(s.clone()));
                    }
                }
            }
            _ => {}
        }
    }

    fn add_encoded(
        &mut self,
        tuples: &EncodedTuples,
    ) {
        for kind in Self::TREES {
            if !tuples.tags(kind).is_empty() {
                self.tree_mut(kind)
                    .insert_batch(&sorted_keys(tuples.tags(kind)));
            }
        }
    }

    fn remove_encoded(
        &mut self,
        tuples: &EncodedTuples,
    ) {
        for kind in Self::TREES {
            if !tuples.tags(kind).is_empty() {
                self.tree_mut(kind)
                    .remove_batch(&sorted_keys(tuples.tags(kind)));
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.tree.is_empty() && self.array_tree.is_empty()
    }

    /// Heap bytes: both trees' nodes.
    fn memory_usage(&self) -> usize {
        self.tree.heap_bytes() + self.array_tree.heap_bytes()
    }

    /// One cursor per range the selection names, chained lazily over one snapshot: a point per
    /// string of a set, one run for a window.
    fn scalars(
        &self,
        strings: StrSel,
    ) -> DocIter {
        DocIter::Strings(Box::new(StrRanges::new(&self.tree, strings.ranges())))
    }

    /// A point lookup into the array tree.
    fn lists_holding(
        &self,
        element: &Arc<String>,
    ) -> DocIter {
        self.rows_at(KeyKind::TagArray, element)
    }

    fn is_scalar_in(
        strings: &StrSel,
        value: &Value,
    ) -> bool {
        matches!(value, Value::String(s) if strings.contains(s))
    }

    fn is_list_holding(
        element: &Arc<String>,
        value: &Value,
    ) -> bool {
        matches!(
            value,
            Value::List(items) if items.iter().any(|v| matches!(v, Value::String(s) if s == element))
        )
    }
}

/// The tag kind, tested through the column that routes to it ([`RangeIndex`]): what these tests
/// exercise is what a query runs.
#[cfg(test)]
mod tests {
    use super::super::range::RangeIndex;
    use super::*;
    use crate::index::IndexQuery;
    use thin_vec::ThinVec;

    fn s(x: &str) -> Arc<String> {
        Arc::new(x.to_string())
    }

    fn val(x: &str) -> Value {
        Value::String(s(x))
    }

    fn list(xs: &[&str]) -> Value {
        Value::List(Arc::new(xs.iter().map(|x| val(x)).collect::<ThinVec<_>>()))
    }

    fn attr() -> Arc<String> {
        s("v")
    }

    fn sorted(it: DocIter) -> Vec<u64> {
        let mut v: Vec<u64> = it.collect();
        v.sort_unstable();
        v
    }

    fn point(
        idx: &RangeIndex,
        x: &str,
    ) -> Vec<u64> {
        sorted(
            idx.query(&IndexQuery::Equal {
                key: attr(),
                value: val(x),
            })
            .expect("a string equality is served"),
        )
    }

    fn range(
        idx: &RangeIndex,
        min: Option<&str>,
        max: Option<&str>,
        include_min: bool,
        include_max: bool,
    ) -> Vec<u64> {
        sorted(
            idx.query(&IndexQuery::Range {
                key: attr(),
                min: min.map(val),
                max: max.map(val),
                include_min,
                include_max,
            })
            .expect("a string range is served"),
        )
    }

    fn holding(
        idx: &RangeIndex,
        x: &str,
    ) -> Vec<u64> {
        sorted(
            idx.query(&IndexQuery::ArrayContains {
                key: attr(),
                value: val(x),
            })
            .expect("a string probe is served"),
        )
    }

    fn sample() -> RangeIndex {
        let mut idx = RangeIndex::new();
        idx.add(&val("bob"), 1);
        idx.add(&val("alice"), 2);
        idx.add(&val("Bob"), 3); // distinct from "bob" — TAG is case-sensitive
        idx.add(&val("carol"), 4);
        idx
    }

    /// Equality is exact, and — unlike the RediSearch TAG tokenizer's default — case-sensitive,
    /// matching the `TagFieldSetCaseSensitive(1)` the RediSearch path sets.
    #[test]
    fn equality_is_exact_and_case_sensitive() {
        let idx = sample();
        assert_eq!(point(&idx, "bob"), vec![1]);
        assert_eq!(point(&idx, "Bob"), vec![3]);
        assert!(point(&idx, "BOB").is_empty());
        assert!(point(&idx, "bo").is_empty(), "no prefix matching");
        assert!(
            point(&idx, "never held").is_empty(),
            "a string the column never held answers empty without touching the tree"
        );
    }

    /// Bytes that RediSearch's tag tokenizer would split on or strip — the separator, whitespace,
    /// backslashes, the escape prefix — are just bytes to a byte-ordered key, so they need no
    /// encoding and cannot collide. `test06_tag_separator` and `test_25_unescaped_string` are the flow
    /// tests this covers.
    #[test]
    fn tokenizer_hostile_strings_are_ordinary_keys() {
        let mut idx = RangeIndex::new();
        let hostile = ["a b", "a\\b", "a_b", "a\u{1}b", "a,b", "", " ", "a|b"];
        for (i, h) in hostile.iter().enumerate() {
            idx.add(&val(h), i as u64);
        }
        for (i, h) in hostile.iter().enumerate() {
            assert_eq!(point(&idx, h), vec![i as u64], "{h:?} must round-trip");
        }
    }

    #[test]
    fn lexicographic_range_bounds() {
        let idx = sample();
        let all = |min, max, inc_min, inc_max| range(&idx, min, max, inc_min, inc_max);
        // Capitals sort below lowercase in codepoint order: "Bob" < "alice" < "bob" < "carol".
        // `''` is the least string: a range with no bound at all is not one Cypher writes.
        assert_eq!(all(Some(""), None, true, true), vec![1, 2, 3, 4]);
        assert_eq!(all(Some("alice"), None, true, true), vec![1, 2, 4]);
        assert_eq!(all(Some("alice"), None, false, true), vec![1, 4]);
        assert_eq!(all(None, Some("bob"), true, true), vec![1, 2, 3]);
        assert_eq!(all(None, Some("bob"), true, false), vec![2, 3]);
        assert_eq!(all(Some("alice"), Some("carol"), true, false), vec![1, 2]);
        // An inverted or empty window is an answer, not a panic.
        assert!(all(Some("z"), Some("a"), true, true).is_empty());
        assert!(all(Some("a"), Some("a"), false, false).is_empty());
    }

    /// A window and a set select by value, so a scan can match a string no batch has added yet.
    #[test]
    fn a_selection_matches_by_value_and_intersects() {
        let window = |min: &str, max: &str| StrSel::window(Some(s(min)), Some(s(max)), true, false);
        assert!(window("a", "b").contains(&s("apple")));
        assert!(!window("a", "b").contains(&s("b")));
        let both = window("a", "c").intersect(window("b", "d"));
        assert!(both.contains(&s("bob")) && !both.contains(&s("apple")) && !both.contains(&s("c")));
        let set = StrSel::set(vec![s("bob"), s("apple"), s("bob")]);
        assert_eq!(set, StrSel::Set(vec![s("apple"), s("bob")]));
        assert_eq!(set.intersect(window("b", "c")), StrSel::Set(vec![s("bob")]));
        assert!(StrSel::none().is_empty() && window("b", "a").is_empty());
        assert!(!window("a", "b").is_empty());
        // On a tie the exclusive end is the tighter one, on either side.
        let open = StrSel::window(Some(s("a")), Some(s("c")), false, true);
        assert_eq!(
            window("a", "c").intersect(open),
            StrSel::window(Some(s("a")), Some(s("c")), false, false)
        );
    }

    /// A string whose last tuple goes leaves nothing behind: no entry, no empty cursor in a
    /// range, and adding it again is like adding it the first time.
    #[test]
    fn a_removed_string_leaves_nothing_behind() {
        let mut idx = sample();
        idx.remove(&val("bob"), 1);
        assert!(point(&idx, "bob").is_empty());
        assert!(
            !idx.tag().tuples().iter().any(|(_, s, _)| s == "bob"),
            "no tuple for bob remains"
        );
        assert_eq!(range(&idx, Some(""), None, true, true), vec![2, 3, 4]);
        idx.add(&val("bob"), 9);
        assert_eq!(point(&idx, "bob"), vec![9]);
    }

    /// The scalar/array split, and the reason for it: `n.v = 'a'` must not match `['a','b']`, and
    /// `'a' IN n.v` must not match the scalar `'a'`.
    #[test]
    fn a_scalar_and_a_list_element_do_not_share_a_key_space() {
        let mut idx = RangeIndex::new();
        idx.add(&val("a"), 10);
        idx.add(&list(&["a", "b"]), 20);

        assert_eq!(point(&idx, "a"), vec![10]);
        assert_eq!(holding(&idx, "a"), vec![20]);
        assert_eq!(holding(&idx, "b"), vec![20]);
        assert!(holding(&idx, "c").is_empty());
        // A range over the scalar tree must not see list elements either.
        assert_eq!(range(&idx, Some(""), None, true, true), vec![10]);
    }

    /// A repeated element contributes one tuple, so removing the value takes everything with it.
    #[test]
    fn duplicate_elements_contribute_one_tuple() {
        let mut idx = RangeIndex::new();
        idx.add(&list(&["a", "a", "b"]), 7);
        assert_eq!(holding(&idx, "a"), vec![7]);
        idx.remove(&list(&["a", "a", "b"]), 7);
        assert!(idx.is_empty(), "no tuple stranded in either tree");
    }

    /// The dangerous re-index case: the same string crossing between the two key spaces. A remove
    /// routed on the *new* value's kind would delete from the wrong tree and strand a tuple that
    /// no later write can reach — and on the scalar side nothing re-checks the index's answer.
    #[test]
    fn a_value_changing_kind_moves_between_trees() {
        let mut idx = RangeIndex::new();
        idx.add(&val("a"), 1);

        idx.remove(&val("a"), 1);
        idx.add(&list(&["a"]), 1);
        assert!(point(&idx, "a").is_empty());
        assert_eq!(holding(&idx, "a"), vec![1]);

        idx.remove(&list(&["a"]), 1);
        idx.add(&val("a"), 1);
        assert_eq!(point(&idx, "a"), vec![1]);
        assert!(holding(&idx, "a").is_empty());

        idx.remove(&val("a"), 1);
        assert!(idx.is_empty());
    }

    /// Tuples encoded on one copy go into another by string: a batch read on an older version,
    /// as `CatchUp::Insert` adds one. Each copy's own changes stay its own.
    #[test]
    fn tuples_move_between_copies_by_string() {
        let mut live = RangeIndex::new();
        live.add(&val("a"), 1);
        let mut batch = live.clone();
        batch.add(&val("b"), 2);
        live.add(&val("c"), 3);
        let mut tuples = EncodedTuples::default();
        batch.push_encoded(&val("b"), 100, &mut tuples);
        batch.push_encoded(&val("a"), 101, &mut tuples);
        batch.push_encoded(&list(&["b", "d"]), 102, &mut tuples);
        live.add_encoded(&mut tuples);
        assert_eq!(point(&live, "a"), vec![1, 101]);
        assert_eq!(point(&live, "b"), vec![100]);
        assert_eq!(point(&live, "c"), vec![3]);
        assert_eq!(holding(&live, "d"), vec![102]);
        // And the copy the tuples were read on is as it was.
        assert_eq!(point(&batch, "b"), vec![2]);
        assert!(point(&batch, "c").is_empty());
    }

    /// Removing a string the copy never held changes nothing.
    #[test]
    fn removing_an_unknown_string_changes_nothing() {
        let mut idx = sample();
        let before = idx.tag().tuples();
        idx.remove(&val("never held"), 1);
        assert_eq!(idx.tag().tuples(), before);
    }

    /// A string range is one run of the tree, in string order, with no cursor per string.
    #[test]
    fn a_range_reads_one_run_in_string_order() {
        let mut idx = RangeIndex::new();
        for (i, s) in ["delta", "alpha", "charlie", "bravo", "echo"]
            .iter()
            .enumerate()
        {
            idx.add(&val(s), i as u64);
        }
        let docs: Vec<u64> = idx
            .query(&IndexQuery::Range {
                key: attr(),
                min: Some(val("b")),
                max: Some(val("d")),
                include_min: true,
                include_max: false,
            })
            .expect("a string range is served")
            .collect();
        assert_eq!(docs, vec![3, 2], "bravo, then charlie");
    }

    /// A union of string selections reads each doc once, however its members overlap, and keeps
    /// out a key both of two excluding members leave out.
    #[test]
    fn a_union_of_strings_reads_each_doc_once() {
        let mut idx = RangeIndex::new();
        for (i, x) in ["a", "b", "c", "x", "y"].iter().enumerate() {
            idx.add(&val(x), i as u64);
        }
        let eq = |x: &str| IndexQuery::Equal {
            key: attr(),
            value: val(x),
        };
        let cmp = |min: Option<&str>, max: Option<&str>, inc: bool| IndexQuery::Range {
            key: attr(),
            min: min.map(val),
            max: max.map(val),
            include_min: inc,
            include_max: inc,
        };
        for (q, want) in [
            (
                IndexQuery::Or(vec![eq("a"), cmp(Some("a"), None, true)]),
                vec![0, 1, 2, 3, 4],
            ),
            (
                IndexQuery::Or(vec![
                    cmp(None, Some("x"), false),
                    cmp(Some("x"), None, false),
                ]),
                vec![0, 1, 2, 4],
            ),
            (
                IndexQuery::Or(vec![
                    cmp(Some("a"), Some("b"), true),
                    cmp(Some("b"), Some("c"), true),
                    eq("y"),
                ]),
                vec![0, 1, 2, 4],
            ),
        ] {
            let docs: Vec<u64> = idx
                .query(&q)
                .expect("a union of strings is served")
                .collect();
            let mut sorted = docs.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(docs.len(), sorted.len(), "a doc read twice: {q:?}");
            assert_eq!(sorted, want, "{q:?}");
        }
    }
}
