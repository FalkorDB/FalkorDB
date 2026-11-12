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
//! So the tag trees key on the string itself: a [`StrBTree`] of `(string, doc)` tuples, ordered by
//! the string's bytes. Equality is one point lookup and exact; a string range is one cursor over
//! one run of the tree, in string order. There is no dictionary, so nothing interns a string or
//! hands out an id, and a string whose last tuple goes leaves nothing behind.
//!
//! # Versions
//!
//! The trees are copy-on-write like the numeric ones: `Clone` is a root clone, a version's fork
//! copies only the path to what it changes, and a reader reads its own version with no lock.
//! Tuples moving between versions — a unit of work's staged changes, an indexer batch read on one
//! version and added to another — carry the string ([`StoredKey::Tag`],
//! [`EncodedTuples::tags`]), the same key every version stores.

use std::ops::Bound;
use std::sync::Arc;

use super::data_structures::str_btree::{StrBTree, StrRanges};
use super::encode::{KeyKind, StoredKey};
use super::numeric::{DocIter, EncodedTuples};
use super::range::RangeKind;
use crate::runtime::value::Value;

/// Which strings a tag predicate selects: a set of them, or a lexicographic window.
///
/// A built column reads it as key ranges of its trees ([`ranges`](Self::ranges)); a column still
/// building answers with a scan, which matches it against each value ([`contains`](Self::contains)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrSel {
    /// These strings, sorted and distinct.
    Set(Vec<Arc<String>>),
    /// Every string in a lexicographic window; `None` leaves that side open.
    Window {
        min: Option<Arc<String>>,
        max: Option<Arc<String>>,
        include_min: bool,
        include_max: bool,
    },
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

    /// Whether this selects no string by construction: an empty set, or a window no string lies
    /// in.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Set(strings) if strings.is_empty()) || self.is_empty_window()
    }

    /// The key ranges of the strings this selects, in order and disjoint: a point per string of a
    /// set, the window itself, or none for a window no string lies in.
    #[must_use]
    pub fn ranges(&self) -> Vec<(Bound<Arc<[u8]>>, Bound<Arc<[u8]>>)> {
        let key = |s: &Arc<String>| -> Arc<[u8]> { Arc::from(s.as_bytes()) };
        match self {
            Self::Set(strings) => strings
                .iter()
                .map(|s| (Bound::Included(key(s)), Bound::Included(key(s))))
                .collect(),
            Self::Window { .. } if self.is_empty_window() => Vec::new(),
            Self::Window {
                min,
                max,
                include_min,
                include_max,
            } => {
                let bound = |v: &Option<Arc<String>>, inclusive: bool| match v {
                    None => Bound::Unbounded,
                    Some(v) if inclusive => Bound::Included(key(v)),
                    Some(v) => Bound::Excluded(key(v)),
                };
                vec![(bound(min, *include_min), bound(max, *include_max))]
            }
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
            Self::Window {
                min,
                max,
                include_min,
                include_max,
            } => {
                let above = min
                    .as_ref()
                    .is_none_or(|m| if *include_min { s >= m } else { s > m });
                let below = max
                    .as_ref()
                    .is_none_or(|m| if *include_max { s <= m } else { s < m });
                above && below
            }
        }
    }

    /// Whether this is a window no string lies in. An inverted window is not a valid
    /// `BTreeMap::range` argument — it panics rather than yielding nothing — so it is answered
    /// before one is built.
    fn is_empty_window(&self) -> bool {
        match self {
            Self::Window {
                min: Some(lo),
                max: Some(hi),
                include_min,
                include_max,
            } => lo > hi || (lo == hi && !(*include_min && *include_max)),
            _ => false,
        }
    }

    /// The strings both select: a conjunction of string predicates on one attribute.
    #[must_use]
    pub fn intersect(
        self,
        other: Self,
    ) -> Self {
        match (self, other) {
            (Self::Set(a), b) | (b, Self::Set(a)) => {
                Self::Set(a.into_iter().filter(|s| b.contains(s)).collect())
            }
            (
                Self::Window {
                    min: amin,
                    max: amax,
                    include_min: ainc_min,
                    include_max: ainc_max,
                },
                Self::Window {
                    min: bmin,
                    max: bmax,
                    include_min: binc_min,
                    include_max: binc_max,
                },
            ) => {
                // The tighter of two bounds; on a tie the exclusive one is tighter.
                let (min, include_min) = match (amin, bmin) {
                    (None, b) => (b, binc_min),
                    (a, None) => (a, ainc_min),
                    (Some(a), Some(b)) if a == b => (Some(a), ainc_min && binc_min),
                    (Some(a), Some(b)) if a > b => (Some(a), ainc_min),
                    (_, b) => (b, binc_min),
                };
                let (max, include_max) = match (amax, bmax) {
                    (None, b) => (b, binc_max),
                    (a, None) => (a, ainc_max),
                    (Some(a), Some(b)) if a == b => (Some(a), ainc_max && binc_max),
                    (Some(a), Some(b)) if a < b => (Some(a), ainc_max),
                    (_, b) => (b, binc_max),
                };
                Self::Window {
                    min,
                    max,
                    include_min,
                    include_max,
                }
            }
        }
    }
}

/// The tag kind of one `(label, attribute)` column: entity ids keyed by their string value.
///
/// `Clone` is `O(1)`: a root clone per tree.
#[derive(Clone, Default)]
pub struct TagIndex {
    /// String scalar values: one tuple per indexed entity.
    tree: StrBTree,
    /// String elements of **list**-valued properties: one tuple per distinct element.
    ///
    /// A separate key space for the same reason the numeric kind keeps one: sharing would make
    /// `WHERE n.v = 'a'` match a node whose `v` is `['a', 'b']`, which is false in Cypher, and
    /// `Equal` has no post-filter to catch it. RediSearch keeps the same split with its
    /// `string:arr` sub-field.
    array_tree: StrBTree,
}

impl TagIndex {
    /// The kinds of key this kind's trees hold, one tree each.
    const TREES: [KeyKind; 2] = [KeyKind::Tag, KeyKind::TagArray];

    /// The tree holding keys of `kind`.
    fn tree(
        &self,
        kind: KeyKind,
    ) -> &StrBTree {
        match kind {
            KeyKind::Tag => &self.tree,
            KeyKind::TagArray => &self.array_tree,
            KeyKind::Scalar | KeyKind::Array => {
                unreachable!("the tag kind holds only tag trees")
            }
        }
    }

    /// The tree holding keys of `kind`, to change.
    fn tree_mut(
        &mut self,
        kind: KeyKind,
    ) -> &mut StrBTree {
        match kind {
            KeyKind::Tag => &mut self.tree,
            KeyKind::TagArray => &mut self.array_tree,
            KeyKind::Scalar | KeyKind::Array => {
                unreachable!("the tag kind holds only tag trees")
            }
        }
    }

    /// The ids stored under the string `s` in the tree of `kind`.
    #[must_use]
    pub fn rows_at(
        &self,
        kind: KeyKind,
        s: &str,
    ) -> DocIter {
        DocIter::Strings(StrRanges::new(
            self.tree(kind),
            vec![(
                Bound::Included(Arc::from(s.as_bytes())),
                Bound::Included(Arc::from(s.as_bytes())),
            )],
        ))
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

/// `tuples`' `(string, doc)` pairs as tree keys, sorted and distinct.
fn sorted_keys(tuples: &[(Arc<String>, u64)]) -> Vec<(&[u8], u64)> {
    let mut keys: Vec<(&[u8], u64)> = tuples.iter().map(|(s, d)| (s.as_bytes(), *d)).collect();
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
    fn for_each_stored_key<'v>(
        &self,
        value: &'v Value,
        mut emit: impl FnMut(KeyKind, StoredKey<'v>),
    ) {
        match value {
            Value::String(s) => emit(KeyKind::Tag, StoredKey::Tag(s)),
            Value::List(items) => {
                for item in items.iter() {
                    if let Value::String(s) = item {
                        emit(KeyKind::TagArray, StoredKey::Tag(s));
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
        DocIter::Strings(StrRanges::new(&self.tree, strings.ranges()))
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
        let window = |min: &str, max: &str| StrSel::Window {
            min: Some(s(min)),
            max: Some(s(max)),
            include_min: true,
            include_max: false,
        };
        assert!(window("a", "b").contains(&s("apple")));
        assert!(!window("a", "b").contains(&s("b")));
        let both = window("a", "c").intersect(window("b", "d"));
        assert!(both.contains(&s("bob")) && !both.contains(&s("apple")) && !both.contains(&s("c")));
        let set = StrSel::set(vec![s("bob"), s("apple"), s("bob")]);
        assert_eq!(set, StrSel::Set(vec![s("apple"), s("bob")]));
        assert_eq!(set.intersect(window("b", "c")), StrSel::Set(vec![s("bob")]));
        assert!(StrSel::none().is_empty() && window("b", "a").is_empty());
        assert!(!window("a", "b").is_empty());
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
}
