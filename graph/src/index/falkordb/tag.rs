//! The tag kind: exact, case-sensitive string values, the native replacement for the TAG half of
//! a RediSearch Range field.
//!
//! # Why a dictionary and not an encoded key
//!
//! The tuple tree keys on a `u64`, and a string does not fit in one. Every scheme that squeezes it
//! in — an 8-byte prefix, a hash — answers `n.name = 'Bob'` with a *superset*, and that is not
//! allowed here: `utilize_index` drops the filter for a plain string equality (the plan is a bare
//! `Node By Index Scan`), so nothing downstream would reject the extra rows.
//!
//! So the string is not encoded, it is **interned**: a per-column [`TagDict`] assigns each distinct
//! string a dense id, and the tree holds `(tag id, doc)`. Equality is exact because the id is
//! exact. Lexicographic order lives in the dictionary rather than in the key, so a string range
//! resolves in two steps — the dictionary names the ids inside the window, and the tree is scanned
//! once per id.
//!
//! # Why the dictionary is shared and append-only
//!
//! The dictionary sits behind an `Arc`, shared by every version of the column, and entries are
//! never removed. Both properties are load-bearing:
//!
//! * **Shared** — so `Clone` stays `O(1)` and the MVCC fork on `Graph::new_version()` does not copy
//!   it. It also makes the background build work: BASE is encoded off the write thread against the
//!   same dictionary the live column uses, so the ids in BASE, DELTA and TOMB are the same ids, and
//!   `install_base` can merge raw `(key, doc)` tuples exactly as the numeric kind does. Two
//!   independent dictionaries would each number their strings from zero and the merge would be
//!   garbage.
//! * **Append-only** — so an id, once handed out, means the same string forever. Reclaiming an id
//!   whose last tuple went away would corrupt any older snapshot still holding that id (a reader on
//!   that version would resolve it to whatever string was interned next). What it costs is a
//!   dictionary entry per string the column has *ever* seen; the entry is one pointer plus the
//!   string, and a dropped column releases the lot.
//!
//! Mutating shared state from a version fork is safe here because the dictionary is a *name table*,
//! not index content: an id an older snapshot never stored simply matches no tuples in that
//! snapshot's tree, which is the right answer.

use std::collections::BTreeMap;
use std::ops::Bound;
use std::sync::Arc;

use parking_lot::RwLock;

use super::encode::KeyKind;
use super::numeric::{DocIter, EncodedTuples, KeySet, Tree, add_sorted, docs_in, remove_sorted};
use super::range::RangeKind;
use crate::runtime::value::Value;

/// The strings one column has interned, and the id each was given.
///
/// Cloning shares the dictionary (an `Arc` bump) rather than copying it — see the module docs for
/// why every version of a column must see the same numbering.
#[derive(Clone, Default)]
pub struct TagDict(Arc<RwLock<DictInner>>);

#[derive(Default)]
struct DictInner {
    /// Ordered by the string, so a lexicographic window is a `BTreeMap::range`. `String`'s `Ord` is
    /// byte order over UTF-8, which is codepoint order — the order Cypher compares strings in.
    ids: BTreeMap<Arc<String>, u64>,
    next: u64,
    /// Bytes the interned strings hold, for [`TagDict::heap_bytes`].
    string_bytes: usize,
}

impl TagDict {
    /// The id for `s`, assigning a fresh one if this is the first time the column has seen it.
    #[must_use]
    pub fn intern(
        &self,
        s: &Arc<String>,
    ) -> u64 {
        if let Some(id) = self.lookup(s) {
            return id;
        }
        let mut inner = self.0.write();
        // Another thread may have interned it between the read and the write lock.
        if let Some(&id) = inner.ids.get(s) {
            return id;
        }
        let id = inner.next;
        inner.next += 1;
        inner.string_bytes += s.capacity();
        inner.ids.insert(s.clone(), id);
        id
    }

    /// The id for `s`, or `None` if the column has never held that string — in which case no tuple
    /// can match it and the caller can answer empty without touching the tree.
    #[must_use]
    pub fn lookup(
        &self,
        s: &Arc<String>,
    ) -> Option<u64> {
        self.0.read().ids.get(s).copied()
    }

    /// The ids of the strings `sel` selects that the column has seen, sorted and distinct.
    ///
    /// For a window this is where a string range costs more than a numeric one: the ids are
    /// assigned in *insertion* order, so the window cannot be a single key range in the tree. It is
    /// proportional to the number of distinct strings in the window, not to the number of matching
    /// entities — including strings whose tuples have since been deleted, which contribute an
    /// empty cursor each (see the module docs on append-only).
    #[must_use]
    pub fn ids(
        &self,
        sel: &StrSel,
    ) -> Vec<u64> {
        let inner = self.0.read();
        let mut ids: Vec<u64> = match sel {
            StrSel::Set(strings) => strings
                .iter()
                .filter_map(|s| inner.ids.get(s).copied())
                .collect(),
            StrSel::Window {
                min,
                max,
                include_min,
                include_max,
            } => {
                if sel.is_empty_window() {
                    return Vec::new();
                }
                let bound = |v: &Option<Arc<String>>, inclusive: bool| match v {
                    None => Bound::Unbounded,
                    Some(v) if inclusive => Bound::Included(v.clone()),
                    Some(v) => Bound::Excluded(v.clone()),
                };
                inner
                    .ids
                    .range((bound(min, *include_min), bound(max, *include_max)))
                    .map(|(_, &id)| id)
                    .collect()
            }
        };
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Number of distinct strings interned. Diagnostics and tests.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.read().ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Heap bytes the dictionary holds: each entry's string and its map slot. Shared by every
    /// version of the column, so each counts it, as each counts the tree pages it shares.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let inner = self.0.read();
        inner.string_bytes
            + inner.ids.len()
                * (size_of::<Arc<String>>() + size_of::<u64>() + size_of::<String>() + 16)
    }
}

/// Which strings a tag predicate selects, by value rather than by dictionary id.
///
/// By value because a column that is still building answers with a scan, and the scan must
/// recognise a string no batch has interned yet: an id set taken from the dictionary at query time
/// would miss it.
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

/// The tag kind of one `(label, attribute)` column: entity ids keyed by the dictionary id of their
/// string value.
///
/// `Clone` is `O(1)`: two root-`Arc` bumps for the trees and one for the shared dictionary.
#[derive(Clone, Default)]
pub struct TagIndex {
    dict: TagDict,
    /// String scalar values: one tuple per indexed entity.
    tree: Tree,
    /// String elements of **list**-valued properties: one tuple per distinct element.
    ///
    /// A separate key space for the same reason the numeric kind keeps one: sharing would make
    /// `WHERE n.v = 'a'` match a node whose `v` is `['a', 'b']`, which is false in Cypher, and
    /// `Equal` has no post-filter to catch it. RediSearch keeps the same split with its
    /// `string:arr` sub-field.
    array_tree: Tree,
}

impl TagIndex {
    /// The kinds of key this kind's trees hold, one tree each.
    const TREES: [KeyKind; 2] = [KeyKind::Tag, KeyKind::TagArray];

    /// The tree holding keys of `kind`.
    fn tree(
        &self,
        kind: KeyKind,
    ) -> &Tree {
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
    ) -> &mut Tree {
        match kind {
            KeyKind::Tag => &mut self.tree,
            KeyKind::TagArray => &mut self.array_tree,
            KeyKind::Scalar | KeyKind::Array => {
                unreachable!("the tag kind holds only tag trees")
            }
        }
    }

    /// The id the column gave `s`, or `None` for a string it never held. Reads the dictionary,
    /// never interns: a unique-constraint check asks this of a value whose write may yet be
    /// refused, and the dictionary is append-only.
    #[must_use]
    pub fn id_of(
        &self,
        s: &Arc<String>,
    ) -> Option<u64> {
        self.dict.lookup(s)
    }

    /// The ids stored under the dictionary id `id` in the tree of `kind`.
    #[must_use]
    pub fn rows_at(
        &self,
        kind: KeyKind,
        id: u64,
    ) -> DocIter {
        docs_in(self.tree(kind), KeySet::Point(id))
    }

    /// The shared dictionary. For tests.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn dict(&self) -> &TagDict {
        &self.dict
    }

    /// Every `(kind, id, doc)` tuple the kind holds, the scalar tree's then the array tree's. For
    /// tests that compare a column with what it must hold.
    #[cfg(test)]
    #[must_use]
    pub fn tuples(&self) -> Vec<(KeyKind, u64, u64)> {
        Self::TREES
            .into_iter()
            .flat_map(|kind| {
                self.tree(kind)
                    .range_tuples(0, u64::MAX)
                    .map(move |(key, doc)| (kind, key, doc))
            })
            .collect()
    }
}

impl RangeKind for TagIndex {
    type Keys = StrSel;
    type Element = Arc<String>;

    /// A string is stored under its dictionary id in the tag tree, and a list under each string
    /// element's id in the tag array tree. Interns, so a background batch encoding entities off the
    /// write thread numbers its strings the way the live column does. Not deduplicated, as
    /// [`for_each_stored_key`](super::encode::for_each_stored_key) is not.
    fn for_each_stored_key(
        &self,
        value: &Value,
        mut emit: impl FnMut(KeyKind, u64),
    ) {
        match value {
            Value::String(s) => emit(KeyKind::Tag, self.dict.intern(s)),
            Value::List(items) => {
                for item in items.iter() {
                    if let Value::String(s) = item {
                        emit(KeyKind::TagArray, self.dict.intern(s));
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
            add_sorted(self.tree_mut(kind), &tuples[kind]);
        }
    }

    fn remove_encoded(
        &mut self,
        tuples: &EncodedTuples,
    ) {
        for kind in Self::TREES {
            remove_sorted(self.tree_mut(kind), &tuples[kind]);
        }
    }

    /// Whether the kind holds no tuples. The dictionary is not consulted: a string interned by a
    /// write that was later removed leaves an entry behind, and that does not make the column
    /// non-empty.
    fn is_empty(&self) -> bool {
        self.tree.is_empty() && self.array_tree.is_empty()
    }

    /// Heap bytes: both trees' pages and the dictionary.
    fn memory_usage(&self) -> usize {
        self.tree.heap_bytes() + self.array_tree.heap_bytes() + self.dict.heap_bytes()
    }

    /// One cursor per distinct string the selection names that the column has interned, chained
    /// lazily ([`TagDict::ids`]).
    fn scalars(
        &self,
        strings: StrSel,
    ) -> DocIter {
        docs_in(&self.tree, KeySet::of_keys(self.dict.ids(&strings)))
    }

    /// A point lookup into the array tree. A string the column never interned is in no list.
    fn lists_holding(
        &self,
        element: &Arc<String>,
    ) -> DocIter {
        let keys = self
            .dict
            .lookup(element)
            .map_or(KeySet::Empty, KeySet::Point);
        docs_in(&self.array_tree, keys)
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
            point(&idx, "never interned").is_empty(),
            "a string the column never held answers empty without touching the tree"
        );
    }

    /// Bytes that RediSearch's tag tokenizer would split on or strip — the separator, whitespace,
    /// backslashes, the escape prefix — are just bytes to a dictionary, so they need no encoding
    /// and cannot collide. `test06_tag_separator` and `test_25_unescaped_string` are the flow
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

    /// A window and a set select by value, so a scan can match a string no batch interned yet.
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

    /// A string deleted from the graph leaves its dictionary entry behind. That must cost nothing
    /// but an empty cursor — never a stale row.
    #[test]
    fn a_removed_string_keeps_its_id_but_yields_nothing() {
        let mut idx = sample();
        idx.remove(&val("bob"), 1);
        assert!(point(&idx, "bob").is_empty());
        assert_eq!(idx.tag().dict().len(), 4, "the dictionary entry stays");
        assert_eq!(
            range(&idx, Some(""), None, true, true),
            vec![2, 3, 4],
            "the emptied id contributes no rows to a range"
        );
        // And re-adding the same string reuses the id rather than minting a second one.
        idx.add(&val("bob"), 9);
        assert_eq!(idx.tag().dict().len(), 4);
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

    /// The property the shared dictionary exists for: a clone of the column (another version, or
    /// the copy an indexer batch encodes through) numbers its strings the way the live column
    /// does, so tuples encoded by one mean the same strings in the other.
    #[test]
    fn a_clone_shares_the_numbering() {
        let mut live = RangeIndex::new();
        live.add(&val("a"), 1);
        let batch = live.clone();
        let mut tuples = EncodedTuples::default();
        batch.push_encoded(&val("b"), 100, &mut tuples);
        batch.push_encoded(&val("a"), 101, &mut tuples);
        live.add_encoded(&mut tuples);
        assert_eq!(point(&live, "a"), vec![1, 101]);
        assert_eq!(point(&live, "b"), vec![100]);
    }
}
