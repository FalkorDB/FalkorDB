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

use super::doc_iter::{DocIter, KeyTuples, Tree, UnionIter, empty_docs};
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
}

impl TagDict {
    /// The id for `s`, assigning a fresh one if this is the first time the column has seen it.
    /// The write path only.
    #[must_use]
    pub fn intern(
        &self,
        s: &Arc<String>,
    ) -> u64 {
        let mut inner = self.0.write();
        if let Some(&id) = inner.ids.get(s) {
            return id;
        }
        let id = inner.next;
        inner.next += 1;
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

    /// Every id whose string lies in the (optionally half-open) lexicographic window.
    ///
    /// This is where a string range costs more than a numeric one: the ids are assigned in
    /// *insertion* order, so the window cannot be a single key range in the tree. It is
    /// proportional to the number of distinct strings in the window, not to the number of matching
    /// entities — including strings whose tuples have since been deleted, which contribute an
    /// empty cursor each (see the module docs on append-only).
    #[must_use]
    pub fn ids_in_window(
        &self,
        min: Option<&Arc<String>>,
        max: Option<&Arc<String>>,
        include_min: bool,
        include_max: bool,
    ) -> Vec<u64> {
        let lo = match min {
            None => Bound::Unbounded,
            Some(v) if include_min => Bound::Included(v.clone()),
            Some(v) => Bound::Excluded(v.clone()),
        };
        let hi = match max {
            None => Bound::Unbounded,
            Some(v) if include_max => Bound::Included(v.clone()),
            Some(v) => Bound::Excluded(v.clone()),
        };
        // An inverted window (`'z' <= x <= 'a'`) is not a valid `BTreeMap::range` argument — it
        // panics rather than yielding nothing — so it is answered here.
        if let (Some(lo), Some(hi)) = (min, max)
            && (lo > hi || (lo == hi && !(include_min && include_max)))
        {
            return Vec::new();
        }
        self.0
            .read()
            .ids
            .range((lo, hi))
            .map(|(_, &id)| id)
            .collect()
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
}

/// A string property index over one `(label, attribute)`: entity ids keyed by the dictionary id of
/// their value.
///
/// `Clone` is `O(1)`: two root-`Arc` bumps for the trees and one for the shared dictionary.
#[derive(Clone, Default)]
pub struct TagIndex {
    dict: TagDict,
    /// Scalar values: one tuple per indexed entity.
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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty index sharing `dict` — how the install builds the replacement column without
    /// renumbering the ids its BASE tuples already carry.
    #[must_use]
    pub fn with_dict(dict: TagDict) -> Self {
        Self {
            dict,
            tree: Tree::default(),
            array_tree: Tree::default(),
        }
    }

    /// The shared dictionary, so a derived column can be built on the same numbering.
    #[must_use]
    pub fn dict(&self) -> TagDict {
        self.dict.clone()
    }

    /// Whether the index holds no tuples. The dictionary is not consulted: a string interned by a
    /// write that was later removed leaves an entry behind, and that does not make the column
    /// non-empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty() && self.array_tree.is_empty()
    }

    /// Append the ids a **stored** value contributes, and say which tree they belong in. Interns,
    /// so this is the write path; a read must use [`TagDict::lookup`], which does not.
    ///
    /// A scalar string contributes one id. A list contributes one per distinct string element, so
    /// `'a' IN n.tags` is a point lookup. Anything else contributes nothing — the other kinds in
    /// the column take it, or no kind does.
    fn encode_stored(
        &self,
        value: &Value,
        out: &mut Vec<u64>,
    ) -> StoredIds {
        out.clear();
        match value {
            Value::String(s) => {
                out.push(self.dict.intern(s));
                StoredIds::Scalar
            }
            Value::List(items) => {
                for item in items.iter() {
                    if let Value::String(s) = item {
                        out.push(self.dict.intern(s));
                    }
                }
                out.sort_unstable();
                out.dedup();
                StoredIds::Array
            }
            _ => StoredIds::Scalar, // nothing to add: `out` is empty
        }
    }

    /// Index `id` under `value`. A no-op for a value holding no strings, and idempotent.
    pub fn add(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        let mut ids = Vec::new();
        let tree = match self.encode_stored(value, &mut ids) {
            StoredIds::Scalar => &mut self.tree,
            StoredIds::Array => &mut self.array_tree,
        };
        for k in ids {
            let _newly_inserted = tree.insert(k, id);
        }
    }

    /// Remove `id` from under `value`. A no-op if it was never indexed.
    ///
    /// Routes on the **old** value's kind, exactly as the numeric side does: a value that changed
    /// from `'a'` to `['a']` must have its scalar tuple removed, not its array tuple.
    pub fn remove(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        let mut ids = Vec::new();
        let tree = match self.encode_stored(value, &mut ids) {
            StoredIds::Scalar => &mut self.tree,
            StoredIds::Array => &mut self.array_tree,
        };
        for k in ids {
            let _was_present = tree.remove(k, id);
        }
    }

    /// Encode `(value, id)` entries to `(tag id, doc)` tuples, dropping values that hold no
    /// strings. Interning happens here, so a background build encoding BASE off the write thread
    /// shares the numbering with the live column.
    #[must_use]
    pub fn encode_entries(
        &self,
        entries: &[(Value, u64)],
    ) -> KeyTuples {
        let mut out = KeyTuples::default();
        let mut ids = Vec::new();
        for (v, id) in entries {
            let dest = match self.encode_stored(v, &mut ids) {
                StoredIds::Scalar => &mut out.scalar,
                StoredIds::Array => &mut out.array,
            };
            dest.extend(ids.iter().map(|&k| (k, *id)));
        }
        out
    }

    /// Bulk-build from `(value, id)` pairs in any order — one sort plus a bottom-up page build,
    /// rather than a traversal per item.
    #[must_use]
    pub fn from_entries<'a>(
        dict: TagDict,
        entries: impl IntoIterator<Item = (&'a Value, u64)>,
    ) -> Self {
        let empty = Self::with_dict(dict);
        let mut out = KeyTuples::default();
        let mut ids = Vec::new();
        for (v, id) in entries {
            let dest = match empty.encode_stored(v, &mut ids) {
                StoredIds::Scalar => &mut out.scalar,
                StoredIds::Array => &mut out.array,
            };
            dest.extend(ids.iter().map(|&k| (k, id)));
        }
        Self::from_encoded(empty.dict, out)
    }

    /// Build directly from already-encoded tuples, on an existing dictionary — how the install
    /// adopts a background-built BASE without re-encoding it.
    #[must_use]
    pub fn from_encoded(
        dict: TagDict,
        mut tuples: KeyTuples,
    ) -> Self {
        let build = |pairs: &mut Vec<(u64, u64)>| {
            pairs.sort_unstable();
            pairs.dedup();
            Tree::from_sorted(pairs)
        };
        Self {
            dict,
            tree: build(&mut tuples.scalar),
            array_tree: build(&mut tuples.array),
        }
    }

    /// Every `(tag id, doc)` tuple, in key order — the install's DELTA/TOMB enumeration.
    #[must_use]
    pub fn encoded_tuples(&self) -> KeyTuples {
        KeyTuples {
            scalar: self.tree.range_tuples(0, u64::MAX).collect(),
            array: self.array_tree.range_tuples(0, u64::MAX).collect(),
        }
    }

    /// Add already-encoded tuples (install: replay DELTA onto BASE).
    pub fn add_encoded(
        &mut self,
        tuples: &mut KeyTuples,
    ) {
        tuples.sort();
        self.tree.insert_batch(&tuples.scalar);
        self.array_tree.insert_batch(&tuples.array);
    }

    /// Remove already-encoded tuples (install: subtract TOMB from BASE).
    pub fn remove_encoded(
        &mut self,
        tuples: &mut KeyTuples,
    ) {
        tuples.sort();
        self.tree.remove_batch(&tuples.scalar);
        self.array_tree.remove_batch(&tuples.array);
    }

    /// Entity ids whose value equals `value` — exact and case-sensitive.
    #[must_use]
    pub fn point(
        &self,
        value: &Arc<String>,
    ) -> DocIter {
        match self.dict.lookup(value) {
            Some(id) => DocIter::One(self.tree.point(id)),
            None => empty_docs(&self.tree),
        }
    }

    /// Entity ids whose value lies in the lexicographic window: one cursor per distinct string in
    /// the window, chained lazily.
    #[must_use]
    pub fn range(
        &self,
        min: Option<&Arc<String>>,
        max: Option<&Arc<String>>,
        include_min: bool,
        include_max: bool,
    ) -> DocIter {
        let ids = self.dict.ids_in_window(min, max, include_min, include_max);
        DocIter::Many(UnionIter::new(
            vec![self.tree.clone()],
            ids.into_iter().map(|id| (0, id, id)).collect(),
        ))
    }

    /// Entity ids whose **list**-valued property contains `value` — `'a' IN n.tags`. A point
    /// lookup into the array tree, so each matching doc appears once.
    #[must_use]
    pub fn array_contains(
        &self,
        value: &Arc<String>,
    ) -> DocIter {
        match self.dict.lookup(value) {
            Some(id) => DocIter::One(self.array_tree.point(id)),
            None => empty_docs(&self.array_tree),
        }
    }

    /// The scalar tree — for the column facade, which composes windows across kinds.
    pub(super) fn tree(&self) -> &Tree {
        &self.tree
    }

    /// The id a query value maps to, or `None` when it is not a string this column has seen.
    pub(super) fn key_of(
        &self,
        value: &Value,
    ) -> Option<u64> {
        match value {
            Value::String(s) => self.dict.lookup(s),
            _ => None,
        }
    }
}

/// Which tree [`TagIndex::encode_stored`] filled its output for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StoredIds {
    /// A scalar (or a value this kind does not index, which contributes nothing).
    Scalar,
    /// A list: the ids are its string elements. May be empty — a list with no string elements is
    /// still an array-kind value, not a scalar.
    Array,
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn ids(it: DocIter) -> Vec<u64> {
        let mut v: Vec<u64> = it.collect();
        v.sort_unstable();
        v
    }

    fn sample() -> TagIndex {
        let mut idx = TagIndex::new();
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
        assert_eq!(ids(idx.point(&s("bob"))), vec![1]);
        assert_eq!(ids(idx.point(&s("Bob"))), vec![3]);
        assert!(ids(idx.point(&s("BOB"))).is_empty());
        assert!(ids(idx.point(&s("bo"))).is_empty(), "no prefix matching");
        assert!(
            ids(idx.point(&s("never interned"))).is_empty(),
            "a string the column never held answers empty without touching the tree"
        );
    }

    /// Bytes that RediSearch's tag tokenizer would split on or strip — the separator, whitespace,
    /// backslashes, the escape prefix — are just bytes to a dictionary, so they need no encoding
    /// and cannot collide. `test06_tag_separator` and `test_25_unescaped_string` are the flow
    /// tests this covers.
    #[test]
    fn tokenizer_hostile_strings_are_ordinary_keys() {
        let mut idx = TagIndex::new();
        let hostile = ["a b", "a\\b", "a_b", "a\u{1}b", "a,b", "", " ", "a|b"];
        for (i, h) in hostile.iter().enumerate() {
            idx.add(&val(h), i as u64);
        }
        for (i, h) in hostile.iter().enumerate() {
            assert_eq!(
                ids(idx.point(&s(h))),
                vec![i as u64],
                "{h:?} must round-trip"
            );
        }
    }

    #[test]
    fn lexicographic_range_bounds() {
        let idx = sample();
        let all = |min: Option<&str>, max: Option<&str>, inc_min, inc_max| {
            ids(idx.range(min.map(s).as_ref(), max.map(s).as_ref(), inc_min, inc_max))
        };
        // Capitals sort below lowercase in codepoint order: "Bob" < "alice" < "bob" < "carol".
        assert_eq!(all(None, None, true, true), vec![1, 2, 3, 4]);
        assert_eq!(all(Some("alice"), None, true, true), vec![1, 2, 4]);
        assert_eq!(all(Some("alice"), None, false, true), vec![1, 4]);
        assert_eq!(all(None, Some("bob"), true, true), vec![1, 2, 3]);
        assert_eq!(all(None, Some("bob"), true, false), vec![2, 3]);
        assert_eq!(all(Some("alice"), Some("carol"), true, false), vec![1, 2]);
        // An inverted or empty window is an answer, not a panic.
        assert!(all(Some("z"), Some("a"), true, true).is_empty());
        assert!(all(Some("a"), Some("a"), false, false).is_empty());
    }

    /// A string deleted from the graph leaves its dictionary entry behind. That must cost nothing
    /// but an empty cursor — never a stale row.
    #[test]
    fn a_removed_string_keeps_its_id_but_yields_nothing() {
        let mut idx = sample();
        idx.remove(&val("bob"), 1);
        assert!(ids(idx.point(&s("bob"))).is_empty());
        assert_eq!(idx.dict.len(), 4, "the dictionary entry stays");
        assert_eq!(
            ids(idx.range(None, None, true, true)),
            vec![2, 3, 4],
            "the emptied id contributes no rows to a range"
        );
        // And re-adding the same string reuses the id rather than minting a second one.
        idx.add(&val("bob"), 9);
        assert_eq!(idx.dict.len(), 4);
        assert_eq!(ids(idx.point(&s("bob"))), vec![9]);
    }

    /// The scalar/array split, and the reason for it: `n.v = 'a'` must not match `['a','b']`, and
    /// `'a' IN n.v` must not match the scalar `'a'`.
    #[test]
    fn a_scalar_and_a_list_element_do_not_share_a_key_space() {
        let mut idx = TagIndex::new();
        idx.add(&val("a"), 10);
        idx.add(&list(&["a", "b"]), 20);

        assert_eq!(ids(idx.point(&s("a"))), vec![10]);
        assert_eq!(ids(idx.array_contains(&s("a"))), vec![20]);
        assert_eq!(ids(idx.array_contains(&s("b"))), vec![20]);
        assert!(ids(idx.array_contains(&s("c"))).is_empty());
        // A range over the scalar tree must not see list elements either.
        assert_eq!(ids(idx.range(None, None, true, true)), vec![10]);
    }

    /// A repeated element contributes one tuple, so removing the value takes everything with it.
    #[test]
    fn duplicate_elements_contribute_one_tuple() {
        let mut idx = TagIndex::new();
        idx.add(&list(&["a", "a", "b"]), 7);
        assert_eq!(ids(idx.array_contains(&s("a"))), vec![7]);
        idx.remove(&list(&["a", "a", "b"]), 7);
        assert!(idx.is_empty(), "no tuple stranded in either tree");
    }

    /// The dangerous re-index case: the same string crossing between the two key spaces. A remove
    /// routed on the *new* value's kind would delete from the wrong tree and strand a tuple that
    /// no later write can reach — and on the scalar side nothing re-checks the index's answer.
    #[test]
    fn a_value_changing_kind_moves_between_trees() {
        let mut idx = TagIndex::new();
        idx.add(&val("a"), 1);

        idx.remove(&val("a"), 1);
        idx.add(&list(&["a"]), 1);
        assert!(ids(idx.point(&s("a"))).is_empty());
        assert_eq!(ids(idx.array_contains(&s("a"))), vec![1]);

        idx.remove(&list(&["a"]), 1);
        idx.add(&val("a"), 1);
        assert_eq!(ids(idx.point(&s("a"))), vec![1]);
        assert!(ids(idx.array_contains(&s("a"))).is_empty());

        idx.remove(&val("a"), 1);
        assert!(idx.is_empty());
    }

    /// Bulk build and incremental build must agree.
    #[test]
    fn bulk_build_matches_incremental() {
        let entries = [
            (val("bob"), 1u64),
            (val("alice"), 2),
            (val("Bob"), 3),
            (val("carol"), 4),
            (Value::Int(7), 5), // not a string: this kind indexes nothing for it
        ];
        let bulk =
            TagIndex::from_entries(TagDict::default(), entries.iter().map(|(v, id)| (v, *id)));
        assert_eq!(
            ids(bulk.range(None, None, true, true)),
            ids(sample().range(None, None, true, true))
        );
    }

    /// The install's round trip: tuples out of one column, back into a column built on the *same*
    /// dictionary. Rebuilding on a fresh dictionary would renumber the strings and the ids in the
    /// tuples would name the wrong ones — which is why `from_encoded` takes the dictionary.
    #[test]
    fn encoded_tuples_round_trip_on_the_shared_dictionary() {
        let mut idx = TagIndex::new();
        idx.add(&val("a"), 10);
        idx.add(&list(&["b"]), 20);

        let rebuilt = TagIndex::from_encoded(idx.dict(), idx.encoded_tuples());
        assert_eq!(ids(rebuilt.point(&s("a"))), vec![10]);
        assert_eq!(ids(rebuilt.array_contains(&s("b"))), vec![20]);
    }

    /// The property the shared dictionary exists for: a BASE encoded off the write thread numbers
    /// its strings the same way the live column does, so the two sets of tuples can be merged.
    #[test]
    fn a_background_encode_shares_the_live_numbering() {
        let mut live = TagIndex::new();
        live.add(&val("a"), 1); // "a" interned by the live column

        // The background job encodes its snapshot through the same dictionary.
        let base = live.encode_entries(&[(val("a"), 100), (val("b"), 101)]);

        let mut installed = TagIndex::from_encoded(live.dict(), base);
        let mut delta = live.encoded_tuples();
        installed.add_encoded(&mut delta);

        assert_eq!(
            ids(installed.point(&s("a"))),
            vec![1, 100],
            "the same string must resolve to the same id on both sides of the install"
        );
        assert_eq!(ids(installed.point(&s("b"))), vec![101]);
    }
}
