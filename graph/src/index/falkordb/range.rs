//! The Range column: one `(label, attribute)` index, made of the three kinds a RediSearch Range
//! field is made of — NUMERIC, TAG and GEO.
//!
//! A Cypher property is not typed, so one column has to hold whatever the entities put in it. The
//! kinds are therefore not alternatives but **co-resident**: a column with `age = 30` on one node,
//! `age = 'thirty'` on another and `loc = point(...)` on a third stores each in the kind that can
//! order it, and a predicate is routed by the type of its operand.
//!
//! Routing is exhaustive and total. Every predicate either reaches a kind, is answered empty
//! because Cypher says it cannot match (a `NULL` operand, or a comparison across types), or is
//! declined — and a decline is a hard error under `index-falkordb`, because there is no other
//! index to try. The three outcomes are kept apart deliberately: an empty answer that should have
//! been a decline is exactly how an unimplemented kind hides.

use std::sync::Arc;

use rustc_hash::FxHashSet;

use super::doc_iter::{DocIter, KeyTuples, UnionIter, empty_docs};
use super::geo::GeoIndex;
use super::numeric::NumericIndex;
use super::tag::{TagDict, TagIndex};
use crate::index::IndexQuery;
use crate::runtime::value::Value;

/// Which kind holds a value, and therefore which kind answers a predicate over it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Numeric,
    Tag,
    Geo,
}

impl Kind {
    /// The kind that indexes `value`, or `None` for a value no kind holds (`NULL`, a list, a map,
    /// a vector). A list is deliberately not a kind: its *elements* are indexed by the numeric and
    /// tag kinds, but the list itself is not a value any predicate can be routed by.
    fn of(value: &Value) -> Option<Self> {
        match value {
            Value::Int(_)
            | Value::Float(_)
            | Value::Bool(_)
            | Value::Datetime(_)
            | Value::Date(_)
            | Value::Time(_)
            | Value::Duration(_) => Some(Self::Numeric),
            Value::String(_) => Some(Self::Tag),
            Value::Point(_) => Some(Self::Geo),
            _ => None,
        }
    }

    /// Position of this kind's tree in the `trees` vector a [`UnionIter`] is built with — see
    /// [`RangeIndex::union_trees`].
    const fn slot(self) -> u8 {
        match self {
            Self::Numeric => 0,
            Self::Tag => 1,
            Self::Geo => 2,
        }
    }
}

/// The string inside a bound, when the bound is one. A free function rather than a closure
/// because it has to be generic over the borrow's lifetime — a closure would tie every call to
/// one.
fn as_str(v: Option<&Value>) -> Option<&Arc<String>> {
    match v {
        Some(Value::String(s)) => Some(s),
        _ => None,
    }
}

/// The encoded contents of one column, per kind — the artifact the online build moves between
/// threads as BASE, DELTA and TOMB.
///
/// Every kind's keys are `u64` (the numeric encoding, a tag dictionary id, a Morton code), so the
/// install merges them as raw tuples without decoding. The tag half only works because the
/// dictionary is shared between the background job and the live column; see [`TagIndex`].
#[derive(Default, Debug, Clone)]
pub struct EncodedTuples {
    pub numeric: KeyTuples,
    pub tag: KeyTuples,
    pub geo: Vec<(u64, u64)>,
}

impl EncodedTuples {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.numeric.is_empty() && self.tag.is_empty() && self.geo.is_empty()
    }

    /// Total tuples across every kind and key space.
    #[must_use]
    pub fn len(&self) -> usize {
        self.numeric.len() + self.tag.len() + self.geo.len()
    }

    /// Numeric scalars only — the shape a column of plain numbers produces, and the constructor
    /// the install tests build a BASE with.
    #[must_use]
    pub fn scalars(scalar: Vec<(u64, u64)>) -> Self {
        Self {
            numeric: KeyTuples::scalars(scalar),
            ..Self::default()
        }
    }

    /// Drop every tuple whose doc fails `keep` — the install's deleted-entity backstop. It has to
    /// sweep every kind and both key spaces, or a deleted node survives in whichever half was
    /// missed.
    pub fn retain_docs(
        &mut self,
        mut keep: impl FnMut(u64) -> bool,
    ) {
        self.numeric.retain_docs(&mut keep);
        self.tag.retain_docs(&mut keep);
        self.geo.retain(|&(_, doc)| keep(doc));
    }
}

/// One indexed `(label, attribute)`: the numeric, tag and geo kinds over the same column.
///
/// `Clone` is `O(1)` — a root-`Arc` bump per tree plus one for the shared tag dictionary — so a
/// graph version forks its index snapshot cheaply.
#[derive(Clone, Default)]
pub struct RangeIndex {
    numeric: NumericIndex,
    tag: TagIndex,
    geo: GeoIndex,
}

impl RangeIndex {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty column sharing this one's tag dictionary — how the install builds a replacement
    /// without renumbering the ids its BASE tuples already carry.
    #[must_use]
    pub fn empty_like(&self) -> Self {
        Self {
            numeric: NumericIndex::new(),
            tag: TagIndex::with_dict(self.tag.dict()),
            geo: GeoIndex::new(),
        }
    }

    /// Bulk-build from `(value, id)` pairs in any order — one sort and a bottom-up page build per
    /// kind, rather than a tree traversal per item.
    #[must_use]
    pub fn from_entries<'a>(entries: impl IntoIterator<Item = (&'a Value, u64)>) -> Self {
        // Materialise once: each kind needs its own pass, and the caller's iterator is single-use.
        let entries: Vec<(&Value, u64)> = entries.into_iter().collect();
        Self {
            numeric: NumericIndex::from_entries(entries.iter().copied()),
            tag: TagIndex::from_entries(TagDict::default(), entries.iter().copied()),
            geo: GeoIndex::from_entries(entries.iter().copied()),
        }
    }

    /// The numeric kind, for the callers that predate the other two (the populate path and tests).
    #[must_use]
    pub fn numeric(&self) -> &NumericIndex {
        &self.numeric
    }

    /// Index `id` under `value`. Each kind takes the part of the value it can represent and
    /// ignores the rest, so a scalar lands in exactly one and a list's elements are split between
    /// the numeric and tag array trees.
    pub fn add(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        self.numeric.add(value, id);
        self.tag.add(value, id);
        self.geo.add(value, id);
    }

    /// Remove `id` from under `value` — the old value on delete/update.
    pub fn remove(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        self.numeric.remove(value, id);
        self.tag.remove(value, id);
        self.geo.remove(value, id);
    }

    /// Add a batch of `(value, id)` entries — the columnar write path's add column. One batched
    /// tree op per kind.
    pub fn add_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
    ) {
        let mut t = self.encode_stream(entries);
        self.add_encoded(&mut t);
    }

    /// Remove a batch of `(value, id)` entries — the write path's remove column.
    pub fn remove_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
    ) {
        let mut t = self.encode_stream(entries);
        self.remove_encoded(&mut t);
    }

    /// Encode an owned batch in **one pass**, offering each entry to every kind in turn.
    ///
    /// The per-kind slice pass ([`encode_entries`](Self::encode_entries)) exists for the background
    /// build, which is handed a materialised BASE anyway. The write path is not: routing through
    /// the slice version made every commit materialise the batch first, for no gain — a kind that
    /// cannot represent a value costs one `match` arm either way.
    fn encode_stream(
        &self,
        entries: impl IntoIterator<Item = (Value, u64)>,
    ) -> EncodedTuples {
        let mut out = EncodedTuples::default();
        let mut scratch = Vec::new();
        for (v, id) in entries {
            NumericIndex::encode_into(&v, id, &mut out.numeric, &mut scratch);
            self.tag.encode_into(&v, id, &mut out.tag, &mut scratch);
            if let Some(k) = GeoIndex::key_of(&v) {
                out.geo.push((k, id));
            }
        }
        out
    }

    /// Encode `(value, id)` entries under every kind — how a background build produces BASE off
    /// the write thread. Interning happens here, against the column's shared dictionary, so the
    /// ids in BASE mean the same strings as the ids in DELTA.
    #[must_use]
    pub fn encode_entries(
        &self,
        entries: Vec<(Value, u64)>,
    ) -> EncodedTuples {
        self.encode_entries_slice(&entries)
    }

    fn encode_entries_slice(
        &self,
        entries: &[(Value, u64)],
    ) -> EncodedTuples {
        EncodedTuples {
            numeric: NumericIndex::encode_entries(entries),
            tag: self.tag.encode_entries(entries),
            geo: GeoIndex::encode_entries(entries),
        }
    }

    /// A new column built from already-encoded tuples, on *this* column's tag dictionary.
    #[must_use]
    pub fn from_encoded_like(
        &self,
        tuples: EncodedTuples,
    ) -> Self {
        Self {
            numeric: NumericIndex::from_encoded(tuples.numeric),
            tag: TagIndex::from_encoded(self.tag.dict(), tuples.tag),
            geo: GeoIndex::from_encoded(tuples.geo),
        }
    }

    /// Every tuple this column holds, encoded — the install's DELTA/TOMB enumeration.
    #[must_use]
    pub fn encoded_tuples(&self) -> EncodedTuples {
        EncodedTuples {
            numeric: self.numeric.encoded_tuples(),
            tag: self.tag.encoded_tuples(),
            geo: self.geo.encoded_tuples(),
        }
    }

    /// Add already-encoded tuples (install: replay DELTA onto BASE).
    pub fn add_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        self.numeric.add_encoded(&mut tuples.numeric);
        self.tag.add_encoded(&mut tuples.tag);
        self.geo.add_encoded(&mut tuples.geo);
    }

    /// Remove already-encoded tuples (install: subtract TOMB from BASE).
    pub fn remove_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        self.numeric.remove_encoded(&mut tuples.numeric);
        self.tag.remove_encoded(&mut tuples.tag);
        self.geo.remove_encoded(&mut tuples.geo);
    }

    /// Whether the column holds no tuples in any kind.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.numeric.is_empty() && self.tag.is_empty() && self.geo.is_empty()
    }

    /// Answer a predicate over this column, or decline it.
    ///
    /// `None` means "no kind here can answer this", which the no-fallback build turns into an
    /// error naming the predicate. An **empty iterator** is a different thing: the predicate was
    /// understood and nothing matches it.
    #[must_use]
    pub fn query(
        &self,
        q: &IndexQuery<Value>,
    ) -> Option<DocIter> {
        match q {
            IndexQuery::Equal { value, .. } => self.point(value),
            IndexQuery::Range {
                min,
                max,
                include_min,
                include_max,
                ..
            } => self.range(min.as_ref(), max.as_ref(), *include_min, *include_max),
            IndexQuery::ArrayContains { value, .. } => self.array_contains(value),
            IndexQuery::Or(children) => self.union(children),
            IndexQuery::And(children) => {
                let refs: Vec<&IndexQuery<Value>> = children.iter().collect();
                self.intersect_refs(&refs)
            }
            IndexQuery::Point { point, radius, .. } => self.within(point, radius),
            // `IN` is desugared into `Or` before the index is consulted.
            IndexQuery::InList { .. } => None,
        }
    }

    /// `n.v = value`, routed by the value's type.
    ///
    /// A `NULL` operand is answered **empty**, not declined: `n.v = null` is unknown in Cypher and
    /// so matches nothing, whatever the column holds. Declining would turn a well-defined query
    /// into an error.
    fn point(
        &self,
        value: &Value,
    ) -> Option<DocIter> {
        match Kind::of(value) {
            Some(Kind::Numeric) => Some(self.numeric.point(value)),
            Some(Kind::Tag) => match value {
                Value::String(s) => Some(self.tag.point(s)),
                _ => unreachable!("Kind::Tag is only ever a string"),
            },
            Some(Kind::Geo) => Some(self.geo.point(value)),
            None if matches!(value, Value::Null) => Some(self.empty()),
            None => None,
        }
    }

    /// `n.v <op> bound`, routed by the bounds' type.
    ///
    /// Cypher orders numbers with numbers and strings with strings; every other comparison is
    /// unknown and matches nothing. So a range whose bounds disagree on kind — or whose bound is a
    /// point, a `NULL`, or anything else unordered — is answered **empty** rather than declined.
    /// That is not a gap in the index; it is the language.
    fn range(
        &self,
        min: Option<&Value>,
        max: Option<&Value>,
        include_min: bool,
        include_max: bool,
    ) -> Option<DocIter> {
        match self.bound_kind(min, max)? {
            // Unbounded on both sides: no operand to route by, and every kind's values would
            // qualify. The numeric tree answers it, as it did before the other kinds existed.
            None | Some(Kind::Numeric) => {
                Some(self.numeric.range(min, max, include_min, include_max))
            }
            Some(Kind::Tag) => {
                Some(
                    self.tag
                        .range(as_str(min), as_str(max), include_min, include_max),
                )
            }
            // Points are not ordered in Cypher — `n.loc > point(...)` is unknown.
            Some(Kind::Geo) => Some(self.empty()),
        }
    }

    /// The kind a range's bounds agree on: `Some(None)` when there are no bounds at all,
    /// `Some(Some(kind))` when they agree, and a directly-answered empty result (via `None` from
    /// [`range`](Self::range)'s `?`) is *not* what this returns — a disagreement or an unorderable
    /// bound yields `Some(Some(Kind::Geo))`-style routing instead. Returns `None` only for a bound
    /// this index cannot reason about at all.
    fn bound_kind(
        &self,
        min: Option<&Value>,
        max: Option<&Value>,
    ) -> Option<Option<Kind>> {
        let kind_of = |v: Option<&Value>| -> Option<Option<Kind>> {
            match v {
                None => Some(None),
                // A `NULL` bound makes the comparison unknown: nothing matches. Routed to the geo
                // arm, which answers empty — the same answer, without a separate code path.
                Some(Value::Null) => Some(Some(Kind::Geo)),
                Some(v) => Kind::of(v).map(Some),
            }
        };
        match (kind_of(min)?, kind_of(max)?) {
            (None, None) => Some(None),
            (Some(k), None) | (None, Some(k)) => Some(Some(k)),
            (Some(a), Some(b)) if a == b => Some(Some(a)),
            // Bounds of different kinds cannot both hold: no value is a number and a string.
            (Some(_), Some(_)) => Some(Some(Kind::Geo)),
        }
    }

    /// `value IN n.v`, where `n.v` is a list — a point lookup in the kind's array tree.
    ///
    /// Only the numeric and tag kinds have one. RediSearch indexes no point inside a list either,
    /// so a point probe is declined and stays visible as a gap.
    fn array_contains(
        &self,
        value: &Value,
    ) -> Option<DocIter> {
        match Kind::of(value) {
            Some(Kind::Numeric) => Some(self.numeric.array_contains(value)),
            Some(Kind::Tag) => match value {
                Value::String(s) => Some(self.tag.array_contains(s)),
                _ => unreachable!("Kind::Tag is only ever a string"),
            },
            _ => None,
        }
    }

    /// `distance(n.v, centre) < radius` — a superset of the disk, which the retained filter
    /// narrows. Declined if the operands are not a point and a number, since then the predicate is
    /// not the shape the planner promised.
    fn within(
        &self,
        centre: &Value,
        radius: &Value,
    ) -> Option<DocIter> {
        let Value::Point(centre) = centre else {
            return None;
        };
        let radius = match radius {
            Value::Int(i) => *i as f64,
            Value::Float(f) => *f,
            _ => return None,
        };
        Some(self.geo.within(centre, radius))
    }

    /// A union of `Equal` leaves on one attribute — what `n.v IN [...]` desugars to.
    ///
    /// Members may span kinds: `n.v IN [1, 'a']` reads the numeric tree for `1` and the tag tree
    /// for `'a'`, and one chained iterator walks both. No doc can be yielded twice, because a
    /// scalar has exactly one type and so appears in exactly one kind.
    ///
    /// A `NULL` member contributes nothing (`x IN [1, null]` matches only `1`). A member whose
    /// type no kind holds declines the **whole** union: answering from the rest would silently
    /// drop the rows that member would have matched.
    fn union(
        &self,
        children: &[IndexQuery<Value>],
    ) -> Option<DocIter> {
        let refs: Vec<&IndexQuery<Value>> = children.iter().collect();
        self.union_refs(&refs)
    }

    /// [`union`](Self::union) over borrowed members, so the facade can hand one column the share of
    /// a cross-attribute union that belongs to it without cloning — `IndexQuery` is deliberately
    /// not `Clone`.
    pub(super) fn union_refs(
        &self,
        children: &[&IndexQuery<Value>],
    ) -> Option<DocIter> {
        fn collect<'a>(
            children: &[&'a IndexQuery<Value>],
            out: &mut Vec<&'a Value>,
            attr: &mut Option<&'a Arc<String>>,
        ) -> Option<()> {
            for child in children {
                match *child {
                    IndexQuery::Equal { key, value } => {
                        match attr {
                            Some(first) if *first != key => return None,
                            Some(_) => {}
                            None => *attr = Some(key),
                        }
                        out.push(value);
                    }
                    // `a IN [..] OR a IN [..]` arrives as an `Or` of `Or`s; the nesting carries no
                    // meaning for one column.
                    IndexQuery::Or(nested) => {
                        let refs: Vec<&IndexQuery<Value>> = nested.iter().collect();
                        collect(&refs, out, attr)?;
                    }
                    _ => return None,
                }
            }
            Some(())
        }

        let mut values = Vec::with_capacity(children.len());
        collect(children, &mut values, &mut None)?;

        let mut windows: Vec<(u8, u64, u64)> = Vec::with_capacity(values.len());
        for v in values {
            match Kind::of(v) {
                Some(kind) => {
                    // A key the column has never held (an unseen string) matches nothing, and
                    // contributes no window rather than an empty cursor.
                    if let Some(k) = self.key_of(kind, v) {
                        windows.push((kind.slot(), k, k));
                    }
                }
                None if matches!(v, Value::Null) => {} // matches nothing, drops out of the union
                None => return None,                   // a type no kind holds: decline the union
            }
        }
        windows.sort_unstable();
        windows.dedup();
        Some(DocIter::Many(UnionIter::new(self.union_trees(), windows)))
    }

    /// A conjunction of predicates that all constrain **one** attribute, collapsed into one scan.
    ///
    /// Numeric conjuncts fold arithmetically into a single window (the planner cannot do it: at
    /// plan time the bounds may still be expressions it cannot compare). String conjuncts fold by
    /// intersecting the dictionary-id sets their windows select. A conjunction mixing the two is
    /// unsatisfiable — no scalar is both a number and a string — and is answered empty.
    ///
    /// A conjunct that is not a foldable leaf but is still servable on its own — a union, an
    /// array-contains, a distance predicate — is answered separately and intersected as a doc set.
    /// `p.name IN [...] AND p.age IN [...]` reaches here one attribute at a time and is the reason
    /// the set path exists at all.
    ///
    /// Returns `None` if a member targets another attribute or is a shape no kind can serve; the
    /// facade then declines, and the no-fallback build reports it.
    pub(super) fn intersect_refs(
        &self,
        children: &[&IndexQuery<Value>],
    ) -> Option<DocIter> {
        #[derive(Default)]
        struct Fold<'a> {
            attr: Option<&'a Arc<String>>,
            numeric: Option<(u64, u64)>,
            /// `None` until a string conjunct is seen; then the ids still in play.
            tags: Option<Vec<u64>>,
            /// A conjunct that nothing can satisfy (`NULL` bound, cross-kind comparison).
            unsatisfiable: bool,
            /// Conjuncts that do not fold into a window but can be answered on their own.
            streams: Vec<&'a IndexQuery<Value>>,
        }

        fn fold<'a>(
            me: &RangeIndex,
            children: &[&'a IndexQuery<Value>],
            acc: &mut Fold<'a>,
        ) -> Option<()> {
            for child in children {
                let key = match *child {
                    IndexQuery::Equal { key, .. } | IndexQuery::Range { key, .. } => key,
                    IndexQuery::And(nested) if !nested.is_empty() => {
                        let refs: Vec<&IndexQuery<Value>> = nested.iter().collect();
                        fold(me, &refs, acc)?;
                        continue;
                    }
                    // Not foldable, but possibly answerable. Check it now rather than at
                    // intersection time so an unservable conjunct still declines the whole
                    // conjunction instead of quietly narrowing it.
                    IndexQuery::Or(_)
                    | IndexQuery::ArrayContains { .. }
                    | IndexQuery::Point { .. } => {
                        let mut keys = Vec::new();
                        if !super::falkordb_index::attributes_of(child, &mut keys) {
                            return None;
                        }
                        let first = *keys.first()?;
                        if keys.iter().any(|k| *k != first) {
                            return None; // this conjunct alone spans columns
                        }
                        match acc.attr {
                            Some(a) if a != first => return None,
                            Some(_) => {}
                            None => acc.attr = Some(first),
                        }
                        me.query(child)?; // servable at all?
                        acc.streams.push(child);
                        continue;
                    }
                    _ => return None,
                };
                match acc.attr {
                    Some(first) if first != key => return None,
                    Some(_) => {}
                    None => acc.attr = Some(key),
                }
                match child {
                    IndexQuery::Equal { value, .. } => match Kind::of(value) {
                        Some(Kind::Numeric) => {
                            let k = NumericIndex::key_of(value)?;
                            acc.numeric = Some(tighten(acc.numeric, (k, k)));
                        }
                        Some(Kind::Tag) => {
                            let ids = me.tag.key_of(value).into_iter().collect();
                            acc.tags = Some(intersect_ids(acc.tags.take(), ids));
                        }
                        // A point equality inside a conjunction: nothing folds it, and the geo
                        // tree cannot be intersected arithmetically. Decline rather than guess.
                        Some(Kind::Geo) => return None,
                        None if matches!(value, Value::Null) => acc.unsatisfiable = true,
                        None => return None,
                    },
                    IndexQuery::Range {
                        min,
                        max,
                        include_min,
                        include_max,
                        ..
                    } => match me.bound_kind(min.as_ref(), max.as_ref())? {
                        None | Some(Kind::Numeric) => {
                            let w = NumericIndex::window(
                                min.as_ref(),
                                max.as_ref(),
                                *include_min,
                                *include_max,
                            )?;
                            acc.numeric = Some(tighten(acc.numeric, w));
                        }
                        Some(Kind::Tag) => {
                            let ids = me.tag.dict().ids_in_window(
                                as_str(min.as_ref()),
                                as_str(max.as_ref()),
                                *include_min,
                                *include_max,
                            );
                            acc.tags = Some(intersect_ids(acc.tags.take(), ids));
                        }
                        Some(Kind::Geo) => acc.unsatisfiable = true,
                    },
                    _ => return None,
                }
            }
            Some(())
        }

        /// The tighter of two inclusive windows.
        fn tighten(
            acc: Option<(u64, u64)>,
            w: (u64, u64),
        ) -> (u64, u64) {
            match acc {
                None => w,
                Some((lo, hi)) => (lo.max(w.0), hi.min(w.1)),
            }
        }

        /// Intersect a running id set with the next conjunct's.
        fn intersect_ids(
            acc: Option<Vec<u64>>,
            mut next: Vec<u64>,
        ) -> Vec<u64> {
            next.sort_unstable();
            next.dedup();
            match acc {
                None => next,
                Some(prev) => {
                    let set: FxHashSet<u64> = next.into_iter().collect();
                    prev.into_iter().filter(|id| set.contains(id)).collect()
                }
            }
        }

        if children.is_empty() {
            return None;
        }
        let mut acc = Fold::default();
        fold(self, children, &mut acc)?;
        acc.attr?; // an `And` of nothing but empty nested `And`s names no column

        // A scalar is a number or a string, never both, so constraining it as both matches nothing.
        if acc.unsatisfiable || (acc.numeric.is_some() && acc.tags.is_some()) {
            return Some(self.empty());
        }
        let folded = match (acc.numeric, acc.tags) {
            (Some((lo, hi)), None) => Some(if lo <= hi {
                DocIter::One(self.numeric.tree().range(lo, hi))
            } else {
                self.empty()
            }),
            (None, Some(ids)) => Some(DocIter::Many(UnionIter::new(
                vec![self.tag.tree().clone()],
                ids.into_iter().map(|id| (0, id, id)).collect(),
            ))),
            (None, None) => None,
            (Some(_), Some(_)) => unreachable!("handled above"),
        };

        // One stream: hand it back lazily, so a `LIMIT` still stops the scan early.
        if acc.streams.is_empty() {
            // `fold` recorded no constraint at all: every conjunct was an empty nested `And`,
            // which names no window.
            return folded;
        }
        if folded.is_none() && acc.streams.len() == 1 {
            return self.query(acc.streams[0]);
        }

        // Several streams. They are in *value* order, not doc order, so this is a set probe rather
        // than a sorted merge — the same trade the cross-column intersection makes.
        let mut seen: FxHashSet<u64> = match folded {
            Some(it) => it.collect(),
            None => self.query(acc.streams.remove(0))?.collect(),
        };
        for q in acc.streams {
            if seen.is_empty() {
                break; // a later conjunct can only remove rows
            }
            let next: FxHashSet<u64> = self.query(q)?.collect();
            seen.retain(|doc| next.contains(doc));
        }
        Some(DocIter::Set(
            seen.into_iter().collect::<Vec<_>>().into_iter(),
        ))
    }

    /// The scalar key `value` has in `kind`, or `None` when this column has never held it.
    fn key_of(
        &self,
        kind: Kind,
        value: &Value,
    ) -> Option<u64> {
        match kind {
            Kind::Numeric => NumericIndex::key_of(value),
            Kind::Tag => self.tag.key_of(value),
            Kind::Geo => GeoIndex::key_of(value),
        }
    }

    /// The scalar trees, in [`Kind::slot`] order — the tree table a cross-kind union indexes into.
    fn union_trees(&self) -> Vec<super::doc_iter::Tree> {
        vec![
            self.numeric.tree().clone(),
            self.tag.tree().clone(),
            self.geo.tree().clone(),
        ]
    }

    /// An iterator over no entries.
    fn empty(&self) -> DocIter {
        empty_docs(self.numeric.tree())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::value::Point;
    use thin_vec::ThinVec;

    fn attr() -> Arc<String> {
        Arc::new("v".to_string())
    }

    fn s(x: &str) -> Value {
        Value::String(Arc::new(x.to_string()))
    }

    fn eq(value: Value) -> IndexQuery<Value> {
        IndexQuery::Equal { key: attr(), value }
    }

    fn range(
        min: Option<Value>,
        max: Option<Value>,
        include_min: bool,
        include_max: bool,
    ) -> IndexQuery<Value> {
        IndexQuery::Range {
            key: attr(),
            min,
            max,
            include_min,
            include_max,
        }
    }

    fn ids(
        idx: &RangeIndex,
        q: &IndexQuery<Value>,
    ) -> Vec<u64> {
        let mut v: Vec<u64> = idx.query(q).expect("servable").collect();
        v.sort_unstable();
        v
    }

    /// A column holding a number, a string and a point at once — the mixed-type column
    /// `test_24_multitype_index` exercises. Each predicate must reach its own kind and see only
    /// its own values.
    fn multitype() -> RangeIndex {
        let mut idx = RangeIndex::new();
        idx.add(&Value::Int(1), 1);
        idx.add(&Value::Float(2.5), 2);
        idx.add(&s("apple"), 3);
        idx.add(&s("banana"), 4);
        idx.add(&Value::Point(Point::new(1.0, 2.0)), 5);
        idx
    }

    #[test]
    fn each_type_reaches_its_own_kind() {
        let idx = multitype();
        assert_eq!(ids(&idx, &eq(Value::Int(1))), vec![1]);
        assert_eq!(ids(&idx, &eq(Value::Float(2.5))), vec![2]);
        assert_eq!(ids(&idx, &eq(s("apple"))), vec![3]);
        assert_eq!(ids(&idx, &eq(Value::Point(Point::new(1.0, 2.0)))), vec![5]);
        // A numeric range sees only the numbers; a string range only the strings.
        assert_eq!(
            ids(
                &idx,
                &range(Some(Value::Int(0)), Some(Value::Int(10)), true, true)
            ),
            vec![1, 2]
        );
        assert_eq!(
            ids(&idx, &range(Some(s("a")), Some(s("z")), true, true)),
            vec![3, 4]
        );
    }

    /// Cypher's cross-type comparisons are unknown, so they match nothing. That is an **answer**,
    /// not a gap — declining would turn a legal query into an error.
    #[test]
    fn cross_type_and_null_comparisons_answer_empty() {
        let idx = multitype();
        assert!(ids(&idx, &eq(Value::Null)).is_empty());
        assert!(ids(&idx, &range(Some(Value::Int(0)), Some(s("z")), true, true)).is_empty());
        assert!(ids(&idx, &range(Some(Value::Null), None, true, true)).is_empty());
        assert!(
            ids(
                &idx,
                &range(Some(Value::Point(Point::new(0.0, 0.0))), None, true, true)
            )
            .is_empty(),
            "points are not ordered"
        );
    }

    /// A value no kind holds is declined, not answered empty — that is what keeps a missing kind
    /// visible instead of masquerading as "nothing matched".
    #[test]
    fn a_value_no_kind_holds_is_declined() {
        let idx = multitype();
        let list = Value::List(Arc::new(
            [Value::Int(1)].into_iter().collect::<ThinVec<_>>(),
        ));
        assert!(idx.query(&eq(list)).is_none());
        assert!(
            idx.query(&IndexQuery::ArrayContains {
                key: attr(),
                value: Value::Point(Point::new(1.0, 2.0)),
            })
            .is_none(),
            "no kind indexes a point inside a list"
        );
    }

    /// `IN [...]` over a mixed list is served from both kinds at once, and each doc appears once.
    #[test]
    fn a_mixed_union_reads_every_kind() {
        let idx = multitype();
        let q = IndexQuery::Or(vec![
            eq(Value::Int(1)),
            eq(s("banana")),
            eq(Value::Point(Point::new(1.0, 2.0))),
        ]);
        assert_eq!(ids(&idx, &q), vec![1, 4, 5]);

        // A NULL member contributes nothing rather than declining the union.
        assert_eq!(
            ids(
                &idx,
                &IndexQuery::Or(vec![eq(Value::Int(1)), eq(Value::Null)])
            ),
            vec![1]
        );
        // A member no kind holds declines the whole union: the rows it would have matched must not
        // silently vanish.
        let list = Value::List(Arc::new(
            [Value::Int(1)].into_iter().collect::<ThinVec<_>>(),
        ));
        assert!(
            idx.query(&IndexQuery::Or(vec![eq(Value::Int(1)), eq(list)]))
                .is_none()
        );
        // Duplicate members must not double-count.
        assert_eq!(
            ids(&idx, &IndexQuery::Or(vec![eq(s("apple")), eq(s("apple"))])),
            vec![3]
        );
    }

    /// String conjuncts fold by intersecting dictionary-id sets, and a conjunction across kinds is
    /// unsatisfiable rather than declined.
    #[test]
    fn conjunctions_fold_within_a_kind_and_die_across_kinds() {
        let idx = multitype();
        // 'a' <= v < 'b' AND v >= 'apple'  ->  just "apple"
        assert_eq!(
            ids(
                &idx,
                &IndexQuery::And(vec![
                    range(Some(s("a")), Some(s("b")), true, false),
                    range(Some(s("apple")), None, true, true),
                ])
            ),
            vec![3]
        );
        // Contradictory string bounds: an answer, and the answer is no rows.
        assert!(
            ids(
                &idx,
                &IndexQuery::And(vec![eq(s("apple")), eq(s("banana"))])
            )
            .is_empty()
        );
        // A number and a string at once: no scalar is both.
        assert!(
            ids(
                &idx,
                &IndexQuery::And(vec![eq(Value::Int(1)), eq(s("apple"))])
            )
            .is_empty()
        );
        // Numeric folding still works.
        assert_eq!(
            ids(
                &idx,
                &IndexQuery::And(vec![
                    range(Some(Value::Int(0)), None, true, true),
                    range(None, Some(Value::Int(2)), true, true),
                ])
            ),
            vec![1]
        );
    }

    /// List elements are split between the kinds' array trees, and a scalar predicate must not see
    /// them (nor an array probe see a scalar).
    #[test]
    fn list_elements_split_between_the_kinds() {
        let mut idx = RangeIndex::new();
        idx.add(&s("a"), 1);
        idx.add(&Value::Int(1), 2);
        idx.add(
            &Value::List(Arc::new(
                [s("a"), Value::Int(1)].into_iter().collect::<ThinVec<_>>(),
            )),
            3,
        );

        let contains = |v: Value| IndexQuery::ArrayContains {
            key: attr(),
            value: v,
        };
        assert_eq!(ids(&idx, &contains(s("a"))), vec![3]);
        assert_eq!(ids(&idx, &contains(Value::Int(1))), vec![3]);
        assert_eq!(ids(&idx, &eq(s("a"))), vec![1], "the scalar only");
        assert_eq!(ids(&idx, &eq(Value::Int(1))), vec![2], "the scalar only");
    }

    /// A distance predicate is served from the geo kind, as a superset the caller's retained
    /// filter narrows.
    #[test]
    fn a_distance_predicate_reaches_the_geo_kind() {
        let mut idx = RangeIndex::new();
        idx.add(&Value::Point(Point::new(51.5074, -0.1278)), 1); // London
        idx.add(&Value::Point(Point::new(48.8566, 2.3522)), 2); // Paris
        idx.add(&s("not a point"), 3);

        let q = IndexQuery::Point {
            key: attr(),
            point: Value::Point(Point::new(51.5, -0.12)),
            radius: Value::Int(20_000),
        };
        assert_eq!(ids(&idx, &q), vec![1]);
    }

    /// Re-indexing across kinds: the remove must route on the *old* value, or the tuple is
    /// stranded in the kind nothing will look at again.
    #[test]
    fn a_value_changing_kind_leaves_nothing_behind() {
        let mut idx = RangeIndex::new();
        idx.add(&Value::Int(1), 1);
        idx.remove(&Value::Int(1), 1);
        idx.add(&s("1"), 1);
        assert!(ids(&idx, &eq(Value::Int(1))).is_empty());
        assert_eq!(ids(&idx, &eq(s("1"))), vec![1]);
        idx.remove(&s("1"), 1);
        assert!(idx.is_empty(), "no tuple stranded in any kind");
    }

    /// The install's round trip has to carry every kind, on the same tag dictionary.
    #[test]
    fn encoded_tuples_round_trip_every_kind() {
        let idx = multitype();
        let rebuilt = idx.empty_like().from_encoded_like(idx.encoded_tuples());
        assert_eq!(ids(&rebuilt, &eq(Value::Int(1))), vec![1]);
        assert_eq!(ids(&rebuilt, &eq(s("apple"))), vec![3]);
        assert_eq!(
            ids(&rebuilt, &eq(Value::Point(Point::new(1.0, 2.0)))),
            vec![5]
        );
    }

    /// Batch maintenance must agree with the one-at-a-time path, for every kind at once.
    #[test]
    fn batch_ops_match_single_ops() {
        let entries = vec![
            (Value::Int(3), 1u64),
            (s("x"), 2),
            (Value::Point(Point::new(1.0, 1.0)), 3),
            (Value::Null, 4), // no kind: dropped by all three
        ];
        let mut batched = RangeIndex::new();
        batched.add_batch(entries.iter().cloned());
        let mut singly = RangeIndex::new();
        for (v, id) in &entries {
            singly.add(v, *id);
        }
        let all = |ix: &RangeIndex| {
            (
                ids(ix, &eq(Value::Int(3))),
                ids(ix, &eq(s("x"))),
                ids(ix, &eq(Value::Point(Point::new(1.0, 1.0)))),
            )
        };
        assert_eq!(all(&batched), all(&singly));

        batched.remove_batch(entries.iter().cloned());
        assert!(batched.is_empty());
    }

    /// Bulk build and incremental build must agree — including on the strings, which is only true
    /// if both interned into the same dictionary shape.
    #[test]
    fn bulk_build_matches_incremental() {
        let entries: Vec<(Value, u64)> = vec![
            (Value::Int(1), 1),
            (Value::Float(2.5), 2),
            (s("apple"), 3),
            (s("banana"), 4),
            (Value::Point(Point::new(1.0, 2.0)), 5),
        ];
        let bulk = RangeIndex::from_entries(entries.iter().map(|(v, id)| (v, *id)));
        let inc = multitype();
        for q in [
            eq(Value::Int(1)),
            eq(s("apple")),
            eq(Value::Point(Point::new(1.0, 2.0))),
            range(Some(s("a")), Some(s("z")), true, true),
        ] {
            assert_eq!(ids(&bulk, &q), ids(&inc, &q));
        }
    }
}
