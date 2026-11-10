//! The Range column: one `(label, attribute)` index, made of the kinds a RediSearch Range field is
//! made of — NUMERIC, TAG and GEO — each a field of [`RangeIndex`] implementing [`RangeKind`].
//!
//! A Cypher property is not typed, so one column has to hold whatever the entities put in it. The
//! kinds are therefore not alternatives but **co-resident**: a column with `v = 30` on one node,
//! `v = 'thirty'` on another and `v = point(...)` on a third stores each in the kind that can
//! order it, and a predicate is routed by the type of its operand.
//!
//! Each kind keys its trees on what orders it — a number's `u64` encoding, a string's bytes, a
//! point's Morton code — one tree per [`KeyKind`], and a column's tuples move between threads and
//! versions as [`EncodedTuples`] whatever their kind.
//!
//! Routing is exhaustive and total. Every predicate either reaches a kind, is answered empty
//! because it cannot match — a `NULL` operand, say, or a scalar predicate and a list predicate on
//! the same attribute — or is declined; and a decline is a hard error under `index-falkordb`,
//! because there is no other index to try. The three outcomes are kept apart deliberately: an empty
//! answer that should have been a decline is exactly how an unimplemented kind hides.

use std::sync::Arc;
use std::{iter, slice};

use rustc_hash::FxHashSet;

use super::encode::{ColumnKey, KeyKind, encode_numeric};
use super::geo::{GeoIndex, cover_of};
use super::numeric::{DistinctIter, DocIter, EncodedTuples, KeySet, NumericIndex};
use super::tag::{StrSel, TagIndex};
use crate::index::IndexQuery;
use crate::runtime::value::Value;

/// What a Range column needs from each of its kinds to store values and look them up.
///
/// A kind owns its trees: the tuples a value is stored under, how a sorted batch of them goes in
/// and out, which entities a set of its keys reaches, and whether a value is one of them. The
/// column reaches the trees only through this trait. What it does itself is route a value or a
/// predicate to a kind ([`Kind::of`]) and build a [`Selection`] of the kind's keys from the query.
///
/// The scan checks ([`is_scalar_in`](Self::is_scalar_in), [`is_list_holding`](Self::is_list_holding))
/// take no kind: a scan standing in for a building column holds the [`Selection`], not the column.
pub trait RangeKind {
    /// A set of this kind's keys: what a predicate over its scalars selects.
    type Keys;

    /// A list element this kind's array tree is probed for: `x IN n.v`.
    type Element;

    /// Call `emit` with each tuple `value` is stored under in this kind, as the tree it goes in and
    /// its key (a string is carried as itself, never numbered here). Nothing when
    /// the kind holds no part of `value`.
    fn for_each_stored_key(
        &self,
        value: &Value,
        emit: impl FnMut(KeyKind, ColumnKey),
    );

    /// Add this kind's share of `tuples`, which are sorted and unique
    /// ([`EncodedTuples::sort_dedup`]).
    fn add_encoded(
        &mut self,
        tuples: &EncodedTuples,
    );

    /// Remove this kind's share of `tuples`, which are sorted and unique.
    fn remove_encoded(
        &mut self,
        tuples: &EncodedTuples,
    );

    /// Whether this kind holds no tuples.
    fn is_empty(&self) -> bool;

    /// Heap bytes this kind's trees hold.
    fn memory_usage(&self) -> usize;

    /// The entities whose value is a scalar with a key in `keys`.
    fn scalars(
        &self,
        keys: Self::Keys,
    ) -> DocIter;

    /// The entities whose value is a list holding `element`.
    fn lists_holding(
        &self,
        element: &Self::Element,
    ) -> DocIter;

    /// Whether `value` is one [`scalars`](Self::scalars) returns for `keys`: what a scan standing
    /// in for a column still building checks, so it keeps exactly what the built column returns.
    fn is_scalar_in(
        keys: &Self::Keys,
        value: &Value,
    ) -> bool;

    /// Whether `value` is one [`lists_holding`](Self::lists_holding) returns for `element`.
    fn is_list_holding(
        element: &Self::Element,
        value: &Value,
    ) -> bool;
}

/// Which kind holds a scalar value, and therefore which kind answers a predicate over it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind<'a> {
    Numeric,
    /// A string, which the tag kind holds.
    Tag(&'a Arc<String>),
    /// A point, which the geo kind holds.
    Geo,
}

impl<'a> Kind<'a> {
    /// The kind that indexes `value`, or `None` for a value no kind holds (`NULL`, a temporal, a
    /// list, a map, a vector). A list is deliberately not a kind: its *elements* are
    /// indexed by the kinds' array trees, but the list itself is not a value any predicate can be
    /// routed by. Temporals are not indexed, as in the RediSearch path ([`encode_numeric`]).
    const fn of(value: &'a Value) -> Option<Self> {
        match value {
            Value::Int(_) | Value::Float(_) | Value::Bool(_) => Some(Self::Numeric),
            Value::String(s) => Some(Self::Tag(s)),
            Value::Point(_) => Some(Self::Geo),
            _ => None,
        }
    }

    /// Whether `self` and `other` are the same kind, whatever their values.
    const fn same(
        &self,
        other: &Self,
    ) -> bool {
        matches!(
            (self, other),
            (Self::Numeric, Self::Numeric) | (Self::Tag(_), Self::Tag(_)) | (Self::Geo, Self::Geo)
        )
    }
}

/// The scalars a predicate selects, per kind: the numeric keys, the strings, and the geo cells.
///
/// A scalar is of one kind, so what a conjunction selects is each kind's share intersected on its
/// own ([`and`](Self::and)) — `n.v = 1 AND n.v = 'a'` selects nothing — and what a union selects
/// is each kind's share united on its own: `n.v IN [1, 'a']` reads the numeric tree for `1` and
/// the tag tree for `'a'`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scalars {
    pub numeric: KeySet,
    pub tag: StrSel,
    pub geo: KeySet,
}

impl Scalars {
    /// Selects no scalar.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            numeric: KeySet::Empty,
            tag: StrSel::none(),
            geo: KeySet::Empty,
        }
    }

    /// The numeric scalars with a key in `keys`, and no string.
    #[must_use]
    pub const fn numeric(keys: KeySet) -> Self {
        Self {
            numeric: keys,
            tag: StrSel::none(),
            geo: KeySet::Empty,
        }
    }

    /// The strings `strings` selects, and no number.
    #[must_use]
    pub const fn tag(strings: StrSel) -> Self {
        Self {
            numeric: KeySet::Empty,
            tag: strings,
            geo: KeySet::Empty,
        }
    }

    /// The points in the geo cells `cells`, and nothing else.
    #[must_use]
    pub const fn geo(cells: KeySet) -> Self {
        Self {
            numeric: KeySet::Empty,
            tag: StrSel::none(),
            geo: cells,
        }
    }

    /// Whether this selects no scalar by construction.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.numeric == KeySet::Empty && self.tag.is_empty() && self.geo == KeySet::Empty
    }

    /// Whether `value` is a scalar this selects.
    fn contains(
        &self,
        value: &Value,
    ) -> bool {
        NumericIndex::is_scalar_in(&self.numeric, value)
            || TagIndex::is_scalar_in(&self.tag, value)
            || GeoIndex::is_scalar_in(&self.geo, value)
    }

    /// What both select, kind by kind.
    fn and(
        self,
        other: Self,
    ) -> Self {
        Self {
            numeric: self.numeric.and(other.numeric),
            tag: self.tag.intersect(other.tag),
            geo: self.geo.and(other.geo),
        }
    }
}

/// Whether `value` is a list holding the element whose array-tree key is `element`.
fn is_held_by(
    element: &ColumnKey,
    value: &Value,
) -> bool {
    match element {
        ColumnKey::Encoded(key) => NumericIndex::is_list_holding(key, value),
        ColumnKey::Tag(s) => TagIndex::is_list_holding(s, value),
    }
}

/// What a column answers a predicate on one attribute with: the entities whose value it selects.
///
/// Built from a query without reading the column, so it serves twice: a `Ready` column looks it up
/// ([`RangeIndex::lookup`]), and a column that is still `Building` hands it to a scan that keeps
/// the entities whose value it [`matches`](Self::matches).
///
/// A stored value is a scalar or a list, never both, so a selection is of one or the other, and a
/// conjunction of a scalar predicate with a list predicate selects nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    /// Scalar values, per kind: `=`, a range, `IN`, and conjunctions and unions of them.
    Scalars(Scalars),
    /// List values holding `first` and each of `rest`, by their array-tree keys: `x IN n.v`, and
    /// conjunctions of it.
    ListsHolding {
        first: ColumnKey,
        rest: Vec<ColumnKey>,
    },
}

impl Selection {
    /// Selects nothing.
    #[must_use]
    pub const fn empty() -> Self {
        Self::Scalars(Scalars::none())
    }

    /// Whether this selects nothing by construction.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Scalars(scalars) if scalars.is_empty())
    }

    /// Whether the column selects an entity holding `value`: the scan's check while the column
    /// is building, which must keep exactly the entities a lookup of the built column returns.
    #[must_use]
    pub fn matches(
        &self,
        value: &Value,
    ) -> bool {
        match self {
            Self::Scalars(scalars) => scalars.contains(value),
            Self::ListsHolding { first, rest } => iter::once(first)
                .chain(rest)
                .all(|element| is_held_by(element, value)),
        }
    }

    /// The selection of a predicate on **one** attribute, or `None` when no kind serves it.
    #[must_use]
    pub fn of(query: &IndexQuery<Value>) -> Option<Self> {
        match query {
            IndexQuery::Equal { value, .. } => Self::equal(value),
            IndexQuery::Range {
                min,
                max,
                include_min,
                include_max,
                ..
            } => Self::range(min.as_ref(), max.as_ref(), *include_min, *include_max),
            // An empty list is declined rather than answered as "no rows": the scan op drops the
            // members it cannot index before the list gets here, so `IN [date(..)]` arrives empty,
            // and "no rows" would miss every entity holding that date.
            IndexQuery::InList {
                list: Value::List(items),
                ..
            } if !items.is_empty() => Self::equal_to_any(items),
            IndexQuery::InList { .. } => None,
            IndexQuery::ArrayContains { value, .. } => Self::lists_holding(value),
            IndexQuery::Or(_) => Self::disjunction(&atoms(query)?),
            IndexQuery::And(_) => Self::conjunction(&atoms(query)?),
            IndexQuery::Point { point, radius, .. } => Self::within(point, radius),
        }
    }

    /// The scalars equal to `value`: `n.v = value`. A `NULL` matches nothing — `n.v = null` is
    /// unknown in Cypher — and is answered empty; a value no kind holds is declined.
    fn equal(value: &Value) -> Option<Self> {
        match Kind::of(value) {
            Some(Kind::Numeric) => Some(Self::Scalars(Scalars::numeric(
                encode_numeric(value).map_or(KeySet::Empty, KeySet::Point),
            ))),
            Some(Kind::Tag(s)) => Some(Self::Scalars(Scalars::tag(StrSel::Set(vec![s.clone()])))),
            // The grid cell, a superset of the equal points: see the geo module on why the filter
            // is always retained over a point operand.
            Some(Kind::Geo) => Some(Self::Scalars(Scalars::geo(
                GeoIndex::key_of(value).map_or(KeySet::Empty, KeySet::Point),
            ))),
            None if matches!(value, Value::Null) => Some(Self::empty()),
            None => None,
        }
    }

    /// The scalars equal to any of `values`: `n.v IN [..]`.
    ///
    /// Each value is routed as [`equal`](Self::equal) routes it, so `x IN [1, null]` matches only
    /// `1`. A value no kind holds declines the **whole** list: answering from the rest would
    /// silently drop the rows that value would have matched.
    fn equal_to_any(values: &[Value]) -> Option<Self> {
        let mut equal = EqualValues::default();
        equal.push(values)?;
        Some(equal.selection())
    }

    /// The selection of `n.v <op> bound`.
    ///
    /// Cypher orders numbers only with numbers, so a range with a `NULL` bound matches nothing and
    /// is answered **empty** rather than declined: that is not a gap in the index, it is the
    /// language. A bound of a type no kind holds is declined.
    ///
    /// Cypher orders strings with strings too, so a range whose bounds are of two kinds
    /// (`n.v >= 1 AND n.v < 'z'` merged into one) is answered empty as well.
    fn range(
        min: Option<&Value>,
        max: Option<&Value>,
        include_min: bool,
        include_max: bool,
    ) -> Option<Self> {
        let mut null_bound = false;
        let mut kind: Option<Kind> = None;
        let mut mixed = false;
        for bound in [min, max].into_iter().flatten() {
            match Kind::of(bound) {
                Some(this) => match &kind {
                    Some(seen) if !seen.same(&this) => mixed = true,
                    Some(_) => {}
                    None => kind = Some(this),
                },
                None if matches!(bound, Value::Null) => null_bound = true,
                None => return None,
            }
        }
        if null_bound || mixed {
            return Some(Self::empty());
        }
        let as_str = |bound: Option<&Value>| match bound {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        };
        Some(Self::Scalars(match kind {
            Some(Kind::Tag(_)) => Scalars::tag(StrSel::window(
                as_str(min),
                as_str(max),
                include_min,
                include_max,
            )),
            // Points compare (`compare_value` orders them by longitude, then latitude), but the
            // Morton key does not follow that order. Every point is the superset: the planner keeps
            // the filter over a point operand, which is never a bare literal, so it narrows the
            // rows to the true comparison.
            Some(Kind::Geo) => Scalars::geo(KeySet::Range(0, u64::MAX)),
            Some(Kind::Numeric) | None => {
                Scalars::numeric(NumericIndex::range_keys(min, max, include_min, include_max))
            }
        }))
    }

    /// The selection of `value IN n.v`, where `n.v` is a list: the lists holding `value`.
    ///
    /// The numeric and tag kinds have array trees, as RediSearch has `numeric:arr` and
    /// `string:arr`. A probe of any other type (a `NULL` included) is declined and stays visible as
    /// a gap: RediSearch indexes no point inside a list either.
    fn lists_holding(value: &Value) -> Option<Self> {
        let first = match Kind::of(value)? {
            Kind::Numeric => match encode_numeric(value) {
                Some(key) => ColumnKey::Encoded(key),
                None => return Some(Self::empty()),
            },
            Kind::Tag(s) => ColumnKey::Tag(s.clone()),
            Kind::Geo => return None,
        };
        Some(Self::ListsHolding {
            first,
            rest: Vec::new(),
        })
    }

    /// The selection of `distance(n.v, centre) < radius` — the geo cells covering the disk's
    /// bounding box, a superset the caller's retained filter narrows. A `NULL` centre or radius
    /// makes the predicate `NULL`, which holds for nothing: answered empty. So does a radius that
    /// is not a number: comparing a distance with a string or a list is `NULL` too. A centre that
    /// is not a point is declined, since then the predicate is not the shape the planner
    /// promised.
    fn within(
        centre: &Value,
        radius: &Value,
    ) -> Option<Self> {
        if matches!(centre, Value::Null) || matches!(radius, Value::Null) {
            return Some(Self::empty());
        }
        let Value::Point(centre) = centre else {
            return None;
        };
        let radius = match radius {
            Value::Int(i) => *i as f64,
            Value::Float(f) => *f,
            _ => return Some(Self::empty()),
        };
        Some(Self::Scalars(Scalars::geo(KeySet::of_windows(cover_of(
            centre, radius,
        )))))
    }

    /// An `OR` of `=`s and `IN` lists, all on one attribute — `a = 1 OR a IN [2, 3]`: the scalars
    /// equal to any of their values. `members` are the `OR`'s [`atoms`], so nested `OR`s are
    /// flattened: the nesting carries no meaning for one column.
    ///
    /// Every member must target the same attribute: answering `a = 1 OR b = 2` from one column
    /// would look up both values in `a` and return wrong rows, not missing ones. Any other member —
    /// a range, an array-contains, a conjunction — declines the union, and so does an empty `IN`
    /// list (for [`of`](Self::of)'s reason) and a value no kind holds.
    pub(super) fn disjunction(members: &[&IndexQuery<Value>]) -> Option<Self> {
        if !on_one_attribute(members) {
            return None;
        }
        let mut equal = EqualValues::default();
        for &member in members {
            let values = match member {
                IndexQuery::Equal { value, .. } => slice::from_ref(value),
                IndexQuery::InList {
                    list: Value::List(items),
                    ..
                } if !items.is_empty() => items.as_slice(),
                _ => return None,
            };
            equal.push(values)?;
        }
        Some(equal.selection())
    }

    /// A conjunction of predicates on **one** attribute: what every conjunct selects
    /// ([`and`](Self::and)). On scalars that is arithmetic on keys ([`KeySet::and`]), so
    /// `n.v = 5 AND n.v > 2` reads one key and `n.v >= 0 AND n.v IN [5, 7]` reads two, not the
    /// window.
    ///
    /// `conjuncts` are atoms — no conjunct is itself an `AND`: an `AND`'s [`atoms`], or one
    /// attribute's share of a conjunction across several, which `intersect_columns` has
    /// flattened.
    ///
    /// A conjunct that selects nothing (a `NULL` operand) empties the conjunction, even beside one
    /// no kind serves: `n.v = null AND <anything>` holds for nothing. Otherwise a conjunct no kind
    /// serves declines the conjunction, and so does one on another attribute.
    pub(super) fn conjunction(conjuncts: &[&IndexQuery<Value>]) -> Option<Self> {
        debug_assert!(
            !conjuncts.iter().any(|q| matches!(q, IndexQuery::And(_))),
            "conjuncts are atoms: a nested `AND` that folds to empty would answer what the flat \
             conjunction declines"
        );
        if !on_one_attribute(conjuncts) {
            return None;
        }
        let selections: Vec<Option<Self>> = conjuncts.iter().copied().map(Self::of).collect();
        if selections.iter().flatten().any(Self::is_empty) {
            return Some(Self::empty());
        }
        selections
            .into_iter()
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .reduce(Self::and)
    }

    /// What both select. A value is a scalar or a list, never both, so a scalar selection and a list
    /// selection select nothing together.
    fn and(
        self,
        other: Self,
    ) -> Self {
        match (self, other) {
            (Self::Scalars(a), Self::Scalars(b)) => Self::Scalars(a.and(b)),
            (
                Self::ListsHolding { first, mut rest },
                Self::ListsHolding {
                    first: other_first,
                    rest: other_rest,
                },
            ) => {
                rest.push(other_first);
                rest.extend(other_rest);
                Self::ListsHolding { first, rest }
            }
            _ => Self::empty(),
        }
    }
}

/// The values a union of equalities selects, per kind: `n.v IN [1, 'a', null]`.
#[derive(Default)]
struct EqualValues {
    keys: Vec<u64>,
    strings: Vec<Arc<String>>,
    cells: Vec<u64>,
}

impl EqualValues {
    /// Add what [`Selection::equal`] selects for each of `values`, or decline (`None`) at a value
    /// no kind holds. A `NULL`, or a number with no key (`NaN`), matches nothing and adds nothing.
    fn push(
        &mut self,
        values: &[Value],
    ) -> Option<()> {
        for value in values {
            match Kind::of(value) {
                Some(Kind::Numeric) => self.keys.extend(encode_numeric(value)),
                Some(Kind::Tag(s)) => self.strings.push(s.clone()),
                Some(Kind::Geo) => self.cells.extend(GeoIndex::key_of(value)),
                None if matches!(value, Value::Null) => {}
                None => return None,
            }
        }
        Some(())
    }

    /// The scalars equal to any of the values pushed.
    fn selection(self) -> Selection {
        Selection::Scalars(Scalars {
            numeric: KeySet::of_keys(self.keys),
            tag: StrSel::set(self.strings),
            geo: KeySet::of_keys(self.cells),
        })
    }
}

/// The atoms of an `AND` or an `OR`: its members, with each member that is the same combinator
/// opened into its own — `And(And(a, b), c)` constrains what `And(a, b, c)` does. A member that is
/// the other combinator is one atom: `a AND (b OR c)` has two. Any other predicate is its own only
/// atom. `None` when the combinator, or one opened in it, has no members: an empty `AND` or `OR`
/// names no column.
fn atoms(query: &IndexQuery<Value>) -> Option<Vec<&IndexQuery<Value>>> {
    let (IndexQuery::And(members) | IndexQuery::Or(members)) = query else {
        return Some(vec![query]);
    };
    if members.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for member in members {
        let same = matches!(
            (query, member),
            (IndexQuery::And(_), IndexQuery::And(_)) | (IndexQuery::Or(_), IndexQuery::Or(_))
        );
        if same {
            out.extend(atoms(member)?);
        } else {
            out.push(member);
        }
    }
    Some(out)
}

/// Whether every one of `queries` constrains the same single attribute: what one column can answer.
fn on_one_attribute(queries: &[&IndexQuery<Value>]) -> bool {
    let mut attributes = Vec::new();
    queries
        .iter()
        .all(|query| attributes_of(query, &mut attributes))
        && attributes.windows(2).all(|pair| pair[0] == pair[1])
}

/// Collect the attributes a predicate constrains, flattening nested composites. `false` when the
/// shape names no attribute at all — an empty `And`/`Or`.
///
/// Deliberately says nothing about *values*: which types a column can serve is decided inside the
/// column, where the kinds live.
pub(super) fn attributes_of<'a>(
    query: &'a IndexQuery<Value>,
    out: &mut Vec<&'a Arc<String>>,
) -> bool {
    match query {
        IndexQuery::Equal { key, .. }
        | IndexQuery::Range { key, .. }
        | IndexQuery::ArrayContains { key, .. }
        | IndexQuery::Point { key, .. }
        | IndexQuery::InList { key, .. } => {
            out.push(key);
            true
        }
        IndexQuery::And(children) | IndexQuery::Or(children) => {
            !children.is_empty() && children.iter().all(|c| attributes_of(c, out))
        }
    }
}

/// One indexed `(label, attribute)`: the kinds of the same column.
///
/// `Clone` is `O(1)` — a root-`Arc` bump per tree — so a graph version forks its index snapshot
/// cheaply.
///
/// A method that asks every kind — the writes, `is_empty`, `memory_usage` — starts with
/// `let Self { numeric, tag, geo } = self`, a pattern without `..`: a new kind is a compile error in each of
/// them until it is asked too. A method that routes to one kind matches on [`Kind`], [`KeyKind`]
/// or [`Selection`], so the compiler finds it once the new kind adds a variant to the one it
/// matches on.
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

    /// The numeric kind, for tests that look at its trees.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn numeric(&self) -> &NumericIndex {
        &self.numeric
    }

    /// The tag kind, for tests that look at its tuples.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn tag(&self) -> &TagIndex {
        &self.tag
    }

    /// Call `emit` with each key a stored `value` takes in this column, and its [`KeyKind`]: a
    /// scalar lands in at most one kind, and a list's elements in the kinds' array trees. A string
    /// is passed as itself ([`ColumnKey::Tag`]): it is numbered only by the column copy its tuple
    /// goes into ([`add_encoded`](Self::add_encoded)).
    pub fn for_each_stored_key(
        &self,
        value: &Value,
        mut emit: impl FnMut(KeyKind, ColumnKey),
    ) {
        let Self { numeric, tag, geo } = self;
        numeric.for_each_stored_key(value, &mut emit);
        tag.for_each_stored_key(value, &mut emit);
        geo.for_each_stored_key(value, emit);
    }

    /// Append the tuples `(value, id)` takes in this column to `out`, each with the tuples of its
    /// kind. Unsorted and possibly repeated: see [`EncodedTuples::sort_dedup`].
    pub fn push_encoded(
        &self,
        value: &Value,
        id: u64,
        out: &mut EncodedTuples,
    ) {
        self.for_each_stored_key(value, |kind, key| out.push(kind, key, id));
    }

    /// Index `id` under `value`.
    pub fn add(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        let mut tuples = EncodedTuples::default();
        self.push_encoded(value, id, &mut tuples);
        self.add_encoded(&mut tuples);
    }

    /// Remove `id` from under `value` — the old value on delete/update. Routes on the **old**
    /// value's kind, or the tuple is stranded in the kind nothing will look at again.
    pub fn remove(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        let mut tuples = EncodedTuples::default();
        self.push_encoded(value, id, &mut tuples);
        self.remove_encoded(&mut tuples);
    }

    /// Add a batch of `(value, id)` entries — one batched tree op per kind.
    ///
    /// `scratch` is the caller's encode buffer and its contents are replaced. Taking it rather than
    /// allocating internally lets a commit that touches many columns allocate once and reuse the
    /// capacity, instead of a fresh buffer per column.
    pub fn add_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut EncodedTuples,
    ) {
        scratch.clear();
        for (value, id) in entries {
            self.push_encoded(&value, id, scratch);
        }
        self.add_encoded(scratch);
    }

    /// Remove a batch of `(value, id)` entries. `scratch` as in [`add_batch`](Self::add_batch).
    pub fn remove_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut EncodedTuples,
    ) {
        scratch.clear();
        for (value, id) in entries {
            self.push_encoded(&value, id, scratch);
        }
        self.remove_encoded(scratch);
    }

    /// Add already-encoded tuples (a unit of work's collapsed adds, or an indexer batch), each
    /// kind into its own trees.
    pub fn add_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.sort_dedup();
        let Self { numeric, tag, geo } = self;
        numeric.add_encoded(tuples);
        tag.add_encoded(tuples);
        geo.add_encoded(tuples);
    }

    /// Remove already-encoded tuples (a unit of work's collapsed removes).
    pub fn remove_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        tuples.sort_dedup();
        let Self { numeric, tag, geo } = self;
        numeric.remove_encoded(tuples);
        tag.remove_encoded(tuples);
        geo.remove_encoded(tuples);
    }

    /// Whether the column holds no tuples in any kind.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let Self { numeric, tag, geo } = self;
        numeric.is_empty() && tag.is_empty() && geo.is_empty()
    }

    /// Heap bytes every kind's trees hold.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let Self { numeric, tag, geo } = self;
        numeric.memory_usage() + tag.memory_usage() + geo.memory_usage()
    }

    /// The scalar tree and key `value` takes, or `None` when no kind indexes it as a scalar (a
    /// list: its elements are in the array trees, which say which lists contain a value, not
    /// which equal a list).
    #[must_use]
    pub fn encode_value(
        &self,
        value: &Value,
    ) -> Option<(KeyKind, ColumnKey)> {
        match Kind::of(value)? {
            Kind::Numeric => {
                encode_numeric(value).map(|k| (KeyKind::Scalar, ColumnKey::Encoded(k)))
            }
            Kind::Tag(s) => Some((KeyKind::Tag, ColumnKey::Tag(s.clone()))),
            Kind::Geo => GeoIndex::key_of(value).map(|k| (KeyKind::Geo, ColumnKey::Encoded(k))),
        }
    }

    /// The ids stored under `key` in the tree of `kind`.
    #[must_use]
    pub fn rows_at(
        &self,
        kind: KeyKind,
        key: &ColumnKey,
    ) -> DocIter {
        match (kind, key) {
            (KeyKind::Scalar, ColumnKey::Encoded(k)) => self.numeric.scalars(KeySet::Point(*k)),
            (KeyKind::Array, ColumnKey::Encoded(k)) => self.numeric.lists_holding(k),
            (KeyKind::Tag | KeyKind::TagArray, ColumnKey::Tag(s)) => self.tag.rows_at(kind, s),
            (KeyKind::Geo, ColumnKey::Encoded(k)) => self.geo.scalars(KeySet::Point(*k)),
            _ => unreachable!("a key of the kind's own type"),
        }
    }

    /// The entities `selection` selects in this column.
    #[must_use]
    pub fn lookup(
        &self,
        selection: Selection,
    ) -> DocIter {
        match selection {
            Selection::Scalars(Scalars { numeric, tag, geo }) => {
                // A scalar is of one kind, so the kinds' streams hold no doc in common, and an
                // empty share is not read.
                let mut streams = Vec::new();
                if numeric != KeySet::Empty {
                    streams.push(self.numeric.scalars(numeric));
                }
                if !tag.is_empty() {
                    streams.push(self.tag.scalars(tag));
                }
                if geo != KeySet::Empty {
                    streams.push(self.geo.scalars(geo));
                }
                match streams.len() {
                    0 => self.numeric.scalars(KeySet::Empty),
                    1 => streams.pop().expect("one stream"),
                    _ => DocIter::Distinct(DistinctIter::new(streams)),
                }
            }
            Selection::ListsHolding { first, rest } => {
                let lists = self.lists_holding(&first);
                if rest.is_empty() {
                    return lists;
                }
                // Every element must be held. A probe yields docs in key order, not id order, so
                // this is a set probe rather than a merge: the first probe's lists are
                // materialised and each later probe can only narrow them.
                let mut found: FxHashSet<u64> = lists.collect();
                for element in rest {
                    if found.is_empty() {
                        break;
                    }
                    found = self
                        .lists_holding(&element)
                        .filter(|doc| found.contains(doc))
                        .collect();
                }
                DocIter::Set(found.into_iter())
            }
        }
    }

    /// The lists holding `element`, in the kind that holds it.
    fn lists_holding(
        &self,
        element: &ColumnKey,
    ) -> DocIter {
        match element {
            ColumnKey::Encoded(key) => self.numeric.lists_holding(key),
            ColumnKey::Tag(s) => self.tag.lists_holding(s),
        }
    }

    /// Answer a predicate on this column, or decline it (`None`).
    #[must_use]
    pub fn query(
        &self,
        query: &IndexQuery<Value>,
    ) -> Option<DocIter> {
        Selection::of(query).map(|selection| self.lookup(selection))
    }

    /// Every `(kind, key, doc)` tuple the column holds, in kind order. For tests that compare a
    /// column with what it must hold.
    #[cfg(test)]
    #[must_use]
    pub fn tuples(&self) -> Vec<(KeyKind, ColumnKey, u64)> {
        let Self { numeric, tag, geo } = self;
        let mut tuples: Vec<_> = numeric
            .tuples()
            .into_iter()
            .chain(geo.tuples())
            .map(|(kind, key, doc)| (kind, ColumnKey::Encoded(key), doc))
            .collect();
        tuples.extend(
            tag.tuples()
                .into_iter()
                .map(|(kind, s, doc)| (kind, ColumnKey::Tag(Arc::new(s)), doc)),
        );
        tuples
    }

    /// The scalars equal to the number `value`, reached as a query reaches them. For tests.
    #[cfg(test)]
    pub(crate) fn point(
        &self,
        value: &Value,
    ) -> DocIter {
        self.lookup(Selection::equal(value).expect("a numeric value"))
    }

    /// The scalars from `min` to `max`, numbers or unbounded, reached as a query reaches them. For
    /// tests.
    #[cfg(test)]
    pub(crate) fn range(
        &self,
        min: Option<&Value>,
        max: Option<&Value>,
        include_min: bool,
        include_max: bool,
    ) -> DocIter {
        self.lookup(Selection::range(min, max, include_min, include_max).expect("numeric bounds"))
    }

    /// The lists holding the number `value`, reached as a query reaches them. For tests.
    #[cfg(test)]
    pub(crate) fn array_contains(
        &self,
        value: &Value,
    ) -> DocIter {
        self.lookup(Selection::lists_holding(value).expect("a numeric value"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::graph::{NodeId, RelationshipId};
    use crate::runtime::value::Point;
    use thin_vec::ThinVec;

    fn attr() -> Arc<String> {
        Arc::new("v".to_string())
    }

    fn eq(value: Value) -> IndexQuery<Value> {
        IndexQuery::Equal { key: attr(), value }
    }

    fn range(
        min: Option<Value>,
        max: Option<Value>,
    ) -> IndexQuery<Value> {
        IndexQuery::Range {
            key: attr(),
            min,
            max,
            include_min: true,
            include_max: true,
        }
    }

    fn contains(value: Value) -> IndexQuery<Value> {
        IndexQuery::ArrayContains { key: attr(), value }
    }

    fn list(values: Vec<Value>) -> Value {
        Value::List(Arc::new(values.into_iter().collect::<ThinVec<_>>()))
    }

    /// The rows of a servable predicate, sorted — checked against the scan's answer, so a
    /// selection that a lookup and a scan disagree on fails here.
    fn ids(
        idx: &RangeIndex,
        q: &IndexQuery<Value>,
        values: &[(Value, u64)],
    ) -> Vec<u64> {
        let selection = Selection::of(q).expect("servable");
        let scanned: Vec<u64> = values
            .iter()
            .filter(|(v, _)| selection.matches(v))
            .map(|&(_, id)| id)
            .collect();
        let mut looked_up: Vec<u64> = idx.lookup(selection).collect();
        looked_up.sort_unstable();
        assert_eq!(
            looked_up, scanned,
            "a lookup and a scan must agree on {q:?}"
        );
        looked_up
    }

    fn sample() -> (RangeIndex, Vec<(Value, u64)>) {
        let values = vec![
            (Value::Int(1), 1),
            (Value::Float(2.5), 2),
            (Value::Int(7), 3),
            (list(vec![Value::Int(1), Value::Int(9)]), 4),
            (Value::String(Arc::new("x".to_string())), 5),
            (list(vec![Value::Int(1), Value::Int(7)]), 6),
        ];
        let mut idx = RangeIndex::new();
        for (v, id) in &values {
            idx.add(v, *id);
        }
        (idx, values)
    }

    #[test]
    fn points_ranges_and_array_probes_agree_with_the_scan() {
        let (idx, vs) = sample();
        assert_eq!(
            ids(&idx, &eq(Value::Int(1)), &vs),
            vec![1],
            "the scalar only"
        );
        assert_eq!(ids(&idx, &eq(Value::Float(2.5)), &vs), vec![2]);
        assert_eq!(
            ids(&idx, &range(Some(Value::Int(0)), Some(Value::Int(5))), &vs),
            vec![1, 2]
        );
        assert_eq!(ids(&idx, &contains(Value::Int(9)), &vs), vec![4]);
        assert_eq!(ids(&idx, &contains(Value::Int(1)), &vs), vec![4, 6]);
    }

    /// `NULL` operands are answered empty — `n.v = null` is unknown in Cypher. A value no kind
    /// holds is declined, which keeps a missing kind visible instead of masquerading as "nothing
    /// matched".
    #[test]
    fn null_answers_empty_and_an_unheld_type_declines() {
        let (idx, vs) = sample();
        assert!(ids(&idx, &eq(Value::Null), &vs).is_empty());
        assert!(ids(&idx, &range(Some(Value::Null), None), &vs).is_empty());
        assert!(idx.query(&eq(Value::Date(1))).is_none());
        assert!(idx.query(&eq(list(vec![Value::Int(1)]))).is_none());
        assert!(
            idx.query(&range(Some(Value::Null), Some(Value::Date(1))))
                .is_none(),
            "a bound no kind holds declines, even beside a NULL one"
        );
        assert!(idx.query(&contains(Value::Null)).is_none());
    }

    /// A union drops `NULL` members, dedups repeated ones, and declines whole if a member's type
    /// has no kind.
    #[test]
    fn a_union_of_values() {
        let (idx, vs) = sample();
        let q = IndexQuery::Or(vec![eq(Value::Int(1)), eq(Value::Null), eq(Value::Int(7))]);
        assert_eq!(ids(&idx, &q, &vs), vec![1, 3]);
        let in_list = IndexQuery::InList {
            key: attr(),
            list: list(vec![Value::Int(7), Value::Int(7)]),
        };
        assert_eq!(ids(&idx, &in_list, &vs), vec![3]);
        assert!(
            idx.query(&IndexQuery::Or(vec![
                eq(Value::Int(1)),
                eq(list(vec![Value::Int(1)]))
            ]))
            .is_none()
        );
        assert!(
            idx.query(&IndexQuery::Or(vec![
                contains(Value::Int(1)),
                contains(Value::Int(9))
            ]))
            .is_none(),
            "a union of array probes is not a union of keys: a list holding both would come twice"
        );
    }

    /// Bounds on one attribute fold into one window and an `IN` list into a point set, so a
    /// conjunction on scalars stays one lookup; array probes intersect by entity.
    #[test]
    fn conjunctions_fold_and_intersect() {
        let (idx, vs) = sample();
        let q = IndexQuery::And(vec![
            range(Some(Value::Int(0)), None),
            range(None, Some(Value::Int(2))),
        ]);
        assert_eq!(ids(&idx, &q, &vs), vec![1]);
        let q = IndexQuery::And(vec![
            IndexQuery::Or(vec![eq(Value::Int(1)), eq(Value::Int(7))]),
            range(Some(Value::Int(5)), None),
        ]);
        assert_eq!(ids(&idx, &q, &vs), vec![3]);
        let q = IndexQuery::And(vec![eq(Value::Int(1)), eq(Value::Null)]);
        assert!(ids(&idx, &q, &vs).is_empty());
        // A `NULL` conjunct empties the conjunction even beside one no kind serves.
        let q = IndexQuery::And(vec![eq(Value::Null), eq(list(vec![Value::Int(1)]))]);
        assert!(ids(&idx, &q, &vs).is_empty());
        let q = IndexQuery::And(vec![eq(Value::Int(1)), eq(list(vec![Value::Int(1)]))]);
        assert!(
            idx.query(&q).is_none(),
            "without one, that conjunct declines it"
        );
        // An `IN` list folds into the window as a point set: it reads its keys, not the window.
        let in_list = |vals: Vec<Value>| IndexQuery::InList {
            key: attr(),
            list: list(vals),
        };
        let q = IndexQuery::And(vec![
            range(Some(Value::Int(2)), None),
            in_list(vec![Value::Int(1), Value::Int(7)]),
        ]);
        assert_eq!(
            Selection::of(&q),
            Some(Selection::Scalars(Scalars::numeric(KeySet::Point(
                encode_numeric(&Value::Int(7)).unwrap()
            )))),
            "one key to read"
        );
        assert_eq!(ids(&idx, &q, &vs), vec![3]);
        let q = IndexQuery::And(vec![
            in_list(vec![Value::Int(1), Value::Int(7)]),
            in_list(vec![Value::Int(7), Value::Int(9)]),
        ]);
        assert_eq!(ids(&idx, &q, &vs), vec![3], "two lists intersect");
        let q = IndexQuery::And(vec![
            range(Some(Value::Int(8)), None),
            in_list(vec![Value::Int(1), Value::Int(7)]),
        ]);
        assert!(
            ids(&idx, &q, &vs).is_empty(),
            "the window excludes every key"
        );
        // Array probes: a list must hold every probed element.
        let q = IndexQuery::And(vec![contains(Value::Int(1)), contains(Value::Int(9))]);
        assert_eq!(ids(&idx, &q, &vs), vec![4]);
        let q = IndexQuery::And(vec![contains(Value::Int(9)), contains(Value::Int(7))]);
        assert!(ids(&idx, &q, &vs).is_empty(), "no list holds both");
        // No value is both a scalar and a list.
        let q = IndexQuery::And(vec![eq(Value::Int(1)), contains(Value::Int(1))]);
        assert_eq!(Selection::of(&q), Some(Selection::empty()));
        assert!(ids(&idx, &q, &vs).is_empty());
    }

    fn s(x: &str) -> Value {
        Value::String(Arc::new(x.to_string()))
    }

    fn range_ex(
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

    fn pt(
        lat: f32,
        lon: f32,
    ) -> Value {
        Value::Point(Point::new(lat, lon))
    }

    /// A column holding numbers, strings and a point at once — the mixed-type column
    /// `test_24_multitype_index` exercises. Each predicate must reach its own kind and see only
    /// its own values.
    fn multitype() -> (RangeIndex, Vec<(Value, u64)>) {
        let values = vec![
            (Value::Int(1), 1),
            (Value::Float(2.5), 2),
            (s("apple"), 3),
            (s("banana"), 4),
            (pt(1.0, 2.0), 5),
        ];
        let mut idx = RangeIndex::new();
        for (v, id) in &values {
            idx.add(v, *id);
        }
        (idx, values)
    }

    #[test]
    fn each_type_reaches_its_own_kind() {
        let (idx, vs) = multitype();
        assert_eq!(ids(&idx, &eq(Value::Int(1)), &vs), vec![1]);
        assert_eq!(ids(&idx, &eq(s("apple")), &vs), vec![3]);
        assert_eq!(ids(&idx, &eq(pt(1.0, 2.0)), &vs), vec![5]);
        // A numeric range sees only the numbers; a string range only the strings.
        assert_eq!(
            ids(&idx, &range(Some(Value::Int(0)), Some(Value::Int(10))), &vs),
            vec![1, 2]
        );
        assert_eq!(
            ids(&idx, &range(Some(s("a")), Some(s("z"))), &vs),
            vec![3, 4]
        );
    }

    /// Cypher's cross-type comparisons are unknown, so they match nothing. That is an **answer**,
    /// not a gap — declining would turn a legal query into an error.
    #[test]
    fn cross_type_comparisons_answer_empty() {
        let (idx, vs) = multitype();
        assert!(ids(&idx, &range(Some(Value::Int(0)), Some(s("z"))), &vs).is_empty());
        assert!(ids(&idx, &range(Some(s("a")), Some(Value::Int(9))), &vs).is_empty());
        assert!(ids(&idx, &range(Some(pt(0.0, 0.0)), Some(Value::Int(9))), &vs).is_empty());
        // A point range is answered with every point: the Morton key does not follow the order
        // points compare in, and the retained filter narrows the superset.
        assert_eq!(ids(&idx, &range(Some(pt(0.0, 0.0)), None), &vs), vec![5]);
        // A `NULL` centre or radius makes `distance(..) < r` `NULL`, and so does a radius that is
        // not a number: no rows, not a decline.
        for (point, radius) in [
            (Value::Null, Value::Int(100)),
            (pt(1.0, 2.0), Value::Null),
            (pt(1.0, 2.0), s("100")),
        ] {
            let q = IndexQuery::Point {
                key: attr(),
                point,
                radius,
            };
            assert!(ids(&idx, &q, &vs).is_empty());
        }
        let q = IndexQuery::Point {
            key: attr(),
            point: Value::Int(1),
            radius: Value::Int(100),
        };
        assert!(idx.query(&q).is_none(), "a centre that is not a point");
    }

    /// `IN [...]` over a mixed list is served from every kind at once, and each doc appears once.
    #[test]
    fn a_mixed_union_reads_every_kind() {
        let (idx, vs) = multitype();
        let q = IndexQuery::Or(vec![eq(Value::Int(1)), eq(s("banana")), eq(pt(1.0, 2.0))]);
        assert_eq!(ids(&idx, &q, &vs), vec![1, 4, 5]);
        let in_list = IndexQuery::InList {
            key: attr(),
            list: list(vec![Value::Float(2.5), s("apple"), s("apple")]),
        };
        assert_eq!(ids(&idx, &in_list, &vs), vec![2, 3]);
        let q = IndexQuery::Or(vec![eq(s("apple")), eq(Value::Null)]);
        assert_eq!(ids(&idx, &q, &vs), vec![3], "a NULL member adds nothing");
    }

    /// String conjuncts fold by intersecting what they select, and a conjunction across kinds is
    /// unsatisfiable rather than declined.
    #[test]
    fn conjunctions_fold_within_a_kind_and_die_across_kinds() {
        let (idx, vs) = multitype();
        // 'a' <= v < 'b' AND v >= 'apple'  ->  just "apple"
        let q = IndexQuery::And(vec![
            range_ex(Some(s("a")), Some(s("b")), true, false),
            range_ex(Some(s("apple")), None, true, true),
        ]);
        assert_eq!(ids(&idx, &q, &vs), vec![3]);
        let q = IndexQuery::And(vec![eq(s("apple")), eq(s("banana"))]);
        assert!(ids(&idx, &q, &vs).is_empty(), "contradictory strings");
        let q = IndexQuery::And(vec![eq(Value::Int(1)), eq(s("apple"))]);
        assert_eq!(
            Selection::of(&q),
            Some(Selection::empty()),
            "no scalar is a number and a string"
        );
        // A mixed union inside a conjunction keeps the share of the kind the rest constrains.
        let q = IndexQuery::And(vec![
            IndexQuery::Or(vec![eq(s("apple")), eq(Value::Int(1))]),
            range(Some(s("a")), None),
        ]);
        assert_eq!(ids(&idx, &q, &vs), vec![3]);
        let q = IndexQuery::And(vec![
            IndexQuery::Or(vec![eq(s("apple")), eq(Value::Int(1))]),
            range(Some(Value::Int(0)), None),
        ]);
        assert_eq!(ids(&idx, &q, &vs), vec![1]);
    }

    /// List elements are split between the kinds' array trees, and a scalar predicate must not see
    /// them (nor an array probe see a scalar). A list may hold elements of both kinds, so probes of
    /// both kinds can hold together.
    #[test]
    fn list_elements_split_between_the_kinds() {
        let vs = vec![
            (s("a"), 1),
            (Value::Int(1), 2),
            (list(vec![s("a"), Value::Int(1)]), 3),
            (list(vec![s("a")]), 4),
        ];
        let mut idx = RangeIndex::new();
        for (v, id) in &vs {
            idx.add(v, *id);
        }
        assert_eq!(ids(&idx, &contains(s("a")), &vs), vec![3, 4]);
        assert_eq!(ids(&idx, &contains(Value::Int(1)), &vs), vec![3]);
        assert_eq!(ids(&idx, &contains(s("never")), &vs), Vec::<u64>::new());
        assert_eq!(ids(&idx, &eq(s("a")), &vs), vec![1], "the scalar only");
        assert_eq!(
            ids(&idx, &eq(Value::Int(1)), &vs),
            vec![2],
            "the scalar only"
        );
        let q = IndexQuery::And(vec![contains(s("a")), contains(Value::Int(1))]);
        assert_eq!(ids(&idx, &q, &vs), vec![3], "a list holding both");
    }

    /// A distance predicate is served from the geo kind, as a superset the caller's retained
    /// filter narrows. Under `AND` with a point equality it stays one lookup of the cell windows.
    #[test]
    fn a_distance_predicate_reaches_the_geo_kind() {
        let vs = vec![
            (pt(51.5074, -0.1278), 1), // London
            (pt(48.8566, 2.3522), 2),  // Paris
            (s("not a point"), 3),
        ];
        let mut idx = RangeIndex::new();
        for (v, id) in &vs {
            idx.add(v, *id);
        }
        let near = |radius: i64| IndexQuery::Point {
            key: attr(),
            point: pt(51.5, -0.12),
            radius: Value::Int(radius),
        };
        assert_eq!(ids(&idx, &near(20_000), &vs), vec![1]);
        let q = IndexQuery::And(vec![near(20_000), eq(pt(51.5074, -0.1278))]);
        assert!(
            matches!(
                Selection::of(&q),
                Some(Selection::Scalars(Scalars {
                    geo: KeySet::Point(_),
                    ..
                }))
            ),
            "the cell of the equality, kept by the cover: {:?}",
            Selection::of(&q)
        );
        assert_eq!(ids(&idx, &q, &vs), vec![1]);
        let q = IndexQuery::And(vec![near(20_000), eq(pt(48.8566, 2.3522))]);
        assert!(ids(&idx, &q, &vs).is_empty(), "Paris is not near London");
        let q = IndexQuery::And(vec![near(20_000), near(400_000)]);
        assert_eq!(
            ids(&idx, &q, &vs),
            vec![1],
            "two covers meet in the smaller"
        );
    }

    /// Two sets of key windows meet in their pairwise overlaps, and a point in a window is kept.
    #[test]
    fn key_windows_intersect() {
        let w = KeySet::Windows(vec![(0, 10), (20, 30), (40, 50)]);
        assert_eq!(
            w.clone().and(KeySet::Range(5, 25)),
            KeySet::Windows(vec![(5, 10), (20, 25)])
        );
        assert_eq!(w.clone().and(KeySet::Range(21, 29)), KeySet::Range(21, 29));
        assert_eq!(w.clone().and(KeySet::Range(11, 19)), KeySet::Empty);
        assert_eq!(w.clone().and(KeySet::Point(45)), KeySet::Point(45));
        assert_eq!(
            w.clone().and(KeySet::of_keys(vec![1, 15, 41])),
            KeySet::Keys(vec![1, 41])
        );
        assert_eq!(
            w.clone().and(KeySet::Windows(vec![(8, 22), (50, 60)])),
            KeySet::Windows(vec![(8, 10), (20, 22), (50, 50)])
        );
        assert!(w.contains(25) && !w.contains(35));
    }

    /// Re-indexing across kinds: the remove must route on the *old* value, or the tuple is
    /// stranded in the kind nothing will look at again.
    #[test]
    fn a_value_changing_kind_leaves_nothing_behind() {
        let mut idx = RangeIndex::new();
        idx.add(&Value::Int(1), 1);
        idx.remove(&Value::Int(1), 1);
        idx.add(&s("1"), 1);
        let vs = [(s("1"), 1)];
        assert!(ids(&idx, &eq(Value::Int(1)), &vs).is_empty());
        assert_eq!(ids(&idx, &eq(s("1")), &vs), vec![1]);
        idx.remove(&s("1"), 1);
        assert!(idx.is_empty(), "no tuple stranded in any kind");
    }

    /// A unique-constraint lookup keys on the value: the number's encoding, or the string itself,
    /// which reads no rows when no entity holds it.
    #[test]
    fn encode_value_keys_on_the_value() {
        let mut idx = RangeIndex::new();
        idx.add(&s("a"), 1);
        let (kind, key) = idx.encode_value(&s("a")).expect("a string is a scalar");
        assert_eq!(kind, KeyKind::Tag);
        assert_eq!(idx.rows_at(kind, &key).collect::<Vec<_>>(), vec![1]);
        let (kind, key) = idx.encode_value(&s("never stored")).expect("a string");
        assert_eq!(idx.rows_at(kind, &key).count(), 0);
        assert_eq!(idx.encode_value(&list(vec![s("a")])), None);
    }

    /// Batch maintenance must agree with the one-at-a-time path, for every kind at once.
    #[test]
    fn batch_ops_match_single_ops() {
        let entries = vec![
            (Value::Int(3), 1u64),
            (list(vec![Value::Int(4), Value::Int(5)]), 2),
            (Value::Null, 3),
            (s("x"), 4),
            (list(vec![s("y"), Value::Int(6)]), 5),
            (pt(1.0, 1.0), 6),
            (list(vec![pt(1.0, 1.0)]), 7), // no kind holds a point in a list
        ];
        let mut scratch = EncodedTuples::default();
        let mut batched = RangeIndex::new();
        batched.add_batch(entries.iter().cloned(), &mut scratch);
        let mut singly = RangeIndex::new();
        for (v, id) in &entries {
            singly.add(v, *id);
        }
        assert_eq!(batched.tuples(), singly.tuples());
        assert_eq!(batched.tuples().len(), 7);
        batched.remove_batch(entries.iter().cloned(), &mut scratch);
        assert!(batched.is_empty());
    }

    /// A value of each [`Value`] variant, and the floats that encode specially. The `match` names
    /// every variant, so a new one does not compile until it is named here: a reminder to sample
    /// it too, which nothing checks.
    fn one_of_each() -> Vec<Value> {
        let values = vec![
            Value::Null,
            Value::Bool(true),
            Value::Int(-3),
            Value::Float(2.5),
            Value::Float(-0.0),
            Value::Float(f64::NAN),
            Value::String(Arc::new("x".to_string())),
            list(vec![Value::Int(1)]),
            Value::Map(Arc::default()),
            Value::Node(NodeId::from(1)),
            Value::Relationship(RelationshipId::from(1)),
            Value::Path(Arc::default()),
            Value::VecF32(Arc::default()),
            Value::Point(Point {
                latitude: 1.0,
                longitude: 2.0,
            }),
            Value::Datetime(1),
            Value::Date(1),
            Value::Time(1),
            Value::Duration(1),
        ];
        for value in &values {
            match value {
                Value::Null
                | Value::Bool(_)
                | Value::Int(_)
                | Value::Float(_)
                | Value::String(_)
                | Value::List(_)
                | Value::Map(_)
                | Value::Node(_)
                | Value::Relationship(_)
                | Value::Path(_)
                | Value::VecF32(_)
                | Value::Point(_)
                | Value::Datetime(_)
                | Value::Date(_)
                | Value::Time(_)
                | Value::Duration(_) => {}
            }
        }
        values
    }

    /// Whether the column would store `value` under any key.
    fn stores(
        idx: &RangeIndex,
        value: &Value,
    ) -> bool {
        let mut any = false;
        idx.for_each_stored_key(value, |_, _| any = true);
        any
    }

    /// A write and a lookup route a value alike: whatever a column stores, the predicate on that
    /// value finds — `n.v = x` for a scalar, `x IN n.v` for each element of a list. The two are
    /// separate code: a lookup routes by [`Kind::of`], a write by what each kind's encoder emits.
    /// A kind that stored a type `Kind::of` sends elsewhere would lose those rows to every lookup,
    /// and comparing with the scan cannot catch it, as the scan checks the same selection.
    #[test]
    fn a_stored_value_is_found_by_its_own_predicate() {
        let rows = |answer: Option<DocIter>| answer.map(Iterator::collect::<Vec<_>>);
        let mut checked = 0;
        for value in one_of_each() {
            let mut idx = RangeIndex::new();
            idx.add(&value, 1);
            if matches!(value, Value::List(_)) || !stores(&idx, &value) {
                continue;
            }
            assert_eq!(
                rows(idx.query(&eq(value.clone()))),
                Some(vec![1]),
                "{value:?} is stored, so `n.v = {value:?}` finds it"
            );
            checked += 1;
        }

        let elements = one_of_each();
        let mut idx = RangeIndex::new();
        idx.add(&list(elements.clone()), 1);
        for element in elements {
            if !stores(&idx, &list(vec![element.clone()])) {
                continue;
            }
            assert_eq!(
                rows(idx.query(&contains(element.clone()))),
                Some(vec![1]),
                "{element:?} is stored as an element, so `{element:?} IN n.v` finds it"
            );
            checked += 1;
        }
        assert_ne!(checked, 0, "the sample reaches a kind");
    }
}
