//! Folded-roots MVCC holder: the FalkorDB index state that lives
//! on [`Graph`](crate::graph::Graph) as its own copy-on-write field, so
//! `Graph::new_version()` forks it in `O(1)` per column and the committed-version
//! `Arc<AtomicRefCell<Graph>>` swap publishes graph + index together — one atomic
//! step, no torn read.
//!
//! Deliberately **pure data — no reference back to `Graph`**. The RediSearch
//! `Indexer` holds an `Arc<Mutex<Option<Arc<AtomicRefCell<Graph>>>>>` pointing
//! back at the graph for background population; that cycle is exactly what this
//! avoids. The write path mutates a version's own copy instead.

use std::collections::HashMap;
use std::sync::Arc;

use crate::entity_type::EntityType;
use crate::graph::graph::NodeId;
use crate::index::{IndexQuery, IndexType};
use crate::runtime::value::Value;

use super::encode::encode_numeric;
use super::numeric::{DocIter, EncodedTuples, NumericIndex};
use super::pending_writes::{IndexWrite, PendingIndexWrites, last_write_wins};

/// Identifies one index column: `(label, attribute)`.
///
/// Keyed by name for now; interned numeric ids would be more compact once the write and read
/// paths both resolve names to ids.
pub type IndexKey = (Arc<String>, Arc<String>);

/// One index column, of a specific kind.
///
/// Index kinds are a **closed set we own**, so this is a static enum — no
/// `Box<dyn>`, no boxed iterators, and exhaustive matching forces every kind to
/// be handled (an unimplemented kind fails loudly, never silently). Kind-specific
/// queries (numeric range/point vs a future vector ANN) are reached by matching
/// the variant; the uniform lifecycle ([`add`](Self::add)/[`remove`](Self::remove))
/// is delegated here. More variants (Text, Vector, Geo) land with their kinds.
#[derive(Clone)]
pub enum IndexColumn {
    Numeric(NumericIndex),
}

impl IndexColumn {
    /// Index `id` under `value` — the new value on create/update. Each kind
    /// interprets the `Value` (numeric encodes it; a future text kind tokenizes).
    pub fn add(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        match self {
            Self::Numeric(idx) => idx.add(value, id),
        }
    }

    /// Remove `id` under `value` — the old value on delete/update.
    pub fn remove(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        match self {
            Self::Numeric(idx) => idx.remove(value, id),
        }
    }

    /// Add a batch of `(value, id)` entries — the columnar write path's add column.
    pub fn add_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut Vec<(u64, u64)>,
    ) {
        match self {
            Self::Numeric(idx) => idx.add_batch(entries, scratch),
        }
    }

    /// Remove a batch of `(value, id)` entries — the columnar write path's remove column.
    pub fn remove_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut Vec<(u64, u64)>,
    ) {
        match self {
            Self::Numeric(idx) => idx.remove_batch(entries, scratch),
        }
    }

    /// The tree key `value` takes in *this* column, or `None` when the kind does not index it
    /// (numeric: non-numeric and `NaN`).
    #[must_use]
    pub fn encode_value(
        &self,
        value: &Value,
    ) -> Option<u64> {
        match self {
            Self::Numeric(_) => encode_numeric(value),
        }
    }

    /// Append the tuple `(value, id)` takes in *this* column to `out`, if the kind indexes
    /// `value`. The one place a value becomes a column's tuples, so a kind that gives
    /// a value several keys changes only this.
    pub fn push_encoded(
        &self,
        value: &Value,
        id: u64,
        out: &mut EncodedTuples,
    ) {
        if let Some(key) = self.encode_value(value) {
            out.scalar.push((key, id));
        }
    }

    /// Add already-encoded tuples (a unit of work's collapsed adds, or an indexer batch).
    pub fn add_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        match self {
            Self::Numeric(idx) => idx.add_encoded(tuples),
        }
    }

    /// Remove already-encoded tuples (a unit of work's collapsed removes).
    pub fn remove_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        match self {
            Self::Numeric(idx) => idx.remove_encoded(tuples),
        }
    }

    /// Whether this column holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Numeric(idx) => idx.is_empty(),
        }
    }
}

/// Build state of a column.
///
/// A column is filled by indexer batches. Each reads the label's next entities from the cursor at
/// the values they hold *then*
/// ([`graph_writes::read_indexer_batch_of`](super::graph_writes::read_indexer_batch_of)), adds
/// them, and moves the cursor past them
/// ([`apply_indexer_batch`](FalkorDbIndex::apply_indexer_batch)). The batch that reaches the end
/// flips the column [`Ready`](Self::Ready). While [`Building`](Self::Building), the read path
/// returns `None` and falls through to RediSearch. `CREATE INDEX` takes one batch over every
/// entity, so its column is `Ready` when it returns.
///
/// A `Building` column holds exactly the current tuples of the entities below its cursor, and
/// nothing at or above it. Live writes keep that true ([`FalkorDbIndex::apply`]). An addition for
/// an entity below the cursor is applied, and one for an entity at or above it is not, since the
/// batch that reaches the entity reads its value then. Every removal is applied, because an id's
/// place can move past the cursor within one unit of work: an edge behind it is deleted and its
/// id reused by an edge ahead of it, so the place no longer tells whether the column holds the
/// deleted edge's tuples. Removing a tuple the column does not hold changes nothing. Additions are
/// dropped before a unit's changes collapse to the last per tuple, so a tuple the unit removed and
/// added back, as the reused id's new edge can, is still removed. This is how C populates an index
/// (`src/index/index_construct.c`), in batches interleaved with writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColumnState {
    /// Some of the label's entities are not indexed yet — reads fall through to RediSearch. `next`
    /// is where the next indexer batch resumes.
    Building { next: BuildCursor },
    /// Every entity is indexed — the column serves reads.
    Ready,
}

/// Where a building column's next indexer batch resumes: the first entity no batch has indexed
/// yet. Also an entity's place in that order, to compare with it ([`Self::precedes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BuildCursor {
    /// A node column's next node.
    Node(NodeId),
    /// An edge column's next `(src, dst)` pair, so a source with more edges than one batch reads
    /// can be resumed partway, as C does. A batch always finishes the multi-edge pair it stops in,
    /// so every edge of an earlier pair is indexed.
    Edge { src: NodeId, dst: NodeId },
}

impl BuildCursor {
    /// Whether this entity's place comes before `next`, a column's cursor: whether a batch has
    /// indexed it. A node by its id; an edge by its `(src, dst)` pair, the order batches read
    /// edges in.
    pub(crate) fn precedes(
        self,
        next: Self,
    ) -> bool {
        match (self, next) {
            (Self::Node(at), Self::Node(next)) => u64::from(at) < u64::from(next),
            (
                Self::Edge { src, dst },
                Self::Edge {
                    src: nsrc,
                    dst: ndst,
                },
            ) => (u64::from(src), u64::from(dst)) < (u64::from(nsrc), u64::from(ndst)),
            _ => unreachable!("a node and an edge are never in one column"),
        }
    }

    /// Where the build of a new `entity` column starts: its first node, or its first pair.
    pub(crate) fn new(entity: EntityType) -> Self {
        match entity {
            EntityType::Node => Self::Node(NodeId::from(0)),
            EntityType::Relationship => Self::Edge {
                src: NodeId::from(0),
                dst: NodeId::from(0),
            },
        }
    }
}

/// One column's index plus its build state.
#[derive(Clone)]
struct ColumnEntry {
    column: IndexColumn,
    state: ColumnState,
}

impl ColumnEntry {
    fn building(&self) -> bool {
        self.state != ColumnState::Ready
    }
}

/// One indexer batch: the encoded tuples of column `key` it read, sorted, and where the column's
/// next batch resumes (`None`: the batch reached the end).
pub(crate) struct IndexerBatch {
    pub(crate) entity: EntityType,
    pub(crate) key: IndexKey,
    pub(crate) tuples: EncodedTuples,
    pub(crate) next: Option<BuildCursor>,
}

/// The graph's FalkorDB indexes: one CoW column per indexed `(label, attr)`.
///
/// Node and edge (relationship) columns live in separate maps — a node label and
/// a relationship type can share a name, so one keyspace would collide. Callers
/// select the map with an [`EntityType`], mirroring the `node_indexer` /
/// `edge_indexer` split on [`Graph`](crate::graph::Graph). An edge column's docs
/// are `edge_id`s; the read path recovers `(src, dst)` from the graph's own
/// `edge_id → (src, dst)` reverse index, so no endpoints are stored here.
///
/// Cloning is `O(1)` per column (each index is a root-`Arc` bump), so it rides
/// `Graph::new_version()` cheaply and shares pages with the prior version until a
/// writer mutates that particular column.
#[derive(Clone, Default)]
pub struct FalkorDbIndex {
    node_columns: HashMap<IndexKey, ColumnEntry>,
    edge_columns: HashMap<IndexKey, ColumnEntry>,
}

impl FalkorDbIndex {
    /// An empty set of indexes.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether no index column (node or edge) has been created.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.node_columns.is_empty() && self.edge_columns.is_empty()
    }

    /// Whether any column exists for `entity` — the write path's fast path: with none, a hook
    /// stages nothing and skips the graph reads staging would cost.
    #[must_use]
    pub fn has_columns(
        &self,
        entity: EntityType,
    ) -> bool {
        !self.columns(entity).is_empty()
    }

    /// Number of index columns (node + edge).
    #[must_use]
    pub fn len(&self) -> usize {
        self.node_columns.len() + self.edge_columns.len()
    }

    /// The column map for `entity` — node labels and relationship types keep
    /// separate keyspaces (a label and a type may share a name).
    fn columns(
        &self,
        entity: EntityType,
    ) -> &HashMap<IndexKey, ColumnEntry> {
        match entity {
            EntityType::Node => &self.node_columns,
            EntityType::Relationship => &self.edge_columns,
        }
    }

    fn columns_mut(
        &mut self,
        entity: EntityType,
    ) -> &mut HashMap<IndexKey, ColumnEntry> {
        match entity {
            EntityType::Node => &mut self.node_columns,
            EntityType::Relationship => &mut self.edge_columns,
        }
    }

    /// Column `(entity, label, attr)`, if it exists.
    fn entry(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<&ColumnEntry> {
        self.columns(entity).get(&(label.clone(), attr.clone()))
    }

    /// Create an empty numeric column for `(entity, label, attr)` in the `Ready`
    /// state, replacing any existing one. Used by the synchronous paths (RDB load /
    /// replica), where the caller fills it via [`build_numeric`](Self::build_numeric)
    /// before any read is served. An edge column's docs are `edge_id`s.
    pub fn create_numeric(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) {
        self.insert_column(
            entity,
            label,
            attr,
            IndexColumn::Numeric(NumericIndex::new()),
            ColumnState::Ready,
        );
    }

    /// Create an empty numeric column in the `Building` state, its cursor at the start, replacing
    /// any existing one. Indexer batches fill it from the start, live writes keep the part below
    /// the cursor current, and reads fall through to RediSearch until the last batch flips it
    /// `Ready`.
    pub fn create_building(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) {
        self.insert_column(
            entity,
            label,
            attr,
            IndexColumn::Numeric(NumericIndex::new()),
            ColumnState::Building {
                next: BuildCursor::new(entity),
            },
        );
    }

    /// Build (or rebuild) the numeric column for `(entity, label, attr)` from
    /// `entries` (any order) in the `Ready` state, replacing any existing one — the
    /// bulk populate path. `NumericIndex::from_entries` bottom-up-loads the sorted
    /// pairs (cheaper than looping incremental adds). Non-numeric / `NaN` values are
    /// skipped by the encoder, so a Range attr holding mixed types indexes only its
    /// numeric values — mirroring the NUMERIC half of the RediSearch Range field.
    pub fn build_numeric<'a>(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        entries: impl IntoIterator<Item = (&'a Value, u64)>,
    ) {
        self.insert_column(
            entity,
            label,
            attr,
            IndexColumn::Numeric(NumericIndex::from_entries(entries)),
            ColumnState::Ready,
        );
    }

    /// Put `column` at `(entity, label, attr)` in `state`, replacing any column there.
    fn insert_column(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        column: IndexColumn,
        state: ColumnState,
    ) {
        self.columns_mut(entity)
            .insert((label.clone(), attr.clone()), ColumnEntry { column, state });
    }

    /// Drop the column for `(entity, label, attr)` in `O(1)` — releases the tree `Arc`. DROP INDEX.
    pub fn drop_column(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) {
        self.columns_mut(entity)
            .remove(&(label.clone(), attr.clone()));
    }

    /// CREATE INDEX on the RDB-load path: create each attribute's column empty, if the index
    /// type has native columns (only Range does). The load fills them with
    /// [`graph_writes::populate`](super::graph_writes::populate) once every entity is in.
    pub fn create_unpopulated(
        &mut self,
        index_type: &IndexType,
        entity: EntityType,
        label: &Arc<String>,
        attrs: &[Arc<String>],
    ) {
        if *index_type == IndexType::Range {
            for attr in attrs {
                self.create_numeric(entity, label, attr);
            }
        }
    }

    /// DROP INDEX: drop each attribute's column, if the index type has native columns. `O(1)`
    /// each, since it only releases the tree `Arc`.
    pub fn drop_index(
        &mut self,
        index_type: &IndexType,
        entity: EntityType,
        label: &Arc<String>,
        attrs: &[Arc<String>],
    ) {
        if *index_type == IndexType::Range {
            for attr in attrs {
                self.drop_column(entity, label, attr);
            }
        }
    }

    /// The numeric column for `(entity, label, attr)`, if one exists and is numeric.
    /// Used by the
    /// populate path and tests, not the read path (which gates on state).
    #[must_use]
    pub fn numeric(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<&NumericIndex> {
        match self.columns(entity).get(&(label.clone(), attr.clone())) {
            Some(ColumnEntry {
                column: IndexColumn::Numeric(idx),
                ..
            }) => Some(idx),
            None => None,
        }
    }

    /// Mutable numeric column for `(entity, label, attr)`, if one exists and is numeric.
    pub fn numeric_mut(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<&mut NumericIndex> {
        match self
            .columns_mut(entity)
            .get_mut(&(label.clone(), attr.clone()))
        {
            Some(ColumnEntry {
                column: IndexColumn::Numeric(idx),
                ..
            }) => Some(idx),
            None => None,
        }
    }

    /// The column for `(entity, label, attr)`, whatever its state.
    #[must_use]
    pub(crate) fn column(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<&IndexColumn> {
        self.entry(entity, label, attr).map(|entry| &entry.column)
    }

    /// Whether a column exists for `(entity, label, attr)` — the write path's staging gate.
    #[must_use]
    pub fn has_column(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> bool {
        self.columns(entity)
            .contains_key(&(label.clone(), attr.clone()))
    }

    /// The tree key `value` takes in column `(entity, label, attr)`, or `None` when there is no
    /// such column or it does not index `value` — the write path's staging gate.
    #[must_use]
    pub fn encode_value(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        value: &Value,
    ) -> Option<u64> {
        self.columns(entity)
            .get(&(label.clone(), attr.clone()))?
            .column
            .encode_value(value)
    }

    /// Apply a unit of work's staged changes and leave `writes` empty. This is the single write
    /// primitive: the write paths only stage from graph state, into `writes`. Owners reach it
    /// through [`Graph::apply_index_writes`](crate::graph::graph::Graph::apply_index_writes).
    ///
    /// `self` must be the index of the private version the changes were staged against. The index
    /// is part of the graph version, so a version that is later refused takes these changes down
    /// with it.
    ///
    /// `place` gives an entity's place in build order, or `None` if the version no longer knows
    /// it, as for an edge it deleted: only a `Building` column asks, and only of additions. It
    /// applies an addition only for an entity below its cursor, and every removal
    /// ([`ColumnState`] says why).
    pub(crate) fn apply(
        &mut self,
        writes: &mut PendingIndexWrites,
        place: impl Fn(EntityType, u64) -> Option<BuildCursor>,
    ) {
        if writes.is_empty() {
            return;
        }
        self.apply_writes(EntityType::Node, writes.drain(EntityType::Node), &place);
        self.apply_writes(
            EntityType::Relationship,
            writes.drain(EntityType::Relationship),
            &place,
        );
    }

    /// One entity kind's part of [`apply`](Self::apply): per column, keep, on a `Building` column,
    /// only the additions for entities below its cursor, collapse each tuple's history to its last
    /// change ([`last_write_wins`]), then **remove** what ended removed and **add** what ended
    /// added, one batched tree op each.
    ///
    /// After the collapse the two sides are disjoint, so their order no longer matters; removes
    /// still go first. A column dropped since its changes were staged is skipped.
    fn apply_writes(
        &mut self,
        entity: EntityType,
        writes: impl IntoIterator<Item = (IndexKey, Vec<IndexWrite>)>,
        place: &impl Fn(EntityType, u64) -> Option<BuildCursor>,
    ) {
        let columns = self.columns_mut(entity);
        // One pair of buffers for the whole unit of work, reused column to column.
        let (mut removes, mut adds) = (EncodedTuples::default(), EncodedTuples::default());
        for (key, mut log) in writes {
            let Some(entry) = columns.get_mut(&key) else {
                continue;
            };
            if let ColumnState::Building { next } = entry.state {
                // A batch reads an entity at or above the cursor when it reaches it, so its
                // additions go. They go before the collapse, so one cannot cancel an earlier
                // removal of the same tuple.
                log.retain(|w| !w.add || place(entity, w.id).is_some_and(|at| at.precedes(next)));
            }
            last_write_wins(&mut log, &mut removes.scalar, &mut adds.scalar);
            entry.column.remove_encoded(&mut removes);
            entry.column.add_encoded(&mut adds);
        }
    }

    /// The first column still `Building`, with its cursor. Node columns first; `None` once every
    /// column is `Ready`. For tests of one column at a time.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn next_building(&self) -> Option<(EntityType, IndexKey, BuildCursor)> {
        [
            (EntityType::Node, &self.node_columns),
            (EntityType::Relationship, &self.edge_columns),
        ]
        .into_iter()
        .find_map(|(entity, columns)| {
            columns.iter().find_map(|(key, entry)| match entry.state {
                ColumnState::Building { next } => Some((entity, key.clone(), next)),
                ColumnState::Ready => None,
            })
        })
    }

    /// Whether any column is still `Building`.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn has_building_columns(&self) -> bool {
        self.node_columns
            .values()
            .chain(self.edge_columns.values())
            .any(ColumnEntry::building)
    }

    /// The build state of column `(entity, label, attr)`, if it exists.
    #[must_use]
    pub(crate) fn column_state(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<ColumnState> {
        self.entry(entity, label, attr).map(|entry| entry.state)
    }

    /// Apply one indexer batch to this version: add its tuples, read from this version, then move
    /// the cursor to `batch.next`, or flip the column `Ready` when it is `None`. `false`, changing
    /// nothing, if the column is gone or no longer `Building`.
    pub(crate) fn apply_indexer_batch(
        &mut self,
        mut batch: IndexerBatch,
    ) -> bool {
        let Some(entry) = self.columns_mut(batch.entity).get_mut(&batch.key) else {
            return false;
        };
        if !entry.building() {
            return false;
        }
        entry.column.add_encoded(&mut batch.tuples);
        entry.state = batch
            .next
            .map_or(ColumnState::Ready, |next| ColumnState::Building { next });
        true
    }

    /// Answer an index query for `(entity, label)` from the numeric column, on a `Ready` column
    /// only. Served: a numeric `Equal`/`Range`, an `IN` list whose values are all numeric, and an
    /// `OR` of equalities and `IN` lists on one attribute. A Range index also holds strings and geo
    /// (served by RediSearch), so anything else returns `None` to fall through — a non-numeric
    /// value, an `And`, an `OR` spanning attributes — and so does a missing column, and so does a
    /// `Building` column (not every entity is indexed yet, so RediSearch answers). Yields docs —
    /// node ids for a node column, `edge_id`s for an edge column. The read path routes here and
    /// falls through to RediSearch on `None`.
    #[must_use]
    pub fn query_numeric(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        query: &IndexQuery<Value>,
    ) -> Option<DocIter> {
        // Which column to ask. `Equal`/`Range` are checked here because the column answers a bound
        // it cannot encode with "no rows" rather than `None`; an `IN` list or an `OR` is checked
        // by the column itself (`NumericIndex::query`), which declines rather than answer part.
        let key = match query {
            IndexQuery::Equal { key, value } if encode_numeric(value).is_some() => key,
            IndexQuery::Range { key, min, max, .. }
                if min.as_ref().is_none_or(|v| encode_numeric(v).is_some())
                    && max.as_ref().is_none_or(|v| encode_numeric(v).is_some()) =>
            {
                key
            }
            // `n.a IN [...]` names one attribute by construction.
            IndexQuery::InList { key, .. } => key,
            // An `OR` is routed by its first member's attribute; the column declines it if any
            // other member names a different one.
            IndexQuery::Or(children) => match children.first() {
                Some(IndexQuery::Equal { key, .. } | IndexQuery::InList { key, .. }) => key,
                _ => return None,
            },
            // A string/geo leaf on a Range index (RediSearch owns those entries), And, Point (geo),
            // ArrayContains → not a numeric predicate.
            _ => return None,
        };
        let entry = self.entry(entity, label, key)?;
        if entry.building() {
            return None; // Building — not every entity is indexed yet; RediSearch answers
        }
        match &entry.column {
            IndexColumn::Numeric(idx) => idx.query(query),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::pending_writes::IndexChanges;
    use super::*;
    use crate::entity_type::EntityType::{Node, Relationship};

    fn arc(s: &str) -> Arc<String> {
        Arc::new(s.to_string())
    }

    /// One unit of work on node column `(label, attr)`: record each `(value, id, add)` in order,
    /// then apply them together.
    fn write(
        idx: &mut FalkorDbIndex,
        label: &Arc<String>,
        attr: &Arc<String>,
        changes: &[(i64, u64, bool)],
    ) {
        let mut staged = IndexChanges::new(Node);
        for &(v, id, add) in changes {
            if add {
                staged.add(idx, label, attr, &Value::Int(v), id);
            } else {
                staged.remove(idx, label, attr, &Value::Int(v), id);
            }
        }
        let mut writes = PendingIndexWrites::default();
        writes.extend(staged);
        idx.apply(&mut writes, |_, id| {
            Some(BuildCursor::Node(NodeId::from(id)))
        });
    }

    /// The build state machine at the column level. A `Building` column gates reads (returns
    /// `None`, so the caller falls back) while indexer batches fill it. A live write reaches it
    /// only below the cursor: one at or above is left to the batch that reaches the entity. The
    /// batch that reaches the end flips the column `Ready`, and from then on a batch changes
    /// nothing.
    #[test]
    fn building_column_gates_reads_until_finished() {
        let (label, attr) = (arc("Person"), arc("age"));
        let column: IndexKey = (label.clone(), attr.clone());
        let eq = |v: i64| IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(v),
        };
        let key = |v: i64| encode_numeric(&Value::Int(v)).unwrap();
        let hits = |idx: &FalkorDbIndex, v: i64| -> Vec<u64> {
            idx.query_numeric(Node, &label, &eq(v))
                .map_or_else(Vec::new, Iterator::collect)
        };
        // What the column holds for `v`, whatever its state.
        let holds = |idx: &FalkorDbIndex, v: i64| -> Vec<u64> {
            idx.numeric(Node, &label, &attr)
                .unwrap()
                .point(&Value::Int(v))
                .collect()
        };
        // `age IN [10, 30]`: a union, gated like any other read.
        let in_10_30 = IndexQuery::InList {
            key: attr.clone(),
            list: Value::List(Arc::new(
                [Value::Int(10), Value::Int(30)].into_iter().collect(),
            )),
        };
        let all = |idx: &FalkorDbIndex| -> Vec<u64> {
            idx.numeric(Node, &label, &attr)
                .unwrap()
                .range(None, None, true, true)
                .collect()
        };

        let mut idx = FalkorDbIndex::new();
        idx.create_building(Node, &label, &attr);
        assert_eq!(
            idx.next_building(),
            Some((Node, column.clone(), BuildCursor::new(Node)))
        );
        assert!(hits(&idx, 10).is_empty(), "empty Building → None");

        // Live writes before any batch: node 2 is created with 30, and node 1 goes 20 -> 25.
        // Both are at or above the cursor, so the column takes neither.
        write(&mut idx, &label, &attr, &[(30, 2, true)]);
        write(&mut idx, &label, &attr, &[(20, 1, false), (25, 1, true)]);
        assert!(
            all(&idx).is_empty(),
            "writes at or above the cursor wait for the batch"
        );
        assert!(
            idx.query_numeric(Node, &label, &in_10_30).is_none(),
            "an IN list on a Building column falls through too"
        );

        // A batch indexes nodes 0..=2 at their current values, and stops before node 3.
        let batch = |tuples: Vec<(u64, u64)>, next| {
            let mut tuples = EncodedTuples::scalars(tuples);
            tuples.sort_dedup();
            IndexerBatch {
                entity: Node,
                key: column.clone(),
                tuples,
                next,
            }
        };
        let next = BuildCursor::Node(NodeId::from(3));
        assert!(idx.apply_indexer_batch(batch(
            vec![(key(30), 2), (key(10), 0), (key(25), 1)],
            Some(next),
        )));
        assert_eq!(idx.next_building(), Some((Node, column.clone(), next)));
        assert!(
            hits(&idx, 10).is_empty(),
            "still Building after a partial batch"
        );

        // Now node 1, below the cursor, goes 25 -> 26 and the column takes it at once. Node 3, at
        // the cursor, gets 45, and node 5, above it, is created with 50: the column takes neither.
        write(&mut idx, &label, &attr, &[(25, 1, false), (26, 1, true)]);
        write(&mut idx, &label, &attr, &[(45, 3, true)]);
        write(&mut idx, &label, &attr, &[(50, 5, true)]);
        assert_eq!(
            holds(&idx, 26),
            vec![1],
            "a write below the cursor is applied"
        );
        assert!(holds(&idx, 25).is_empty());
        assert!(holds(&idx, 45).is_empty(), "a write at the cursor is not");
        assert!(
            holds(&idx, 50).is_empty(),
            "a write above the cursor is not"
        );

        // The last batch reaches the end, reading nodes 3 and 5 as they are now.
        assert!(idx.apply_indexer_batch(batch(vec![(key(45), 3), (key(50), 5)], None)));
        assert!(!idx.has_building_columns());
        assert_eq!(
            idx.column_state(Node, &label, &attr),
            Some(ColumnState::Ready)
        );
        assert_eq!(hits(&idx, 10), vec![0]);
        assert_eq!(
            idx.query_numeric(Node, &label, &in_10_30)
                .map(Iterator::collect::<Vec<u64>>),
            Some(vec![0, 2]),
            "a Ready column serves the IN list"
        );
        assert_eq!(hits(&idx, 26), vec![1]);
        assert!(hits(&idx, 20).is_empty() && hits(&idx, 25).is_empty());
        assert_eq!(hits(&idx, 45), vec![3]);
        assert_eq!(hits(&idx, 50), vec![5]);
        assert_eq!(all(&idx), vec![0, 1, 2, 3, 5]);

        // A batch on a `Ready` column, or on a column that is gone, changes nothing.
        assert!(!idx.apply_indexer_batch(batch(vec![(key(99), 9)], None)));
        assert!(!idx.apply_indexer_batch(IndexerBatch {
            key: (label.clone(), arc("gone")),
            ..batch(vec![(key(99), 9)], None)
        }));
        assert!(hits(&idx, 99).is_empty());
        assert_eq!(all(&idx), vec![0, 1, 2, 3, 5]);
    }

    #[test]
    fn new_version_is_copy_on_write() {
        let (label, attr) = (arc("Person"), arc("age"));

        let mut v1 = FalkorDbIndex::new();
        v1.create_numeric(Node, &label, &attr);
        v1.numeric_mut(Node, &label, &attr)
            .unwrap()
            .add(&Value::Int(30), 1);

        let mut v2 = v1.clone(); // the O(1) fork
        v2.numeric_mut(Node, &label, &attr)
            .unwrap()
            .add(&Value::Int(40), 2);

        let all = |v: &FalkorDbIndex| -> Vec<u64> {
            v.numeric(Node, &label, &attr)
                .unwrap()
                .range(None, None, true, true)
                .collect()
        };
        assert_eq!(all(&v1), vec![1]); // committed version unchanged
        assert_eq!(all(&v2), vec![1, 2]); // new version sees the write
    }

    /// `create_numeric` installs a column; before it, the key is absent.
    #[test]
    fn create_then_lookup() {
        let (label, attr) = (arc("Person"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        assert!(idx.is_empty());
        assert!(idx.numeric(Node, &label, &attr).is_none());

        idx.create_numeric(Node, &label, &attr);
        assert_eq!(idx.len(), 1);
        assert!(idx.numeric(Node, &label, &attr).is_some());
        assert!(idx.numeric(Node, &label, &attr).unwrap().is_empty());
    }

    /// `apply_writes` lands each unit of work's changes in the right column.
    #[test]
    fn writes_apply_into_columns() {
        let (label, attr) = (arc("Person"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        idx.create_numeric(Node, &label, &attr);

        write(&mut idx, &label, &attr, &[(7, 1, true), (9, 2, true)]);
        write(&mut idx, &label, &attr, &[(7, 1, false)]);

        let ids: Vec<u64> = idx
            .numeric(Node, &label, &attr)
            .unwrap()
            .range(None, None, true, true)
            .collect();
        assert_eq!(ids, vec![2]);
    }

    /// A node label and a relationship type can share a name; their columns must be independent
    /// (separate keyspaces), or edge writes would corrupt node reads and vice-versa. #51.
    #[test]
    fn node_and_edge_columns_do_not_collide() {
        let (name, attr) = (arc("Rated"), arc("score")); // same name as a node label AND a rel type
        let mut idx = FalkorDbIndex::new();
        idx.create_numeric(Node, &name, &attr);
        idx.create_numeric(Relationship, &name, &attr);
        assert_eq!(idx.len(), 2, "one node column + one edge column");

        idx.numeric_mut(Node, &name, &attr)
            .unwrap()
            .add(&Value::Int(1), 10);
        idx.numeric_mut(Relationship, &name, &attr)
            .unwrap()
            .add(&Value::Int(1), 20);

        let ids = |e| -> Vec<u64> {
            idx.numeric(e, &name, &attr)
                .unwrap()
                .range(None, None, true, true)
                .collect()
        };
        assert_eq!(ids(Node), vec![10], "node column sees only the node doc");
        assert_eq!(
            ids(Relationship),
            vec![20],
            "edge column sees only the edge doc"
        );

        // Dropping the edge column leaves the node column intact.
        idx.drop_column(Relationship, &name, &attr);
        assert!(idx.numeric(Relationship, &name, &attr).is_none());
        assert!(idx.numeric(Node, &name, &attr).is_some());
    }

    /// Reads route only numeric `Equal`/`Range` leaves to the numeric column, for either entity.
    /// A node query on a shared name must not see the edge column, and vice-versa.
    #[test]
    fn query_numeric_routes_only_numeric_leaves_per_entity() {
        let (label, attr) = (arc("Person"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        idx.create_numeric(Node, &label, &attr);
        idx.numeric_mut(Node, &label, &attr)
            .unwrap()
            .add(&Value::Int(30), 1);
        let key = attr.clone();

        // Numeric Equal / Range → routed to the node column, correct ids.
        let eq = IndexQuery::Equal {
            key: key.clone(),
            value: Value::Int(30),
        };
        let hit: Vec<u64> = idx.query_numeric(Node, &label, &eq).unwrap().collect();
        assert_eq!(hit, vec![1]);
        let rg = IndexQuery::Range {
            key: key.clone(),
            min: Some(Value::Int(10)),
            max: None,
            include_min: true,
            include_max: true,
        };
        assert!(idx.query_numeric(Node, &label, &rg).is_some());

        // A string on the same Range index must fall through (RediSearch owns the TAG entries).
        let str_eq = IndexQuery::Equal {
            key: key.clone(),
            value: Value::String(Arc::new("x".to_string())),
        };
        assert!(idx.query_numeric(Node, &label, &str_eq).is_none());
        // Composite, missing-column, and the wrong entity all fall through.
        let eq2 = IndexQuery::Equal {
            key: key.clone(),
            value: Value::Int(30),
        };
        assert!(
            idx.query_numeric(Node, &label, &IndexQuery::And(vec![eq2]))
                .is_none()
        );
        let other = IndexQuery::Equal {
            key: arc("height"),
            value: Value::Int(5),
        };
        assert!(idx.query_numeric(Node, &label, &other).is_none());
        assert!(
            idx.query_numeric(Relationship, &label, &eq).is_none(),
            "no edge column of this name"
        );
    }

    /// `n.a IN [...]` reaches the facade as one `InList`, and `a IN [..] OR a = ..` as an `Or` of
    /// lists and equalities. Each must be routed to that attribute's column, and anything that
    /// cannot be answered whole must be declined. A facade that declined every one would still
    /// return the right rows, because RediSearch answers instead, so only a test at this level
    /// notices.
    #[test]
    fn query_numeric_serves_in_lists_and_same_attribute_unions() {
        let (label, attr) = (arc("Person"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        idx.create_numeric(Node, &label, &attr);
        let column = idx.numeric_mut(Node, &label, &attr).unwrap();
        column.add(&Value::Int(30), 1);
        column.add(&Value::Int(40), 2);
        column.add(&Value::Int(50), 3);

        let in_list = |key: &Arc<String>, values: Vec<Value>| IndexQuery::InList {
            key: key.clone(),
            list: Value::List(Arc::new(values.into_iter().collect())),
        };
        let ints = |key: &Arc<String>, values: &[i64]| {
            in_list(key, values.iter().copied().map(Value::Int).collect())
        };
        let eq = |key: &Arc<String>, v: i64| IndexQuery::Equal {
            key: key.clone(),
            value: Value::Int(v),
        };
        // Union order is unspecified, so compare sorted ids.
        let served = |entity, q: &IndexQuery<Value>| -> Option<Vec<u64>> {
            idx.query_numeric(entity, &label, q).map(|it| {
                let mut ids: Vec<u64> = it.collect();
                ids.sort_unstable();
                ids
            })
        };

        assert_eq!(
            served(Node, &ints(&attr, &[30, 40])),
            Some(vec![1, 2]),
            "an IN list is served from its attribute's column"
        );
        assert_eq!(
            served(
                Node,
                &IndexQuery::Or(vec![ints(&attr, &[30]), ints(&attr, &[50])])
            ),
            Some(vec![1, 3]),
            "`a IN [..] OR a IN [..]` is served as one union"
        );
        assert_eq!(
            served(
                Node,
                &IndexQuery::Or(vec![eq(&attr, 40), ints(&attr, &[50])])
            ),
            Some(vec![2, 3]),
            "`a = .. OR a IN [..]` is served as one union"
        );

        assert_eq!(
            served(
                Node,
                &in_list(
                    &attr,
                    vec![Value::Int(30), Value::String(Arc::new("x".to_string()))]
                )
            ),
            None,
            "a non-numeric value declines the whole list"
        );
        assert_eq!(
            served(Node, &in_list(&attr, Vec::new())),
            None,
            "an empty list is declined, not answered with no rows"
        );
        assert_eq!(
            served(
                Node,
                &IndexQuery::Or(vec![eq(&attr, 30), eq(&arc("height"), 40)])
            ),
            None,
            "an OR spanning two attributes cannot be answered from one column"
        );
        assert_eq!(
            served(Node, &IndexQuery::Or(vec![])),
            None,
            "an empty OR is not match-all"
        );
        assert_eq!(
            served(Node, &ints(&arc("height"), &[30])),
            None,
            "no column for the list's attribute"
        );
        assert_eq!(
            served(Relationship, &ints(&attr, &[30, 40])),
            None,
            "no edge column of this name"
        );
    }
}
