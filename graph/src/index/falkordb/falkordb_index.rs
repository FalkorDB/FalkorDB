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
use crate::index::{IndexQuery, IndexType};
use crate::runtime::value::Value;

use super::encode::encode_numeric;
use super::numeric::{DocIter, NumericIndex};
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

    /// Add already-encoded tuples (a unit of work's collapsed adds).
    pub fn add_encoded(
        &mut self,
        pairs: &mut [(u64, u64)],
    ) {
        match self {
            Self::Numeric(idx) => idx.add_encoded(pairs),
        }
    }

    /// Remove already-encoded tuples (a unit of work's collapsed removes).
    pub fn remove_encoded(
        &mut self,
        pairs: &mut [(u64, u64)],
    ) {
        match self {
            Self::Numeric(idx) => idx.remove_encoded(pairs),
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

/// Build state of a column, for online (background) index build.
///
/// `CREATE INDEX` on existing data returns immediately with the column
/// [`Building`](Self::Building). The column is then filled in steps
/// ([`graph_writes::build_step`](super::graph_writes::build_step)), each its own unit of work: it
/// indexes the next batch of the label's entities at the values they hold *then*, and moves the
/// cursor past them. The step that reaches the end flips the column [`Ready`](Self::Ready). While
/// `Building`, the read path returns `None` and falls back.
///
/// Nothing written during the build has to be remembered for later. Live writes maintain the
/// column as usual, including for entities no step has reached yet: a step that reaches one later
/// reads its current value, and adding a tuple the column already holds is a no-op. A removed
/// tuple the column never held is a no-op too. This is how C populates an index
/// (`src/index/index_construct.c`), in batches interleaved with writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColumnState {
    /// Some of the label's entities are not indexed yet — reads fall back. `next` is where the next
    /// step resumes.
    Building { next: BuildCursor },
    /// Every entity is indexed — the column serves reads.
    Ready,
}

/// Where a building column's next step resumes: the first entity no step has indexed yet.
///
/// For a node column, `row` is a node id and `col` is unused. For an edge column, `row` is a source
/// node id and `col` a destination within that row, so a row with more edges than one step reads
/// can be resumed partway, as C does. A step always finishes the multi-edge pair it stops in, so
/// every edge before `(row, col)` is indexed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BuildCursor {
    pub(crate) row: u64,
    pub(crate) col: u64,
}

/// One column's index plus its build state.
#[derive(Clone)]
struct ColumnEntry {
    column: IndexColumn,
    state: ColumnState,
}

impl ColumnEntry {
    /// A column that already holds all its data and serves reads immediately —
    /// the synchronous populate path.
    fn ready(column: IndexColumn) -> Self {
        Self {
            column,
            state: ColumnState::Ready,
        }
    }
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

    /// Create an empty numeric column for `(entity, label, attr)` in the `Ready`
    /// state, replacing any existing one. Used by the synchronous paths (RDB load /
    /// replica), where the caller fills it via [`build_numeric_sorted`](Self::build_numeric_sorted)
    /// before any read is served. An edge column's docs are `edge_id`s.
    pub fn create_numeric(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) {
        self.columns_mut(entity).insert(
            (label.clone(), attr.clone()),
            ColumnEntry::ready(IndexColumn::Numeric(NumericIndex::new())),
        );
    }

    /// Create an empty numeric column in the `Building` state, its cursor at the start — the
    /// online-build path. Returns immediately: live writes maintain the column from now on, build
    /// steps fill in the entities that already exist, and reads fall back until the last step
    /// flips it `Ready`. Replaces any existing column.
    pub fn create_building(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) {
        self.columns_mut(entity).insert(
            (label.clone(), attr.clone()),
            ColumnEntry {
                column: IndexColumn::Numeric(NumericIndex::new()),
                state: ColumnState::Building {
                    next: BuildCursor::default(),
                },
            },
        );
    }

    /// Build (or rebuild) the numeric column for `(entity, label, attr)` in the `Ready` state from
    /// numeric-encoded tuples **sorted ascending and unique**, replacing any existing one — the
    /// RDB-load populate path. Bottom-up-loads the tuples, which is cheaper than looping
    /// incremental adds. The caller encodes, so a Range attr holding mixed types indexes only its
    /// numeric values — mirroring the NUMERIC half of the RediSearch Range field.
    pub fn build_numeric_sorted(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        pairs: &[(u64, u64)],
    ) {
        self.columns_mut(entity).insert(
            (label.clone(), attr.clone()),
            ColumnEntry::ready(IndexColumn::Numeric(NumericIndex::from_sorted_tuples(
                pairs,
            ))),
        );
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
    /// State-agnostic (inspects a `Building` or `Ready` column alike) — used by the
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
        self.columns(entity)
            .get(&(label.clone(), attr.clone()))
            .map(|entry| &entry.column)
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
    pub(crate) fn apply(
        &mut self,
        writes: &mut PendingIndexWrites,
    ) {
        if writes.is_empty() {
            return;
        }
        self.apply_writes(EntityType::Node, writes.drain(EntityType::Node));
        self.apply_writes(
            EntityType::Relationship,
            writes.drain(EntityType::Relationship),
        );
    }

    /// One entity kind's part of [`apply`](Self::apply): per column, collapse each tuple's
    /// history to its last change ([`last_write_wins`]), then **remove** what ended removed and
    /// **add** what ended added, one batched tree op each.
    ///
    /// After the collapse the two sides are disjoint, so their order no longer matters; removes
    /// still go first. A column dropped since its changes were staged is skipped.
    fn apply_writes(
        &mut self,
        entity: EntityType,
        writes: impl IntoIterator<Item = (IndexKey, Vec<IndexWrite>)>,
    ) {
        let columns = self.columns_mut(entity);
        // One pair of buffers for the whole unit of work, reused column to column.
        let (mut removes, mut adds) = (Vec::new(), Vec::new());
        for (key, mut log) in writes {
            let Some(entry) = columns.get_mut(&key) else {
                continue;
            };
            last_write_wins(&mut log, &mut removes, &mut adds);
            entry.column.remove_encoded(&mut removes);
            entry.column.add_encoded(&mut adds);
        }
    }

    /// The first column still `Building`, with its cursor — what the next build step advances.
    /// Node columns first; `None` once every column is `Ready`.
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
    #[must_use]
    pub(crate) fn has_building_columns(&self) -> bool {
        self.next_building().is_some()
    }

    /// The build state of column `(entity, label, attr)`, if it exists.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn column_state(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<ColumnState> {
        self.columns(entity)
            .get(&(label.clone(), attr.clone()))
            .map(|entry| entry.state)
    }

    /// Record one build step on column `key`: add the tuples it read, then move the cursor to
    /// `next`, or flip the column `Ready` when `next` is `None` (the step reached the end). A
    /// no-op for a column that is gone or no longer `Building`.
    pub(crate) fn advance_build(
        &mut self,
        entity: EntityType,
        key: &IndexKey,
        tuples: &mut [(u64, u64)],
        next: Option<BuildCursor>,
    ) {
        let Some(entry) = self.columns_mut(entity).get_mut(key) else {
            return;
        };
        if entry.state == ColumnState::Ready {
            return;
        }
        entry.column.add_encoded(tuples);
        entry.state = next.map_or(ColumnState::Ready, |next| ColumnState::Building { next });
    }

    /// Answer an index query for `(entity, label)` from the numeric column — but ONLY a **numeric**
    /// `Equal`/`Range` leaf on a `Ready` column. A Range index also holds strings and geo (served by
    /// RediSearch), so a non-numeric or composite predicate returns `None` to fall through; so does a
    /// missing column, and so does a `Building` column (not every entity is indexed yet, so the read
    /// falls back to a scan — today's `UNDER CONSTRUCTION` behavior). Yields docs — node ids for a node
    /// column, `edge_id`s for an edge column. The read path routes here and falls back on `None`.
    #[must_use]
    pub fn query_numeric(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        query: &IndexQuery<Value>,
    ) -> Option<DocIter> {
        let (key, servable) = match query {
            IndexQuery::Equal { key, value } => (key, encode_numeric(value).is_some()),
            IndexQuery::Range { key, min, max, .. } => (
                key,
                min.as_ref().is_none_or(|v| encode_numeric(v).is_some())
                    && max.as_ref().is_none_or(|v| encode_numeric(v).is_some()),
            ),
            // A union of `Equal`s — what `n.a IN [...]` becomes before it reaches the index.
            // Every member must target this same attribute and encode numerically; a mixed or
            // empty list is declined whole, because answering with only the numeric members
            // would drop rows another kind still holds. `NumericIndex::union` re-checks the
            // encodability, so this only has to agree on the key.
            IndexQuery::Or(children) if !children.is_empty() => {
                // Flatten nested unions: `a IN [..] OR a IN [..]` arrives as an `Or` of `Or`s.
                fn keys_of<'a>(
                    children: &'a [IndexQuery<Value>],
                    out: &mut Vec<&'a Arc<String>>,
                ) -> bool {
                    children.iter().all(|c| match c {
                        IndexQuery::Equal { key, .. } => {
                            out.push(key);
                            true
                        }
                        IndexQuery::Or(nested) => !nested.is_empty() && keys_of(nested, out),
                        _ => false,
                    })
                }
                let mut member_keys = Vec::new();
                if !keys_of(children, &mut member_keys) || member_keys.is_empty() {
                    return None;
                }
                let first_key = member_keys[0];
                let same_key = member_keys.iter().all(|k| *k == first_key);
                (first_key, same_key)
            }
            // And / Point (geo) / InList / ArrayContains → not a numeric leaf.
            _ => return None,
        };
        if !servable {
            // A string/geo leaf on a Range index (RediSearch owns those entries), or a union
            // whose members do not all target this attribute.
            return None;
        }
        let entry = self.columns(entity).get(&(label.clone(), key.clone()))?;
        if entry.state != ColumnState::Ready {
            return None; // Building — not every entity is indexed yet, fall back to a scan
        }
        match &entry.column {
            IndexColumn::Numeric(idx) => idx.query(query),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::encode::encode_numeric;
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
        idx.apply(&mut writes);
    }

    /// The online-build state machine at the column level. A `Building` column gates reads (returns
    /// `None`, so the caller falls back) while live writes and build steps both fill it; the step
    /// that reaches the end flips it `Ready`, and from then on a step changes nothing.
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
            Some((Node, column.clone(), BuildCursor::default()))
        );
        assert!(hits(&idx, 10).is_empty(), "empty Building → None");

        // Live writes during the build: node 2 is created with 30, and node 1 goes 20 -> 25
        // before any step reached it — removing the 20 the column never held is a no-op.
        write(&mut idx, &label, &attr, &[(30, 2, true)]);
        write(&mut idx, &label, &attr, &[(20, 1, false), (25, 1, true)]);
        assert!(
            hits(&idx, 30).is_empty(),
            "still Building → None even with live writes in the column"
        );

        // A step indexes nodes 0..=2 at their current values — including node 1's 25 and node 2's
        // 30, which the writes already added — and stops before node 3.
        let mut batch = vec![(key(30), 2), (key(10), 0), (key(25), 1)];
        let next = BuildCursor { row: 3, col: 0 };
        idx.advance_build(Node, &column, &mut batch, Some(next));
        assert_eq!(idx.next_building(), Some((Node, column.clone(), next)));
        assert!(
            hits(&idx, 10).is_empty(),
            "still Building after a partial step"
        );

        // The last step reaches the end.
        idx.advance_build(Node, &column, &mut [(key(40), 3)], None);
        assert!(!idx.has_building_columns());
        assert_eq!(
            idx.column_state(Node, &label, &attr),
            Some(ColumnState::Ready)
        );
        assert_eq!(hits(&idx, 10), vec![0]);
        assert_eq!(hits(&idx, 25), vec![1]);
        assert!(
            hits(&idx, 20).is_empty(),
            "the removed value was never indexed"
        );
        assert_eq!(
            hits(&idx, 30),
            vec![2],
            "added by the write and the step, held once"
        );
        assert_eq!(all(&idx), vec![0, 1, 2, 3]);

        // A step on a `Ready` column, or on a column that is gone, changes nothing.
        idx.advance_build(Node, &column, &mut [(key(99), 9)], None);
        idx.advance_build(
            Node,
            &(label.clone(), arc("gone")),
            &mut [(key(99), 9)],
            None,
        );
        assert!(hits(&idx, 99).is_empty());
        assert_eq!(all(&idx), vec![0, 1, 2, 3]);
    }

    /// Forking a new version (what `Graph::new_version` does) and mutating it
    /// leaves the prior version — the snapshot a reader may still hold — untouched.
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

    /// `n.a IN [...]` reaches the facade as an `Or` of `Equal`s. A same-attribute union — nested
    /// `Or`s included — must be routed to that attribute's column, and any union that cannot be
    /// answered whole must be declined. A facade that declined every union would still return the
    /// right rows, because RediSearch answers instead, so only a test at this level notices.
    #[test]
    fn query_numeric_serves_a_same_attribute_union() {
        let (label, attr) = (arc("Person"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        idx.create_numeric(Node, &label, &attr);
        let column = idx.numeric_mut(Node, &label, &attr).unwrap();
        column.add(&Value::Int(30), 1);
        column.add(&Value::Int(40), 2);
        column.add(&Value::Int(50), 3);

        let eq = |key: &Arc<String>, v: Value| IndexQuery::Equal {
            key: key.clone(),
            value: v,
        };
        let int = |key: &Arc<String>, v: i64| eq(key, Value::Int(v));
        // Union order is unspecified, so compare sorted ids.
        let served = |entity, q: &IndexQuery<Value>| -> Option<Vec<u64>> {
            idx.query_numeric(entity, &label, q).map(|it| {
                let mut ids: Vec<u64> = it.collect();
                ids.sort_unstable();
                ids
            })
        };

        assert_eq!(
            served(Node, &IndexQuery::Or(vec![int(&attr, 30), int(&attr, 40)])),
            Some(vec![1, 2]),
            "a same-attribute union is served from the column"
        );
        assert_eq!(
            served(
                Node,
                &IndexQuery::Or(vec![
                    IndexQuery::Or(vec![int(&attr, 30)]),
                    IndexQuery::Or(vec![int(&attr, 50)]),
                ])
            ),
            Some(vec![1, 3]),
            "`a IN [..] OR a IN [..]` arrives as an Or of Ors and must flatten, not decline"
        );

        assert_eq!(
            served(
                Node,
                &IndexQuery::Or(vec![int(&attr, 30), int(&arc("height"), 40)])
            ),
            None,
            "a union spanning two attributes cannot be answered from one column"
        );
        assert_eq!(
            served(
                Node,
                &IndexQuery::Or(vec![
                    int(&attr, 30),
                    eq(&attr, Value::String(Arc::new("x".to_string()))),
                ])
            ),
            None,
            "a non-numeric member declines the whole union"
        );
        assert_eq!(
            served(Node, &IndexQuery::Or(vec![])),
            None,
            "an empty union is not match-all"
        );
        assert_eq!(
            served(Node, &IndexQuery::Or(vec![int(&arc("height"), 30)])),
            None,
            "no column for the union's attribute"
        );
        assert_eq!(
            served(
                Relationship,
                &IndexQuery::Or(vec![int(&attr, 30), int(&attr, 40)])
            ),
            None,
            "no edge column of this name"
        );
    }
}
