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

use roaring::RoaringTreemap;

use crate::entity_type::EntityType;
use crate::graph::graph::NodeId;
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
/// [`Building`](Self::Building). The column is then filled in steps, each its own unit of work: it
/// reads the next batch of the label's entities at the values they hold *then*
/// ([`graph_writes::read_build_batch`](super::graph_writes::read_build_batch)), adds them, and
/// moves the cursor past them ([`apply_build`](FalkorDbIndex::apply_build)). The step that reaches
/// the end flips the column [`Ready`](Self::Ready). While `Building`, the read path returns `None`
/// and falls through to RediSearch.
///
/// Live writes maintain the column as usual, including for entities no step has reached yet: a
/// step that reaches one later reads its current value, and adding a tuple the column already
/// holds is a no-op. A removed tuple the column never held is a no-op too. This is how C populates
/// an index (`src/index/index_construct.c`), in batches interleaved with writes.
///
/// A step reads its batch on a snapshot and a write may commit before the step does, and the
/// other way round. Neither throws its work away; each merges the other's
/// ([`apply_build`](FalkorDbIndex::apply_build),
/// [`adopt_build_progress`](FalkorDbIndex::adopt_build_progress)). For that a `Building` column
/// remembers the entities written since the last step committed (`touched`), and a version
/// remembers its own changes to the column until it is published (`log`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColumnState {
    /// Some of the label's entities are not indexed yet — reads fall through to RediSearch. `next`
    /// is where the next step resumes.
    Building { next: BuildCursor },
    /// Every entity is indexed — the column serves reads.
    Ready,
}

/// Where a building column's next step resumes: the first entity no step has indexed yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BuildCursor {
    /// A node column's next node.
    Node(NodeId),
    /// An edge column's next `(src, dst)` pair, so a source with more edges than one step reads
    /// can be resumed partway, as C does. A step always finishes the multi-edge pair it stops in,
    /// so every edge of an earlier pair is indexed.
    Edge { src: NodeId, dst: NodeId },
}

impl BuildCursor {
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
struct ColumnEntry {
    column: IndexColumn,
    state: ColumnState,
    /// Which column this is: the index's count of columns created, so a build batch read from one
    /// column never lands in another created under the same name since.
    generation: u64,
    /// While `Building`: the entities written since a build step last committed, to any column.
    /// Their tuples in the column are already right, so a step that read them before those writes
    /// leaves them alone. Empty once `Ready`.
    touched: Arc<RoaringTreemap>,
    /// While `Building`, until this version is published: its own changes to the column, in the
    /// order made — what a writer replays onto build steps committed after its fork was made. A
    /// new version starts it empty.
    log: Vec<IndexWrite>,
}

impl ColumnEntry {
    /// This entry in a new version: the column shared copy-on-write, without this version's log.
    fn new_version(&self) -> Self {
        Self {
            column: self.column.clone(),
            state: self.state,
            generation: self.generation,
            touched: Arc::clone(&self.touched),
            log: Vec::new(),
        }
    }

    fn building(&self) -> bool {
        self.state != ColumnState::Ready
    }
}

/// One build step's batch, kept so it can be merged into a version other than the one it was
/// read on: the encoded tuples of column `key` it read, sorted, and where the column's next step
/// resumes (`None`: the batch reached the end). `read_at` is the index's count of build steps
/// committed when the batch was read.
pub(crate) struct BuildBatch {
    pub(crate) entity: EntityType,
    pub(crate) key: IndexKey,
    pub(crate) generation: u64,
    pub(crate) read_at: u64,
    pub(crate) tuples: Vec<(u64, u64)>,
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
/// [`new_version`](Self::new_version) is `O(1)` per column (each index is a root-`Arc` bump), so it
/// rides `Graph::new_version()` cheaply and shares pages with the prior version until a writer
/// mutates that particular column.
#[derive(Default)]
pub struct FalkorDbIndex {
    node_columns: HashMap<IndexKey, ColumnEntry>,
    edge_columns: HashMap<IndexKey, ColumnEntry>,
    /// Build steps committed into this index's history. A batch is merged only into the index it
    /// was read on or one with the same count, so `touched` — kept since the last step — covers
    /// every write since the batch was read.
    build_commits: u64,
    /// Columns created in this index's history: the last one's [`ColumnEntry::generation`]. Only
    /// writers create columns, and they commit one at a time, so along the published versions it
    /// only grows: no later column gets the generation a batch was read with.
    columns_created: u64,
    /// Whether this version records its changes to `Building` columns in their logs: a new
    /// version does, until it is published.
    logging: bool,
}

impl FalkorDbIndex {
    /// An empty set of indexes.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The index of a new version: every column shared copy-on-write, and a change log of its own,
    /// started empty and kept until the version is published, in case its writer has to replay it
    /// onto build steps committed meanwhile ([`adopt_build_progress`](Self::adopt_build_progress)).
    #[must_use]
    pub fn new_version(&self) -> Self {
        let fork = |columns: &HashMap<IndexKey, ColumnEntry>| {
            columns
                .iter()
                .map(|(key, entry)| (key.clone(), entry.new_version()))
                .collect()
        };
        Self {
            node_columns: fork(&self.node_columns),
            edge_columns: fork(&self.edge_columns),
            build_commits: self.build_commits,
            columns_created: self.columns_created,
            logging: true,
        }
    }

    /// Stop recording, and drop what was recorded: this version is being published, and nobody
    /// replays a published version's changes.
    pub(crate) fn stop_logging(&mut self) {
        self.logging = false;
        for entry in self.entries_mut() {
            entry.log = Vec::new();
        }
    }

    /// How many changes this version's logs hold.
    #[cfg(test)]
    pub(crate) fn logged(&self) -> usize {
        self.entries().map(|entry| entry.log.len()).sum()
    }

    /// Build steps committed into this index's history; see [`BuildBatch::read_at`].
    pub(crate) const fn build_commits(&self) -> u64 {
        self.build_commits
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

    /// Every column, node and edge.
    fn entries(&self) -> impl Iterator<Item = &ColumnEntry> {
        self.node_columns.values().chain(self.edge_columns.values())
    }

    fn entries_mut(&mut self) -> impl Iterator<Item = &mut ColumnEntry> {
        self.node_columns
            .values_mut()
            .chain(self.edge_columns.values_mut())
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
    /// state, replacing any existing one. Used by RDB load ([`create_unpopulated`](Self::create_unpopulated)),
    /// whose populate replaces it with the built column
    /// ([`build_numeric`](Self::build_numeric)) before any read is served. An edge
    /// column's docs are `edge_id`s.
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

    /// Create an empty numeric column in the `Building` state, its cursor at the start — the
    /// online-build path. Returns immediately: live writes maintain the column from now on, build
    /// steps fill in the entities that already exist, and reads fall through to RediSearch until
    /// the last step flips it `Ready`. Replaces any existing column.
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

    /// Put `column` at `(entity, label, attr)` in `state`, replacing any column there, as a column
    /// of a generation of its own.
    fn insert_column(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        column: IndexColumn,
        state: ColumnState,
    ) {
        self.columns_created += 1;
        let entry = ColumnEntry {
            column,
            state,
            generation: self.columns_created,
            touched: Arc::default(),
            log: Vec::new(),
        };
        self.columns_mut(entity)
            .insert((label.clone(), attr.clone()), entry);
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
    /// State-agnostic (inspects a `Building` or `Ready` column alike) — used by tests, not the
    /// read path (which gates on state).
    #[must_use]
    pub fn numeric(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<&NumericIndex> {
        self.entry(entity, label, attr)
            .map(|entry| match &entry.column {
                IndexColumn::Numeric(idx) => idx,
            })
    }

    /// Mutable numeric column for `(entity, label, attr)`, if one exists and is numeric.
    pub fn numeric_mut(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<&mut NumericIndex> {
        self.columns_mut(entity)
            .get_mut(&(label.clone(), attr.clone()))
            .map(|entry| match &mut entry.column {
                IndexColumn::Numeric(idx) => idx,
            })
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
        self.entry(entity, label, attr).is_some()
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
        self.entry(entity, label, attr)?.column.encode_value(value)
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

    /// One entity kind's part of [`apply`](Self::apply): each column's changes go through
    /// [`apply_changes`], logged while this version logs. A column dropped since its changes were
    /// staged is skipped.
    fn apply_writes(
        &mut self,
        entity: EntityType,
        writes: impl IntoIterator<Item = (IndexKey, Vec<IndexWrite>)>,
    ) {
        let logging = self.logging;
        let columns = self.columns_mut(entity);
        // One pair of buffers for the whole unit of work, reused column to column.
        let (mut removes, mut adds) = (Vec::new(), Vec::new());
        for (key, mut log) in writes {
            let Some(entry) = columns.get_mut(&key) else {
                continue;
            };
            apply_changes(entry, &mut log, &mut removes, &mut adds, logging);
        }
    }

    /// The first column still `Building`, with its cursor and generation — what the next build
    /// step advances. Node columns first; `None` once every column is `Ready`.
    #[must_use]
    pub(crate) fn next_building(&self) -> Option<(EntityType, IndexKey, BuildCursor, u64)> {
        [
            (EntityType::Node, &self.node_columns),
            (EntityType::Relationship, &self.edge_columns),
        ]
        .into_iter()
        .find_map(|(entity, columns)| {
            columns.iter().find_map(|(key, entry)| match entry.state {
                ColumnState::Building { next } => {
                    Some((entity, key.clone(), next, entry.generation))
                }
                ColumnState::Ready => None,
            })
        })
    }

    /// Whether any column is still `Building`.
    #[must_use]
    pub(crate) fn has_building_columns(&self) -> bool {
        self.entries().any(ColumnEntry::building)
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
        self.entry(entity, label, attr).map(|entry| entry.state)
    }

    /// Commit one build step into this version: add the batch's tuples, except those of entities
    /// written since a step last committed, then move the cursor to `batch.next`, or flip the
    /// column `Ready` when it is `None`, and start every column's `touched` afresh.
    ///
    /// The batch may have been read on an older version, though never one before the last step
    /// committed. Its tuple for an entity nobody has written since is still right. An entity
    /// written since is in `touched`, and its tuples in this column are already right: writes
    /// maintain the column ahead of the cursor too. So is one written before the batch was read;
    /// skipping it is merely unnecessary.
    ///
    /// `false`, changing nothing, if another step committed since the batch was read (its
    /// `touched` would no longer cover every write since), or if the column is gone, was
    /// re-created since (its generation differs), or is no longer `Building`.
    ///
    /// The batch's tuples are filtered in place and left in it, so it can be applied again to a
    /// later version, which skips at least the entities this one did.
    pub(crate) fn apply_build(
        &mut self,
        batch: &mut BuildBatch,
    ) -> bool {
        if batch.read_at != self.build_commits {
            return false;
        }
        let Some(entry) = self.columns_mut(batch.entity).get_mut(&batch.key) else {
            return false;
        };
        if entry.generation != batch.generation || !entry.building() {
            return false;
        }
        batch.tuples.retain(|&(_, id)| !entry.touched.contains(id));
        entry.column.add_encoded(&mut batch.tuples);
        entry.state = batch
            .next
            .map_or(ColumnState::Ready, |next| ColumnState::Building { next });
        self.build_commits += 1;
        for entry in self.entries_mut() {
            if !entry.touched.is_empty() {
                entry.touched = Arc::default();
            }
        }
        true
    }

    /// Take the build steps committed in `published` since this version forked from an older
    /// one: for each column `Building` here, take `published`'s copy of it — its tuples, cursor
    /// and state — and replay this version's own changes to it (its log) on top. For a writer
    /// whose version is stale only because build steps committed meanwhile: they changed no
    /// graph data, so its writes still hold, and only the index needs the steps.
    ///
    /// Replaying the changes rather than comparing columns is what keeps a removal: a writer that
    /// re-valued or deleted an entity a step had not reached removed nothing from its own copy,
    /// but must remove the tuple a step committed since.
    ///
    /// A column `Ready` here was `Ready` when the version forked, and no step touches it. A column
    /// this version created, or re-created, is its own (another generation) and stays as it is.
    /// The count of build steps is `published`'s.
    pub(crate) fn adopt_build_progress(
        &mut self,
        published: &Self,
    ) {
        self.build_commits = published.build_commits;
        for entity in [EntityType::Node, EntityType::Relationship] {
            let theirs = published.columns(entity);
            let (mut removes, mut adds) = (Vec::new(), Vec::new());
            for (key, mine) in self.columns_mut(entity) {
                if !mine.building() {
                    continue;
                }
                let Some(theirs) = theirs.get(key) else {
                    continue;
                };
                if theirs.generation != mine.generation {
                    continue;
                }
                let mut log = std::mem::take(&mut mine.log);
                *mine = theirs.new_version();
                if !log.is_empty() {
                    apply_changes(mine, &mut log, &mut removes, &mut adds, false);
                }
            }
        }
    }

    /// Answer an index query for `(entity, label)` from the numeric column — but ONLY a **numeric**
    /// `Equal`/`Range` leaf on a `Ready` column. A Range index also holds strings and geo (served by
    /// RediSearch), so a non-numeric or composite predicate returns `None` to fall through; so does a
    /// missing column, and so does a `Building` column (not every entity is indexed yet, so
    /// RediSearch answers). Yields docs — node ids for a node column, `edge_id`s for an edge column.
    /// The read path routes here and falls through to RediSearch on `None`.
    #[must_use]
    pub fn query_numeric(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        query: &IndexQuery<Value>,
    ) -> Option<DocIter> {
        let (key, numeric) = match query {
            IndexQuery::Equal { key, value } => (key, encode_numeric(value).is_some()),
            IndexQuery::Range { key, min, max, .. } => (
                key,
                min.as_ref().is_none_or(|v| encode_numeric(v).is_some())
                    && max.as_ref().is_none_or(|v| encode_numeric(v).is_some()),
            ),
            // And / Or / Point (geo) / InList / ArrayContains → not a numeric leaf.
            _ => return None,
        };
        if !numeric {
            return None; // string / geo on a Range index — RediSearch owns those entries
        }
        let entry = self.entry(entity, label, key)?;
        if entry.building() {
            return None; // Building — not every entity is indexed yet; RediSearch answers
        }
        match &entry.column {
            IndexColumn::Numeric(idx) => idx.query(query),
        }
    }
}

/// Apply a unit of work's changes to `entry`, given in the order made: collapse each tuple's
/// history to its last change ([`last_write_wins`]), then **remove** what ended removed and
/// **add** what ended added, one batched tree op each. After the collapse the two sides are
/// disjoint, so their order no longer matters; removes still go first. A `Building` column also
/// notes them ([`record`]). `removes` and `adds` are scratch buffers, reused across calls.
fn apply_changes(
    entry: &mut ColumnEntry,
    log: &mut [IndexWrite],
    removes: &mut Vec<(u64, u64)>,
    adds: &mut Vec<(u64, u64)>,
    logging: bool,
) {
    last_write_wins(log, removes, adds);
    if entry.building() {
        record(entry, removes, adds, logging);
    }
    entry.column.remove_encoded(removes);
    entry.column.add_encoded(adds);
}

/// Note a unit of work's collapsed changes to a `Building` column: their entities in `touched`
/// and, if `logging`, the changes in its log, removes first as they are applied.
fn record(
    entry: &mut ColumnEntry,
    removes: &[(u64, u64)],
    adds: &[(u64, u64)],
    logging: bool,
) {
    let touched = Arc::make_mut(&mut entry.touched);
    for (&(key, id), add) in removes
        .iter()
        .map(|r| (r, false))
        .chain(adds.iter().map(|a| (a, true)))
    {
        if logging {
            entry.log.push(IndexWrite { key, id, add });
        }
        touched.insert(id);
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
        let generation = idx.next_building().unwrap().3;
        assert_eq!(
            idx.next_building(),
            Some((Node, column.clone(), BuildCursor::new(Node), generation))
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
        let batch = |mut tuples: Vec<(u64, u64)>, next, read_at| {
            tuples.sort_unstable();
            BuildBatch {
                entity: Node,
                key: column.clone(),
                generation,
                read_at,
                tuples,
                next,
            }
        };
        let next = BuildCursor::Node(NodeId::from(3));
        assert!(idx.apply_build(&mut batch(
            vec![(key(30), 2), (key(10), 0), (key(25), 1)],
            Some(next),
            0
        )));
        assert_eq!(
            idx.next_building(),
            Some((Node, column.clone(), next, generation))
        );
        assert!(
            hits(&idx, 10).is_empty(),
            "still Building after a partial step"
        );

        // The last step reaches the end.
        assert!(
            !idx.apply_build(&mut batch(vec![(key(40), 3)], None, 0)),
            "a batch read before the last step committed"
        );
        assert!(idx.apply_build(&mut batch(vec![(key(40), 3)], None, 1)));
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
        assert!(!idx.apply_build(&mut batch(vec![(key(99), 9)], None, 2)));
        assert!(!idx.apply_build(&mut BuildBatch {
            key: (label.clone(), arc("gone")),
            ..batch(vec![(key(99), 9)], None, 2)
        }));
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

        let mut v2 = v1.new_version(); // the O(1) fork
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
}
