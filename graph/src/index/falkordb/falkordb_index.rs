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

use rustc_hash::FxHashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::entity_type::EntityType;
use crate::index::{IndexQuery, IndexType};
use crate::runtime::value::Value;

use super::encode::{KeySpace, encode_numeric, for_each_stored_key};
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
        scratch: &mut EncodedTuples,
    ) {
        match self {
            Self::Numeric(idx) => idx.add_batch(entries, scratch),
        }
    }

    /// Remove a batch of `(value, id)` entries — the columnar write path's remove column.
    pub fn remove_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut EncodedTuples,
    ) {
        match self {
            Self::Numeric(idx) => idx.remove_batch(entries, scratch),
        }
    }

    /// Call `emit` with each tree key `value` takes in *this* column, and the key space it goes
    /// in. Nothing when the kind does not index it (numeric: non-numeric, temporal and `NaN`); one
    /// key per numeric element, in [`KeySpace::Array`], for a list.
    pub fn for_each_stored_key(
        &self,
        value: &Value,
        emit: impl FnMut(KeySpace, u64),
    ) {
        match self {
            Self::Numeric(_) => for_each_stored_key(value, emit),
        }
    }

    /// Append the tuples `(value, id)` takes in *this* column to `out`, each in its key space.
    /// Unsorted and possibly repeated — see [`EncodedTuples::sort_dedup`].
    pub fn push_encoded(
        &self,
        value: &Value,
        id: u64,
        out: &mut EncodedTuples,
    ) {
        match self {
            Self::Numeric(_) => out.push_numeric(value, id),
        }
    }

    /// A new column of *this* column's kind, built from encoded tuples arriving sorted and unique,
    /// one stream per key space — how the install builds the finished column without hard-coding
    /// a kind at the call site.
    #[must_use]
    pub fn new_like_from_sorted(
        &self,
        scalar: impl IntoIterator<Item = (u64, u64)>,
        array: impl IntoIterator<Item = (u64, u64)>,
    ) -> Self {
        match self {
            Self::Numeric(_) => Self::Numeric(NumericIndex::from_sorted_tuples(scalar, array)),
        }
    }

    /// Every `(key, doc)` tuple this column holds in key space `space`, encoded and in order, read
    /// lazily from a snapshot — how the install streams TOMB and DELTA. Encoded, not `Value`s:
    /// TOMB must use the same equivalence relation as the column it is subtracted from, and
    /// decoding to re-encode would be a second trip through a many-to-one map.
    pub fn tuples(
        &self,
        space: KeySpace,
    ) -> impl Iterator<Item = (u64, u64)> + use<> {
        match self {
            Self::Numeric(idx) => idx.tuples(space),
        }
    }

    /// Add already-encoded tuples (a unit of work's collapsed adds).
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

/// What the native index has to say about a query.
///
/// The three cases are deliberately distinct, because the no-fallback build treats them
/// differently and collapsing any two of them is a bug:
///
/// * `Some(Rows(..))` — served natively.
/// * `Some(NotReady)` — the column exists and will serve this, but its background build has not
///   installed the base yet. The caller must fall back to a scan. This is a *timing* state, not
///   a capability gap, so it must never surface as an error.
/// * `None` — the native index cannot serve this predicate at all. Under `index-falkordb` that
///   is a hard error, because there is no other index to answer it.
///
/// Generic in the row iterator so the read path can retype the rows — raw docs to `NodeId`, or to
/// `(src, dst, edge_id)` — without flattening the three cases back into an `Option` on the way.
pub enum NumericAnswer<I = DocIter> {
    Rows(I),
    NotReady,
}

impl<I> NumericAnswer<I> {
    /// Map the rows, leaving `NotReady` alone. `f` runs only when there are rows, so a caller may
    /// move setup into it that would be wasted on a not-ready column.
    pub fn map_rows<J>(
        self,
        f: impl FnOnce(I) -> J,
    ) -> NumericAnswer<J> {
        match self {
            Self::Rows(rows) => NumericAnswer::Rows(f(rows)),
            Self::NotReady => NumericAnswer::NotReady,
        }
    }
}

/// Build state of a column, for online (background) index build.
///
/// `CREATE INDEX` on existing data returns immediately with the column in
/// [`Building`](Self::Building); a background job scans the pre-existing snapshot (BASE)
/// off-thread and one install commit adopts it, flipping to [`Ready`](Self::Ready).
/// While `Building` the read path returns `None` (falls back to a scan). Live writes
/// maintain the column normally — the column's own tree *is* the DELTA.
///
/// **Reconciliation**, so a concurrently deleted or updated entity's stale snapshot entry
/// never resurrects, is by TOMB: the tuples destroyed since `CREATE INDEX`, subtracted from
/// BASE before DELTA is replayed. The graph's free-id lists are applied first as a cheap
/// backstop ([`graph_writes::install_base`](super::graph_writes::install_base)), but they are
/// only that — an id leaves its free list when it is reused, so TOMB is the real defence.
#[derive(Clone)]
pub enum ColumnState {
    /// Base not yet installed — reads scan-fall-back.
    ///
    /// `tomb` is the catch-up log of tuples *destroyed* since `CREATE INDEX`. It is the same CoW
    /// tuple tree as the column, for two reasons: it forks in `O(1)` with the version (a
    /// `RoaringTreemap` deep-clones, which is `O(writes²)` across a build), and it dedups under
    /// the column's own equivalence relation, which a set of raw values would not.
    ///
    /// Adds are *not* recorded here — they go straight into the column tree, which is the DELTA.
    /// Only removes are deferred, because a remove may target a row that still lives only in the
    /// not-yet-installed BASE, where no version can name it yet.
    ///
    /// `epoch` is the build's identity: a token from [`NEXT_BUILD_EPOCH`], unique across the
    /// process, assigned at `create_building` and carried by the background job. `install_base`
    /// no-ops unless the column's current epoch matches the job's — so if the column is dropped
    /// and re-created mid-build, the stale job installs nothing into the new column (which has a
    /// fresh epoch) and a new job builds it instead.
    Building { tomb: IndexColumn, epoch: u64 },
    /// Base installed (or the graph was empty at create) — the column serves reads.
    Ready,
}

/// One column's index plus its build state.
#[derive(Clone)]
struct ColumnEntry {
    column: IndexColumn,
    state: ColumnState,
}

impl ColumnEntry {
    /// A column that already holds all its data and serves reads immediately —
    /// the synchronous populate / bulk-build path.
    fn ready(column: IndexColumn) -> Self {
        Self {
            column,
            state: ColumnState::Ready,
        }
    }

    /// Defer these *destroyed* tuples into TOMB, so the install can subtract them from a BASE
    /// that was scanned before they died. A no-op once `Ready` — after install there is no stale
    /// BASE left to correct.
    fn note_tomb(
        &mut self,
        removed: &mut EncodedTuples,
    ) {
        if let ColumnState::Building { tomb, .. } = &mut self.state {
            tomb.add_encoded(removed);
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

/// Build-epoch source for every graph in the process (see [`ColumnState::Building`]).
///
/// Process-wide rather than per graph so that no two builds anywhere share an epoch. The build
/// registry keys in-flight builds by graph *name*, column and epoch, and a per-graph counter
/// restarted at `1` for a graph deleted and re-created under the same name: its first build then
/// had the same key as the old graph's build still running, and waited for that job to finish
/// before it could be dispatched. A fork that takes an epoch and is rolled back only skips a
/// number. Starts at `1`, so `0` is never a live building epoch.
static NEXT_BUILD_EPOCH: AtomicU64 = AtomicU64::new(1);

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
    /// replica), where the caller fills it via [`build_numeric`](Self::build_numeric)
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

    /// Create an empty numeric column in the `Building` state — the online-build path.
    /// Returns immediately; live writes maintain the column (adds into it, removes also into
    /// TOMB) and reads fall back to a scan until [`install_base`](Self::install_base) adopts
    /// the pre-existing snapshot and flips it `Ready`. Returns the build **epoch** the
    /// background job must carry (see [`ColumnState::Building`]); replaces any existing column
    /// with a fresh epoch.
    pub fn create_building(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> u64 {
        let epoch = NEXT_BUILD_EPOCH.fetch_add(1, Ordering::Relaxed);
        self.columns_mut(entity).insert(
            (label.clone(), attr.clone()),
            ColumnEntry {
                column: IndexColumn::Numeric(NumericIndex::new()),
                state: ColumnState::Building {
                    tomb: IndexColumn::Numeric(NumericIndex::new()),
                    epoch,
                },
            },
        );
        epoch
    }

    /// Build (or rebuild) the numeric column for `(entity, label, attr)` in the `Ready` state from
    /// numeric-encoded tuples, each half **sorted ascending and unique**, replacing any existing
    /// one — the RDB-load populate path. Bottom-up-loads the tuples, which is cheaper than looping
    /// incremental adds. The caller encodes, so a Range attr holding mixed types indexes only its
    /// numeric values — mirroring the NUMERIC half of the RediSearch Range field.
    pub fn build_numeric_sorted(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        tuples: EncodedTuples,
    ) {
        self.columns_mut(entity).insert(
            (label.clone(), attr.clone()),
            ColumnEntry::ready(IndexColumn::Numeric(NumericIndex::from_sorted_tuples(
                tuples.scalar,
                tuples.array,
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

    /// Call `emit` with each tree key `value` takes in column `(entity, label, attr)` and the key
    /// space it goes in — nothing when there is no such column or it does not index `value`. The
    /// write path's staging gate.
    pub fn for_each_stored_key(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        value: &Value,
        emit: impl FnMut(KeySpace, u64),
    ) {
        if let Some(entry) = self.columns(entity).get(&(label.clone(), attr.clone())) {
            entry.column.for_each_stored_key(value, emit);
        }
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
        let (mut removes, mut adds) = (EncodedTuples::default(), EncodedTuples::default());
        for (key, mut log) in writes {
            let Some(entry) = columns.get_mut(&key) else {
                continue;
            };
            last_write_wins(&mut log, &mut removes, &mut adds);
            // TOMB for the install's BASE subtraction, *and* the column tree itself: I-W4'
            // requires DELTA to hold each touched entity's final state, not an append-only
            // add-log. Skipping the tree here would install two rows for `v0 -> v1 -> v2`.
            entry.note_tomb(&mut removes);
            entry.column.remove_encoded(&mut removes);
            entry.column.add_encoded(&mut adds);
        }
    }

    /// Install a background-built BASE and flip the column to `Ready` — **one commit**, per
    /// I-B3. No-op (returning `false`) if the column is gone, already `Ready`, or carries a
    /// different epoch, which is how a drop-and-recreate mid-build makes the stale job inert.
    ///
    /// ```text
    /// column := (BASE \ TOMB) ∪ DELTA,   state := Ready
    /// ```
    ///
    /// Built in one pass. BASE arrives sorted from the job, TOMB and DELTA are read in order from
    /// their trees, and [`installed_tuples`] merges the three into the stream the new column is
    /// packed from, page by page. Nothing is collected, and no existing tree is patched.
    ///
    /// **TOMB subtracts from BASE only, never from DELTA.** A tuple destroyed and recreated during
    /// the build (`v0 -> v1 -> v0`, or an id reused with the same value) is in both TOMB and DELTA,
    /// and must survive. By I-W4' no DELTA tuple is invalid at the version being installed into.
    ///
    /// Each half of `base` must be sorted and unique, which
    /// [`graph_writes::base_encoded`](super::graph_writes::base_encoded) guarantees.
    ///
    /// Chunking this would be wrong as well as slow: a partially-subtracted BASE is not a valid
    /// index at any version, and `MvccGraph::commit` pays an `O(attribute-store)`
    /// `trim_attr_stores()` per commit, so N/1024 commits multiply a cost the single install pays
    /// once.
    pub fn install_base(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        epoch: u64,
        base: EncodedTuples,
    ) -> bool {
        let Some(entry) = self
            .columns_mut(entity)
            .get_mut(&(label.clone(), attr.clone()))
        else {
            return false;
        };
        let ColumnState::Building { tomb, epoch: e } = &entry.state else {
            return false; // already Ready — nothing to install into
        };
        if *e != epoch {
            return false; // stale job: the column was dropped and re-created since it started
        }
        // Once per key space: a tuple only ever meets TOMB and DELTA tuples of its own tree.
        let scalar = installed_tuples(
            base.scalar,
            tomb.tuples(KeySpace::Scalar),
            entry.column.tuples(KeySpace::Scalar),
        );
        let array = installed_tuples(
            base.array,
            tomb.tuples(KeySpace::Array),
            entry.column.tuples(KeySpace::Array),
        );
        entry.column = entry.column.new_like_from_sorted(scalar, array);
        entry.state = ColumnState::Ready;
        true
    }

    /// A conjunction spanning several attributes: intersect one column per attribute.
    ///
    /// Each attribute's own conjuncts are folded by its own column first (see
    /// `NumericIndex::intersect_same_key`), so this only combines one doc stream per attribute.
    ///
    /// **Set probe, not a sorted merge.** `DocIter` yields docs in *value* order, not id order —
    /// deliberately, since Cypher guarantees no order without `ORDER BY`, and merging would cost
    /// a comparison per doc and force every branch to stay live. Value-ordered streams cannot be
    /// merge-intersected, so the first attribute is materialised into a set and the rest probe it.
    /// Memory is proportional to the first attribute's match count; there is no cardinality
    /// estimate to pick the smallest side with, so the order is the query's.
    ///
    /// Every attribute needs a `Ready` numeric column. One still `Building` makes the whole
    /// conjunction `NotReady` — intersecting against a half-built column would silently drop rows,
    /// which is worse than scanning.
    fn intersect_columns(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        children: &[IndexQuery<Value>],
    ) -> Option<NumericAnswer> {
        // Regroup the leaves per attribute, keeping each attribute's own conjuncts together so
        // its column can fold them into one window.
        fn group<'a>(
            children: &'a [IndexQuery<Value>],
            out: &mut Vec<(&'a Arc<String>, Vec<&'a IndexQuery<Value>>)>,
        ) -> Option<()> {
            for child in children {
                let key = match child {
                    IndexQuery::Equal { key, .. } | IndexQuery::Range { key, .. } => key,
                    IndexQuery::And(nested) => {
                        group(nested, out)?;
                        continue;
                    }
                    _ => return None,
                };
                match out.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, qs)) => qs.push(child),
                    None => out.push((key, vec![child])),
                }
            }
            Some(())
        }
        let mut per_attr: Vec<(&Arc<String>, Vec<&IndexQuery<Value>>)> = Vec::new();
        group(children, &mut per_attr)?;
        if per_attr.len() < 2 {
            return None; // the caller already handles the single-column case
        }

        let mut result: Option<FxHashSet<u64>> = None;
        for (attr, conjuncts) in per_attr {
            let entry = self.columns(entity).get(&(label.clone(), attr.clone()))?;
            if !matches!(entry.state, ColumnState::Ready) {
                return Some(NumericAnswer::NotReady);
            }
            let IndexColumn::Numeric(idx) = &entry.column;
            let folded = idx.intersect_refs(&conjuncts)?;
            result = Some(match result {
                None => folded.collect(),
                Some(seen) => folded.filter(|doc| seen.contains(doc)).collect(),
            });
            if result.as_ref().is_some_and(FxHashSet::is_empty) {
                break; // a later conjunct can only remove rows, never add them
            }
        }
        result.map(|docs| NumericAnswer::Rows(DocIter::Set(docs.into_iter())))
    }

    /// Every `(entity, label, attr, epoch)` column currently in the `Building` state —
    /// the work list the background-build controller spawns jobs for. The epoch identifies
    /// the build so a stale job can't publish a re-created column.
    #[must_use]
    pub fn building_columns(&self) -> Vec<(EntityType, Arc<String>, Arc<String>, u64)> {
        let mut out = Vec::new();
        for (entity, columns) in [
            (EntityType::Node, &self.node_columns),
            (EntityType::Relationship, &self.edge_columns),
        ] {
            for ((label, attr), entry) in columns {
                if let ColumnState::Building { epoch, .. } = &entry.state {
                    out.push((entity, label.clone(), attr.clone(), *epoch));
                }
            }
        }
        out
    }

    /// Answer an index query for `(entity, label)` from the numeric columns — but ONLY a **numeric**
    /// `Equal`/`Range`/union/array-contains leaf, or a conjunction of numeric `Equal`/`Range`
    /// leaves. Yields docs — node ids for a node column, `edge_id`s for an edge column.
    ///
    /// Three outcomes, and the caller must keep them apart (see [`NumericAnswer`]):
    ///
    /// * [`Rows`](NumericAnswer::Rows) — served here.
    /// * [`NotReady`](NumericAnswer::NotReady) — the column exists but is still `Building`, so its
    ///   base is not installed. The caller scans; this is a timing state, never an error.
    /// * `None` — not a numeric leaf, or no such column. A Range index also holds strings and geo
    ///   (RediSearch's, in the dark-launch build), so those land here too. Under `index-falkordb`
    ///   there is no other index, and the caller turns this into an error.
    #[must_use]
    pub fn query_numeric(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        query: &IndexQuery<Value>,
    ) -> Option<NumericAnswer> {
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
            // A conjunction of numeric `Equal`/`Range` leaves. On ONE attribute — `v = 5 AND v > 2`,
            // or two bounds the planner could not fold because they were still expressions —
            // `NumericIndex` collapses it arithmetically into a single window. Across several
            // attributes it is an intersection of one column per attribute, handed to
            // `intersect_columns`. Any other member (a string bound, an array-contains, a union)
            // declines the whole conjunction.
            IndexQuery::And(children) if !children.is_empty() => {
                fn leaves<'a>(
                    children: &'a [IndexQuery<Value>],
                    out: &mut Vec<(&'a Arc<String>, bool)>,
                ) -> bool {
                    children.iter().all(|c| match c {
                        IndexQuery::Equal { key, value } => {
                            out.push((key, encode_numeric(value).is_some()));
                            true
                        }
                        IndexQuery::Range { key, min, max, .. } => {
                            let numeric = min.as_ref().is_none_or(|v| encode_numeric(v).is_some())
                                && max.as_ref().is_none_or(|v| encode_numeric(v).is_some());
                            out.push((key, numeric));
                            true
                        }
                        IndexQuery::And(nested) => !nested.is_empty() && leaves(nested, out),
                        _ => false,
                    })
                }
                let mut found = Vec::new();
                if !leaves(children, &mut found) || found.is_empty() {
                    return None;
                }
                if !found.iter().all(|(_, numeric)| *numeric) {
                    return None; // a non-numeric member: no column can answer the conjunction
                }
                let first = found[0].0;
                if found.iter().any(|(k, _)| *k != first) {
                    // Spans several attributes: intersect one column per attribute.
                    return self.intersect_columns(entity, label, children);
                }
                (first, true)
            }
            // `value IN n.prop`, where the property holds a list. Served from the column's
            // separate array tree — see `NumericIndex::array_contains`. Servable when the probed
            // value encodes; the elements themselves were filtered at write time.
            IndexQuery::ArrayContains { key, value } => (key, encode_numeric(value).is_some()),
            // Point (geo) / InList → not a numeric leaf.
            _ => return None,
        };
        if !servable {
            // A string/geo leaf on a Range index (RediSearch owns those entries), or a union
            // whose members do not all target this attribute.
            return None;
        }
        let entry = self.columns(entity).get(&(label.clone(), key.clone()))?;
        if !matches!(entry.state, ColumnState::Ready) {
            // Building: the base is not installed yet. This is NOT an unsupported predicate —
            // the column will serve it once the build finishes — so the caller must scan rather
            // than error. `NotReady` keeps that distinct from `None`, which under the
            // no-fallback build is a hard failure.
            return Some(NumericAnswer::NotReady);
        }
        match &entry.column {
            IndexColumn::Numeric(idx) => idx.query(query).map(NumericAnswer::Rows),
        }
    }
}

/// `(BASE \ TOMB) ∪ DELTA` as one sorted, unique stream: the tuples an install builds the
/// finished column from. Each input must be sorted ascending and unique.
///
/// TOMB is subtracted from BASE before the union, so a tuple in both TOMB and DELTA survives —
/// the same result as adopting BASE, removing TOMB, then adding DELTA.
fn installed_tuples(
    base: impl IntoIterator<Item = (u64, u64)>,
    tomb: impl IntoIterator<Item = (u64, u64)>,
    delta: impl IntoIterator<Item = (u64, u64)>,
) -> impl Iterator<Item = (u64, u64)> {
    let mut base = base.into_iter().peekable();
    let mut tomb = tomb.into_iter().peekable();
    let mut delta = delta.into_iter().peekable();
    std::iter::from_fn(move || {
        // The next BASE tuple TOMB does not remove.
        let surviving = loop {
            let Some(&b) = base.peek() else {
                break None;
            };
            while tomb.next_if(|&t| t < b).is_some() {}
            if tomb.peek() == Some(&b) {
                base.next();
                continue;
            }
            break Some(b);
        };
        match (surviving, delta.peek().copied()) {
            (None, None) => None,
            (Some(b), None) => base.next().map(|_| b),
            (None, Some(d)) => delta.next().map(|_| d),
            (Some(b), Some(d)) if b < d => base.next().map(|_| b),
            (Some(b), Some(d)) if d < b => delta.next().map(|_| d),
            // In both: emit once.
            (Some(b), Some(_)) => {
                base.next();
                delta.next();
                Some(b)
            }
        }
    })
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

    /// The install's merge is `(BASE \ TOMB) ∪ DELTA` exactly, on inputs drawn to overlap: a
    /// small key space, so BASE, TOMB and DELTA share many tuples and every case of the merge is
    /// hit, including a tuple in all three.
    #[test]
    fn installed_tuples_is_base_minus_tomb_plus_delta() {
        use std::collections::BTreeSet;
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..200 {
            let mut draw = |n: u64| -> BTreeSet<(u64, u64)> {
                (0..next() % n).map(|_| (next() % 8, next() % 8)).collect()
            };
            let (base, tomb, delta) = (draw(40), draw(20), draw(20));
            let expected: Vec<(u64, u64)> = base
                .difference(&tomb)
                .copied()
                .collect::<BTreeSet<_>>()
                .union(&delta)
                .copied()
                .collect();
            let got: Vec<(u64, u64)> = installed_tuples(
                base.iter().copied(),
                tomb.iter().copied(),
                delta.iter().copied(),
            )
            .collect();
            assert_eq!(got, expected, "base {base:?} tomb {tomb:?} delta {delta:?}");
        }
    }

    /// A tuple destroyed and recreated during the build is in TOMB and in DELTA. TOMB subtracts
    /// from BASE only, so it survives.
    #[test]
    fn a_tuple_in_tomb_and_delta_survives_the_install() {
        let got: Vec<(u64, u64)> =
            installed_tuples([(1, 1), (2, 2)], [(1, 1), (2, 2)], [(2, 2)]).collect();
        assert_eq!(got, vec![(2, 2)]);
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

    /// Rows of an answer that must be `Rows`. Panics on the other two, rather than reporting them
    /// as "no rows" — a test that expected rows and got `NotReady` or `None` has found a bug, and
    /// collapsing all three into an empty `Vec` is exactly what hid one.
    fn rows(answer: Option<NumericAnswer>) -> Vec<u64> {
        match answer {
            Some(NumericAnswer::Rows(it)) => it.collect(),
            Some(NumericAnswer::NotReady) => panic!("column is still Building"),
            None => panic!("predicate is not servable by the numeric index"),
        }
    }

    /// The online-build state machine at the column level: a `Building` column gates reads
    /// (answers `NotReady`, so the caller scan-falls-back rather than erroring), a stale epoch
    /// cannot publish, and the single install commit subtracts TOMB from the stale BASE before
    /// replaying DELTA.
    #[test]
    fn building_column_gates_reads_until_finished() {
        let (label, attr) = (arc("Person"), arc("age"));
        let eq = |v: i64| IndexQuery::Equal {
            key: attr.clone(),
            value: Value::Int(v),
        };
        let key = |v: i64| encode_numeric(&Value::Int(v)).unwrap();
        let hits = |idx: &FalkorDbIndex, v: i64| -> Vec<u64> {
            rows(idx.query_numeric(Node, &label, &eq(v)))
        };
        // `NotReady`, specifically — not `None`. A `Building` column is a timing state and the
        // caller scans; `None` means unservable, which the no-fallback build turns into a query
        // error. Collapsing the two errored every read against a still-building index.
        let building = |idx: &FalkorDbIndex, v: i64| {
            matches!(
                idx.query_numeric(Node, &label, &eq(v)),
                Some(NumericAnswer::NotReady)
            )
        };

        let mut idx = FalkorDbIndex::new();
        let epoch = idx.create_building(Node, &label, &attr);
        assert!(building(&idx, 10), "empty Building → NotReady");

        // Writes landing during the build. 30 is created (DELTA); 20 is destroyed, which goes to
        // TOMB *and* to the column tree — the pair that stops the stale base resurrecting it.
        write(&mut idx, &label, &attr, &[(30, 2, true)]);
        write(&mut idx, &label, &attr, &[(20, 1, false)]);
        assert!(
            building(&idx, 30),
            "still Building → NotReady even with a live delta"
        );
        assert_eq!(idx.building_columns().len(), 1);

        // A stale epoch — as if the column had been dropped and re-created — installs nothing and
        // must not flip the column `Ready`.
        assert!(
            !idx.install_base(
                Node,
                &label,
                &attr,
                epoch + 1,
                EncodedTuples::scalars(vec![(key(99), 9)])
            ),
            "stale epoch cannot install"
        );
        assert!(building(&idx, 99), "stale epoch cannot publish");
        assert_eq!(
            idx.building_columns().len(),
            1,
            "still Building after a stale install"
        );

        // The real install, one commit: BASE holds the snapshot-era rows 10 and 20.
        assert!(idx.install_base(
            Node,
            &label,
            &attr,
            epoch,
            EncodedTuples::scalars(vec![(key(10), 0), (key(20), 1)])
        ));
        assert!(idx.building_columns().is_empty(), "no longer building");
        assert_eq!(hits(&idx, 10), vec![0], "untouched base row survives");
        assert!(
            hits(&idx, 20).is_empty(),
            "TOMB must stop the deleted row resurrecting from the stale base"
        );
        assert_eq!(
            hits(&idx, 30),
            vec![2],
            "DELTA row preserved by the install"
        );
        assert!(hits(&idx, 99).is_empty(), "stale-epoch value never landed");
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
        assert_eq!(rows(idx.query_numeric(Node, &label, &eq)), vec![1]);
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
        // A conjunction on ONE attribute is served — it folds to a single window.
        let eq2 = IndexQuery::Equal {
            key: key.clone(),
            value: Value::Int(30),
        };
        assert!(matches!(
            idx.query_numeric(Node, &label, &IndexQuery::And(vec![eq2])),
            Some(NumericAnswer::Rows(_))
        ));
        // A conjunction spanning TWO attributes needs a column for each; `height` has none, so
        // it is declined rather than answered from `age` alone (served case: see
        // `cross_attribute_conjunctions_intersect_columns`).
        assert!(
            idx.query_numeric(
                Node,
                &label,
                &IndexQuery::And(vec![
                    IndexQuery::Equal {
                        key: key.clone(),
                        value: Value::Int(30),
                    },
                    IndexQuery::Equal {
                        key: arc("height"),
                        value: Value::Int(5),
                    },
                ])
            )
            .is_none()
        );
        // Missing column and the wrong entity still fall through.
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

    /// A conjunction across two attributes is answered by intersecting their columns.
    ///
    /// The streams are in **value** order, not id order, so this cannot be a sorted merge — the
    /// first attribute is materialised into a set and the rest probe it. Verified against the
    /// RediSearch build: same rows, different order, which Cypher permits without `ORDER BY`.
    #[test]
    fn cross_attribute_conjunctions_intersect_columns() {
        let label = arc("C");
        let (a, b) = (arc("a"), arc("b"));
        let mut idx = FalkorDbIndex::new();
        idx.create_numeric(Node, &label, &a);
        idx.create_numeric(Node, &label, &b);
        // ids 0..9; a = id % 2, b = id
        for id in 0..10u64 {
            idx.numeric_mut(Node, &label, &a)
                .unwrap()
                .add(&Value::Int((id % 2) as i64), id);
            idx.numeric_mut(Node, &label, &b)
                .unwrap()
                .add(&Value::Int(id as i64), id);
        }
        let eq = |k: &Arc<String>, v: i64| IndexQuery::Equal {
            key: k.clone(),
            value: Value::Int(v),
        };
        let gt = |k: &Arc<String>, v: i64| IndexQuery::Range {
            key: k.clone(),
            min: Some(Value::Int(v)),
            max: None,
            include_min: false,
            include_max: true,
        };
        let got = |q: IndexQuery<Value>| {
            let mut v = rows(idx.query_numeric(Node, &label, &q));
            v.sort_unstable();
            v
        };

        // odd ids above 4
        assert_eq!(
            got(IndexQuery::And(vec![eq(&a, 1), gt(&b, 4)])),
            vec![5, 7, 9]
        );
        // Each attribute's own conjuncts are folded by its column first, so three conjuncts
        // across two attributes still means two streams.
        assert_eq!(
            got(IndexQuery::And(vec![eq(&a, 1), gt(&b, 4), eq(&b, 7)])),
            vec![7]
        );
        // An empty intersection is an answer, not a decline.
        assert!(got(IndexQuery::And(vec![eq(&a, 0), eq(&b, 7)])).is_empty());

        // A missing column cannot be intersected — decline rather than answer from a subset,
        // which would return rows the conjunction excludes.
        assert!(
            idx.query_numeric(
                Node,
                &label,
                &IndexQuery::And(vec![eq(&a, 1), eq(&arc("nope"), 1)])
            )
            .is_none()
        );
    }

    /// A column still building makes the whole conjunction `NotReady`. Intersecting against a
    /// half-populated column would silently drop rows, which is worse than scanning.
    #[test]
    fn a_building_column_makes_the_intersection_not_ready() {
        let label = arc("C");
        let (a, b) = (arc("a"), arc("b"));
        let mut idx = FalkorDbIndex::new();
        idx.create_numeric(Node, &label, &a);
        idx.numeric_mut(Node, &label, &a)
            .unwrap()
            .add(&Value::Int(1), 1);
        idx.create_building(Node, &label, &b);

        let q = IndexQuery::And(vec![
            IndexQuery::Equal {
                key: a,
                value: Value::Int(1),
            },
            IndexQuery::Equal {
                key: b,
                value: Value::Int(1),
            },
        ]);
        assert!(matches!(
            idx.query_numeric(Node, &label, &q),
            Some(NumericAnswer::NotReady)
        ));
    }
}
