//! Folded-roots MVCC holder (PR2 · P3/P4a): the FalkorDB index state that lives
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

use crate::entity_type::EntityType;
use crate::index::IndexQuery;
use crate::runtime::value::Value;

use super::doc_iter::DocIter;
use super::numeric::NumericIndex;
use super::range::{EncodedTuples, RangeIndex};

/// Identifies one index column: `(label, attribute)`.
///
/// Placeholder key shape for P4 — a later step may switch to interned numeric
/// ids for compactness once the write/read wiring already resolves names.
pub type IndexKey = (Arc<String>, Arc<String>);

/// A staged batch of maintenance for one commit: `(label, attr) → [(value, id)]`,
/// applied per column by [`FalkorDbIndex::merge`].
pub type StagedColumns = HashMap<IndexKey, Vec<(Value, u64)>>;

/// One index column, of a specific kind.
///
/// Index kinds are a **closed set we own**, so this is a static enum — no
/// `Box<dyn>`, no boxed iterators, and exhaustive matching forces every kind to
/// be handled (an unimplemented kind fails loudly, never silently). The uniform
/// lifecycle ([`add`](Self::add)/[`remove`](Self::remove)) is delegated here.
///
/// `Range` is the one kind a `CREATE INDEX ... ON (n.p)` makes, and it is itself the union of
/// three value kinds (numeric, tag, geo) over one column — see [`RangeIndex`], which owns the
/// routing between them. A Vector variant lands with its kind; Fulltext stays on RediSearch,
/// which is why there is no variant for it.
#[derive(Clone)]
pub enum IndexColumn {
    Range(RangeIndex),
}

impl IndexColumn {
    /// Index `id` under `value` — the new value on create/update.
    pub fn add(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        match self {
            Self::Range(idx) => idx.add(value, id),
        }
    }

    /// Remove `id` under `value` — the old value on delete/update.
    pub fn remove(
        &mut self,
        value: &Value,
        id: u64,
    ) {
        match self {
            Self::Range(idx) => idx.remove(value, id),
        }
    }

    /// Add a batch of `(value, id)` entries — the columnar write path's add column.
    pub fn add_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
    ) {
        match self {
            Self::Range(idx) => idx.add_batch(entries),
        }
    }

    /// Remove a batch of `(value, id)` entries — the columnar write path's remove column.
    pub fn remove_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
    ) {
        match self {
            Self::Range(idx) => idx.remove_batch(entries),
        }
    }

    /// Encode `(value, id)` entries under *this* column's kind, dropping whatever the kind
    /// does not index. Lets the background job encode BASE off-thread, so the install commit
    /// pays only the tree build.
    ///
    /// Takes `&self`, not an associated function, because the tag kind's encoding is relative to
    /// *this column's* dictionary — a BASE encoded against any other numbering is unmergeable.
    #[must_use]
    pub fn encode_entries(
        &self,
        entries: Vec<(Value, u64)>,
    ) -> EncodedTuples {
        match self {
            Self::Range(idx) => idx.encode_entries(entries),
        }
    }

    /// A new column of *this* column's kind, built from already-encoded tuples — how the
    /// install adopts BASE without hard-coding a kind at the call site. Carries this column's
    /// tag dictionary across, for the same reason [`encode_entries`](Self::encode_entries) takes
    /// `&self`.
    #[must_use]
    pub fn new_like_from_encoded(
        &self,
        tuples: EncodedTuples,
    ) -> Self {
        match self {
            Self::Range(idx) => Self::Range(idx.from_encoded_like(tuples)),
        }
    }

    /// Every `(key, doc)` tuple this column holds, encoded — the install's DELTA/TOMB
    /// enumeration. Encoded, not `Value`s: TOMB must use the same equivalence relation as
    /// the column it will be subtracted from, and decoding to re-encode would be a second
    /// trip through a many-to-one map.
    #[must_use]
    pub fn encoded_tuples(&self) -> EncodedTuples {
        match self {
            Self::Range(idx) => idx.encoded_tuples(),
        }
    }

    /// Add already-encoded tuples (install: replay DELTA onto BASE).
    pub fn add_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        match self {
            Self::Range(idx) => idx.add_encoded(tuples),
        }
    }

    /// Remove already-encoded tuples (install: subtract TOMB from BASE).
    pub fn remove_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        match self {
            Self::Range(idx) => idx.remove_encoded(tuples),
        }
    }

    /// Whether this column holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Range(idx) => idx.is_empty(),
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
pub enum IndexAnswer<I = DocIter> {
    Rows(I),
    NotReady,
}

impl<I> IndexAnswer<I> {
    /// Map the rows, leaving `NotReady` alone. `f` runs only when there are rows, so a caller may
    /// move setup into it that would be wasted on a not-ready column.
    pub fn map_rows<J>(
        self,
        f: impl FnOnce(I) -> J,
    ) -> IndexAnswer<J> {
        match self {
            Self::Rows(rows) => IndexAnswer::Rows(f(rows)),
            Self::NotReady => IndexAnswer::NotReady,
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
/// BASE before DELTA is replayed. The graph's `deleted_nodes` / `deleted_relationships`
/// bitmaps are applied first as a cheap backstop
/// ([`Graph::install_index_base`](crate::graph::Graph)), but they are only that — the
/// bitmaps are a free list and are cleared on id reuse, so TOMB is the real defence.
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
    /// `epoch` is the build's identity: a monotonic token assigned at `create_building` and carried
    /// by the background job. `install_base` no-ops unless the column's current epoch matches the
    /// job's — so if the column is dropped and re-created mid-build, the stale job installs nothing
    /// into the new column (which has a fresh epoch) and a new job builds it instead.
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
        removed: &[(Value, u64)],
    ) {
        if let ColumnState::Building { tomb, .. } = &mut self.state {
            tomb.add_batch(removed.iter().cloned());
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
    /// Monotonic build-epoch source (see [`ColumnState::Building`]). Bumped by `create_building`,
    /// which only runs under the serialized write path (forked from the latest committed version),
    /// so it increases monotonically across the committed lineage — every online build gets a
    /// process-unique id. `0` is never a live building epoch (the counter pre-increments).
    next_build_epoch: u64,
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

    /// Create an empty Range column for `(entity, label, attr)` in the `Ready`
    /// state, replacing any existing one. Used by the synchronous paths (RDB load /
    /// replica), where the caller fills it via [`build_column`](Self::build_column)
    /// before any read is served. An edge column's docs are `edge_id`s.
    pub fn create_column(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) {
        self.columns_mut(entity).insert(
            (label.clone(), attr.clone()),
            ColumnEntry::ready(IndexColumn::Range(RangeIndex::new())),
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
        self.next_build_epoch += 1;
        let epoch = self.next_build_epoch;
        let column = RangeIndex::new();
        self.columns_mut(entity).insert(
            (label.clone(), attr.clone()),
            ColumnEntry {
                // TOMB shares the column's tag dictionary: its tuples are subtracted from the
                // column's, so both sides have to number the same strings the same way.
                state: ColumnState::Building {
                    tomb: IndexColumn::Range(column.empty_like()),
                    epoch,
                },
                column: IndexColumn::Range(column),
            },
        );
        epoch
    }

    /// Build (or rebuild) the column for `(entity, label, attr)` from `entries` (any order) in the
    /// `Ready` state, replacing any existing one — the bulk populate path. Each kind bottom-up-loads
    /// its sorted pairs, which is cheaper than looping incremental adds. A value no kind indexes
    /// (`NULL`, `NaN`, a map) is skipped, exactly as the RediSearch Range field skips it.
    pub fn build_column<'a>(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        entries: impl IntoIterator<Item = (&'a Value, u64)>,
    ) {
        self.columns_mut(entity).insert(
            (label.clone(), attr.clone()),
            ColumnEntry::ready(IndexColumn::Range(RangeIndex::from_entries(entries))),
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

    /// The column for `(entity, label, attr)`, if one exists.
    /// State-agnostic (inspects a `Building` or `Ready` column alike) — used by the
    /// populate path and tests, not the read path (which gates on state).
    #[must_use]
    pub fn column(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<&RangeIndex> {
        match self.columns(entity).get(&(label.clone(), attr.clone())) {
            Some(ColumnEntry {
                column: IndexColumn::Range(idx),
                ..
            }) => Some(idx),
            None => None,
        }
    }

    /// Mutable column for `(entity, label, attr)`, if one exists.
    pub fn column_mut(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<&mut RangeIndex> {
        match self
            .columns_mut(entity)
            .get_mut(&(label.clone(), attr.clone()))
        {
            Some(ColumnEntry {
                column: IndexColumn::Range(idx),
                ..
            }) => Some(idx),
            None => None,
        }
    }

    /// The numeric kind of the column for `(entity, label, attr)` — the accessor that predates
    /// the tag and geo kinds, kept for the callers that only ever mean numbers.
    #[must_use]
    pub fn numeric(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<&NumericIndex> {
        self.column(entity, label, attr).map(RangeIndex::numeric)
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

    /// Apply one commit's batched maintenance for `entity`: **remove** the old-value
    /// tuples, then **add** the new-value tuples, one batched tree op per touched
    /// column. This is the single merge primitive — the caller only stages
    /// `(value, id)` from graph state; the per-column batching lives here, and both
    /// the steady-state write path and the background-build install go through it.
    ///
    /// Removes precede adds so a `(value, id)` removed and re-added in the same commit
    /// nets to present. Each side is a batched op, so on disk this is one page-flush
    /// pass per column — the merge is storage-agnostic (only the medium behind the
    /// page store changes).
    pub fn merge(
        &mut self,
        entity: EntityType,
        adds: StagedColumns,
        removes: StagedColumns,
    ) {
        let columns = self.columns_mut(entity);
        for (key, entries) in removes {
            if let Some(entry) = columns.get_mut(&key) {
                // TOMB for the install's BASE subtraction, *and* the column tree itself: I-W4'
                // requires DELTA to hold each touched entity's final state, not an append-only
                // add-log. Skipping the tree here would install two rows for `v0 -> v1 -> v2`.
                entry.note_tomb(&entries);
                entry.column.remove_batch(entries);
            }
        }
        for (key, entries) in adds {
            if let Some(entry) = columns.get_mut(&key) {
                entry.column.add_batch(entries);
            }
        }
    }

    /// Install a background-built BASE and flip the column to `Ready` — **one commit**, per
    /// I-B3. No-op (returning `false`) if the column is gone, already `Ready`, or carries a
    /// different epoch, which is how a drop-and-recreate mid-build makes the stale job inert.
    ///
    /// ```text
    /// new := BASE                 (adopted wholesale — the job already built the tree)
    /// new.remove(TOMB)            (removes strictly first)
    /// new.add(DELTA)              (then adds)
    /// column := new, state := Ready
    /// ```
    ///
    /// **Order is load-bearing.** Install starts *from* BASE, so every snapshot-era tuple is
    /// present and the only question is TOMB-vs-DELTA. Applying DELTA first would then delete
    /// exactly `TOMB ∩ DELTA` — every tuple destroyed and recreated during the build, e.g.
    /// `v0 -> v1 -> v0` or an id reused with the same value. By I-W4' no DELTA tuple is invalid
    /// at the version being installed into, so TOMB never needs to fire after DELTA.
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
        let mut new = entry.column.new_like_from_encoded(base);
        let mut tomb_tuples = tomb.encoded_tuples();
        new.remove_encoded(&mut tomb_tuples);
        let mut delta = entry.column.encoded_tuples();
        new.add_encoded(&mut delta);
        entry.column = new;
        entry.state = ColumnState::Ready;
        true
    }

    /// A conjunction spanning several attributes: intersect one column per attribute.
    ///
    /// Each attribute's own conjuncts are folded by its own column first (see
    /// `RangeIndex::intersect_refs`), so this only combines one doc stream per attribute.
    ///
    /// **Set probe, not a sorted merge.** `DocIter` yields docs in *value* order, not id order —
    /// deliberately, since Cypher guarantees no order without `ORDER BY`, and merging would cost
    /// a comparison per doc and force every branch to stay live. Value-ordered streams cannot be
    /// merge-intersected, so the first attribute is materialised into a set and the rest probe it.
    /// Memory is proportional to the first attribute's match count; there is no cardinality
    /// estimate to pick the smallest side with, so the order is the query's.
    ///
    /// Every attribute needs a `Ready` column. One still `Building` makes the whole
    /// conjunction `NotReady` — intersecting against a half-built column would silently drop rows,
    /// which is worse than scanning.
    fn intersect_columns(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        children: &[IndexQuery<Value>],
    ) -> Option<IndexAnswer> {
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
                    // A union (or array-contains, or distance) conjunct belongs to whichever
                    // attribute it constrains, and its column answers it whole. Without this,
                    // `p.name IN [..] AND p.age IN [..]` — two servable unions — was declined for
                    // being neither one column's problem nor groupable into two.
                    IndexQuery::Or(_)
                    | IndexQuery::ArrayContains { .. }
                    | IndexQuery::Point { .. } => {
                        let mut keys = Vec::new();
                        if !attributes_of(child, &mut keys) {
                            return None;
                        }
                        let first = *keys.first()?;
                        if keys.iter().any(|k| *k != first) {
                            return None; // one conjunct spanning columns is not groupable
                        }
                        first
                    }
                    IndexQuery::InList { .. } => return None,
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
                return Some(IndexAnswer::NotReady);
            }
            let IndexColumn::Range(idx) = &entry.column;
            let folded = idx.intersect_refs(&conjuncts)?;
            result = Some(match result {
                None => folded.collect(),
                Some(seen) => folded.filter(|doc| seen.contains(doc)).collect(),
            });
            if result.as_ref().is_some_and(FxHashSet::is_empty) {
                break; // a later conjunct can only remove rows, never add them
            }
        }
        result.map(|docs| {
            IndexAnswer::Rows(DocIter::Set(
                docs.into_iter().collect::<Vec<_>>().into_iter(),
            ))
        })
    }

    /// A union spanning several attributes: union one column per attribute.
    ///
    /// `n.name IN [...] OR n.age = 33` is answerable, and unlike the conjunction case it has to be
    /// answered by *combining columns* rather than by one of them: asking a single column for both
    /// values would look the age up among the names and return rows the query never asked for.
    ///
    /// **Materialised, not chained.** A doc may satisfy several members (a node whose name matches
    /// *and* whose age is 33), and `OR` must yield it once; the streams are in value order per
    /// column, so there is nothing to merge-dedup against. The set is the dedup.
    ///
    /// Every attribute needs a `Ready` column. One still `Building` makes the whole union
    /// `NotReady` — dropping a member's rows would be a wrong answer, not a slower one.
    fn union_columns(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        children: &[IndexQuery<Value>],
    ) -> Option<IndexAnswer> {
        // Regroup the members per attribute, flattening the nesting `a IN [..] OR b IN [..]`
        // arrives with, so each column is asked once for its whole share.
        fn group<'a>(
            children: &'a [IndexQuery<Value>],
            out: &mut Vec<(&'a Arc<String>, Vec<&'a IndexQuery<Value>>)>,
        ) -> Option<()> {
            for child in children {
                let key = match child {
                    IndexQuery::Equal { key, .. } => key,
                    IndexQuery::Or(nested) => {
                        group(nested, out)?;
                        continue;
                    }
                    // A range or other combinator inside a union is not a shape any column has
                    // been asked to serve yet.
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

        let mut docs: FxHashSet<u64> = FxHashSet::default();
        for (attr, members) in per_attr {
            let entry = self.columns(entity).get(&(label.clone(), attr.clone()))?;
            if !matches!(entry.state, ColumnState::Ready) {
                return Some(IndexAnswer::NotReady);
            }
            let IndexColumn::Range(idx) = &entry.column;
            docs.extend(idx.union_refs(&members)?);
        }
        Some(IndexAnswer::Rows(DocIter::Set(
            docs.into_iter().collect::<Vec<_>>().into_iter(),
        )))
    }

    /// Encode `entries` under the kind of the column `(entity, label, attr)`, if it exists.
    /// `None` when there is no such column — the build was for a column since dropped.
    #[must_use]
    pub fn encode_for_column(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        entries: Vec<(Value, u64)>,
    ) -> Option<EncodedTuples> {
        self.columns(entity)
            .get(&(label.clone(), attr.clone()))
            .map(|entry| entry.column.encode_entries(entries))
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

    /// Answer an index query for `(entity, label)`. Yields docs — node ids for a node column,
    /// `edge_id`s for an edge column.
    ///
    /// This layer answers only *which column(s)* a predicate needs and whether they can be read
    /// yet; whether the predicate itself is servable is [`RangeIndex::query`]'s business, since
    /// that depends on the value's type and therefore on the kinds inside the column.
    ///
    /// Three outcomes, and the caller must keep them apart (see [`IndexAnswer`]):
    ///
    /// * [`Rows`](IndexAnswer::Rows) — served here. Possibly zero rows: a predicate Cypher says
    ///   cannot match (`n.v = null`, a cross-type comparison) is *answered*, not declined.
    /// * [`NotReady`](IndexAnswer::NotReady) — the column exists but is still `Building`, so its
    ///   base is not installed. The caller scans; this is a timing state, never an error.
    /// * `None` — no such column, or no kind in it can serve this predicate. Under
    ///   `index-falkordb` there is no other index, and the caller turns this into an error.
    #[must_use]
    pub fn query_column(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        query: &IndexQuery<Value>,
    ) -> Option<IndexAnswer> {
        let mut keys = Vec::new();
        if !attributes_of(query, &mut keys) || keys.is_empty() {
            return None; // a shape with no attribute to route by (an empty composite, an `InList`)
        }
        let first = keys[0];
        if keys.iter().any(|k| *k != first) {
            // Several attributes: combine one column per attribute — intersect for a conjunction,
            // union for a disjunction. What neither can do is answer from a single column: looking
            // every value up in one of them would return rows the query never asked for.
            return match query {
                IndexQuery::And(children) => self.intersect_columns(entity, label, children),
                IndexQuery::Or(children) => self.union_columns(entity, label, children),
                _ => None,
            };
        }
        let entry = self.columns(entity).get(&(label.clone(), first.clone()))?;
        if !matches!(entry.state, ColumnState::Ready) {
            // Building: the base is not installed yet. This is NOT an unsupported predicate —
            // the column will serve it once the build finishes — so the caller must scan rather
            // than error. `NotReady` keeps that distinct from `None`, which under the
            // no-fallback build is a hard failure.
            return Some(IndexAnswer::NotReady);
        }
        match &entry.column {
            IndexColumn::Range(idx) => idx.query(query).map(IndexAnswer::Rows),
        }
    }
}

/// Collect the attributes a predicate constrains, flattening nested composites. `false` when the
/// shape names no attribute at all — an empty `And`/`Or`, or an `InList` the runtime should have
/// desugared into a union before the index ever saw it.
///
/// Deliberately says nothing about *values*: which types a column can serve is decided inside the
/// column, where the kinds live. Duplicating that judgement here is what made the previous version
/// declare a string leaf unservable before the tag kind existed to be asked.
pub(super) fn attributes_of<'a>(
    query: &'a IndexQuery<Value>,
    out: &mut Vec<&'a Arc<String>>,
) -> bool {
    match query {
        IndexQuery::Equal { key, .. }
        | IndexQuery::Range { key, .. }
        | IndexQuery::ArrayContains { key, .. }
        | IndexQuery::Point { key, .. } => {
            out.push(key);
            true
        }
        IndexQuery::And(children) | IndexQuery::Or(children) => {
            !children.is_empty() && children.iter().all(|c| attributes_of(c, out))
        }
        IndexQuery::InList { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::encode::encode_numeric;
    use super::*;
    use crate::entity_type::EntityType::{Node, Relationship};

    fn arc(s: &str) -> Arc<String> {
        Arc::new(s.to_string())
    }

    /// Rows of an answer that must be `Rows`. Panics on the other two, rather than reporting them
    /// as "no rows" — a test that expected rows and got `NotReady` or `None` has found a bug, and
    /// collapsing all three into an empty `Vec` is exactly what hid one.
    fn rows(answer: Option<IndexAnswer>) -> Vec<u64> {
        match answer {
            Some(IndexAnswer::Rows(it)) => it.collect(),
            Some(IndexAnswer::NotReady) => panic!("column is still Building"),
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
        let staged = |v: i64, id: u64| {
            let mut m: StagedColumns = HashMap::default();
            m.insert((label.clone(), attr.clone()), vec![(Value::Int(v), id)]);
            m
        };
        let key = |v: i64| encode_numeric(&Value::Int(v)).unwrap();
        let hits = |idx: &FalkorDbIndex, v: i64| -> Vec<u64> {
            rows(idx.query_column(Node, &label, &eq(v)))
        };
        // `NotReady`, specifically — not `None`. A `Building` column is a timing state and the
        // caller scans; `None` means unservable, which the no-fallback build turns into a query
        // error. Collapsing the two errored every read against a still-building index.
        let building = |idx: &FalkorDbIndex, v: i64| {
            matches!(
                idx.query_column(Node, &label, &eq(v)),
                Some(IndexAnswer::NotReady)
            )
        };

        let mut idx = FalkorDbIndex::new();
        let epoch = idx.create_building(Node, &label, &attr);
        assert!(building(&idx, 10), "empty Building → NotReady");

        // Writes landing during the build. 30 is created (DELTA); 20 is destroyed, which goes to
        // TOMB *and* to the column tree — the pair that stops the stale base resurrecting it.
        idx.merge(Node, staged(30, 2), HashMap::default());
        idx.merge(Node, HashMap::default(), staged(20, 1));
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
        v1.create_column(Node, &label, &attr);
        v1.column_mut(Node, &label, &attr)
            .unwrap()
            .add(&Value::Int(30), 1);

        let mut v2 = v1.clone(); // the O(1) fork
        v2.column_mut(Node, &label, &attr)
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

        idx.create_column(Node, &label, &attr);
        assert_eq!(idx.len(), 1);
        assert!(idx.numeric(Node, &label, &attr).is_some());
        assert!(idx.numeric(Node, &label, &attr).unwrap().is_empty());
    }

    /// `merge` batches adds+removes into the right columns (removes before adds).
    #[test]
    fn merge_batches_into_columns() {
        let (label, attr) = (arc("Person"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        idx.create_column(Node, &label, &attr);

        let staged = |vals: &[(i64, u64)]| -> StagedColumns {
            let mut m: StagedColumns = HashMap::new();
            m.insert(
                (label.clone(), attr.clone()),
                vals.iter().map(|&(v, id)| (Value::Int(v), id)).collect(),
            );
            m
        };
        idx.merge(Node, staged(&[(7, 1), (9, 2)]), StagedColumns::new());
        idx.merge(Node, StagedColumns::new(), staged(&[(7, 1)]));

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
        idx.create_column(Node, &name, &attr);
        idx.create_column(Relationship, &name, &attr);
        assert_eq!(idx.len(), 2, "one node column + one edge column");

        idx.column_mut(Node, &name, &attr)
            .unwrap()
            .add(&Value::Int(1), 10);
        idx.column_mut(Relationship, &name, &attr)
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

    /// Reads route each leaf to the column its attribute names, for either entity. A node query on
    /// a shared name must not see the edge column, and vice-versa.
    #[test]
    fn query_routes_leaves_to_the_named_column_per_entity() {
        let (label, attr) = (arc("Person"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        idx.create_column(Node, &label, &attr);
        for (v, id) in [(Value::Int(30), 1), (Value::String(arc("x")), 2)] {
            idx.column_mut(Node, &label, &attr).unwrap().add(&v, id);
        }
        let key = attr.clone();

        // Numeric Equal / Range → routed to the node column, correct ids.
        let eq = IndexQuery::Equal {
            key: key.clone(),
            value: Value::Int(30),
        };
        assert_eq!(rows(idx.query_column(Node, &label, &eq)), vec![1]);
        let rg = IndexQuery::Range {
            key: key.clone(),
            min: Some(Value::Int(10)),
            max: None,
            include_min: true,
            include_max: true,
        };
        assert!(idx.query_column(Node, &label, &rg).is_some());

        // A string on the same column reaches the tag kind — the routing is by the operand's
        // type, and both kinds live in the one column.
        let str_eq = IndexQuery::Equal {
            key: key.clone(),
            value: Value::String(arc("x")),
        };
        assert_eq!(rows(idx.query_column(Node, &label, &str_eq)), vec![2]);
        assert!(
            rows(idx.query_column(
                Node,
                &label,
                &IndexQuery::Equal {
                    key: key.clone(),
                    value: Value::String(arc("never stored")),
                }
            ))
            .is_empty()
        );
        // A conjunction on ONE attribute is served — it folds to a single window.
        let eq2 = IndexQuery::Equal {
            key: key.clone(),
            value: Value::Int(30),
        };
        assert!(matches!(
            idx.query_column(Node, &label, &IndexQuery::And(vec![eq2])),
            Some(IndexAnswer::Rows(_))
        ));
        // A conjunction spanning TWO attributes is not: that needs an intersection across
        // columns, which this column-scoped lookup cannot do.
        assert!(
            idx.query_column(
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
        assert!(idx.query_column(Node, &label, &other).is_none());
        assert!(
            idx.query_column(Relationship, &label, &eq).is_none(),
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
        idx.create_column(Node, &label, &a);
        idx.create_column(Node, &label, &b);
        // ids 0..9; a = id % 2, b = id
        for id in 0..10u64 {
            idx.column_mut(Node, &label, &a)
                .unwrap()
                .add(&Value::Int((id % 2) as i64), id);
            idx.column_mut(Node, &label, &b)
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
            let mut v = rows(idx.query_column(Node, &label, &q));
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
            idx.query_column(
                Node,
                &label,
                &IndexQuery::And(vec![eq(&a, 1), eq(&arc("nope"), 1)])
            )
            .is_none()
        );
    }

    /// `p.name IN [..] AND p.age IN [..]` — a conjunction whose children are *unions*, on two
    /// attributes. Each column answers its own union and the results intersect.
    ///
    /// It reaches the index in exactly this shape (the runtime desugars `IN` into an `Or` before
    /// the index is consulted), and it is the shape `test05_test_in_operator_string_props`
    /// exercises. It used to be declined for being neither a single column's problem nor a
    /// conjunction of plain leaves.
    #[test]
    fn a_conjunction_of_unions_intersects_the_columns() {
        let label = arc("person");
        let (name, age) = (arc("name"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        idx.create_column(Node, &label, &name);
        idx.create_column(Node, &label, &age);
        for (id, n, a) in [(0u64, "Gal Derriere", 26i64), (1, "Lucy Yanfital", 30)] {
            idx.column_mut(Node, &label, &name)
                .unwrap()
                .add(&Value::String(arc(n)), id);
            idx.column_mut(Node, &label, &age)
                .unwrap()
                .add(&Value::Int(a), id);
        }
        let names = IndexQuery::Or(vec![
            IndexQuery::Equal {
                key: name.clone(),
                value: Value::String(arc("Gal Derriere")),
            },
            IndexQuery::Equal {
                key: name.clone(),
                value: Value::String(arc("Lucy Yanfital")),
            },
        ]);
        let ages = IndexQuery::Or(vec![IndexQuery::Equal {
            key: age.clone(),
            value: Value::Int(30),
        }]);

        // Each union alone.
        let mut both = rows(idx.query_column(Node, &label, &names));
        both.sort_unstable();
        assert_eq!(both, vec![0, 1]);
        assert_eq!(rows(idx.query_column(Node, &label, &ages)), vec![1]);

        // And their conjunction: only Lucy is 30.
        assert_eq!(
            rows(idx.query_column(Node, &label, &IndexQuery::And(vec![names, ages]))),
            vec![1]
        );
    }

    /// `p.name IN [..] OR p.age = 33` — a union spanning two columns. Each column answers its own
    /// members and the results are combined, deduplicated: a doc satisfying both members is one
    /// row in Cypher, not two.
    #[test]
    fn a_union_across_columns_combines_and_dedups() {
        let label = arc("person");
        let (name, age) = (arc("name"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        idx.create_column(Node, &label, &name);
        idx.create_column(Node, &label, &age);
        // Doc 1 satisfies BOTH members — the dedup case.
        for (id, n, a) in [(0u64, "Gal", 26i64), (1, "Lucy", 33), (2, "Omri", 33)] {
            idx.column_mut(Node, &label, &name)
                .unwrap()
                .add(&Value::String(arc(n)), id);
            idx.column_mut(Node, &label, &age)
                .unwrap()
                .add(&Value::Int(a), id);
        }
        let q = IndexQuery::Or(vec![
            IndexQuery::Or(vec![
                IndexQuery::Equal {
                    key: name.clone(),
                    value: Value::String(arc("Gal")),
                },
                IndexQuery::Equal {
                    key: name.clone(),
                    value: Value::String(arc("Lucy")),
                },
            ]),
            IndexQuery::Equal {
                key: age.clone(),
                value: Value::Int(33),
            },
        ]);
        let mut got = rows(idx.query_column(Node, &label, &q));
        got.sort_unstable();
        assert_eq!(got, vec![0, 1, 2]);
        assert_eq!(
            got.len(),
            3,
            "Lucy matches both members and must appear once"
        );

        // A missing column cannot be unioned: answering from the rest would drop its rows.
        let partial = IndexQuery::Or(vec![
            IndexQuery::Equal {
                key: name,
                value: Value::String(arc("Gal")),
            },
            IndexQuery::Equal {
                key: arc("nope"),
                value: Value::Int(1),
            },
        ]);
        assert!(idx.query_column(Node, &label, &partial).is_none());
    }

    /// A column still building makes the whole conjunction `NotReady`. Intersecting against a
    /// half-populated column would silently drop rows, which is worse than scanning.
    #[test]
    fn a_building_column_makes_the_intersection_not_ready() {
        let label = arc("C");
        let (a, b) = (arc("a"), arc("b"));
        let mut idx = FalkorDbIndex::new();
        idx.create_column(Node, &label, &a);
        idx.column_mut(Node, &label, &a)
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
            idx.query_column(Node, &label, &q),
            Some(IndexAnswer::NotReady)
        ));
    }
}
