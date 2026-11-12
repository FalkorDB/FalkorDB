//! Folded-roots MVCC holder: the FalkorDB index state that lives
//! on [`Graph`](crate::graph::Graph) as its own copy-on-write field, so
//! `Graph::new_version()` forks it in `O(1)` per column and the committed-version
//! `Arc<Graph>` swap publishes graph + index together — one atomic
//! step, no torn read.
//!
//! Deliberately **pure data — no reference back to `Graph`**. The RediSearch
//! `Indexer` holds an `Arc<ArcSwapOption<Graph>>` pointing
//! back at the graph for background population; that cycle is exactly what this
//! avoids. The write path mutates a version's own copy instead.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use roaring::RoaringTreemap;
use rustc_hash::FxHashSet;

use crate::entity_type::EntityType;
use crate::graph::endpoint_index::EndpointIndex;
use crate::graph::graph::{Graph, NodeId, RelationshipId};
use crate::index::{IndexQuery, IndexType};
use crate::runtime::value::Value;

use super::cch::{CchError, CchIndex, CchIndexDef, Hierarchy};
use super::encode::KeyKind;
use super::graph_writes;
use super::numeric::{DocIter, EncodedTuples};
use super::pending_writes::{IndexWrite, PendingIndexWrites, last_write_wins};
use super::range::{RangeIndex, Selection, attributes_of};
use super::versioned::{Absorbed, BuildState, IndexEntry, VersionedIndex};

/// Identifies one index column: `(label, attribute)`.
///
/// Keyed by name for now; interned numeric ids would be more compact once the write and read
/// paths both resolve names to ids.
pub type IndexKey = (Arc<String>, Arc<String>);

/// One index column, of a specific kind.
///
/// Index kinds are a **closed set we own**, so this is a static enum — no
/// `Box<dyn>`, no boxed iterators, and exhaustive matching forces every kind to
/// be handled (an unimplemented kind fails loudly, never silently). A Range column
/// ([`RangeIndex`]) holds the kinds of one attribute together, as a RediSearch Range field
/// does; the numeric kind is the only one so far. More variants (Text, Vector) land with their
/// kinds.
#[derive(Clone)]
pub enum IndexColumn {
    Range(RangeIndex),
}

impl IndexColumn {
    /// Index `id` under `value` — the new value on create/update. Each kind takes the part of
    /// the value it can represent.
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
        scratch: &mut EncodedTuples,
    ) {
        match self {
            Self::Range(idx) => idx.add_batch(entries, scratch),
        }
    }

    /// Remove a batch of `(value, id)` entries — the columnar write path's remove column.
    pub fn remove_batch(
        &mut self,
        entries: impl IntoIterator<Item = (Value, u64)>,
        scratch: &mut EncodedTuples,
    ) {
        match self {
            Self::Range(idx) => idx.remove_batch(entries, scratch),
        }
    }

    /// The scalar tree and key `value` takes in *this* column, or `None` when no kind indexes it
    /// as a scalar (a list, whose elements are keys of the array trees instead; a temporal; a map).
    #[must_use]
    pub fn encode_value(
        &self,
        value: &Value,
    ) -> Option<(KeyKind, u64)> {
        match self {
            Self::Range(idx) => idx.encode_value(value),
        }
    }

    /// Call `emit` with each key a stored `value` takes in *this* column, and its [`KeyKind`].
    /// Nothing when no kind indexes it; one key per indexed element, of an array kind, for a list.
    pub fn for_each_stored_key(
        &self,
        value: &Value,
        emit: impl FnMut(KeyKind, u64),
    ) {
        match self {
            Self::Range(idx) => idx.for_each_stored_key(value, emit),
        }
    }

    /// Append the tuples `(value, id)` takes in *this* column to `out`, each with the tuples of its
    /// kind. The one place a value becomes a column's tuples. Unsorted and possibly repeated: see
    /// [`EncodedTuples::sort_dedup`].
    pub fn push_encoded(
        &self,
        value: &Value,
        id: u64,
        out: &mut EncodedTuples,
    ) {
        match self {
            Self::Range(idx) => idx.push_encoded(value, id, out),
        }
    }

    /// Add already-encoded tuples (a unit of work's collapsed adds, or an indexer batch).
    pub fn add_encoded(
        &mut self,
        tuples: &mut EncodedTuples,
    ) {
        match self {
            Self::Range(idx) => idx.add_encoded(tuples),
        }
    }

    /// Remove already-encoded tuples (a unit of work's collapsed removes).
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

    /// The ids stored under `key` in the tree of `kind`, a key
    /// [`encode_value`](Self::encode_value) gave.
    #[must_use]
    pub fn rows_at(
        &self,
        kind: KeyKind,
        key: u64,
    ) -> DocIter {
        match self {
            Self::Range(idx) => idx.rows_at(kind, key),
        }
    }

    /// The entities `selection` selects ([`Selection::of`]).
    #[must_use]
    pub fn lookup(
        &self,
        selection: Selection,
    ) -> DocIter {
        match self {
            Self::Range(idx) => idx.lookup(selection),
        }
    }

    /// Refresh `written` in column `name`: their tuples as `from` has them come out, and as `to`
    /// has them go in, while `state` is `Building` only for an entity `to` places below the cursor.
    /// Removing a tuple the column does not hold changes nothing, and an entity whose value did not
    /// change takes its tuple out and puts it back. What a writer's rebase and an indexer batch's
    /// catch-up both do ([`VersionedIndex::writer_rebase`], [`VersionedIndex::indexer_catch_up`]).
    ///
    /// This is `from`'s copy of the column, so the tuples coming out are read with no label check:
    /// the copy holds none for an entity without the label in `from`, and removing one changes
    /// nothing. The tuples going in are checked ([`graph_writes::entity_tuples`]).
    fn refresh(
        &mut self,
        state: &BuildState<BuildCursor>,
        (entity, key): &(EntityType, IndexKey),
        written: &RoaringTreemap,
        from: &Graph,
        to: &Graph,
    ) {
        if written.is_empty() {
            return;
        }
        let (mut removes, mut adds) = (EncodedTuples::default(), EncodedTuples::default());
        let attr = &key.1;
        match entity {
            EntityType::Node => {
                if let Some(idx) = from.get_node_attribute_id(attr) {
                    for id in written {
                        if let Some(value) =
                            from.get_node_attribute_by_idx(NodeId::from(id), idx as u16)
                        {
                            self.push_encoded(&value, id, &mut removes);
                        }
                    }
                }
            }
            EntityType::Relationship => {
                if let Some(idx) = from.get_relationship_attribute_id(attr) {
                    for id in written {
                        if let Some(value) = from
                            .get_relationship_attribute_by_idx(RelationshipId::from(id), idx as u16)
                        {
                            self.push_encoded(&value, id, &mut removes);
                        }
                    }
                }
            }
        }
        // Placed before their values are read, so an entity past the cursor costs a place, not an
        // attribute read.
        match state {
            BuildState::Ready => {
                graph_writes::entity_tuples(to, *entity, key, written, self, &mut adds);
            }
            BuildState::Building { next, .. } => {
                let below = placed_in(written, *entity, None, Some(*next), to);
                graph_writes::entity_tuples(to, *entity, key, &below, self, &mut adds);
            }
        }
        self.remove_encoded(&mut removes);
        self.add_encoded(&mut adds);
    }
}

/// The entities of `written` whose place in a column's build order, in `version`, is at or after
/// `from` and before `next`. A `None` end leaves that side open. Nodes are found by cutting the
/// set at the ends; each edge costs one endpoint lookup. An edge that `version` no longer has is
/// left out.
fn placed_in(
    written: &RoaringTreemap,
    entity: EntityType,
    from: Option<BuildCursor>,
    next: Option<BuildCursor>,
    version: &Graph,
) -> RoaringTreemap {
    match entity {
        EntityType::Node => {
            let mut placed = written.clone();
            if let Some(BuildCursor::Node(from)) = from {
                placed.remove_range(..u64::from(from));
            }
            if let Some(BuildCursor::Node(next)) = next {
                placed.remove_range(u64::from(next)..);
            }
            placed
        }
        EntityType::Relationship => written
            .iter()
            .filter(|&id| {
                version.build_place(entity, id).is_some_and(|at| {
                    from.is_none_or(|from| !at.precedes(from))
                        && next.is_none_or(|next| at.precedes(next))
                })
            })
            .collect(),
    }
}

/// A lookup a column has agreed to answer: which of its entities may hold one value.
///
/// [`FalkorDbIndex::equal_candidates`] makes it without reading the column, so a caller can
/// collect one per entity and run only those it gets to. A live cursor per entity would instead
/// hold a snapshot and a descent stack each.
pub struct EqualCandidates<'a> {
    column: &'a IndexColumn,
    kind: KeyKind,
    key: u64,
}

impl EqualCandidates<'_> {
    /// Every entity holding the value, and possibly others: a column may give several values one
    /// key (the numeric kind keys `1`, `1.0` and `true` alike). Node ids for a node column,
    /// `edge_id`s for an edge column.
    #[must_use]
    pub fn rows(&self) -> DocIter {
        self.column.rows_at(self.kind, self.key)
    }

    /// Which lookup this is: its column and the key it reads. Two lookups with the same id answer
    /// with the same rows.
    #[must_use]
    pub fn id(&self) -> (usize, KeyKind, u64) {
        (std::ptr::from_ref(self.column).addr(), self.kind, self.key)
    }
}

/// A column is a versioned index ([`BuildState`] has the rules its build keeps). It is found by
/// entity and `(label, attr)`, writers stage its changes as a log of encoded tuples, and a batch
/// reads the label's entities in build order ([`BuildCursor`]) into encoded tuples.
impl VersionedIndex for IndexColumn {
    type Cursor = BuildCursor;
    type Changes = Vec<IndexWrite>;
    type Batch = EncodedTuples;
    type Name = (EntityType, IndexKey);
    /// One pair of remove and add buffers for a whole unit of work, reused column to column.
    type Scratch = (EncodedTuples, EncodedTuples);

    fn entry<'a>(
        index: &'a FalkorDbIndex,
        (entity, key): &(EntityType, IndexKey),
    ) -> Option<&'a IndexEntry<Self>> {
        index.columns(*entity).get(key)
    }

    fn entry_mut<'a>(
        index: &'a mut FalkorDbIndex,
        (entity, key): &(EntityType, IndexKey),
    ) -> Option<&'a mut IndexEntry<Self>> {
        index.columns_mut(*entity).get_mut(key)
    }

    /// Keep, while `Building`, only the additions for entities below the cursor, collapse each
    /// tuple's history to its last change ([`last_write_wins`]), then **remove** what ended
    /// removed and **add** what ended added, one batched tree op each.
    ///
    /// After the collapse the two sides are disjoint, so their order no longer matters; removes
    /// still go first.
    fn writer_record(
        &mut self,
        state: &mut BuildState<BuildCursor>,
        mut log: Vec<IndexWrite>,
        place: impl Fn(u64) -> Option<BuildCursor>,
        (removes, adds): &mut (EncodedTuples, EncodedTuples),
    ) {
        if let BuildState::Building { next, written, .. } = state {
            written.by_this_version.extend(log.iter().map(|w| w.id));
            let next = *next;
            // A batch reads an entity at or above the cursor when it reaches it, so its
            // additions go. They go before the collapse, so one cannot cancel an earlier
            // removal of the same tuple.
            log.retain(|w| !w.add || place(w.id).is_some_and(|at| at.precedes(next)));
        }
        last_write_wins(&mut log, removes, adds);
        self.remove_encoded(removes);
        self.add_encoded(adds);
    }

    fn writer_rebase(
        &mut self,
        state: &BuildState<BuildCursor>,
        name: &(EntityType, IndexKey),
        written: &RoaringTreemap,
        published: &Graph,
        fork: &Graph,
    ) -> Absorbed<BuildCursor> {
        self.refresh(state, name, written, published, fork);
        Absorbed::Yes
    }

    fn indexer_catch_up(
        &mut self,
        state: &BuildState<BuildCursor>,
        name: &(EntityType, IndexKey),
        written: &RoaringTreemap,
        batch_version: &Graph,
        published: &Graph,
    ) -> Absorbed<BuildCursor> {
        self.refresh(state, name, written, batch_version, published);
        Absorbed::Yes
    }

    fn indexer_insert(
        &mut self,
        (entity, key): &(EntityType, IndexKey),
        batch: &EncodedTuples,
        written: &RoaringTreemap,
        from: BuildCursor,
        next: Option<BuildCursor>,
        published: &Graph,
    ) {
        let mut adds = EncodedTuples::default();
        for kind in KeyKind::ALL {
            adds[kind].extend(batch[kind].iter().filter(|&&(_, id)| !written.contains(id)));
        }
        // Add the written entities in the batch's range, with their values in `published`. Those
        // below `from` are already right in this copy. Their places are found before any value is
        // read, so a written entity outside the range costs no attribute read under the host's
        // locks.
        let in_range = placed_in(written, *entity, Some(from), next, published);
        graph_writes::entity_tuples(published, *entity, key, &in_range, self, &mut adds);
        self.add_encoded(&mut adds);
    }

    fn batch_len(batch: &EncodedTuples) -> usize {
        batch.len()
    }

    /// The column's next `budget` nodes or edges from `from`, at their values in `version`,
    /// encoded and sorted ([`graph_writes::scan`]).
    fn indexer_read(
        &self,
        version: &Graph,
        (_, (label, attr)): &(EntityType, IndexKey),
        from: BuildCursor,
        budget: usize,
    ) -> (EncodedTuples, Option<BuildCursor>) {
        graph_writes::scan(version, label, attr, from, budget, self)
    }

    fn indexer_add(
        &mut self,
        batch: &mut EncodedTuples,
    ) {
        self.add_encoded(batch);
    }

    fn memory_usage(&self) -> usize {
        match self {
            Self::Range(idx) => idx.memory_usage(),
        }
    }
}

/// How to answer a query by scanning while its columns are still building: keep the entities the
/// columns will answer the query with once they are built. Built from the selections those columns
/// would look up, the scan keeps exactly what they will answer. It is the only check the scanned
/// rows get: the planner removes a predicate it hands to an index scan.
///
/// `A` names an attribute: as the query names it, or as a scan has resolved it
/// ([`Self::resolve`]), so a scan looks each name up once rather than once per entity.
#[derive(Clone, Debug)]
pub struct ScanFilter<A = Arc<String>> {
    /// Each attribute the query constrains, with what the built column for it selects. One for a
    /// query on one attribute; one per attribute for a conjunction across several.
    conditions: Vec<(A, Selection)>,
}

impl ScanFilter {
    /// The scan filter of a query on one attribute.
    fn one(
        attribute: Arc<String>,
        selection: Selection,
    ) -> Self {
        Self {
            conditions: vec![(attribute, selection)],
        }
    }
}

impl<A> ScanFilter<A> {
    /// The same filter with each attribute resolved by `resolve`, or `None` when one does not
    /// resolve. A scan resolves names to the graph's attribute ids: no entity holds a name the
    /// graph has never seen, so such a filter matches nothing.
    pub fn resolve<B>(
        self,
        mut resolve: impl FnMut(&A) -> Option<B>,
    ) -> Option<ScanFilter<B>> {
        let conditions = self
            .conditions
            .into_iter()
            .map(|(attribute, selection)| Some((resolve(&attribute)?, selection)))
            .collect::<Option<_>>()?;
        Some(ScanFilter { conditions })
    }

    /// Whether the built columns answer the query with an entity whose attributes `value_of`
    /// reads: its value of every attribute is one its column answers with
    /// ([`Selection::matches`]). An entity missing one of the attributes does not match, as no
    /// column holds it.
    #[must_use]
    pub fn matches(
        &self,
        value_of: impl Fn(&A) -> Option<Value>,
    ) -> bool {
        self.conditions.iter().all(|(attribute, selection)| {
            value_of(attribute).is_some_and(|value| selection.matches(&value))
        })
    }
}

/// Where a building column's next indexer batch resumes: the first entity no batch has indexed
/// yet. Also an entity's place in that order, to compare with it ([`Self::precedes`]).
///
/// A marker, not an iterator: it holds a place and nothing of the scan that reached it. Each batch
/// opens a new scan from it, of the label's matrix for nodes or the type's tensor for edges, reads
/// up to its budget, and returns the next marker (`graph_writes::scan`). The order is the same for
/// every label, node id or `(src, dst)`, so the cursor names none: the scan reads only the label's
/// entities, and a refresh keeps only those the label has (`graph_writes::entity_tuples`).
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

    /// Where entity `id` sits in build order, as a cursor: a node by its id, an edge by its
    /// endpoints in `endpoints`, a version's edge endpoints. An edge that version deleted has none
    /// (`None`).
    pub(crate) fn from_id(
        entity: EntityType,
        id: u64,
        endpoints: &EndpointIndex,
    ) -> Option<Self> {
        match entity {
            EntityType::Node => Some(Self::Node(NodeId::from(id))),
            EntityType::Relationship => endpoints.get(id).map(|(src, dst)| Self::Edge {
                src: NodeId::from(src),
                dst: NodeId::from(dst),
            }),
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
    node_columns: HashMap<IndexKey, IndexEntry<IndexColumn>>,
    edge_columns: HashMap<IndexKey, IndexEntry<IndexColumn>>,
    /// Indexes created in this version and those before it: the next one's generation.
    indexes_created: u64,
    /// Indexer batches published in this graph before this version, and this one if it is a
    /// batch's ([`Self::set_batches_published`]). It never falls and rises with every batch
    /// published, so between a version and a later published one with the same count, no batch
    /// was published. Writers may have been: the data version tells.
    batches_published: u64,
    /// CCH shortest-path indexes, one [`IndexEntry`] per definition, keyed by the whole
    /// [`CchIndexDef`](super::cch::CchIndexDef) (the kind's [`VersionedIndex::Name`]). A created
    /// entry is `Building` until the index builder adopts its hierarchy, so readers check the
    /// state.
    ///
    /// A `BTreeMap`, not a `HashMap`: it iterates in definition order, the same in every
    /// process, so whatever lists the indexes (a save, a listing) is deterministic. Found by
    /// key, one entry per definition. Empty until one is created, so a graph with no CCH index
    /// pays one empty map.
    ///
    /// Cloned with the rest of the index by `Graph::new_version`, which clones one `Arc<Cch>`
    /// per index: no hierarchy data is copied, a reader that already holds the old version
    /// keeps it, and rollback costs nothing. `pub(super)` for the CCH modules that find or
    /// update an entry in place (`CchIndex`'s [`VersionedIndex`] lookup); everything else goes
    /// through the CCH methods below.
    pub(super) cch_indexes: BTreeMap<CchIndexDef, IndexEntry<CchIndex>>,
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

    /// Heap bytes every index holds in this version, node and edge columns and CCH
    /// hierarchies ([`VersionedIndex::memory_usage`]): what a build with `index-falkordb` adds
    /// to `indices_sz_mb` in `GRAPH.MEMORY USAGE`. Walks every page of every column.
    ///
    /// A CCH index's map key is its own copy of the definition, so its type list is counted
    /// here, next to the one the entry counts.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let columns: usize = self
            .node_columns
            .values()
            .chain(self.edge_columns.values())
            .map(IndexEntry::memory_usage)
            .sum();
        let hierarchies: usize = self
            .cch_indexes
            .iter()
            .map(|(def, entry)| size_of_val(def.rel_types()) + entry.memory_usage())
            .sum();
        columns + hierarchies
    }

    /// The column map for `entity` — node labels and relationship types keep
    /// separate keyspaces (a label and a type may share a name).
    fn columns(
        &self,
        entity: EntityType,
    ) -> &HashMap<IndexKey, IndexEntry<IndexColumn>> {
        match entity {
            EntityType::Node => &self.node_columns,
            EntityType::Relationship => &self.edge_columns,
        }
    }

    fn columns_mut(
        &mut self,
        entity: EntityType,
    ) -> &mut HashMap<IndexKey, IndexEntry<IndexColumn>> {
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
    ) -> Option<&IndexEntry<IndexColumn>> {
        self.columns(entity).get(&(label.clone(), attr.clone()))
    }

    /// Create an empty column for `(entity, label, attr)` in the `Ready`
    /// state, replacing any existing one. Used by the RDB-load path
    /// ([`create_unpopulated`](Self::create_unpopulated)), whose populate rebuilds it once every
    /// entity is in, before any read is served. An edge column's docs are `edge_id`s.
    pub fn create_column(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) {
        self.insert_column(
            entity,
            label,
            attr,
            IndexColumn::Range(RangeIndex::new()),
            BuildState::Ready,
        );
    }

    /// Create an empty column in the `Building` state, its cursor at the start, replacing
    /// any existing one. Indexer batches fill it from the start, live writes keep the part below
    /// the cursor current, and a read it would serve scans instead ([`ScanFilter`]) until the last
    /// batch flips it `Ready`.
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
            IndexColumn::Range(RangeIndex::new()),
            BuildState::building(BuildCursor::new(entity)),
        );
    }

    /// Put `column` at `(entity, label, attr)` in `state`, replacing any column there.
    fn insert_column(
        &mut self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        column: IndexColumn,
        state: BuildState<BuildCursor>,
    ) {
        self.indexes_created += 1;
        let generation = self.indexes_created;
        self.columns_mut(entity).insert(
            (label.clone(), attr.clone()),
            IndexEntry::new(column, state, generation),
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
                self.create_column(entity, label, attr);
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

    /// The column for `(entity, label, attr)`, if one exists. Used by the populate path and tests,
    /// not the read path (which gates on state).
    #[must_use]
    pub fn column(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<&RangeIndex> {
        match self.columns(entity).get(&(label.clone(), attr.clone())) {
            Some(IndexEntry {
                index: IndexColumn::Range(idx),
                ..
            }) => Some(idx),
            None => None,
        }
    }

    /// The column for `(entity, label, attr)`, to change.
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
            Some(IndexEntry {
                index: IndexColumn::Range(idx),
                ..
            }) => Some(idx),
            None => None,
        }
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

    /// Whether column `(entity, label, attr)` exists and is `Ready`: whether this version's index
    /// serves reads on it. A `Building` column answers with a [`ScanFilter`] instead.
    #[must_use]
    pub fn is_ready(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> bool {
        self.entry(entity, label, attr)
            .is_some_and(|entry| !entry.is_building())
    }

    /// How many entities column `(entity, label, attr)`'s batches have read
    /// ([`BuildState::Building`]), or `None` if the column is `Ready` or does not exist.
    pub(crate) fn building_read(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<u64> {
        match self.entry(entity, label, attr)?.state {
            BuildState::Building { read, .. } => Some(read),
            BuildState::Ready => None,
        }
    }

    /// Ready this version to be published: each index's sets of written entities are settled
    /// ([`IndexEntry::publish`]).
    pub(crate) fn publish(&mut self) {
        self.node_columns
            .values_mut()
            .chain(self.edge_columns.values_mut())
            .for_each(IndexEntry::publish);
        self.cch_indexes.values_mut().for_each(IndexEntry::publish);
    }

    /// Every column, by kind and key.
    pub(crate) fn column_names(&self) -> Vec<(EntityType, IndexKey)> {
        [EntityType::Node, EntityType::Relationship]
            .into_iter()
            .flat_map(|entity| {
                self.columns(entity)
                    .keys()
                    .map(move |key| (entity, key.clone()))
            })
            .collect()
    }

    /// Set this version's [`Self::batches_published`] as it is published after `published`:
    /// `published`'s count, plus one if this is an indexer batch's version. Only the publish
    /// calls this, so nothing that builds a version has to keep the count.
    pub(crate) fn set_batches_published(
        &mut self,
        published: &Self,
        indexer_batch: bool,
    ) {
        self.batches_published = published.batches_published + u64::from(indexer_batch);
    }

    /// Whether an indexer batch was published between `older`, an earlier version, and this
    /// published one.
    pub(crate) fn batch_published_since(
        &self,
        older: &Self,
    ) -> bool {
        self.batches_published != older.batches_published
    }

    /// Call `emit` with each key `value` takes in column `(entity, label, attr)` and its
    /// [`KeyKind`]; nothing when there is no such column or it does not index `value`. The write
    /// path's staging gate.
    pub fn for_each_stored_key(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        value: &Value,
        emit: impl FnMut(KeyKind, u64),
    ) {
        if let Some(entry) = self.entry(entity, label, attr) {
            entry.index.for_each_stored_key(value, emit);
        }
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
    /// ([`BuildState`] says why).
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

    /// One entity kind's part of [`apply`](Self::apply): each column records its changes as its
    /// state allows ([`IndexEntry::writer_record`]). A column dropped since its changes were staged
    /// is skipped.
    fn apply_writes(
        &mut self,
        entity: EntityType,
        writes: impl IntoIterator<Item = (IndexKey, Vec<IndexWrite>)>,
        place: &impl Fn(EntityType, u64) -> Option<BuildCursor>,
    ) {
        let columns = self.columns_mut(entity);
        let mut scratch = <IndexColumn as VersionedIndex>::Scratch::default();
        for (key, log) in writes {
            let Some(entry) = columns.get_mut(&key) else {
                continue;
            };
            entry.writer_record(log, |id| place(entity, id), &mut scratch);
        }
    }

    /// Answer an index query for `(entity, label)`. Yields docs — node ids for a node column,
    /// `edge_id`s for an edge column.
    ///
    /// This layer resolves only *which column(s)* a predicate needs and whether they can be read
    /// yet; what the predicate selects in a column is [`Selection::of`]'s business, since that
    /// depends on the operand's type and therefore on the kinds inside the column.
    ///
    /// Three outcomes, and the caller must keep them apart, because the no-fallback build treats
    /// each differently:
    ///
    /// * `Some(Ok(rows))` — served here, from `Ready` columns. Possibly zero rows: a predicate
    ///   Cypher says cannot match (`n.v = null`, a `NULL` bound) is *answered*, not declined.
    /// * `Some(Err(scan_filter))` — a shape the columns serve, on a column that is still
    ///   `Building`. The caller scans and keeps what the [`ScanFilter`] matches. This is a timing
    ///   state, not a capability gap, so it must never surface as an error.
    /// * `None` — no such column, or no kind in it serves this predicate (a value of a type no
    ///   kind holds, a union mixing ranges in, a union spanning attributes). Under
    ///   `index-falkordb` there is no other index, and the caller turns this into an error.
    #[must_use]
    pub fn query_column(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        query: &IndexQuery<Value>,
    ) -> Option<Result<DocIter, ScanFilter>> {
        let mut keys = Vec::new();
        if !attributes_of(query, &mut keys) || keys.is_empty() {
            return None; // a shape with no attribute to route by (an empty composite)
        }
        let first = keys[0];
        if keys.iter().any(|k| *k != first) {
            // Several attributes: a conjunction intersects one column per attribute. What cannot be
            // done is answer from a single column: looking every value up in one of them would
            // return rows the query never asked for.
            return match query {
                IndexQuery::And(children) => self.intersect_columns(entity, label, children),
                _ => None,
            };
        }
        let entry = self.entry(entity, label, first)?;
        // Translate the query even while the column is building: a shape no kind serves is
        // declined whatever the column's state, and that must stay `None`. Only a shape it would
        // serve is a `ScanFilter`.
        let selection = Selection::of(query)?;
        if entry.is_building() {
            // Not every entity is indexed yet.
            return Some(Err(ScanFilter::one(first.clone(), selection)));
        }
        Some(Ok(entry.index.lookup(selection)))
    }

    /// A conjunction spanning several attributes: intersect one column per attribute.
    ///
    /// Each attribute's own conjuncts are first collapsed by [`Selection::conjunction`], so this
    /// only combines one doc stream per attribute. A conjunct that itself spans attributes
    /// (`a = 1 AND (b = 2 OR c = 3)`) is not regrouped and declines.
    ///
    /// **Set probe, not a sorted merge.** `DocIter` yields docs in *value* order, not id order —
    /// deliberately, since Cypher guarantees no order without `ORDER BY`, and merging would cost
    /// a comparison per doc and force every branch to stay live. Value-ordered streams cannot be
    /// merge-intersected, so the first attribute is materialised into a set and the rest probe it.
    /// Memory is proportional to the first attribute's match count; there is no cardinality
    /// estimate to pick the smallest side with, so the order is the query's.
    ///
    /// Every attribute needs a column. A missing one declines the conjunction rather than
    /// answering from the attributes that have one, which would return rows the conjunction
    /// excludes. One still `Building` makes the whole conjunction a scan (`Err`, a
    /// [`ScanFilter`]): intersecting against a half-built column would silently drop rows, which
    /// is worse than scanning.
    fn intersect_columns(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        children: &[IndexQuery<Value>],
    ) -> Option<Result<DocIter, ScanFilter>> {
        // Regroup the conjuncts per attribute, flattening nested conjunctions, so each column is
        // asked once for its whole share.
        fn group<'a>(
            children: &'a [IndexQuery<Value>],
            out: &mut Vec<(&'a Arc<String>, Vec<&'a IndexQuery<Value>>)>,
        ) -> Option<()> {
            for child in children {
                if let IndexQuery::And(nested) = child {
                    group(nested, out)?;
                    continue;
                }
                let mut keys = Vec::new();
                if !attributes_of(child, &mut keys) {
                    return None;
                }
                let key = *keys.first()?;
                if keys.iter().any(|k| *k != key) {
                    return None; // one conjunct spanning columns is not groupable
                }
                match out.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, qs)) => qs.push(child),
                    None => out.push((key, vec![child])),
                }
            }
            Some(())
        }
        let mut per_attr: Vec<(&Arc<String>, Vec<&IndexQuery<Value>>)> = Vec::new();
        group(children, &mut per_attr)?;

        // Every column is checked, and every attribute's share translated, before any is read: a
        // missing column or an unservable share declines whatever the others' state, and a
        // building column makes it a scan only if every attribute could be answered.
        let mut columns = Vec::with_capacity(per_attr.len());
        for (attr, conjuncts) in per_attr {
            let entry = self.entry(entity, label, attr)?;
            let selection = Selection::conjunction(&conjuncts)?;
            columns.push((attr, entry, selection));
        }
        if columns.iter().any(|(_, entry, _)| entry.is_building()) {
            // The scan standing in has to check every attribute, as the intersection would.
            return Some(Err(ScanFilter {
                conditions: columns
                    .into_iter()
                    .map(|(attr, _, selection)| (attr.clone(), selection))
                    .collect(),
            }));
        }

        let mut result: Option<FxHashSet<u64>> = None;
        for (_, entry, selection) in columns {
            let rows = entry.index.lookup(selection);
            result = Some(match result {
                None => rows.collect(),
                Some(seen) => rows.filter(|doc| seen.contains(doc)).collect(),
            });
            if result.as_ref().is_some_and(FxHashSet::is_empty) {
                break; // a later conjunct can only remove rows, never add them
            }
        }
        result.map(|docs| Ok(DocIter::Set(docs.into_iter())))
    }

    /// "Which entities of `(entity, label)` may hold `value` at `attr`?", or `None` when this
    /// index cannot answer: there is no such column, the column is still `Building` (it has not
    /// seen every entity, so it cannot rule one out), or its kind does not index `value` as a
    /// scalar (a temporal, or a list: its elements are in the array trees, which say which lists
    /// contain a value, not which equal a list).
    ///
    /// `None` is not "no entities". An answer, even an empty one, covers every entity holding
    /// `value`, which is what lets a unique constraint trust it. The question names no index kind,
    /// so a value a new kind indexes is answered by that kind with no change at the caller.
    #[must_use]
    pub fn equal_candidates(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        value: &Value,
    ) -> Option<EqualCandidates<'_>> {
        let entry = self.entry(entity, label, attr)?;
        if entry.is_building() {
            return None;
        }
        let (kind, key) = entry.index.encode_value(value)?;
        Some(EqualCandidates {
            column: &entry.index,
            kind,
            key,
        })
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
            columns
                .iter()
                .find_map(|(key, entry)| Some((entity, key.clone(), entry.cursor()?)))
        })
    }

    /// Whether any column is still `Building`. Walks the columns' states without hashing a name,
    /// so a per-write check that only matters during a build costs next to nothing outside one.
    #[must_use]
    pub(crate) fn has_building_columns(&self) -> bool {
        self.node_columns
            .values()
            .chain(self.edge_columns.values())
            .any(IndexEntry::is_building)
    }

    /// Whether any index, a column or a CCH hierarchy, is still `Building`: what every writer's
    /// commit asks before it looks for builds to start. The CCH map is empty on almost every
    /// graph, so this costs what [`Self::has_building_columns`] does.
    #[must_use]
    pub(crate) fn has_building_indexes(&self) -> bool {
        self.has_building_columns() || self.cch_indexes.values().any(IndexEntry::is_building)
    }

    /// The build state of column `(entity, label, attr)`, if it exists.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn column_state(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<BuildState<BuildCursor>> {
        self.entry(entity, label, attr)
            .map(|entry| entry.state.clone())
    }
}

/// CCH shortest-path indexes, held next to the columns. A new index is held `Building`, and the
/// index builder builds its hierarchy in the background ([`CchIndex`]'s
/// [`VersionedIndex`](super::versioned::VersionedIndex) impl); an RDB load hands one built
/// already. Nothing here reads the graph.
///
/// MVCC: an entry holds its hierarchy as `Arc<Cch>`, and `Graph::new_version` clones the `Arc`.
/// Every write here adds or removes an entry on the version it is called on only,
/// so a reader holding an older version keeps its hierarchy, and rolling a write back is
/// dropping the new version. Every write is all-or-nothing.
impl FalkorDbIndex {
    /// The entry held over `def` in any state, looked up by its definition, the whole key
    /// ([`VersionedIndex::entry`]).
    #[must_use]
    pub(crate) fn cch_entry(
        &self,
        def: &CchIndexDef,
    ) -> Option<&IndexEntry<CchIndex>> {
        CchIndex::entry(self, def)
    }

    /// The index held over exactly `def`, if it is ready to answer: an entry still building, or
    /// whose build failed, is not returned.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn cch_index(
        &self,
        def: &CchIndexDef,
    ) -> Option<&CchIndex> {
        self.cch_entry(def)
            .filter(|entry| !entry.is_building() && entry.index.cch().is_some())
            .map(|entry| &entry.index)
    }

    /// Every held CCH definition, ready or not, in definition order.
    pub(crate) fn cch_defs(&self) -> impl Iterator<Item = &CchIndexDef> {
        self.cch_indexes.keys()
    }

    /// Hold `index`, built and `Ready`, as a new index: the next generation, like a
    /// column's. The caller has checked that no entry over the same definition is held
    /// ([`Self::cch_entry`]), before building it.
    #[cfg(test)]
    pub(crate) fn add_cch(
        &mut self,
        index: CchIndex,
    ) {
        self.hold_cch(index, BuildState::Ready);
    }

    /// Hold a new index over `def`, `Building`, the next generation: the commit of this version
    /// starts its build ([`MvccGraph::commit`](crate::graph::mvcc_graph::MvccGraph::commit)).
    /// The caller has checked that no entry over `def` is held ([`Self::cch_entry`]).
    pub(crate) fn add_cch_building(
        &mut self,
        def: CchIndexDef,
    ) {
        self.hold_cch(CchIndex::pending(def), BuildState::building(()));
    }

    /// [`Self::add_cch`] and [`Self::add_cch_building`].
    fn hold_cch(
        &mut self,
        index: CchIndex,
        state: BuildState<()>,
    ) {
        self.indexes_created += 1;
        let entry = IndexEntry::new(index, state, self.indexes_created);
        let held = self.cch_indexes.insert(entry.index.def.clone(), entry);
        debug_assert!(
            held.is_none(),
            "a CCH index over this definition was already held"
        );
    }

    /// Set the built entry over `def` back to `Building` under a new generation, with no
    /// hierarchy: a write changed the graph in a way it cannot take. The commit of this version
    /// hands it to the index builder, as for a `CREATE`; older versions keep their entry and its
    /// hierarchy. The definition is held, so an RDB saves it as for any entry.
    pub(crate) fn rebuild_cch_in_background(
        &mut self,
        def: &CchIndexDef,
    ) {
        let generation = self.next_generation();
        let entry = self.cch_indexes.get_mut(def).expect("held");
        entry.index.hierarchy = Hierarchy::Pending;
        entry.index.restart_count_reset();
        entry.state = BuildState::building(());
        entry.generation = generation;
    }

    /// A new generation, for an index created or built again on this version: the version's
    /// count of indexes created, counted up.
    pub(crate) fn next_generation(&mut self) -> u64 {
        self.indexes_created += 1;
        self.indexes_created
    }

    /// Stop holding the index over `def`.
    ///
    /// # Errors
    /// [`CchError::NotFound`] if no entry over `def` is held.
    pub(crate) fn drop_cch(
        &mut self,
        def: &CchIndexDef,
    ) -> Result<(), CchError> {
        self.cch_indexes
            .remove(def)
            .map(drop)
            .ok_or(CchError::NotFound)
    }
}

#[cfg(test)]
mod tests {
    use super::super::encode::encode_numeric;
    use super::super::pending_writes::IndexChanges;
    use super::super::versioned::IndexerBatch;
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

    /// Rows of an answer that must be `Some(Ok(..))`. Panics on the other two, rather than
    /// reporting them as "no rows" — a test that expected rows and got a `ScanFilter` or `None`
    /// has found a bug, and collapsing all three into an empty `Vec` is exactly what hid one.
    fn rows(answer: Option<Result<DocIter, ScanFilter>>) -> Vec<u64> {
        match answer {
            Some(Ok(it)) => it.collect(),
            Some(Err(_)) => panic!("column is still Building"),
            None => panic!("predicate is not servable by the native index"),
        }
    }

    /// The build state machine at the column level. A `Building` column gates reads (answers
    /// with a `ScanFilter`, so the caller scans rather than erroring) while indexer batches fill
    /// it. A live write reaches it only below the cursor: one at or above is left to the batch
    /// that reaches the entity. The batch that reaches the end flips the column `Ready`, and from
    /// then on a batch changes nothing.
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
            rows(idx.query_column(Node, &label, &eq(v)))
        };
        // A `ScanFilter`, specifically — not `None`. A `Building` column is a timing state and
        // the caller scans; `None` means unservable, which the no-fallback build turns into a
        // query error. Collapsing the two errored every read against a still-building index.
        let building = |idx: &FalkorDbIndex, v: i64| {
            matches!(idx.query_column(Node, &label, &eq(v)), Some(Err(_)))
        };
        // What the column holds for `v`, whatever its state.
        let holds = |idx: &FalkorDbIndex, v: i64| -> Vec<u64> {
            idx.column(Node, &label, &attr)
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
            idx.column(Node, &label, &attr)
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
        assert!(building(&idx, 10), "empty Building → ScanFilter");

        // Live writes before any batch: node 2 is created with 30, and node 1 goes 20 -> 25.
        // Both are at or above the cursor, so the column takes neither.
        write(&mut idx, &label, &attr, &[(30, 2, true)]);
        write(&mut idx, &label, &attr, &[(20, 1, false), (25, 1, true)]);
        assert!(
            all(&idx).is_empty(),
            "writes at or above the cursor wait for the batch"
        );
        assert!(
            building(&idx, 30),
            "still Building → ScanFilter after live writes"
        );
        assert!(
            matches!(idx.query_column(Node, &label, &in_10_30), Some(Err(_))),
            "an IN list on a Building column is answered with a ScanFilter too"
        );
        // A shape the column declines stays `None` mid-build: that is a gap, not a timing state,
        // and answering with a `ScanFilter` would let it scan quietly until the build finished.
        let in_mixed = IndexQuery::InList {
            key: attr.clone(),
            list: Value::List(Arc::new(
                [Value::Int(10), Value::String(arc("x"))]
                    .into_iter()
                    .collect(),
            )),
        };
        assert!(
            idx.query_column(Node, &label, &in_mixed).is_none(),
            "a mixed IN list is unservable whatever the column's state"
        );

        // A batch indexes nodes 0..=2 at their current values, and stops before node 3.
        let batch = |tuples: Vec<(u64, u64)>, next| {
            let mut tuples = EncodedTuples::scalars(tuples);
            tuples.sort_dedup();
            IndexerBatch::<IndexColumn> {
                name: (Node, column.clone()),
                work: tuples,
                next,
                read: 3,
                catch_up_limit: None,
            }
        };
        let next = BuildCursor::Node(NodeId::from(3));
        assert!(idx.apply_indexer_batch(&mut batch(
            vec![(key(30), 2), (key(10), 0), (key(25), 1)],
            Some(next),
        )));
        assert_eq!(idx.next_building(), Some((Node, column.clone(), next)));
        assert!(building(&idx, 10), "still Building after a partial batch");

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
        assert!(idx.apply_indexer_batch(&mut batch(vec![(key(45), 3), (key(50), 5)], None)));
        assert!(!idx.has_building_columns());
        assert_eq!(
            idx.column_state(Node, &label, &attr),
            Some(BuildState::Ready)
        );
        assert_eq!(hits(&idx, 10), vec![0]);
        assert_eq!(
            rows(idx.query_column(Node, &label, &in_10_30)),
            vec![0, 2],
            "a Ready column serves the IN list"
        );
        assert_eq!(hits(&idx, 26), vec![1]);
        assert!(hits(&idx, 20).is_empty() && hits(&idx, 25).is_empty());
        assert_eq!(hits(&idx, 45), vec![3]);
        assert_eq!(hits(&idx, 50), vec![5]);
        assert_eq!(all(&idx), vec![0, 1, 2, 3, 5]);

        // A batch on a `Ready` column, or on a column that is gone, changes nothing.
        assert!(!idx.apply_indexer_batch(&mut batch(vec![(key(99), 9)], None)));
        assert!(!idx.apply_indexer_batch(&mut IndexerBatch {
            name: (Node, (label.clone(), arc("gone"))),
            ..batch(vec![(key(99), 9)], None)
        }));
        assert!(hits(&idx, 99).is_empty());
        assert_eq!(all(&idx), vec![0, 1, 2, 3, 5]);
    }

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
            v.column(Node, &label, &attr)
                .unwrap()
                .range(None, None, true, true)
                .collect()
        };
        assert_eq!(all(&v1), vec![1]); // committed version unchanged
        assert_eq!(all(&v2), vec![1, 2]); // new version sees the write
    }

    /// Tuples for entities `0..n`, entity `i` holding `value(i)`.
    fn tuples(
        n: u64,
        value: impl Fn(u64) -> i64,
    ) -> EncodedTuples {
        EncodedTuples::scalars(
            (0..n)
                .map(|i| (encode_numeric(&Value::Int(value(i))).unwrap(), i))
                .collect(),
        )
    }

    /// A column built from `tuples`, as an indexer batch builds one.
    fn column_of(mut tuples: EncodedTuples) -> IndexColumn {
        let mut column = IndexColumn::Range(RangeIndex::new());
        column.indexer_add(&mut tuples);
        column
    }

    /// A column reports its pages, at sizes from empty to several levels of branches.
    ///
    /// Distinct ints are far apart once encoded, so every leaf stays in the 16-byte
    /// array-of-structs form: the tuples are a floor. Above each leaf a branch holds one child
    /// and one 16-byte separator, under 64 bytes a leaf between them.
    #[test]
    fn a_column_reports_its_pages() {
        let mut smaller = 0;
        for n in [0, 1, 255, 256, 257, 10_000, 200_000] {
            let size = VersionedIndex::memory_usage(&column_of(tuples(n, |i| i as i64)));
            let floor = usize::try_from(n).unwrap() * 16;
            let leaves = floor.div_ceil(256 * 16);
            assert!(
                (floor..=floor + 64 * leaves).contains(&size),
                "{n} tuples reported {size} bytes, not {floor} plus at most {} for branches",
                64 * leaves
            );
            assert!(
                size > smaller || n == 0,
                "{n} tuples reported no more than fewer did"
            );
            smaller = size;
        }
    }

    /// The compact leaf form counts too: every entity holding one value packs a tuple into a
    /// one-byte index entry plus its doc, well under the 16 bytes the other form takes.
    #[test]
    fn a_compact_column_reports_its_pages() {
        for n in [256, 10_000, 200_000] {
            let size = VersionedIndex::memory_usage(&column_of(tuples(n, |_| 7)));
            let n = usize::try_from(n).unwrap();
            assert!(
                (n..=n * 8).contains(&size),
                "{n} tuples of one value reported {size} bytes"
            );
        }
    }

    /// A column reports the pages of both its trees: list elements' tuples, in the array tree,
    /// count as scalars' do.
    #[test]
    fn a_column_reports_both_trees_pages() {
        let scalars = tuples(10_000, |i| i as i64);
        let mut elements = EncodedTuples::default();
        elements[KeyKind::Array].clone_from(&scalars[KeyKind::Scalar]);
        let mut both = scalars.clone();
        both[KeyKind::Array].clone_from(&scalars[KeyKind::Scalar]);
        let size =
            |tuples: &EncodedTuples| VersionedIndex::memory_usage(&column_of(tuples.clone()));

        assert!(size(&scalars) > 0);
        assert_eq!(size(&elements), size(&scalars));
        assert_eq!(size(&both), 2 * size(&scalars));
    }

    /// Versions that share pages each count them: a fork reports what its parent does, and
    /// writing the fork changes only the fork's number.
    #[test]
    fn each_version_counts_the_pages_it_shares() {
        let (label, attr) = (arc("Person"), arc("age"));
        let adds = |ids: std::ops::Range<u64>| -> Vec<(i64, u64, bool)> {
            ids.map(|id| (id as i64, id, true)).collect()
        };
        let mut v1 = FalkorDbIndex::new();
        v1.create_column(Node, &label, &attr);
        write(&mut v1, &label, &attr, &adds(0..10_000));
        let before = v1.memory_usage();
        assert!(before > 0);

        let mut v2 = v1.clone();
        assert_eq!(v2.memory_usage(), before);
        write(&mut v2, &label, &attr, &adds(10_000..20_000));
        assert_eq!(v1.memory_usage(), before);
        assert!(v2.memory_usage() > before);
    }

    /// The index reports every column, node and edge, each by its own pages.
    #[test]
    fn the_index_counts_node_and_edge_columns() {
        let (label, attr) = (arc("Rated"), arc("score")); // a node label and a relationship type
        let with = |columns: &[(EntityType, u64)]| -> FalkorDbIndex {
            let mut idx = FalkorDbIndex::new();
            for &(entity, n) in columns {
                idx.create_column(entity, &label, &attr);
                idx.column_mut(entity, &label, &attr)
                    .unwrap()
                    .add_encoded(&mut tuples(n, |i| i as i64));
            }
            idx
        };
        let own = |n| VersionedIndex::memory_usage(&column_of(tuples(n, |i| i as i64)));

        assert_eq!(FalkorDbIndex::new().memory_usage(), 0);
        assert_eq!(with(&[(Node, 1_000)]).memory_usage(), own(1_000));
        assert_eq!(with(&[(Relationship, 3_000)]).memory_usage(), own(3_000));
        assert_eq!(
            with(&[(Node, 1_000), (Relationship, 3_000)]).memory_usage(),
            own(1_000) + own(3_000)
        );

        let mut idx = with(&[(Node, 1_000), (Relationship, 3_000)]);
        idx.drop_column(Node, &label, &attr);
        assert_eq!(idx.memory_usage(), own(3_000));
    }

    /// A `Building` index also holds the entities writers changed since its last batch, until a
    /// batch catches up on them. A `Ready` one holds none, and reports its pages alone.
    #[test]
    fn a_building_index_counts_what_writers_left_for_its_next_batch() {
        let (label, attr) = (arc("Person"), arc("age"));
        let name = (Node, (label.clone(), attr.clone()));
        let mut idx = FalkorDbIndex::new();
        idx.create_building(Node, &label, &attr);
        let pages = idx.memory_usage();

        let written: roaring::RoaringTreemap = (0..10_000).step_by(3).collect();
        let BuildState::Building { written: w, .. } =
            &mut IndexColumn::entry_mut(&mut idx, &name).unwrap().state
        else {
            panic!("create_building leaves the column Building");
        };
        w.since_last_indexer_step = Arc::new(written.clone());
        assert!(written.serialized_size() > 0);
        assert_eq!(idx.memory_usage(), pages + written.serialized_size());

        idx.create_column(Node, &label, &attr);
        assert_eq!(idx.memory_usage(), 0);
    }

    /// `create_column` installs a column; before it, the key is absent.
    #[test]
    fn create_then_lookup() {
        let (label, attr) = (arc("Person"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        assert!(idx.is_empty());
        assert!(idx.column(Node, &label, &attr).is_none());

        idx.create_column(Node, &label, &attr);
        assert_eq!(idx.len(), 1);
        assert!(idx.column(Node, &label, &attr).is_some());
        assert!(idx.column(Node, &label, &attr).unwrap().is_empty());
    }

    /// `apply_writes` lands each unit of work's changes in the right column.
    #[test]
    fn writes_apply_into_columns() {
        let (label, attr) = (arc("Person"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        idx.create_column(Node, &label, &attr);

        write(&mut idx, &label, &attr, &[(7, 1, true), (9, 2, true)]);
        write(&mut idx, &label, &attr, &[(7, 1, false)]);

        let ids: Vec<u64> = idx
            .column(Node, &label, &attr)
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
            idx.column(e, &name, &attr)
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
        assert!(idx.column(Relationship, &name, &attr).is_none());
        assert!(idx.column(Node, &name, &attr).is_some());
    }

    /// Reads route only numeric `Equal`/`Range` leaves to the numeric column, for either entity.
    /// A node query on a shared name must not see the edge column, and vice-versa.
    #[test]
    fn query_numeric_routes_only_numeric_leaves_per_entity() {
        let (label, attr) = (arc("Person"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        idx.create_column(Node, &label, &attr);
        idx.column_mut(Node, &label, &attr)
            .unwrap()
            .add(&Value::Int(30), 1);
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

        // A null bound holds for nothing, so it is answered with no rows rather than declined.
        let null_min = IndexQuery::Range {
            key: key.clone(),
            min: Some(Value::Null),
            max: None,
            include_min: false,
            include_max: true,
        };
        assert_eq!(
            rows(idx.query_column(Node, &label, &null_min)),
            Vec::<u64>::new()
        );
        let null_max = IndexQuery::Range {
            key: key.clone(),
            min: Some(Value::Int(10)),
            max: Some(Value::Null),
            include_min: true,
            include_max: false,
        };
        assert_eq!(
            rows(idx.query_column(Node, &label, &null_max)),
            Vec::<u64>::new()
        );

        // A string on the same Range index has no kind to serve it yet.
        let str_eq = IndexQuery::Equal {
            key: key.clone(),
            value: Value::String(Arc::new("x".to_string())),
        };
        assert!(idx.query_column(Node, &label, &str_eq).is_none());
        // A conjunction on ONE attribute is served — it folds to a single window.
        let eq2 = IndexQuery::Equal {
            key: key.clone(),
            value: Value::Int(30),
        };
        assert_eq!(
            rows(idx.query_column(Node, &label, &IndexQuery::And(vec![eq2]))),
            vec![1]
        );
        // A conjunction spanning TWO attributes needs a column for each; `height` has none, so
        // it is declined rather than answered from `age` alone (served case: see
        // `cross_attribute_conjunctions_intersect_columns`).
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

    /// `n.a IN [...]` reaches the facade as one `InList`, and `a IN [..] OR a = ..` as an `Or` of
    /// lists and equalities. Each must be routed to that attribute's column, and anything that
    /// cannot be answered whole must be declined. A facade that declined every one would still
    /// return the right rows, because RediSearch answers instead, so only a test at this level
    /// notices.
    #[test]
    fn query_numeric_serves_in_lists_and_same_attribute_unions() {
        let (label, attr) = (arc("Person"), arc("age"));
        let mut idx = FalkorDbIndex::new();
        idx.create_column(Node, &label, &attr);
        let column = idx.column_mut(Node, &label, &attr).unwrap();
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
            match idx.query_column(entity, &label, q)? {
                Ok(it) => {
                    let mut ids: Vec<u64> = it.collect();
                    ids.sort_unstable();
                    Some(ids)
                }
                Err(_) => panic!("column is still Building"),
            }
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

    /// `equal_candidates` answers only when its answer covers every entity holding the value, and
    /// declines otherwise. A decline must never look like an empty answer: to a unique constraint,
    /// "no entities" means "no duplicate".
    #[test]
    fn equal_candidates_declines_what_it_cannot_rule_out() {
        let (label, attr) = (arc("Person"), arc("age"));
        let candidates = |idx: &FalkorDbIndex, value: &Value| {
            idx.equal_candidates(Node, &label, &attr, value)
                .map(|c| c.rows().collect::<Vec<_>>())
        };
        let mut idx = FalkorDbIndex::new();
        assert!(
            candidates(&idx, &Value::Int(1)).is_none(),
            "no column: declined"
        );

        // A column whose kind keys several values alike answers with all of them; the caller
        // re-checks each against the graph.
        let one = [Value::Int(1), Value::Float(1.0), Value::Bool(true)];
        idx.create_column(Node, &label, &attr);
        let mut staged = IndexChanges::new(Node);
        for (value, id) in one.iter().zip([10, 11, 12]) {
            staged.add(&idx, &label, &attr, value, id);
        }
        let mut writes = PendingIndexWrites::default();
        writes.extend(staged);
        idx.apply(&mut writes, |_, id| {
            Some(BuildCursor::Node(NodeId::from(id)))
        });
        for value in &one {
            assert_eq!(candidates(&idx, value), Some(vec![10, 11, 12]), "{value:?}");
        }
        assert_eq!(
            candidates(&idx, &Value::Int(2)),
            Some(vec![]),
            "an empty answer from a Ready column means no entity holds the value"
        );
        assert!(
            candidates(&idx, &Value::String(arc("1"))).is_none(),
            "a numeric column does not index strings, so it cannot say none holds one"
        );
        assert!(
            candidates(&idx, &Value::Float(f64::NAN)).is_none(),
            "nor NaN"
        );
        assert!(
            idx.equal_candidates(Relationship, &label, &attr, &Value::Int(1))
                .is_none(),
            "no edge column of this name"
        );

        // A building column has not seen every entity, even when it already holds this one: a
        // batch put node 10 in and stopped before node 11.
        idx.create_building(Node, &label, &attr);
        let mut work = EncodedTuples::scalars(vec![(encode_numeric(&Value::Int(1)).unwrap(), 10)]);
        work.sort_dedup();
        assert!(idx.apply_indexer_batch(&mut IndexerBatch::<IndexColumn> {
            name: (Node, (label.clone(), attr.clone())),
            work,
            next: Some(BuildCursor::Node(NodeId::from(11))),
            read: 11,
            catch_up_limit: None,
        }));
        assert_eq!(
            idx.column(Node, &label, &attr)
                .unwrap()
                .point(&Value::Int(1))
                .collect::<Vec<_>>(),
            vec![10]
        );
        assert!(
            candidates(&idx, &Value::Int(1)).is_none(),
            "a Building column cannot rule an entity out"
        );
    }

    /// A conjunction across two attributes is answered by intersecting their columns.
    ///
    /// The streams are in **value** order, not id order, so this cannot be a sorted merge — the
    /// first attribute is materialised into a set and the rest probe it.
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

    /// A column still building makes the whole conjunction a scan. Intersecting against a
    /// half-populated column would silently drop rows, which is worse than scanning.
    #[test]
    fn a_building_column_makes_the_intersection_a_scan() {
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
        assert!(matches!(idx.query_column(Node, &label, &q), Some(Err(_))));
    }
}
