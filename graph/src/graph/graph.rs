//! Core graph data structure and operations.
//!
//! This module contains the main [`Graph`] struct which represents a property graph
//! using sparse matrices for efficient storage and graph operations.
//!
//! ## Graph Model
//!
//! The graph supports:
//! - **Nodes**: Identified by 64-bit IDs, can have multiple labels and properties
//! - **Relationships**: Directed edges with a type, source/destination, and properties
//! - **Properties**: Key-value pairs stored in columnar [`AttributeStore`]s
//! - **Indexes**: Range and full-text indexes on node properties
//!
//! ## Storage Layout
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                         Graph Structure                             │
//! ├──────────────────────────┬───────────────────────────────────────────┤
//! │ adjacancy_matrix         │ Boolean matrix: src x dst -> bool        │
//! │                          │ (union of all relationship types)        │
//! ├──────────────────────────┼───────────────────────────────────────────┤
//! │ labels_matices[i]        │ Diagonal matrix per label:               │
//! │                          │ node_id x node_id -> bool                │
//! ├──────────────────────────┼───────────────────────────────────────────┤
//! │ node_labels_matrix       │ Matrix: node_id x label_id -> bool       │
//! │                          │ (maps each node to all its labels)       │
//! ├──────────────────────────┼───────────────────────────────────────────┤
//! │ relationship_matrices[i] │ Tensor per type: src x dst x edge_id     │
//! │                          │ (supports multiple edges between same    │
//! │                          │  src/dst pair via 3rd dimension)         │
//! ├──────────────────────────┼───────────────────────────────────────────┤
//! │ relationship_type_matrix │ Matrix: edge_id x type_id -> bool        │
//! │                          │ (maps each edge to its type)             │
//! ├──────────────────────────┼───────────────────────────────────────────┤
//! │ node_attrs               │ AttributeStore: node properties          │
//! │ relationship_attrs       │ AttributeStore: edge properties          │
//! ├──────────────────────────┼───────────────────────────────────────────┤
//! │ node_indexer             │ Secondary indexes on node properties     │
//! │ cache                    │ LRU cache for parsed query plans         │
//! └──────────────────────────┴───────────────────────────────────────────┘
//! ```
//!
//! ## ID Allocation and Recycling
//!
//! Deleted node/edge IDs are tracked in `RoaringTreemap` bitmaps and reused
//! before allocating fresh IDs. The `reserve_node` / `reserve_relationship`
//! methods first reclaim from deleted IDs, then extend the ID space. This
//! keeps matrices compact and avoids unbounded ID growth.
//!
//! ## Versioning
//!
//! `Graph::new_version()` creates a shallow copy suitable for a write
//! transaction. Matrices use Copy-on-Write (see [`super::cow::Cow`]) so
//! they are only duplicated when the writer actually mutates them. The
//! `version` counter is incremented on each write transaction.
//!
//! ## Query Plan Caching
//!
//! The graph caches parsed and planned queries in an LRU cache. On cache hit,
//! the plan is returned directly without reparsing. The cache key is the
//! raw query string. Plans are invalidated when the UDF version changes.

use std::{
    collections::HashMap,
    hash::Hash,
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use atomic_refcell::AtomicRefCell;
use lru::LruCache;
use orx_tree::DynTree;
use parking_lot::{Mutex, MutexGuard};
use roaring::RoaringTreemap;

use crate::{
    entity_type::EntityType,
    graph::{
        attribute_store::{AttrNameMap, AttributeStore},
        constraint::{Constraint, ConstraintStatus, ConstraintType},
        endpoint_index::EndpointIndex,
        graphblas::{
            matrix::{Descriptor, Dup, Matrix},
            serialization::{Encode, EncodeState, PayloadEntry, Writer},
            tensor::Tensor,
            versioned_matrix::{self, VersionedMatrix},
        },
    },
    index::{
        Field,
        indexer::{Document, IndexInfo, IndexOptions, IndexQuery, IndexType, Indexer},
    },
    parser::cypher::Parser,
    planner::{IR, Planner, binder::Binder, optimizer::optimize},
    runtime::{orderset::OrderSet, value::Value, vec_distance},
    threadpool::spawn,
};

/// Result of query parsing and planning.
///
/// Contains the execution plan along with metadata about parsing performance.
pub struct Plan {
    /// The execution plan tree
    pub plan: Arc<DynTree<IR>>,
    /// Whether this plan was retrieved from cache
    pub cached: bool,
    /// Query parameters extracted from CYPHER prefix, already evaluated
    pub parameters: HashMap<String, Value>,
    /// Time spent parsing the query
    pub parse_duration: Duration,
    /// Time spent planning/optimizing the query
    pub plan_duration: Duration,
    /// Byte offset in the original query where the actual query (without
    /// CYPHER params prefix) begins. Used by the slowlog to separate the
    /// parameter portion from the query text.
    pub params_offset: usize,
}

/// Opaque identifier for a node label.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LabelId(pub usize);

/// Opaque identifier for a relationship type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeId(pub usize);

/// Opaque identifier for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NodeId(u64);

/// Opaque identifier for a relationship (edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct RelationshipId(u64);

/// Which halves of a node's adjacency to enumerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    Outgoing,
    Incoming,
    Both,
}

impl std::str::FromStr for EdgeDirection {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "outgoing" => Ok(Self::Outgoing),
            "incoming" => Ok(Self::Incoming),
            "both" => Ok(Self::Both),
            _ => Err(()),
        }
    }
}

impl From<LabelId> for usize {
    fn from(val: LabelId) -> Self {
        val.0
    }
}

impl From<TypeId> for usize {
    fn from(val: TypeId) -> Self {
        val.0
    }
}

impl From<u64> for NodeId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<NodeId> for u64 {
    fn from(value: NodeId) -> Self {
        value.0
    }
}

impl From<u64> for RelationshipId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<RelationshipId> for u64 {
    fn from(value: RelationshipId) -> Self {
        value.0
    }
}

impl Plan {
    #[must_use]
    pub const fn new(
        plan: Arc<DynTree<IR>>,
        cached: bool,
        parameters: HashMap<String, Value>,
        parse_duration: Duration,
        plan_duration: Duration,
        params_offset: usize,
    ) -> Self {
        Self {
            plan,
            cached,
            parameters,
            parse_duration,
            plan_duration,
            params_offset,
        }
    }
}

/// Detailed memory usage breakdown returned by [`Graph::memory_usage_report`].
///
/// All sizes are in bytes; the command handler converts to megabytes.
pub struct MemoryUsageReport {
    pub label_matrices_sz: usize,
    pub relation_matrices_sz: usize,
    pub node_block_storage_sz: usize,
    pub node_attr_by_label: Vec<(Arc<String>, usize)>,
    pub unlabeled_node_attr_sz: usize,
    pub edge_block_storage_sz: usize,
    pub edge_attr_by_type: Vec<(Arc<String>, usize)>,
    pub indices_sz: usize,
}

/// The main graph data structure.
///
/// Stores nodes, relationships, labels, and properties using sparse matrices
/// for efficient graph operations. Supports:
/// - Node and relationship creation/deletion
/// - Label and property assignment
/// - Index-based lookups
/// - Query plan caching
///
/// # Thread Safety
///
/// The Graph is `Send + Sync` but not internally synchronized. Use [`MvccGraph`]
/// for concurrent access with proper read/write isolation.
/// A relationship removed by [`Graph::delete_relationships`] or
/// [`Graph::delete_implicit_edges`]: its id, **its type**, and its endpoints.
///
/// The type is captured here because it cannot be recovered afterwards — the
/// edge is gone by the time effects are built — and v3's `DELETE_EDGE` groups by
/// it. `index_remove_edge_docs` is not a substitute: it is only populated for
/// types that actually carry an index.
/// One label a deleted node carried, as [`Graph::delete_nodes`] reports them.
///
/// Named for the same reason as [`DeletedEdge`] and `AttrRef`: two `u64`s side
/// by side, where transposing them attributes every label to the wrong node and
/// nothing about the types would notice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeletedNodeLabel {
    pub node: u64,
    pub label: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeletedEdge {
    /// The edge's own id.
    pub id: RelationshipId,
    /// Its relationship type.
    pub type_id: u64,
    /// Its endpoints, in wire order. Named rather than positional because both
    /// are `NodeId`: a transposition here reverses every deleted edge and
    /// nothing about the types would catch it.
    pub src: NodeId,
    pub dst: NodeId,
}

pub struct Graph {
    /// Graph name (Redis key name)
    name: String,
    /// Maximum node capacity (for matrix sizing)
    node_cap: u64,
    /// Maximum relationship capacity (for matrix sizing)
    relationship_cap: u64,
    /// Number of node IDs reserved (including deleted)
    reserved_node_count: u64,
    /// Number of relationship IDs reserved (including deleted)
    reserved_relationship_count: u64,
    /// Current count of active nodes
    node_count: u64,
    /// Current count of active relationships
    relationship_count: u64,
    /// Bitmap of deleted node IDs (for ID reuse)
    deleted_nodes: RoaringTreemap,
    /// Bitmap of deleted relationship IDs
    deleted_relationships: RoaringTreemap,
    /// Empty matrix for operations
    zero_matrix: VersionedMatrix<bool>,
    /// Combined adjacency matrix (all relationship types)
    adjacancy_matrix: VersionedMatrix<bool>,
    /// Matrix mapping nodes to their labels
    node_labels_matrix: VersionedMatrix<bool>,
    /// Matrix mapping relationships to their types
    relationship_type_matrix: VersionedMatrix<bool>,
    /// Per-label matrices (label ID → node membership)
    labels_matices: Vec<VersionedMatrix<bool>>,
    /// Per-type relationship tensors (type ID → src×dst×edge_id)
    relationship_matrices: Vec<Tensor>,
    /// Graph-wide reverse index: `edge_id` → `(src, dst)` for O(1) endpoint
    /// lookup, as contiguous slots indexed by edge id. Edge IDs are densely
    /// allocated, so slots are far more compact than a hash map (~31 B/edge
    /// with control bytes and load-factor slack). Wrapped in `Arc` so MVCC
    /// `new_version` is O(1); the first edge mutation per version pays one
    /// `Arc::make_mut` deep clone of the tier it writes to — not of the whole
    /// index — and node-only writes pay nothing.
    ///
    /// [`EndpointIndex`] sizes its fields to the endpoints it holds, so this is
    /// 6 bytes per edge on a graph of millions of nodes rather than a flat 8 —
    /// and it has no ceiling, where the previous `(src << 32) | dst` packing
    /// refused any node id past 2^32.
    edge_endpoints: Arc<EndpointIndex>,
    /// The graph's one attribute-name dictionary: name → id and id → name, shared by
    /// both stores below so an id means the same attribute everywhere it appears — in a
    /// span, the RDB, a `GRAPH.EFFECT`, or a compact reply.
    ///
    /// Deliberately *not* per store. Two dictionaries numbered the same name
    /// differently for nodes and relationships, and because effects put a bare id on
    /// the wire, an RDB-seeded replica resolved it against different numbering and wrote
    /// to the wrong attribute (#2457). C has the same shape: one `attributes` array on
    /// `GraphContext`, two `DataBlock`s.
    attrs_name: AttrNameMap,
    /// Node property storage
    node_attrs: AttributeStore,
    /// Relationship property storage
    relationship_attrs: AttributeStore,
    /// Index manager for node property indexes
    node_indexer: Indexer,
    /// Index manager for edge property indexes
    edge_indexer: Indexer,
    /// Label names (ID → name mapping)
    node_labels: Vec<Arc<String>>,
    /// Label name → ID, the reverse of `node_labels`.
    ///
    /// A label test is a per-row operation (`n:Person` for every row of a scan),
    /// so resolving the name must not walk the whole schema: with 500 labels
    /// registered, `MATCH (n) WHERE n:Person RETURN count(n)` over 10k nodes
    /// measured 2.1x the same query on a one-label schema, all of it in the
    /// walk. Maintained only by `intern_label`, so it cannot drift.
    node_labels_index: FxHashMap<String, LabelId>,
    /// Relationship type names (ID → name mapping)
    relationship_types: Vec<Arc<String>>,
    /// LRU cache for query plans
    cache: Arc<Mutex<LruCache<String, Arc<PlanTree>>>>,
    /// Graph constraints (unique, mandatory)
    constraints: Vec<Constraint>,
    /// Version counter (incremented on each write transaction)
    pub version: u64,
    /// Schema version (incremented only on schema changes: new labels, relationship types, or attributes)
    pub schema_version: u64,
}

/// Wrapper for plan trees to implement Send+Sync.
/// Also stores the UDF version at cache time for invalidation.
struct PlanTree {
    plan: DynTree<IR>,
    udf_version: u64,
}

#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for PlanTree {}
unsafe impl Sync for PlanTree {}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

/// Populates an index in the background for existing nodes.
///
/// Each batch borrows the latest committed graph (via the Indexer's shared
/// graph reference) to get a fresh label matrix and attribute store,
/// ensuring nodes added by write transactions between batches are visible.
///
/// The Indexer's serialization lock serializes each batch with write-path
/// `commit_index` calls, so they never run concurrently.  Within a
/// batch the lock is held, preventing writes from committing index
/// changes.  Between batches the lock is released, allowing writes
/// to proceed.
/// Which kind of entity a background population job is building an
/// index for. Controls how ids are enumerated and how attributes are
/// resolved; every other step of the batch loop is shared.
#[derive(Copy, Clone)]
enum IndexKind {
    Node,
    Edge,
}

/// Cursor position between batches.
///
/// - For nodes: `row` is the next node id to visit; the other fields
///   are always `None` (per-row cursor suffices).
/// - For edges: `row` is the next `src` row in the relationship tensor.
///   When `within_row_dst` is `Some(d)`, all edges at `(row, dst)`
///   with `dst < d` have been indexed and must be skipped on resume.
///   When additionally `within_pair_edge_id` is `Some(e)`, the pair
///   at `(row, d)` has multiple edges and the ones with
///   `edge_id <= e` have already been indexed — the remainder of the
///   same-pair group must still be processed. Matches the C
///   implementation's `(prev_src, prev_dst, prev_eid)` skip pattern
///   in `src/index/index_construct.c:_Index_PopulateEdgeIndex`.
#[derive(Clone, Copy, Default)]
struct BatchCursor {
    row: u64,
    within_row_dst: Option<u64>,
    within_pair_edge_id: Option<u64>,
}

fn populate_index(
    kind: IndexKind,
    label: Arc<String>,
    indexer: Indexer,
) {
    // Capture the field snapshot and ticket in one read-side critical
    // section so the documents we build match the generation we own.
    let Some(snapshot) = indexer.acquire_population_snapshot(&label) else {
        return;
    };
    populate_index_batch(
        kind,
        label,
        indexer,
        snapshot.fields,
        0,
        BatchCursor::default(),
        snapshot.ticket,
    );
}

/// Processes one batch of index population and spawns the next batch.
///
/// Node and edge indexes share this loop; the only kind-specific bits
/// are which matrix we walk (label matrix vs relationship tensor) and
/// which attribute store we read.
fn populate_index_batch(
    kind: IndexKind,
    label: Arc<String>,
    indexer: Indexer,
    attrs: HashMap<Arc<String>, Vec<Arc<Field>>>,
    mut progress: u64,
    cursor: BatchCursor,
    ticket: crate::index::indexer::PopulationTicket,
) {
    spawn(
        move || {
            const BATCH_SIZE: usize = 10_000;

            let exhausted;
            let mut next_cursor = cursor;
            let mut do_recurse = false;

            // Hold the Indexer's serialization lock for the entire batch
            // AND through the final ticket-release / recurse decision. Without
            // this, `drop_index_bg` can remove the index entry between
            // batch completion and ticket release. If a subsequent CREATE
            // recreates an entry under the same label, release still targets
            // the original generation via `ticket.generation_id`.
            {
                // Deliberately no global lock: a batch only touches ticket
                // bookkeeping and document add/delete, never the RediSearch spec
                // lifecycle, and taking it per batch would serialize background
                // population against the main thread.
                let lock = indexer.write_lock();
                let guard = lock.lock();

                if !indexer.is_ticket_current(&ticket) {
                    // Index entry was recreated under us — our captured
                    // `attrs` is stale and would commit partial-spec docs
                    // into the new rs_idx. The fresh populate spawned by
                    // recreate will repopulate from cursor 0 with the full
                    // current schema. Release only this generation's
                    // ticket so stale workers never decrement the fresh
                    // generation's pending counter.
                    indexer.release_population_ticket(&ticket);
                    return;
                }

                if indexer.is_cancelled() || indexer.ticket_pending_changes(&ticket) > 1 {
                    indexer.release_population_ticket(&ticket);
                    return;
                }

                let Some(graph) = indexer.get_graph() else {
                    // Graph not yet committed — reschedule this batch.
                    // MvccGraph::commit() will set the indexer's graph
                    // reference, so the next attempt will find it.
                    drop(guard);
                    drop(lock);
                    std::thread::sleep(Duration::from_millis(1));
                    populate_index_batch(kind, label, indexer, attrs, progress, cursor, ticket);
                    return;
                };

                // Collect this batch's entities.
                //
                // For nodes: walk the label matrix from `cursor.row`
                // (keyed by node id).
                // For edges: walk the relationship tensor from
                // `cursor.row` (keyed by `src` row). When
                // `cursor.within_row_dst` is `Some`, skip entries
                // `(src, dst)` where `src == cursor.row` and
                // `dst <= within_row_dst` — those were already
                // processed in a prior batch that ran out of
                // BATCH_SIZE mid-row. Matches C's `(prev_src, prev_dst)`
                // skip in `_Index_PopulateEdgeIndex`.
                let ids: Vec<u64>;
                let edge_triples: Vec<(u64, u64, u64)>;
                let scanned_count: usize;
                {
                    let g = graph.borrow();
                    match kind {
                        IndexKind::Node => {
                            ids = g
                                .get_label_matrix(&label)
                                .map(|lm| {
                                    lm.iter(cursor.row, u64::MAX)
                                        .take(BATCH_SIZE)
                                        .map(|(n, _)| n)
                                        .collect()
                                })
                                .unwrap_or_default();
                            edge_triples = Vec::new();
                            scanned_count = ids.len();
                        }
                        IndexKind::Edge => {
                            let skip_src = cursor.row;
                            let skip_dst = cursor.within_row_dst;
                            let skip_eid = cursor.within_pair_edge_id;
                            let triples: Vec<(u64, u64, u64)> = g
                                .get_relationship_matrix(&label)
                                .map(|t| {
                                    t.iter(cursor.row, u64::MAX, false)
                                        .filter(|(src, dst, eid)| {
                                            // Row mismatch: past the resume
                                            // src entirely, always included.
                                            if *src != skip_src {
                                                return true;
                                            }
                                            // Within the resume row: skip
                                            // completed columns; within the
                                            // resume column (multi-edge),
                                            // skip the already-indexed edge
                                            // ids.
                                            match (skip_dst, skip_eid) {
                                                (Some(d), Some(e)) if *dst == d => *eid > e,
                                                (Some(d), _) => *dst > d,
                                                _ => true,
                                            }
                                        })
                                        .take(BATCH_SIZE)
                                        .collect()
                                })
                                .unwrap_or_default();
                            ids = Vec::new();
                            scanned_count = triples.len();
                            edge_triples = triples;
                        }
                    }
                }

                let mut batch: Vec<Document> = Vec::with_capacity(scanned_count);

                // Build a document for `id`, allocating the RSDoc only once we
                // know the entity has at least one indexed field. Populating an
                // index created before its fields are seeded would otherwise
                // create-then-free a document for every field-less entity (e.g.
                // scanning a 1.7M-edge tensor), pure allocator churn. Returns
                // `None` when no indexed field is present. `make_doc` builds the
                // right key kind (node id vs edge triple) lazily.
                let build_doc =
                    |id: u64, is_edge: bool, g: &Graph, make_doc: &dyn Fn() -> Document| {
                        let mut doc: Option<Document> = None;
                        for (attr, fields) in &attrs {
                            let value = if is_edge {
                                g.get_relationship_attribute(RelationshipId(id), attr)
                            } else {
                                g.get_node_attribute(NodeId(id), attr)
                            };
                            if let Some(value) = value {
                                let doc = doc.get_or_insert_with(make_doc);
                                for field in fields {
                                    doc.set(field, &value);
                                }
                            }
                        }
                        doc
                    };

                // Advance `next_cursor` based on the *last scanned id*,
                // not the last emitted doc, so we don't get stuck when
                // most entities have no indexed attributes (few docs
                // produced) but there are still more to scan.
                match kind {
                    IndexKind::Node => {
                        let last_id = ids.last().copied();
                        for id in ids {
                            let g = graph.borrow();
                            if let Some(doc) = build_doc(id, false, &g, &|| Document::new(id)) {
                                batch.push(doc);
                            }
                        }
                        if let Some(id) = last_id {
                            next_cursor = BatchCursor {
                                row: id + 1,
                                within_row_dst: None,
                                within_pair_edge_id: None,
                            };
                        }
                    }
                    IndexKind::Edge => {
                        let last_pos = edge_triples.last().map(|(s, d, e)| (*s, *d, *e));
                        for (src, dst, eid) in edge_triples {
                            let g = graph.borrow();
                            if let Some(doc) =
                                build_doc(eid, true, &g, &|| Document::new_edge(src, dst, eid))
                            {
                                batch.push(doc);
                            }
                        }
                        if let Some((last_src, last_dst, last_eid)) = last_pos {
                            next_cursor = BatchCursor {
                                row: last_src,
                                within_row_dst: Some(last_dst),
                                within_pair_edge_id: Some(last_eid),
                            };
                        }
                    }
                }

                exhausted = scanned_count < BATCH_SIZE;

                if !batch.is_empty() {
                    progress += batch.len() as u64;
                    let mut add_docs = HashMap::new();
                    add_docs.insert(label.clone(), batch);
                    indexer.commit(&mut add_docs, &mut HashMap::new());
                    indexer.update_progress(&label, progress);
                }

                if exhausted {
                    indexer.release_population_ticket(&ticket);
                } else {
                    do_recurse = true;
                }
                // guard dropped here
            }

            if do_recurse {
                populate_index_batch(kind, label, indexer, attrs, progress, next_cursor, ticket);
            }
        },
        Some(0),
    );
}

fn drop_index_bg(
    label: Arc<String>,
    node_indexer: Indexer,
) {
    spawn(
        move || {
            // Indexer lock only, never the global lock: `Indexer::remove` swaps the
            // index map, and the `Index` it drops takes no lock either. Reaching for
            // the global lock *under* the indexer lock is the AB-BA that hung
            // `test_index_create` for six hours (issue #726).
            //
            // The lock serializes against `populate_index_batch`, which holds it for
            // a whole batch, so the populate worker can't be mid-batch here.
            let lock = node_indexer.write_lock();
            let _guard = lock.lock();
            node_indexer.remove(&label);
        },
        Some(0),
    );
}

/// Default for the NODE_CREATION_BUFFER configuration; the config
/// registration and `CONFIGURATION_NODE_CREATION_BUFFER` in the root crate
/// reference this constant so the default lives in one place.
pub const DEFAULT_NODE_CREATION_BUFFER: u64 = 16384;

/// Effective NODE_CREATION_BUFFER configuration value: the chunk size (in
/// entities) that matrix capacities grow by.
///
/// Set once at module init from the normalized config (power of two, >= 128);
/// immutable afterwards.
pub static NODE_CREATION_BUFFER: AtomicU64 = AtomicU64::new(DEFAULT_NODE_CREATION_BUFFER);

/// Capacity growth step for entity matrices. Sparse matrices carry a
/// row-pointer array sized by the matrix dimension, so doubling the
/// capacity wastes up to `4 * cap` bytes per matrix at large graph sizes.
/// Growing 25% at a time (rounded up to a NODE_CREATION_BUFFER chunk)
/// bounds that slop while keeping resizes rare: each resize triggers
/// GraphBLAS format conversions costing O(entries), so smaller growth
/// steps measurably slow bulk inserts.
fn grow_cap(
    mut cap: u64,
    needed: u64,
) -> u64 {
    let chunk = NODE_CREATION_BUFFER.load(Ordering::Relaxed);
    while needed > cap {
        cap = (cap + (cap / 4).max(chunk)).next_multiple_of(chunk);
    }
    cap
}

/// Append the next `count` reclaimable ids from `pool` to `out`.
///
/// `base` is how many of the pool's ids are already reserved, so this yields
/// exactly what `pool.iter().skip(base).take(count)` would.
///
/// It gets there by rank rather than by walking. `select(base)` finds the
/// base-th id by summing container cardinalities — a container holds 65,536
/// ids, so that is on the order of sixteen steps for a million-id pool — and
/// `Iter::advance_to` then seeks to that value. Expressed as `skip(base)` it
/// walked `base` elements instead, and `CREATE` reserves once per BATCH_SIZE
/// rows with `base` only reset at commit, so a create of N ids walked the pool
/// N/BATCH_SIZE times: O(N^2 / BATCH_SIZE). A 1M-node create over a 1M-id pool
/// spent about 2s of its 2.5s there.
///
/// (`select` is only cheap per *batch*. Per id — the shape this replaced
/// earlier — N calls of O(containers) is its own quadratic.)
fn reclaim_ids<T: From<u64>>(
    pool: &RoaringTreemap,
    base: u64,
    count: u64,
    out: &mut Vec<T>,
) {
    if count == 0 {
        return;
    }
    let Some(start) = pool.select(base) else {
        return;
    };
    let mut iter = pool.iter();
    iter.advance_to(start);
    out.extend(iter.take(count as usize).map(T::from));
}

impl Graph {
    #[must_use]
    pub fn new(
        n: u64,
        e: u64,
        cache_size: usize,
        version: u64,
        name: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            node_cap: n,
            relationship_cap: e,
            reserved_node_count: 0,
            reserved_relationship_count: 0,
            node_count: 0,
            relationship_count: 0,
            deleted_nodes: RoaringTreemap::new(),
            deleted_relationships: RoaringTreemap::new(),
            zero_matrix: VersionedMatrix::<bool>::new(0, 0),
            adjacancy_matrix: VersionedMatrix::<bool>::new(n, n),
            node_labels_matrix: VersionedMatrix::<bool>::new(0, 0),
            relationship_type_matrix: VersionedMatrix::<bool>::new(0, 0),
            labels_matices: Vec::new(),
            relationship_matrices: Vec::new(),
            edge_endpoints: Arc::new(EndpointIndex::default()),
            attrs_name: AttrNameMap::default(),
            node_attrs: AttributeStore::new(),
            relationship_attrs: AttributeStore::new(),
            node_indexer: Indexer::default(),
            edge_indexer: Indexer::default(),
            node_labels: Vec::new(),
            node_labels_index: FxHashMap::default(),
            relationship_types: Vec::new(),
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_size.max(1)).expect("cache_size.max(1) is always >= 1"),
            ))),
            constraints: Vec::new(),
            version,
            schema_version: 0,
        }
    }

    /// Restore a graph from decoded RDB data.
    ///
    /// Used by the RDB load path to construct a fully-populated graph
    /// without going through the mutation API.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn restore(
        name: &str,
        cache_size: usize,
        node_count: u64,
        relationship_count: u64,
        deleted_nodes: RoaringTreemap,
        deleted_relationships: RoaringTreemap,
        adjacancy_matrix: VersionedMatrix<bool>,
        node_labels_matrix: VersionedMatrix<bool>,
        relationship_type_matrix: VersionedMatrix<bool>,
        labels_matices: Vec<VersionedMatrix<bool>>,
        relationship_matrices: Vec<Tensor>,
        node_labels: Vec<Arc<String>>,
        relationship_types: Vec<Arc<String>>,
        attrs_name: AttrNameMap,
        node_attrs: AttributeStore,
        relationship_attrs: AttributeStore,
    ) -> Self {
        // Rebuild the graph-wide reverse index after RDB load to ensure
        // complete sync with the decoded edges.
        // Two scans rather than one scan into a `Vec` of triples: materialising
        // them would add ~24 bytes per edge of peak memory on top of the graph
        // being restored, which is enough to fail a load that would otherwise
        // fit. The first scan takes only maxima, which sizes the index once, so
        // the second neither widens nor reallocates.
        //
        // The first pass also recovers where the index's width tiers begin.
        // Sizing everything to the graph's widest endpoint would be simpler and
        // would flatten the tiering on every load; under incremental growth an
        // edge lands in the widest tier any earlier edge needed, so tier `r`
        // begins at the smallest edge id whose own endpoints need at least `r`.
        // Three scalars, no per-edge state.
        let mut max_edge_id = 0u64;
        let mut starts = [u64::MAX; 3];
        for tensor in &relationship_matrices {
            for (src, dst, edge_id) in tensor.iter_edges() {
                max_edge_id = max_edge_id.max(edge_id);
                let rank = EndpointIndex::rank_for(src.max(dst));
                for r in 1..=rank {
                    let slot = &mut starts[usize::from(r) - 1];
                    *slot = (*slot).min(edge_id);
                }
            }
        }
        let len = if relationship_matrices.iter().any(|t| t.edge_count() > 0) {
            max_edge_id + 1
        } else {
            0
        };
        // A tier nothing reached starts at the end, i.e. stays empty. Clamped
        // monotone so a later tier never begins before an earlier one.
        let mut acc = len;
        for slot in starts.iter_mut().rev() {
            acc = acc.min(*slot);
            *slot = acc;
        }
        let to_slot = |v: u64| usize::try_from(v).expect("edge id exceeds usize");
        let mut edge_endpoints = EndpointIndex::default();
        edge_endpoints.prepare_tiers(starts.map(to_slot), to_slot(len));
        for tensor in &relationship_matrices {
            for (src, dst, edge_id) in tensor.iter_edges() {
                edge_endpoints.set(edge_id, src, dst);
            }
        }

        let chunk = NODE_CREATION_BUFFER.load(Ordering::Relaxed);
        let node_cap = node_count + deleted_nodes.len();
        let relationship_cap = relationship_count + deleted_relationships.len();
        let schema_version = (node_labels.len() + relationship_types.len()) as u64;
        Self {
            name: name.to_string(),
            node_cap: node_cap.next_multiple_of(chunk).max(64),
            relationship_cap: relationship_cap.next_multiple_of(chunk).max(64),
            reserved_node_count: 0,
            reserved_relationship_count: 0,
            node_count,
            relationship_count,
            deleted_nodes,
            deleted_relationships,
            zero_matrix: VersionedMatrix::<bool>::new(0, 0),
            adjacancy_matrix,
            node_labels_matrix,
            relationship_type_matrix,
            labels_matices,
            relationship_matrices,
            edge_endpoints: Arc::new(edge_endpoints),
            attrs_name,
            node_attrs,
            relationship_attrs,
            node_indexer: Indexer::default(),
            edge_indexer: Indexer::default(),
            node_labels_index: node_labels
                .iter()
                .enumerate()
                .map(|(i, l)| (l.as_str().to_string(), LabelId(i)))
                .collect(),
            node_labels,
            relationship_types,
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_size.max(1)).expect("cache_size.max(1) is always >= 1"),
            ))),
            constraints: Vec::new(),
            version: 0,
            schema_version,
        }
    }

    /// Rebuild derived matrices after RDB load.
    ///
    /// - `relationship_type_matrix`: `(edge_id, type_index) = true` for all edges
    /// - Tensor backward (`mt`): transpose of forward (`m`)
    pub fn rebuild_derived_matrices(&mut self) {
        // Resize all node-dimension matrices to match the restored graph capacity.
        // Decoded matrices may have dimensions from the original graph's node_cap,
        // which can differ from the restored node_cap.
        self.resize_node_matrices();
        let rc = self.relationship_cap;
        self.relationship_type_matrix
            .resize(rc, self.relationship_types.len() as u64);

        // Rebuild relationship_type_matrix and tensor backward matrices
        for (type_idx, tensor) in self.relationship_matrices.iter_mut().enumerate() {
            // Rebuild backward (transpose) matrix from forward matrix in one operation
            tensor.rebuild_backward();

            // Iterate edges matrix to rebuild relationship_type_matrix
            for (_, _, edge_id) in tensor.iter(0, u64::MAX, false) {
                self.relationship_type_matrix
                    .set(edge_id, type_idx as u64, true);
            }
        }
    }

    /// Reclaim attribute-arena growth slop on blocks touched by this version.
    /// Called once at MVCC commit.
    pub fn trim_attr_stores(&mut self) {
        self.node_attrs.trim();
        self.relationship_attrs.trim();
    }

    #[must_use]
    pub fn new_version(&self) -> Self {
        debug_assert_eq!(self.reserved_node_count, 0);
        debug_assert_eq!(self.reserved_relationship_count, 0);
        // One dictionary clone per version instead of the two the split tables cost.
        let attrs_name = self.attrs_name.clone();
        let node_attrs = self.node_attrs.new_version();
        let relationship_attrs = self.relationship_attrs.new_version();

        // Tensor::dup() is copy-on-write; edge_endpoints is behind an Arc, so
        // the clone below is O(1) and the deep copy is deferred to the first
        // edge mutation in the new version.
        let relationship_matrices: Vec<Tensor> =
            self.relationship_matrices.iter().map(Tensor::dup).collect();

        Self {
            name: self.name.clone(),
            node_cap: self.node_cap,
            relationship_cap: self.relationship_cap,
            reserved_node_count: 0,
            reserved_relationship_count: 0,
            node_count: self.node_count,
            relationship_count: self.relationship_count,
            deleted_nodes: self.deleted_nodes.clone(),
            deleted_relationships: self.deleted_relationships.clone(),
            zero_matrix: self.zero_matrix.dup(),
            adjacancy_matrix: self.adjacancy_matrix.dup(),
            node_labels_matrix: self.node_labels_matrix.dup(),
            relationship_type_matrix: self.relationship_type_matrix.dup(),
            labels_matices: self
                .labels_matices
                .iter()
                .map(VersionedMatrix::dup)
                .collect(),
            relationship_matrices,
            edge_endpoints: self.edge_endpoints.clone(),
            attrs_name,
            node_attrs,
            relationship_attrs,
            node_indexer: self.node_indexer.clone(),
            edge_indexer: self.edge_indexer.clone(),
            node_labels: self.node_labels.clone(),
            node_labels_index: self.node_labels_index.clone(),
            relationship_types: self.relationship_types.clone(),
            cache: self.cache.clone(),
            constraints: self.constraints.clone(),
            version: self.version + 1,
            schema_version: self.schema_version,
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn node_count(&self) -> u64 {
        self.node_count
    }

    /// Returns the number of nodes with the given label.
    #[must_use]
    pub fn label_node_count(
        &self,
        label: &str,
    ) -> u64 {
        self.get_label_matrix(label).map_or(
            0,
            super::graphblas::versioned_matrix::VersionedMatrix::nvals,
        )
    }

    #[must_use]
    pub const fn relationship_count(&self) -> u64 {
        self.relationship_count
    }

    /// Number of nodes with the given label (by label index).
    #[must_use]
    pub fn label_node_count_by_idx(
        &self,
        label_idx: usize,
    ) -> u64 {
        self.labels_matices[label_idx].nvals()
    }

    /// Number of edges of the given relationship type (by type index).
    #[must_use]
    pub fn type_edge_count(
        &self,
        type_idx: usize,
    ) -> u64 {
        self.relationship_matrices[type_idx].edge_count()
    }

    /// Number of distinct property keys across nodes and relationships.
    ///
    /// Just the dictionary's length: there is one table per graph and it is keyed
    /// by name, so entries are distinct by construction. This used to union two
    /// per-store tables through a `HashSet`; after they were merged the union
    /// became the same table twice.
    #[must_use]
    pub const fn property_key_count(&self) -> usize {
        self.attrs_name.len()
    }

    #[must_use]
    pub const fn node_cap(&self) -> u64 {
        self.node_cap
    }

    #[must_use]
    pub const fn labels_count(&self) -> usize {
        self.node_labels.len()
    }

    #[must_use]
    pub fn get_labels(&self) -> &[Arc<String>] {
        &self.node_labels
    }

    #[must_use]
    pub fn get_label_by_id(
        &self,
        id: LabelId,
    ) -> Arc<String> {
        self.node_labels[id.0].clone()
    }

    #[must_use]
    pub fn get_types(&self) -> &[Arc<String>] {
        &self.relationship_types
    }

    #[must_use]
    pub fn get_type(
        &self,
        id: TypeId,
    ) -> Option<Arc<String>> {
        self.relationship_types.get(id.0).cloned()
    }

    /// The graph's attribute names in id order; position is the id.
    ///
    /// Was a deduplicated union of two per-store tables, walked once per property on the
    /// relationship reply path. With one dictionary it is a plain iteration and the
    /// per-call `HashSet` is gone.
    pub fn get_attrs(&self) -> impl Iterator<Item = &Arc<String>> + '_ {
        self.attrs_name.iter()
    }

    /// Register `label`, or return the id it already has. The only writer of
    /// `node_labels` and `node_labels_index`, so the two stay in step.
    fn intern_label(
        &mut self,
        label: &Arc<String>,
    ) -> LabelId {
        if let Some(&id) = self.node_labels_index.get(label.as_str()) {
            return id;
        }
        let id = LabelId(self.node_labels.len());
        self.node_labels.push(label.clone());
        self.node_labels_index
            .insert(label.as_str().to_string(), id);
        id
    }

    pub fn get_label_id_mut(
        &mut self,
        label: &str,
    ) -> LabelId {
        if let Some(id) = self.get_label_id(label) {
            return id;
        }

        let id = self.intern_label(&Arc::new(label.to_string()));
        self.labels_matices
            .push(VersionedMatrix::<bool>::new(self.node_cap, self.node_cap));
        // Registering a label changes a matrix dimension — `node_labels_matrix`
        // is nodes x labels — so it is reconciled here, where the dimension
        // changed, rather than by whichever write reaches the new column first.
        // Once per label ever, and `resize` is four comparisons when nothing
        // else moved.
        self.resize();
        id
    }

    pub fn get_label_id(
        &self,
        label: &str,
    ) -> Option<LabelId> {
        self.node_labels_index.get(label).copied()
    }

    pub fn get_type_id(
        &self,
        relationship_type: &str,
    ) -> Option<TypeId> {
        self.relationship_types
            .iter()
            .position(|t| t.as_str() == relationship_type)
            .map(TypeId)
    }

    /// Check if a node has a specific label.
    #[must_use]
    pub fn node_has_label(
        &self,
        node_id: NodeId,
        label: &str,
    ) -> bool {
        self.get_label_id(label).is_some_and(|label_id| {
            self.node_labels_matrix
                .get(node_id.0, label_id.0 as u64)
                .is_some()
        })
    }

    /// Check if a node has a specific label by id (no string lookup).
    #[must_use]
    pub fn node_has_label_id(
        &self,
        node_id: NodeId,
        label_id: LabelId,
    ) -> bool {
        self.node_labels_matrix
            .get(node_id.0, label_id.0 as u64)
            .is_some()
    }

    /// Check if an edge has a specific relationship type.
    #[must_use]
    pub fn edge_has_type(
        &self,
        edge_id: RelationshipId,
        type_name: &str,
    ) -> bool {
        self.get_type_id(type_name).is_some_and(|type_id| {
            self.relationship_type_matrix
                .get(edge_id.0, type_id.0 as u64)
                .is_some()
        })
    }

    /// Get-or-create a relationship type by name, returning its `TypeId`.
    pub fn get_type_id_mut(
        &mut self,
        relationship_type: &str,
    ) -> TypeId {
        if let Some(pos) = self
            .relationship_types
            .iter()
            .position(|t| t.as_str() == relationship_type)
            .map(TypeId)
        {
            return pos;
        }

        self.relationship_types
            .push(Arc::new(relationship_type.to_string()));
        self.relationship_matrices.insert(
            self.relationship_types.len() - 1,
            Tensor::new(self.node_cap, self.node_cap),
        );
        TypeId(self.relationship_types.len() - 1)
    }

    pub fn get_plan(
        &self,
        query: &str,
    ) -> Result<Plan, String> {
        let mut parse_duration = Duration::ZERO;
        let mut plan_duration = Duration::ZERO;

        let mut parser = Parser::new(query);
        let (parameters, query_no_params) = parser.parse_parameters()?;
        let params_offset = query.len() - query_no_params.len();
        let query = query_no_params;

        // The optimizer wants the same values the runtime will see.
        let param_values = parameters.clone();

        let current_udf_version = crate::runtime::functions::udf_version();

        {
            let mut cache = self.cache.lock();
            if let Some(plan) = cache.get(query)
                && plan.udf_version == current_udf_version
            {
                let plan = plan.clone();
                drop(cache);
                let optimize_plan = optimize(&plan.plan, self, &param_values);
                return Ok(Plan::new(
                    Arc::new(optimize_plan),
                    true,
                    parameters,
                    parse_duration,
                    plan_duration,
                    params_offset,
                ));
            }
        }

        let start = Instant::now();
        let raw_ir = parser.parse()?;
        let (ir, scope_vars) = Binder::default().bind(raw_ir)?;
        ir.validate()?;
        parse_duration = start.elapsed();

        let mut planner = Planner::new(scope_vars);
        let start = Instant::now();
        let plan = planner.plan(ir);
        let optimize_plan = optimize(&plan, self, &param_values);
        plan_duration = start.elapsed();

        // Only cache the plan if UDF version hasn't changed during planning.
        // A drift means the plan may reference stale UDF bindings.
        if crate::runtime::functions::udf_version() == current_udf_version {
            self.cache.lock().push(
                query.to_string(),
                Arc::new(PlanTree {
                    plan,
                    udf_version: current_udf_version,
                }),
            );
        }
        Ok(Plan::new(
            Arc::new(optimize_plan),
            false,
            parameters,
            parse_duration,
            plan_duration,
            params_offset,
        ))
    }

    #[must_use]
    pub fn get_label_matrix(
        &self,
        label: &str,
    ) -> Option<&VersionedMatrix<bool>> {
        self.get_label_id(label)
            .map(|id| &self.labels_matices[id.0])
    }

    fn get_label_matrix_mut(
        &mut self,
        label: &Arc<String>,
    ) -> &mut VersionedMatrix<bool> {
        let id = self.intern_label(label);
        if id.0 == self.labels_matices.len() {
            let m = VersionedMatrix::<bool>::new(self.node_cap, self.node_cap);
            self.labels_matices.insert(id.0, m);
            self.resize();
        }
        &mut self.labels_matices[id.0]
    }

    fn get_relationship_matrix_mut(
        &mut self,
        relationship_type: &Arc<String>,
    ) -> &mut Tensor {
        if !self.relationship_types.contains(relationship_type) {
            self.relationship_types.push(relationship_type.clone());

            self.relationship_matrices.insert(
                self.relationship_types.len() - 1,
                Tensor::new(self.node_cap, self.node_cap),
            );
            // As for labels above: `relationship_type_matrix` is relationships
            // x types, so registering a type changes one of its dimensions.
            // `create_relationships_bulk` reconciles too, but that is the write
            // catching up rather than the change accounting for itself.
            self.resize();
        }

        self.relationship_types
            .iter()
            .position(|l| l.as_str() == relationship_type.as_str())
            .map(|i| &mut self.relationship_matrices[i])
            .expect("relationship type was just inserted")
    }

    #[must_use]
    pub fn get_relationship_matrix(
        &self,
        relationship_type: &Arc<String>,
    ) -> Option<&Tensor> {
        if !self.relationship_types.contains(relationship_type) {
            return None;
        }

        self.relationship_types
            .iter()
            .position(|l| l.as_str() == relationship_type.as_str())
            .map(|i| &self.relationship_matrices[i])
    }

    #[must_use]
    pub fn get_node_attribute_id(
        &self,
        attr: &Arc<String>,
    ) -> Option<usize> {
        self.attrs_name.get_index_of(attr)
    }

    #[must_use]
    pub const fn node_attribute_count(&self) -> usize {
        self.attrs_name.len()
    }

    #[must_use]
    pub fn get_relationship_attribute_id(
        &self,
        attr: &Arc<String>,
    ) -> Option<usize> {
        self.attrs_name.get_index_of(attr)
    }

    /// The id for `attr`, or `None` if this graph has never seen the name.
    ///
    /// Was an O(N) scan over a union of two per-store tables, per property serialized.
    /// There is one dictionary now, so it is the dictionary's own lookup — and node,
    /// relationship and "global" ids are the same thing.
    #[must_use]
    pub fn get_global_attribute_id(
        &self,
        attr: &Arc<String>,
    ) -> Option<usize> {
        self.attrs_name.get_index_of(attr)
    }

    pub fn return_node_id(
        &mut self,
        id: NodeId,
    ) {
        self.reserved_node_count -= 1;
        self.deleted_nodes.insert(id.into());
    }

    pub fn return_relationship_id(
        &mut self,
        id: RelationshipId,
    ) {
        self.reserved_relationship_count -= 1;
        self.deleted_relationships.insert(id.into());
    }

    pub fn reserve_node(&mut self) -> NodeId {
        if self.reserved_node_count < self.deleted_nodes.len() {
            let id = self.deleted_nodes.select(self.reserved_node_count).unwrap();
            self.reserved_node_count += 1;
            return NodeId(id);
        }
        self.reserved_node_count += 1;
        NodeId(self.node_count + self.reserved_node_count - 1)
    }

    /// Increment the reserved node counter without allocating a specific ID.
    /// Used by effect replay where the actual ID comes from the primary.
    pub const fn inc_reserved_node_count(&mut self) {
        self.reserved_node_count += 1;
    }

    /// Reserve `n` node ids without choosing which.
    ///
    /// The effects apply path takes the ids from the primary, so it needs the
    /// counter moved but not an allocation — and it knows `n` up front, so it
    /// should not have to say so one at a time.
    pub const fn add_reserved_node_count(
        &mut self,
        n: u64,
    ) {
        self.reserved_node_count += n;
    }

    /// Reserve `count` node ids, failing instead of aborting when `count` cannot be
    /// allocated.
    ///
    /// `GRAPH.BULK` reserves from a client-declared count, so the allocation size is
    /// attacker-influenced: `Vec::with_capacity` panicked on the capacity overflow, and the
    /// panic hook (`src/module_init.rs`) exits the process, so it took the server down
    /// (#2426). `try_reserve_exact` turns that into an error the command can report.
    pub fn reserve_nodes(
        &mut self,
        count: usize,
    ) -> Result<Vec<NodeId>, String> {
        let mut ids = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|_| format!("failed to reserve {count} node ids"))?;
        let count = count as u64;
        let deleted_len = self.deleted_nodes.len();
        let available = deleted_len.saturating_sub(self.reserved_node_count);
        let reclaimed = count.min(available);

        // First reclaim from deleted nodes.
        //
        // One rank lookup for the batch, then an ordered walk of it — see
        // [`reclaim_ids`]. Walking to `base` per batch, which is what
        // `iter().skip(base)` did, made a large create quadratic.
        let base = self.reserved_node_count;
        self.reserved_node_count += reclaimed;
        reclaim_ids(&self.deleted_nodes, base, reclaimed, &mut ids);

        // Allocate remaining from the end
        let remaining = count - reclaimed;
        let start = self.node_count + self.reserved_node_count;
        self.reserved_node_count += remaining;
        ids.extend((start..start + remaining).map(NodeId));

        Ok(ids)
    }

    pub fn create_nodes(
        &mut self,
        nodes: &RoaringTreemap,
    ) {
        self.node_count += nodes.len();
        self.reserved_node_count -= nodes.len();
        self.deleted_nodes -= nodes;

        // Ensure capacity covers the highest node ID (effects replay may
        // insert IDs above the current count when applied one-by-one).
        if let Some(max_id) = nodes.max() {
            let needed = max_id + 1;
            if needed > self.node_cap {
                self.node_cap = grow_cap(self.node_cap, needed);
                self.resize_node_matrices();
            }
        }

        self.resize();
    }

    #[must_use]
    pub fn max_node_id(&self) -> u64 {
        if self.node_count == 0 {
            return 0;
        }
        self.node_count + self.deleted_nodes.len() - 1
    }

    #[must_use]
    pub fn max_relationship_id(&self) -> u64 {
        if self.relationship_count == 0 {
            return 0;
        }
        self.relationship_count + self.deleted_relationships.len() - 1
    }

    /// Set node attributes from a record's row-major layout.
    ///
    /// The map form makes the caller allocate a `Vec` per entity and this
    /// function sort them back into the id order the wire already had. See
    /// [`AttributeStore::insert_attrs_rows`].
    /// Set node attributes from a record's row-major layout, for nodes already
    /// known to share one label set.
    ///
    /// The labels come from the caller rather than from this graph, and on the
    /// replication path that is the point: a v3 `UPDATE_NODE` states the label
    /// set the **primary** saw, so indexing under it makes the replica's index
    /// hold what the primary's holds. Re-deriving would make the replica agree
    /// with its own label matrix instead — which is the same thing only while
    /// nothing has diverged, and silently different once something has.
    ///
    /// It is also the cheap form, for the same reason
    /// [`Self::set_relationships_attributes_of_type`] is.
    /// [`Self::set_nodes_attributes`] walks `node_labels_matrix.iter(id, id)`
    /// per node to answer the same question; here the record's grouping has
    /// already established that every id shares the set, so the
    /// indexed-attribute test runs once for the whole record and what remains
    /// per node is an insert.
    ///
    /// There is no label-deriving row-major counterpart any more. Both callers
    /// on the apply path — `CREATE_NODE` and `UPDATE_NODE` — carry a
    /// `LabelSet`, so one existed only to re-derive what the record had already
    /// stated. `set_nodes_attributes` stays for the query path, where the node
    /// is being read anyway and no record has spoken for it.
    ///
    /// `label_ids` must be within this graph's dictionary — `apply` bounds-checks
    /// them through `checked_label_ids` before calling, the way C's
    /// `ApplyUpdateEdge` checks its `r_id`.
    pub fn set_nodes_attributes_rows_of_labels(
        &mut self,
        ids: &[u64],
        label_ids: &[u64],
        attr_ids: &[u16],
        rows: &[Value],
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> Result<(usize, usize), String> {
        let (nremoved, nset) = self.node_attrs.insert_attrs_rows(ids, attr_ids, rows)?;

        if self.node_indexer.has_indices() {
            for &label_id in label_ids {
                let Some(label) = self.node_labels.get(label_id as usize) else {
                    continue;
                };
                // Once per (label, attribute) for the whole record, rather than
                // once per (node, label, attribute).
                let indexed = attr_ids.iter().any(|&attr_id| {
                    self.attrs_name
                        .get(attr_id as usize)
                        .is_some_and(|key| self.node_indexer.has_indexed_attr(label, key))
                });
                if indexed {
                    let docs = index_add_docs.entry(label_id).or_default();
                    docs.extend(ids.iter().copied());
                }
            }
        }
        Ok((nremoved, nset))
    }

    pub fn set_nodes_attributes(
        &mut self,
        attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> Result<(usize, usize), String> {
        let (nremoved, nset) = self.node_attrs.insert_attrs(attrs)?;

        if self.node_indexer.has_indices() {
            for (id, attrs) in attrs {
                for (_, label_id) in self.node_labels_matrix.iter(*id, *id) {
                    let label = &self.node_labels[label_id as usize];
                    for (attr_id, _) in attrs {
                        let Some(key) = self.attrs_name.get(*attr_id as usize) else {
                            continue;
                        };
                        if self.node_indexer.has_indexed_attr(label, key) {
                            index_add_docs.entry(label_id).or_default().insert(*id);
                        }
                    }
                }
            }
        }
        Ok((nremoved, nset))
    }

    /// Import attributes for nodes created in the current transaction.
    ///
    /// `new_labels` maps each created node to the label ids set this
    /// transaction (a created node has no committed labels, so this is its
    /// complete label set). Reading labels from here instead of
    /// `node_labels_matrix` keeps the write path off `iter`, whose `wait`
    /// forces a pending-delta merge per query — O(accumulated delta).
    pub fn import_node_attrs(
        &mut self,
        attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
        new_labels: &FxHashMap<u64, Vec<u64>>,
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> usize {
        let nset = self.node_attrs.import_attrs(attrs);

        if self.node_indexer.has_indices() {
            for (id, attrs) in attrs {
                let Some(label_ids) = new_labels.get(id) else {
                    continue;
                };
                for &label_id in label_ids {
                    let label = &self.node_labels[label_id as usize];
                    for (attr_id, _) in attrs {
                        let Some(key) = self.attrs_name.get(*attr_id as usize) else {
                            continue;
                        };
                        if self.node_indexer.has_indexed_attr(label, key) {
                            index_add_docs.entry(label_id).or_default().insert(*id);
                        }
                    }
                }
            }
        }
        nset
    }

    /// Import pre-resolved node attributes directly into the cache.
    /// Used by bulk insert to avoid per-node OrderMap allocations.
    ///
    /// Marks the imported nodes for (re)indexing, as
    /// [`import_node_attrs`](Self::import_node_attrs) does. `label_ids` is the label set the
    /// caller is applying to every node in `data` — a bulk token carries one label set for
    /// all its rows, so it is passed once instead of looked up per node.
    ///
    /// Keeping the tracking here rather than at the call site matters: the alternative is an
    /// unwritten rule that attributes must be imported before labels are set, and the natural
    /// order (the one [`Pending::commit`] uses) is the opposite, so a future refactor would
    /// silently stop indexing bulk-loaded rows.
    ///
    /// Tracking runs **before** the import because `import_attrs_resolved` drains `data`.
    pub fn import_node_attrs_resolved(
        &mut self,
        data: &mut Vec<(u64, Vec<(u16, Value)>)>,
        label_ids: &[LabelId],
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> usize {
        if self.node_indexer.has_indices() {
            for (id, attrs) in data.iter() {
                for label_id in label_ids {
                    let label = &self.node_labels[label_id.0];
                    for (attr_id, _) in attrs {
                        let Some(key) = self.attrs_name.get(*attr_id as usize) else {
                            continue;
                        };
                        if self.node_indexer.has_indexed_attr(label, key) {
                            index_add_docs
                                .entry(label_id.0 as u64)
                                .or_default()
                                .insert(*id);
                        }
                    }
                }
            }
        }
        self.node_attrs.import_attrs_resolved(data)
    }

    /// Resolve a node attribute name to its index, creating if needed.
    pub fn get_or_create_node_attr_id(
        &mut self,
        attr: &Arc<String>,
    ) -> u16 {
        self.attrs_name.get_or_create(attr)
    }

    /// Resolve a node attribute name to its index, if it exists.
    #[must_use]
    pub fn get_node_attr_id(
        &self,
        attr: &Arc<String>,
    ) -> Option<u16> {
        self.attrs_name.get_index_of(attr).map(|i| i as u16)
    }

    /// Node attribute name for an index.
    #[must_use]
    pub fn node_attr_name(
        &self,
        attr_id: u16,
    ) -> Option<Arc<String>> {
        self.attrs_name.get(attr_id as usize).cloned()
    }

    /// Resolve a relationship attribute name to its index, if it exists.
    #[must_use]
    pub fn get_rel_attr_id(
        &self,
        attr: &Arc<String>,
    ) -> Option<u16> {
        self.attrs_name.get_index_of(attr).map(|i| i as u16)
    }

    /// Relationship attribute name for an index.
    #[must_use]
    pub fn rel_attr_name(
        &self,
        attr_id: u16,
    ) -> Option<Arc<String>> {
        self.attrs_name.get(attr_id as usize).cloned()
    }

    /// Import pre-resolved relationship attributes directly into the cache.
    ///
    /// Marks the imported edges for (re)indexing, exactly as
    /// [`import_relationship_attrs`](Self::import_relationship_attrs) does; the caller
    /// publishes them with [`commit_edge_index`](Self::commit_edge_index). Tracking runs
    /// **before** the import because `import_attrs_resolved` drains `data`.
    pub fn import_relationship_attrs_resolved(
        &mut self,
        data: &mut Vec<(u64, Vec<(u16, Value)>)>,
        type_id: TypeId,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> usize {
        self.track_edge_index_updates_of_type(
            type_id,
            data.iter().map(|(id, attrs)| (id, attrs)),
            index_add_edge_docs,
        );
        self.relationship_attrs.import_attrs_resolved(data)
    }

    /// Resolve a relationship attribute name to its index, creating if needed.
    pub fn get_or_create_rel_attr_id(
        &mut self,
        attr: &Arc<String>,
    ) -> u16 {
        self.attrs_name.get_or_create(attr)
    }

    pub fn import_relationship_attrs(
        &mut self,
        attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> usize {
        let nset = self.relationship_attrs.import_attrs(attrs);
        self.track_edge_index_updates(attrs, index_add_edge_docs);
        nset
    }

    /// Mark every `(type_id, edge_id)` whose changed attributes are
    /// indexed so the next `commit_edge_index` pass rebuilds their
    /// documents. Shared by the import and set paths.
    /// As [`track_edge_index_updates`](Self::track_edge_index_updates), for a caller that
    /// already knows the type every edge in `attrs` carries — a bulk token has exactly one.
    ///
    /// `get_relationship_type_id` iterates `relationship_type_matrix`, and `iter` waits on the
    /// matrix, forcing a pending-delta merge. Doing that once per edge right after
    /// `create_relationships_bulk` enqueued a large delta is the per-entity overhead the bulk
    /// path exists to avoid, so the type is resolved once by the caller instead.
    fn track_edge_index_updates_of_type<'a>(
        &self,
        type_id: TypeId,
        attrs: impl IntoIterator<Item = (&'a u64, &'a Vec<(u16, Value)>)>,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) {
        if !self.edge_indexer.has_indices() {
            return;
        }
        let type_name = &self.relationship_types[type_id.0];
        for (id, attrs) in attrs {
            for (attr_id, _) in attrs {
                let Some(key) = self.attrs_name.get(*attr_id as usize) else {
                    continue;
                };
                if self.edge_indexer.has_indexed_attr(type_name, key) {
                    index_add_edge_docs
                        .entry(type_id.0 as u64)
                        .or_default()
                        .insert(*id);
                }
            }
        }
    }

    fn track_edge_index_updates<'a>(
        &self,
        attrs: impl IntoIterator<Item = (&'a u64, &'a Vec<(u16, Value)>)>,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) {
        if !self.edge_indexer.has_indices() {
            return;
        }
        for (id, attrs) in attrs {
            let type_id = self.get_relationship_type_id(RelationshipId(*id));
            let type_name = &self.relationship_types[type_id.0];
            for (attr_id, _) in attrs {
                let Some(key) = self.attrs_name.get(*attr_id as usize) else {
                    continue;
                };
                if self.edge_indexer.has_indexed_attr(type_name, key) {
                    index_add_edge_docs
                        .entry(type_id.0 as u64)
                        .or_default()
                        .insert(*id);
                }
            }
        }
    }

    /// Bulk set node labels using parallel row/col slices (2 FFI calls per matrix).
    ///
    /// `all_new` asserts every `(node, label)` pair is fresh — the node was
    /// created in this transaction — allowing the unchecked delta insert.
    /// `SET n:Label` on pre-existing nodes may re-add a committed pair, which
    /// must go through the checked path to keep `dp ∩ m = ∅`.
    /// Give every id in `ids` every label in `label_ids`.
    ///
    /// The pair form takes the cartesian product already expanded, which means
    /// the caller allocates `ids x labels` twice over and this function then
    /// regroups it back into the per-label lists it wanted in the first place.
    /// A million-node create with one label spends about 24 MB of scratch
    /// saying something the two input slices already said.
    ///
    /// Here the product stays an iterator: `set_all` consumes one, and each
    /// label's diagonal is just `ids`.
    pub fn set_node_labels_product(
        &mut self,
        ids: &[u64],
        label_ids: &[u64],
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
        all_new: bool,
    ) {
        // Which labels are indexed, decided once per label rather than once per
        // pair. Collected before the mutations below so the immutable borrows
        // of `node_labels` and `node_indexer` end first.
        let indexed: Vec<u64> = label_ids
            .iter()
            .copied()
            .filter(|&lid| {
                self.node_labels
                    .get(lid as usize)
                    .is_some_and(|label| self.node_indexer.has_index(label))
            })
            .collect();
        for lid in indexed {
            let label = self.node_labels[lid as usize].clone();
            for &id in ids {
                // The node has to hold an attribute this label *indexes*, not
                // merely some attribute.
                //
                // The point is symmetry with the remove side rather than a
                // demonstrated leak. `delete_nodes` has always tested exactly
                // this, so with the looser test here a node whose attributes
                // are none of them indexed was added to the index and never
                // matched by a removal. Whether RediSearch materializes a
                // document for a node with no indexed field is not observable
                // from outside the module — it is linked in, so `FT.INFO` is
                // not reachable — so this makes the two predicates the same
                // rather than fixing something measured.
                if self
                    .attr_names(&self.node_attrs, id)
                    .any(|attr| self.node_indexer.has_indexed_attr(&label, &attr))
                {
                    index_add_docs.entry(lid).or_default().insert(id);
                }
            }
        }

        for &lid in label_ids {
            let diag = ids.iter().map(|&id| (id, id));
            if all_new {
                self.labels_matices[lid as usize].set_all::<true>(diag);
            } else {
                self.labels_matices[lid as usize].set_all::<false>(diag);
            }
        }

        // The cartesian product, handed to GraphBLAS as two index lists rather
        // than enumerated here. `GrB_Matrix_assign` forms `ids x label_ids`
        // itself, so a million nodes taking one label is one call instead of a
        // million `setElement`s — and the slices are already exactly the `I`
        // and `J` it wants, so nothing is built to pass them.
        //
        // `set_product::<false>` cannot take that path — an existing node's
        // pair may already be live in the committed base, which only a
        // per-entry probe can tell — so it degrades to the iterator form.
        if all_new {
            self.node_labels_matrix.set_product::<true>(ids, label_ids);
        } else {
            self.node_labels_matrix.set_product::<false>(ids, label_ids);
        }
    }

    pub fn set_nodes_labels_bulk(
        &mut self,
        label_rows: &[u64],
        label_cols: &[u64],
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
        all_new: bool,
    ) {
        self.resize();

        // Collect entries grouped by label for per-label matrices
        let num_labels = self.labels_matices.len();
        let mut by_label: Vec<Vec<u64>> = vec![Vec::new(); num_labels];

        for (&id, &label_id) in label_rows.iter().zip(label_cols.iter()) {
            by_label[label_id as usize].push(id);

            let label = &self.node_labels[label_id as usize];
            if self.node_indexer.has_index(label) && self.node_attrs.has_attributes(id) {
                index_add_docs.entry(label_id).or_default().insert(id);
            }
        }

        let pairs = label_rows.iter().copied().zip(label_cols.iter().copied());
        if all_new {
            self.node_labels_matrix.set_all::<true>(pairs);
        } else {
            self.node_labels_matrix.set_all::<false>(pairs);
        }

        for (lid, ids) in by_label.into_iter().enumerate() {
            if !ids.is_empty() {
                let diag = ids.iter().map(|&id| (id, id));
                if all_new {
                    self.labels_matices[lid].set_all::<true>(diag);
                } else {
                    self.labels_matices[lid].set_all::<false>(diag);
                }
            }
        }
    }

    pub fn remove_nodes_labels(
        &mut self,
        label_rows: &[u64],
        label_cols: &[u64],
        remove_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) {
        self.resize();

        for (&id, &label_id) in label_rows.iter().zip(label_cols.iter()) {
            self.node_labels_matrix.remove(id, label_id);
            self.labels_matices[label_id as usize].remove(id, id);
            let label = &self.node_labels[label_id as usize];
            if self.node_indexer.has_index(label) {
                remove_docs.entry(label_id).or_default().insert(id);
            }
        }
    }

    /// Delete `deleted_nodes`, returning the `(node_id, label_id)` pairs they
    /// carried.
    ///
    /// The pairs are a by-product, not extra work: clearing the label matrices
    /// already has to collect them. Effects replication needs them because the
    /// label set is what tells a replica which label-scoped indexes to clear,
    /// and by the time the effect is encoded — after `commit` — the labels are
    /// gone from the graph. Same reason `delete_relationships` hands back its
    /// endpoints. They arrive grouped by node in ascending node order; a node
    /// with no labels contributes none, so a caller reconciling against
    /// `deleted_nodes` must expect gaps.
    pub fn delete_nodes(
        &mut self,
        deleted_nodes: &RoaringTreemap,
        remove_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> Result<Vec<DeletedNodeLabel>, String> {
        self.deleted_nodes |= deleted_nodes;
        self.node_count -= deleted_nodes.len();

        // Every removal below is a per-entity tombstone, and every lookup below
        // is a row seek. Nothing here touches an entry that does not belong to a
        // deleted node, so the cost is O(|deleted|) rather than O(graph).
        //
        // It used to build diagonal mask matrices and hand them to `remove_mask`,
        // whose `element_wise_multiply` takes `m` as an operand: GraphBLAS then
        // walks the base matrix's vectors, so a delete of any size cost O(graph).
        // Deleting one node from a 200,000-node graph spent ~240M instructions,
        // against ~564k in the C implementation, which is flat in graph size.
        // `VersionedMatrix::remove` marks the same tombstone one entry at a time
        // (`dm[i,j] = true` when `m` holds the pair, else drop it from `dp`),
        // which is what `remove_nodes_labels` already does for label removal.
        //
        // The two are equivalent because `dp ∩ m = ∅` holds for
        // `VersionedMatrix<bool>`: its `set` clears the `dm` tombstone when `m`
        // already holds the pair instead of writing a shadowing `dp` entry. So
        // `remove_mask`'s two effects — tombstone `mask ∩ m`, drop `mask ∩ dp` —
        // can never both apply to one entry, which is exactly the choice `remove`
        // makes per entry. (The valued matrices *do* allow `dp` to shadow `m`;
        // `remove` is defined only on the boolean ones.)
        //
        // That invariant is covered by
        // `delta_invariants_hold_across_mutation_sequences`, and this specific
        // substitution is machine-checked as
        // `eff_removeMask_eq_foldl_remove` in `proofs/versioned_matrix`.
        // Which labels each deleted node carries. Collected first because the
        // iterator borrows `node_labels_matrix` for the duration and the removals
        // below need it mutably.
        //
        // `seek` is a row jump (`GxB_rowIterator_seekRow`), so one iterator
        // re-seeked per deleted row replaces both the previous full scan and the
        // per-node iterator construction that the full scan had replaced. There
        // is no size threshold: this is O(|deleted|) for every delete, so there
        // is no shape of input the old whole-matrix scan would win.
        let mut pairs: Vec<DeletedNodeLabel> = Vec::with_capacity(deleted_nodes.len() as usize);
        {
            let mut it = self.node_labels_matrix.iter(0, self.node_cap);
            for node_id in deleted_nodes {
                it.seek(node_id, node_id);
                pairs.extend(
                    it.by_ref()
                        .map(|(node, label)| DeletedNodeLabel { node, label }),
                );
            }
        }

        for &DeletedNodeLabel {
            node: node_id,
            label: label_id,
        } in &pairs
        {
            let lid = label_id as usize;

            let label = &self.node_labels[lid];
            if self.node_indexer.has_index(label) {
                for attr in self.attr_names(&self.node_attrs, node_id) {
                    if self.node_indexer.has_indexed_attr(label, &attr) {
                        remove_docs.entry(label_id).or_default().insert(node_id);
                        break;
                    }
                }
            }

            self.labels_matices[lid].remove(node_id, node_id);
            self.node_labels_matrix.remove(node_id, label_id);
        }

        self.node_attrs.remove_all(deleted_nodes);
        Ok(pairs)
    }

    pub fn get_node_relationships(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = (NodeId, NodeId, RelationshipId)> + '_ {
        self.relationship_matrices
            .iter()
            .flat_map(move |m| m.iter(id.0, id.0, false).chain(m.iter(id.0, id.0, true)))
            .map(|(src, dest, id)| {
                let src_node = NodeId(src);
                let dest_node = NodeId(dest);
                (src_node, dest_node, RelationshipId(id))
            })
    }

    /// Returns an iterator over all relationship tensors (one per type).
    pub fn relationship_matrices_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.relationship_matrices.iter()
    }

    /// Get all relationships for a node, optionally filtered by relationship types.
    /// When `types` is empty, returns relationships of all types (equivalent to `[*]`).
    /// Tuples are always `(src, dst, edge_id)` in original edge orientation,
    /// regardless of `direction`.
    pub fn get_node_relationships_by_type(
        &self,
        id: NodeId,
        types: &[Arc<String>],
        direction: EdgeDirection,
    ) -> impl Iterator<Item = (NodeId, NodeId, RelationshipId)> + '_ {
        let matrices: Vec<&Tensor> = if types.is_empty() {
            self.relationship_matrices.iter().collect()
        } else {
            types
                .iter()
                .filter_map(|t| self.get_relationship_matrix(t))
                .collect()
        };
        let (with_outgoing, with_incoming) = match direction {
            EdgeDirection::Outgoing => (true, false),
            EdgeDirection::Incoming => (false, true),
            EdgeDirection::Both => (true, true),
        };
        matrices
            .into_iter()
            .flat_map(move |m| {
                let outgoing = with_outgoing
                    .then(|| m.iter(id.0, id.0, false))
                    .into_iter()
                    .flatten();
                // A self-loop appears in both halves; when the outgoing half is
                // also enumerated, drop it from the incoming half.
                let incoming = with_incoming
                    .then(|| {
                        m.iter(id.0, id.0, true)
                            .filter(move |(src, _, _)| !with_outgoing || *src != id.0)
                    })
                    .into_iter()
                    .flatten();
                outgoing.chain(incoming)
            })
            .map(|(src, dest, id)| (NodeId(src), NodeId(dest), RelationshipId(id)))
    }

    /// Count the number of incoming edges to a node.
    #[must_use]
    pub fn get_node_indegree(
        &self,
        id: NodeId,
    ) -> usize {
        self.relationship_matrices
            .iter()
            .map(|m| m.col_degree(id.0) as usize)
            .sum()
    }

    /// Count the number of incoming edges to a node, filtered by relationship types.
    #[must_use]
    pub fn get_node_indegree_by_type(
        &self,
        id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        types
            .iter()
            .filter_map(|t| self.get_relationship_matrix(t))
            .map(|m| m.col_degree(id.0) as usize)
            .sum()
    }

    /// Count the number of outgoing edges from a node.
    #[must_use]
    pub fn get_node_outdegree(
        &self,
        id: NodeId,
    ) -> usize {
        self.relationship_matrices
            .iter()
            .map(|m| m.row_degree(id.0) as usize)
            .sum()
    }

    /// Count the number of outgoing edges from a node, filtered by relationship types.
    #[must_use]
    pub fn get_node_outdegree_by_type(
        &self,
        id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        types
            .iter()
            .filter_map(|t| self.get_relationship_matrix(t))
            .map(|m| m.row_degree(id.0) as usize)
            .sum()
    }

    #[must_use]
    pub fn get_nodes(
        &self,
        labels: &OrderSet<Arc<String>>,
        min_row: u64,
    ) -> Box<dyn Iterator<Item = NodeId>> {
        if labels.is_empty() {
            // Full scan: live node IDs are exactly `0..=max_node_id` minus the
            // deleted set. A range walk with a roaring-bitmap membership check
            // avoids the per-element GraphBLAS row-iterator overhead, which
            // dominates the cost of unfiltered `MATCH (n)` scans.
            //
            // It is also why no matrix of live nodes is kept: the bitmap
            // already answers the question, and maintaining a parallel
            // GraphBLAS diagonal made every write pay structural upkeep
            // proportional to the graph rather than to the write.
            if self.node_count == 0 {
                return Box::new(std::iter::empty());
            }
            let max_id = self.max_node_id();
            if self.deleted_nodes.is_empty() {
                return Box::new((min_row..=max_id).map(NodeId));
            }
            let deleted = self.deleted_nodes.clone();
            return Box::new(
                (min_row..=max_id)
                    .filter_map(move |id| (!deleted.contains(id)).then_some(NodeId(id))),
            );
        }
        if labels.len() == 1 {
            if let Some(label_matrix) = self.get_label_matrix(&labels[0]) {
                return Box::new(
                    label_matrix
                        .iter(min_row, u64::MAX)
                        .map(|(id, _)| NodeId(id)),
                );
            }
            return Box::new(std::iter::empty());
        }
        let matrices = labels
            .iter()
            .map(|label| self.get_label_matrix(label))
            .collect::<Option<Vec<_>>>();
        Box::new(
            matrices
                .map_or_else(
                    || self.zero_matrix.extract().iter(min_row, u64::MAX),
                    |mut matrices| {
                        let mut iter = matrices.iter_mut();
                        let mut m = iter.next().unwrap().extract();
                        for label_matrix in iter {
                            m.element_wise_multiply(
                                None,
                                None,
                                Some(&label_matrix.extract()),
                                None,
                            );
                        }
                        m.iter(min_row, u64::MAX)
                    },
                )
                .map(|(id, _)| NodeId(id)),
        )
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn get_node_label_ids(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = LabelId> {
        self.node_labels_matrix
            .iter(id.0, id.0)
            .map(|(_, l)| LabelId(l as usize))
    }

    pub fn get_node_labels(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = Arc<String>> {
        self.get_node_label_ids(id)
            .map(move |label_id| self.node_labels[label_id.0].clone())
    }

    #[must_use]
    pub fn get_node_attribute(
        &self,
        id: NodeId,
        attr: &Arc<String>,
    ) -> Option<Value> {
        self.attr_by_name(&self.node_attrs, id.0, attr)
    }

    /// Fetches a node attribute using a pre-resolved attribute index.
    /// Use `get_node_attribute_id` to resolve the index once, then call
    /// this method for each node to avoid repeated string lookups.
    #[must_use]
    pub fn get_node_attribute_by_idx(
        &self,
        id: NodeId,
        attr_idx: u16,
    ) -> Option<Value> {
        self.node_attrs.get_attr_by_idx(id.0, attr_idx)
    }

    /// Batch variant of `get_node_attribute_by_idx`.
    /// Pushes one `Value` per id into `out`, substituting `default` for
    /// missing entries (so callers don't allocate a temp `Vec<Option<_>>`).
    pub fn get_node_attributes_by_idx(
        &self,
        ids: &[NodeId],
        attr_idx: u16,
        default: &Value,
        out: &mut Vec<Value>,
    ) {
        // SAFETY: NodeId is `#[repr(transparent)]` over u64.
        let keys: &[u64] = unsafe { std::slice::from_raw_parts(ids.as_ptr().cast(), ids.len()) };
        self.node_attrs
            .get_attrs_by_idx_batch_into(keys, attr_idx, default, out);
    }

    pub fn reserve_relationship(&mut self) -> RelationshipId {
        if self.reserved_relationship_count < self.deleted_relationships.len() {
            let id = self
                .deleted_relationships
                .select(self.reserved_relationship_count)
                .unwrap();
            self.reserved_relationship_count += 1;
            return RelationshipId(id);
        }
        self.reserved_relationship_count += 1;
        RelationshipId(self.relationship_count + self.reserved_relationship_count - 1)
    }

    /// Increment the reserved relationship counter without allocating a specific ID.
    /// Used by effect replay where the actual ID comes from the primary.
    pub const fn inc_reserved_relationship_count(&mut self) {
        self.reserved_relationship_count += 1;
    }

    /// Reserve `n` relationship ids without choosing which. See
    /// [`Self::add_reserved_node_count`].
    pub const fn add_reserved_relationship_count(
        &mut self,
        n: u64,
    ) {
        self.reserved_relationship_count += n;
    }

    /// Reserve `count` relationship ids. Fallible for the same reason as
    /// [`Self::reserve_nodes`]: `GRAPH.BULK` sizes this from a client-declared count.
    pub fn reserve_relationships(
        &mut self,
        count: usize,
    ) -> Result<Vec<RelationshipId>, String> {
        let mut ids = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|_| format!("failed to reserve {count} relationship ids"))?;
        let count = count as u64;
        let deleted_len = self.deleted_relationships.len();
        let available = deleted_len.saturating_sub(self.reserved_relationship_count);
        let reclaimed = count.min(available);

        // First reclaim from deleted relationships — same shape as
        // `reserve_nodes`, see [`reclaim_ids`].
        let base = self.reserved_relationship_count;
        self.reserved_relationship_count += reclaimed;
        reclaim_ids(&self.deleted_relationships, base, reclaimed, &mut ids);

        // Allocate remaining from the end
        let remaining = count - reclaimed;
        let start = self.relationship_count + self.reserved_relationship_count;
        self.reserved_relationship_count += remaining;
        ids.extend((start..start + remaining).map(RelationshipId));

        Ok(ids)
    }

    /// Create relationships of a single type using flat arrays.
    /// Avoids HashMap overhead while using individual GraphBLAS set calls.
    pub fn create_relationships_bulk(
        &mut self,
        type_name: &Arc<String>,
        srcs: &[u64],
        dsts: &[u64],
        rel_ids: &[u64],
    ) {
        let count = srcs.len() as u64;
        self.relationship_count += count;
        self.reserved_relationship_count -= count;

        for &id in rel_ids {
            if self.deleted_relationships.is_empty() {
                break;
            }
            self.deleted_relationships.remove(id);
        }

        if let Some(&max_id) = rel_ids.iter().max() {
            let needed = max_id + 1;
            if needed > self.relationship_cap {
                self.relationship_cap = grow_cap(self.relationship_cap, needed);
                self.resize_relationship_matrices();
            }
        }

        self.get_relationship_matrix_mut(type_name);
        let type_idx = self
            .relationship_types
            .iter()
            .position(|t| t.as_str() == type_name.as_str())
            .unwrap();

        self.resize();

        self.relationship_matrices[type_idx].set_all_from_slices(srcs, dsts, rel_ids);

        // Maintain the graph-wide reverse index alongside the tensor edges.
        // Reserve exactly: MVCC clones reset capacity to `len`, so amortized
        // doubling here would only leave ~2x slack behind, never save reallocs.
        let endpoints = Arc::make_mut(&mut self.edge_endpoints);
        // Size for the whole batch first: growing per edge is a reallocation
        // per edge, since the index reserves exactly rather than doubling.
        if let (Some(&max_id), Some(max_endpoint)) = (
            rel_ids.iter().max(),
            srcs.iter().chain(dsts.iter()).max().copied(),
        ) {
            endpoints.prepare(max_id, max_endpoint);
        }
        for ((&src, &dst), &id) in srcs.iter().zip(dsts.iter()).zip(rel_ids.iter()) {
            endpoints.set(id, src, dst);
        }

        self.adjacancy_matrix
            .set_all::<false>(srcs.iter().copied().zip(dsts.iter().copied()));

        let type_id = type_idx as u64;
        let type_ids: Vec<u64> = vec![type_id; rel_ids.len()];
        self.relationship_type_matrix
            .set_all::<true>(rel_ids.iter().copied().zip(type_ids.iter().copied()));
    }

    /// Fold oversized delta-plus into the base for all shared matrices at
    /// the end of a GRAPH.BULK command, preventing dp accumulation across
    /// commands. `fold_latched` latches the fold decision (via `wait`) and
    /// executes it immediately — mutations are done here, so the mid-tx
    /// fold pathology cannot occur, and deferring to the next version's
    /// `dup`/`flush` would leave the final command's deltas unfolded.
    pub fn flush_for_bulk(&mut self) {
        self.node_labels_matrix.fold_latched();
        for m in &mut self.labels_matices {
            m.fold_latched();
        }
        self.adjacancy_matrix.fold_latched();
        self.relationship_type_matrix.fold_latched();
        for t in &mut self.relationship_matrices {
            t.fold_latched();
        }
    }

    /// Prepare a just-committed version for publication: fold every delta that
    /// has grown comparable to its base into that base, then materialize the
    /// committed base (`m`) layer of every matrix and tensor.
    ///
    /// Folding bounds a committed version's delta memory; see
    /// [`VersionedMatrix::fold_oversized`] for why the sub-hatch deltas are
    /// deliberately left for the next `flush`.
    ///
    /// Waiting the bases is what makes publication safe: readers reach bases
    /// lock-free (dp/dm reads go through the mutex-guarded `Matrix::wait`), so
    /// a pending base in a visible snapshot lets concurrent readers corrupt
    /// GrB state. No-op per matrix when already synced, and it has to run
    /// after the fold, which leaves the base pending.
    pub fn fold_oversized_deltas(&mut self) {
        self.zero_matrix.wait_base();
        self.adjacancy_matrix.fold_oversized();
        self.adjacancy_matrix.wait_base();
        self.node_labels_matrix.fold_oversized();
        self.node_labels_matrix.wait_base();
        self.relationship_type_matrix.fold_oversized();
        self.relationship_type_matrix.wait_base();
        for m in &mut self.labels_matices {
            m.fold_oversized();
            m.wait_base();
        }
        for t in &mut self.relationship_matrices {
            t.fold_oversized();
            t.wait_base();
        }
    }

    /// Materialize all pending GraphBLAS operations on every matrix.
    /// Called from pthread_atfork prepare handler to ensure no internal
    /// GraphBLAS locks are held at fork time.
    pub fn wait_all(&self) {
        self.zero_matrix.wait_all();
        self.adjacancy_matrix.wait_all();
        self.node_labels_matrix.wait_all();
        self.relationship_type_matrix.wait_all();
        for m in &self.labels_matices {
            m.wait_all();
        }
        for t in &self.relationship_matrices {
            t.wait_all();
        }
    }

    /// Returns true if every matrix is fully synced — i.e. `wait_all`
    /// has been called and no writer has since enqueued pending ops.
    /// Used by the post-fork child handler to verify the parent's
    /// pre_fork sync took effect before emitting an RDB.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        self.zero_matrix.is_synced()
            && self.adjacancy_matrix.is_synced()
            && self.node_labels_matrix.is_synced()
            && self.relationship_type_matrix.is_synced()
            && self.labels_matices.iter().all(VersionedMatrix::is_synced)
            && self.relationship_matrices.iter().all(Tensor::is_synced)
    }

    pub fn set_relationships_attributes(
        &mut self,
        attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> Result<(usize, usize), String> {
        let (nremoved, nset) = self.relationship_attrs.insert_attrs(attrs)?;
        self.track_edge_index_updates(attrs, index_add_edge_docs);
        Ok((nremoved, nset))
    }

    /// As [`Self::set_relationships_attributes`], for edges already known to
    /// share one type.
    ///
    /// The difference is entirely in the index bookkeeping: the general form
    /// calls `get_relationship_type_id` per edge, and that is a
    /// `relationship_type_matrix.iter(id, id)` each time — the per-entity delta
    /// iteration `import_node_attrs` documents as O(accumulated delta). Every
    /// caller that already knows the type should state it.
    ///
    /// The v3 `UPDATE_EDGE` record does, since #2570: its group key is the
    /// relationship type, so one record is one type by construction.
    pub fn set_relationships_attributes_of_type(
        &mut self,
        type_id: TypeId,
        attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> Result<(usize, usize), String> {
        let (nremoved, nset) = self.relationship_attrs.insert_attrs(attrs)?;
        self.track_edge_index_updates_of_type(type_id, attrs, index_add_edge_docs);
        Ok((nremoved, nset))
    }

    #[must_use]
    pub fn is_node_deleted(
        &self,
        id: NodeId,
    ) -> bool {
        self.deleted_nodes.contains(id.0)
    }

    #[must_use]
    pub fn deleted_nodes_count(&self) -> u64 {
        self.deleted_nodes.len()
    }

    /// The first node id that has never been handed out.
    ///
    /// Ids below this are either live or sitting in the recycle bin; ids at or
    /// above it are untouched. Derived, not stored — `reserve_node` computes the
    /// next id the same way — which is exactly why a replica whose `node_count`
    /// or bin has drifted from the primary's will start allocating ids the
    /// primary would not, the moment it is promoted.
    ///
    /// **Not `max_node_id() + 1`**, though the arithmetic agrees whenever the
    /// graph holds a live node. `max_node_id` returns a 0 *sentinel* for an
    /// empty graph, which is indistinguishable from a graph whose highest id is
    /// 0 — so a caller asking "has this id been handed out?" reads id 0 as used
    /// on a graph that has never allocated anything. Substituting it made
    /// `recreating_a_recycled_id_is_allowed` reject the very first
    /// `CREATE_NODE` of id 0 with `NodeAlreadyLive`.
    #[must_use]
    pub fn node_id_high_water(&self) -> u64 {
        self.node_count + self.deleted_nodes.len()
    }

    /// The lowest id in `nodes` this graph cannot create — one it has already
    /// handed out and not freed — or `None` when every one of them is
    /// available.
    ///
    /// Available means in the recycle bin *or* never allocated, and the caller
    /// gets the answer rather than the bin. Handing out the set of
    /// not-yet-recycled ids let a caller assemble that judgement itself, and
    /// there is only one right way to assemble it.
    ///
    /// `high_water` is the caller's, not this graph's current one:
    /// `apply_effects` freezes it at buffer entry because records within a
    /// buffer are not in id order — a create of 500..600 may precede 0..500 —
    /// so mid-buffer `node_count` is a count rather than a bound.
    ///
    /// Two roaring operations for the whole set rather than a probe per id.
    #[must_use]
    pub fn first_uncreatable_node(
        &self,
        nodes: &RoaringTreemap,
        high_water: u64,
    ) -> Option<u64> {
        (nodes - &self.deleted_nodes)
            .min()
            .filter(|&id| id < high_water)
    }

    /// The lowest id in `nodes` that is already in the recycle bin, or `None`
    /// when none of them is.
    ///
    /// The half of "is this node live" that only the bin can answer; the other
    /// half is the high-water comparison, which is the caller's because the
    /// mark is.
    #[must_use]
    pub fn first_recycled_node(
        &self,
        nodes: &RoaringTreemap,
    ) -> Option<u64> {
        (nodes & &self.deleted_nodes).min()
    }

    #[must_use]
    pub const fn deleted_nodes(&self) -> &RoaringTreemap {
        &self.deleted_nodes
    }

    #[must_use]
    pub fn deleted_relationships_count(&self) -> u64 {
        self.deleted_relationships.len()
    }

    #[must_use]
    pub const fn deleted_relationships(&self) -> &RoaringTreemap {
        &self.deleted_relationships
    }

    #[must_use]
    pub fn label_matrices(&self) -> &[VersionedMatrix<bool>] {
        &self.labels_matices
    }

    #[must_use]
    pub const fn adjacency_matrix(&self) -> &VersionedMatrix<bool> {
        &self.adjacancy_matrix
    }

    #[must_use]
    pub fn relationship_tensors(&self) -> &[Tensor] {
        &self.relationship_matrices
    }

    #[must_use]
    pub fn is_relationship_deleted(
        &self,
        id: RelationshipId,
    ) -> bool {
        self.deleted_relationships.contains(id.0)
    }

    pub fn delete_relationships(
        &mut self,
        rels: &RoaringTreemap,
        index_remove_edge_docs: &mut FxHashMap<u64, FxHashMap<u64, (u64, u64)>>,
    ) -> Result<Vec<DeletedEdge>, String> {
        if rels.is_empty() {
            return Ok(Vec::new());
        }
        let num_types = self.relationship_matrices.len();

        // --- Phase 1: resolve (type, src, dst) per edge without mutating state ---
        // Endpoints come from the graph-wide reverse index (O(1)); the type from
        // a single re-seekable iterator over `relationship_type_matrix`. Walking
        // `rels` (O(deleted)) with per-edge seeks avoids scanning every edge in
        // the [min, max] id range. Only edges that resolve to both endpoints and
        // a type are recorded in `resolved`; stale/non-existent ids are skipped
        // so they can't corrupt counters or bitmaps in phase 2.
        let mut by_type: Vec<Vec<(u64, u64, u64)>> = vec![Vec::new(); num_types];
        let mut resolved = RoaringTreemap::new();
        {
            let min_id = rels.min().expect("rels is non-empty");
            let mut type_iter = self.relationship_type_matrix.iter(min_id, min_id);
            #[allow(clippy::cast_possible_truncation)]
            for edge_id in rels {
                let Some((src, dst)) = self.endpoints_for_edge(edge_id) else {
                    continue;
                };
                type_iter.seek(edge_id, edge_id);
                if let Some((_, type_idx)) = type_iter.next() {
                    by_type[type_idx as usize].push((edge_id, src, dst));
                    resolved.insert(edge_id);
                }
            }
        }

        // --- Phase 2: mutate state for the actually-resolved edges only ---
        self.deleted_relationships |= &resolved;
        self.relationship_count -= resolved.len();
        self.relationship_attrs.remove_all(&resolved);

        let mut endpoints: Vec<DeletedEdge> = Vec::with_capacity(resolved.len() as usize);
        // (edge_id, type_id) pairs for a single bulk removal from the type matrix.
        let mut tm_rows: Vec<u64> = Vec::with_capacity(endpoints.capacity());
        let mut tm_cols: Vec<u64> = Vec::with_capacity(endpoints.capacity());
        // (src, dst) pairs emptied from a tensor — candidates for adjacency removal.
        let mut adj_candidates: Vec<(u64, u64)> = Vec::new();

        for (type_idx, type_rels) in by_type.iter().enumerate() {
            if type_rels.is_empty() {
                continue;
            }
            let type_id = type_idx as u64;

            // Stage index document removals for indexed relationship types.
            if self
                .edge_indexer
                .has_index(&self.relationship_types[type_idx])
            {
                let docs = index_remove_edge_docs.entry(type_id).or_default();
                for &(edge_id, src, dst) in type_rels {
                    docs.insert(edge_id, (src, dst));
                }
            }

            for &(edge_id, src, dst) in type_rels {
                tm_rows.push(edge_id);
                tm_cols.push(type_id);
                endpoints.push(DeletedEdge {
                    id: RelationshipId(edge_id),
                    type_id,
                    src: NodeId(src),
                    dst: NodeId(dst),
                });
                self.clear_edge_endpoint(edge_id);
            }

            adj_candidates.extend(self.relationship_matrices[type_idx].remove_all(type_rels));
        }

        // Remove every deleted edge from relationship_type_matrix in one masked op.
        if !tm_rows.is_empty() {
            let mut type_mask =
                Matrix::<bool>::new(self.relationship_cap, self.relationship_types.len() as u64);
            type_mask.build(&tm_rows, &tm_cols);
            self.relationship_type_matrix.remove_mask(&type_mask);
        }

        // Update adjacancy_matrix for pairs that lost all edges.
        if !adj_candidates.is_empty() {
            if num_types > 1 {
                // Multiple types — keep only pairs empty across all tensors.
                adj_candidates.retain(|&(src, dst)| {
                    !self
                        .relationship_matrices
                        .iter()
                        .any(|tensor| tensor.get(src, dst).next().is_some())
                });
            }
            if !adj_candidates.is_empty() {
                let node_cap = self.node_cap;
                let adj_rows: Vec<u64> = adj_candidates.iter().map(|&(src, _)| src).collect();
                let adj_cols: Vec<u64> = adj_candidates.iter().map(|&(_, dst)| dst).collect();
                let mut adj_mask = Matrix::<bool>::new(node_cap, node_cap);
                adj_mask.build(&adj_rows, &adj_cols);
                self.adjacancy_matrix.remove_mask(&adj_mask);
            }
        }

        Ok(endpoints)
    }

    ///
    /// Instead of discovering edges per-node during the delete operator, this
    /// method iterates each tensor once for all deleted nodes and batch-removes
    /// the edges. Edges already in `explicit_rels` are skipped (they're handled
    /// by `delete_relationships`).
    ///
    /// The adjacency matrix is NOT updated for node pairs where both endpoints
    /// are deleted — those entries are unreachable since the nodes themselves
    /// are gone.
    /// Returns the list of implicitly deleted edges as `(edge_id, src, dst)`
    /// so the caller can record them for effects/replication.
    pub fn delete_implicit_edges(
        &mut self,
        deleted_nodes: &RoaringTreemap,
        explicit_rels: &RoaringTreemap,
        index_remove_edge_docs: &mut FxHashMap<u64, FxHashMap<u64, (u64, u64)>>,
    ) -> Result<Vec<DeletedEdge>, String> {
        if self.relationship_matrices.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_implicit: Vec<DeletedEdge> = Vec::new();
        // Pairs where only one endpoint is deleted — need adjacency check
        let mut check_adj_pairs: std::collections::HashSet<(u64, u64)> =
            std::collections::HashSet::default();

        for type_idx in 0..self.relationship_matrices.len() {
            let mut rels: Vec<(u64, u64, u64)> = Vec::new();

            // Collect all edges for deleted nodes from this tensor.
            //
            // One iterator pair for the whole type, re-seeked per node, rather
            // than a fresh pair per node: building one allocates a
            // `GxB_Iterator` per layer and waits the tensor, so per-node
            // construction cost scales with the deleted set and the number of
            // relationship types while having nothing to do with how many edges
            // those nodes actually have. Deleting a million edgeless nodes over
            // six types built twelve million iterators to yield nothing.
            {
                let tensor = &self.relationship_matrices[type_idx];
                let mut outgoing = tensor.iter(0, 0, false);
                let mut incoming = tensor.iter(0, 0, true);
                for node_id in deleted_nodes {
                    outgoing.seek(node_id, node_id);
                    for (src, dst, edge_id) in &mut outgoing {
                        if !explicit_rels.contains(edge_id) {
                            rels.push((edge_id, src, dst));
                        }
                    }
                    // Incoming edges — skip if source is also a deleted node
                    // (those edges are already collected from the source's
                    // outgoing iteration), and skip self-loops already found above.
                    incoming.seek(node_id, node_id);
                    for (src, dst, edge_id) in &mut incoming {
                        if src != node_id
                            && !deleted_nodes.contains(src)
                            && !explicit_rels.contains(edge_id)
                        {
                            rels.push((edge_id, src, dst));
                        }
                    }
                }
            }

            if rels.is_empty() {
                continue;
            }

            // Batch remove from relationship_type_matrix using bulk mask
            let type_id = type_idx as u64;
            let tm_rows: Vec<u64> = rels.iter().map(|&(id, _, _)| id).collect();
            let tm_cols: Vec<u64> = vec![type_id; rels.len()];
            let mut type_mask =
                Matrix::<bool>::new(self.relationship_cap, self.relationship_types.len() as u64);
            type_mask.build(&tm_rows, &tm_cols);

            let del_keys: RoaringTreemap = rels.iter().map(|&(id, _, _)| id).collect();
            self.deleted_relationships |= &del_keys;
            let type_name = &self.relationship_types[type_idx];
            let is_indexed = self.edge_indexer.has_index(type_name);
            for &(edge_id, src, dst) in &rels {
                all_implicit.push(DeletedEdge {
                    id: RelationshipId(edge_id),
                    type_id,
                    src: NodeId(src),
                    dst: NodeId(dst),
                });

                // Stage an edge-index document removal so cascade
                // deletes don't leave stale index hits on the next
                // query. Matches `delete_relationships`' handling.
                if is_indexed {
                    index_remove_edge_docs
                        .entry(type_id)
                        .or_default()
                        .insert(edge_id, (src, dst));
                }
            }
            self.relationship_type_matrix.remove_mask(&type_mask);
            self.relationship_attrs.remove_all(&del_keys);

            // Drop deleted edges from the graph-wide reverse index.
            for &(edge_id, _, _) in &rels {
                self.clear_edge_endpoint(edge_id);
            }

            // Batch-remove from tensor — remove_all uses bulk mask operations
            let emptied = self.relationship_matrices[type_idx].remove_all(&rels);
            for (src, dst) in emptied {
                // Only check adjacency if the other endpoint is NOT deleted
                if !deleted_nodes.contains(src) || !deleted_nodes.contains(dst) {
                    check_adj_pairs.insert((src, dst));
                }
            }
        }

        self.relationship_count -= all_implicit.len() as u64;

        // Update adjacency_matrix only for pairs where one endpoint survives
        let mut adj_mask = Matrix::<bool>::new(self.node_cap, self.node_cap);
        for (src, dst) in check_adj_pairs {
            let has_edges = self
                .relationship_matrices
                .iter()
                .any(|tensor| tensor.get(src, dst).next().is_some());
            if !has_edges {
                adj_mask.set(src, dst, true);
            }
        }
        if adj_mask.nvals() > 0 {
            self.adjacancy_matrix.remove_mask(&adj_mask);
        }

        Ok(all_implicit)
    }

    pub fn get_src_dest_relationships(
        &self,
        src: NodeId,
        dest: NodeId,
        types: &[Arc<String>],
    ) -> impl Iterator<Item = RelationshipId> + use<> {
        let iters: Vec<_> = if types.is_empty() {
            &self.relationship_types
        } else {
            types
        }
        .iter()
        .filter_map(|relationship_type| self.get_relationship_matrix(relationship_type))
        .map(|relationship_matrix| relationship_matrix.get(src.0, dest.0))
        .collect();

        iters.into_iter().flat_map(|iter| iter.map(RelationshipId))
    }

    /// Build a relationship matrix summing only the given types (no
    /// source/destination label restriction). Returns `None` when `types` is
    /// non-empty but none of the types exist in the schema (caller should
    /// short-circuit to an empty result).
    #[must_use]
    pub fn build_relationship_matrix_unrestricted(
        &self,
        types: &[Arc<String>],
    ) -> Option<Matrix<bool>> {
        let matrices = types
            .iter()
            .filter_map(|relationship_type| self.get_relationship_matrix(relationship_type))
            .collect::<Vec<_>>();
        if !types.is_empty() && matrices.is_empty() {
            return None;
        }
        let mut iter = matrices.into_iter();
        let mut m = iter.next().map_or_else(
            || self.adjacancy_matrix.extract(),
            super::graphblas::tensor::Tensor::extract,
        );
        for relationship_matrix in iter {
            // The raw fwd layers may hold pending work; set_pattern's GrB ops
            // would finish it internally, racing other readers on the shared
            // handles. No-op when already synced.
            relationship_matrix.wait_fwd();
            m.set_pattern(
                Some(relationship_matrix.fwd_dm()),
                relationship_matrix.fwd_m(),
                Some(Descriptor::C),
            );
            m.set_pattern(None, relationship_matrix.fwd_dp(), None);
        }
        Some(m)
    }

    /// Resolve a set of label names to ids. Returns `None` if any label is not
    /// in the schema (which means no node could match).
    #[must_use]
    pub fn resolve_label_ids(
        &self,
        labels: &OrderSet<Arc<String>>,
    ) -> Option<Vec<LabelId>> {
        labels.iter().map(|l| self.get_label_id(l)).collect()
    }

    /// Build a relationship matrix combining the given types and filtering by
    /// source/destination labels.
    #[must_use]
    pub fn build_relationship_matrix(
        &self,
        types: &[Arc<String>],
        src_labels: &OrderSet<Arc<String>>,
        dest_labels: &OrderSet<Arc<String>>,
    ) -> Matrix<bool> {
        let matrices = types
            .iter()
            .filter_map(|relationship_type| self.get_relationship_matrix(relationship_type))
            .collect::<Vec<_>>();
        let src_labels_matrices = src_labels
            .iter()
            .map(|label| self.get_label_matrix(label))
            .collect::<Option<Vec<_>>>();
        let dest_labels_matrices = dest_labels
            .iter()
            .map(|label| self.get_label_matrix(label))
            .collect::<Option<Vec<_>>>();
        let no_match = (!types.is_empty() && matrices.is_empty())
            || src_labels_matrices.is_none()
            || dest_labels_matrices.is_none();

        let src_labels_matrices = src_labels_matrices.unwrap_or_default();
        let dest_labels_matrices = dest_labels_matrices.unwrap_or_default();

        if no_match {
            self.zero_matrix.extract()
        } else {
            let mut iter = matrices.into_iter();
            let mut m = iter.next().map_or_else(
                || self.adjacancy_matrix.extract(),
                super::graphblas::tensor::Tensor::extract,
            );
            for relationship_matrix in iter {
                m.element_wise_add(None, None, Some(&relationship_matrix.extract()), None);
            }

            if !src_labels_matrices.is_empty() {
                let mut iter = src_labels_matrices.iter();
                let mut src_matrix = iter.next().unwrap().extract();
                for label_matrix in iter {
                    src_matrix.element_wise_multiply(
                        None,
                        None,
                        Some(&label_matrix.extract()),
                        None,
                    );
                }
                m.rmxm(&src_matrix);
            }
            if !dest_labels_matrices.is_empty() {
                let mut iter = dest_labels_matrices.iter();
                let mut dest_matrix = iter.next().unwrap().extract();
                for label_matrix in iter {
                    dest_matrix.element_wise_multiply(
                        None,
                        None,
                        Some(&label_matrix.extract()),
                        None,
                    );
                }
                m.lmxm(&dest_matrix);
            }
            m
        }
    }

    /// Convenience wrapper: build the matrix and iterate it in one call.
    pub fn get_relationships(
        &self,
        types: &[Arc<String>],
        src_labels: &OrderSet<Arc<String>>,
        dest_labels: &OrderSet<Arc<String>>,
        from_id: Option<NodeId>,
        to_id: Option<NodeId>,
    ) -> impl Iterator<Item = (NodeId, NodeId)> + use<> {
        let m = self.build_relationship_matrix(types, src_labels, dest_labels);
        let (min_row, max_row) = from_id.map_or((0, u64::MAX), |id| (id.0, id.0));
        m.iter(min_row, max_row)
            .filter(move |(_, dest)| to_id.is_none() || to_id.unwrap().0 == *dest)
            .map(|(src, dest)| (NodeId(src), NodeId(dest)))
    }

    #[must_use]
    pub fn get_relationship_type_id(
        &self,
        id: RelationshipId,
    ) -> TypeId {
        self.type_id_for_edge(id)
            .expect("relationship must have a type in type_matrix")
    }

    /// The edge's type, or `None` if it has no row in the type matrix.
    ///
    /// The fallible counterpart of [`Self::get_relationship_type_id`], in the
    /// same relation as [`Self::endpoints_for_edge`] to
    /// `get_relationship_endpoints`. An edge deleted earlier in this same
    /// transaction is exactly this case: `commit` clears the matrices before
    /// effects are built, so anything reading a type on the emit path has to
    /// tolerate the edge already being gone rather than panic on it.
    #[must_use]
    pub fn type_id_for_edge(
        &self,
        id: RelationshipId,
    ) -> Option<TypeId> {
        #[allow(clippy::cast_possible_truncation)]
        self.relationship_type_matrix
            .iter(id.0, id.0)
            .map(|(_, l)| TypeId(l as usize))
            .next()
    }

    /// Decode the (src, dst) endpoints for an edge from the graph-wide reverse
    /// index. `None` if the edge does not exist — the fallible counterpart of
    /// [`Self::get_relationship_endpoints`], which panics.
    #[must_use]
    pub fn endpoints_for_edge(
        &self,
        edge_id: u64,
    ) -> Option<(u64, u64)> {
        self.edge_endpoints.get(edge_id)
    }

    /// Clear an edge's endpoint slot in the reverse index (on deletion).
    /// Leaves the slot as a tombstone; the vector never shrinks.
    fn clear_edge_endpoint(
        &mut self,
        edge_id: u64,
    ) {
        Arc::make_mut(&mut self.edge_endpoints).clear(edge_id);
    }

    /// Returns (src, dst) for an edge via the maintained reverse index.
    #[must_use]
    pub fn get_relationship_endpoints(
        &self,
        id: RelationshipId,
    ) -> (NodeId, NodeId) {
        if let Some((src, dst)) = self.endpoints_for_edge(id.0) {
            return (NodeId(src), NodeId(dst));
        }

        panic!("relationship {} not found", id.0);
    }

    /// Iterate the relationship type matrix over a range of edge IDs.
    /// Returns `(edge_id, type_index)` pairs.
    #[must_use]
    pub fn relationship_type_matrix_iter(
        &self,
        min_edge_id: u64,
        max_edge_id: u64,
    ) -> versioned_matrix::Iter {
        self.relationship_type_matrix.iter(min_edge_id, max_edge_id)
    }

    #[must_use]
    pub fn get_relationship_attribute(
        &self,
        id: RelationshipId,
        attr: &Arc<String>,
    ) -> Option<Value> {
        self.attr_by_name(&self.relationship_attrs, id.0, attr)
    }

    /// Fetches a relationship attribute using a pre-resolved attribute
    /// index. Use `get_relationship_attribute_id` to resolve the index
    /// once, then call this per relationship to avoid repeated string
    /// lookups.
    #[must_use]
    pub fn get_relationship_attribute_by_idx(
        &self,
        id: RelationshipId,
        attr_idx: u16,
    ) -> Option<Value> {
        self.relationship_attrs.get_attr_by_idx(id.0, attr_idx)
    }

    /// Borrow the relationship attribute store directly.
    ///
    /// Lets a caller hand a GraphBLAS user-defined operator a minimal, read-only
    /// handle to just the relationship weights (see `algo.MSF`'s weight operator)
    /// rather than a raw pointer to the whole graph.
    #[must_use]
    pub const fn relationship_attrs(&self) -> &AttributeStore {
        &self.relationship_attrs
    }

    /// Batch variant of `get_relationship_attribute_by_idx`.
    /// Pushes one `Value` per id into `out`, substituting `default` for
    /// missing entries (so callers don't allocate a temp `Vec<Option<_>>`).
    pub fn get_relationship_attributes_by_idx(
        &self,
        ids: &[RelationshipId],
        attr_idx: u16,
        default: &Value,
        out: &mut Vec<Value>,
    ) {
        // SAFETY: RelationshipId is `#[repr(transparent)]` over u64.
        let keys: &[u64] = unsafe { std::slice::from_raw_parts(ids.as_ptr().cast(), ids.len()) };
        self.relationship_attrs
            .get_attrs_by_idx_batch_into(keys, attr_idx, default, out);
    }

    fn resize_node_matrices(&mut self) {
        self.adjacancy_matrix.resize(self.node_cap, self.node_cap);
        self.node_labels_matrix
            .resize(self.node_cap, self.labels_matices.len() as u64);
        for label_matrix in &mut self.labels_matices {
            label_matrix.resize(self.node_cap, self.node_cap);
        }
        for relationship_matrix in &mut self.relationship_matrices {
            relationship_matrix.resize(self.node_cap, self.node_cap);
        }
    }

    fn resize_relationship_matrices(&mut self) {
        self.relationship_type_matrix
            .resize(self.relationship_cap, self.relationship_types.len() as u64);
    }

    fn resize(&mut self) {
        if self.node_count > self.node_cap {
            self.node_cap = grow_cap(self.node_cap, self.node_count);
            self.resize_node_matrices();
        }

        if self.labels_matices.len() as u64 > self.node_labels_matrix.ncols() {
            self.node_labels_matrix
                .resize(self.node_cap, self.labels_matices.len() as u64);
        }

        if self.relationship_count > self.relationship_cap {
            self.relationship_cap = grow_cap(self.relationship_cap, self.relationship_count);
            self.resize_relationship_matrices();
        }

        if self.relationship_types.len() as u64 > self.relationship_type_matrix.ncols() {
            self.relationship_type_matrix
                .resize(self.relationship_cap, self.relationship_types.len() as u64);
        }
    }

    // ---- attribute name resolution ----------------------------------------
    //
    // `AttributeStore` is an id-keyed structure and does not own the name table;
    // the graph does. These three resolve between the two, so the store's API stays
    // honest about what it can answer on its own and no caller has to pass the
    // graph's dictionary back into it.

    /// Value of a named attribute on an entity in `store`.
    fn attr_by_name(
        &self,
        store: &AttributeStore,
        key: u64,
        attr: &Arc<String>,
    ) -> Option<Value> {
        let idx = self.attrs_name.get_index_of(attr)? as u16;
        store.get_attr_by_idx(key, idx)
    }

    /// Names of the attributes an entity in `store` carries.
    fn attr_names<'a>(
        &'a self,
        store: &'a AttributeStore,
        key: u64,
    ) -> impl Iterator<Item = Arc<String>> + 'a {
        store
            .get_attr_ids(key)
            .filter_map(move |id| self.attrs_name.get(id as usize).cloned())
    }

    /// Name/value pairs for an entity in `store`, in one storage pass.
    fn attr_pairs<'a>(
        &'a self,
        store: &'a AttributeStore,
        key: u64,
    ) -> impl Iterator<Item = (Arc<String>, Value)> + 'a {
        store
            .get_all_attrs_by_id(key)
            .filter_map(move |(id, value)| {
                self.attrs_name.get(id as usize).map(|n| (n.clone(), value))
            })
    }

    pub fn get_node_attrs(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = Arc<String>> + '_ {
        self.attr_names(&self.node_attrs, id.0)
    }

    /// Get all attribute names and values for a node in a single storage pass.
    pub fn get_node_all_attrs(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = (Arc<String>, Value)> + '_ {
        self.attr_pairs(&self.node_attrs, id.0)
    }

    pub fn get_node_all_attrs_by_id(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = (u16, Value)> + '_ {
        self.node_attrs.get_all_attrs_by_id(id.0)
    }

    /// Number of attributes stored for a node (0 if none).
    #[must_use]
    pub fn get_node_attr_count(
        &self,
        id: NodeId,
    ) -> usize {
        self.node_attrs.attr_count(id.0)
    }

    pub fn get_relationship_attrs(
        &self,
        id: RelationshipId,
    ) -> impl Iterator<Item = Arc<String>> + '_ {
        self.attr_names(&self.relationship_attrs, id.0)
    }

    /// Get all attribute names and values for a relationship in a single storage pass.
    pub fn get_relationship_all_attrs(
        &self,
        id: RelationshipId,
    ) -> impl Iterator<Item = (Arc<String>, Value)> + '_ {
        self.attr_pairs(&self.relationship_attrs, id.0)
    }

    pub fn get_relationship_all_attrs_by_id(
        &self,
        id: RelationshipId,
    ) -> impl Iterator<Item = (u16, Value)> + '_ {
        self.relationship_attrs.get_all_attrs_by_id(id.0)
    }

    /// Number of attributes stored for a relationship (0 if none).
    #[must_use]
    pub fn get_relationship_attr_count(
        &self,
        id: RelationshipId,
    ) -> usize {
        self.relationship_attrs.attr_count(id.0)
    }

    pub fn create_index(
        &mut self,
        index_type: &IndexType,
        entity_type: &EntityType,
        label: &Arc<String>,
        attrs: &Vec<Arc<String>>,
        options: Option<IndexOptions>,
    ) -> Result<(), String> {
        match entity_type {
            EntityType::Node => {
                let len = self.get_label_matrix_mut(label).nvals();
                self.node_indexer
                    .create_index(index_type, label, attrs, len, options)?;
                // Register attribute names so they appear in
                // db.propertyKeys() even if no entity yet uses them.
                for attr in attrs {
                    self.add_node_attribute_name(attr);
                }
                populate_index(IndexKind::Node, label.clone(), self.node_indexer.clone());
            }
            EntityType::Relationship => {
                // get-or-create the relationship type so that
                // db.relationshipTypes() reflects indexed (but still
                // empty) types.
                let len = self.get_relationship_matrix_mut(label).edge_count();
                self.edge_indexer
                    .create_index(index_type, label, attrs, len, options)?;
                for attr in attrs {
                    self.add_rel_attribute_name(attr);
                }
                populate_index(IndexKind::Edge, label.clone(), self.edge_indexer.clone());
            }
        }
        Ok(())
    }

    /// Create an index and populate it synchronously (for RDB load).
    /// Unlike `create_index`, this doesn't spawn async tasks.
    pub fn create_index_sync(
        &mut self,
        index_type: &IndexType,
        entity_type: &EntityType,
        label: &Arc<String>,
        attrs: &Vec<Arc<String>>,
        options: Option<IndexOptions>,
    ) -> Result<(), String> {
        match entity_type {
            EntityType::Node => {
                let len = self.get_label_matrix_mut(label).nvals();
                self.node_indexer
                    .create_index(index_type, label, attrs, len, options)?;
                // Same bookkeeping as `create_index`: surface the
                // indexed attr names in `db.propertyKeys()` even if no
                // entity has yet written to them. Critical for
                // RDB-load and EFFECT_CREATE_INDEX (replica) paths,
                // which go through this sync variant.
                for attr in attrs {
                    self.add_node_attribute_name(attr);
                }
                // Don't spawn async — caller will populate via populate_index_sync
            }
            EntityType::Relationship => {
                // get-or-create the relationship type so
                // `db.relationshipTypes()` reflects indexed (but still
                // empty) types even on replicas / after RDB load.
                let len = self.get_relationship_matrix_mut(label).edge_count();
                self.edge_indexer
                    .create_index(index_type, label, attrs, len, options)?;
                for attr in attrs {
                    self.add_rel_attribute_name(attr);
                }
            }
        }
        Ok(())
    }

    /// Synchronously populate all pending indexes.
    /// Used after RDB load when the graph is fully constructed.
    pub fn populate_indexes_sync(&mut self) {
        let node_snapshots = self.node_indexer.acquire_population_snapshots();
        for snapshot in node_snapshots {
            let label = snapshot.ticket.label().clone();
            let attrs = snapshot.fields;
            if let Some(lm) = self.get_label_matrix(&label) {
                // Pre-resolve attribute indices to avoid string lookups per node
                let resolved_attrs: Vec<(u16, Vec<_>)> = attrs
                    .iter()
                    .filter_map(|(attr, fields)| {
                        self.get_node_attribute_id(attr)
                            .map(|idx| (idx as u16, fields.clone()))
                    })
                    .collect();

                let mut batch = Vec::new();
                for (n, _) in lm.iter(0, u64::MAX) {
                    // Allocate the RSDoc only once a field is present — nodes
                    // without an indexed attribute produce no document (and no
                    // create+free churn).
                    let mut doc: Option<Document> = None;
                    for (attr_idx, fields) in &resolved_attrs {
                        if let Some(value) = self.get_node_attribute_by_idx(NodeId(n), *attr_idx) {
                            let doc = doc.get_or_insert_with(|| Document::new(n));
                            for field in fields {
                                doc.set(field, &value);
                            }
                        }
                    }
                    if let Some(doc) = doc {
                        batch.push(doc);
                    }
                }
                if !batch.is_empty() {
                    let mut add_docs = HashMap::new();
                    add_docs.insert(label.clone(), batch);
                    self.node_indexer.commit(&mut add_docs, &mut HashMap::new());
                }
            }
            self.node_indexer
                .release_population_ticket(&snapshot.ticket);
        }

        // Edge indexes: symmetric to the node path, but walk the
        // relationship tensor and emit `Document::new_edge(src, dst, eid)`
        // so RediSearch keys stay the 24-byte `[src, dst, edge_id]`
        // triple that `Index_RemoveEdge` expects on delete. Stream
        // the tensor iterator directly so we don't materialize every
        // `(src, dst, eid)` triple for large relationship types on
        // RDB load.
        let edge_snapshots = self.edge_indexer.acquire_population_snapshots();
        for snapshot in edge_snapshots {
            let type_name = snapshot.ticket.label().clone();
            let attrs = snapshot.fields;
            if let Some(tensor) = self.get_relationship_matrix(&type_name) {
                let mut batch = Vec::new();
                for (src, dst, eid) in tensor.iter(0, u64::MAX, false) {
                    // Allocate the RSDoc only once a field is present — edges
                    // without an indexed attribute produce no document (and no
                    // create+free churn).
                    let mut doc: Option<Document> = None;
                    for (attr, fields) in &attrs {
                        if let Some(value) = self.attr_by_name(&self.relationship_attrs, eid, attr)
                        {
                            let doc = doc.get_or_insert_with(|| Document::new_edge(src, dst, eid));
                            for field in fields {
                                doc.set(field, &value);
                            }
                        }
                    }
                    if let Some(doc) = doc {
                        batch.push(doc);
                    }
                }
                if !batch.is_empty() {
                    let mut add_docs = HashMap::new();
                    add_docs.insert(type_name.clone(), batch);
                    self.edge_indexer.commit(&mut add_docs, &mut HashMap::new());
                }
            }
            self.edge_indexer
                .release_population_ticket(&snapshot.ticket);
        }
    }

    /// The indexer serialization locks, cloned so a caller can take them **before**
    /// borrowing the graph.
    ///
    /// That is the order `populate_index_batch` uses (indexer lock held across its
    /// graph borrow), so anything that needs a *mutable* borrow of the same version —
    /// the index undo path — must take these first or the two collide, which
    /// `AtomicRefCell` reports by panicking.
    #[must_use]
    pub fn index_locks(&self) -> (Arc<Mutex<()>>, Arc<Mutex<()>>) {
        (
            self.node_indexer.write_lock(),
            self.edge_indexer.write_lock(),
        )
    }

    /// Write node index documents to RediSearch, taking the node indexer lock.
    pub fn commit_index(
        &mut self,
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
        remove_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) {
        let lock = self.node_indexer.write_lock();
        let guard = lock.lock();
        self.commit_index_kind(&guard, IndexKind::Node, index_add_docs, remove_docs);
    }

    /// As [`Self::commit_index`], for a caller that already holds the node indexer
    /// lock from [`Self::index_locks`] — which it must, to have borrowed the graph in
    /// the sanctioned order.
    pub fn commit_index_locked(
        &mut self,
        guard: &MutexGuard<'_, ()>,
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
        remove_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) {
        self.commit_index_kind(guard, IndexKind::Node, index_add_docs, remove_docs);
    }

    /// Write edge index documents to RediSearch, taking the edge indexer lock.
    pub fn commit_edge_index(
        &mut self,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
        remove_edge_docs: &mut FxHashMap<u64, FxHashMap<u64, (u64, u64)>>,
    ) {
        let lock = self.edge_indexer.write_lock();
        let guard = lock.lock();
        self.commit_edge_index_locked(&guard, index_add_edge_docs, remove_edge_docs);
    }

    /// As [`Self::commit_edge_index`], for a caller already holding the edge indexer
    /// lock from [`Self::index_locks`].
    pub fn commit_edge_index_locked(
        &mut self,
        _guard: &MutexGuard<'_, ()>,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
        remove_edge_docs: &mut FxHashMap<u64, FxHashMap<u64, (u64, u64)>>,
    ) {
        if index_add_edge_docs.is_empty() && remove_edge_docs.is_empty() {
            return;
        }

        let indexer = &self.edge_indexer;

        let mut add_docs: HashMap<Arc<String>, Vec<Document>> = HashMap::new();
        for (type_id, ids) in index_add_edge_docs.drain() {
            let name = &self.relationship_types[type_id as usize];
            let fields = indexer.get_fields(name);

            // Resolve `(src, dst)` only for the edge ids we actually
            // need, instead of materializing the full tensor every
            // commit. With N ids out of M tensor entries we stop the
            // scan as soon as all N are found. Edges being indexed
            // are expected to still be live in the tensor (deletes
            // go through the remove path), so any id left in
            // `pending` after the scan is treated the same as before
            // — skipped below.
            let mut pending: FxHashSet<u64> = ids.iter().collect();
            let mut endpoints: FxHashMap<u64, (u64, u64)> =
                FxHashMap::with_capacity_and_hasher(ids.len() as usize, FxBuildHasher);
            if let Some(t) = self.relationship_matrices.get(type_id as usize) {
                for (src, dst, eid) in t.iter(0, u64::MAX, false) {
                    if pending.remove(&eid) {
                        endpoints.insert(eid, (src, dst));
                        if pending.is_empty() {
                            break;
                        }
                    }
                }
            }

            let mut docs = Vec::with_capacity(ids.len() as usize);
            for id in ids {
                let Some(&(src, dst)) = endpoints.get(&id) else {
                    // Edge vanished between track time and commit;
                    // nothing to index.
                    continue;
                };
                let mut doc = Document::new_edge(src, dst, id);
                for (key, fields) in &fields {
                    if let Some(value) = self.attr_by_name(&self.relationship_attrs, id, key) {
                        for field in fields {
                            doc.set(field, &value);
                        }
                    }
                }
                docs.push(doc);
            }
            add_docs.insert(name.clone(), docs);
        }

        // Removes: (src, dst) captured at delete time lets us
        // reconstruct the 24-byte `[src, dst, edge_id]` RediSearch key
        // — matches FalkorDB C's `Index_RemoveEdge` in
        // `src/index/index_edge.c`.
        let mut remove: HashMap<Arc<String>, HashMap<u64, (u64, u64)>> = HashMap::new();
        for (type_id, edges) in remove_edge_docs.drain() {
            let name = &self.relationship_types[type_id as usize];
            remove.insert(name.clone(), edges.into_iter().collect());
        }

        indexer.commit_edge(&mut add_docs, &mut remove);
    }

    /// Shared body for `commit_index` / `commit_edge_index`: resolve the
    /// id sets (keyed by label or type id) into `Document`s against the
    /// matching attribute store, then hand them to the underlying
    /// `Indexer::commit`.
    fn commit_index_kind(
        &mut self,
        _guard: &MutexGuard<'_, ()>,
        kind: IndexKind,
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
        remove_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) {
        if index_add_docs.is_empty() && remove_docs.is_empty() {
            return;
        }

        let (indexer, names, attr_store) = match kind {
            IndexKind::Node => (&self.node_indexer, &self.node_labels, &self.node_attrs),
            IndexKind::Edge => unreachable!("use commit_edge_index for edges"),
        };

        let mut add_docs = HashMap::new();
        for (slot, ids) in index_add_docs.drain() {
            let name = &names[slot as usize];
            let fields = indexer.get_fields(name);
            let mut docs = vec![];
            for id in ids {
                let mut doc = Document::new(id);
                for (key, fields) in &fields {
                    if let Some(value) = self.attr_by_name(attr_store, id, key) {
                        for field in fields {
                            doc.set(field, &value);
                        }
                    }
                }
                docs.push(doc);
            }
            add_docs.insert(name.clone(), docs);
        }

        let mut remove = HashMap::new();
        for (slot, ids) in remove_docs.drain() {
            let name = &names[slot as usize];
            remove.insert(name.clone(), ids);
        }

        indexer.commit(&mut add_docs, &mut remove);
    }

    pub fn drop_index(
        &mut self,
        index_type: &IndexType,
        entity_type: &EntityType,
        label: &Arc<String>,
        attrs: &[Arc<String>],
    ) -> Result<usize, String> {
        // Expand an empty `attrs` to the full set of fields of `index_type`
        // (matches the `target_attrs` derivation in `Indexer::drop_index`);
        // otherwise an empty-attr drop would bypass constraint protection.
        let indexer = match entity_type {
            EntityType::Node => &self.node_indexer,
            EntityType::Relationship => &self.edge_indexer,
        };
        let effective_attrs: Vec<Arc<String>> = if attrs.is_empty() {
            indexer
                .get_fields(label)
                .into_iter()
                .filter(|(_, fields)| fields.iter().any(|f| f.ty == *index_type))
                .map(|(attr, _)| attr)
                .collect()
        } else {
            attrs.to_vec()
        };

        // Check if any UNIQUE constraint depends on this index
        for attr in &effective_attrs {
            if self.constraint_depends_on_index(entity_type, label, attr, index_type) {
                return Err("Index supports constraint".to_string());
            }
        }

        let (indexer, total, kind) = match entity_type {
            EntityType::Node => {
                let total = self.get_label_matrix(label).map_or(
                    0,
                    super::graphblas::versioned_matrix::VersionedMatrix::nvals,
                );
                (&mut self.node_indexer, total, IndexKind::Node)
            }
            EntityType::Relationship => {
                let total = self
                    .get_relationship_matrix(label)
                    .map_or(0, Tensor::edge_count);
                (&mut self.edge_indexer, total, IndexKind::Edge)
            }
        };

        let lock = indexer.write_lock();
        let _guard = lock.lock();

        let reindex = indexer.drop_index(label, attrs, index_type, total);

        match reindex {
            Some((dropped, remaining)) if dropped > 0 => {
                if remaining > 0 {
                    indexer.recreate_index(label)?;
                    populate_index(kind, label.clone(), indexer.clone());
                } else {
                    drop_index_bg(label.clone(), indexer.clone());
                }
                Ok(dropped)
            }
            _ => {
                // Include every requested attr so multi-attribute
                // drops don't silently trim the list in the error.
                let attr_list = attrs
                    .iter()
                    .map(|a| a.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                Err(format!(
                    "Unable to drop index on :{label}({attr_list}): no such index."
                ))
            }
        }
    }

    #[must_use]
    pub fn is_indexed(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
        index_type: &IndexType,
    ) -> bool {
        self.node_indexer.is_label_indexed(label, field, index_type)
    }

    pub fn get_indexed_nodes(
        &self,
        label: &Arc<String>,
        query: IndexQuery<Value>,
    ) -> impl Iterator<Item = NodeId> + use<> {
        self.node_indexer.query(label, query).map(NodeId)
    }

    #[must_use]
    pub fn is_edge_indexed(
        &self,
        label: &Arc<String>,
        field: &Arc<String>,
        index_type: &IndexType,
    ) -> bool {
        self.edge_indexer.is_label_indexed(label, field, index_type)
    }

    pub fn get_indexed_edges(
        &self,
        label: &Arc<String>,
        query: IndexQuery<Value>,
    ) -> impl Iterator<Item = (NodeId, NodeId, RelationshipId)> + use<> {
        // Edge index documents carry `(src, dst, edge_id)` in their
        // 24-byte key (set by `Document::new_edge`), so the result
        // iterator materializes endpoints directly — no relationship
        // tensor scan. Matches FalkorDB C's `EdgeIndexKey` layout.
        self.edge_indexer
            .query_edges(label, query)
            .map(|(src, dst, eid)| (NodeId(src), NodeId(dst), RelationshipId(eid)))
    }

    /// Get all edges of a given type (fallback when index can't be utilized).
    pub fn get_all_edges(
        &self,
        label: &Arc<String>,
    ) -> Vec<(NodeId, NodeId, RelationshipId)> {
        self.get_relationship_matrix(label)
            .map_or_else(std::vec::Vec::new, |tensor| {
                tensor
                    .iter(0, u64::MAX, false)
                    .map(|(src, dst, eid)| (NodeId(src), NodeId(dst), RelationshipId(eid)))
                    .collect()
            })
    }

    pub fn fulltext_query_nodes(
        &self,
        label: &Arc<String>,
        query: &str,
    ) -> Result<impl Iterator<Item = (NodeId, f64)> + use<>, String> {
        self.node_indexer
            .fulltext_query(label, query)
            .map(|r| r.map(|(id, score)| (NodeId(id), score)))
    }

    /// Execute a fulltext query against an *edge* index, yielding
    /// `(src, dst, edge_id, score)` tuples. Mirrors
    /// [`fulltext_query_nodes`] but goes through the `edge_indexer`
    /// and reads the 24-byte edge-index key produced by
    /// `Document::new_edge`.
    pub fn fulltext_query_edges(
        &self,
        label: &Arc<String>,
        query: &str,
    ) -> Result<impl Iterator<Item = (NodeId, NodeId, RelationshipId, f64)> + use<>, String> {
        self.edge_indexer
            .fulltext_query_edges(label, query)
            .map(|r| {
                r.map(|(src, dst, eid, score)| {
                    (NodeId(src), NodeId(dst), RelationshipId(eid), score)
                })
            })
    }

    /// Execute a KNN vector query against a node label's index,
    /// yielding `(NodeId, distance)` pairs ordered by ascending
    /// distance.
    ///
    /// Distances are computed manually from the query vector and each
    /// entity's stored vector — RediSearch's per-result score is *not*
    /// the KNN distance, and reading it returns 0. For each result we
    /// look up the entity's vector property and apply the index's
    /// similarity function. Results are eagerly materialized into a
    /// `Vec` so the caller can drop the graph borrow before iterating.
    pub fn vector_query_nodes(
        &self,
        label: &Arc<String>,
        field: &str,
        vector: Arc<thin_vec::ThinVec<f32>>,
        k: usize,
    ) -> Result<impl Iterator<Item = (NodeId, f64)> + use<>, String> {
        let attr = Arc::new(field.to_string());
        if let Some(expected) = self.node_indexer.get_vector_dimension(label, &attr)
            && expected as usize != vector.len()
        {
            return Err(format!(
                "Vector dimension mismatch, expected {expected} but got {}",
                vector.len()
            ));
        }
        let metric = self.node_indexer.get_vector_metric(label, &attr);
        let query_vec = Arc::clone(&vector);
        let raw_iter = self.node_indexer.vector_query(label, field, vector, k)?;

        // Resolve the attribute name to its numeric slot once, rather than
        // re-hashing the attribute string for every KNN result. If the
        // attribute is unknown there are no vectors to score.
        let mut out: Vec<(NodeId, f64)> = Vec::with_capacity(k);
        let Some(attr_idx) = self.get_node_attribute_id(&attr).map(|i| i as u16) else {
            return Ok(out.into_iter());
        };
        // Collect the candidate ids first, then fetch their vectors in one
        // fused batch pass. This amortizes the per-shard read lock and gives
        // the attribute cache sequential access instead of one isolated
        // lookup per KNN result.
        let node_ids: Vec<NodeId> = raw_iter.map(|(id, _score)| NodeId(id)).collect();
        let mut vecs: Vec<Value> = Vec::with_capacity(node_ids.len());
        self.get_node_attributes_by_idx(&node_ids, attr_idx, &Value::Null, &mut vecs);
        for (node_id, entity) in node_ids.into_iter().zip(vecs) {
            let Value::VecF32(entity_vec) = entity else {
                continue;
            };
            if let Some(d) = vec_distance::distance(metric.as_deref(), &query_vec, &entity_vec) {
                out.push((node_id, d));
            }
        }
        // HNSW returns rows in (approximate) score order, but the
        // distance we recompute here may diverge slightly from
        // VecSim's internal ranking (different precision, different
        // reduction order). Sort by the value we actually expose so
        // the documented ascending-distance contract holds.
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out.into_iter())
    }

    /// Execute a KNN vector query against an *edge* index, yielding
    /// `(src, dst, edge_id, distance)` tuples. Distances are computed
    /// from the query vector and each edge's stored vector — see
    /// [`vector_query_nodes`] for why.
    pub fn vector_query_edges(
        &self,
        label: &Arc<String>,
        field: &str,
        vector: Arc<thin_vec::ThinVec<f32>>,
        k: usize,
    ) -> Result<impl Iterator<Item = (NodeId, NodeId, RelationshipId, f64)> + use<>, String> {
        let attr = Arc::new(field.to_string());
        if let Some(expected) = self.edge_indexer.get_vector_dimension(label, &attr)
            && expected as usize != vector.len()
        {
            return Err(format!(
                "Vector dimension mismatch, expected {expected} but got {}",
                vector.len()
            ));
        }
        let metric = self.edge_indexer.get_vector_metric(label, &attr);
        let query_vec = Arc::clone(&vector);
        let raw_iter = self
            .edge_indexer
            .vector_query_edges(label, field, vector, k)?;

        // Resolve the attribute slot once instead of per KNN result.
        let mut out: Vec<(NodeId, NodeId, RelationshipId, f64)> = Vec::with_capacity(k);
        let Some(attr_idx) = self.get_relationship_attribute_id(&attr).map(|i| i as u16) else {
            return Ok(out.into_iter());
        };
        // Collect candidate triples first, then fetch their vectors in one
        // fused batch pass (see `vector_query_nodes`).
        let triples: Vec<(NodeId, NodeId, RelationshipId)> = raw_iter
            .map(|(src, dst, eid, _score)| (NodeId(src), NodeId(dst), RelationshipId(eid)))
            .collect();
        let edge_ids: Vec<RelationshipId> = triples.iter().map(|&(_, _, eid)| eid).collect();
        let mut vecs: Vec<Value> = Vec::with_capacity(edge_ids.len());
        self.get_relationship_attributes_by_idx(&edge_ids, attr_idx, &Value::Null, &mut vecs);
        for ((src, dst, edge_id), entity) in triples.into_iter().zip(vecs) {
            let Value::VecF32(entity_vec) = entity else {
                continue;
            };
            if let Some(d) = vec_distance::distance(metric.as_deref(), &query_vec, &entity_vec) {
                out.push((src, dst, edge_id, d));
            }
        }
        // See `vector_query_nodes` for why we sort by recomputed
        // distance instead of trusting the upstream HNSW order.
        out.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out.into_iter())
    }

    #[must_use]
    pub fn index_info(&self) -> Vec<IndexInfo> {
        let mut infos = self.node_indexer.index_info();
        for info in &mut infos {
            info.entity_type = String::from("NODE");
        }
        let mut edge_infos = self.edge_indexer.index_info();
        for info in &mut edge_infos {
            info.entity_type = String::from("RELATIONSHIP");
        }
        infos.extend(edge_infos);
        infos
    }

    // ── Constraint management ─────────────────────────────────────────

    #[must_use]
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    pub const fn constraints_mut(&mut self) -> &mut Vec<Constraint> {
        &mut self.constraints
    }

    /// Create a constraint with validation of existing data.
    pub fn create_constraint(
        &mut self,
        ct: ConstraintType,
        entity_type: EntityType,
        label: Arc<String>,
        properties: Vec<Arc<String>>,
    ) -> Result<bool, String> {
        if ct == ConstraintType::Unique
            && !self.has_supporting_index(&entity_type, &label, &properties)
        {
            return Err("missing supporting exact-match index".into());
        }

        // Check for duplicates
        let existing_idx = self
            .constraints
            .iter()
            .position(|c| c.matches(&ct, &entity_type, &label, &properties));
        if let Some(idx) = existing_idx {
            if self.constraints[idx].status == ConstraintStatus::Failed {
                // Remove the failed constraint so it can be recreated
                self.constraints.remove(idx);
            } else {
                return Err("Constraint already exists".into());
            }
        }

        // Register the label and the property names before the constraint is
        // stored. Without this a constraint on a not-yet-seen property is
        // persisted with attribute id 0 — `encode_constraint_block` resolves
        // each property with `position(..).unwrap_or(0)` — and reads back as a
        // different property entirely across any RDB save or replica sync.
        // Replication needs the ids for the same reason. C does this in
        // `GraphHub_AddConstraint`.
        match entity_type {
            EntityType::Node => {
                self.get_label_id_mut(&label);
            }
            EntityType::Relationship => {
                self.get_type_id_mut(&label);
            }
        }
        for property in &properties {
            self.add_node_attribute_name(property);
        }

        let mut constraint = Constraint::new(ct, entity_type, label, properties);

        // Get entity count to decide sync vs async validation
        let count = self.get_constraint_entity_count(&constraint);

        if count <= 10_000 {
            // Synchronous validation for small datasets
            if self.validate_constraint(&constraint) {
                constraint.status = ConstraintStatus::Operational;
            } else {
                constraint.status = ConstraintStatus::Failed;
            }
            self.constraints.push(constraint);
            Ok(false) // no async validation needed
        } else {
            // Large dataset: mark under construction, caller spawns background validation
            constraint.status = ConstraintStatus::UnderConstruction;
            self.constraints.push(constraint);
            Ok(true) // async validation needed
        }
    }

    /// Add a constraint directly (for replication/restore). No validation.
    pub fn add_constraint_raw(
        &mut self,
        constraint: Constraint,
    ) {
        self.constraints.push(constraint);
    }

    /// Install a constraint with the status the primary reached, adding it if
    /// this graph does not have it yet and updating the status if it does.
    ///
    /// Idempotent because the primary announces a constraint **twice**: once
    /// when it is created, still under construction, and again when validation
    /// finishes and it is either operational or failed. A plain add would leave
    /// a duplicate; a plain "already exists, ignore" would leave the replica
    /// believing it is still under construction forever, which is what makes
    /// the second announcement load-bearing rather than noise.
    pub fn upsert_constraint_raw(
        &mut self,
        ct: ConstraintType,
        entity_type: EntityType,
        label: &Arc<String>,
        properties: &[Arc<String>],
        status: ConstraintStatus,
    ) {
        if let Some(existing) = self.constraints.iter_mut().find(|c| {
            c.ct == ct
                && c.entity_type == entity_type
                && c.label == *label
                && c.properties == properties
        }) {
            existing.status = status;
            return;
        }
        let mut constraint = Constraint::new(ct, entity_type, label.clone(), properties.to_vec());
        constraint.status = status;
        self.constraints.push(constraint);
    }

    fn get_constraint_entity_count(
        &self,
        constraint: &Constraint,
    ) -> u64 {
        match constraint.entity_type {
            EntityType::Node => self.label_node_count(&constraint.label),
            EntityType::Relationship => self
                .get_relationship_matrix(&constraint.label)
                .map_or(0, Tensor::edge_count),
        }
    }

    fn validate_constraint(
        &self,
        constraint: &Constraint,
    ) -> bool {
        match constraint.ct {
            ConstraintType::Mandatory => self.validate_mandatory_constraint(constraint),
            ConstraintType::Unique => self.validate_unique_constraint(constraint),
        }
    }

    /// Validate all constraints currently under construction and update their status.
    pub fn validate_pending_constraints(&mut self) {
        // Collect indices of pending constraints to avoid borrow conflicts
        let pending_indices: Vec<usize> = self
            .constraints
            .iter()
            .enumerate()
            .filter(|(_, c)| c.status == ConstraintStatus::UnderConstruction)
            .map(|(i, _)| i)
            .collect();

        for i in pending_indices {
            let valid = self.validate_constraint(&self.constraints[i].clone());
            self.constraints[i].status = if valid {
                ConstraintStatus::Operational
            } else {
                ConstraintStatus::Failed
            };
        }
    }

    /// Read-only phase of async constraint validation: compute the outcome
    /// (valid / invalid) for every constraint currently under construction,
    /// without mutating state. Pair with
    /// [`apply_constraint_validation_results`] under a write lock.
    ///
    /// Returns `(constraint_id, valid)` rather than indices because
    /// `drop_constraint` uses `swap_remove`, which invalidates positional
    /// references between the compute and apply phases.
    #[must_use]
    pub fn compute_pending_constraint_results(&self) -> Vec<(u64, bool)> {
        self.constraints
            .iter()
            .filter(|c| c.status == ConstraintStatus::UnderConstruction)
            .map(|c| (c.id, self.validate_constraint(c)))
            .collect()
    }

    /// Apply results computed by [`compute_pending_constraint_results`].
    /// Status is updated only for constraints that are still under
    /// construction and still present (a concurrent drop may have removed
    /// them, and `swap_remove` may have shuffled others into their slot).
    pub fn apply_constraint_validation_results(
        &mut self,
        results: Vec<(u64, bool)>,
    ) {
        for (id, valid) in results {
            if let Some(c) = self
                .constraints
                .iter_mut()
                .find(|c| c.id == id && c.status == ConstraintStatus::UnderConstruction)
            {
                c.status = if valid {
                    ConstraintStatus::Operational
                } else {
                    ConstraintStatus::Failed
                };
            }
        }
    }

    fn validate_mandatory_constraint(
        &self,
        constraint: &Constraint,
    ) -> bool {
        match constraint.entity_type {
            EntityType::Node => {
                let Some(lm) = self.get_label_matrix(&constraint.label) else {
                    return true;
                };
                for (node_id, _) in lm.iter(0, u64::MAX) {
                    for prop in &constraint.properties {
                        if self
                            .get_node_attribute(NodeId(node_id), prop)
                            .is_none_or(|val| matches!(val, Value::Null))
                        {
                            return false;
                        }
                    }
                }
                true
            }
            EntityType::Relationship => {
                let Some(tensor) = self.get_relationship_matrix(&constraint.label) else {
                    return true;
                };
                for (_, _, edge_id) in tensor.iter(0, u64::MAX, false) {
                    for prop in &constraint.properties {
                        if self
                            .get_relationship_attribute(RelationshipId(edge_id), prop)
                            .is_none_or(|val| matches!(val, Value::Null))
                        {
                            return false;
                        }
                    }
                }
                true
            }
        }
    }

    fn validate_unique_constraint(
        &self,
        constraint: &Constraint,
    ) -> bool {
        match constraint.entity_type {
            EntityType::Node => {
                let Some(lm) = self.get_label_matrix(&constraint.label) else {
                    return true;
                };
                let mut seen: FxHashSet<Vec<u8>> = FxHashSet::default();
                for (node_id, _) in lm.iter(0, u64::MAX) {
                    let key = Self::build_composite_key(&constraint.properties, |prop| {
                        self.get_node_attribute(NodeId(node_id), prop)
                    });
                    if key.is_empty() {
                        continue; // All NULL → skip
                    }
                    if !seen.insert(key) {
                        return false;
                    }
                }
                true
            }
            EntityType::Relationship => {
                let Some(tensor) = self.get_relationship_matrix(&constraint.label) else {
                    return true;
                };
                let mut seen: FxHashSet<Vec<u8>> = FxHashSet::default();
                for (_, _, edge_id) in tensor.iter(0, u64::MAX, false) {
                    let key = Self::build_composite_key(&constraint.properties, |prop| {
                        self.get_relationship_attribute(RelationshipId(edge_id), prop)
                    });
                    if key.is_empty() {
                        continue;
                    }
                    if !seen.insert(key) {
                        return false;
                    }
                }
                true
            }
        }
    }

    #[must_use]
    pub fn build_composite_key(
        properties: &[Arc<String>],
        mut get: impl FnMut(&Arc<String>) -> Option<Value>,
    ) -> Vec<u8> {
        let mut all_null = true;
        let mut key = Vec::new();
        for prop in properties {
            match get(prop) {
                Some(v) if !matches!(v, Value::Null) => {
                    all_null = false;
                    key.extend_from_slice(format!("{v:?}").as_bytes());
                }
                _ => {
                    key.push(0); // NULL marker
                }
            }
            key.push(b'|');
        }
        if all_null { Vec::new() } else { key }
    }

    /// Drop a constraint by type, entity type, label and properties.
    pub fn drop_constraint(
        &mut self,
        ct: &ConstraintType,
        entity_type: &EntityType,
        label: &str,
        properties: &[Arc<String>],
    ) -> Result<(), String> {
        let idx = self
            .constraints
            .iter()
            .position(|c| c.matches(ct, entity_type, label, properties));
        match idx {
            Some(i) => {
                self.constraints.swap_remove(i);
                Ok(())
            }
            None => Err("Unable to drop constraint, no such constraint.".into()),
        }
    }

    /// Check if a unique constraint for the given label and properties has
    /// a supporting exact-match index for each property.
    #[must_use]
    pub fn has_supporting_index(
        &self,
        entity_type: &EntityType,
        label: &Arc<String>,
        properties: &[Arc<String>],
    ) -> bool {
        let indexer = match entity_type {
            EntityType::Node => &self.node_indexer,
            EntityType::Relationship => &self.edge_indexer,
        };
        // Check if the index exists (regardless of operational status)
        // since index population may still be in progress
        properties
            .iter()
            .all(|prop| indexer.has_field_for_label(label, prop, &IndexType::Range))
    }

    /// Check if any UNIQUE constraint depends on an index for the given label and attribute.
    #[must_use]
    pub fn constraint_depends_on_index(
        &self,
        entity_type: &EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        index_type: &IndexType,
    ) -> bool {
        // Only Range indexes support UNIQUE constraints
        if *index_type != IndexType::Range {
            return false;
        }
        self.constraints.iter().any(|c| {
            c.ct == ConstraintType::Unique
                && c.entity_type == *entity_type
                && c.label.as_str() == label.as_str()
                && c.properties.iter().any(|p| p.as_str() == attr.as_str())
        })
    }

    pub fn cancel_indexing(&self) {
        self.node_indexer.cancel();
        self.edge_indexer.cancel();
    }

    /// Takes `&self` (not `&mut self`): it only publishes the graph
    /// reference into the node/edge indexers' own `Mutex`-guarded fields,
    /// via `Indexer::set_graph`. This lets `MvccGraph::commit()` call it
    /// through an immutable borrow of the graph, so the background index
    /// population thread's own immutable borrows of the same
    /// `AtomicRefCell<Graph>` are never blocked by (or racing against) a
    /// concurrent mutable borrow here -- see `MvccGraph::commit()`.
    pub fn set_indexer_graph(
        &self,
        graph: Arc<AtomicRefCell<Self>>,
    ) {
        self.node_indexer.set_graph(graph.clone());
        self.edge_indexer.set_graph(graph);
    }

    /// Build a materialized boolean adjacency matrix filtered by relationship types.
    /// If `rel_types` is empty, returns the full adjacency matrix.
    /// The caller owns the returned `Matrix`.
    #[must_use]
    pub fn build_adjacency_matrix(
        &self,
        rel_types: &[Arc<String>],
    ) -> Matrix<bool> {
        if rel_types.is_empty() {
            self.adjacancy_matrix.extract()
        } else if let [rel_type] = rel_types {
            // Single type: the extract already materializes an owned bool
            // matrix; skip the extra new + eWiseAdd pass.
            self.get_type_id(rel_type).map_or_else(
                || Matrix::<bool>::new(self.node_cap, self.node_cap),
                |type_id| self.relationship_matrices[usize::from(type_id)].extract(),
            )
        } else {
            let mut result = Matrix::<bool>::new(self.node_cap, self.node_cap);
            for rel_type in rel_types {
                if let Some(type_id) = self.get_type_id(rel_type) {
                    let m = self.relationship_matrices[usize::from(type_id)].extract();
                    result.element_wise_add(None, None, Some(&m), None);
                }
            }
            result
        }
    }

    /// Build a materialized boolean adjacency matrix that is symmetric (A + A^T).
    /// Used for undirected graph algorithms (WCC, CDLP, MSF).
    #[must_use]
    pub fn build_symmetric_adjacency_matrix(
        &self,
        rel_types: &[Arc<String>],
    ) -> Matrix<bool> {
        let a = self.build_adjacency_matrix(rel_types);
        let at = a.transpose();
        let mut result = Matrix::<bool>::new(self.node_cap, self.node_cap);
        result.element_wise_add(None, Some(&a), Some(&at), None);
        result
    }

    /// Compute a detailed breakdown of memory usage for `GRAPH.MEMORY USAGE`.
    ///
    /// `samples` controls how many entities are sampled per label/type when
    /// estimating attribute memory.  Clamped to \[1, 10000\].
    #[must_use]
    pub fn memory_usage_report(
        &self,
        samples: usize,
    ) -> MemoryUsageReport {
        let samples = samples.clamp(1, 10000);

        // --- label matrices ---
        let mut label_matrices_sz: usize = 0;
        label_matrices_sz += self.node_labels_matrix.memory_usage();
        for lm in &self.labels_matices {
            label_matrices_sz += lm.memory_usage();
        }

        // --- relation matrices ---
        // Per-type tensors plus the combined adjacency matrix and the zero
        // matrix, matching what C's `Graph_memoryUsage` folds in here.
        let mut relation_matrices_sz: usize =
            self.adjacancy_matrix.memory_usage() + self.zero_matrix.memory_usage();
        for rm in &self.relationship_matrices {
            relation_matrices_sz += rm.memory_usage();
        }

        // --- node block storage ---
        // Everything a node costs that is not its property values: the
        // attribute store's unattributed bytes (the per-node slot, which is
        // what makes an attribute-less node not free, and the counterpart of
        // C's per-node DataBlock item) and the deleted-id bitmap. Node
        // existence itself costs nothing beyond those — it is the id range
        // minus that bitmap, not a stored structure. Property values are
        // reported under the attribute components, and
        // `structural_memory_usage` excludes exactly those, so the two halves
        // cover the store without overlap.
        let node_block_storage_sz: usize =
            self.node_attrs.structural_memory_usage() + self.deleted_nodes.serialized_size();

        // --- edge block storage ---
        // Mirrors the node side: the matrix recording each edge's existence and
        // type, plus the graph-wide edge_id → endpoint reverse index (one slot
        // per edge id, 4, 6, 8 or 16 bytes wide depending on the tier).
        let edge_block_storage_sz: usize = self.relationship_type_matrix.memory_usage()
            + self.relationship_attrs.structural_memory_usage()
            + self.deleted_relationships.serialized_size()
            + self.edge_endpoints.memory_usage();

        // --- node attributes by label (sampling) ---
        let mut node_attr_by_label: Vec<(Arc<String>, usize)> = Vec::new();

        // Track processed node IDs (deduplicated) so that multi-label nodes are
        // counted once, under their first (lowest-index) label — matching the C
        // implementation — and so unlabeled nodes can be detected afterwards.
        // A dense bitvector indexed by node id gives O(1) membership/insert,
        // far cheaper than a RoaringTreemap when touching every node.
        // Sized by `node_cap` (the matrix dimension) rather than `max_node_id()`:
        // effects replay can leave sparse high ids whose value exceeds
        // `node_count + deleted_nodes.len() - 1`, but every id present in the
        // label/all-node matrices is strictly below `node_cap`.
        let mut seen = vec![false; self.node_cap() as usize];
        let mut total_labeled: u64 = 0;

        for (label_idx, label_name) in self.node_labels.iter().enumerate() {
            let label_matrix = &self.labels_matices[label_idx];
            let total_for_label = label_matrix.nvals();
            if total_for_label == 0 {
                node_attr_by_label.push((label_name.clone(), 0));
                continue;
            }

            // Iterate the full label matrix.  For attribute estimation, only
            // sample nodes that have NOT been processed by a previous label
            // (avoids double-counting multi-label nodes, matching C).
            let mut sampled_mem: usize = 0;
            let mut sampled_count: usize = 0;
            let mut unprocessed_count: u64 = 0;
            for (node_id, _) in label_matrix.iter(0, u64::MAX) {
                let slot = &mut seen[node_id as usize];
                if !*slot {
                    *slot = true;
                    total_labeled += 1;
                    unprocessed_count += 1;
                    if sampled_count < samples {
                        sampled_mem += self.estimate_entity_attr_size(&self.node_attrs, node_id);
                        sampled_count += 1;
                    }
                }
            }

            let estimated = if sampled_count > 0 {
                (sampled_mem as u64 * unprocessed_count / sampled_count as u64) as usize
            } else {
                0
            };
            node_attr_by_label.push((label_name.clone(), estimated));
        }

        // --- unlabeled node attributes ---
        let total_nodes = self.node_count;
        let unlabeled_count = total_nodes.saturating_sub(total_labeled);
        let unlabeled_node_attr_sz = if unlabeled_count > 0 {
            // Sample unlabeled nodes, skipping labeled ones. Live ids are the
            // id range minus the deleted set — the same walk `get_nodes` uses
            // for an unfiltered scan.
            let mut sampled_mem: usize = 0;
            let mut sampled_count: usize = 0;
            for node_id in 0..=self.max_node_id() {
                if sampled_count >= samples {
                    break;
                }
                if self.deleted_nodes.contains(node_id) || seen[node_id as usize] {
                    continue;
                }
                sampled_mem += self.estimate_entity_attr_size(&self.node_attrs, node_id);
                sampled_count += 1;
            }
            if sampled_count > 0 {
                (sampled_mem as u64 * unlabeled_count / sampled_count as u64) as usize
            } else {
                0
            }
        } else {
            0
        };

        // --- edge attributes by type (sampling) ---
        let mut edge_attr_by_type: Vec<(Arc<String>, usize)> = Vec::new();
        for (type_idx, type_name) in self.relationship_types.iter().enumerate() {
            let tensor = &self.relationship_matrices[type_idx];
            let total_for_type = tensor.edge_count();
            if total_for_type == 0 {
                edge_attr_by_type.push((type_name.clone(), 0));
                continue;
            }

            let mut sampled_mem: usize = 0;
            let mut sampled_count: usize = 0;
            for (_, _, edge_id) in tensor.iter(0, u64::MAX, false) {
                if sampled_count >= samples {
                    break;
                }
                sampled_mem += self.estimate_entity_attr_size(&self.relationship_attrs, edge_id);
                sampled_count += 1;
            }

            let estimated = if sampled_count > 0 {
                (sampled_mem as u64 * total_for_type / sampled_count as u64) as usize
            } else {
                0
            };
            edge_attr_by_type.push((type_name.clone(), estimated));
        }

        // --- indices ---
        let indices_sz = self.node_indexer.memory_usage() + self.edge_indexer.memory_usage();

        MemoryUsageReport {
            label_matrices_sz,
            relation_matrices_sz,
            node_block_storage_sz,
            node_attr_by_label,
            unlabeled_node_attr_sz,
            edge_block_storage_sz,
            edge_attr_by_type,
            indices_sz,
        }
    }

    fn estimate_entity_attr_size(
        &self,
        store: &AttributeStore,
        entity_id: u64,
    ) -> usize {
        store.entity_memory_usage(entity_id)
    }

    /// Encode a single payload entry.
    pub fn encode_payload(
        &self,
        w: &mut dyn Writer,
        p: &PayloadEntry,
    ) {
        match p.state {
            EncodeState::Nodes => {
                self.node_attrs.encode_with_range(
                    w,
                    &self.deleted_nodes,
                    self.max_node_id(),
                    p.count,
                    p.offset,
                );
            }
            EncodeState::DeletedNodes => {
                self.deleted_nodes.encode_with_range(w, p.count, p.offset);
            }
            EncodeState::Edges => {
                self.relationship_attrs.encode_with_range(
                    w,
                    &self.deleted_relationships,
                    self.max_relationship_id(),
                    p.count,
                    p.offset,
                );
            }
            EncodeState::DeletedEdges => {
                self.deleted_relationships
                    .encode_with_range(w, p.count, p.offset);
            }
            EncodeState::LabelsMatrices => {
                let label_matrices = self.label_matrices();
                w.write_unsigned(label_matrices.len() as u64);
                for (i, lm) in label_matrices.iter().enumerate() {
                    w.write_unsigned(i as u64);
                    lm.encode(w);
                }
            }
            EncodeState::RelationMatrices => {
                let tensors = self.relationship_tensors();
                for (i, tensor) in tensors.iter().enumerate() {
                    w.write_unsigned(i as u64);
                    tensor.encode(w);
                }
            }
            EncodeState::AdjMatrix => self.adjacancy_matrix.encode(w),
            EncodeState::LblsMatrix => self.node_labels_matrix.encode(w),
            _ => {}
        }
    }

    /// Get node attribute names.
    #[must_use]
    pub fn get_node_attribute_names(&self) -> Vec<Arc<String>> {
        self.attrs_name.iter().cloned().collect()
    }

    /// Get relationship attribute names.
    #[must_use]
    pub fn get_relationship_attribute_names(&self) -> Vec<Arc<String>> {
        self.attrs_name.iter().cloned().collect()
    }

    /// Register a node attribute name (get-or-create). Used by effect
    /// replication to pre-register attribute names on the replica.
    pub fn add_node_attribute_name(
        &mut self,
        name: &str,
    ) {
        let arc = Arc::new(name.to_string());
        if self.attrs_name.get_index_of(&arc).is_none() {
            self.attrs_name.insert(arc);
        }
    }

    /// Register a relationship attribute name (get-or-create). Used by effect
    /// replication to pre-register attribute names on the replica.
    pub fn add_rel_attribute_name(
        &mut self,
        name: &str,
    ) {
        // Identical to `add_node_attribute_name` now that there is one dictionary. Both
        // are kept because the effects wire still carries an ATTR_NODE/ATTR_REL
        // discriminator, which is vestigial from here on and is removed with the
        // `EFFECTS_VERSION` bump in #2419.
        self.attrs_name.insert(Arc::new(name.to_string()));
    }

    /// The graph's attribute names, in id order — index *is* the id.
    ///
    /// Was a node ∪ relationship union over two dictionaries; with one dictionary it is
    /// simply that dictionary's contents, and the RDB's flat table is the same list.
    #[must_use]
    pub fn build_global_attrs(&self) -> Vec<Arc<String>> {
        self.attrs_name.iter().cloned().collect()
    }
}

#[cfg(test)]
mod attr_id_space_tests {
    use super::super::graphblas::test_init::ensure_init;
    use super::*;

    fn attr(s: &str) -> Arc<String> {
        Arc::new(s.to_string())
    }

    /// A name has one id, whichever kind of entity introduced it.
    ///
    /// This is the invariant behind #2457. With a dictionary per store, `since` was
    /// registered only in the relationship table, so it got relationship-local id 0 —
    /// colliding with the *node* attribute already holding node-local id 0. Effects put a
    /// bare id on the wire, so a replica whose dictionary came from an RDB (one unified
    /// table) resolved that 0 to the wrong attribute.
    #[test]
    fn one_id_per_name_across_entity_kinds() {
        ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, "attr_id_space");

        // Node attributes first, so a per-store numbering would start relationships back
        // at 0 and collide with `a`.
        let a = g.get_or_create_node_attr_id(&attr("a"));
        let b = g.get_or_create_node_attr_id(&attr("b"));
        // Then a relationship-only attribute.
        let since = g.get_or_create_rel_attr_id(&attr("since"));

        assert_eq!(a, 0);
        assert_eq!(b, 1);
        // The bug: this was 0 — the relationship table's first slot.
        assert_eq!(
            since, 2,
            "a relationship attribute must not restart numbering"
        );

        // Every accessor agrees, because there is only one dictionary to disagree about.
        for (name, expected) in [("a", 0usize), ("b", 1), ("since", 2)] {
            let n = attr(name);
            assert_eq!(g.get_node_attribute_id(&n), Some(expected));
            assert_eq!(g.get_relationship_attribute_id(&n), Some(expected));
            assert_eq!(g.get_global_attribute_id(&n), Some(expected));
        }

        // Re-registering under the other entity kind must not mint a second id.
        assert_eq!(g.get_or_create_rel_attr_id(&attr("a")), 0);
        assert_eq!(g.get_or_create_node_attr_id(&attr("since")), 2);
        assert_eq!(g.build_global_attrs().len(), 3);
    }

    /// The RDB's flat list and the live dictionary are the same numbering.
    ///
    /// A full sync seeds a replica from `build_global_attrs()`. If that list disagreed
    /// with the ids the master stamps into effects, the replica would misresolve them —
    /// which is exactly how #2457 manifested.
    #[test]
    fn rdb_attr_list_matches_live_ids() {
        ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, "attr_id_rdb");
        g.get_or_create_node_attr_id(&attr("a"));
        let since = g.get_or_create_rel_attr_id(&attr("since"));
        g.get_or_create_node_attr_id(&attr("b"));

        let list = g.build_global_attrs();

        // The load-bearing one: an id the master stamps into an effect has to index the
        // same name in the list a replica is seeded from. Split numbering broke exactly
        // this — `since` was relationship-local 0, and position 0 of the list is `a`.
        assert_eq!(
            list.get(since as usize).map(|s| s.as_str()),
            Some("since"),
            "the id put on the wire does not index its own name in the RDB list"
        );

        for (id, name) in list.iter().enumerate() {
            assert_eq!(
                g.get_global_attribute_id(name),
                Some(id),
                "RDB position {id} disagrees with the live id for {name}"
            );
        }
    }
}

#[cfg(test)]
mod reclaim_ids_tests {
    use super::*;

    /// The plain walk `reclaim_ids` has to stay equivalent to.
    fn walk(
        pool: &RoaringTreemap,
        base: u64,
        count: u64,
    ) -> Vec<u64> {
        pool.iter()
            .skip(base as usize)
            .take(count as usize)
            .collect()
    }

    fn reclaim(
        pool: &RoaringTreemap,
        base: u64,
        count: u64,
    ) -> Vec<u64> {
        let mut out: Vec<u64> = Vec::new();
        reclaim_ids(pool, base, count, &mut out);
        out
    }

    /// Reclaiming in batches must hand out the same ids as one big reclaim.
    /// This is the property the whole optimisation rests on.
    #[test]
    fn batched_reclaim_matches_single_walk() {
        let pool: RoaringTreemap = (0..5_000u64).map(|i| i * 3).collect();

        let mut batched: Vec<u64> = Vec::new();
        let mut base = 0;
        while base < pool.len() {
            let count = 97.min(pool.len() - base);
            reclaim_ids(&pool, base, count, &mut batched);
            base += count;
        }

        assert_eq!(batched, walk(&pool, 0, pool.len()));
    }

    /// Rank-based seeking has to agree with walking at *every* base, including
    /// across container boundaries — `select` sums container cardinalities, so a
    /// pool spread over several containers is where an off-by-one would show.
    #[test]
    fn agrees_with_walk_at_every_base() {
        let pool: RoaringTreemap = (0..300u64)
            .map(|i| i * 1_000)
            .chain(70_000..70_400)
            .chain(1_000_000..1_000_050)
            .collect();

        for base in 0..pool.len() {
            for count in [1u64, 13, 97] {
                assert_eq!(
                    reclaim(&pool, base, count),
                    walk(&pool, base, count),
                    "base {base}, count {count}"
                );
            }
        }
    }

    /// Ids freed below where reclaiming had reached shift every later position.
    /// Rank is read from the pool on each call, so the answer tracks the pool
    /// with no state to go stale — this is what the previous cursor-based
    /// version needed a cardinality guard to get right.
    #[test]
    fn reflects_a_pool_that_changed_between_calls() {
        let mut pool: RoaringTreemap = (100..200u64).collect();

        assert_eq!(reclaim(&pool, 0, 10), (100..110).collect::<Vec<_>>());

        // Free some lower ids, as returning a pending-created id does.
        pool.insert(0);
        pool.insert(1);

        assert_eq!(reclaim(&pool, 10, 10), walk(&pool, 10, 10));
    }

    /// Asking for more than the pool holds yields what there is, and a base past
    /// the end yields nothing rather than panicking.
    #[test]
    fn handles_requests_past_the_end() {
        let pool: RoaringTreemap = (0..5u64).collect();

        assert_eq!(reclaim(&pool, 0, 5), vec![0, 1, 2, 3, 4]);
        assert_eq!(reclaim(&pool, 0, 9), vec![0, 1, 2, 3, 4]);
        assert!(reclaim(&pool, 5, 3).is_empty());
        assert!(reclaim(&pool, 99, 3).is_empty());
    }

    /// A zero-count request appends nothing.
    #[test]
    fn zero_count_yields_nothing() {
        let pool: RoaringTreemap = (0..100u64).collect();
        assert!(reclaim(&pool, 0, 0).is_empty());
        assert!(reclaim(&pool, 50, 0).is_empty());
    }

    /// An empty pool has nothing to reclaim at any base.
    #[test]
    fn empty_pool_yields_nothing() {
        let pool = RoaringTreemap::new();
        assert!(reclaim(&pool, 0, 10).is_empty());
        assert!(reclaim(&pool, 7, 10).is_empty());
    }
}
