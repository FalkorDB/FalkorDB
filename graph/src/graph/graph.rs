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
//! │ all_nodes_matrix         │ Diagonal matrix: node_id -> bool         │
//! │                          │ (set for every live node)                │
//! ├──────────────────────────┼───────────────────────────────────────────┤
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
    sync::Arc,
    time::{Duration, Instant},
};

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use atomic_refcell::AtomicRefCell;
use lru::LruCache;
use orx_tree::DynTree;
use parking_lot::Mutex;
use roaring::RoaringTreemap;

use crate::{
    entity_type::EntityType,
    graph::{
        attribute_store::AttributeStore,
        constraint::{Constraint, ConstraintStatus, ConstraintType},
        graphblas::{
            matrix::{
                Descriptor, Dup, Get, MaskedElementWiseAdd, MaskedElementWiseMultiply, Matrix, MxM,
                New, Remove, Set, Size, Transpose,
            },
            serialization::{Encode, EncodeState, PayloadEntry, Writer},
            tensor::{Tensor, compound_key},
            versioned_matrix::{self, VersionedMatrix},
        },
    },
    index::{
        Field,
        indexer::{Document, IndexInfo, IndexOptions, IndexQuery, IndexType, Indexer},
    },
    parser::{ast::ExprIR, cypher::Parser},
    planner::{IR, Planner, binder::Binder, optimizer::optimize},
    runtime::{
        eval::evaluate_param, ordermap::OrderMap, orderset::OrderSet, value::Value, vec_distance,
    },
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
    /// Query parameters extracted from CYPHER prefix
    pub parameters: HashMap<String, DynTree<ExprIR<Arc<String>>>>,
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
pub struct TypeId(pub(crate) usize);

/// Opaque identifier for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NodeId(u64);

/// Opaque identifier for a relationship (edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct RelationshipId(u64);

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
        parameters: HashMap<String, DynTree<ExprIR<Arc<String>>>>,
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

/// Pre-built attribute snapshots for RDB save.
/// Built before Redis forks so the child never accesses fjall.
pub struct RdbSnapshots {
    pub nodes: FxHashMap<u64, Arc<Vec<(u16, Value)>>>,
    pub relationships: FxHashMap<u64, Arc<Vec<(u16, Value)>>>,
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
    zero_matrix: VersionedMatrix,
    /// Combined adjacency matrix (all relationship types)
    adjacancy_matrix: VersionedMatrix,
    /// Matrix mapping nodes to their labels
    node_labels_matrix: VersionedMatrix,
    /// Matrix mapping relationships to their types
    relationship_type_matrix: VersionedMatrix,
    /// Matrix with all nodes (for full scans)
    all_nodes_matrix: VersionedMatrix,
    /// Per-label matrices (label ID → node membership)
    labels_matices: Vec<VersionedMatrix>,
    /// Per-type relationship tensors (type ID → src×dst×edge_id)
    relationship_matrices: Vec<Tensor>,
    /// Graph-wide reverse index: edge_id → compound_key(src, dst) for O(1)
    /// endpoint lookup. Edge IDs are globally unique, so a single map covers
    /// all relationship types.
    edge_id_to_key: FxHashMap<u64, u64>,
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
    mut indexer: Indexer,
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

                let build_doc_with_fields =
                    |doc: &mut Document, id: u64, is_edge: bool, g: &Graph| -> bool {
                        let mut has_fields = false;
                        for (attr, fields) in &attrs {
                            let value = if is_edge {
                                g.get_relationship_attribute(RelationshipId(id), attr)
                            } else {
                                g.get_node_attribute(NodeId(id), attr)
                            };
                            if let Some(value) = value {
                                for field in fields {
                                    doc.set(field, &value);
                                }
                                has_fields = true;
                            }
                        }
                        has_fields
                    };

                // Advance `next_cursor` based on the *last scanned id*,
                // not the last emitted doc, so we don't get stuck when
                // most entities have no indexed attributes (few docs
                // produced) but there are still more to scan.
                match kind {
                    IndexKind::Node => {
                        let last_id = ids.last().copied();
                        for id in ids {
                            let mut doc = Document::new(id);
                            let g = graph.borrow();
                            if build_doc_with_fields(&mut doc, id, false, &g) {
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
                            let mut doc = Document::new_edge(src, dst, eid);
                            let g = graph.borrow();
                            if build_doc_with_fields(&mut doc, eid, true, &g) {
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
    mut node_indexer: Indexer,
) {
    spawn(
        move || {
            // Serialize with `populate_index_batch`, which holds the same
            // lock for the duration of a batch. Without this, the populate
            // worker can be mid-batch when we remove the label.
            let lock = node_indexer.write_lock();
            let _guard = lock.lock();
            node_indexer.remove(&label);
        },
        Some(0),
    );
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
            zero_matrix: VersionedMatrix::new(0, 0),
            adjacancy_matrix: VersionedMatrix::new(n, n),
            node_labels_matrix: VersionedMatrix::new(0, 0),
            relationship_type_matrix: VersionedMatrix::new(0, 0),
            all_nodes_matrix: VersionedMatrix::new(n, n),
            labels_matices: Vec::new(),
            relationship_matrices: Vec::new(),
            edge_id_to_key: FxHashMap::default(),
            node_attrs: AttributeStore::new(&format!("{name}/nodes"), version),
            relationship_attrs: AttributeStore::new(&format!("{name}/relationships"), version),
            node_indexer: Indexer::default(),
            edge_indexer: Indexer::default(),
            node_labels: Vec::new(),
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
        adjacancy_matrix: VersionedMatrix,
        node_labels_matrix: VersionedMatrix,
        relationship_type_matrix: VersionedMatrix,
        all_nodes_matrix: VersionedMatrix,
        labels_matices: Vec<VersionedMatrix>,
        relationship_matrices: Vec<Tensor>,
        node_labels: Vec<Arc<String>>,
        relationship_types: Vec<Arc<String>>,
        node_attrs: AttributeStore,
        relationship_attrs: AttributeStore,
    ) -> Self {
        // Rebuild the graph-wide reverse index after RDB load to ensure
        // complete sync with the decoded edges.
        let mut edge_id_to_key: FxHashMap<u64, u64> = FxHashMap::default();
        for tensor in &relationship_matrices {
            for (key, edge_id) in tensor.edge_iter(0, u64::MAX) {
                edge_id_to_key.insert(edge_id, key);
            }
        }

        let node_cap = node_count + deleted_nodes.len();
        let relationship_cap = relationship_count + deleted_relationships.len();
        let schema_version = (node_labels.len() + relationship_types.len()) as u64;
        Self {
            name: name.to_string(),
            node_cap: node_cap.next_power_of_two().max(64),
            relationship_cap: relationship_cap.next_power_of_two().max(64),
            reserved_node_count: 0,
            reserved_relationship_count: 0,
            node_count,
            relationship_count,
            deleted_nodes,
            deleted_relationships,
            zero_matrix: VersionedMatrix::new(0, 0),
            adjacancy_matrix,
            node_labels_matrix,
            relationship_type_matrix,
            all_nodes_matrix,
            labels_matices,
            relationship_matrices,
            edge_id_to_key,
            node_attrs,
            relationship_attrs,
            node_indexer: Indexer::default(),
            edge_indexer: Indexer::default(),
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
    /// - `all_nodes_matrix`: diagonal `(id, id) = true` for all live nodes
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

        // Rebuild all_nodes_matrix from all live node IDs (0..=max_id skipping deleted).
        // Cannot rebuild only from label matrices: unlabeled nodes do not appear in any
        // label matrix but still need to be in all_nodes_matrix for MATCH (n) scans.
        if self.node_count > 0 {
            let max_id = self.node_count + self.deleted_nodes.len() - 1;
            for id in 0..=max_id {
                if !self.deleted_nodes.contains(id) {
                    self.all_nodes_matrix.set(id, id, true);
                }
            }
        }

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

    #[must_use]
    pub fn new_version(&self) -> Self {
        debug_assert_eq!(self.reserved_node_count, 0);
        debug_assert_eq!(self.reserved_relationship_count, 0);
        let node_attrs = self.node_attrs.new_version(self.version + 1);
        let relationship_attrs = self.relationship_attrs.new_version(self.version + 1);

        // Tensor::dup() is copy-on-write; the graph-wide edge_id_to_key is
        // cloned once below.
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
            all_nodes_matrix: self.all_nodes_matrix.dup(),
            labels_matices: self
                .labels_matices
                .iter()
                .map(VersionedMatrix::dup)
                .collect(),
            relationship_matrices,
            edge_id_to_key: self.edge_id_to_key.clone(),
            node_attrs,
            relationship_attrs,
            node_indexer: self.node_indexer.clone(),
            edge_indexer: self.edge_indexer.clone(),
            node_labels: self.node_labels.clone(),
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

    pub const fn node_count(&self) -> u64 {
        self.node_count
    }

    /// Returns the number of nodes with the given label.
    #[must_use]
    pub fn label_node_count(
        &self,
        label: &str,
    ) -> u64 {
        self.get_label_matrix(label).map_or(0, Size::nvals)
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
    #[must_use]
    pub fn property_key_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for name in self.node_attrs.attrs_name.iter() {
            seen.insert(name.as_str());
        }
        for name in self.relationship_attrs.attrs_name.iter() {
            seen.insert(name.as_str());
        }
        seen.len()
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

    pub fn get_attrs(&self) -> impl Iterator<Item = &Arc<String>> + '_ {
        let mut seen = std::collections::HashSet::new();
        self.node_attrs
            .attrs_name
            .iter()
            .chain(self.relationship_attrs.attrs_name.iter())
            .filter(move |a| seen.insert(a.as_str().to_owned()))
    }

    pub fn get_label_id_mut(
        &mut self,
        label: &str,
    ) -> LabelId {
        if let Some(pos) = self
            .node_labels
            .iter()
            .position(|l| l.as_str() == label)
            .map(LabelId)
        {
            return pos;
        }

        self.node_labels.push(Arc::new(label.to_string()));
        self.labels_matices
            .push(VersionedMatrix::new(self.node_cap, self.node_cap));
        LabelId(self.node_labels.len() - 1)
    }

    pub fn get_label_id(
        &self,
        label: &str,
    ) -> Option<LabelId> {
        self.node_labels
            .iter()
            .position(|l| l.as_str() == label)
            .map(LabelId)
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

        // Evaluate parameter expressions to values for the optimizer.
        let param_values: HashMap<String, Value> = parameters
            .iter()
            .filter_map(|(k, v)| evaluate_param(&v.root()).ok().map(|val| (k.clone(), val)))
            .collect();

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

    pub fn get_label_matrix(
        &self,
        label: &str,
    ) -> Option<&VersionedMatrix> {
        self.node_labels
            .iter()
            .position(|l| l.as_str() == label)
            .map(|i| &self.labels_matices[i])
    }

    fn get_label_matrix_mut(
        &mut self,
        label: &Arc<String>,
    ) -> &mut VersionedMatrix {
        if !self.node_labels.contains(label) {
            self.node_labels.push(label.clone());

            let m = VersionedMatrix::new(self.node_cap, self.node_cap);
            self.labels_matices.insert(self.node_labels.len() - 1, m);
        }

        self.node_labels
            .iter()
            .position(|l| l.as_str() == label.as_str())
            .map(|i| &mut self.labels_matices[i])
            .expect("label was just inserted")
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
        }

        self.relationship_types
            .iter()
            .position(|l| l.as_str() == relationship_type.as_str())
            .map(|i| &mut self.relationship_matrices[i])
            .expect("relationship type was just inserted")
    }

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
        self.node_attrs.get_attr_id(attr)
    }

    #[must_use]
    pub const fn node_attribute_count(&self) -> usize {
        self.node_attrs.attrs_name.len()
    }

    #[must_use]
    pub fn get_relationship_attribute_id(
        &self,
        attr: &Arc<String>,
    ) -> Option<usize> {
        self.relationship_attrs.get_attr_id(attr)
    }

    /// Return the global property ID for `attr`, matching the index in `get_attrs()`.
    /// Node attrs come first; relationship-only attrs follow.
    #[must_use]
    pub fn get_global_attribute_id(
        &self,
        attr: &Arc<String>,
    ) -> Option<usize> {
        self.get_attrs().position(|a| a == attr)
    }

    /// Convert a relationship-local attribute ID to the global property ID.
    #[must_use]
    pub fn rel_attr_id_to_global(
        &self,
        local_id: u16,
    ) -> Option<usize> {
        let name = self.relationship_attrs.attrs_name.get(local_id as usize)?;
        self.get_attrs().position(|a| a == name)
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

    pub fn reserve_nodes(
        &mut self,
        count: usize,
    ) -> Vec<NodeId> {
        let count = count as u64;
        let mut ids = Vec::with_capacity(count as usize);
        let deleted_len = self.deleted_nodes.len();
        let available = deleted_len.saturating_sub(self.reserved_node_count);
        let reclaimed = count.min(available);

        // First reclaim from deleted nodes
        let base = self.reserved_node_count;
        self.reserved_node_count += reclaimed;
        for i in base..base + reclaimed {
            let id = self.deleted_nodes.select(i).unwrap();
            ids.push(NodeId(id));
        }

        // Allocate remaining from the end
        let remaining = count - reclaimed;
        let start = self.node_count + self.reserved_node_count;
        self.reserved_node_count += remaining;
        ids.extend((start..start + remaining).map(NodeId));

        ids
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
                while needed > self.node_cap {
                    self.node_cap *= 2;
                }
                self.resize_node_matrices();
            }
        }

        self.resize();

        self.all_nodes_matrix
            .set_all(nodes.iter().map(|id| (id, id)));
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

    pub fn set_nodes_attributes(
        &mut self,
        attrs: &FxHashMap<u64, OrderMap<Arc<String>, Value>>,
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> Result<(usize, usize), String> {
        let (nremoved, nset) = self.node_attrs.insert_attrs(attrs)?;

        if self.node_indexer.has_indices() {
            for (id, attrs) in attrs {
                for (_, label_id) in self.node_labels_matrix.iter(*id, *id) {
                    let label = &self.node_labels[label_id as usize];
                    for key in attrs.keys() {
                        if self.node_indexer.has_indexed_attr(label, key) {
                            index_add_docs.entry(label_id).or_default().insert(*id);
                        }
                    }
                }
            }
        }
        Ok((nremoved, nset))
    }

    pub fn import_node_attrs(
        &mut self,
        attrs: &FxHashMap<u64, OrderMap<Arc<String>, Value>>,
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> usize {
        let nset = self.node_attrs.import_attrs(attrs);

        if self.node_indexer.has_indices() {
            for (id, attrs) in attrs {
                for (_, label_id) in self.node_labels_matrix.iter(*id, *id) {
                    let label = &self.node_labels[label_id as usize];
                    for key in attrs.keys() {
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
    pub fn import_node_attrs_resolved(
        &mut self,
        data: &mut Vec<(u64, Vec<(u16, Value)>)>,
    ) -> usize {
        self.node_attrs.import_attrs_resolved(data)
    }

    /// Resolve a node attribute name to its index, creating if needed.
    pub fn get_or_create_node_attr_id(
        &mut self,
        attr: &Arc<String>,
    ) -> u16 {
        self.node_attrs.get_or_create_attr_id(attr)
    }

    /// Import pre-resolved relationship attributes directly into the cache.
    pub fn import_relationship_attrs_resolved(
        &mut self,
        data: &mut Vec<(u64, Vec<(u16, Value)>)>,
    ) -> usize {
        self.relationship_attrs.import_attrs_resolved(data)
    }

    /// Resolve a relationship attribute name to its index, creating if needed.
    pub fn get_or_create_rel_attr_id(
        &mut self,
        attr: &Arc<String>,
    ) -> u16 {
        self.relationship_attrs.get_or_create_attr_id(attr)
    }

    pub fn import_relationship_attrs(
        &mut self,
        attrs: &FxHashMap<u64, OrderMap<Arc<String>, Value>>,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> usize {
        let nset = self.relationship_attrs.import_attrs(attrs);
        self.track_edge_index_updates(attrs, index_add_edge_docs);
        nset
    }

    /// Mark every `(type_id, edge_id)` whose changed attributes are
    /// indexed so the next `commit_edge_index` pass rebuilds their
    /// documents. Shared by the import and set paths.
    fn track_edge_index_updates(
        &self,
        attrs: &FxHashMap<u64, OrderMap<Arc<String>, Value>>,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) {
        if !self.edge_indexer.has_indices() {
            return;
        }
        for (id, attrs) in attrs {
            let type_id = self.get_relationship_type_id(RelationshipId(*id));
            let type_name = &self.relationship_types[type_id.0];
            for key in attrs.keys() {
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
    pub fn set_nodes_labels_bulk(
        &mut self,
        label_rows: &[u64],
        label_cols: &[u64],
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
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

        self.node_labels_matrix
            .set_all(label_rows.iter().copied().zip(label_cols.iter().copied()));

        for (lid, ids) in by_label.into_iter().enumerate() {
            if !ids.is_empty() {
                self.labels_matices[lid].set_all(ids.iter().map(|&id| (id, id)));
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

    pub fn delete_nodes(
        &mut self,
        deleted_nodes: &RoaringTreemap,
        remove_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> Result<(), String> {
        self.deleted_nodes |= deleted_nodes;
        self.node_count -= deleted_nodes.len();

        // Build a diagonal mask matrix from all deleted node IDs
        let n = self.node_cap;
        let mut diag_mask = Matrix::new(n, n);
        for id in deleted_nodes {
            diag_mask.set(id, id, true);
        }

        // Bulk-remove from all_nodes_matrix
        self.all_nodes_matrix.remove_mask(&diag_mask);

        // Build per-label masks and nlm_mask using a single scan of the
        // node_labels_matrix instead of one iterator per deleted node.
        let num_labels = self.labels_matices.len();
        let mut label_masks: Vec<Option<Matrix>> = vec![None; num_labels];
        let mut nlm_mask = Matrix::new(
            self.node_labels_matrix.nrows().max(1),
            self.node_labels_matrix
                .ncols()
                .max(num_labels as u64)
                .max(1),
        );

        // Single scan: iterate all entries in node_labels_matrix and filter
        // by deleted_nodes membership (O(1) bitmap check per entry).
        for (node_id, label_id) in self.node_labels_matrix.iter(0, n) {
            if !deleted_nodes.contains(node_id) {
                continue;
            }
            let lid = label_id as usize;
            let lm = label_masks[lid].get_or_insert_with(|| Matrix::new(n, n));
            lm.set(node_id, node_id, true);

            let label = &self.node_labels[lid];
            if self.node_indexer.has_index(label) {
                for attr in self.node_attrs.get_attrs(node_id) {
                    if self.node_indexer.has_indexed_attr(label, &attr) {
                        remove_docs.entry(label_id).or_default().insert(node_id);
                        break;
                    }
                }
            }

            nlm_mask.set(node_id, label_id, true);
        }

        // Bulk-remove from per-label matrices
        for (lid, mask_opt) in label_masks.into_iter().enumerate() {
            if let Some(mask) = mask_opt {
                self.labels_matices[lid].remove_mask(&mask);
            }
        }

        // Bulk-remove from node_labels_matrix
        if nlm_mask.nvals() > 0 {
            self.node_labels_matrix.remove_mask(&nlm_mask);
        }

        self.node_attrs.remove_all(deleted_nodes);
        Ok(())
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
    pub fn get_node_relationships_by_type(
        &self,
        id: NodeId,
        types: &[Arc<String>],
    ) -> impl Iterator<Item = (NodeId, NodeId, RelationshipId)> + '_ {
        let matrices: Vec<&Tensor> = if types.is_empty() {
            self.relationship_matrices.iter().collect()
        } else {
            types
                .iter()
                .filter_map(|t| self.get_relationship_matrix(t))
                .collect()
        };
        matrices
            .into_iter()
            .flat_map(move |m| {
                m.iter(id.0, id.0, false).chain(
                    m.iter(id.0, id.0, true)
                        .filter(move |(src, _, _)| *src != id.0),
                )
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
            .flat_map(move |m| m.iter(id.0, id.0, true))
            .count()
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
            .flat_map(|m| m.iter(id.0, id.0, true))
            .count()
    }

    /// Count the number of outgoing edges from a node.
    #[must_use]
    pub fn get_node_outdegree(
        &self,
        id: NodeId,
    ) -> usize {
        self.relationship_matrices
            .iter()
            .flat_map(move |m| m.iter(id.0, id.0, false))
            .count()
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
            .flat_map(|m| m.iter(id.0, id.0, false))
            .count()
    }

    #[must_use]
    pub fn get_nodes(
        &self,
        labels: &OrderSet<Arc<String>>,
        min_row: u64,
    ) -> Box<dyn Iterator<Item = NodeId>> {
        if labels.is_empty() {
            // Full scan: live node IDs are exactly `0..=max_node_id` minus the
            // deleted set, identical to the diagonal of `all_nodes_matrix`.
            // A range walk with a roaring-bitmap membership check avoids the
            // per-element GraphBLAS row-iterator overhead, which dominates the
            // cost of unfiltered `MATCH (n)` scans.
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
                    || self.zero_matrix.to_matrix().iter(min_row, u64::MAX),
                    |mut matrices| {
                        let mut iter = matrices.iter_mut();
                        let mut m = iter.next().unwrap().to_matrix();
                        for label_matrix in iter {
                            m.element_wise_multiply(
                                None,
                                None,
                                Some(&label_matrix.to_matrix()),
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
        self.node_attrs.get_attr(id.0, attr)
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

    pub fn reserve_relationships(
        &mut self,
        count: usize,
    ) -> Vec<RelationshipId> {
        let count = count as u64;
        let mut ids = Vec::with_capacity(count as usize);
        let deleted_len = self.deleted_relationships.len();
        let available = deleted_len.saturating_sub(self.reserved_relationship_count);
        let reclaimed = count.min(available);

        // First reclaim from deleted relationships
        let base = self.reserved_relationship_count;
        self.reserved_relationship_count += reclaimed;
        for i in base..base + reclaimed {
            let id = self.deleted_relationships.select(i).unwrap();
            ids.push(RelationshipId(id));
        }

        // Allocate remaining from the end
        let remaining = count - reclaimed;
        let start = self.relationship_count + self.reserved_relationship_count;
        self.reserved_relationship_count += remaining;
        ids.extend((start..start + remaining).map(RelationshipId));

        ids
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
                while needed > self.relationship_cap {
                    self.relationship_cap *= 2;
                }
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
        self.edge_id_to_key.reserve(rel_ids.len());
        for ((&src, &dst), &id) in srcs.iter().zip(dsts.iter()).zip(rel_ids.iter()) {
            self.edge_id_to_key.insert(id, compound_key(src, dst));
        }

        self.adjacancy_matrix
            .set_all(srcs.iter().copied().zip(dsts.iter().copied()));

        let type_id = type_idx as u64;
        let type_ids: Vec<u64> = vec![type_id; rel_ids.len()];
        self.relationship_type_matrix
            .set_all(rel_ids.iter().copied().zip(type_ids.iter().copied()));
    }

    /// Flush delta-plus into base for all shared matrices.
    /// Reduces dp accumulation across multiple GRAPH.BULK commands.
    pub fn flush_for_bulk(&mut self) {
        self.all_nodes_matrix.flush();
        self.node_labels_matrix.flush();
        for m in &mut self.labels_matices {
            m.flush();
        }
        self.adjacancy_matrix.flush();
        self.relationship_type_matrix.flush();
    }

    /// Materialize all pending GraphBLAS operations on every matrix.
    /// Called from pthread_atfork prepare handler to ensure no internal
    /// GraphBLAS locks are held at fork time.
    pub fn wait_all(&self) {
        self.zero_matrix.wait_all();
        self.adjacancy_matrix.wait_all();
        self.node_labels_matrix.wait_all();
        self.relationship_type_matrix.wait_all();
        self.all_nodes_matrix.wait_all();
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
            && self.all_nodes_matrix.is_synced()
            && self.labels_matices.iter().all(VersionedMatrix::is_synced)
            && self.relationship_matrices.iter().all(Tensor::is_synced)
    }

    pub fn set_relationships_attributes(
        &mut self,
        attrs: &FxHashMap<u64, OrderMap<Arc<String>, Value>>,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) -> Result<(usize, usize), String> {
        let (nremoved, nset) = self.relationship_attrs.insert_attrs(attrs)?;
        self.track_edge_index_updates(attrs, index_add_edge_docs);
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
    pub fn label_matrices(&self) -> &[VersionedMatrix] {
        &self.labels_matices
    }

    #[must_use]
    pub const fn adjacency_matrix(&self) -> &VersionedMatrix {
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
    ) -> Result<Vec<(RelationshipId, NodeId, NodeId)>, String> {
        let del_keys = rels;

        let min_id = del_keys.min().unwrap_or(0);
        let max_id = del_keys.max().unwrap_or(0);
        let num_types = self.relationship_matrices.len();
        let mut by_type: Vec<Vec<(u64, u64, u64)>> = vec![Vec::new(); num_types];

        // Resolve endpoints BEFORE mutating any graph state, so a stale or
        // non-existent edge id can't corrupt the deleted bitmap / counters.
        // Build by_type using the graph-wide reverse index (edge_id_to_key) for
        // O(1) endpoint lookup. The me matrix rows are compound_keys
        // (src<<32|dst), not edge IDs, so an edge-ID range scan can't be used.
        #[allow(clippy::cast_possible_truncation)]
        for (edge_id, type_idx) in self.relationship_type_matrix.iter(min_id, max_id) {
            if del_keys.contains(edge_id) {
                if let Some((src, dst)) = self.endpoints_for_edge(edge_id) {
                    by_type[type_idx as usize].push((edge_id, src, dst));
                }
            }
        }

        // Endpoints resolved — now apply state mutations.
        self.deleted_relationships.extend(rels.iter());
        self.relationship_count -= rels.len();
        self.relationship_attrs.remove_all(del_keys);

        let mut endpoints: Vec<(RelationshipId, NodeId, NodeId)> =
            Vec::with_capacity(rels.len() as usize);

        // Track (src, dst) pairs that were emptied from at least one tensor —
        // only these are candidates for adjacency matrix removal.
        let mut adj_candidates: Vec<(u64, u64)> = Vec::new();

        for (type_idx, type_rels) in by_type.iter().enumerate() {
            if type_rels.is_empty() {
                continue;
            }

            // Stage index document removals for indexed relationship types
            let type_id = type_idx as u64;
            let type_name = &self.relationship_types[type_idx];
            if self.edge_indexer.has_index(type_name) {
                for &(edge_id, src, dst) in type_rels {
                    index_remove_edge_docs
                        .entry(type_id)
                        .or_default()
                        .insert(edge_id, (src, dst));
                }
            }

            // Batch remove from relationship_type_matrix using bulk build_bool
            let tm_rows: Vec<u64> = type_rels.iter().map(|&(id, _, _)| id).collect();
            let tm_cols: Vec<u64> = vec![type_id; type_rels.len()];
            let mut type_mask =
                Matrix::new(self.relationship_cap, self.relationship_types.len() as u64);
            type_mask.build_bool(&tm_rows, &tm_cols);
            self.relationship_type_matrix.remove_mask(&type_mask);

            for &(edge_id, src, dst) in type_rels {
                endpoints.push((RelationshipId(edge_id), NodeId(src), NodeId(dst)));
                self.edge_id_to_key.remove(&edge_id);
            }
            let emptied = self.relationship_matrices[type_idx].remove_all(type_rels);
            adj_candidates.extend(emptied);
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
                let mut adj_mask = Matrix::new(node_cap, node_cap);
                adj_mask.build_bool(&adj_rows, &adj_cols);
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
    ) -> Result<Vec<(RelationshipId, NodeId, NodeId)>, String> {
        if self.relationship_matrices.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_implicit: Vec<(RelationshipId, NodeId, NodeId)> = Vec::new();
        // Pairs where only one endpoint is deleted — need adjacency check
        let mut check_adj_pairs: std::collections::HashSet<(u64, u64)> =
            std::collections::HashSet::default();

        for type_idx in 0..self.relationship_matrices.len() {
            let mut rels: Vec<(u64, u64, u64)> = Vec::new();

            // Collect all edges for deleted nodes from this tensor
            for node_id in deleted_nodes {
                // Outgoing edges
                for (src, dst, edge_id) in
                    self.relationship_matrices[type_idx].iter(node_id, node_id, false)
                {
                    if !explicit_rels.contains(edge_id) {
                        rels.push((edge_id, src, dst));
                    }
                }
                // Incoming edges — skip if source is also a deleted node
                // (those edges are already collected from the source's
                // outgoing iteration), and skip self-loops already found above.
                for (src, dst, edge_id) in
                    self.relationship_matrices[type_idx].iter(node_id, node_id, true)
                {
                    if src != node_id
                        && !deleted_nodes.contains(src)
                        && !explicit_rels.contains(edge_id)
                    {
                        rels.push((edge_id, src, dst));
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
                Matrix::new(self.relationship_cap, self.relationship_types.len() as u64);
            type_mask.build_bool(&tm_rows, &tm_cols);

            let del_keys: RoaringTreemap = rels.iter().map(|&(id, _, _)| id).collect();
            self.deleted_relationships |= &del_keys;
            let type_name = &self.relationship_types[type_idx];
            let is_indexed = self.edge_indexer.has_index(type_name);
            for &(edge_id, src, dst) in &rels {
                all_implicit.push((RelationshipId(edge_id), NodeId(src), NodeId(dst)));

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
                self.edge_id_to_key.remove(&edge_id);
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
        let mut adj_mask = Matrix::new(self.node_cap, self.node_cap);
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

        iters
            .into_iter()
            .flat_map(|iter| iter.map(|(_, id)| RelationshipId(id)))
    }

    /// Build a relationship matrix summing only the given types (no
    /// source/destination label restriction). Returns `None` when `types` is
    /// non-empty but none of the types exist in the schema (caller should
    /// short-circuit to an empty result).
    pub fn build_relationship_matrix_unrestricted(
        &self,
        types: &[Arc<String>],
    ) -> Option<Matrix> {
        let matrices = types
            .iter()
            .filter_map(|relationship_type| self.get_relationship_matrix(relationship_type))
            .collect::<Vec<_>>();
        if !types.is_empty() && matrices.is_empty() {
            return None;
        }
        let mut iter = matrices.into_iter();
        let mut m = iter.next().map_or_else(
            || self.adjacancy_matrix.to_matrix(),
            |t| t.matrix().to_matrix(),
        );
        for relationship_matrix in iter {
            m.element_wise_add(
                Some(relationship_matrix.matrix().dm()),
                None,
                Some(relationship_matrix.matrix().m()),
                Some(Descriptor::C),
            );
            m.element_wise_add(None, None, Some(relationship_matrix.matrix().dp()), None);
        }
        Some(m)
    }

    /// Resolve a set of label names to ids. Returns `None` if any label is not
    /// in the schema (which means no node could match).
    pub fn resolve_label_ids(
        &self,
        labels: &OrderSet<Arc<String>>,
    ) -> Option<Vec<LabelId>> {
        labels.iter().map(|l| self.get_label_id(l)).collect()
    }

    /// Build a relationship matrix combining the given types and filtering by
    /// source/destination labels.
    pub fn build_relationship_matrix(
        &self,
        types: &[Arc<String>],
        src_labels: &OrderSet<Arc<String>>,
        dest_labels: &OrderSet<Arc<String>>,
    ) -> Matrix {
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
            self.zero_matrix.to_matrix()
        } else {
            let mut iter = matrices.into_iter();
            let mut m = iter.next().map_or_else(
                || self.adjacancy_matrix.to_matrix(),
                |t| t.matrix().to_matrix(),
            );
            for relationship_matrix in iter {
                m.element_wise_add(
                    None,
                    None,
                    Some(&relationship_matrix.matrix().to_matrix()),
                    None,
                );
            }

            if !src_labels_matrices.is_empty() {
                let mut iter = src_labels_matrices.iter();
                let mut src_matrix = iter.next().unwrap().to_matrix();
                for label_matrix in iter {
                    src_matrix.element_wise_multiply(
                        None,
                        None,
                        Some(&label_matrix.to_matrix()),
                        None,
                    );
                }
                m.rmxm(&src_matrix);
            }
            if !dest_labels_matrices.is_empty() {
                let mut iter = dest_labels_matrices.iter();
                let mut dest_matrix = iter.next().unwrap().to_matrix();
                for label_matrix in iter {
                    dest_matrix.element_wise_multiply(
                        None,
                        None,
                        Some(&label_matrix.to_matrix()),
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
        #[allow(clippy::cast_possible_truncation)]
        self.relationship_type_matrix
            .iter(id.0, id.0)
            .map(|(_, l)| TypeId(l as usize))
            .next()
            .expect("relationship must have a type in type_matrix")
    }

    /// Decode the (src, dst) endpoints for an edge from the graph-wide reverse
    /// index. Returns None if the edge_id is not present.
    #[must_use]
    fn endpoints_for_edge(
        &self,
        edge_id: u64,
    ) -> Option<(u64, u64)> {
        self.edge_id_to_key
            .get(&edge_id)
            .map(|&key| (key >> 32, key & 0xFFFF_FFFF))
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
        self.relationship_attrs.get_attr(id.0, attr)
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
        self.all_nodes_matrix.resize(self.node_cap, self.node_cap);
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
            while self.node_count > self.node_cap {
                self.node_cap *= 2;
            }
            self.resize_node_matrices();
        }

        if self.labels_matices.len() as u64 > self.node_labels_matrix.ncols() {
            self.node_labels_matrix
                .resize(self.node_cap, self.labels_matices.len() as u64);
        }

        if self.relationship_count > self.relationship_cap {
            while self.relationship_count > self.relationship_cap {
                self.relationship_cap *= 2;
            }
            self.resize_relationship_matrices();
        }

        if self.relationship_types.len() as u64 > self.relationship_type_matrix.ncols() {
            self.relationship_type_matrix
                .resize(self.relationship_cap, self.relationship_types.len() as u64);
        }
    }

    pub fn get_node_attrs(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = Arc<String>> + '_ {
        self.node_attrs.get_attrs(id.0)
    }

    /// Get all attribute names and values for a node in a single storage pass.
    pub fn get_node_all_attrs(
        &self,
        id: NodeId,
    ) -> Vec<(Arc<String>, Value)> {
        self.node_attrs.get_all_attrs(id.0)
    }

    pub fn get_node_all_attrs_by_id(
        &self,
        id: NodeId,
    ) -> Arc<Vec<(u16, Value)>> {
        self.node_attrs.get_all_attrs_by_id(id.0)
    }

    pub fn get_relationship_attrs(
        &self,
        id: RelationshipId,
    ) -> impl Iterator<Item = Arc<String>> + '_ {
        self.relationship_attrs.get_attrs(id.0)
    }

    /// Get all attribute names and values for a relationship in a single storage pass.
    pub fn get_relationship_all_attrs(
        &self,
        id: RelationshipId,
    ) -> Vec<(Arc<String>, Value)> {
        self.relationship_attrs.get_all_attrs(id.0)
    }

    pub fn get_relationship_all_attrs_by_id(
        &self,
        id: RelationshipId,
    ) -> Arc<Vec<(u16, Value)>> {
        self.relationship_attrs.get_all_attrs_by_id(id.0)
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
                    let mut doc = Document::new(n);
                    let mut has_fields = false;
                    for (attr_idx, fields) in &resolved_attrs {
                        let value = self.get_node_attribute_by_idx(NodeId(n), *attr_idx);
                        if let Some(value) = value {
                            for field in fields {
                                doc.set(field, &value);
                            }
                            has_fields = true;
                        }
                    }
                    if has_fields {
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
                    let mut doc = Document::new_edge(src, dst, eid);
                    let mut has_fields = false;
                    for (attr, fields) in &attrs {
                        if let Some(value) = self.relationship_attrs.get_attr(eid, attr) {
                            for field in fields {
                                doc.set(field, &value);
                            }
                            has_fields = true;
                        }
                    }
                    if has_fields {
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

    pub fn commit_attrs(&mut self) -> Result<(), String> {
        self.node_attrs.commit()?;
        self.relationship_attrs.commit()?;
        Ok(())
    }

    /// Invalidate dirty cache entries written during a failed write transaction.
    pub fn rollback_cache(&mut self) {
        self.node_attrs.rollback_cache();
        self.relationship_attrs.rollback_cache();
    }

    /// Drop rollback-saved state after a query commits successfully.
    pub fn clear_rollback_state(&mut self) {
        self.node_attrs.clear_rollback_state();
        self.relationship_attrs.clear_rollback_state();
    }

    /// Flush dirty cache entries to fjall and evict clean entries if over budget.
    pub fn maybe_flush_caches(&self) -> Result<(), String> {
        const FLUSH_BATCH: usize = 1024;
        if self.node_attrs.cache().over_budget() {
            self.node_attrs.flush_dirty_to_fjall(FLUSH_BATCH)?;
        }
        if self.relationship_attrs.cache().over_budget() {
            self.relationship_attrs.flush_dirty_to_fjall(FLUSH_BATCH)?;
        }
        Ok(())
    }

    /// Returns `true` if any attribute store has cold data in fjall that
    /// would be unsafe to read from a fork child.
    pub fn needs_rdb_snapshot(&self) -> bool {
        self.node_attrs.has_fjall_data() || self.relationship_attrs.has_fjall_data()
    }

    /// Pre-populate attribute caches from fjall for RDB save.
    pub fn build_rdb_snapshots(&self) -> RdbSnapshots {
        let node_snap = self
            .node_attrs
            .build_rdb_snapshot(&self.deleted_nodes, self.max_node_id());
        let rel_snap = self
            .relationship_attrs
            .build_rdb_snapshot(&self.deleted_relationships, self.max_relationship_id());
        RdbSnapshots {
            nodes: node_snap,
            relationships: rel_snap,
        }
    }

    pub fn commit_index(
        &mut self,
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
        remove_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) {
        self.commit_index_kind(IndexKind::Node, index_add_docs, remove_docs);
    }

    pub fn commit_edge_index(
        &mut self,
        index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
        remove_edge_docs: &mut FxHashMap<u64, FxHashMap<u64, (u64, u64)>>,
    ) {
        if index_add_edge_docs.is_empty() && remove_edge_docs.is_empty() {
            return;
        }

        let indexer = &mut self.edge_indexer;
        let lock = indexer.write_lock();
        let _guard = lock.lock();

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
                    if let Some(value) = self.relationship_attrs.get_attr(id, key) {
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
        kind: IndexKind,
        index_add_docs: &mut FxHashMap<u64, RoaringTreemap>,
        remove_docs: &mut FxHashMap<u64, RoaringTreemap>,
    ) {
        if index_add_docs.is_empty() && remove_docs.is_empty() {
            return;
        }

        let (indexer, names, attr_store) = match kind {
            IndexKind::Node => (&mut self.node_indexer, &self.node_labels, &self.node_attrs),
            IndexKind::Edge => unreachable!("use commit_edge_index for edges"),
        };

        let lock = indexer.write_lock();
        let _guard = lock.lock();

        let mut add_docs = HashMap::new();
        for (slot, ids) in index_add_docs.drain() {
            let name = &names[slot as usize];
            let fields = indexer.get_fields(name);
            let mut docs = vec![];
            for id in ids {
                let mut doc = Document::new(id);
                for (key, fields) in &fields {
                    if let Some(value) = attr_store.get_attr(id, key) {
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
                let total = self
                    .get_label_matrix(label)
                    .map_or(0, super::graphblas::matrix::Size::nvals);
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
                    let attrs = self.get_node_all_attrs(NodeId(node_id));
                    for prop in &constraint.properties {
                        if !attrs
                            .iter()
                            .any(|(name, val)| name == prop && !matches!(val, Value::Null))
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
                    let attrs = self.get_relationship_all_attrs(RelationshipId(edge_id));
                    for prop in &constraint.properties {
                        if !attrs
                            .iter()
                            .any(|(name, val)| name == prop && !matches!(val, Value::Null))
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
                    let attrs = self.get_node_all_attrs(NodeId(node_id));
                    let key = Self::build_composite_key(&constraint.properties, &attrs);
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
                    let attrs = self.get_relationship_all_attrs(RelationshipId(edge_id));
                    let key = Self::build_composite_key(&constraint.properties, &attrs);
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
        attrs: &[(Arc<String>, Value)],
    ) -> Vec<u8> {
        let mut all_null = true;
        let mut key = Vec::new();
        for prop in properties {
            let value = attrs.iter().find(|(name, _)| name == prop).map(|(_, v)| v);
            match value {
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

    /// Delete fjall keyspaces for both node and relationship attribute stores.
    /// Called during graph destruction to release persisted attribute data.
    pub fn delete_keyspaces(&self) {
        self.node_attrs.delete_keyspace();
        self.relationship_attrs.delete_keyspace();
    }

    pub fn set_indexer_graph(
        &mut self,
        graph: Arc<AtomicRefCell<Self>>,
    ) {
        self.node_indexer.set_graph(graph.clone());
        self.edge_indexer.set_graph(graph);
    }

    /// Build a materialized boolean adjacency matrix filtered by relationship types.
    /// If `rel_types` is empty, returns the full adjacency matrix.
    /// The caller owns the returned `Matrix`.
    pub fn build_adjacency_matrix(
        &self,
        rel_types: &[Arc<String>],
    ) -> Matrix {
        if rel_types.is_empty() {
            self.adjacancy_matrix.to_matrix()
        } else {
            let mut result = Matrix::new(self.node_cap, self.node_cap);
            for rel_type in rel_types {
                if let Some(type_id) = self.get_type_id(rel_type) {
                    let m = self.relationship_matrices[usize::from(type_id)]
                        .matrix()
                        .to_matrix();
                    result.element_wise_add(None, None, Some(&m), None);
                }
            }
            result
        }
    }

    /// Build a materialized boolean adjacency matrix that is symmetric (A + A^T).
    /// Used for undirected graph algorithms (WCC, CDLP, MSF).
    pub fn build_symmetric_adjacency_matrix(
        &self,
        rel_types: &[Arc<String>],
    ) -> Matrix {
        let a = self.build_adjacency_matrix(rel_types);
        let at = a.transpose();
        let mut result = Matrix::new(self.node_cap, self.node_cap);
        result.element_wise_add(None, Some(&a), Some(&at), None);
        result
    }

    /// Build a diagonal boolean matrix of nodes matching any of the given labels.
    /// If `labels` is empty, returns the all_nodes matrix.
    pub fn build_node_mask_matrix(
        &self,
        labels: &[Arc<String>],
    ) -> Matrix {
        if labels.is_empty() {
            self.all_nodes_matrix.to_matrix()
        } else {
            let mut result = Matrix::new(self.node_cap, self.node_cap);
            for label in labels {
                if let Some(label_id) = self.get_label_id(label) {
                    let m = self.labels_matices[usize::from(label_id)].to_matrix();
                    result.element_wise_add(None, None, Some(&m), None);
                }
            }
            result
        }
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let mut size = 0usize;
        size += self.adjacancy_matrix.memory_usage();
        size += self.node_labels_matrix.memory_usage();
        size += self.relationship_type_matrix.memory_usage();
        size += self.all_nodes_matrix.memory_usage();
        for label_matrix in &self.labels_matices {
            size += label_matrix.memory_usage();
        }
        for relationship_matrix in &self.relationship_matrices {
            size += relationship_matrix.memory_usage();
        }
        size += self.node_attrs.memory_usage();
        // Graph-wide edge_id → compound_key reverse index: one (u64, u64) entry
        // per relationship, plus one SwissTable control byte per slot.
        size += self.edge_id_to_key.capacity() * (std::mem::size_of::<(u64, u64)>() + 1);
        // size += self.relationship_attrs.memory_usage();
        // size += self.node_indexer.memory_usage();
        size
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
        let mut relation_matrices_sz: usize = 0;
        for rm in &self.relationship_matrices {
            relation_matrices_sz += rm.memory_usage();
        }

        // --- node block storage ---
        let node_block_storage_sz: usize =
            self.node_attrs.structural_memory_usage() + self.deleted_nodes.serialized_size();

        // --- edge block storage ---
        // Includes the graph-wide edge_id → compound_key reverse index, which
        // holds one (u64, u64) entry per relationship (plus a SwissTable
        // control byte per slot).
        let edge_block_storage_sz: usize = self.relationship_attrs.structural_memory_usage()
            + self.deleted_relationships.serialized_size()
            + self.edge_id_to_key.capacity() * (std::mem::size_of::<(u64, u64)>() + 1);

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
            // Sample unlabeled nodes from all_nodes_matrix, skipping labeled ones.
            let mut sampled_mem: usize = 0;
            let mut sampled_count: usize = 0;
            for (node_id, _) in self.all_nodes_matrix.iter(0, u64::MAX) {
                if sampled_count >= samples {
                    break;
                }
                if seen[node_id as usize] {
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
        let mut sz: usize = 0;
        for (_, val) in store.get_all_attrs_by_id(entity_id).iter() {
            sz += std::mem::size_of::<u16>() + std::mem::size_of::<Value>() + val.heap_size();
        }
        sz
    }

    /// Encode a single payload entry.
    pub fn encode_payload(
        &self,
        w: &mut dyn Writer,
        p: &PayloadEntry,
        global_attrs: &[Arc<String>],
        snapshots: Option<&RdbSnapshots>,
    ) {
        match p.state {
            EncodeState::Nodes => {
                self.node_attrs.encode_with_range(
                    w,
                    &self.deleted_nodes,
                    self.max_node_id(),
                    global_attrs,
                    p.count,
                    p.offset,
                    snapshots.map(|s| &s.nodes),
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
                    global_attrs,
                    p.count,
                    p.offset,
                    snapshots.map(|s| &s.relationships),
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
    pub fn get_node_attribute_names(&self) -> Vec<Arc<String>> {
        self.node_attrs.attrs_name.iter().cloned().collect()
    }

    /// Get relationship attribute names.
    pub fn get_relationship_attribute_names(&self) -> Vec<Arc<String>> {
        self.relationship_attrs.attrs_name.iter().cloned().collect()
    }

    /// Register a node attribute name (get-or-create). Used by effect
    /// replication to pre-register attribute names on the replica.
    pub fn add_node_attribute_name(
        &mut self,
        name: &str,
    ) {
        let arc = Arc::new(name.to_string());
        if self.node_attrs.attrs_name.get_index_of(&arc).is_none() {
            self.node_attrs.attrs_name.insert(arc);
        }
    }

    /// Register a relationship attribute name (get-or-create). Used by effect
    /// replication to pre-register attribute names on the replica.
    pub fn add_rel_attribute_name(
        &mut self,
        name: &str,
    ) {
        let arc = Arc::new(name.to_string());
        if self
            .relationship_attrs
            .attrs_name
            .get_index_of(&arc)
            .is_none()
        {
            self.relationship_attrs.attrs_name.insert(arc);
        }
    }

    /// Build the unified global attribute list (node attrs ∪ relationship attrs, in order).
    pub fn build_global_attrs(&self) -> Vec<Arc<String>> {
        let mut attrs = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for name in self.node_attrs.attrs_name.iter() {
            if seen.insert(name.clone()) {
                attrs.push(name.clone());
            }
        }
        for name in self.relationship_attrs.attrs_name.iter() {
            if seen.insert(name.clone()) {
                attrs.push(name.clone());
            }
        }
        attrs
    }
}
