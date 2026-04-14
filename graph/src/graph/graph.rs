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

use atomic_refcell::AtomicRefCell;
use itertools::Itertools;
use lru::LruCache;
use orx_tree::DynTree;
use parking_lot::Mutex;
use roaring::RoaringTreemap;

use crate::{
    entity_type::EntityType,
    graph::{
        attribute_store::AttributeStore,
        graphblas::{
            matrix::{
                Dup, MaskedElementWiseAdd, MaskedElementWiseMultiply, Matrix, MxM, New, Remove,
                Set, Size, Transpose,
            },
            serialization::{Encode, EncodeState, PayloadEntry, Writer},
            tensor::Tensor,
            versioned_matrix::VersionedMatrix,
        },
    },
    index::{
        Field,
        indexer::{Document, IndexInfo, IndexOptions, IndexQuery, IndexType, Indexer},
    },
    parser::{ast::ExprIR, cypher::Parser},
    planner::{IR, Planner, binder::Binder, optimizer::optimize},
    runtime::{ordermap::OrderMap, orderset::OrderSet, pending::PendingRelationship, value::Value},
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
}

/// Opaque identifier for a node label.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LabelId(pub usize);

/// Opaque identifier for a relationship type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeId(usize);

/// Opaque identifier for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u64);

/// Opaque identifier for a relationship (edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    ) -> Self {
        Self {
            plan,
            cached,
            parameters,
            parse_duration,
            plan_duration,
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
    /// Node property storage
    node_attrs: AttributeStore,
    /// Relationship property storage
    relationship_attrs: AttributeStore,
    /// Index manager for property indexes
    node_indexer: Indexer,
    /// Label names (ID → name mapping)
    node_labels: Vec<Arc<String>>,
    /// Relationship type names (ID → name mapping)
    relationship_types: Vec<Arc<String>>,
    /// LRU cache for query plans
    cache: Arc<Mutex<LruCache<String, PlanTree>>>,
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
fn populate_index(
    label: Arc<String>,
    node_indexer: Indexer,
) {
    let attrs = node_indexer.get_fields(&label);
    populate_index_batch(label, node_indexer, attrs, 0, 0);
}

/// Processes one batch of index population and spawns the next batch.
fn populate_index_batch(
    label: Arc<String>,
    mut node_indexer: Indexer,
    attrs: HashMap<Arc<String>, Vec<Arc<Field>>>,
    mut progress: u64,
    min_row: u64,
) {
    spawn(
        move || {
            const BATCH_SIZE: usize = 10_000;

            if node_indexer.is_cancelled() || node_indexer.pending_changes(&label) > 1 {
                node_indexer.enable(&label);
                return;
            }

            let exhausted;
            let mut next_min_row = min_row;

            // Hold the Indexer's serialization lock for the entire batch so
            // that write-path `commit_index` calls wait until this batch
            // finishes.  This guarantees no concurrent index mutations.
            {
                let lock = node_indexer.write_lock();
                let guard = lock.lock();

                let mut batch = Vec::with_capacity(BATCH_SIZE);

                if let Some(graph) = node_indexer.get_graph()
                    && let Some(lm) = graph.borrow().get_label_matrix(&label)
                {
                    for (n, _) in lm.iter(min_row, u64::MAX).take(BATCH_SIZE) {
                        let mut doc = Document::new(n);
                        let mut has_fields = false;
                        for (attr, fields) in &attrs {
                            let value = graph.borrow().get_node_attribute(NodeId(n), attr);
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
                    next_min_row = batch.last().map_or(next_min_row, |doc| doc.id() + 1);
                } else {
                    // Graph not yet committed — reschedule this batch.
                    // MvccGraph::commit() will set the indexer's graph
                    // reference, so the next attempt will find it.
                    drop(guard);
                    drop(lock);
                    std::thread::sleep(Duration::from_millis(1));
                    populate_index_batch(label, node_indexer, attrs, progress, min_row);
                    return;
                }

                exhausted = batch.len() < BATCH_SIZE;

                if !batch.is_empty() {
                    progress += batch.len() as u64;
                    let mut add_docs = HashMap::new();
                    add_docs.insert(label.clone(), batch);
                    node_indexer.commit(&mut add_docs, &mut HashMap::new());
                    node_indexer.update_progress(&label, progress);
                }
                // guard dropped here — lock released between batches
            }

            if exhausted {
                node_indexer.enable(&label);
            } else {
                populate_index_batch(label, node_indexer, attrs, progress, next_min_row);
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
            node_attrs: AttributeStore::new(&format!("{name}/nodes"), version),
            relationship_attrs: AttributeStore::new(&format!("{name}/relationships"), version),
            node_indexer: Indexer::default(),
            node_labels: Vec::new(),
            relationship_types: Vec::new(),
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_size.max(1)).expect("cache_size.max(1) is always >= 1"),
            ))),
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
            node_attrs,
            relationship_attrs,
            node_indexer: Indexer::default(),
            node_labels,
            relationship_types,
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_size.max(1)).expect("cache_size.max(1) is always >= 1"),
            ))),
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

        // Rebuild all_nodes_matrix from per-label matrices
        // Each label matrix is diagonal (node_id, node_id), so we just need
        // to collect all live node IDs across all labels
        for lm in &self.labels_matices {
            for (node_id, _) in lm.iter(0, u64::MAX) {
                self.all_nodes_matrix.set(node_id, node_id, true);
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
            relationship_matrices: self.relationship_matrices.iter().map(Tensor::dup).collect(),
            node_attrs,
            relationship_attrs,
            node_indexer: self.node_indexer.clone(),
            node_labels: self.node_labels.clone(),
            relationship_types: self.relationship_types.clone(),
            cache: self.cache.clone(),
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
        self.node_attrs
            .attrs_name
            .iter()
            .chain(self.relationship_attrs.attrs_name.iter())
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
        let (parameters, query) = parser.parse_parameters()?;

        let current_udf_version = crate::runtime::functions::udf_version();

        {
            let mut cache = self.cache.lock();
            if let Some(plan) = cache.get(query) {
                if plan.udf_version == current_udf_version {
                    let optimize_plan = optimize(&plan.plan, self);
                    return Ok(Plan::new(
                        Arc::new(optimize_plan),
                        true,
                        parameters,
                        parse_duration,
                        plan_duration,
                    ));
                }
                // UDF version mismatch — discard stale cache entry
                cache.pop(query);
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
        let optimize_plan = optimize(&plan, self);
        plan_duration = start.elapsed();

        // Only cache the plan if UDF version hasn't changed during planning.
        // A drift means the plan may reference stale UDF bindings.
        if crate::runtime::functions::udf_version() == current_udf_version {
            self.cache.lock().push(
                query.to_string(),
                PlanTree {
                    plan,
                    udf_version: current_udf_version,
                },
            );
        }
        Ok(Plan::new(
            Arc::new(optimize_plan),
            false,
            parameters,
            parse_duration,
            plan_duration,
        ))
    }

    fn get_label_matrix(
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

    fn get_relationship_matrix(
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

        for id in nodes {
            self.all_nodes_matrix.set(id, id, true);
        }
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
        attrs: &HashMap<u64, OrderMap<Arc<String>, Value>>,
        index_add_docs: &mut HashMap<u64, RoaringTreemap>,
    ) -> Result<usize, String> {
        let nremoved = self.node_attrs.insert_attrs(attrs)?;

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
        Ok(nremoved)
    }

    pub fn import_node_attrs(
        &mut self,
        attrs: &HashMap<u64, OrderMap<Arc<String>, Value>>,
        index_add_docs: &mut HashMap<u64, RoaringTreemap>,
    ) {
        self.node_attrs.import_attrs(attrs);

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
    }

    pub fn import_relationship_attrs(
        &mut self,
        attrs: &HashMap<u64, OrderMap<Arc<String>, Value>>,
    ) {
        self.relationship_attrs.import_attrs(attrs);
    }

    pub fn set_nodes_labels(
        &mut self,
        nodes_labels: &mut Matrix,
        index_add_docs: &mut HashMap<u64, RoaringTreemap>,
    ) {
        self.resize();

        for (id, label_id) in nodes_labels.iter(0, u64::MAX) {
            self.node_labels_matrix.set(id, label_id, true);
            self.labels_matices[label_id as usize].set(id, id, true);
            let label = &self.node_labels[label_id as usize];
            if self.node_indexer.has_index(label) && self.node_attrs.has_attributes(id) {
                index_add_docs.entry(label_id).or_default().insert(id);
            }
        }
    }

    pub fn remove_nodes_labels(
        &mut self,
        nodes_labels: &mut Matrix,
        remove_docs: &mut HashMap<u64, RoaringTreemap>,
    ) {
        self.resize();

        for (id, label_id) in nodes_labels.iter(0, u64::MAX) {
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
        remove_docs: &mut HashMap<u64, RoaringTreemap>,
    ) -> Result<(), String> {
        self.deleted_nodes |= deleted_nodes;
        self.node_count -= deleted_nodes.len();

        for id in deleted_nodes {
            self.all_nodes_matrix.remove(id, id);

            for (_, label_id) in self.node_labels_matrix.iter(id, id) {
                let label = &self.node_labels[label_id as usize];
                self.labels_matices[label_id as usize].remove(id, id);
                if self.node_indexer.has_index(label) {
                    for attr in self.node_attrs.get_attrs(id) {
                        if self.node_indexer.has_indexed_attr(label, &attr) {
                            remove_docs.entry(label_id).or_default().insert(id);
                            break;
                        }
                    }
                }
            }

            for label_id in 0..self.labels_matices.len() {
                self.node_labels_matrix.remove(id, label_id as _);
            }
        }
        self.node_attrs.remove_all(deleted_nodes)?;
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
            .flat_map(move |m| m.iter(id.0, id.0, false).chain(m.iter(id.0, id.0, true)))
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
            return Box::new(
                self.all_nodes_matrix
                    .iter(min_row, u64::MAX)
                    .map(|(id, _)| NodeId(id)),
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

    pub fn create_relationships(
        &mut self,
        relationships: &HashMap<RelationshipId, PendingRelationship>,
    ) {
        self.relationship_count += relationships.len() as u64;
        self.reserved_relationship_count -= relationships.len() as u64;

        for id in relationships.keys() {
            if self.deleted_relationships.is_empty() {
                break;
            }
            self.deleted_relationships.remove(id.0);
        }

        // Ensure capacity covers the highest relationship ID (effects replay
        // may insert IDs above the current count when applied one-by-one).
        if let Some(max_id) = relationships.keys().map(|id| id.0).max() {
            let needed = max_id + 1;
            if needed > self.relationship_cap {
                while needed > self.relationship_cap {
                    self.relationship_cap *= 2;
                }
                self.resize_relationship_matrices();
            }
        }

        for (
            id,
            PendingRelationship {
                type_name,
                from: start,
                to: end,
                ..
            },
        ) in relationships
        {
            self.get_relationship_matrix_mut(type_name)
                .set(start.0, end.0, id.0);
        }

        self.resize();

        for (
            id,
            PendingRelationship {
                type_name,
                from: start,
                to: end,
            },
        ) in relationships
        {
            self.adjacancy_matrix.set(start.0, end.0, true);
            self.relationship_type_matrix.set(
                id.0,
                self.relationship_types
                    .iter()
                    .position(|p| p.as_str() == type_name.as_str())
                    .unwrap() as u64,
                true,
            );
        }
    }

    pub fn set_relationships_attributes(
        &mut self,
        attrs: &HashMap<u64, OrderMap<Arc<String>, Value>>,
    ) -> Result<usize, String> {
        let nremoved = self.relationship_attrs.insert_attrs(attrs)?;
        Ok(nremoved)
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
        rels: &HashMap<RelationshipId, (NodeId, NodeId)>,
    ) -> Result<(), String> {
        self.deleted_relationships
            .extend(rels.keys().map(|id| id.0));
        self.relationship_count -= rels.len() as u64;

        // Collect unique (src, dst) pairs to check adjacancy_matrix after deletion
        let pairs: std::collections::HashSet<(u64, u64)> =
            rels.values().map(|(src, dst)| (src.0, dst.0)).collect();

        for (type_id, rels) in &rels
            .iter()
            .map(|(id, (src, dst))| (id.0, src.0, dst.0))
            .into_group_map_by(|(id, _, _)| self.get_relationship_type_id(RelationshipId(*id)))
        {
            for (id, _, _) in rels {
                self.relationship_type_matrix.remove(*id, type_id.0 as u64);
                self.relationship_attrs.remove(*id)?;
            }
            let typ = self.relationship_types[type_id.0].clone();
            self.get_relationship_matrix_mut(&typ).remove_all(rels);
        }

        // Update adjacancy_matrix: remove entries where no typed edges remain
        for (src, dst) in pairs {
            let has_edges = self
                .relationship_matrices
                .iter()
                .any(|tensor| tensor.get(src, dst).next().is_some());
            if !has_edges {
                self.adjacancy_matrix.remove(src, dst);
            }
        }

        Ok(())
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

    /// Build a relationship matrix combining the given types and filtering by
    /// source/destination labels.  The returned matrix can be cached and reused
    /// across multiple calls to [`iter_relationship_matrix`].
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

    /// Iterate a prebuilt relationship matrix, optionally restricting to a
    /// single source row (`from_id`) and/or a single destination column
    /// (`to_id`).
    pub fn iter_relationship_matrix(
        m: &Matrix,
        from_id: Option<NodeId>,
        to_id: Option<NodeId>,
    ) -> impl Iterator<Item = (NodeId, NodeId)> + use<> {
        let (min_row, max_row) = from_id.map_or((0, u64::MAX), |id| (id.0, id.0));
        m.iter(min_row, max_row)
            .filter(move |(_, dest)| to_id.is_none() || to_id.unwrap().0 == *dest)
            .map(|(src, dest)| (NodeId(src), NodeId(dest)))
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

    #[must_use]
    pub fn get_relationship_attribute(
        &self,
        id: RelationshipId,
        attr: &Arc<String>,
    ) -> Option<Value> {
        self.relationship_attrs.get_attr(id.0, attr)
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
    ) -> impl Iterator<Item = (Arc<String>, Value)> + '_ {
        self.node_attrs.get_all_attrs(id.0)
    }

    pub fn get_node_all_attrs_by_id(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = (u16, Value)> + '_ {
        self.node_attrs
            .get_all_attrs_by_id(id.0)
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
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
    ) -> impl Iterator<Item = (Arc<String>, Value)> + '_ {
        self.relationship_attrs.get_all_attrs(id.0)
    }

    pub fn get_relationship_all_attrs_by_id(
        &self,
        id: RelationshipId,
    ) -> impl Iterator<Item = (u16, Value)> + '_ {
        self.relationship_attrs
            .get_all_attrs_by_id(id.0)
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
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
                self.start_populate_index(label);
            }
            EntityType::Relationship => {}
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
                // Don't spawn async — caller will populate via populate_index_sync
            }
            EntityType::Relationship => {}
        }
        Ok(())
    }

    /// Synchronously populate all pending indexes.
    /// Used after RDB load when the graph is fully constructed.
    pub fn populate_indexes_sync(&mut self) {
        let fields_by_label = self.node_indexer.get_all_pending_fields();
        for (label, attrs) in fields_by_label {
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
                self.node_indexer.enable(&label);
            }
        }
    }

    fn start_populate_index(
        &self,
        label: &Arc<String>,
    ) {
        populate_index(label.clone(), self.node_indexer.clone());
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

    pub fn commit_index(
        &mut self,
        index_add_docs: &mut HashMap<u64, RoaringTreemap>,
        remove_docs: &mut HashMap<u64, RoaringTreemap>,
    ) {
        let lock = self.node_indexer.write_lock();
        let _guard = lock.lock();

        let mut add_docs = HashMap::new();
        for (label_id, ids) in index_add_docs.drain() {
            let label = &self.node_labels[label_id as usize];
            let fields = self.node_indexer.get_fields(label);
            let mut docs = vec![];
            for id in ids {
                let mut doc = Document::new(id);
                for (key, fields) in &fields {
                    if let Some(value) = self.node_attrs.get_attr(id, key) {
                        for field in fields {
                            doc.set(field, &value);
                        }
                    }
                }
                docs.push(doc);
            }
            add_docs.insert(label.clone(), docs);
        }

        let mut remove = HashMap::new();
        for (label_id, ids) in remove_docs.drain() {
            let label = &self.node_labels[label_id as usize];
            remove.insert(label.clone(), ids);
        }

        self.node_indexer.commit(&mut add_docs, &mut remove);
    }

    pub fn drop_index(
        &mut self,
        index_type: &IndexType,
        entity_type: &EntityType,
        label: &Arc<String>,
        attrs: &Vec<Arc<String>>,
    ) -> Result<usize, String> {
        match entity_type {
            EntityType::Node => {
                let total = self
                    .get_label_matrix(label)
                    .map_or(0, super::graphblas::matrix::Size::nvals);
                let reindex = self
                    .node_indexer
                    .drop_index(label, attrs, index_type, total);

                if let Some((dropped, remaining)) = reindex {
                    if dropped > 0 {
                        if remaining > 0 {
                            self.node_indexer.recreate_index(label)?;
                            self.start_populate_index(label);
                        } else {
                            drop_index_bg(label.clone(), self.node_indexer.clone());
                        }
                    }
                    return Ok(dropped);
                }
            }
            EntityType::Relationship => {}
        }
        Ok(0)
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

    pub fn fulltext_query_nodes(
        &self,
        label: &Arc<String>,
        query: &str,
    ) -> Result<impl Iterator<Item = (NodeId, f64)> + use<>, String> {
        self.node_indexer
            .fulltext_query(label, query)
            .map(|r| r.map(|(id, score)| (NodeId(id), score)))
    }

    #[must_use]
    pub fn index_info(&self) -> Vec<IndexInfo> {
        self.node_indexer.index_info()
    }

    pub fn cancel_indexing(&self) {
        self.node_indexer.cancel();
    }

    pub fn set_indexer_graph(
        &mut self,
        graph: Arc<AtomicRefCell<Self>>,
    ) {
        self.node_indexer.set_graph(graph);
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
            self.node_attrs.memory_usage() + self.deleted_nodes.serialized_size();

        // --- edge block storage ---
        let edge_block_storage_sz: usize =
            self.relationship_attrs.memory_usage() + self.deleted_relationships.serialized_size();

        // --- node attributes by label (sampling) ---
        let mut node_attr_by_label: Vec<(Arc<String>, usize)> = Vec::new();

        // Track all labeled node IDs (deduplicated) for unlabeled-node detection
        // and track processed node IDs to avoid double-counting multi-label nodes.
        let mut labeled_node_ids = roaring::RoaringTreemap::new();
        let mut processed_node_ids = roaring::RoaringTreemap::new();

        for (label_idx, label_name) in self.node_labels.iter().enumerate() {
            let label_matrix = &self.labels_matices[label_idx];
            let total_for_label = label_matrix.nvals();
            if total_for_label == 0 {
                node_attr_by_label.push((label_name.clone(), 0));
                continue;
            }

            // Iterate the full label matrix, collecting all node IDs for the
            // labeled-node bitmap.  For attribute estimation, only sample nodes
            // that have NOT been processed by a previous label (avoids
            // double-counting multi-label nodes, matching the C implementation).
            let mut sampled_mem: usize = 0;
            let mut sampled_count: usize = 0;
            let mut unprocessed_count: u64 = 0;
            for (node_id, _) in label_matrix.iter(0, u64::MAX) {
                labeled_node_ids.insert(node_id);
                if !processed_node_ids.contains(node_id) {
                    unprocessed_count += 1;
                    processed_node_ids.insert(node_id);
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
        let total_labeled = labeled_node_ids.len();
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
                if labeled_node_ids.contains(node_id) {
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
        let indices_sz = self.node_indexer.memory_usage();

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
