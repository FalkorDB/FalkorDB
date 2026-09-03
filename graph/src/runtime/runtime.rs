//! Query execution engine.
//!
//! This module contains the [`Runtime`] struct which executes query plans
//! against the graph. The runtime builds a tree of [`BatchOp`] operators
//! that process data in batches of up to 1024 rows.
//!
//! ## Execution Model
//!
//! ```text
//!  IR Plan Tree                BatchOp Tree (built by run_batch)
//! ┌──────────┐               ┌──────────────────┐
//! │  Return  │  ────────►    │  ProjectOp        │◄── yields Batch<'a>
//! │    │     │               │    │              │
//! │  Filter  │               │  FilterOp         │◄── sets selection vector
//! │    │     │               │    │              │
//! │  Expand  │               │  CondTraverseOp   │◄── expands per-row
//! │    │     │               │    │              │
//! │ NodeScan │               │  NodeByLabelScan  │◄── produces BATCH_SIZE rows
//! └──────────┘               └──────────────────┘
//!
//!  query() drives the root BatchOp, collecting result rows into ResultSummary.
//! ```
//!
//! ## Key Types
//!
//! - [`Runtime`]: Main execution context (carries `Pool`, graph ref, plan)
//! - [`ResultSummary`]: Query result with collected rows and statistics
//! - [`BatchOp`]: Enum-dispatch operator tree (28+ variants)
//! - [`Batch`]: Columnar batch of up to 1024 rows
//!
//! ## Write Operations
//!
//! Write operations (CREATE, DELETE, SET) are batched in [`Pending`] and
//! applied atomically by [`CommitOp`] at the end of the query.

#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
use crate::{
    graph::graph::{Graph, LabelId, NodeId, RelationshipId},
    identifier_limits::validate_identifier_len,
    index::indexer::{IndexOptions, IndexType, TextIndexOptions, VectorIndexOptions},
    parser::ast::{ExprIR, QueryExpr, Variable},
    planner::IR,
    runtime::{
        batch::{Batch, BatchBuilder, BatchOp, BatchRow, Column, NullBitmap, classify_column},
        ops::{
            AggregateOp, AllShortestPathsOp, ApplyOp, CartesianProductOp, CommitOp, CondTraverseOp,
            CondVarLenTraverseOp, CreateOp, DeleteOp, DistinctOp, EdgeByFulltextScanOp,
            EdgeByIndexScanOp, EdgeByVectorScanOp, ExpandIntoOp, FilterOp, ForEachOp,
            IncludePendingOp, LimitOp, LoadCsvOp, MergeOp, NodeByFulltextScanOp, NodeByIdSeekOp,
            NodeByIndexScanOp, NodeByLabelAndIdScanOp, NodeByLabelScanOp, NodeByVectorScanOp,
            OptionalOp, OrApplyMultiplexerOp, PathBuilderOp, ProcedureCallOp, ProjectOp, RemoveOp,
            SemiApplyOp, SetOp, SkipOp, SortOp, UnionOp, UnwindOp, ValueHashJoinOp,
        },
        ordermap::OrderMap,
        orderset::OrderSet,
        pending::Pending,
        row::{Row, RowView},
        value::{DeletedNode, DeletedRelationship, Value, ValuesDeduper},
    },
};
use atomic_refcell::AtomicRefCell;
use chrono::{DateTime, Utc};
use once_cell::unsync::Lazy;
use orx_tree::{Bfs, Dyn, DynNode, DynTree, MemoryPolicy, NodeIdx, NodeRef};
use roaring::RoaringTreemap;
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    marker::PhantomData,
    sync::Arc,
    time::{Duration, Instant},
};

/// Query result containing statistics and returned tuples.
pub struct ResultSummary<'a> {
    /// Mutation statistics (nodes created, etc.)
    pub stats: QueryStatistics,
    /// Result batches from the execution pipeline
    pub result: Vec<Batch<'a>>,
}

/// Statistics about query execution and mutations performed.
#[derive(Default)]
pub struct QueryStatistics {
    pub labels_added: usize,
    pub labels_removed: usize,
    pub nodes_created: u64,
    pub relationships_created: usize,
    pub nodes_deleted: u64,
    pub relationships_deleted: usize,
    pub properties_set: usize,
    pub properties_removed: usize,
    pub indexes_created: usize,
    pub indexes_dropped: usize,
    /// Total execution time in milliseconds
    pub execution_time: f64,
    /// Whether the query plan was retrieved from cache
    pub cached: bool,
}

/// Upper bound on attr-id memo entries; queries with more distinct
/// attribute keys (typically per-row computed keys) fall back to the
/// name-map lookup for the excess.
const ATTR_ID_MEMO_CAP: usize = 32;

/// The query execution context.
///
/// Runtime holds all state needed to execute a query plan:
/// - Graph reference and parameters
/// - Pending mutations (for deferred writes)
/// - Statistics tracking
/// - Variable bindings cache
///
/// # Lifecycle
/// 1. Create Runtime with graph, parameters, and plan
/// 2. Call `run()` to execute
/// 3. Pending mutations applied at end of execution
/// 4. Return `ResultSummary` with results and stats
pub struct Runtime<'a> {
    /// Query parameters ($param syntax)
    pub parameters: HashMap<String, Value>,
    /// Graph being queried (shared, thread-safe reference)
    pub g: Arc<AtomicRefCell<Graph>>,
    /// Whether this is a write query
    pub write: bool,
    /// Batched mutations (lazy-initialized)
    pub pending: Lazy<RefCell<Pending>>,
    /// Execution statistics
    pub stats: RefCell<QueryStatistics>,
    /// Query execution plan tree
    pub plan: Arc<DynTree<IR>>,
    /// The host value holding this query's locks (see [`crate::locks`]). Reached
    /// through [`Runtime::write_escalation`] by the operators that mutate shared
    /// state, so the dependency is explicit at the point of use.
    write_escalation: &'a dyn crate::locks::WriteEscalation,
    /// Deduplication state for DISTINCT operations, keyed by the DISTINCT
    /// expression's node index plus its aggregation group hash. Uses a
    /// cheap `FxHashMap` over a `(NodeIdx, u64)` tuple so the hot per-row
    /// lookup avoids the previous `format!`-built `String` key (heap alloc +
    /// `NodeIdx` Debug formatting) and SipHash. `NodeIdx` is `Copy`, and the
    /// expression trees it points into are immutable for the query lifetime.
    pub value_dedupers:
        RefCell<rustc_hash::FxHashMap<(NodeIdx<Dyn<ExprIR<Variable>>>, u64), ValuesDeduper>>,
    /// Variables to return in query results
    pub return_names: Vec<Variable>,
    /// Debug mode: record operator execution
    pub inspect: bool,
    /// Debug records of operator execution
    pub record: RefCell<Vec<(NodeIdx<Dyn<IR>>, Result<Row, String>)>>,
    /// Folder for LOAD CSV operations
    pub import_folder: String,
    /// Cache of deleted nodes for result consistency
    pub deleted_nodes: RefCell<HashMap<NodeId, DeletedNode>>,
    /// Cache of deleted relationships for result consistency
    pub deleted_relationships: RefCell<HashMap<RelationshipId, DeletedRelationship>>,
    /// Cache for MERGE pattern matching — stores only the created entity bindings (variable id → value)
    pub merge_pattern_cache: RefCell<HashMap<u64, Vec<(u32, Value)>>>,
    /// Pointer-identity memo of resolved node attribute ids. Plan
    /// expressions hold the same `Arc<String>` across all rows, so after
    /// the first resolution a lookup is a short pointer scan instead of a
    /// string hash + equality probe. Entries hold the `Arc` so a memoized
    /// address can never be freed and recycled by a different name
    /// mid-query. Capped so per-row computed keys can't grow it unboundedly.
    node_attr_id_memo: RefCell<Vec<(Arc<String>, u16)>>,
    /// Relationship-space counterpart of `node_attr_id_memo` (the two id
    /// spaces are independent).
    rel_attr_id_memo: RefCell<Vec<(Arc<String>, u16)>>,
    /// Maximum number of result rows to return. Negative means unlimited.
    pub result_set_size: i64,
    /// Effects buffer built before commit, for replication.
    pub effects_buffer: RefCell<Option<Vec<u8>>>,
    /// Total number of effect records across all commits in this query.
    pub effects_count: Cell<u64>,
    /// Whether commits should serialize an effects buffer. Callers clear
    /// this when replication has no possible consumer (no AOF, no replica
    /// has ever attached); the replication layer then falls back to
    /// verbatim query propagation, which Redis discards for free.
    pub build_effects: Cell<bool>,
    /// Timestamp captured at the start of the transaction/query.
    /// Used by `date.transaction()`, `localtime.transaction()`, and `localdatetime.transaction()`
    /// so every call in the same transaction returns the same value.
    pub transaction_timestamp: DateTime<Utc>,
    /// Whether profiling is enabled for this query.
    pub profile: bool,
    /// Per-operator profile data: (records_produced, exclusive_time).
    pub profile_data: RefCell<HashMap<NodeIdx<Dyn<IR>>, (usize, Duration)>>,
    /// Accumulator for child time subtraction during profiling.
    pub profile_child_time: Cell<Duration>,
    /// Optional deadline for query timeout enforcement.
    pub deadline: Option<Instant>,
    /// Maximum memory (bytes) a single query may consume. 0 = unlimited.
    pub mem_capacity: i64,
    /// Function pointer to read the current thread's net memory usage.
    pub current_usage_fn: Option<fn() -> usize>,
    /// Preserves the `'a` lifetime parameter used by borrowing methods.
    _marker: PhantomData<&'a ()>,
}

pub trait GetVariables {
    fn get_variables(&self) -> Vec<Variable>;
}

impl<T: MemoryPolicy> GetVariables for DynNode<'_, IR, T> {
    fn get_variables(&self) -> Vec<Variable> {
        let mut vars = vec![];
        for node in self.walk::<Bfs>() {
            match node {
                IR::Optional(variables) => vars.extend(variables.iter().cloned()),
                IR::ProcedureCall {
                    yields: named_outputs,
                    ..
                } => {
                    vars.extend(named_outputs.clone());
                }
                IR::Unwind { var: variable, .. } => vars.push(variable.clone()),
                IR::Create(query_graph)
                | IR::Merge {
                    pattern: query_graph,
                    ..
                } => {
                    for node in query_graph.nodes() {
                        vars.push(node.alias.clone());
                    }
                    for relationship in query_graph.relationships() {
                        vars.push(relationship.alias.clone());
                    }
                    for path in query_graph.paths() {
                        vars.push(path.var.clone());
                    }
                }
                IR::ForEach { var, .. } | IR::LoadCsv { var, .. } => {
                    vars.push(var.clone());
                }
                IR::Delete { .. }
                | IR::Argument(_)
                | IR::Set(_)
                | IR::Remove(_)
                | IR::Filter(_)
                | IR::CartesianProduct
                | IR::ValueHashJoin { .. }
                | IR::Union
                | IR::Apply
                | IR::SemiApply
                | IR::AntiSemiApply
                | IR::OrApplyMultiplexer(_)
                | IR::Sort(_)
                | IR::Skip(_)
                | IR::Limit(_)
                | IR::Distinct
                | IR::Commit
                | IR::IncludePending { .. }
                | IR::CreateIndex { .. }
                | IR::DropIndex { .. } => {}
                IR::NodeByLabelScan { node, .. }
                | IR::AllNodeScan(node)
                | IR::NodeByIndexScan { node, .. }
                | IR::NodeByLabelAndIdScan { node, .. }
                | IR::NodeByIdSeek { node, .. } => {
                    vars.push(node.alias.clone());
                }
                IR::NodeByFulltextScan { node, score, .. }
                | IR::NodeByVectorScan { node, score, .. } => {
                    vars.push(node.clone());
                    if let Some(score) = score {
                        vars.push(score.clone());
                    }
                }
                IR::EdgeByFulltextScan { edge, score, .. }
                | IR::EdgeByVectorScan { edge, score, .. } => {
                    vars.push(edge.clone());
                    if let Some(score) = score {
                        vars.push(score.clone());
                    }
                }
                IR::CondTraverse {
                    relationship: query_relationship,
                    ..
                }
                | IR::EdgeByIndexScan {
                    relationship: query_relationship,
                    ..
                }
                | IR::AllShortestPaths(query_relationship)
                | IR::ExpandInto {
                    relationship: query_relationship,
                    ..
                } => {
                    vars.push(query_relationship.alias.clone());
                    vars.push(query_relationship.from.alias.clone());
                    vars.push(query_relationship.to.alias.clone());
                }
                IR::CondVarLenTraverse {
                    relationship: query_relationship,
                    path_var,
                    ..
                } => {
                    vars.push(query_relationship.alias.clone());
                    vars.push(query_relationship.from.alias.clone());
                    vars.push(query_relationship.to.alias.clone());
                    if let Some(path_var) = path_var {
                        vars.push(path_var.clone());
                    }
                }
                IR::PathBuilder(query_paths) => {
                    for path in query_paths {
                        vars.push(path.var.clone());
                    }
                }
                IR::Aggregate {
                    names: variables, ..
                } => {
                    vars.extend(variables.iter().cloned());
                }
                IR::Project { exprs: items, .. } => {
                    vars.extend(items.iter().map(|v| v.0.clone()));
                    break;
                }
            }
        }
        vars
    }
}

pub(crate) trait ReturnNames {
    fn get_return_names(&self) -> Vec<Variable>;
}

impl ReturnNames for DynNode<'_, IR> {
    fn get_return_names(&self) -> Vec<Variable> {
        match self.data() {
            IR::Project { exprs: trees, .. } => trees.iter().map(|v| v.0.clone()).collect(),
            IR::Commit => self
                .get_child(0)
                .map_or(vec![], |child| child.get_return_names()),
            IR::ProcedureCall {
                yields: named_outputs,
                ..
            } => named_outputs.clone(),
            IR::NodeByFulltextScan { node, score, .. }
            | IR::NodeByVectorScan { node, score, .. } => {
                let mut v = vec![node.clone()];
                if let Some(score) = score {
                    v.push(score.clone());
                }
                v
            }
            IR::EdgeByFulltextScan { edge, score, .. }
            | IR::EdgeByVectorScan { edge, score, .. } => {
                let mut v = vec![edge.clone()];
                if let Some(score) = score {
                    v.push(score.clone());
                }
                v
            }
            IR::Sort(_) | IR::Skip(_) | IR::Limit(_) | IR::Distinct => {
                self.child(0).get_return_names()
            }
            IR::Union => self.child(0).get_return_names(),
            IR::Aggregate { names, .. } => names.clone(),
            _ => vec![],
        }
    }
}

impl<'a> Runtime<'a> {
    #[inline]
    pub fn inspect_batch(
        &self,
        idx: NodeIdx<Dyn<IR>>,
        result: &Result<Batch<'_>, String>,
    ) {
        if self.inspect {
            match result {
                Ok(batch) => {
                    let mut record = self.record.borrow_mut();
                    for row in batch.active_indices() {
                        record.push((idx, Ok(BatchRow::new(batch, row).to_owned_row())));
                    }
                }
                Err(err) => {
                    self.record.borrow_mut().push((idx, Err(err.clone())));
                }
            }
        }
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        g: Arc<AtomicRefCell<Graph>>,
        parameters: HashMap<String, Value>,
        write: bool,
        plan: Arc<DynTree<IR>>,
        inspect: bool,
        import_folder: String,
        result_set_size: i64,
        profile: bool,
        timeout_ms: Option<u64>,
        mem_capacity: i64,
        current_usage_fn: Option<fn() -> usize>,
        write_escalation: &'a dyn crate::locks::WriteEscalation,
    ) -> Self {
        let return_names = plan.root().get_return_names();
        let pending = Lazy::new((|| RefCell::new(Pending::new())) as fn() -> RefCell<Pending>);
        if write {
            pending.borrow_mut().set_schema_baseline(&g);
        }
        Self {
            parameters,
            g,
            write,
            pending,
            stats: RefCell::new(QueryStatistics::default()),
            plan,
            write_escalation,
            return_names,
            value_dedupers: RefCell::new(rustc_hash::FxHashMap::default()),
            inspect,
            record: RefCell::new(vec![]),
            import_folder,
            deleted_nodes: RefCell::new(HashMap::new()),
            deleted_relationships: RefCell::new(HashMap::new()),
            merge_pattern_cache: RefCell::new(HashMap::new()),
            node_attr_id_memo: RefCell::new(Vec::new()),
            rel_attr_id_memo: RefCell::new(Vec::new()),
            result_set_size,
            effects_buffer: RefCell::new(None),
            effects_count: Cell::new(0),
            build_effects: Cell::new(true),
            transaction_timestamp: Utc::now(),
            profile,
            profile_data: RefCell::new(HashMap::new()),
            profile_child_time: Cell::new(Duration::ZERO),
            deadline: timeout_ms.map(|ms| Instant::now() + Duration::from_millis(ms)),
            mem_capacity,
            current_usage_fn,
            _marker: PhantomData,
        }
    }

    /// Check if the query has exceeded its timeout deadline.
    /// Returns `Err("Query timed out")` if the deadline has passed.
    #[inline]
    pub fn check_timeout(&self) -> Result<(), String> {
        if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            return Err("Query timed out".to_string());
        }
        Ok(())
    }

    #[inline]
    pub fn check_mem_capacity(&self) -> Result<(), String> {
        if self.mem_capacity > 0
            && let Some(usage_fn) = self.current_usage_fn
            && usage_fn() as i64 > self.mem_capacity
        {
            return Err("Query's mem consumption exceeded capacity".to_string());
        }
        Ok(())
    }

    /// Call `runtime.write_escalation().upgrade_to_write()?` before mutating
    /// shared state (the index, or the published graph version).
    ///
    /// Not a guard: the host owns the locks for the whole query — they must
    /// outlive `query()` so the host can commit — so this only *changes mode*,
    /// never releases.
    #[must_use]
    pub fn write_escalation(&self) -> &'a dyn crate::locks::WriteEscalation {
        self.write_escalation
    }

    /// Undo the index documents earlier `Commit`s published, after this query
    /// failed. See [`Pending::resync_published_indexes`].
    pub fn resync_published_indexes(
        &self,
        committed: &Arc<AtomicRefCell<Graph>>,
    ) {
        self.pending
            .borrow_mut()
            .resync_published_indexes(committed, &self.g);
    }

    /// Write this `Commit`'s index documents to RediSearch. Writer mode only.
    pub fn commit_deferred_indexes(&self) {
        self.pending.borrow_mut().commit_deferred_indexes(&self.g);
    }

    pub fn query(&'a self) -> Result<ResultSummary<'a>, String> {
        let start = Instant::now();
        let idx = self.plan.root().idx();
        let labels_count = self.g.borrow().labels_count();
        let mut result = vec![];
        let mut batch_op = self.run_batch(idx)?;
        if self.result_set_size >= 0 {
            let limit = self.result_set_size as usize;
            let mut total: usize = 0;
            for batch_result in &mut batch_op {
                let mut batch = batch_result?;
                total += batch.active_len();
                if total >= limit {
                    let keep = batch.active_len() - (total - limit);
                    let sel: Vec<u16> = batch
                        .active_indices()
                        .take(keep)
                        .map(|i| i as u16)
                        .collect();
                    batch.set_selection(sel);
                    result.push(batch);
                    // Drain remaining batches so CommitOp and pending
                    // mutations run to completion, but only for write
                    // queries that actually have a Commit in the plan.
                    if self.write {
                        for remaining in &mut batch_op {
                            remaining?;
                        }
                    }
                    break;
                }
                result.push(batch);
            }
        } else {
            for batch_result in &mut batch_op {
                let batch = batch_result?;
                result.push(batch);
            }
        }
        let run_duration = start.elapsed();

        self.stats.borrow_mut().labels_added += self.g.borrow().labels_count() - labels_count;
        self.stats.borrow_mut().execution_time = run_duration.as_secs_f64() * 1000.0;
        Ok(ResultSummary {
            stats: self.stats.take(),
            result,
        })
    }

    /// Creates a single-row default batch.
    fn default_batch(&self) -> Batch<'_> {
        let mut builder = BatchBuilder::new();
        builder.push_row(&Row::new());
        builder.finish()
    }

    /// Walk IR ancestors from `idx` upward to find the effective limit.
    /// Returns `None` if a non-transparent operation is encountered before
    /// a Limit node, or if no Limit ancestor exists.
    /// Only transparent operators (Project, Skip) are safe to propagate
    /// through — Sort is NOT transparent because it needs all rows.
    fn effective_limit(
        &self,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Option<usize> {
        let mut cur = idx;
        while let Some(parent) = self.plan.node(cur).parent() {
            match parent.data() {
                IR::Limit(expr) => {
                    let val = super::eval::ExprEval::from_runtime(self)
                        .eval(expr, expr.root().idx(), super::eval::NO_ROW, None)
                        .ok()?;
                    return match val {
                        Value::Int(n) if n >= 0 => Some(n as usize),
                        _ => None,
                    };
                }
                // These operators pass rows through 1:1 or 1:N — limit still
                // provides a useful early-stop hint through them.
                IR::Project { .. }
                | IR::Skip(_)
                | IR::CondTraverse { .. }
                | IR::ExpandInto { .. } => {}
                // Everything else is a barrier.
                _ => {
                    return None;
                }
            }
            cur = parent.idx();
        }
        None
    }

    /// Walk IR ancestors from `idx` upward to find the effective skip.
    /// Returns 0 if a row-reducing or eager operation is encountered before
    /// a Skip node, or if no Skip ancestor exists.
    fn effective_skip(
        &self,
        idx: NodeIdx<Dyn<IR>>,
    ) -> usize {
        let mut cur = idx;
        while let Some(parent) = self.plan.node(cur).parent() {
            match parent.data() {
                IR::Skip(expr) => {
                    let val = super::eval::ExprEval::from_runtime(self)
                        .eval(expr, expr.root().idx(), super::eval::NO_ROW, None)
                        .ok();
                    return match val {
                        Some(Value::Int(n)) if n >= 0 => n as usize,
                        _ => 0,
                    };
                }
                // Safe pass-through operators.
                IR::Project { .. } | IR::Limit(_) => {}
                // Everything else is a barrier.
                _ => {
                    return 0;
                }
            }
            cur = parent.idx();
        }
        0
    }

    /// Combined downstream row budget for `idx`: `effective_limit + effective_skip`,
    /// or `None` when no usable `Limit` ancestor exists. Accounts for both limit
    /// and skip so a capped operator produces enough rows for a downstream
    /// `SkipOp` + `LimitOp` pipeline.
    fn record_cap(
        &self,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Option<usize> {
        self.effective_limit(idx)
            .map(|l| l.saturating_add(self.effective_skip(idx)))
    }

    /// Returns the IR child indices that must be built before `idx` itself.
    /// Mirrors the per-variant recursion pattern that `build_batch_op` expects.
    fn children_to_recurse(
        &self,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Vec<NodeIdx<Dyn<IR>>> {
        let node = self.plan.node(idx);
        match node.data() {
            IR::Union | IR::Argument(_) | IR::CreateIndex { .. } | IR::DropIndex { .. } => {
                Vec::new()
            }
            IR::CartesianProduct => node.children().map(|c| c.idx()).collect(),
            IR::ValueHashJoin { .. } => {
                vec![node.child(0).idx(), node.child(1).idx()]
            }
            IR::Optional(_) | IR::Merge { .. } | IR::ForEach { .. } => {
                if node.num_children() > 1 {
                    vec![node.child(0).idx()]
                } else {
                    Vec::new()
                }
            }
            _ => node.get_child(0).map(|c| vec![c.idx()]).unwrap_or_default(),
        }
    }

    /// Iteratively builds a batch-mode operator tree for the given IR node.
    ///
    /// The recursive form was prone to stack overflow under ASAN: the body
    /// is a giant match over ~60 IR variants, so each frame holds the full
    /// `BatchOp` enum (~41 KB). Deep plans (Optional/Aggregate/Delete/ForEach
    /// chains) under ASAN's 2-3x frame inflation exhausted the default 2 MB
    /// thread stack mid-walk. Postorder iteration moves the per-node state
    /// to the heap so depth is bounded only by available memory.
    pub fn run_batch(
        &'a self,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Result<BatchOp<'a>, String> {
        enum Frame<I> {
            Pre(I),
            Post(I, usize),
        }
        let mut work: Vec<Frame<NodeIdx<Dyn<IR>>>> = vec![Frame::Pre(idx)];
        let mut built: Vec<BatchOp<'a>> = Vec::new();
        while let Some(f) = work.pop() {
            match f {
                Frame::Pre(cur) => {
                    let kids = self.children_to_recurse(cur);
                    let n = kids.len();
                    work.push(Frame::Post(cur, n));
                    for kid in kids.into_iter().rev() {
                        work.push(Frame::Pre(kid));
                    }
                }
                Frame::Post(cur, n) => {
                    let start = built.len() - n;
                    let kids: Vec<BatchOp<'a>> = built.drain(start..).collect();
                    let op = self.build_batch_op(cur, kids)?;
                    built.push(op);
                }
            }
        }
        built.pop().ok_or_else(|| String::from("empty plan tree"))
    }

    /// Constructs a single `BatchOp` for `idx`, consuming the already-built
    /// children that were produced by `run_batch`'s postorder walk.
    fn build_batch_op(
        &'a self,
        idx: NodeIdx<Dyn<IR>>,
        children: Vec<BatchOp<'a>>,
    ) -> Result<BatchOp<'a>, String> {
        let mut children = children.into_iter();
        let pop_or_once = |it: &mut std::vec::IntoIter<BatchOp<'a>>| -> BatchOp<'a> {
            it.next()
                .unwrap_or_else(|| BatchOp::Once(Some(self.default_batch())))
        };
        let pop_or_argument = |it: &mut std::vec::IntoIter<BatchOp<'a>>| -> BatchOp<'a> {
            it.next()
                .unwrap_or_else(|| BatchOp::Argument(Some(self.default_batch())))
        };
        match self.plan.node(idx).data() {
            IR::NodeByLabelScan { .. } | IR::AllNodeScan(_) => {
                let child = pop_or_once(&mut children);
                let ir = self.plan.node(idx).data();
                let node_pattern = match ir {
                    IR::NodeByLabelScan { node } => node,
                    IR::AllNodeScan(n) => n,
                    _ => unreachable!(),
                };
                Ok(BatchOp::NodeByLabelScan(NodeByLabelScanOp::new(
                    self,
                    Box::new(child),
                    node_pattern,
                    idx,
                )))
            }
            IR::IncludePending { node } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::IncludePending(IncludePendingOp::new(
                    self,
                    Box::new(child),
                    node,
                    idx,
                )))
            }
            IR::Filter(tree) => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Filter(FilterOp::new(
                    self,
                    Box::new(child),
                    tree,
                    idx,
                )))
            }
            IR::Project {
                exprs: trees,
                copies: copy_from_parent,
            } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Project(ProjectOp::new(
                    self,
                    Box::new(child),
                    trees,
                    copy_from_parent,
                    idx,
                )))
            }
            IR::Skip(skip) => {
                let child = pop_or_once(&mut children);
                let Value::Int(skip) = {
                    let this = &self;
                    let idx = skip.root().idx();
                    super::eval::ExprEval::from_runtime(this).eval(
                        skip,
                        idx,
                        super::eval::NO_ROW,
                        None,
                    )
                }?
                else {
                    return Err(String::from("Skip operator requires an integer argument"));
                };
                if skip < 0 {
                    return Err(format!("SKIP must be a non-negative integer, got {skip}"));
                }
                Ok(BatchOp::Skip(SkipOp::new(
                    self,
                    Box::new(child),
                    skip as usize,
                    idx,
                )))
            }
            IR::Limit(limit) => {
                let child = pop_or_once(&mut children);
                let Value::Int(limit) = {
                    let this = &self;
                    let idx = limit.root().idx();
                    super::eval::ExprEval::from_runtime(this).eval(
                        limit,
                        idx,
                        super::eval::NO_ROW,
                        None,
                    )
                }?
                else {
                    return Err(String::from("Limit operator requires an integer argument"));
                };
                if limit < 0 {
                    return Err(format!("LIMIT must be a non-negative integer, got {limit}"));
                }
                Ok(BatchOp::Limit(LimitOp::new(
                    self,
                    Box::new(child),
                    limit as usize,
                    idx,
                )))
            }
            IR::Distinct => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Distinct(DistinctOp::new(
                    self,
                    Box::new(child),
                    idx,
                )))
            }
            IR::Sort(trees) => {
                let limit = self.effective_limit(idx);
                let skip = self.effective_skip(idx);
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Sort(SortOp::new(
                    self,
                    Box::new(child),
                    trees,
                    idx,
                    limit,
                    skip,
                )))
            }
            IR::Aggregate {
                keys,
                aggregations: agg,
                projections: copy_from_parent,
                ..
            } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Aggregate(AggregateOp::new(
                    self,
                    Box::new(child),
                    keys,
                    agg,
                    copy_from_parent,
                    idx,
                )))
            }
            IR::Unwind {
                expr: list,
                var: name,
            } => {
                // A downstream Skip/Limit bounds how many expanded rows are
                // needed; pass it through so packing stays lazy under `LIMIT`.
                let record_cap = self.record_cap(idx);
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Unwind(UnwindOp::new(
                    self,
                    Box::new(child),
                    list,
                    name,
                    record_cap,
                    idx,
                )))
            }
            IR::CondTraverse {
                relationship: relationship_pattern,
                emit_relationship,
                sibling_edges,
                transposed,
                chain,
                optional,
                bind_relationship,
            } => {
                // Account for both limit and skip so the traverse produces
                // enough rows for a downstream SkipOp + LimitOp pipeline.
                let record_cap = self.record_cap(idx);
                let child = pop_or_once(&mut children);
                Ok(BatchOp::CondTraverse(CondTraverseOp::new(
                    self,
                    Box::new(child),
                    relationship_pattern,
                    *emit_relationship,
                    sibling_edges,
                    *transposed,
                    chain,
                    *optional,
                    *bind_relationship,
                    idx,
                    record_cap,
                )))
            }
            IR::ExpandInto {
                relationship: relationship_pattern,
                emit_relationship,
                sibling_edges,
            } => {
                // Account for both limit and skip so the traverse produces
                // enough rows for a downstream SkipOp + LimitOp pipeline.
                let record_cap = self.record_cap(idx);
                let child = pop_or_once(&mut children);
                Ok(BatchOp::ExpandInto(ExpandIntoOp::new(
                    self,
                    Box::new(child),
                    relationship_pattern,
                    *emit_relationship,
                    sibling_edges,
                    idx,
                    record_cap,
                )))
            }
            IR::NodeByIdSeek { node, filter } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::NodeByIdSeek(NodeByIdSeekOp::new(
                    self,
                    Box::new(child),
                    node,
                    filter,
                    idx,
                )))
            }
            IR::NodeByIndexScan { node, index, query } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::NodeByIndexScan(NodeByIndexScanOp::new(
                    self,
                    Box::new(child),
                    node,
                    index,
                    query,
                    idx,
                )))
            }
            IR::EdgeByIndexScan {
                relationship,
                query,
                transposed,
            } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::EdgeByIndexScan(EdgeByIndexScanOp::new(
                    self,
                    Box::new(child),
                    relationship,
                    query,
                    *transposed,
                    idx,
                )))
            }
            IR::CartesianProduct => {
                let child = pop_or_once(&mut children);
                let right_children: Vec<BatchOp<'_>> = children.collect();
                Ok(BatchOp::CartesianProduct(CartesianProductOp::new(
                    self,
                    Box::new(child),
                    right_children,
                    idx,
                )))
            }
            IR::ValueHashJoin { lhs_exp, rhs_exp } => {
                let child = pop_or_once(&mut children);
                let right = children
                    .next()
                    .ok_or_else(|| String::from("ValueHashJoin missing right child"))?;
                Ok(BatchOp::ValueHashJoin(ValueHashJoinOp::new(
                    self,
                    Box::new(child),
                    Box::new(right),
                    lhs_exp,
                    rhs_exp,
                    idx,
                )))
            }
            IR::Apply => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Apply(ApplyOp::new(self, Box::new(child), idx)))
            }
            IR::SemiApply | IR::AntiSemiApply => {
                let is_anti = matches!(self.plan.node(idx).data(), IR::AntiSemiApply);
                let child = pop_or_once(&mut children);
                Ok(BatchOp::SemiApply(SemiApplyOp::new(
                    self,
                    Box::new(child),
                    is_anti,
                    idx,
                )))
            }
            IR::Optional(vars) => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Optional(OptionalOp::new(
                    self,
                    Box::new(child),
                    vars,
                    idx,
                )))
            }
            IR::Create(pattern) => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Create(CreateOp::new(
                    self,
                    Box::new(child),
                    pattern,
                    idx,
                )))
            }
            IR::Delete { exprs: trees, .. } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Delete(DeleteOp::new(
                    self,
                    Box::new(child),
                    trees,
                    idx,
                )))
            }
            IR::Set(items) => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Set(SetOp::new(self, Box::new(child), items, idx)))
            }
            IR::Remove(items) => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Remove(RemoveOp::new(
                    self,
                    Box::new(child),
                    items,
                    idx,
                )))
            }
            IR::Merge {
                pattern,
                on_create: on_create_set_items,
                on_match: on_match_set_items,
            } => {
                let child = pop_or_argument(&mut children);
                Ok(BatchOp::Merge(MergeOp::new(
                    self,
                    Box::new(child),
                    pattern,
                    on_create_set_items,
                    on_match_set_items,
                    idx,
                )))
            }
            IR::Commit => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::Commit(CommitOp::new(self, Box::new(child), idx)?))
            }
            IR::ForEach { list, var } => {
                // ForEach has 1 or 2 children:
                //   - If 2 children: child(0) = input from preceding clause, child(1) = body sub-plan
                //   - If 1 child: child(0) = body sub-plan, input comes via Argument
                //     (Argument allows set_argument_batch to inject the parent env)
                let child = pop_or_argument(&mut children);
                Ok(BatchOp::ForEach(ForEachOp::new(
                    self,
                    Box::new(child),
                    list,
                    var,
                    idx,
                )))
            }
            IR::Union => Ok(BatchOp::Union(UnionOp::new(self, idx))),
            IR::PathBuilder(paths) => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::PathBuilder(PathBuilderOp::new(
                    self,
                    Box::new(child),
                    paths,
                    idx,
                )))
            }
            IR::LoadCsv {
                file_path,
                headers,
                delimiter,
                var,
            } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::LoadCsv(LoadCsvOp::new(
                    self,
                    Box::new(child),
                    file_path,
                    headers,
                    delimiter,
                    var,
                    idx,
                )))
            }
            IR::ProcedureCall {
                func,
                args: trees,
                yields: name_outputs,
            } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::ProcedureCall(ProcedureCallOp::new(
                    self,
                    Box::new(child),
                    func,
                    trees,
                    name_outputs,
                    idx,
                )?))
            }
            IR::NodeByFulltextScan {
                node,
                label,
                query,
                score,
            } => {
                // A downstream Skip/Limit bounds how many results are needed;
                // pass it through so the scan drains just enough index results.
                let record_cap = self.record_cap(idx);
                let child = pop_or_once(&mut children);
                Ok(BatchOp::NodeByFulltextScan(NodeByFulltextScanOp::new(
                    self,
                    Box::new(child),
                    node,
                    label,
                    query,
                    score,
                    record_cap,
                    idx,
                )))
            }
            IR::EdgeByFulltextScan {
                edge,
                label,
                query,
                score,
            } => {
                // A downstream Skip/Limit bounds how many results are needed;
                // pass it through so the scan drains just enough index results.
                let record_cap = self.record_cap(idx);
                let child = pop_or_once(&mut children);
                Ok(BatchOp::EdgeByFulltextScan(EdgeByFulltextScanOp::new(
                    self,
                    Box::new(child),
                    edge,
                    label,
                    query,
                    score,
                    record_cap,
                    idx,
                )))
            }
            IR::NodeByVectorScan {
                node,
                label,
                attr,
                k,
                vector,
                score,
            } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::NodeByVectorScan(NodeByVectorScanOp::new(
                    self,
                    Box::new(child),
                    node,
                    label,
                    attr,
                    k,
                    vector,
                    score,
                    idx,
                )))
            }
            IR::EdgeByVectorScan {
                edge,
                label,
                attr,
                k,
                vector,
                score,
            } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::EdgeByVectorScan(EdgeByVectorScanOp::new(
                    self,
                    Box::new(child),
                    edge,
                    label,
                    attr,
                    k,
                    vector,
                    score,
                    idx,
                )))
            }
            IR::NodeByLabelAndIdScan { node, filter } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::NodeByLabelAndIdScan(NodeByLabelAndIdScanOp::new(
                    self,
                    Box::new(child),
                    node,
                    filter,
                    idx,
                )))
            }
            IR::CondVarLenTraverse {
                relationship: relationship_pattern,
                edge_filter,
                emit_path,
                path_var,
                ..
            } => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::CondVarLenTraverse(CondVarLenTraverseOp::new(
                    self,
                    Box::new(child),
                    relationship_pattern,
                    edge_filter.as_ref(),
                    *emit_path,
                    path_var.as_ref().map(|v| v.id),
                    idx,
                )))
            }
            IR::AllShortestPaths(relationship_pattern) => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::AllShortestPaths(AllShortestPathsOp::new(
                    self,
                    Box::new(child),
                    relationship_pattern,
                    idx,
                )))
            }
            IR::OrApplyMultiplexer(anti_flags) => {
                let child = pop_or_once(&mut children);
                Ok(BatchOp::OrApplyMultiplexer(OrApplyMultiplexerOp::new(
                    self,
                    Box::new(child),
                    anti_flags,
                    idx,
                )))
            }
            IR::Argument(_) => Ok(BatchOp::Argument(Some(self.default_batch()))),
            IR::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options,
            } => {
                if !self.write {
                    return Err(String::from(
                        "graph.RO_QUERY is to be executed only on read-only queries",
                    ));
                }
                let index_options = match options {
                    Some(expr) => {
                        let val = {
                            let this = &self;
                            let idx = expr.root().idx();
                            super::eval::ExprEval::from_runtime(this).eval(
                                expr,
                                idx,
                                super::eval::NO_ROW,
                                None,
                            )
                        }?;
                        match val {
                            Value::Map(map) => map_to_index_options(index_type, &map)?,
                            _ => return Err("Index options must be a map".into()),
                        }
                    }
                    None => None,
                };
                // Index DDL mutates the shared, non-MVCC index directly (not via
                // `pending`) and calls host FFI that needs the global lock, so
                // become a writer first — same contract as `CommitOp`.
                self.write_escalation().upgrade_to_write()?;
                self.g.borrow_mut().create_index(
                    index_type,
                    entity_type,
                    label,
                    attrs,
                    index_options,
                )?;
                self.stats.borrow_mut().indexes_created += attrs.len();
                Ok(BatchOp::Once(None))
            }
            IR::DropIndex {
                label,
                attrs,
                index_type,
                entity_type,
            } => {
                if !self.write {
                    return Err(String::from(
                        "graph.RO_QUERY is to be executed only on read-only queries",
                    ));
                }

                // See `CreateIndex` above: DDL runs in writer mode.
                self.write_escalation().upgrade_to_write()?;
                let dropped =
                    self.g
                        .borrow_mut()
                        .drop_index(index_type, entity_type, label, attrs)?;
                self.stats.borrow_mut().indexes_dropped += dropped;
                Ok(BatchOp::Once(None))
            }
        }
    }

    pub fn evaluate_id_filter<R: super::row::RowView + ?Sized>(
        &self,
        filter: &Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
        vars: &R,
    ) -> Result<Option<RoaringTreemap>, String> {
        let mut min = 0u64;
        let mut max = self.g.borrow().max_node_id();
        for (expr, op) in filter {
            let id = match {
                let this = &self;
                let idx = expr.root().idx();
                super::eval::ExprEval::from_runtime(this).eval(expr, idx, Some(vars), None)
            }? {
                Value::Int(id) => id,
                _ => {
                    return Err(String::from("Node ID must be an integer"));
                }
            };
            // IDs are non-negative: negative equality/upper bounds cannot match,
            // while negative lower bounds are trivially satisfied.
            if id < 0 {
                match op {
                    ExprIR::Eq | ExprIR::Lt | ExprIR::Le => return Ok(None),
                    ExprIR::Gt | ExprIR::Ge => continue,
                    _ => unreachable!(),
                }
            }
            let id = id as u64;
            match op {
                ExprIR::Eq => {
                    if id < min || id > max {
                        return Ok(None);
                    }
                    min = id;
                    max = id;
                }
                ExprIR::Gt => {
                    if id >= max {
                        return Ok(None);
                    }
                    min = std::cmp::max(min, id + 1);
                }
                ExprIR::Ge => {
                    if id > max {
                        return Ok(None);
                    }
                    min = std::cmp::max(min, id);
                }
                ExprIR::Lt => {
                    if id <= min {
                        return Ok(None);
                    }
                    max = std::cmp::min(max, id - 1);
                }
                ExprIR::Le => {
                    if id < min {
                        return Ok(None);
                    }
                    max = std::cmp::min(max, id);
                }
                _ => {
                    unreachable!()
                }
            }
        }
        let mut result = RoaringTreemap::new();
        result.insert_range(min..=max);
        Ok(Some(result))
    }

    /// Resolve an attribute name to its id (creating it if needed) and
    /// record the pending node property change.
    pub(crate) fn set_pending_node_attr(
        &self,
        id: NodeId,
        key: &Arc<String>,
        value: Value,
    ) -> Result<(), String> {
        validate_identifier_len(key, "Property name")?;
        let attr_id = Self::memo_lookup(&self.node_attr_id_memo, key).unwrap_or_else(|| {
            let attr_id = self.g.borrow_mut().get_or_create_node_attr_id(key);
            Self::memo_insert(&self.node_attr_id_memo, key, attr_id);
            attr_id
        });
        self.pending
            .borrow_mut()
            .set_node_attribute(id, attr_id, value)
    }

    /// Resolve an attribute name to its id (creating it if needed) and
    /// record the pending relationship property change.
    pub(crate) fn set_pending_relationship_attr(
        &self,
        id: RelationshipId,
        key: &Arc<String>,
        value: Value,
    ) -> Result<(), String> {
        validate_identifier_len(key, "Property name")?;
        let attr_id = Self::memo_lookup(&self.rel_attr_id_memo, key).unwrap_or_else(|| {
            let attr_id = self.g.borrow_mut().get_or_create_rel_attr_id(key);
            Self::memo_insert(&self.rel_attr_id_memo, key, attr_id);
            attr_id
        });
        self.pending
            .borrow_mut()
            .set_relationship_attribute(id, attr_id, value)
    }

    fn memo_lookup(
        memo: &RefCell<Vec<(Arc<String>, u16)>>,
        attr: &Arc<String>,
    ) -> Option<u16> {
        memo.borrow()
            .iter()
            .find(|(name, _)| Arc::ptr_eq(name, attr))
            .map(|&(_, id)| id)
    }

    fn memo_insert(
        memo: &RefCell<Vec<(Arc<String>, u16)>>,
        attr: &Arc<String>,
        id: u16,
    ) {
        let mut memo = memo.borrow_mut();
        // Bounded memo with no eviction: once full, further names simply
        // aren't memoized and fall back to the graph's name table lookup.
        if memo.len() < ATTR_ID_MEMO_CAP {
            memo.push((attr.clone(), id));
        }
    }

    /// Memoized `Graph::get_node_attr_id`. Never memoizes a miss: the id
    /// may be created later in the same query.
    fn node_attr_id(
        &self,
        g: &Graph,
        attr: &Arc<String>,
    ) -> Option<u16> {
        if let Some(id) = Self::memo_lookup(&self.node_attr_id_memo, attr) {
            return Some(id);
        }
        let id = g.get_node_attr_id(attr)?;
        Self::memo_insert(&self.node_attr_id_memo, attr, id);
        Some(id)
    }

    /// Memoized `Graph::get_rel_attr_id`.
    fn rel_attr_id(
        &self,
        g: &Graph,
        attr: &Arc<String>,
    ) -> Option<u16> {
        if let Some(id) = Self::memo_lookup(&self.rel_attr_id_memo, attr) {
            return Some(id);
        }
        let id = g.get_rel_attr_id(attr)?;
        Self::memo_insert(&self.rel_attr_id_memo, attr, id);
        Some(id)
    }

    pub fn get_node_attribute(
        &self,
        id: NodeId,
        attr: &Arc<String>,
    ) -> Option<Value> {
        let deleted = self.deleted_nodes.borrow();
        if !deleted.is_empty()
            && let Some(dn) = deleted.get(&id)
        {
            if let Some(value) = dn.attrs.get(attr) {
                return Some(value.clone());
            }
            return None;
        }
        drop(deleted);
        self.get_node_attribute_no_delete_check(id, attr)
    }

    /// Like `get_node_attribute` but skips the deleted_nodes check.
    /// Use when the caller has already verified no deletions exist.
    pub fn get_node_attribute_no_delete_check(
        &self,
        id: NodeId,
        attr: &Arc<String>,
    ) -> Option<Value> {
        let g = self.g.borrow();
        let attr_id = self.node_attr_id(&g, attr)?;
        if let Some(value) = self.pending.borrow().get_node_attribute(id, attr_id) {
            return Some(value.clone());
        }
        g.get_node_attribute_by_idx(id, attr_id)
    }

    /// Like `get_relationship_attribute` but skips the deleted check.
    pub fn get_relationship_attribute_no_delete_check(
        &self,
        id: RelationshipId,
        attr: &Arc<String>,
    ) -> Option<Value> {
        let g = self.g.borrow();
        let attr_id = self.rel_attr_id(&g, attr)?;
        if let Some(value) = self
            .pending
            .borrow()
            .get_relationship_attribute(id, attr_id)
        {
            return Some(value.clone());
        }
        g.get_relationship_attribute_by_idx(id, attr_id)
    }

    pub fn get_relationship_attribute(
        &self,
        id: RelationshipId,
        attr: &Arc<String>,
    ) -> Option<Value> {
        let deleted = self.deleted_relationships.borrow();
        if !deleted.is_empty()
            && let Some(dn) = deleted.get(&id)
        {
            if let Some(value) = dn.attrs.get(attr) {
                return Some(value.clone());
            }
            return None;
        }
        drop(deleted);
        let g = self.g.borrow();
        if let Some(attr_id) = self.rel_attr_id(&g, attr)
            && let Some(value) = self
                .pending
                .borrow()
                .get_relationship_attribute(id, attr_id)
        {
            return Some(value.clone());
        }
        g.get_relationship_attribute(id, attr)
    }

    /// Materializes a property column for a batch of node IDs.
    ///
    /// Resolves the attribute index once, then fetches the value for each node.
    /// Checks deleted_nodes and pending mutations (same as `get_node_attribute`).
    /// Returns a typed Column plus a NullBitmap.
    pub fn materialize_node_property(
        &self,
        node_ids: &[NodeId],
        attr: &Arc<String>,
    ) -> (Column, NullBitmap) {
        classify_column(self.materialize_node_property_values(node_ids, attr))
    }

    /// Like [`materialize_node_property`](Self::materialize_node_property) but
    /// returns the raw per-node values without classifying them into a typed
    /// column. Callers that must preserve exact value types — e.g. join-key
    /// evaluation, where coercing a mixed int/float column to all-float would
    /// lose integer precision past 2^53 and change which keys compare equal —
    /// use this and classify losslessly themselves.
    pub fn materialize_node_property_values(
        &self,
        node_ids: &[NodeId],
        attr: &Arc<String>,
    ) -> Vec<Value> {
        let g = self.g.borrow();
        let attr_idx = self.node_attr_id(&g, attr);

        let deleted = self.deleted_nodes.borrow();
        let pending = self.pending.borrow();

        let mut values = Vec::with_capacity(node_ids.len());
        if deleted.is_empty() && !pending.has_node_attrs() {
            // Hot read-only path: a single batch call covers all node ids.
            if let Some(idx) = attr_idx {
                g.get_node_attributes_by_idx(node_ids, idx, &Value::Null, &mut values);
            } else {
                values.resize(node_ids.len(), Value::Null);
            }
        } else {
            for &id in node_ids {
                let val = deleted.get(&id).map_or_else(
                    || {
                        attr_idx
                            .and_then(|idx| {
                                pending
                                    .get_node_attribute(id, idx)
                                    .cloned()
                                    .or_else(|| g.get_node_attribute_by_idx(id, idx))
                            })
                            .unwrap_or(Value::Null)
                    },
                    |dn| dn.attrs.get(attr).cloned().unwrap_or(Value::Null),
                );
                values.push(val);
            }
        }
        drop(g);
        drop(deleted);
        drop(pending);

        values
    }

    /// Bulk-read `attr` for `rel_ids`, the relationship counterpart of
    /// [`materialize_node_property`](Self::materialize_node_property). Same
    /// shape, same reason: on the read-only path a single batched attribute-store
    /// call covers every id, instead of one lookup per row.
    pub fn materialize_relationship_property(
        &self,
        rel_ids: &[RelationshipId],
        attr: &Arc<String>,
    ) -> (Column, NullBitmap) {
        classify_column(self.materialize_relationship_property_values(rel_ids, attr))
    }

    /// Like
    /// [`materialize_relationship_property`](Self::materialize_relationship_property)
    /// but returns the raw per-relationship values without classifying them
    /// into a typed column, for callers that must preserve exact value types.
    pub fn materialize_relationship_property_values(
        &self,
        rel_ids: &[RelationshipId],
        attr: &Arc<String>,
    ) -> Vec<Value> {
        let g = self.g.borrow();
        let attr_idx = self.rel_attr_id(&g, attr);

        let deleted = self.deleted_relationships.borrow();
        let pending = self.pending.borrow();

        let mut values = Vec::with_capacity(rel_ids.len());
        if deleted.is_empty() && !pending.has_relationship_attrs() {
            // Hot read-only path: a single batch call covers all relationship ids.
            if let Some(idx) = attr_idx {
                g.get_relationship_attributes_by_idx(rel_ids, idx, &Value::Null, &mut values);
            } else {
                values.resize(rel_ids.len(), Value::Null);
            }
        } else {
            for &id in rel_ids {
                let val = deleted.get(&id).map_or_else(
                    || {
                        attr_idx
                            .and_then(|idx| {
                                pending
                                    .get_relationship_attribute(id, idx)
                                    .cloned()
                                    .or_else(|| g.get_relationship_attribute_by_idx(id, idx))
                            })
                            .unwrap_or(Value::Null)
                    },
                    |dr| dr.attrs.get(attr).cloned().unwrap_or(Value::Null),
                );
                values.push(val);
            }
        }
        drop(g);
        drop(deleted);
        drop(pending);

        values
    }

    pub fn get_node_labels(
        &self,
        id: NodeId,
    ) -> OrderSet<Arc<String>> {
        let g = self.g.borrow();
        if let Some(dn) = self.deleted_nodes.borrow().get(&id) {
            return dn.labels.iter().map(|l| g.get_label_by_id(*l)).collect();
        }
        let mut labels = g.get_node_label_ids(id).collect::<OrderSet<_>>();
        self.pending.borrow().update_node_labels(id, &mut labels);

        labels.iter().map(|l| g.get_label_by_id(*l)).collect()
    }

    /// Whether `id` carries the label named `name` — the whole answer, for
    /// every node this query can see, without building the node's label set.
    ///
    /// This is the single place a label test is decided, and it decides it in
    /// the same order [`Self::get_node_labels`] does: a node deleted by this
    /// query answers from the labels captured at the delete, a label this query
    /// staged answers from the staged state, and everything else is one bit of
    /// the committed label matrix.
    ///
    /// `n:Person` reaches the runtime as a `hasLabels` call, and answering it
    /// through `get_node_labels` costs two `OrderSet`s and an `Arc<String>`
    /// clone per stored label, then compares label *names*, for every row. On
    /// the benchmark graph `MATCH (n) WHERE n:Person RETURN count(n)` spent
    /// 48.9M instructions over 15k nodes — 3.3k a row — against 22.4M on the C
    /// engine, while the same scan filtering on a property (`n.id < 5`) costs
    /// 2.0M. A label test is one bit in the label matrix; this reads that bit.
    pub fn node_has_label(
        &self,
        id: NodeId,
        name: &str,
    ) -> bool {
        // A name the graph has never registered is on no node at all, not even
        // one this query just created: `CREATE (:L)` registers `L` before it
        // stages the label. Resolving a known name is a walk over the label
        // names — a handful of entries, no allocation.
        let Some(label_id) = self.label_id(name) else {
            return false;
        };
        self.node_has_label_id(id, label_id)
    }

    /// The registered id for a label name, or `None` when the graph has never
    /// seen it.
    ///
    /// Split out so a caller testing the same label across many rows resolves
    /// the name once instead of per row — see the `hasLabels` column kernel.
    #[must_use]
    pub fn label_id(
        &self,
        name: &str,
    ) -> Option<LabelId> {
        self.g.borrow().get_label_id(name)
    }

    /// [`Self::node_has_label`] with the name already resolved.
    ///
    /// Same three-way answer in the same order: a node this query deleted
    /// answers from the labels captured at the delete, a label this query
    /// staged answers from the staged state, and everything else is one bit of
    /// the committed label matrix.
    #[must_use]
    pub fn node_has_label_id(
        &self,
        id: NodeId,
        label_id: LabelId,
    ) -> bool {
        if let Some(deleted) = self.deleted_nodes.borrow().get(&id) {
            return deleted.labels.contains(&label_id);
        }
        self.pending
            .borrow()
            .node_has_label(id, label_id)
            .unwrap_or_else(|| self.g.borrow().node_has_label_id(id, label_id))
    }

    pub fn get_node_attrs(
        &self,
        id: NodeId,
    ) -> OrderMap<Arc<String>, Value> {
        if let Some(dn) = self.deleted_nodes.borrow().get(&id) {
            let attrs = dn
                .attrs
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            return attrs;
        }
        let g = self.g.borrow();
        let mut actual = OrderMap::from_unique_keys(g.get_node_all_attrs(id));
        self.pending.borrow().update_node_attrs(id, &mut actual, &g);
        actual
    }

    pub fn get_relationship_attrs(
        &self,
        id: RelationshipId,
    ) -> OrderMap<Arc<String>, Value> {
        if let Some(dr) = self.deleted_relationships.borrow().get(&id) {
            let attrs = dr
                .attrs
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            return attrs;
        }
        let g = self.g.borrow();
        let mut actual = OrderMap::from_unique_keys(g.get_relationship_all_attrs(id));
        self.pending
            .borrow()
            .update_relationship_attrs(id, &mut actual, &g);
        actual
    }

    pub fn get_relationship_endpoints(
        &self,
        id: RelationshipId,
    ) -> (NodeId, NodeId) {
        if let Some(dr) = self.deleted_relationships.borrow().get(&id) {
            return (dr.src, dr.dst);
        }
        if let Some(endpoints) = self.pending.borrow().get_created_relationship_endpoints(id) {
            return endpoints;
        }
        self.g.borrow().get_relationship_endpoints(id)
    }

    pub fn get_relationship_type(
        &self,
        id: RelationshipId,
    ) -> Option<Arc<String>> {
        if let Some(dr) = self.deleted_relationships.borrow().get(&id) {
            return Some(dr.type_name.clone());
        }
        if let Some(type_name) = self.pending.borrow().get_relationship_type(id) {
            return Some(type_name);
        }
        let g = self.g.borrow();
        g.get_type(g.get_relationship_type_id(id))
    }

    pub fn get_node_indegree(
        &self,
        id: NodeId,
    ) -> usize {
        if self.deleted_nodes.borrow().contains_key(&id)
            || self.pending.borrow().is_node_deleted(id)
        {
            return 0;
        }
        let g = self.g.borrow();
        let base = g.get_node_indegree(id);
        let pending = self.pending.borrow();
        let added = pending.pending_indegree(id, &[]);
        let removed = pending.pending_deleted_indegree(id, &[], &g);
        base + added - removed
    }

    pub fn get_node_indegree_by_type(
        &self,
        id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        if self.deleted_nodes.borrow().contains_key(&id)
            || self.pending.borrow().is_node_deleted(id)
        {
            return 0;
        }
        let g = self.g.borrow();
        let base = g.get_node_indegree_by_type(id, types);
        let pending = self.pending.borrow();
        let added = pending.pending_indegree(id, types);
        let removed = pending.pending_deleted_indegree(id, types, &g);
        base + added - removed
    }

    pub fn get_node_outdegree(
        &self,
        id: NodeId,
    ) -> usize {
        if self.deleted_nodes.borrow().contains_key(&id)
            || self.pending.borrow().is_node_deleted(id)
        {
            return 0;
        }
        let g = self.g.borrow();
        let base = g.get_node_outdegree(id);
        let pending = self.pending.borrow();
        let added = pending.pending_outdegree(id, &[]);
        let removed = pending.pending_deleted_outdegree(id, &[], &g);
        base + added - removed
    }

    pub fn get_node_outdegree_by_type(
        &self,
        id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        if self.deleted_nodes.borrow().contains_key(&id)
            || self.pending.borrow().is_node_deleted(id)
        {
            return 0;
        }
        let g = self.g.borrow();
        let base = g.get_node_outdegree_by_type(id, types);
        let pending = self.pending.borrow();
        let added = pending.pending_outdegree(id, types);
        let removed = pending.pending_deleted_outdegree(id, types, &g);
        base + added - removed
    }
}

fn map_to_index_options(
    index_type: &IndexType,
    kv_map: &OrderMap<Arc<String>, Value>,
) -> Result<Option<IndexOptions>, String> {
    let get = |key: &str| -> Option<&Value> {
        kv_map
            .iter()
            .find_map(|(k, v)| if k.as_str() == key { Some(v) } else { None })
    };
    match index_type {
        IndexType::Fulltext => {
            let weight = match get("weight") {
                Some(Value::Float(f)) => Some(*f),
                Some(Value::Int(i)) => Some(*i as f64),
                None => None,
                _ => return Err("Weight must be numeric".into()),
            };
            let nostem = match get("nostem") {
                Some(Value::Bool(b)) => Some(*b),
                None => None,
                _ => return Err("Nostem must be bool".into()),
            };
            // Phonetic accepts either a bool (true = enable, false =
            // disable) or the algorithm code 'dm:en'. The Rust binding
            // sets only RediSearch's default phonetic flag, which maps
            // to Double Metaphone English — other algorithm codes
            // (dm:fr / dm:pt / dm:es) aren't wired up here.
            let phonetic = match get("phonetic") {
                Some(Value::Bool(b)) => Some(*b),
                Some(Value::String(s)) => {
                    if s.eq_ignore_ascii_case("dm:en") {
                        Some(true)
                    } else {
                        return Err(format!(
                            "Unsupported phonetic algorithm '{s}'; only 'dm:en' is supported"
                        ));
                    }
                }
                None => None,
                _ => return Err("Phonetic must be bool or string".into()),
            };
            let language = match get("language") {
                Some(Value::String(s)) => Some(s.clone()),
                None => None,
                _ => return Err("Language must be string".into()),
            };
            let stopwords = match get("stopwords") {
                Some(Value::List(list)) => {
                    let mut words = Vec::with_capacity(list.len());
                    for v in list.iter() {
                        match v {
                            Value::String(s) => words.push(s.clone()),
                            _ => {
                                return Err("Stopwords must be an array of strings".into());
                            }
                        }
                    }
                    Some(words)
                }
                None => None,
                _ => return Err("Stopwords must be array".into()),
            };
            let options = IndexOptions::Text(TextIndexOptions {
                weight,
                nostem,
                phonetic,
                language,
                stopwords,
            });
            Ok(Some(options))
        }
        IndexType::Range => Ok(None),
        IndexType::Vector => {
            let dimension = match get("dimension") {
                Some(Value::Int(n)) => {
                    if *n < 0 {
                        return Err("Invalid vector index configuration: dimension must be a non-negative integer".into());
                    }
                    *n as u32
                }
                None => 0,
                _ => {
                    return Err(
                        "Invalid vector index configuration: dimension must be an integer".into(),
                    );
                }
            };
            let similarity_function =
                match get("similarityFunction") {
                    Some(Value::String(s)) => Some(s.to_string()),
                    None => None,
                    _ => return Err(
                        "Invalid vector index configuration: similarityFunction must be a string"
                            .into(),
                    ),
                };
            let m = match get("M") {
                Some(Value::Int(n)) if *n < 0 => {
                    return Err(
                        "Invalid vector index configuration: M must be a non-negative integer"
                            .into(),
                    );
                }
                Some(Value::Int(n)) => Some(*n as usize),
                None => None,
                _ => return Err("Invalid vector index configuration: M must be an integer".into()),
            };
            let ef_construction = match get("efConstruction") {
                Some(Value::Int(n)) if *n < 0 => {
                    return Err("Invalid vector index configuration: efConstruction must be a non-negative integer".into());
                }
                Some(Value::Int(n)) => Some(*n as usize),
                None => None,
                _ => {
                    return Err(
                        "Invalid vector index configuration: efConstruction must be an integer"
                            .into(),
                    );
                }
            };
            let ef_runtime = match get("efRuntime") {
                Some(Value::Int(n)) if *n < 0 => {
                    return Err("Invalid vector index configuration: efRuntime must be a non-negative integer".into());
                }
                Some(Value::Int(n)) => Some(*n as usize),
                None => None,
                _ => {
                    return Err(
                        "Invalid vector index configuration: efRuntime must be an integer".into(),
                    );
                }
            };
            Ok(Some(IndexOptions::Vector(VectorIndexOptions {
                dimension,
                similarity_function,
                m,
                ef_construction,
                ef_runtime,
            })))
        }
    }
}
