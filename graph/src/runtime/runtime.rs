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
//!  query() drives the root BatchOp, collecting Env rows into ResultSummary.
//! ```
//!
//! ## Key Types
//!
//! - [`Runtime`]: Main execution context (carries `Pool`, graph ref, plan)
//! - [`ResultSummary`]: Query result with collected rows and statistics
//! - [`BatchOp`]: Enum-dispatch operator tree (28+ variants)
//! - [`Batch`]: Columnar/env-backed batch of up to 1024 rows
//! - [`Env`]: Tuple of variable bindings (pool-backed)
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
    graph::graph::{Graph, NodeId, RelationshipId},
    index::indexer::{IndexOptions, IndexType, TextIndexOptions, VectorIndexOptions},
    parser::ast::{ExprIR, QueryExpr, Variable},
    planner::IR,
    runtime::{
        batch::{Batch, BatchOp, Column, NullBitmap, classify_column},
        bitset::BitSet,
        env::Env,
        ops::{
            AggregateOp, AllShortestPathsOp, ApplyOp, CartesianProductOp, CommitOp, CondTraverseOp,
            CondVarLenTraverseOp, CreateOp, DeleteOp, DistinctOp, EdgeByFulltextScanOp,
            EdgeByIndexScanOp, ExpandIntoOp, FilterOp, ForEachOp, LimitOp, LoadCsvOp, MergeOp,
            NodeByFulltextScanOp, NodeByIdSeekOp, NodeByIndexScanOp, NodeByLabelAndIdScanOp,
            NodeByLabelScanOp, OptionalOp, OrApplyMultiplexerOp, PathBuilderOp, ProcedureCallOp,
            ProjectOp, RemoveOp, SemiApplyOp, SetOp, SkipOp, SortOp, UnionOp, UnwindOp,
            ValueHashJoinOp,
        },
        ordermap::OrderMap,
        orderset::OrderSet,
        pending::Pending,
        pool::Pool,
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
    fmt::Debug,
    sync::Arc,
    time::{Duration, Instant},
};

pub use super::eval::ValueIter;

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
    /// Deduplication state for DISTINCT operations
    pub value_dedupers: RefCell<HashMap<String, ValuesDeduper>>,
    /// Variables to return in query results
    pub return_names: Vec<Variable>,
    /// Debug mode: record operator execution
    pub inspect: bool,
    /// Debug records of operator execution
    pub record: RefCell<Vec<(NodeIdx<Dyn<IR>>, Result<(Vec<Value>, BitSet), String>)>>,
    /// Folder for LOAD CSV operations
    pub import_folder: String,
    /// Cache of deleted nodes for result consistency
    pub deleted_nodes: RefCell<HashMap<NodeId, DeletedNode>>,
    /// Cache of deleted relationships for result consistency
    pub deleted_relationships: RefCell<HashMap<RelationshipId, DeletedRelationship>>,
    /// Cache for MERGE pattern matching — stores only the created entity bindings (variable id → value)
    pub merge_pattern_cache: RefCell<HashMap<u64, Vec<(u32, Value)>>>,
    /// Per-query object pool for Env backing Vec<Value> buffers.
    /// Owned externally and borrowed here to avoid self-referential lifetimes.
    pub env_pool: &'a Pool<Value>,
    /// Maximum number of result rows to return. Negative means unlimited.
    pub result_set_size: i64,
    /// Effects buffer built before commit, for replication.
    pub effects_buffer: RefCell<Option<Vec<u8>>>,
    /// Total number of effect records across all commits in this query.
    pub effects_count: Cell<u64>,
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
                | IR::Argument
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
                | IR::CreateIndex { .. }
                | IR::DropIndex { .. } => {}
                IR::NodeByLabelScan(node)
                | IR::AllNodeScan(node)
                | IR::NodeByIndexScan { node, .. }
                | IR::NodeByLabelAndIdScan { node, .. }
                | IR::NodeByIdSeek { node, .. } => {
                    vars.push(node.alias.clone());
                }
                IR::NodeByFulltextScan { node, score, .. } => {
                    vars.push(node.clone());
                    if let Some(score) = score {
                        vars.push(score.clone());
                    }
                }
                IR::EdgeByFulltextScan { edge, score, .. } => {
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
                | IR::CondVarLenTraverse {
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
            IR::NodeByFulltextScan { node, score, .. } => {
                let mut v = vec![node.clone()];
                if let Some(score) = score {
                    v.push(score.clone());
                }
                v
            }
            IR::EdgeByFulltextScan { edge, score, .. } => {
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

impl Debug for Env<'_> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_list().entries(self.as_ref().iter()).finish()
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
                    for env in batch.active_env_iter() {
                        record.push((idx, Ok(env.to_raw())));
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
        env_pool: &'a Pool<Value>,
        result_set_size: i64,
        profile: bool,
        timeout_ms: Option<u64>,
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
            return_names,
            value_dedupers: RefCell::new(HashMap::new()),
            inspect,
            record: RefCell::new(vec![]),
            import_folder,
            deleted_nodes: RefCell::new(HashMap::new()),
            deleted_relationships: RefCell::new(HashMap::new()),
            merge_pattern_cache: RefCell::new(HashMap::new()),
            env_pool,
            result_set_size,
            effects_buffer: RefCell::new(None),
            effects_count: Cell::new(0),
            transaction_timestamp: Utc::now(),
            profile,
            profile_data: RefCell::new(HashMap::new()),
            profile_child_time: Cell::new(Duration::ZERO),
            deadline: timeout_ms.map(|ms| Instant::now() + Duration::from_millis(ms)),
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

    /// Apply deferred index operations to RediSearch. Must be called only
    /// after the full query succeeds.
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
        let envs = vec![Env::new(self.env_pool)];
        Batch::from_envs(envs)
    }

    /// Resolves the first child of `idx` into a `BatchOp`, falling back to a
    /// single-row default batch when no child exists.
    fn child_batch_op(
        &'a self,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Result<BatchOp<'a>, String> {
        self.plan.node(idx).get_child(0).map_or_else(
            || Ok(BatchOp::Once(Some(self.default_batch()))),
            |child| self.run_batch(child.idx()),
        )
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
                    let env = Env::new(self.env_pool);
                    let val = super::eval::ExprEval::from_runtime(self)
                        .eval(expr, expr.root().idx(), Some(&env), None)
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
                    let env = Env::new(self.env_pool);
                    let val = super::eval::ExprEval::from_runtime(self)
                        .eval(expr, expr.root().idx(), Some(&env), None)
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

    /// Builds a batch-mode operator tree for the given IR node.
    pub fn run_batch(
        &'a self,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Result<BatchOp<'a>, String> {
        match self.plan.node(idx).data() {
            IR::NodeByLabelScan(_) | IR::AllNodeScan(_) => {
                let child = self.child_batch_op(idx)?;
                let (IR::NodeByLabelScan(node_pattern) | IR::AllNodeScan(node_pattern)) =
                    self.plan.node(idx).data()
                else {
                    unreachable!()
                };
                Ok(BatchOp::NodeByLabelScan(NodeByLabelScanOp::new(
                    self,
                    Box::new(child),
                    node_pattern,
                    idx,
                )))
            }
            IR::Filter(tree) => {
                let child = self.child_batch_op(idx)?;
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
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::Project(ProjectOp::new(
                    self,
                    Box::new(child),
                    trees,
                    copy_from_parent,
                    idx,
                )))
            }
            IR::Skip(skip) => {
                let child = self.child_batch_op(idx)?;
                let Value::Int(skip) = {
                    let this = &self;
                    let idx = skip.root().idx();
                    let env: &Env<'_> = &Env::new(self.env_pool);
                    super::eval::ExprEval::from_runtime(this).eval(skip, idx, Some(env), None)
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
                let child = self.child_batch_op(idx)?;
                let Value::Int(limit) = {
                    let this = &self;
                    let idx = limit.root().idx();
                    let env: &Env<'_> = &Env::new(self.env_pool);
                    super::eval::ExprEval::from_runtime(this).eval(limit, idx, Some(env), None)
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
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::Distinct(DistinctOp::new(
                    self,
                    Box::new(child),
                    idx,
                )))
            }
            IR::Sort(trees) => {
                let limit = self.effective_limit(idx);
                let skip = self.effective_skip(idx);
                let child = self.child_batch_op(idx)?;
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
                let child = self.child_batch_op(idx)?;
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
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::Unwind(UnwindOp::new(
                    self,
                    Box::new(child),
                    list,
                    name,
                    idx,
                )))
            }
            IR::CondTraverse {
                relationship: relationship_pattern,
                emit_relationship,
                sibling_edges,
                transposed,
            } => {
                // Account for both limit and skip so the traverse produces
                // enough rows for a downstream SkipOp + LimitOp pipeline.
                let record_cap = self
                    .effective_limit(idx)
                    .map(|l| l + self.effective_skip(idx));
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::CondTraverse(CondTraverseOp::new(
                    self,
                    Box::new(child),
                    relationship_pattern,
                    *emit_relationship,
                    sibling_edges,
                    *transposed,
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
                let record_cap = self
                    .effective_limit(idx)
                    .map(|l| l + self.effective_skip(idx));
                let child = self.child_batch_op(idx)?;
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
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::NodeByIdSeek(NodeByIdSeekOp::new(
                    self,
                    Box::new(child),
                    node,
                    filter,
                    idx,
                )))
            }
            IR::NodeByIndexScan { node, index, query } => {
                let child = self.child_batch_op(idx)?;
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
                let child = self.child_batch_op(idx)?;
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
                let child = self.child_batch_op(idx)?;
                let right_children: Vec<BatchOp<'_>> = self
                    .plan
                    .node(idx)
                    .children()
                    .skip(1)
                    .map(|c| self.run_batch(c.idx()))
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(BatchOp::CartesianProduct(CartesianProductOp::new(
                    self,
                    Box::new(child),
                    right_children,
                    idx,
                )))
            }
            IR::ValueHashJoin { lhs_exp, rhs_exp } => {
                let child = self.child_batch_op(idx)?;
                let right = self.run_batch(self.plan.node(idx).child(1).idx())?;
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
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::Apply(ApplyOp::new(self, Box::new(child), idx)))
            }
            IR::SemiApply | IR::AntiSemiApply => {
                let is_anti = matches!(self.plan.node(idx).data(), IR::AntiSemiApply);
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::SemiApply(SemiApplyOp::new(
                    self,
                    Box::new(child),
                    is_anti,
                    idx,
                )))
            }
            IR::Optional(vars) => {
                let child = if self.plan.node(idx).num_children() > 1 {
                    self.run_batch(self.plan.node(idx).child(0).idx())?
                } else {
                    BatchOp::Once(Some(self.default_batch()))
                };
                Ok(BatchOp::Optional(OptionalOp::new(
                    self,
                    Box::new(child),
                    vars,
                    idx,
                )))
            }
            IR::Create(pattern) => {
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::Create(CreateOp::new(
                    self,
                    Box::new(child),
                    pattern,
                    idx,
                )))
            }
            IR::Delete { exprs: trees, .. } => {
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::Delete(DeleteOp::new(
                    self,
                    Box::new(child),
                    trees,
                    idx,
                )))
            }
            IR::Set(items) => {
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::Set(SetOp::new(self, Box::new(child), items, idx)))
            }
            IR::Remove(items) => {
                let child = self.child_batch_op(idx)?;
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
                let child = if self.plan.node(idx).num_children() > 1 {
                    self.run_batch(self.plan.node(idx).child(0).idx())?
                } else {
                    BatchOp::Argument(Some(self.default_batch()))
                };
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
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::Commit(CommitOp::new(self, Box::new(child), idx)?))
            }
            IR::ForEach { list, var } => {
                // ForEach has 1 or 2 children:
                //   - If 2 children: child(0) = input from preceding clause, child(1) = body sub-plan
                //   - If 1 child: child(0) = body sub-plan, input comes via Argument
                //     (Argument allows set_argument_batch to inject the parent env)
                let node = self.plan.node(idx);
                let child = if node.num_children() > 1 {
                    self.run_batch(node.child(0).idx())?
                } else {
                    BatchOp::Argument(Some(self.default_batch()))
                };
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
                let child = self.child_batch_op(idx)?;
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
                let child = self.child_batch_op(idx)?;
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
                let child = self.child_batch_op(idx)?;
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
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::NodeByFulltextScan(NodeByFulltextScanOp::new(
                    self,
                    Box::new(child),
                    node,
                    label,
                    query,
                    score,
                    idx,
                )))
            }
            IR::EdgeByFulltextScan {
                edge,
                label,
                query,
                score,
            } => {
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::EdgeByFulltextScan(EdgeByFulltextScanOp::new(
                    self,
                    Box::new(child),
                    edge,
                    label,
                    query,
                    score,
                    idx,
                )))
            }
            IR::NodeByLabelAndIdScan { node, filter } => {
                let child = self.child_batch_op(idx)?;
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
            } => {
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::CondVarLenTraverse(CondVarLenTraverseOp::new(
                    self,
                    Box::new(child),
                    relationship_pattern,
                    edge_filter.as_ref(),
                    idx,
                )))
            }
            IR::AllShortestPaths(relationship_pattern) => {
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::AllShortestPaths(AllShortestPathsOp::new(
                    self,
                    Box::new(child),
                    relationship_pattern,
                    idx,
                )))
            }
            IR::OrApplyMultiplexer(anti_flags) => {
                let child = self.child_batch_op(idx)?;
                Ok(BatchOp::OrApplyMultiplexer(OrApplyMultiplexerOp::new(
                    self,
                    Box::new(child),
                    anti_flags,
                    idx,
                )))
            }
            IR::Argument => Ok(BatchOp::Argument(Some(self.default_batch()))),
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
                            let env: &Env<'_> = &Env::new(self.env_pool);
                            super::eval::ExprEval::from_runtime(this).eval(
                                expr,
                                idx,
                                Some(env),
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

                let dropped =
                    self.g
                        .borrow_mut()
                        .drop_index(index_type, entity_type, label, attrs)?;
                self.stats.borrow_mut().indexes_dropped += dropped;
                Ok(BatchOp::Once(None))
            }
        }
    }

    pub fn run_iter_expr(
        &self,
        ir: &DynTree<ExprIR<Variable>>,
        idx: NodeIdx<Dyn<ExprIR<Variable>>>,
        env: &Env<'_>,
    ) -> Result<ValueIter, String> {
        super::eval::ExprEval::from_runtime(self).eval_iter_expr(ir, idx, Some(env))
    }

    pub fn evaluate_id_filter(
        &self,
        filter: &Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
        vars: &Env<'_>,
    ) -> Result<Option<RoaringTreemap>, String> {
        let mut min = 0u64;
        let mut max = self.g.borrow().max_node_id();
        for (expr, op) in filter {
            let id = match {
                let this = &self;
                let idx = expr.root().idx();
                super::eval::ExprEval::from_runtime(this).eval(expr, idx, Some(vars), None)
            }? {
                Value::Int(id) => id as u64,
                _ => {
                    return Err(String::from("Node ID must be an integer"));
                }
            };
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

    pub fn get_node_attribute(
        &self,
        id: NodeId,
        attr: &Arc<String>,
    ) -> Option<Value> {
        if let Some(dn) = self.deleted_nodes.borrow().get(&id) {
            if let Some(value) = dn.attrs.get(attr) {
                return Some(value.clone());
            }
            return None;
        }
        self.get_node_attribute_no_delete_check(id, attr)
    }

    /// Like `get_node_attribute` but skips the deleted_nodes check.
    /// Use when the caller has already verified no deletions exist.
    pub fn get_node_attribute_no_delete_check(
        &self,
        id: NodeId,
        attr: &Arc<String>,
    ) -> Option<Value> {
        if let Some(value) = self.pending.borrow().get_node_attribute(id, attr) {
            return Some(value.clone());
        }
        self.g.borrow().get_node_attribute(id, attr)
    }

    /// Like `get_relationship_attribute` but skips the deleted check.
    pub fn get_relationship_attribute_no_delete_check(
        &self,
        id: RelationshipId,
        attr: &Arc<String>,
    ) -> Option<Value> {
        if let Some(value) = self.pending.borrow().get_relationship_attribute(id, attr) {
            return Some(value.clone());
        }
        self.g.borrow().get_relationship_attribute(id, attr)
    }

    pub fn get_relationship_attribute(
        &self,
        id: RelationshipId,
        attr: &Arc<String>,
    ) -> Option<Value> {
        if let Some(dn) = self.deleted_relationships.borrow().get(&id) {
            if let Some(value) = dn.attrs.get(attr) {
                return Some(value.clone());
            }
            return None;
        }
        if let Some(value) = self.pending.borrow().get_relationship_attribute(id, attr) {
            return Some(value.clone());
        }
        self.g.borrow().get_relationship_attribute(id, attr)
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
        let g = self.g.borrow();

        let attr_idx = g.get_node_attribute_id(attr).map(|idx| idx as u16);

        let deleted = self.deleted_nodes.borrow();
        let pending = self.pending.borrow();

        let mut values = Vec::with_capacity(node_ids.len());
        for &id in node_ids {
            let val = deleted.get(&id).map_or_else(
                || {
                    pending.get_node_attribute(id, attr).map_or_else(
                        || {
                            attr_idx
                                .and_then(|idx| g.get_node_attribute_by_idx(id, idx))
                                .unwrap_or(Value::Null)
                        },
                        Clone::clone,
                    )
                },
                |dn| dn.attrs.get(attr).cloned().unwrap_or(Value::Null),
            );
            values.push(val);
        }
        drop(g);
        drop(deleted);
        drop(pending);

        classify_column(values)
    }

    pub fn get_node_labels(
        &self,
        id: NodeId,
    ) -> OrderSet<Arc<String>> {
        if let Some(dn) = self.deleted_nodes.borrow().get(&id) {
            return dn
                .labels
                .iter()
                .map(|l| self.g.borrow().get_label_by_id(*l))
                .collect();
        }
        let mut labels = self
            .g
            .borrow()
            .get_node_label_ids(id)
            .collect::<OrderSet<_>>();
        self.pending.borrow().update_node_labels(id, &mut labels);

        labels
            .iter()
            .map(|l| self.g.borrow().get_label_by_id(*l))
            .collect()
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
        let mut actual = OrderMap::from_vec(self.g.borrow().get_node_all_attrs(id));
        self.pending.borrow().update_node_attrs(id, &mut actual);
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
        let mut actual = OrderMap::from_vec(self.g.borrow().get_relationship_all_attrs(id));
        self.pending
            .borrow()
            .update_relationship_attrs(id, &mut actual);
        actual
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
        self.g
            .borrow()
            .get_type(self.g.borrow().get_relationship_type_id(id))
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
