//! Batch-mode aggregate operator — groups input rows and computes aggregate
//! functions.
//!
//! This is a *blocking* operator: it consumes all batches from the child on
//! the first `next()` call, groups rows by key expressions, runs
//! aggregation functions, and then yields one result row per group in batches.
//!
//! ```text
//!  Input rows (streamed from child)
//!       │
//!       ▼
//!  ┌─────────────────────────────────────┐
//!  │  Group by key exprs (hash map)      │
//!  │  key -> (key_env, accumulator_env)  │
//!  └──────────────────┬──────────────────┘
//!                     │  for each group:
//!                     │    finalize accumulators
//!                     ▼
//!            output one row per group
//! ```
//!
//! When all key and aggregation-input expressions are simple (variable
//! passthrough or `entity.property`), the operator uses a vectorized path
//! that extracts values in bulk via [`Runtime::materialize_node_property`]
//! instead of per-row `run_expr` evaluation.  For complex expressions or
//! `DISTINCT` aggregations it falls back to per-row evaluation.

use crate::parser::ast::{ExprIR, QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp, Column, NullBitmap},
    env::Env,
    functions::{FnType, GraphFn},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, DynNode, DynTree, NodeIdx, NodeRef};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use thin_vec::{ThinVec, thin_vec};

// ---------------------------------------------------------------------------
// GroupKey — collision-free composite grouping key
// ---------------------------------------------------------------------------

/// A composite grouping key — a vector of evaluated key values.
///
/// Uses `Value::hash` for bucket placement and `Value::eq` for collision
/// resolution, eliminating the silent-merge bug of raw `u64` hash keys.
struct GroupKey(Vec<Value>);

impl PartialEq for GroupKey {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.0 == other.0
    }
}

impl Eq for GroupKey {}

impl Hash for GroupKey {
    fn hash<H: Hasher>(
        &self,
        state: &mut H,
    ) {
        for v in &self.0 {
            v.hash(state);
        }
    }
}

// ---------------------------------------------------------------------------
// Vectorizable expression analysis
// ---------------------------------------------------------------------------

/// How a key expression can be evaluated in bulk.
enum KeyExprKind {
    /// Simple variable passthrough: `GROUP BY n`
    Variable(Variable),
    /// Property access: `GROUP BY n.age`
    Property { var: Variable, attr: Arc<String> },
}

/// How an aggregation input expression can be evaluated in bulk.
enum AggInputKind {
    /// Simple variable: `sum(x)`
    Variable(Variable),
    /// Property access: `sum(n.age)`
    Property { var: Variable, attr: Arc<String> },
}

/// A single aggregation that can be evaluated via the vectorized path.
struct VectorizableAgg {
    /// The aggregation function (e.g., sum, count, min, max, collect).
    func: Arc<GraphFn>,
    /// How the aggregation input is computed.
    input: Option<AggInputKind>,
    /// The variable used as the accumulator slot in the acc Env.
    acc_var: Variable,
}

/// Full analysis of a vectorizable aggregate operator.
struct AggAnalysis {
    key_kinds: Vec<KeyExprKind>,
    agg_kinds: Vec<VectorizableAgg>,
}

/// Cached analysis result.
/// `None` = not yet analyzed.
/// `Some(None)` = analyzed, not vectorizable (use per-row fallback).
/// `Some(Some(..))` = analyzed, vectorizable fast path.
type CachedAggAnalysis = Option<Option<AggAnalysis>>;

// ---------------------------------------------------------------------------
// AggregateOp
// ---------------------------------------------------------------------------

pub struct AggregateOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Option<Box<BatchOp<'a>>>,
    keys: &'a [(Variable, QueryExpr<Variable>)],
    agg: &'a [(Variable, QueryExpr<Variable>)],
    copy_from_parent: &'a [(Variable, Variable)],
    default_acc: Option<Env<'a>>,
    errors: std::vec::IntoIter<String>,
    groups: std::collections::hash_map::IntoIter<GroupKey, (Env<'a>, Env<'a>)>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// Lazily-initialized vectorizable analysis cache.
    vectorized: CachedAggAnalysis,
}

impl<'a> AggregateOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        keys: &'a [(Variable, QueryExpr<Variable>)],
        agg: &'a [(Variable, QueryExpr<Variable>)],
        copy_from_parent: &'a [(Variable, Variable)],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let mut default_acc = Env::new(runtime.env_pool);
        for (_var, t) in agg {
            Self::set_agg_expr_zero(&t.root(), &mut default_acc);
        }

        Self {
            runtime,
            child: Some(child),
            keys,
            agg,
            copy_from_parent,
            default_acc: Some(default_acc),
            errors: Vec::new().into_iter(),
            groups: HashMap::new().into_iter(),
            idx,
            vectorized: None,
        }
    }

    // -----------------------------------------------------------------------
    // Expression analysis
    // -----------------------------------------------------------------------

    /// Analyzes key and aggregation expression trees to determine if the
    /// vectorized fast path can be used.
    fn analyze(&self) -> Option<AggAnalysis> {
        let mut key_kinds = Vec::with_capacity(self.keys.len());
        for (_target, tree) in self.keys {
            let root = tree.root();
            match root.data() {
                ExprIR::Variable(var) => {
                    key_kinds.push(KeyExprKind::Variable(var.clone()));
                }
                ExprIR::Property(attr) => {
                    if root.num_children() != 1 {
                        return None;
                    }
                    if let ExprIR::Variable(var) = root.child(0).data() {
                        key_kinds.push(KeyExprKind::Property {
                            var: var.clone(),
                            attr: attr.clone(),
                        });
                    } else {
                        return None;
                    }
                }
                _ => return None,
            }
        }

        let mut agg_kinds = Vec::with_capacity(self.agg.len());
        for (_target, tree) in self.agg {
            {
                let agg = Self::analyze_agg_tree(tree)?;
                agg_kinds.push(agg);
            }
        }

        Some(AggAnalysis {
            key_kinds,
            agg_kinds,
        })
    }

    /// Analyzes a single aggregation expression tree.
    /// Returns `Some` if the root is a simple aggregate function call with
    /// a vectorizable input.  Falls back to per-row for complex trees
    /// (e.g., `{min: min(x), max: max(x)}`) where aggregates are nested.
    fn analyze_agg_tree(tree: &QueryExpr<Variable>) -> Option<VectorizableAgg> {
        let root = tree.root();
        let ExprIR::FuncInvocation(func) = root.data() else {
            // Root is not an aggregate function (e.g., Map, Add, etc.)
            // — fall back to per-row which can recurse to find all aggregates.
            return None;
        };
        if !func.is_aggregate() {
            return None;
        }

        let num_children = root.num_children();
        if num_children < 1 {
            return None;
        }

        // Last child is the accumulator variable.
        let ExprIR::Variable(acc_var) = root.child(num_children - 1).data() else {
            return None;
        };

        // Check for DISTINCT — fall back to per-row.
        if num_children == 2 && matches!(root.child(0).data(), ExprIR::Distinct) {
            return None;
        }

        // Analyze the input argument (child 0).
        let input = if num_children == 1 {
            // No input args — this is count(*) / count() — only the accumulator var.
            None
        } else if num_children == 2 {
            // Single argument: check if it's a simple variable or property.
            let arg = root.child(0);
            match arg.data() {
                ExprIR::Variable(var) => Some(AggInputKind::Variable(var.clone())),
                ExprIR::Property(attr) => {
                    if arg.num_children() != 1 {
                        return None;
                    }
                    if let ExprIR::Variable(var) = arg.child(0).data() {
                        Some(AggInputKind::Property {
                            var: var.clone(),
                            attr: attr.clone(),
                        })
                    } else {
                        return None;
                    }
                }
                ExprIR::Bool(true) | ExprIR::Integer(_) | ExprIR::Float(_) | ExprIR::String(_)
                    if func.name.eq_ignore_ascii_case("count") =>
                {
                    // count(*) is represented as count(1) (or other non-null literal).
                    None
                }
                _ => return None, // complex expression or non-count literal
            }
        } else {
            // Multi-argument aggregation (e.g., percentileDisc(n.age, 0.5)).
            // Fall back to per-row for these.
            return None;
        };

        Some(VectorizableAgg {
            func: func.clone(),
            input,
            acc_var: acc_var.clone(),
        })
    }

    // -----------------------------------------------------------------------
    // Vectorized consume
    // -----------------------------------------------------------------------

    /// Consumes all input using the vectorized fast path.
    fn consume_input_vectorized(
        &mut self,
        analysis: &AggAnalysis,
    ) {
        let child = self.child.take().unwrap();
        let default_acc = self.default_acc.take().unwrap();

        let mut groups: HashMap<GroupKey, (Env<'a>, Env<'a>)> = HashMap::new();
        let mut errors: Vec<String> = Vec::new();

        // Pre-insert default group for keyless aggregation.
        if self.keys.is_empty() {
            let key_env = Env::new(self.runtime.env_pool);
            groups.insert(
                GroupKey(vec![]),
                (key_env, default_acc.clone_pooled(self.runtime.env_pool)),
            );
        }

        for batch_result in child {
            let batch = match batch_result {
                Ok(b) => b,
                Err(e) => {
                    errors.push(e);
                    break;
                }
            };

            let active: Vec<usize> = batch.active_indices().collect();
            let num_active = active.len();
            if num_active == 0 {
                continue;
            }

            // --- Phase 1: Extract key columns in bulk ---
            let Ok(key_columns) =
                Self::extract_key_columns(self.runtime, &batch, &active, &analysis.key_kinds)
            else {
                // Vectorized extraction failed for this batch.
                // Fall back to per-row for this batch only.
                Self::consume_batch_per_row(
                    self.runtime,
                    self.keys,
                    self.agg,
                    self.copy_from_parent,
                    &batch,
                    &default_acc,
                    &mut groups,
                    &mut errors,
                );
                continue;
            };

            // --- Phase 2: Extract aggregation input columns in bulk ---
            let Ok(agg_input_columns) =
                Self::extract_agg_input_columns(self.runtime, &batch, &active, &analysis.agg_kinds)
            else {
                Self::consume_batch_per_row(
                    self.runtime,
                    self.keys,
                    self.agg,
                    self.copy_from_parent,
                    &batch,
                    &default_acc,
                    &mut groups,
                    &mut errors,
                );
                continue;
            };

            // --- Phase 3: Group rows and accumulate ---
            // Fast path: keyless single-aggregate where the function registered
            // a `batch_agg`. Skips per-row function calls, validation, and
            // per-row `acc.take`/`acc.insert`.
            if self.keys.is_empty()
                && self.copy_from_parent.is_empty()
                && analysis.agg_kinds.len() == 1
                && let FnType::Aggregation {
                    batch_agg: Some(batch_fn),
                    ..
                } = &analysis.agg_kinds[0].func.fn_type
            {
                let agg = &analysis.agg_kinds[0];
                let entry = groups.get_mut(&GroupKey(vec![])).unwrap();
                let acc = &mut entry.1;
                let prev = acc.take(&agg.acc_var).unwrap_or(Value::Null);
                let inputs: &[Value] = if agg.input.is_some() {
                    &agg_input_columns[0]
                } else {
                    &[]
                };
                match batch_fn(self.runtime, inputs, num_active, prev) {
                    Ok(new_val) => acc.insert(&agg.acc_var, new_val),
                    Err(e) => {
                        errors.push(e);
                        break;
                    }
                }
                continue;
            }

            for row_idx in 0..num_active {
                let key_values: Vec<Value> =
                    key_columns.iter().map(|col| col[row_idx].clone()).collect();
                let group_key = GroupKey(key_values);

                let entry = groups.entry(group_key).or_insert_with(|| {
                    let mut key_env = Env::new(self.runtime.env_pool);
                    for (ki, (name, _tree)) in self.keys.iter().enumerate() {
                        key_env.insert(name, key_columns[ki][row_idx].clone());
                    }
                    // Capture copy_from_parent values from the first row of this group.
                    for (old_var, new_var) in self.copy_from_parent {
                        let val = batch.get(active[row_idx], old_var.id);
                        key_env.insert(new_var, val.clone());
                    }
                    (key_env, default_acc.clone_pooled(self.runtime.env_pool))
                });

                let acc = &mut entry.1;
                for (agg_idx, agg) in analysis.agg_kinds.iter().enumerate() {
                    let input_val = match &agg.input {
                        Some(_) => agg_input_columns[agg_idx][row_idx].clone(),
                        None => Value::Bool(true), // count(*)
                    };

                    // Skip Null inputs for aggregation — standard Cypher behavior.
                    if matches!(input_val, Value::Null) {
                        continue;
                    }

                    let prev = acc.take(&agg.acc_var).unwrap_or(Value::Null);

                    let args = thin_vec![input_val, prev];

                    if let Err(e) = agg.func.validate_args_type(&args[..1]) {
                        // Restore the accumulator that was taken above.
                        if let Some(prev_val) = args.into_iter().nth(1) {
                            acc.insert(&agg.acc_var, prev_val);
                        }
                        errors.push(e);
                        break;
                    }

                    match (agg.func.func)(self.runtime, args) {
                        Ok(new_val) => acc.insert(&agg.acc_var, new_val),
                        Err(e) => {
                            errors.push(e);
                            break;
                        }
                    }
                }
            }
        }

        self.errors = errors.into_iter();
        self.groups = groups.into_iter();
    }

    /// Extracts key values for all active rows in a batch.
    fn extract_key_columns(
        runtime: &'a Runtime<'a>,
        batch: &Batch<'a>,
        active: &[usize],
        key_kinds: &[KeyExprKind],
    ) -> Result<Vec<Vec<Value>>, ()> {
        let mut key_columns = Vec::with_capacity(key_kinds.len());
        for kind in key_kinds {
            match kind {
                KeyExprKind::Variable(var) => {
                    let col: Vec<Value> = active
                        .iter()
                        .map(|&row| batch.get(row, var.id).clone())
                        .collect();
                    key_columns.push(col);
                }
                KeyExprKind::Property { var, attr } => {
                    let node_ids = batch.extract_node_ids(var.id).ok_or(())?;
                    let active_ids: Vec<_> = active.iter().map(|&i| node_ids[i]).collect();
                    let (col, nulls) = runtime.materialize_node_property(&active_ids, attr);
                    key_columns.push(column_to_values(&col, &nulls, active.len()));
                }
            }
        }
        Ok(key_columns)
    }

    /// Extracts aggregation input values for all active rows in a batch.
    fn extract_agg_input_columns(
        runtime: &'a Runtime<'a>,
        batch: &Batch<'a>,
        active: &[usize],
        agg_kinds: &[VectorizableAgg],
    ) -> Result<Vec<Vec<Value>>, ()> {
        let mut agg_columns = Vec::with_capacity(agg_kinds.len());
        for agg in agg_kinds {
            match &agg.input {
                None => {
                    // count(*) — no input column needed, placeholder.
                    agg_columns.push(Vec::new());
                }
                Some(AggInputKind::Variable(var)) => {
                    let col: Vec<Value> = active
                        .iter()
                        .map(|&row| batch.get(row, var.id).clone())
                        .collect();
                    agg_columns.push(col);
                }
                Some(AggInputKind::Property { var, attr }) => {
                    let node_ids = batch.extract_node_ids(var.id).ok_or(())?;
                    let active_ids: Vec<_> = active.iter().map(|&i| node_ids[i]).collect();
                    let (col, nulls) = runtime.materialize_node_property(&active_ids, attr);
                    agg_columns.push(column_to_values(&col, &nulls, active.len()));
                }
            }
        }
        Ok(agg_columns)
    }

    // -----------------------------------------------------------------------
    // Per-row fallback
    // -----------------------------------------------------------------------

    /// Consumes all input using the per-row fallback path.
    fn consume_input_per_row(&mut self) {
        let child = self.child.take().unwrap();
        let default_acc = self.default_acc.take().unwrap();

        let mut groups: HashMap<GroupKey, (Env<'a>, Env<'a>)> = HashMap::new();
        let mut errors: Vec<String> = Vec::new();

        // Pre-insert default group for keyless aggregation.
        if self.keys.is_empty() {
            let key_env = Env::new(self.runtime.env_pool);
            groups.insert(
                GroupKey(vec![]),
                (key_env, default_acc.clone_pooled(self.runtime.env_pool)),
            );
        }

        for batch_result in child {
            let batch = match batch_result {
                Ok(b) => b,
                Err(e) => {
                    errors.push(e);
                    break;
                }
            };
            Self::consume_batch_per_row(
                self.runtime,
                self.keys,
                self.agg,
                self.copy_from_parent,
                &batch,
                &default_acc,
                &mut groups,
                &mut errors,
            );
        }

        self.errors = errors.into_iter();
        self.groups = groups.into_iter();
    }

    /// Processes a single batch using per-row evaluation.
    /// Shared by both the per-row fallback path and the vectorized path
    /// (when a specific batch can't use bulk extraction).
    #[allow(clippy::too_many_arguments)]
    fn consume_batch_per_row(
        runtime: &'a Runtime<'a>,
        keys: &[(Variable, QueryExpr<Variable>)],
        agg: &[(Variable, QueryExpr<Variable>)],
        copy_from_parent: &[(Variable, Variable)],
        batch: &Batch<'a>,
        default_acc: &Env<'a>,
        groups: &mut HashMap<GroupKey, (Env<'a>, Env<'a>)>,
        errors: &mut Vec<String>,
    ) {
        for vars in batch.active_env_iter() {
            let (key_values, key_env) = match (|| {
                let mut key_values = Vec::with_capacity(keys.len());
                let mut key_env = Env::new(runtime.env_pool);
                for (name, tree) in keys {
                    let value = ExprEval::from_runtime(runtime).eval(
                        tree,
                        tree.root().idx(),
                        Some(vars),
                        None,
                    )?;
                    key_env.insert(name, value.clone());
                    key_values.push(value);
                }
                // Capture copy_from_parent values from the input row.
                for (old_var, new_var) in copy_from_parent {
                    if let Some(val) = vars.get(old_var) {
                        key_env.insert(new_var, val.clone());
                    }
                }
                Ok::<(Vec<Value>, Env<'_>), String>((key_values, key_env))
            })() {
                Ok(kv) => kv,
                Err(e) => {
                    errors.push(e);
                    continue;
                }
            };

            let group_key = GroupKey(key_values);

            let entry = groups
                .entry(group_key)
                .or_insert_with(|| (key_env, default_acc.clone_pooled(runtime.env_pool)));

            // Compute group hash for DISTINCT tracking.
            let agg_group_key = entry.0.hash_u64();

            let mut curr = vars.clone_pooled(runtime.env_pool);
            for (_, tree) in agg {
                if let Err(e) = Self::run_agg_expr(
                    runtime,
                    tree,
                    tree.root().idx(),
                    &mut curr,
                    &mut entry.1,
                    agg_group_key,
                ) {
                    errors.push(e);
                    break;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Aggregation expression evaluation (per-row)
    // -----------------------------------------------------------------------

    fn run_agg_expr(
        runtime: &Runtime,
        ir: &DynTree<ExprIR<Variable>>,
        idx: NodeIdx<Dyn<ExprIR<Variable>>>,
        curr: &mut Env<'a>,
        acc: &mut Env<'a>,
        agg_group_key: u64,
    ) -> Result<(), String> {
        match ir.node(idx).data() {
            ExprIR::FuncInvocation(func) if func.is_aggregate() => {
                let num_children = ir.node(idx).num_children();
                if num_children < 2 {
                    return Err(String::from(
                        "Aggregation function must have at least one argument",
                    ));
                }

                let ExprIR::Variable(key) = ir.node(idx).child(num_children - 1).data() else {
                    return Err(String::from(
                        "Aggregation function must end with a variable",
                    ));
                };

                let prev_value = acc.take(key).unwrap_or(Value::Null);

                let arg_results: Result<ThinVec<Value>, String> = (0..num_children - 1)
                    .map(|i| {
                        let child = ir.node(idx).child(i);

                        ExprEval::from_runtime(runtime).eval(
                            ir,
                            child.idx(),
                            Some(curr),
                            Some(agg_group_key),
                        )
                    })
                    .collect();

                let mut args = match arg_results {
                    Ok(a) => a,
                    Err(e) => {
                        acc.insert(key, prev_value);
                        return Err(e);
                    }
                };

                // Skip Null inputs for aggregation — standard Cypher behavior.
                // count(*) has no explicit args so num_children == 2 with Distinct
                // or 1 arg child; for functions with a single input, skip if Null.
                if args.len() == 1 && matches!(args[0], Value::Null) {
                    acc.insert(key, prev_value);
                    return Ok(());
                }

                if num_children == 2 && matches!(ir.node(idx).child(0).data(), ExprIR::Distinct) {
                    let arg = args.remove(0);
                    if let Value::List(values) = arg {
                        args = Arc::unwrap_or_clone(values);
                    } else {
                        acc.insert(key, prev_value);
                        return Err(String::from("DISTINCT should return a list"));
                    }
                }

                if let Err(e) = func.validate_args_type(&args) {
                    acc.insert(key, prev_value);
                    return Err(e);
                }

                if let Err(e) = func.validate_args_domain(&args) {
                    acc.insert(key, prev_value);
                    return Err(e);
                }

                args.push(prev_value);

                let new_value = (func.func)(runtime, args)?;
                acc.insert(key, new_value);
            }
            _ => {
                for child in ir.node(idx).children() {
                    Self::run_agg_expr(runtime, ir, child.idx(), curr, acc, agg_group_key)?;
                }
            }
        }
        Ok(())
    }

    fn set_agg_expr_zero(
        ir: &DynNode<ExprIR<Variable>>,
        env: &mut Env<'a>,
    ) {
        match ir.data() {
            ExprIR::FuncInvocation(func) if func.is_aggregate() => {
                if let FnType::Aggregation { initial: zero, .. } = &func.fn_type {
                    let ExprIR::Variable(key) = ir.child(ir.num_children() - 1).data() else {
                        unreachable!();
                    };
                    env.insert(key, zero.clone());
                }
            }
            _ => {
                for child in ir.children() {
                    Self::set_agg_expr_zero(&child, env);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

impl<'a> Iterator for AggregateOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Consume all input on first call.
        if self.child.is_some() {
            // Lazily analyze expressions.
            if self.vectorized.is_none() {
                self.vectorized = Some(self.analyze());
            }

            if let Some(Some(ref analysis)) = self.vectorized {
                // SAFETY: We only read from `analysis` (an immutable reference
                // into `self.vectorized`), while `consume_input_vectorized`
                // mutates other fields (`child`, `default_acc`, `groups`,
                // `errors`).  Rust's borrow checker can't see that these are
                // disjoint, so we use a pointer cast to split the borrow.
                let analysis_ptr = analysis as *const AggAnalysis;
                unsafe {
                    self.consume_input_vectorized(&*analysis_ptr);
                }
            } else {
                self.consume_input_per_row();
            }
        }

        // Drain errors first.
        if let Some(e) = self.errors.next() {
            return Some(Err(e));
        }

        // Emit finalized groups in batches.
        let mut envs = Vec::with_capacity(BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            let Some((_group_key, (key, mut acc))) = self.groups.next() else {
                break;
            };
            match (|| {
                // Build a combined env with key values at both post-projection
                // (name) and pre-projection (original_var) IDs, plus all
                // accumulator values.  Acc values take precedence on collision.
                let mut combined = key.clone_pooled(self.runtime.env_pool);
                for (name, tree) in self.keys {
                    if let ExprIR::Variable(original_var) = tree.root().data()
                        && let Some(value) = key.get(name)
                    {
                        combined.insert(original_var, value.clone());
                    }
                }
                combined.merge(&acc);
                for (name, tree) in self.agg {
                    let val = {
                        let this = &self.runtime;
                        let idx = tree.root().idx();
                        let env: &Env<'_> = &combined;
                        crate::runtime::eval::ExprEval::from_runtime(this).eval(
                            tree,
                            idx,
                            Some(env),
                            None,
                        )
                    }?;
                    acc.insert(name, val.clone());
                    combined.insert(name, val);
                }
                // Insert pre-projection key variable values into acc so
                // downstream operators can find them, but skip any slot that
                // is already occupied by an aggregation output to avoid
                // overwriting computed results (e.g., a MapProjection result).
                for (name, tree) in self.keys {
                    if let ExprIR::Variable(original_var) = tree.root().data()
                        && let Some(value) = key.get(name)
                        && !self
                            .agg
                            .iter()
                            .any(|(agg_name, _)| agg_name.id == original_var.id)
                    {
                        acc.insert(original_var, value.clone());
                    }
                }
                acc.merge(&key);
                // Unbind internal accumulator variables so they don't leak
                // to downstream operators and collide with variables in
                // subsequent scopes that reuse the same slot IDs.
                for (_, tree) in self.agg {
                    unbind_agg_accumulators(&tree.root(), &mut acc);
                }
                Ok::<Env<'_>, String>(acc)
            })() {
                Ok(env) => envs.push(env),
                Err(e) => return Some(Err(e)),
            }
        }

        if envs.is_empty() {
            None
        } else {
            Some(Ok(Batch::from_envs(envs)))
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Converts a `Column` + `NullBitmap` from `materialize_node_property` into
/// a `Vec<Value>` for use in grouping and accumulation.
fn column_to_values(
    col: &Column,
    nulls: &NullBitmap,
    len: usize,
) -> Vec<Value> {
    match col {
        Column::Ints(data) => (0..len)
            .map(|i| {
                if nulls.is_null(i) {
                    Value::Null
                } else {
                    Value::Int(data[i])
                }
            })
            .collect(),
        Column::Floats(data) => (0..len)
            .map(|i| {
                if nulls.is_null(i) {
                    Value::Null
                } else {
                    Value::Float(data[i])
                }
            })
            .collect(),
        Column::Values(data) => data.clone(),
        _ => vec![Value::Null; len],
    }
}

/// Recursively walks an expression tree and unbinds each aggregate function's
/// accumulator variable from the environment. This mirrors the recursion in
/// `set_agg_expr_zero` / `run_agg_expr`: when an aggregate `FuncInvocation` is
/// found, its last child (the accumulator `Variable`) is unbound; otherwise we
/// recurse into children.
fn unbind_agg_accumulators(
    ir: &DynNode<ExprIR<Variable>>,
    acc: &mut Env,
) {
    match ir.data() {
        ExprIR::FuncInvocation(func) if func.is_aggregate() => {
            let num_children = ir.num_children();
            if num_children >= 2
                && let ExprIR::Variable(acc_var) = ir.child(num_children - 1).data()
            {
                acc.unbind(acc_var);
            }
        }
        _ => {
            for child in ir.children() {
                unbind_agg_accumulators(&child, acc);
            }
        }
    }
}

/// Computes a u64 hash of an `Env`, used as an opaque group identifier
/// for DISTINCT tracking in the per-row fallback path.
trait HashU64 {
    fn hash_u64(&self) -> u64;
}

impl HashU64 for Env<'_> {
    fn hash_u64(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}
