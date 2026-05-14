//! Batch-mode project operator — evaluates return expressions and reshapes rows.
//!
//! For each active row in the input batch, evaluates projection expressions
//! and carry-forward variables to produce a new batch with only the projected
//! columns.
//!
//! When all projection expressions are simple property accesses
//! (e.g., `n.age`, `n.name`), the operator uses a fast columnar path:
//! extract node IDs → materialize property columns in bulk.
//! For expressions containing function calls, arithmetic, etc., it falls
//! back to per-row `run_expr` evaluation.

use std::sync::Arc;

use crate::parser::ast::{ExprIR, QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{Batch, BatchOp, Column},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

/// A projection expression that can be evaluated in batch via property materialization.
enum ProjectionKind {
    /// Property access: `var.attr` — materializable as a typed column.
    Property { var: Variable, attr: Arc<String> },
    /// Simple variable passthrough (e.g., `RETURN n`).
    Variable(Variable),
}

/// Cached analysis of whether all projections are batchable.
enum ProjectionCache {
    /// Not yet analyzed (lazy initialization pending).
    NotAnalyzed,
    /// Analyzed: not all projections are batchable — use per-row evaluation.
    NotBatchable,
    /// Analyzed: all projections are batchable — use vectorized path.
    Batchable(Vec<ProjectionKind>),
}

pub struct ProjectOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    trees: &'a [(Variable, QueryExpr<Variable>)],
    copy_from_parent: &'a [(Variable, Variable)],
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// Lazily-initialized projection analysis cache.
    vectorized: ProjectionCache,
}

impl<'a> ProjectOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        trees: &'a [(Variable, QueryExpr<Variable>)],
        copy_from_parent: &'a [(Variable, Variable)],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            trees,
            copy_from_parent,
            idx,
            vectorized: ProjectionCache::NotAnalyzed,
        }
    }

    /// Analyzes all projection trees. Returns `Some(kinds)` if all are simple
    /// property accesses or variable passthroughs, `None` otherwise.
    fn analyze_projections(&self) -> Option<Vec<ProjectionKind>> {
        let mut kinds = Vec::with_capacity(self.trees.len());
        for (_target, tree) in self.trees {
            let root = tree.root();
            match root.data() {
                ExprIR::Property(attr) => {
                    if root.num_children() != 1 {
                        return None;
                    }
                    if let ExprIR::Variable(var) = root.child(0).data() {
                        kinds.push(ProjectionKind::Property {
                            var: var.clone(),
                            attr: attr.clone(),
                        });
                    } else {
                        return None;
                    }
                }
                ExprIR::Variable(var) => {
                    kinds.push(ProjectionKind::Variable(var.clone()));
                }
                _ => return None,
            }
        }
        Some(kinds)
    }

    /// Evaluates all projections using batch property materialization.
    fn eval_vectorized(
        &self,
        batch: Batch<'a>,
        kinds: &[ProjectionKind],
    ) -> Result<Batch<'a>, Batch<'a>> {
        let cap = self.trees.len() + self.copy_from_parent.len();
        let active_count = batch.active_len();
        let mut result_envs = Vec::with_capacity(active_count);

        // Process each projection.
        let active: Vec<usize> = batch.active_indices().collect();

        // Pre-initialize result envs.
        for &row_idx in &active {
            let mut env = Env::with_capacity(cap, self.runtime.env_pool);
            env.origin_row = batch.env_ref(row_idx).origin_row;
            result_envs.push(env);
        }

        for (proj_idx, kind) in kinds.iter().enumerate() {
            let target = &self.trees[proj_idx].0;
            match kind {
                ProjectionKind::Property { var, attr } => {
                    let Some(node_ids) = batch.extract_node_ids(var.id) else {
                        return Err(batch);
                    };
                    // Gather only active node IDs.
                    let active_ids: Vec<_> = active.iter().map(|&i| node_ids[i]).collect();
                    let (col, nulls) = self.runtime.materialize_node_property(&active_ids, attr);
                    // Distribute values to result envs.
                    match &col {
                        Column::Ints(data) => {
                            for (env_idx, &val) in data.iter().enumerate() {
                                let v = if nulls.is_null(env_idx) {
                                    Value::Null
                                } else {
                                    Value::Int(val)
                                };
                                result_envs[env_idx].insert(target, v);
                            }
                        }
                        Column::Floats(data) => {
                            for (env_idx, &val) in data.iter().enumerate() {
                                let v = if nulls.is_null(env_idx) {
                                    Value::Null
                                } else {
                                    Value::Float(val)
                                };
                                result_envs[env_idx].insert(target, v);
                            }
                        }
                        Column::Values(data) => {
                            for (env_idx, val) in data.iter().enumerate() {
                                result_envs[env_idx].insert(target, val.clone());
                            }
                        }
                        _ => return Err(batch),
                    }
                }
                ProjectionKind::Variable(var) => {
                    for (env_idx, &row_idx) in active.iter().enumerate() {
                        let val = batch.get(row_idx, var.id);
                        result_envs[env_idx].insert(target, val.clone());
                    }
                }
            }
        }

        // Handle copy_from_parent.
        for (old_var, new_var) in self.copy_from_parent {
            for (env_idx, &row_idx) in active.iter().enumerate() {
                let val = batch.get(row_idx, old_var.id);
                result_envs[env_idx].insert(new_var, val.clone());
            }
        }

        Ok(Batch::from_envs(result_envs))
    }

    /// Per-row fallback: evaluates each projection expression row-by-row.
    fn eval_per_row(
        &self,
        batch: &Batch<'a>,
    ) -> Result<Batch<'a>, String> {
        let cap = self.trees.len() + self.copy_from_parent.len();
        let mut result_envs = Vec::with_capacity(batch.active_len());

        for env in batch.active_env_iter() {
            let mut return_vars = Env::with_capacity(cap, self.runtime.env_pool);
            return_vars.origin_row = env.origin_row;
            for (name, tree) in self.trees {
                let res = ExprEval::from_runtime(self.runtime).eval(
                    tree,
                    tree.root().idx(),
                    Some(env),
                    None,
                );
                return_vars.insert(name, res?);
            }
            let mut vars = env.clone_pooled(self.runtime.env_pool);
            for (old_var, new_var) in self.copy_from_parent {
                match vars.take(old_var) {
                    Some(value) => return_vars.insert(new_var, value),
                    None if vars.is_bound(old_var) => {
                        return_vars.insert(new_var, Value::Null);
                    }
                    None => {}
                }
            }
            result_envs.push(return_vars);
        }

        Ok(Batch::from_envs(result_envs))
    }
}

impl<'a> Iterator for ProjectOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Lazily analyze projections on the first call.
        if matches!(self.vectorized, ProjectionCache::NotAnalyzed) {
            self.vectorized = self
                .analyze_projections()
                .map_or(ProjectionCache::NotBatchable, ProjectionCache::Batchable);
        }

        let batch = match self.child.next()? {
            Ok(batch) => batch,
            Err(e) => return Some(Err(e)),
        };

        // Try vectorized path if all projections are simple property accesses.
        if let ProjectionCache::Batchable(kinds) = &self.vectorized {
            match self.eval_vectorized(batch, kinds) {
                Ok(result) => return Some(Ok(result)),
                Err(batch) => {
                    // Vectorized path failed (e.g. variable not a node).
                    // Disable for future batches and fall through to per-row
                    // on the same batch.
                    self.vectorized = ProjectionCache::NotBatchable;
                    return Some(self.eval_per_row(&batch));
                }
            }
        }

        // Per-row fallback path.
        Some(self.eval_per_row(&batch))
    }
}
