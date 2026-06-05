//! Batch-mode project operator — evaluates return expressions and reshapes rows.
//!
//! For each active row in the input batch, evaluates projection expressions
//! and carry-forward variables to produce a new batch with only the projected
//! columns.
//!
//! Projections are always assembled columnarly through a [`BatchBuilder`],
//! never an intermediate `Env`. Simple property accesses (e.g. `n.age`) and
//! variable passthroughs (e.g. `RETURN n`) are gathered as bulk columns;
//! expressions containing function calls, arithmetic, etc. are evaluated per
//! row via `BatchRow`.

use std::sync::Arc;

use crate::parser::ast::{ExprIR, QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{Batch, BatchBuilder, BatchOp, BatchRow, Column},
    row::Row,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

/// How a single projection expression is evaluated when building the output batch.
enum ProjectionKind {
    /// Property access: `var.attr` — materializable as a bulk column read.
    Property { var: Variable, attr: Arc<String> },
    /// Simple variable passthrough (e.g., `RETURN n`).
    Variable(Variable),
    /// Arbitrary expression (function call, arithmetic, etc.) evaluated per row.
    Expr,
}

pub struct ProjectOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    trees: &'a [(Variable, QueryExpr<Variable>)],
    copy_from_parent: &'a [(Variable, Variable)],
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// Lazily-initialized per-projection evaluation plan.
    plan: Option<Vec<ProjectionKind>>,
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
            plan: None,
        }
    }

    /// Classifies every projection tree into an evaluation strategy. Property
    /// accesses and variable passthroughs use the bulk columnar path; anything
    /// else falls back to per-row expression evaluation.
    fn classify_projections(&self) -> Vec<ProjectionKind> {
        self.trees
            .iter()
            .map(|(_target, tree)| {
                let root = tree.root();
                match root.data() {
                    ExprIR::Property(attr) if root.num_children() == 1 => {
                        if let ExprIR::Variable(var) = root.child(0).data() {
                            ProjectionKind::Property {
                                var: var.clone(),
                                attr: attr.clone(),
                            }
                        } else {
                            ProjectionKind::Expr
                        }
                    }
                    ExprIR::Variable(var) => ProjectionKind::Variable(var.clone()),
                    _ => ProjectionKind::Expr,
                }
            })
            .collect()
    }

    /// Evaluates all projections columnarly and produces the output batch.
    ///
    /// Property and variable projections are gathered as bulk columns; arbitrary
    /// expressions are evaluated per row. The output batch is assembled directly
    /// through [`BatchBuilder`], never materializing an intermediate `Env`.
    fn eval(
        &self,
        batch: &Batch<'a>,
        plan: &[ProjectionKind],
    ) -> Result<Batch<'a>, String> {
        let active: Vec<usize> = batch.active_indices().collect();
        let cap = self.trees.len() + self.copy_from_parent.len();

        // Bulk-materialize each projection column where possible. `None` marks a
        // projection that must be evaluated per row.
        let mut columns: Vec<Option<Vec<Value>>> = Vec::with_capacity(self.trees.len());
        for kind in plan {
            let col = match kind {
                ProjectionKind::Property { var, attr } => {
                    batch.extract_node_ids(var.id).and_then(|node_ids| {
                        let active_ids: Vec<_> = active.iter().map(|&i| node_ids[i]).collect();
                        let (col, nulls) =
                            self.runtime.materialize_node_property(&active_ids, attr);
                        match col {
                            Column::Ints(data) => Some(
                                data.iter()
                                    .enumerate()
                                    .map(|(i, &v)| {
                                        if nulls.is_null(i) {
                                            Value::Null
                                        } else {
                                            Value::Int(v)
                                        }
                                    })
                                    .collect(),
                            ),
                            Column::Floats(data) => Some(
                                data.iter()
                                    .enumerate()
                                    .map(|(i, &v)| {
                                        if nulls.is_null(i) {
                                            Value::Null
                                        } else {
                                            Value::Float(v)
                                        }
                                    })
                                    .collect(),
                            ),
                            Column::Values(data) => Some(data),
                            _ => None,
                        }
                    })
                }
                ProjectionKind::Variable(var) => Some(
                    active
                        .iter()
                        .map(|&row| batch.value_at(var.id, row).unwrap_or(Value::Null))
                        .collect(),
                ),
                ProjectionKind::Expr => None,
            };
            columns.push(col);
        }

        // Transpose the projected columns into the output batch row by row.
        let mut builder = BatchBuilder::new();
        for (out_idx, &row) in active.iter().enumerate() {
            let view = BatchRow::new(batch, row);
            let mut result = Row::with_capacity(cap);
            result.origin_row = batch.origin_row(row);
            for (proj_idx, (target, tree)) in self.trees.iter().enumerate() {
                let val = match &columns[proj_idx] {
                    Some(col) => col[out_idx].clone(),
                    None => ExprEval::from_runtime(self.runtime).eval(
                        tree,
                        tree.root().idx(),
                        Some(&view),
                        None,
                    )?,
                };
                result.insert(target, val);
            }
            for (old_var, new_var) in self.copy_from_parent {
                match batch.value_at(old_var.id, row) {
                    Some(value) if !matches!(value, Value::Null) => {
                        result.insert(new_var, value);
                    }
                    _ if batch.is_bound_at(old_var.id, row) => {
                        result.insert(new_var, Value::Null);
                    }
                    _ => {}
                }
            }
            builder.push_row(&result);
        }
        Ok(builder.finish())
    }
}

impl<'a> Iterator for ProjectOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Lazily classify projections on the first call.
        if self.plan.is_none() {
            self.plan = Some(self.classify_projections());
        }

        let batch = match self.child.next()? {
            Ok(batch) => batch,
            Err(e) => return Some(Err(e)),
        };

        let plan = self.plan.as_ref().expect("plan initialized above");
        Some(self.eval(&batch, plan))
    }
}
