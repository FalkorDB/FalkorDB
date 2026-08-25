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
                    // The stored values are what the projection emits, so take
                    // them unclassified: classifying and rebuilding `Value`s
                    // costs two extra passes, and its float lane would promote
                    // a mixed int/float column, printing a stored
                    // 9007199254740993 as 9.00719925474099e+15.
                    batch.extract_node_ids(var.id).map(|node_ids| {
                        let active_ids: Vec<_> = active.iter().map(|&i| node_ids[i]).collect();
                        self.runtime
                            .materialize_node_property_values(&active_ids, attr)
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

        // Fast path: every projection bulk-materialized into a column and there
        // are no carry-forward variables, so move the columns straight into the
        // output batch. This skips the per-row `Row` allocation and the
        // column -> Row -> column double-clone the general path below incurs (the
        // dominant `ProjectOp` cost on row-heavy projections such as `RETURN n`
        // or `RETURN n.id`). Parity-exact: `set_column` applies the same
        // `classify_stored_column` that `BatchBuilder::finish` would over the
        // same materialized values and binds the slot identically, and origins
        // are carried over per active row exactly as the row path does.
        if !active.is_empty()
            && !self.trees.is_empty()
            && self.copy_from_parent.is_empty()
            && columns.iter().all(Option::is_some)
        {
            let mut out = Batch::new(0);
            for (proj_idx, (target, _tree)) in self.trees.iter().enumerate() {
                let col = columns[proj_idx]
                    .take()
                    .expect("all projection columns materialized on the fast path");
                out.set_column(target.id, Column::Values(col));
            }
            let origins: Vec<u32> = active.iter().map(|&row| batch.origin_row(row)).collect();
            if origins.iter().any(|&o| o != 0) {
                out.set_origin_rows(origins);
            }
            return Ok(out);
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
