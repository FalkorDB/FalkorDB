//! Batch-mode project operator — evaluates return expressions and reshapes rows.
//!
//! For each active row in the input batch, evaluates projection expressions
//! and carry-forward variables to produce a new batch with only the projected
//! columns.
//!
//! Projections are always assembled columnarly through a [`BatchBuilder`],
//! never an intermediate `Env`. Every projection — a property read, a variable
//! passthrough or a computed expression — is evaluated by [`VectorEval`] as one
//! column, so there is no per-projection strategy to classify and no shape that
//! drops the whole operator to row-at-a-time evaluation.

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchBuilder, BatchOp, Column},
    row::Row,
    runtime::Runtime,
    value::Value,
    vector_expr::VectorEval,
};
use orx_tree::{Dyn, NodeIdx};

pub struct ProjectOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    trees: &'a [(Variable, QueryExpr<Variable>)],
    copy_from_parent: &'a [(Variable, Variable)],
    pub(crate) idx: NodeIdx<Dyn<IR>>,
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
        }
    }

    /// Evaluates all projections columnarly and produces the output batch.
    ///
    /// Property and variable projections are gathered as bulk columns; arbitrary
    /// expressions are evaluated per row. The output batch is assembled directly
    /// through [`BatchBuilder`], never materializing an intermediate `Env`.
    fn eval(
        &self,
        batch: &Batch<'a>,
    ) -> Result<Batch<'a>, String> {
        let active: Vec<usize> = batch.active_indices().collect();
        let cap = self.trees.len() + self.copy_from_parent.len();

        // One column per projection, each evaluated over every active row at
        // once. `VectorEval` handles the shapes this operator used to classify
        // by hand — `n.age` is still a single bulk attribute fetch — and any
        // node it has no columnar arm for degrades to per-row evaluation of
        // that subtree alone.
        let eval = VectorEval::new(self.runtime);
        let mut columns: Vec<Vec<Value>> = Vec::with_capacity(self.trees.len());
        for (_target, tree) in self.trees {
            columns.push(eval.eval_values(&tree.root(), batch, &active)?);
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
        if !active.is_empty() && !self.trees.is_empty() && self.copy_from_parent.is_empty() {
            let mut out = Batch::new(0);
            for ((target, _tree), col) in self.trees.iter().zip(columns) {
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
            let mut result = Row::with_capacity(cap);
            result.origin_row = batch.origin_row(row);
            for (proj_idx, (target, _tree)) in self.trees.iter().enumerate() {
                result.insert(target, columns[proj_idx][out_idx].clone());
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
        let batch = match self.child.next()? {
            Ok(batch) => batch,
            Err(e) => return Some(Err(e)),
        };

        Some(self.eval(&batch))
    }
}
