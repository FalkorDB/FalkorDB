//! Batch-mode filter operator — evaluates a boolean predicate on each row.
//!
//! Pulls a batch from the child operator, evaluates the predicate columnarly,
//! and returns only the passing rows via a selection vector (zero-copy
//! filtering).
//!
//! ```text
//!  active rows ──► VectorEval(predicate) ──► mask ──► selection vector
//! ```
//!
//! The predicate is evaluated by [`VectorEval`], which walks the expression
//! tree column-at-a-time: one bulk property fetch per property reference and
//! one pass per operator, instead of one tree walk per row. Operators it has no
//! columnar arm for degrade to per-row evaluation of that subtree alone, so
//! there is no whole-predicate fallback to choose between — every filter takes
//! the same path.
//!
//! A row survives only when the predicate is `true`; `false` and `null` both
//! drop it, and any non-boolean result is a type error.

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
    value::Value,
    vector_expr::{ExprColumn, VectorEval},
};
use orx_tree::{Dyn, NodeIdx};

pub struct FilterOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    tree: &'a QueryExpr<Variable>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> FilterOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        tree: &'a QueryExpr<Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            tree,
            idx,
        }
    }

    /// Evaluates the filter for one batch, returning the surviving rows.
    ///
    /// Returns `Ok(Some(batch))` with passing rows, `Ok(None)` when every row
    /// was filtered out, or `Err` on a predicate evaluation error.
    fn eval(
        &mut self,
        mut batch: Batch<'a>,
    ) -> Result<Option<Batch<'a>>, String> {
        let rows: Vec<usize> = batch.active_indices().collect();
        if rows.is_empty() {
            return Ok(None);
        }

        let root = self.tree.root();
        let verdict = VectorEval::new(self.runtime).eval(&root, &batch, &rows)?;

        let passing = match verdict {
            // The common shape: a comparison or boolean tree is already a mask.
            // `null` rows are marked in the bitmap and drop with the `false`s —
            // but a predicate over non-null data marks none, so the bitmap is
            // consulted once for the whole batch rather than once per row.
            ExprColumn::Bools(mask, nulls) if !nulls.any_null() => rows
                .iter()
                .enumerate()
                .filter(|&(i, _)| mask[i])
                .map(|(_, &row)| row as u16)
                .collect(),
            ExprColumn::Bools(mask, nulls) => rows
                .iter()
                .enumerate()
                .filter(|&(i, _)| mask[i] && !nulls.is_null(i))
                .map(|(_, &row)| row as u16)
                .collect(),
            // Any other shape — a bare boolean property, a function call, a
            // subtree that fell back per row — is read one row at a time. A
            // `null` drops the row (a missing property is not an error); any
            // other non-boolean is a type error, reported for the first row
            // that carries one, as the per-row path does.
            other => {
                let mut passing = Vec::new();
                for (i, &row) in rows.iter().enumerate() {
                    match other.get(i) {
                        Value::Bool(true) => passing.push(row as u16),
                        Value::Bool(false) | Value::Null => {}
                        value => {
                            return Err(format!(
                                "Type mismatch: expected Boolean but was {}",
                                value.name()
                            ));
                        }
                    }
                }
                passing
            }
        };

        if passing.is_empty() {
            Ok(None)
        } else {
            batch.set_selection(passing);
            Ok(Some(batch))
        }
    }
}

impl<'a> Iterator for FilterOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let batch = match self.child.next()? {
                Ok(batch) => batch,
                Err(e) => return Some(Err(e)),
            };

            match self.eval(batch) {
                Ok(Some(result)) => return Some(Ok(result)),
                Ok(None) => {} // all rows filtered out — pull next batch
                Err(e) => return Some(Err(e)),
            }
        }
    }
}
