//! Batch-mode sort operator — orders result rows by one or more expressions.
//!
//! This is a *blocking* operator: it consumes all batches from the child on
//! the first `next()` call, evaluates sort-key expressions for each
//! row, sorts in-memory using a stable sort with per-key ascending/descending
//! control, and then yields rows in sorted batches.
//!
//! ```text
//!  Child batches (all consumed on first call)
//!       │
//!       ▼
//!  ┌──────────────────────────────────┐
//!  │ Evaluate sort keys per row       │
//!  │ [(env, [(val, desc), ...])]      │
//!  └──────────────┬───────────────────┘
//!                 │
//!       stable sort (multi-key)
//!                 │
//!       ┌────────▼────────┐
//!       │ reversed Vec    │  stored reversed so pop() is O(1)
//!       │ yield BATCH_SIZE│
//!       │ at a time       │
//!       └─────────────────┘
//! ```
//!
//! When primary sort keys are equal, a deterministic tiebreaker compares
//! env values slot-by-slot.

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    runtime::Runtime,
    value::{CompareValue, Value},
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use std::cmp::Ordering;

pub struct SortOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Option<Box<BatchOp<'a>>>,
    trees: &'a [(QueryExpr<Variable>, bool)],
    /// The concatenated input buffer (fully materialised on the first `next`),
    /// the row order to emit, and a cursor into that order.
    sorted: Option<Batch<'a>>,
    order: Vec<usize>,
    pos: usize,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// When set, only the top `limit` rows (after skip) are needed.
    /// Allows truncation after sorting to avoid excess work.
    limit: Option<usize>,
    /// Number of rows to skip before the limit applies.
    skip: usize,
}

impl<'a> SortOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        trees: &'a [(QueryExpr<Variable>, bool)],
        idx: NodeIdx<Dyn<IR>>,
        limit: Option<usize>,
        skip: usize,
    ) -> Self {
        Self {
            runtime,
            child: Some(child),
            trees,
            sorted: None,
            order: Vec::new(),
            pos: 0,
            idx,
            limit,
            skip,
        }
    }
}

impl<'a> Iterator for SortOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Consume and sort all input on the first call.
        if let Some(child) = self.child.take() {
            // Concatenate every child batch into one columnar buffer with no
            // owned `Row` materialised per input row.
            let mut combined_builder = BatchBuilder::new();
            for batch_result in child {
                match batch_result {
                    Ok(batch) => combined_builder.push_batch_active(&batch),
                    Err(e) => return Some(Err(e)),
                }
            }
            let combined = combined_builder.finish();
            let total = combined.len();
            if total == 0 {
                return None;
            }

            // Evaluate the sort keys once per row through a borrowed columnar
            // view (so `rand()` and other expression keys still work) with no
            // `Row` allocation.
            let mut keys: Vec<Vec<(Value, bool)>> = Vec::with_capacity(total);
            for row in 0..total {
                let view = BatchRow::new(&combined, row);
                let mut row_keys = Vec::with_capacity(self.trees.len());
                for (tree, desc) in self.trees {
                    match ExprEval::from_runtime(self.runtime).eval(
                        tree,
                        tree.root().idx(),
                        Some(&view),
                        None,
                    ) {
                        Ok(value) => row_keys.push((value, *desc)),
                        Err(e) => return Some(Err(e)),
                    }
                }
                keys.push(row_keys);
            }

            let num_columns = combined.num_columns();
            let mut order: Vec<usize> = (0..total).collect();
            order.sort_by(|&a, &b| {
                let primary = keys[a].iter().zip(&keys[b]).fold(
                    Ordering::Equal,
                    |acc, ((va, desc_a), (vb, _))| {
                        if acc != Ordering::Equal {
                            return acc;
                        }
                        let (ordering, _) = va.compare_value(vb);
                        if *desc_a {
                            ordering.reverse()
                        } else {
                            ordering
                        }
                    },
                );
                if primary != Ordering::Equal {
                    return primary;
                }
                // Deterministic tiebreaker: compare bound slots position-by-position.
                // Columns unbound in every row read back as `Null` on both sides
                // and so never change the ordering.
                for id in 0..num_columns {
                    let va = combined.value_at(id as u32, a).unwrap_or(Value::Null);
                    let vb = combined.value_at(id as u32, b).unwrap_or(Value::Null);
                    let (ordering, _) = va.compare_value(&vb);
                    if ordering != Ordering::Equal {
                        return ordering;
                    }
                }
                // Final total-order fallback: rows still equal here are identical
                // in every compared column (or differ only in columns that read
                // back as `Null` on both sides). `sort_by` is unstable, so break
                // the remaining ties by original row index to keep output order
                // deterministic.
                a.cmp(&b)
            });

            // When a limit is known, drop the rows the Skip/Limit operators above
            // will never consume.
            if let Some(limit) = self.limit {
                order.truncate(limit + self.skip);
            }

            self.sorted = Some(combined);
            self.order = order;
            self.pos = 0;
        }

        // Emit the sorted rows BATCH_SIZE at a time by gathering from the
        // combined buffer in sorted order.
        if self.pos >= self.order.len() {
            return None;
        }
        let end = (self.pos + BATCH_SIZE).min(self.order.len());
        let out = {
            let combined = self.sorted.as_ref().unwrap();
            combined.gather(&self.order[self.pos..end])
        };
        self.pos = end;
        Some(Ok(out))
    }
}
