//! Batch-mode unit-subquery operator — runs a `CALL {}` body that returns
//! nothing.
//!
//! A subquery with no `RETURN` is a *unit subquery*: it runs for every input
//! row and contributes no rows and no variables of its own, so the input row
//! reaches the next clause exactly once however many rows the body produced
//! internally.
//!
//! ```text
//!  Input batch [row0, row1, row2]
//!       │
//!  stamp origin_row
//!       │
//!  ┌────▼─────────────┐
//!  │ Body sub-plan     │  single instance, all rows at once
//!  └────┬─────────────┘
//!       │
//!  drained to exhaustion, results discarded
//!       │
//!  ┌────▼──────────────────────────┐
//!  │ emit [row0, row1, row2] as-is │
//!  └───────────────────────────────┘
//! ```
//!
//! Draining matters: the body's writes are its whole point, and a body that
//! fans out internally performs them once per row it produced. Only the row
//! count is capped, which is why this cannot be expressed as a limit.
//!
//! The C engine calls its counterpart `SubqueryForeach`.

use crate::planner::{IR, subtree_contains};
use crate::runtime::{
    batch::{Batch, BatchBuilder, BatchOp, BatchRow},
    row::RowView,
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct UnitSubqueryOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    body_idx: NodeIdx<Dyn<IR>>,
    /// False when the body collapses rows, so it must see one row at a time.
    can_batch: bool,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> UnitSubqueryOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let body_idx = runtime.plan.node(idx).child(1).idx();

        // A body that collapses rows, such as `WITH ... LIMIT 1` or an
        // aggregation, would apply once across the whole batch rather than once
        // per input row, and its writes would happen the wrong number of times.
        // Apply draws the same line for the same reason.
        let can_batch = !subtree_contains(&runtime.plan, body_idx, |ir| {
            matches!(
                ir,
                IR::Aggregate { .. }
                    | IR::CartesianProduct
                    | IR::Optional(_)
                    | IR::Apply
                    | IR::UnitSubquery
                    | IR::Merge { .. }
                    | IR::Union
                    | IR::Sort(_)
                    | IR::Limit(_)
                    | IR::Skip(_)
                    | IR::Distinct
            )
        });

        Self {
            runtime,
            child,
            body_idx,
            can_batch,
            idx,
        }
    }

    /// Runs the body once over `arg`, discarding every row it produces.
    fn drain_body(
        &self,
        arg: Batch<'a>,
    ) -> Result<(), String> {
        let mut body = self.runtime.run_batch(self.body_idx)?;
        body.set_argument_batch(arg);
        for result in body.by_ref() {
            result?;
        }
        Ok(())
    }
}

impl<'a> Iterator for UnitSubqueryOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.child.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        if self.can_batch {
            if let Err(e) = self.drain_body(batch.clone_active_rows_seq_origin()) {
                return Some(Err(e));
            }
            return Some(Ok(batch));
        }

        for row in batch.active_indices() {
            let mut arg = BatchBuilder::new();
            arg.push_row(&BatchRow::new(&batch, row).to_owned_row());
            if let Err(e) = self.drain_body(arg.finish()) {
                return Some(Err(e));
            }
            // Each invocation is a separate CALL {}, so DISTINCT inside the
            // body must not remember rows from the previous one.
            self.runtime.value_dedupers.borrow_mut().clear();
        }

        Some(Ok(batch))
    }
}
