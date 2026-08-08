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

use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct UnitSubqueryOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    body_idx: NodeIdx<Dyn<IR>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> UnitSubqueryOp<'a> {
    /// Builds the operator over `child`, the input rows, resolving the body
    /// from child(1) of the plan node.
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let body_idx = runtime.plan.node(idx).child(1).idx();

        Self {
            runtime,
            child,
            body_idx,
            idx,
        }
    }
}

impl<'a> Iterator for UnitSubqueryOp<'a> {
    type Item = Result<Batch<'a>, String>;

    /// Runs the body over the whole input batch, discards whatever it
    /// produced, and yields the batch unchanged.
    ///
    /// The body is drained rather than short-circuited: its writes are the
    /// reason it exists, and one that fans out internally performs them once
    /// per row it produced.
    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.child.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        let mut body = match self.runtime.run_batch(self.body_idx) {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };
        body.set_argument_batch(batch.clone_active_rows_seq_origin());

        for result in body.by_ref() {
            if let Err(e) = result {
                return Some(Err(e));
            }
        }
        drop(body);

        Some(Ok(batch))
    }
}
