//! Union operator — concatenates results from multiple sub-plans.
//!
//! Implements Cypher `UNION` / `UNION ALL`. Iterates through each child
//! sub-plan in order and yields all batches from the first child before
//! moving to the second, and so on.
//!
//! ```text
//!  ┌───────────┐  ┌───────────┐  ┌───────────┐
//!  │ Branch 0  │  │ Branch 1  │  │ Branch 2  │
//!  │ (created  │  │ (created  │  │ (created  │
//!  │  lazily)  │  │  lazily)  │  │  lazily)  │
//!  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
//!        │              │              │
//!   drain all      drain all      drain all
//!   batches        batches        batches
//!        │              │              │
//!        └──── sequential order ───────┘
//!                       │
//!                  output stream
//! ```
//!
//! Each branch is lazily constructed via `run_batch` when the previous branch
//! is exhausted. An optional stored argument batch is propagated to each
//! branch for correlated sub-queries.

use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct UnionOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) current: Option<Box<BatchOp<'a>>>,
    current_child: usize,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// Stored argument batch to propagate to each branch when created.
    pub(crate) argument_batch: Option<Batch<'a>>,
}

impl<'a> UnionOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            current: None,
            current_child: 0,
            idx,
            argument_batch: None,
        }
    }

    /// Store an argument batch to be passed to each Union branch.
    pub fn store_argument_batch(
        &mut self,
        batch: Batch<'a>,
    ) {
        self.argument_batch = Some(batch.into_compacted());
    }
}

impl<'a> Iterator for UnionOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut current) = self.current {
                if let Some(result) = current.next() {
                    return Some(result);
                }
                self.current = None;
                self.current_child += 1;
            }
            let current_node = self.runtime.plan.node(self.idx);
            if self.current_child >= current_node.num_children() {
                return None;
            }
            let child_idx = current_node.child(self.current_child).idx();
            match self.runtime.run_batch(child_idx) {
                Ok(mut child_op) => {
                    // Propagate stored argument batch to each new branch
                    if let Some(ref batch) = self.argument_batch {
                        child_op.set_argument_batch(batch.clone());
                    }
                    self.current = Some(Box::new(child_op));
                }
                Err(e) => {
                    self.current_child = current_node.num_children();
                    return Some(Err(e));
                }
            }
        }
    }
}
