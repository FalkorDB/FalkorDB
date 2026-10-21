//! Batch-mode skip operator — discards the first N rows across batches.
//!
//! Pulls batches from the child operator, skipping the first `remaining_skip`
//! active rows. Entire batches are dropped when they fall within the skip
//! window. A partial batch is trimmed by rebuilding the selection vector to
//! exclude the leading entries. Once the skip count is exhausted, subsequent
//! batches pass through unchanged.
//!
//! ```text
//!  SKIP 5:
//!
//!  batch 1 (3 rows) ──► drop entirely (remaining: 5 -> 2)
//!  batch 2 (4 rows) ──► skip first 2, pass last 2 (remaining: 2 -> 0)
//!  batch 3           ──► pass through unchanged
//! ```

use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx};

pub struct SkipOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    remaining_skip: usize,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> SkipOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        skip: usize,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            remaining_skip: skip,
            idx,
        }
    }
}

impl<'a> Iterator for SkipOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut batch = match self.child.next()? {
                Ok(batch) => batch,
                Err(e) => return Some(Err(e)),
            };

            if self.remaining_skip == 0 {
                // Skip window exhausted — pass through unchanged.
                return Some(Ok(batch));
            }

            let active = batch.active_len();

            if self.remaining_skip >= active {
                // Entire batch falls within the skip window — drop it.
                self.remaining_skip -= active;
                continue;
            }

            // Partial skip: collect active indices, drop the first
            // `remaining_skip` entries, and build a new selection vector.
            let new_sel: Vec<u16> = batch
                .active_indices()
                .skip(self.remaining_skip)
                .map(|i| i as u16)
                .collect();
            self.remaining_skip = 0;
            batch.set_selection(new_sel);
            return Some(Ok(batch));
        }
    }
}
