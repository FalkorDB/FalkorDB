//! Batch-mode limit operator — caps the number of rows yielded across batches.
//!
//! Pulls batches from the child operator, passing through at most `remaining`
//! active rows total. Entire batches are passed through when they fall within
//! the limit window. A partial batch is trimmed by rebuilding the selection
//! vector to include only the first `remaining` active entries. Once the
//! limit count is exhausted, subsequent calls return `None`.
//!
//! ```text
//!  LIMIT 5:
//!
//!  batch 1 (3 rows) ──► pass through (remaining: 5 -> 2)
//!  batch 2 (4 rows) ──► trim to 2 rows (remaining: 2 -> 0)
//!  batch 3           ──► None (exhausted)
//! ```

use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx};

pub struct LimitOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    remaining: usize,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> LimitOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        limit: usize,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            remaining: limit,
            idx,
        }
    }
}

// TODO: implement size_hint for all operators

impl<'a> Iterator for LimitOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.remaining == 0 {
                return None;
            }

            let mut batch = match self.child.next()? {
                Ok(batch) => batch,
                Err(e) => return Some(Err(e)),
            };

            let active = batch.active_len();

            if active == 0 {
                // Skip empty batches.
                continue;
            }

            if self.remaining >= active {
                // Entire batch fits within the limit window — pass through.
                self.remaining -= active;
                return Some(Ok(batch));
            }

            // Partial limit: collect active indices, keep only the first
            // `remaining` entries, and build a new selection vector.
            let new_sel: Vec<u16> = batch
                .active_indices()
                .take(self.remaining)
                .map(|i| i as u16)
                .collect();
            self.remaining = 0;
            batch.set_selection(new_sel);
            return Some(Ok(batch));
        }
    }
}
