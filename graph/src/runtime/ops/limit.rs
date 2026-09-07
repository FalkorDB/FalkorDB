//! Batch-mode limit operator — caps the number of rows yielded across batches.
//!
//! Pulls batches from the child operator, passing through at most `remaining`
//! active rows total. Entire batches are passed through when they fall within
//! the limit window. A partial batch is trimmed by rebuilding the selection
//! vector to include only the first `remaining` active entries. Once the
//! limit count is exhausted, subsequent calls return `None`.
//!
//! Special case: LIMIT 0 returns one empty batch (to preserve schema and
//! execute child side effects) then exhausts.
//!
//! ```text
//!  LIMIT 5:
//!
//!  batch 1 (3 rows) ──► pass through (remaining: 5 -> 2)
//!  batch 2 (4 rows) ──► trim to 2 rows (remaining: 2 -> 0)
//!  batch 3           ──► None (exhausted)
//!
//!  LIMIT 0:
//!
//!  batch 1 (N rows) ──► trim to 0 rows, return empty batch
//!  batch 2           ──► None (exhausted)
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
    /// For LIMIT 0, we still need to pull one batch from the child to
    /// execute any side effects and preserve output schema, then return
    /// an empty batch. This flag tracks whether we've done that.
    limit_zero_emitted: bool,
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
            limit_zero_emitted: false,
            idx,
        }
    }
}

// TODO: implement size_hint for all operators

impl<'a> Iterator for LimitOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Special handling for LIMIT 0: we must still pull one batch from
        // the child to execute any side effects (e.g., pending mutations in
        // write queries) and to preserve the output schema. We return an
        // empty batch once, then exhaust.
        if self.remaining == 0 {
            if !self.limit_zero_emitted {
                self.limit_zero_emitted = true;
                // Pull one batch from child to execute side effects, then
                // return an empty batch with the same schema.
                if let Some(child_result) = self.child.next() {
                    let mut batch = child_result?;
                    // Trim to 0 active rows while preserving schema.
                    batch.set_selection(Vec::new());
                    return Some(Ok(batch));
                }
                // Child yielded nothing — return empty batch from default.
                let mut batch = self.runtime.default_batch();
                batch.set_selection(Vec::new());
                return Some(Ok(batch));
            }
            return None;
        }

        loop {
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
