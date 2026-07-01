//! Vectorized block nested loop cartesian product operator.
//!
//! Materializes the right sub-plan(s) once on first use, then cross-joins
//! blocks of left rows with the materialized right rows.
//!
//! ```text
//!  Left child (streaming)          Right child(ren) (materialized once)
//!  ┌──────────────────┐            ┌──────────────────┐
//!  │ batch 1: [L1,L2] │            │ [R1, R2, R3]     │  <-- cached in memory
//!  │ batch 2: [L3]    │            └──────────────────┘
//!  │ ...               │
//!  └──────────────────┘
//!                 \                       /
//!                  \─── cross-join ──────/
//!                          │
//!                  ┌───────┴───────┐
//!                  │ L1+R1, L1+R2, │
//!                  │ L1+R3, L2+R1, │
//!                  │ L2+R2, L2+R3, │
//!                  │ L3+R1, ...    │
//!                  └───────────────┘
//! ```
//!
//! With multiple right children, each branch is materialized independently
//! and their cross-product is computed into a single flat vector before
//! joining with left rows.

use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx};

pub struct CartesianProductOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Pre-built right branch operators, seeded via set_argument_batch.
    pub(crate) right_children: Vec<BatchOp<'a>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// Lazily materialized right-side rows. `None` means not yet computed.
    materialized_right: Option<Batch<'a>>,
    /// Current block of left-side rows being cross-joined.
    left_batch: Option<Batch<'a>>,
    /// Current position within `left_batch`.
    left_pos: usize,
    /// Current position within `materialized_right`.
    right_pos: usize,
}

impl<'a> CartesianProductOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        right_children: Vec<BatchOp<'a>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            right_children,
            idx,
            materialized_right: None,
            left_batch: None,
            left_pos: 0,
            right_pos: 0,
        }
    }

    /// Materializes all right sub-plans into a single columnar [`Batch`].
    ///
    /// For a single right child, runs the sub-plan once and concatenates all
    /// rows. For multiple right children, materializes each independently and
    /// computes their cross-product into a single flat batch.
    fn materialize_right(&mut self) -> Result<Batch<'a>, String> {
        let mut branch_results: Vec<Batch<'a>> = Vec::with_capacity(self.right_children.len());

        for child in &mut self.right_children {
            let mut batches: Vec<Batch<'a>> = Vec::new();
            for result in child.by_ref() {
                batches.push(result?);
            }
            branch_results.push(Batch::concat(&batches));
        }

        // Single branch: no cross-product needed.
        if branch_results.len() == 1 {
            return Ok(branch_results.pop().unwrap());
        }

        // Multi-branch: iteratively cross-product all branches.
        let mut accumulated = branch_results.remove(0);
        for branch in branch_results {
            if accumulated.is_empty() || branch.is_empty() {
                return Ok(BatchBuilder::new().finish());
            }
            let mut builder = BatchBuilder::new();
            for left in accumulated.active_indices() {
                let origin = accumulated.origin_row(left);
                for right in branch.active_indices() {
                    builder.push_merged(&accumulated, left, &branch, right, origin);
                }
            }
            accumulated = builder.finish();
        }

        Ok(accumulated)
    }
}

impl<'a> Iterator for CartesianProductOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Lazy materialization of right side (runs once).
        if self.materialized_right.is_none() {
            match self.materialize_right() {
                Ok(right) => {
                    if right.is_empty() {
                        return None;
                    }
                    self.materialized_right = Some(right);
                }
                Err(e) => return Some(Err(e)),
            }
        }

        let right_len = self.materialized_right.as_ref().unwrap().len();
        let mut builder = BatchBuilder::new();

        loop {
            // Produce cross-product rows from current (left_pos, right_pos).
            let left_len = self.left_batch.as_ref().map_or(0, Batch::len);
            while builder.len() < BATCH_SIZE && self.left_pos < left_len {
                while builder.len() < BATCH_SIZE && self.right_pos < right_len {
                    let left = self.left_batch.as_ref().unwrap();
                    let right = self.materialized_right.as_ref().unwrap();
                    let origin = left.origin_row(self.left_pos);
                    builder.push_merged(left, self.left_pos, right, self.right_pos, origin);
                    self.right_pos += 1;
                }
                if self.right_pos >= right_len {
                    self.left_pos += 1;
                    self.right_pos = 0;
                }
            }

            // If batch is full, return it (positions preserved for next call).
            if builder.len() >= BATCH_SIZE {
                return Some(Ok(builder.finish()));
            }

            // Current left block exhausted. Load next batch from left child.
            self.left_batch = None;
            self.left_pos = 0;
            self.right_pos = 0;

            match self.child.next() {
                Some(Ok(batch)) => {
                    self.left_batch = Some(batch.into_compacted());
                }
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    if builder.is_empty() {
                        return None;
                    }
                    return Some(Ok(builder.finish()));
                }
            }
        }
    }
}
