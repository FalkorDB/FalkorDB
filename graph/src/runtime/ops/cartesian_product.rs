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
//! With multiple right children, each branch is materialized independently and
//! the branches are then walked as an odometer: memory is the *sum* of the
//! branch sizes, never their product, and rows leave the operator
//! [`BATCH_SIZE`] at a time so timeout and memory-capacity checks get a chance
//! to fire.

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
    /// Lazily materialized right-side rows, one dense batch per right branch.
    /// `None` means not yet computed.
    materialized_right: Option<Vec<Batch<'a>>>,
    /// Current block of left-side rows being cross-joined.
    left_batch: Option<Batch<'a>>,
    /// Current position within `left_batch`.
    left_pos: usize,
    /// Odometer position within each materialized right branch.
    right_pos: Vec<usize>,
}

/// Advances `pos` to the next combination of right-branch rows, rightmost digit
/// first. Returns `false` once it wraps back to all-zeros, i.e. the current left
/// row has been cross-joined against every combination.
fn advance(
    pos: &mut [usize],
    rights: &[Batch],
) -> bool {
    for (p, branch) in pos.iter_mut().zip(rights).rev() {
        *p += 1;
        if *p < branch.len() {
            return true;
        }
        *p = 0;
    }
    false
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
            right_pos: Vec::new(),
        }
    }

    /// Runs each right sub-plan once, collecting its rows into one dense
    /// [`Batch`] per branch. The branches are *not* cross-joined here — that is
    /// streamed row by row in [`Iterator::next`] — so the memory held is the sum
    /// of the branch sizes rather than their product.
    fn materialize_right(&mut self) -> Result<Vec<Batch<'a>>, String> {
        let mut branch_results: Vec<Batch<'a>> = Vec::with_capacity(self.right_children.len());

        for child in &mut self.right_children {
            let mut batches: Vec<Batch<'a>> = Vec::new();
            for result in child.by_ref() {
                batches.push(result?);
            }
            branch_results.push(Batch::concat(&batches));
        }

        Ok(branch_results)
    }
}

impl<'a> Iterator for CartesianProductOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Lazy materialization of right side (runs once).
        if self.materialized_right.is_none() {
            match self.materialize_right() {
                Ok(rights) => {
                    // An empty branch makes the whole cross-product empty.
                    if rights.iter().any(Batch::is_empty) {
                        return None;
                    }
                    self.right_pos = vec![0; rights.len()];
                    self.materialized_right = Some(rights);
                }
                Err(e) => return Some(Err(e)),
            }
        }

        let rights = self.materialized_right.as_ref().unwrap();
        let mut cursor: Vec<(&Batch<'a>, usize)> = Vec::with_capacity(rights.len());
        let mut builder = BatchBuilder::new();

        loop {
            // Produce cross-product rows from current (left_pos, right_pos).
            let left_len = self.left_batch.as_ref().map_or(0, Batch::len);
            if let [right] = rights.as_slice() {
                // Single branch — the overwhelmingly common shape. One cursor
                // means no odometer: scan the branch's rows in a tight loop with
                // the position held in a local.
                let right_len = right.len();
                let mut pos = self.right_pos[0];
                while builder.len() < BATCH_SIZE && self.left_pos < left_len {
                    let left = self.left_batch.as_ref().unwrap();
                    let origin = left.origin_row(self.left_pos);
                    while builder.len() < BATCH_SIZE && pos < right_len {
                        builder.push_merged(left, self.left_pos, right, pos, origin);
                        pos += 1;
                    }
                    if pos >= right_len {
                        self.left_pos += 1;
                        pos = 0;
                    }
                }
                self.right_pos[0] = pos;
            } else {
                while builder.len() < BATCH_SIZE && self.left_pos < left_len {
                    let left = self.left_batch.as_ref().unwrap();
                    let origin = left.origin_row(self.left_pos);
                    cursor.clear();
                    cursor.extend(rights.iter().zip(self.right_pos.iter().copied()));
                    builder.push_merged_many(left, self.left_pos, &cursor, origin);
                    if !advance(&mut self.right_pos, rights) {
                        // Odometer wrapped: this left row is fully cross-joined.
                        self.left_pos += 1;
                    }
                }
            }

            // If batch is full, return it (positions preserved for next call).
            if builder.len() >= BATCH_SIZE {
                return Some(Ok(builder.finish()));
            }

            // Current left block exhausted. Load next batch from left child.
            self.left_batch = None;
            self.left_pos = 0;
            self.right_pos.fill(0);

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
