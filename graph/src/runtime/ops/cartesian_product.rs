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
//! Each right child is materialized independently and the branches are then
//! walked as an odometer: memory is the *sum* of the branch sizes, never their
//! product, and rows leave the operator [`BATCH_SIZE`] at a time so timeout and
//! memory-capacity checks get a chance to fire.
//!
//! Which branch supplies which output column is fixed for the operator's whole
//! life, so it is resolved once into a [`MergePlan`] and a row then costs one
//! indexed read per column regardless of how many branches there are. That makes
//! a single right branch the odometer's degenerate case rather than a shape
//! worth special-casing.

use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, MergePlan},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx};

pub struct CartesianProductOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Pre-built right branch operators, seeded via set_argument_batch.
    pub(crate) right_children: Vec<BatchOp<'a>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// Lazily materialized right-hand side. `None` means not yet computed.
    right: Option<RightSide<'a>>,
    /// Current block of left-side rows being cross-joined.
    left_batch: Option<Batch<'a>>,
    /// Current position within `left_batch`.
    left_pos: usize,
}

/// The materialized right-hand side: one dense [`Batch`] per right branch, each
/// holding that branch's *entire* output rather than one batch of its stream,
/// together with the odometer cursor over them and the [`MergePlan`] that says
/// which branch supplies which output column.
///
/// The cursor lives here rather than beside the branches in the operator so the
/// two cannot fall out of step: `cursor.len() == branches.len()` holds by
/// construction, and stepping it is [`advance`](Self::advance)'s business alone —
/// no caller does position arithmetic.
struct RightSide<'a> {
    /// One dense batch per branch — a whole branch, not a batch of its stream.
    branches: Vec<Batch<'a>>,
    /// `cursor[i]` is the current row of `branches[i]`.
    cursor: Vec<usize>,
    /// `lens[i] == branches[i].len()`, carried separately so stepping the
    /// odometer touches two flat `usize` slices and never chases a pointer into
    /// a [`Batch`] — this is a per-row cost on every shape.
    lens: Vec<usize>,
    /// Resolved once at materialization: the branches, and therefore which of
    /// them binds each column, are fixed for the operator's whole life.
    plan: MergePlan,
}

impl<'a> RightSide<'a> {
    fn new(branches: Vec<Batch<'a>>) -> Self {
        // Every branch comes out of `Batch::concat`, which drops the selection
        // vector, so the odometer indexes rows `0..len()` directly instead of
        // walking `active_indices()`.
        debug_assert!(branches.iter().all(|b| b.selection().is_none()));
        Self {
            plan: MergePlan::for_rights(&branches),
            cursor: vec![0; branches.len()],
            lens: branches.iter().map(Batch::len).collect(),
            branches,
        }
    }

    /// An empty branch makes the whole cross-product empty.
    fn is_empty(&self) -> bool {
        self.branches.iter().any(Batch::is_empty)
    }

    /// Steps to the next combination of branch rows, rightmost branch fastest —
    /// the row order the nested per-branch merge produced. Returns `false` once
    /// the odometer wraps back to all-zeros, i.e. the current left row has been
    /// cross-joined against every combination.
    ///
    /// With no branches at all (a degenerate plan) this reports a wrap
    /// immediately, so the operator passes its left rows through unchanged.
    fn advance(&mut self) -> bool {
        for (pos, &len) in self.cursor.iter_mut().zip(&self.lens).rev() {
            *pos += 1;
            if *pos < len {
                return true;
            }
            *pos = 0;
        }
        false
    }

    /// Rewinds the odometer for a fresh block of left rows.
    fn rewind(&mut self) {
        self.cursor.fill(0);
    }
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
            right: None,
            left_batch: None,
            left_pos: 0,
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
        // Lazy materialization of right side (runs once). The result is recorded
        // even when a branch came back empty, so a re-poll after `None`
        // short-circuits below instead of running every right sub-plan again.
        if self.right.is_none() {
            match self.materialize_right() {
                Ok(branches) => self.right = Some(RightSide::new(branches)),
                Err(e) => return Some(Err(e)),
            }
        }

        let right = self.right.as_mut().unwrap();
        if right.is_empty() {
            return None;
        }

        let mut builder = BatchBuilder::new();

        loop {
            // Produce cross-product rows from the current (left_pos, cursor).
            // One shape for every branch count: `advance` owns the wrap, so this
            // loop does no position arithmetic of its own.
            //
            // The nesting is what keeps it cheap. Resolving the left batch and
            // reading its `origin_row` are invariant across every combination of
            // one left row, so they are hoisted out of the inner loop rather than
            // repeated per output row — for a 1,000-row branch that is once per
            // 1,000 rows instead of a million times.
            let left_len = self.left_batch.as_ref().map_or(0, Batch::len);
            while builder.len() < BATCH_SIZE && self.left_pos < left_len {
                let left = self.left_batch.as_ref().unwrap();
                let origin = left.origin_row(self.left_pos);
                // Emit this left row against every combination, or until the
                // batch fills — in which case the cursor holds where to resume.
                while builder.len() < BATCH_SIZE {
                    builder.push_planned(
                        left,
                        self.left_pos,
                        &right.branches,
                        &right.cursor,
                        &right.plan,
                        origin,
                    );
                    if !right.advance() {
                        // Odometer wrapped: this left row is fully cross-joined.
                        self.left_pos += 1;
                        break;
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
            right.rewind();

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

#[cfg(test)]
mod tests {
    use super::RightSide;
    use crate::runtime::batch::{Batch, BatchBuilder, Column, MergePlan};

    /// A dense batch binding each `(var_id, values)` pair as an `Int` column.
    fn batch(cols: &[(u32, &[i64])]) -> Batch<'static> {
        let mut b = Batch::new(0);
        for (id, vals) in cols {
            b.set_column(*id, Column::Ints(vals.to_vec()));
        }
        b
    }

    /// Asserts two batches agree row-for-row on every column's value *and* its
    /// bound bit — `Value` has no `PartialEq`, so values compare via `Debug`.
    fn assert_same(
        expected: &Batch,
        actual: &Batch,
    ) {
        assert_eq!(expected.len(), actual.len(), "row count");
        let width = expected.num_columns().max(actual.num_columns()) as u32;
        for row in 0..expected.len() {
            assert_eq!(
                expected.origin_row(row),
                actual.origin_row(row),
                "origin_row at row {row}"
            );
            for id in 0..width {
                assert_eq!(
                    expected.is_bound_at(id, row),
                    actual.is_bound_at(id, row),
                    "bound bit of col {id} at row {row}"
                );
                assert_eq!(
                    format!("{:?}", expected.value_at(id, row)),
                    format!("{:?}", actual.value_at(id, row)),
                    "value of col {id} at row {row}"
                );
            }
        }
    }

    /// The odometer + [`MergePlan`] must produce byte-for-byte what the original
    /// implementation did: cross-multiply the branches into one flat batch by
    /// chaining `push_merged`, then join that against the left rows. Covers a
    /// slot only the left binds (0), a slot two branches bind so the later one
    /// must win (1), slots single branches bind (2, 3), and a slot nobody binds
    /// (4).
    #[test]
    fn push_planned_matches_chained_push_merged() {
        let left = {
            let mut b = batch(&[(0, &[10, 20])]);
            b.set_column(4, Column::Unbound);
            b
        };
        let branches = vec![
            batch(&[(1, &[1, 2, 3])]),
            batch(&[(1, &[7, 8]), (2, &[70, 80])]),
            batch(&[(3, &[100])]),
        ];

        // Reference: the deleted algorithm, verbatim.
        let mut accumulated = branches[0].clone();
        for branch in &branches[1..] {
            let mut builder = BatchBuilder::new();
            for l in accumulated.active_indices() {
                let origin = accumulated.origin_row(l);
                for r in branch.active_indices() {
                    builder.push_merged(&accumulated, l, branch, r, origin);
                }
            }
            accumulated = builder.finish();
        }
        let mut expected = BatchBuilder::new();
        for l in left.active_indices() {
            let origin = left.origin_row(l);
            for r in accumulated.active_indices() {
                expected.push_merged(&left, l, &accumulated, r, origin);
            }
        }
        let expected = expected.finish();

        // Under test: stream the branches as an odometer.
        let mut right = RightSide::new(branches);
        let mut actual = BatchBuilder::new();
        for l in 0..left.len() {
            let origin = left.origin_row(l);
            loop {
                actual.push_planned(
                    &left,
                    l,
                    &right.branches,
                    &right.cursor,
                    &right.plan,
                    origin,
                );
                if !right.advance() {
                    break;
                }
            }
        }
        let actual = actual.finish();

        assert_eq!(expected.len(), 2 * 3 * 2 * 1, "2 left rows x 3 x 2 x 1");
        assert_same(&expected, &actual);
    }

    /// The odometer steps its rightmost branch fastest and reports the wrap only
    /// after every combination has been visited exactly once.
    #[test]
    fn odometer_visits_every_combination_rightmost_first() {
        let mut right = RightSide::new(vec![batch(&[(0, &[0, 1, 2])]), batch(&[(1, &[0, 1])])]);
        let mut seen = vec![right.cursor.clone()];
        while right.advance() {
            seen.push(right.cursor.clone());
        }
        assert_eq!(
            seen,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![1, 0],
                vec![1, 1],
                vec![2, 0],
                vec![2, 1]
            ]
        );
        // The wrap left the cursor rewound, ready for the next left row.
        assert_eq!(right.cursor, vec![0, 0]);
    }

    /// One branch is the odometer's degenerate case, not a special case: it
    /// advances until its rows run out, then wraps.
    #[test]
    fn odometer_with_one_branch_counts_its_rows() {
        let mut right = RightSide::new(vec![batch(&[(0, &[5, 6, 7])])]);
        let mut visited = 1;
        while right.advance() {
            visited += 1;
        }
        assert_eq!(visited, 3);
        assert_eq!(right.cursor, vec![0]);
    }

    /// An empty branch makes the whole product empty, whichever branch it is.
    #[test]
    fn any_empty_branch_makes_the_product_empty() {
        let empty = Batch::new(0);
        assert!(RightSide::new(vec![batch(&[(0, &[1])]), empty.clone()]).is_empty());
        assert!(RightSide::new(vec![empty, batch(&[(0, &[1])])]).is_empty());
        assert!(!RightSide::new(vec![batch(&[(0, &[1])])]).is_empty());
    }

    /// A degenerate plan with no right branches at all wraps immediately, so the
    /// operator emits each left row once instead of panicking.
    #[test]
    fn odometer_with_no_branches_wraps_immediately() {
        let mut right = RightSide::new(Vec::new());
        assert!(!right.is_empty());
        assert!(!right.advance());

        let left = batch(&[(0, &[10, 20])]);
        let mut out = BatchBuilder::new();
        for l in 0..left.len() {
            out.push_planned(&left, l, &right.branches, &right.cursor, &right.plan, 0);
        }
        assert_same(&left, &out.finish());
    }

    /// `MergePlan` resolves the owner of every column once; a later branch wins
    /// where two bind the same slot.
    #[test]
    fn merge_plan_lets_the_later_branch_win() {
        let left = batch(&[(0, &[10])]);
        let branches = vec![batch(&[(1, &[1])]), batch(&[(1, &[2])])];
        let plan = MergePlan::for_rights(&branches);
        assert_eq!(plan.width(), 2);

        let mut out = BatchBuilder::new();
        out.push_planned(&left, 0, &branches, &[0, 0], &plan, 0);
        let out = out.finish();
        assert_eq!(format!("{:?}", out.value_at(0, 0)), "Some(Int(10))");
        assert_eq!(format!("{:?}", out.value_at(1, 0)), "Some(Int(2))");
    }
}
