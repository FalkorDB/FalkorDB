//! Batch-mode OR-apply multiplexer operator — evaluates multiple existence-check branches.
//!
//! Implements disjunctive patterns like `WHERE EXISTS {...} OR EXISTS {...}`.
//!
//! ```text
//!  Input batch [row0, row1, row2]
//!       │
//!  stamp origin_row
//!       │
//!  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
//!  │ Branch 1│  │ Branch 2│  │ Branch 3│    each gets a copy of all rows
//!  └────┬────┘  └────┬────┘  └────┬────┘
//!       │            │            │
//!  matched: {0,2}  matched: {1}  matched: {}
//!       │            │            │
//!       └────── OR (union) ──────┘
//!                    │
//!            overall: {0, 1, 2}
//!                    │
//!            selection vector
//! ```
//!
//! For each input batch, runs each branch sub-plan once with all active rows.
//! Uses `origin_row` on output envs to determine which input rows matched
//! each branch. A row is emitted when any branch produces a result (or, for
//! `anti` branches, when a branch produces NO results).

use std::collections::HashSet;

use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct OrApplyMultiplexerOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    anti_flags: &'a [bool],
    branch_indices: Vec<NodeIdx<Dyn<IR>>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> OrApplyMultiplexerOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        anti_flags: &'a [bool],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let num_branches = runtime.plan.node(idx).num_children() - 1;
        let mut branch_indices = Vec::with_capacity(num_branches);

        for i in 1..=num_branches {
            let branch_idx = runtime.plan.node(idx).child(i).idx();
            branch_indices.push(branch_idx);
        }

        Self {
            runtime,
            child,
            anti_flags,
            branch_indices,
            idx,
        }
    }
}

impl<'a> Iterator for OrApplyMultiplexerOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut batch = match self.child.next()? {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };

            let active: Vec<usize> = batch.active_indices().collect();

            // Track which origin_rows are matched across all branches.
            let mut overall_matched: HashSet<u32> = HashSet::new();

            for (branch_num, branch_idx) in self.branch_indices.iter().enumerate() {
                // Build a fresh argument batch for each branch (each subtree
                // consumes it); origin_row is stamped sequentially in active
                // order so branch results correlate back to the outer rows.
                let mut subtree = match self.runtime.run_batch(*branch_idx) {
                    Ok(s) => s,
                    Err(e) => return Some(Err(e)),
                };
                subtree.set_argument_batch(batch.clone_active_rows_seq_origin());

                // Collect which origin_rows this branch matched.
                let mut branch_matched: HashSet<u32> = HashSet::new();
                for sub_result in subtree.by_ref() {
                    match sub_result {
                        Ok(sub_batch) => {
                            for row in sub_batch.active_indices() {
                                branch_matched.insert(sub_batch.origin_row(row));
                            }
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                drop(subtree);

                let is_anti = self.anti_flags[branch_num];
                for (i, _) in active.iter().enumerate() {
                    let has_result = branch_matched.contains(&(i as u32));
                    if has_result ^ is_anti {
                        overall_matched.insert(i as u32);
                    }
                }
            }

            let mut passing = Vec::new();
            for (i, &row_idx) in active.iter().enumerate() {
                if overall_matched.contains(&(i as u32)) {
                    passing.push(row_idx as u16);
                }
            }

            if !passing.is_empty() {
                batch.set_selection(passing);
                return Some(Ok(batch));
            }
        }
    }
}
