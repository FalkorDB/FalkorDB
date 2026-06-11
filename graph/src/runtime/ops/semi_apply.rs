//! Batch-mode semi-apply operator — existence-based filtering via a sub-plan.
//!
//! Implements Cypher `WHERE EXISTS { ... }` (semi) and
//! `WHERE NOT EXISTS { ... }` (anti-semi) patterns.
//!
//! ```text
//!  Input batch [row0, row1, row2]
//!       │
//!  stamp origin_row
//!       │
//!  ┌────▼─────────────┐
//!  │ Right sub-plan    │  single instance, all rows at once
//!  └────┬─────────────┘
//!       │
//!  collect matched origin_rows
//!       │
//!  ┌────▼──────────────────────────────────┐
//!  │ Semi mode:  keep rows WITH matches    │
//!  │ Anti mode:  keep rows WITHOUT matches │
//!  └──────────────────────────────────────┘
//! ```
//!
//! For each input batch, runs the right sub-plan once with all active rows as
//! the argument batch. Uses `origin_row` on each output env to determine which
//! input rows had matches, then builds a selection vector accordingly.

use std::collections::HashSet;

use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct SemiApplyOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    is_anti: bool,
    right_child_idx: NodeIdx<Dyn<IR>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> SemiApplyOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        is_anti: bool,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let right_child_idx = runtime.plan.node(idx).child(1).idx();

        Self {
            runtime,
            child,
            is_anti,
            right_child_idx,
            idx,
        }
    }
}

impl<'a> Iterator for SemiApplyOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut batch = match self.child.next()? {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };

            let active: Vec<usize> = batch.active_indices().collect();

            // Build argument batch with origin_row stamped on each env.
            let arg_batch = batch.clone_active_rows_seq_origin();

            // Create ONE subtree for all rows.
            let mut subtree = match self.runtime.run_batch(self.right_child_idx) {
                Ok(s) => s,
                Err(e) => return Some(Err(e)),
            };
            subtree.set_argument_batch(arg_batch);

            // Collect which origin_rows produced at least one result.
            let mut matched = HashSet::new();
            for sub_result in subtree.by_ref() {
                match sub_result {
                    Ok(sub_batch) => {
                        for row in sub_batch.active_indices() {
                            matched.insert(sub_batch.origin_row(row));
                        }
                    }
                    Err(e) => return Some(Err(e)),
                }
            }
            drop(subtree);

            // Build selection based on match/anti semantics.
            let mut passing = Vec::new();
            for (i, &row_idx) in active.iter().enumerate() {
                let has_result = matched.contains(&(i as u32));
                if has_result ^ self.is_anti {
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
