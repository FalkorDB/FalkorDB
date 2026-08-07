//! Batch-mode distinct operator — deduplicates result rows across batches.
//!
//! Pulls batches from the child operator and filters out rows whose
//! projected return columns have been seen before (by hash). Uses
//! `batch.get()` to read return-name variables and `set_selection`
//! for zero-copy filtering.
//!
//! ```text
//!  Input batch
//!       │
//!  ┌────▼──────────────────────────────────┐
//!  │ for each active row:                  │
//!  │   hash(return_col1, return_col2, ...) │
//!  │   seen before? ──► skip               │
//!  │   new?         ──► add to selection   │
//!  └────┬──────────────────────────────────┘
//!       │
//!  output batch with selection vector (zero-copy)
//! ```
//!
//! The deduplication state persists across batches so that duplicates
//! appearing in later batches are still filtered.

use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::{ReturnNames, Runtime},
    value::{Value, ValuesDeduper},
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

pub struct DistinctOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    deduper: ValuesDeduper,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> DistinctOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            // Pre-size the dedup set so it does not rehash through several small
            // capacities on every query. DISTINCT result sets in traversal/
            // expansion queries are commonly in the hundreds-to-low-thousands.
            deduper: ValuesDeduper::with_capacity(1024),
            idx,
        }
    }
}

impl<'a> Iterator for DistinctOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut batch = match self.child.next()? {
                Ok(batch) => batch,
                Err(e) => return Some(Err(e)),
            };

            let child_names = self.runtime.plan.node(self.idx).child(0).get_return_names();
            let names = if child_names.is_empty() {
                &self.runtime.return_names
            } else {
                &child_names
            };

            let mut passing = Vec::with_capacity(batch.active_len());

            for row in batch.active_indices() {
                let mut hasher = FxHasher::default();
                for name in names {
                    batch
                        .value_at(name.id, row)
                        .unwrap_or(Value::Null)
                        .hash(&mut hasher);
                }
                if self.deduper.has_hash(hasher.finish()) {
                    continue;
                }
                passing.push(row as u16);
            }

            if passing.is_empty() {
                continue;
            }

            batch.set_selection(passing);
            return Some(Ok(batch));
        }
    }
}
