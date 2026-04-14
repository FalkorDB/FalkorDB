//! Batch-mode commit operator — materializes pending mutations.
//!
//! This is a *blocking* operator: it drains all child batches first
//! (collecting all result environments), then calls `pending.commit()`
//! to apply batched creates, deletes, and property changes to the
//! underlying graph. After the commit succeeds, the collected
//! environments are yielded as batches.
//!
//! ```text
//!  Child (Create/Delete/Set/Merge ops accumulate into Pending)
//!       │
//!       ▼  drain ALL batches
//!  ┌────────────────────┐
//!  │ collected batches   │
//!  └─────────┬──────────┘
//!            │
//!   pending.commit() ──► apply to graph
//!            │
//!       yield collected batches
//! ```
//!
//! Only allowed in write queries; returns an error for `GRAPH.RO_QUERY`.

use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx};

pub struct CommitOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Option<Box<BatchOp<'a>>>,
    results: Vec<Batch<'a>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> CommitOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Result<Self, String> {
        if !runtime.write {
            return Err(String::from(
                "graph.RO_QUERY is to be executed only on read-only queries",
            ));
        }
        Ok(Self {
            runtime,
            child: Some(child),
            results: Vec::new(),
            idx,
        })
    }
}

impl<'a> Iterator for CommitOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // On first call, drain the entire child and commit.
        if let Some(mut child) = self.child.take() {
            loop {
                match child.next() {
                    Some(Ok(batch)) => {
                        self.results.push(batch);
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => break,
                }
            }
            if let Err(e) = self
                .runtime
                .pending
                .borrow_mut()
                .commit(&self.runtime.g, &self.runtime.stats)
            {
                return Some(Err(e));
            }
            // Commit succeeded — build effects buffer from pending data, then clear.
            {
                let pending = self.runtime.pending.borrow();
                if pending.effects_count() > 0 {
                    let mut buf_ref = self.runtime.effects_buffer.borrow_mut();
                    let buf = buf_ref.get_or_insert_with(Vec::new);
                    let n_effects = pending.build_effects_buffer(&self.runtime.g, buf);
                    self.runtime
                        .effects_count
                        .set(self.runtime.effects_count.get() + n_effects);
                }
            }
            self.runtime.pending.borrow_mut().clear();
            // Update schema baseline so the next commit in this query only
            // emits newly added schema entries.
            self.runtime
                .pending
                .borrow_mut()
                .set_schema_baseline(&self.runtime.g);
            // Reverse once so we can pop from the end in O(1) while preserving order.
            self.results.reverse();
        }

        // Yield collected batches one at a time.
        self.results.pop().map(Ok)
    }
}
