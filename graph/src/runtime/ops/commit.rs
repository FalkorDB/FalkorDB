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
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct CommitOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Option<Box<BatchOp<'a>>>,
    results: Vec<Batch<'a>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// When true, this Commit is the root of the plan (no parent) and
    /// no downstream operator will consume the batches. We drain the
    /// child but discard the batches to avoid allocating a large result
    /// vector that nobody reads.
    is_root: bool,
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
        let is_root = runtime.plan.node(idx).parent().is_none();
        Ok(Self {
            runtime,
            child: Some(child),
            results: Vec::new(),
            idx,
            is_root,
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
                        if !self.is_root {
                            self.results.push(batch);
                        }
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => break,
                }
            }
            // Applying `pending` mutates only this query's *private* MVCC graph
            // version, so it needs no lock at all — keep it outside the writer
            // window, which the host holds its global lock for.
            if let Err(e) = self
                .runtime
                .pending
                .borrow_mut()
                .commit(&self.runtime.g, &self.runtime.stats)
            {
                return Some(Err(e));
            }
            // Publishing index documents *is* a mutation of shared, non-MVCC
            // state, so escalate to writer mode first. The host takes its global
            // lock and the per-graph write lock and keeps them for the rest of the
            // query (see `crate::query_lock`); idempotent, so nested Commits after
            // the first are free.
            if let Err(e) = crate::query_lock::upgrade_to_write() {
                return Some(Err(e));
            }
            // Publish this commit's index documents now, so an operator above us
            // can scan what an earlier subquery wrote (e.g.
            // `CREATE (n:L) WITH n MATCH (m:L) WHERE m.id > 0`). Safe against
            // concurrent readers because we hold the write lock; safe against this
            // query failing later because `resync_published_indexes` brings the
            // index back in line with committed state — the same guarantee C gets
            // from its undo log.
            self.runtime.commit_deferred_indexes();
            // Commit succeeded — build effects buffer from pending data, then clear.
            {
                let pending = self.runtime.pending.borrow();
                let estimated = pending.effects_count();
                if estimated > 0 {
                    if self.runtime.build_effects.get() {
                        let mut buf_ref = self.runtime.effects_buffer.borrow_mut();
                        let buf = buf_ref.get_or_insert_with(Vec::new);
                        let n_effects = pending.build_effects_buffer(&self.runtime.g, buf);
                        self.runtime
                            .effects_count
                            .set(self.runtime.effects_count.get() + n_effects);
                    } else {
                        // Keep the count accurate for `modified` bookkeeping
                        // even when the buffer itself is not needed.
                        self.runtime
                            .effects_count
                            .set(self.runtime.effects_count.get() + estimated);
                    }
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
