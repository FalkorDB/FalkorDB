//! Shared columnar emit for the node-scan operators.
//!
//! Every node-scan op — [`NodeByLabelScanOp`](super::node_by_label_scan::NodeByLabelScanOp),
//! [`NodeByIdSeekOp`](super::node_by_id_seek::NodeByIdSeekOp),
//! [`NodeByLabelAndIdScanOp`](super::node_by_label_and_id_scan::NodeByLabelAndIdScanOp)
//! and [`NodeByIndexScanOp`](super::node_by_index_scan::NodeByIndexScanOp) — has the
//! same shape: pull a parent batch from the child, produce a node-id iterator for
//! each active parent row, then pack up to [`BATCH_SIZE`] `(parent_row, node_id)`
//! pairs into one columnar output batch. The ops differ only in *how* each row's
//! iterator is produced (and any per-node filtering, which they fold into the
//! pushed iterator); the pack-and-gather emit is identical and lives here.
//!
//! ```text
//!  child BatchOp ──► parent batch ──► set_batch()
//!                         │
//!            for each active parent row:
//!              op-specific node-id iterator ──► push(row, iter)
//!                         │
//!              ┌──────────┴───────────┐
//!              │ pack ≤ BATCH_SIZE    │  emit()
//!              │ (parent_row, nodeid) │
//!              └──────────┬───────────┘
//!                         │
//!     gather parent columns + origin per node id, attach Column::NodeIds
//! ```

use std::collections::VecDeque;

use crate::graph::graph::NodeId;
use crate::runtime::batch::{BATCH_SIZE, Batch, Column};

/// Owns the parent batch being expanded plus the queue of per-row node-id
/// iterators, and performs the shared pack-and-gather emit for the node-scan ops.
pub(crate) struct ScanEmitter<'a> {
    /// Alias the scanned node is bound to; the emitted [`Column::NodeIds`] column.
    alias_id: u32,
    /// Parent batch currently being expanded. Emitted rows are produced by
    /// `gather`ing this batch, which replicates every carried-forward parent
    /// column (and correlation origin) once per matching node id.
    batch: Option<Batch<'a>>,
    /// Per-parent-row node-id iterators keyed by their parent row index within
    /// `batch`: `(parent_row, node_iterator)`. Any per-node filtering is folded
    /// into the iterator by the op that produced it.
    pending: VecDeque<(usize, Box<dyn Iterator<Item = NodeId> + 'a>)>,
}

impl<'a> ScanEmitter<'a> {
    pub(crate) const fn new(alias_id: u32) -> Self {
        Self {
            alias_id,
            batch: None,
            pending: VecDeque::new(),
        }
    }

    /// True when the pending queue is drained and the op must refill it from its
    /// child before the next [`emit`](Self::emit).
    pub(crate) fn needs_refill(&self) -> bool {
        self.pending.is_empty()
    }

    /// Install the parent batch whose rows the queued iterators expand. Called
    /// once per refill, after pushing that batch's per-row iterators.
    pub(crate) fn set_batch(
        &mut self,
        batch: Batch<'a>,
    ) {
        self.batch = Some(batch);
    }

    /// Queue a parent row's node-id iterator. The op folds any per-node filter
    /// (id range, extra-label verification, …) into `iter`.
    pub(crate) fn push(
        &mut self,
        row: usize,
        iter: Box<dyn Iterator<Item = NodeId> + 'a>,
    ) {
        self.pending.push_back((row, iter));
    }

    /// Pack up to [`BATCH_SIZE`] `(parent row, node id)` pairs from the pending
    /// queue into one columnar batch. `gather` replicates each parent row's
    /// columns (and correlation origin) once per matching node id; the scanned
    /// node is attached as a [`Column::NodeIds`]. The per-row index vector is
    /// only needed when the parent carries columns — a leaf scan skips it and
    /// emits a standalone [`Column::NodeIds`] batch. Returns `None` when the
    /// queue drained without yielding a node, in which case the caller refills.
    pub(crate) fn emit(&mut self) -> Option<Batch<'a>> {
        let should_expand_batch = self.batch.as_ref().is_some_and(|b| b.num_columns() > 0);
        let mut indices: Vec<usize> = if should_expand_batch {
            Vec::with_capacity(BATCH_SIZE)
        } else {
            Vec::new()
        };
        let mut ids: Vec<NodeId> = Vec::with_capacity(BATCH_SIZE);
        while ids.len() < BATCH_SIZE {
            let Some((row, iter)) = self.pending.front_mut() else {
                break;
            };
            if let Some(nid) = iter.next() {
                if should_expand_batch {
                    indices.push(*row);
                }
                ids.push(nid);
            } else {
                self.pending.pop_front();
            }
        }
        if ids.is_empty() {
            return None;
        }
        let batch = self
            .batch
            .as_ref()
            .expect("batch is set while pending is non-empty");
        let mut out = if should_expand_batch {
            batch.gather(&indices)
        } else {
            Batch::new(0)
        };
        out.set_column(self.alias_id, Column::NodeIds(ids));
        Some(out)
    }
}
