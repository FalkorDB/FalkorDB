//! Shared batched-result emit: packs per-row expansions into columnar batches.
//!
//! Several operators have the same shape: pull a parent batch from the child,
//! produce a per-active-row iterator of results, then pack up to [`BATCH_SIZE`]
//! `(parent_row, result)` pairs into one columnar output batch. The operators
//! differ only in *how* each row's iterator is produced (and any per-row
//! filtering, which they fold into the pushed iterator); the pack-and-gather
//! emit is identical and lives here.
//!
//! The shape of the emitted result columns is chosen by the [`GatherItem`] type
//! parameter, which knows how to [`scatter`](GatherItem::scatter) a packed
//! `Vec` of results into one or more columns on the gathered batch. A single id
//! (`NodeId`/`RelationshipId`) binds one column; an emitter built with
//! [`new_without_alias`](BatchedResultEmitter::new_without_alias) binds none and
//! just gathers the matched input rows forward; richer items (scored scans,
//! edge-with-endpoints, unwound values) bind several columns at once.
//!
//! ```text
//!  child BatchOp ──► parent batch ──► set_batch()
//!                         │
//!            for each active parent row:
//!              op-specific result iterator ──► push(row, iter)
//!                         │
//!              ┌──────────┴───────────┐
//!              │ pack ≤ BATCH_SIZE    │  emit()
//!              │ (parent_row, result) │
//!              └──────────┬───────────┘
//!                         │
//!     gather parent columns + origin per result, then scatter the result
//!     columns (if any)
//! ```

use std::collections::VecDeque;

use crate::graph::graph::{NodeId, RelationshipId};
use crate::runtime::batch::{BATCH_SIZE, Batch, Column};
use crate::runtime::value::Value;

/// A per-row result that knows how to scatter a packed batch of itself into one
/// or more output columns. [`Binding`](GatherItem::Binding) carries the
/// per-operator metadata (which alias slots to bind, and any layout flags).
pub(crate) trait GatherItem {
    /// Per-operator binding metadata consumed by [`scatter`](Self::scatter).
    type Binding;

    /// Scatter the packed results into columns on `out`, which has already been
    /// gathered to one row per result.
    fn scatter(
        items: Vec<Self>,
        binding: &Self::Binding,
        out: &mut Batch,
    ) where
        Self: Sized;
}

impl GatherItem for NodeId {
    /// `Some(alias)` binds the node-id column; `None` binds no column.
    type Binding = Option<u32>;

    fn scatter(
        items: Vec<Self>,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        if let Some(alias) = binding {
            out.set_column(*alias, Column::NodeIds(items));
        }
    }
}

impl GatherItem for RelationshipId {
    /// `Some(alias)` binds the relationship-id column; `None` binds no column.
    type Binding = Option<u32>;

    fn scatter(
        items: Vec<Self>,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        if let Some(alias) = binding {
            out.set_column(*alias, Column::RelIds(items));
        }
    }
}

/// Binding for a scan that yields an id plus an optional relevance score
/// (fulltext / vector index scans).
pub(crate) struct ScoredColumn {
    /// Alias of the bound entity (node or relationship).
    pub(crate) id: u32,
    /// Alias of the score column, when a score yield variable is present.
    pub(crate) score: Option<u32>,
}

impl GatherItem for (NodeId, f64) {
    type Binding = ScoredColumn;

    fn scatter(
        items: Vec<Self>,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        // Hoist the score-bound check out of the loop: when a score column is
        // bound, `unzip` keeps the id and score columns the same length by
        // construction; otherwise we never build the score column at all.
        if let Some(score_alias) = binding.score {
            let (nodes, scores): (Vec<NodeId>, Vec<f64>) = items.into_iter().unzip();
            out.set_column(binding.id, Column::NodeIds(nodes));
            out.set_column(score_alias, Column::Floats(scores));
        } else {
            let nodes: Vec<NodeId> = items.into_iter().map(|(id, _)| id).collect();
            out.set_column(binding.id, Column::NodeIds(nodes));
        }
    }
}

impl GatherItem for (RelationshipId, f64) {
    type Binding = ScoredColumn;

    fn scatter(
        items: Vec<Self>,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        if let Some(score_alias) = binding.score {
            let (edges, scores): (Vec<RelationshipId>, Vec<f64>) = items.into_iter().unzip();
            out.set_column(binding.id, Column::RelIds(edges));
            out.set_column(score_alias, Column::Floats(scores));
        } else {
            let edges: Vec<RelationshipId> = items.into_iter().map(|(id, _)| id).collect();
            out.set_column(binding.id, Column::RelIds(edges));
        }
    }
}

/// Binding for an edge scan that yields both endpoints and the edge id. The
/// per-row iterator yields `(src, dst, edge)` in graph-tensor order;
/// `transposed` flips which pattern endpoint (from/to) each graph side fills.
pub(crate) struct EdgeEndpoints {
    /// Alias of the pattern's `from` endpoint.
    pub(crate) from: u32,
    /// Alias of the pattern's `to` endpoint, or `None` for a self-loop pattern
    /// where both endpoints share one alias (bound once, via `from`).
    pub(crate) to: Option<u32>,
    /// Alias of the relationship variable.
    pub(crate) edge: u32,
    /// When true, the graph `src`/`dst` map to the pattern's `to`/`from`.
    pub(crate) transposed: bool,
}

impl GatherItem for (NodeId, NodeId, RelationshipId) {
    type Binding = EdgeEndpoints;

    fn scatter(
        items: Vec<Self>,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        let mut froms = Vec::with_capacity(items.len());
        let mut edges = Vec::with_capacity(items.len());
        // Hoist the `to`-bound check out of the loop. A distinct `to` endpoint
        // is gathered alongside `from` (same length by construction); a
        // self-loop pattern shares one alias, so we bind it once via `from` and
        // never build the `to` column.
        if let Some(to_alias) = binding.to {
            let mut tos = Vec::with_capacity(items.len());
            for (src, dst, edge) in items {
                let (from_node, to_node) = if binding.transposed {
                    (dst, src)
                } else {
                    (src, dst)
                };
                froms.push(from_node);
                tos.push(to_node);
                edges.push(edge);
            }
            out.set_column(binding.from, Column::NodeIds(froms));
            out.set_column(to_alias, Column::NodeIds(tos));
            out.set_column(binding.edge, Column::RelIds(edges));
        } else {
            for (src, dst, edge) in items {
                froms.push(if binding.transposed { dst } else { src });
                edges.push(edge);
            }
            out.set_column(binding.from, Column::NodeIds(froms));
            out.set_column(binding.edge, Column::RelIds(edges));
        }
    }
}

impl GatherItem for Value {
    /// Alias of the bound variable. `set_column` upgrades the value column to
    /// the best lossless stored shape (ints/floats), so this preserves the
    /// column specialization a row builder would have produced.
    type Binding = u32;

    fn scatter(
        items: Vec<Self>,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        out.set_column(*binding, Column::Values(items));
    }
}

/// A parent row's queued results. `One` carries a single result inline (no heap
/// allocation) for the common case where a row yields exactly one; `Many` boxes
/// an iterator for rows that yield several.
enum RowIter<'a, I> {
    One(Option<I>),
    Many(Box<dyn Iterator<Item = I> + 'a>),
}

impl<I> Iterator for RowIter<'_, I> {
    type Item = I;

    #[inline]
    fn next(&mut self) -> Option<I> {
        match self {
            Self::Many(iter) => iter.next(),
            Self::One(item) => item.take(),
        }
    }
}

/// Owns the parent batch being expanded plus the queue of per-row result
/// iterators, and performs the shared pack-and-gather emit.
pub(crate) struct BatchedResultEmitter<'a, I: GatherItem> {
    /// Per-operator binding describing how packed results scatter into columns.
    binding: I::Binding,
    /// Parent batch currently being expanded. Emitted rows are produced by
    /// `gather`ing this batch, which replicates every carried-forward parent
    /// column (and correlation origin) once per result.
    batch: Option<Batch<'a>>,
    /// Per-parent-row result iterators keyed by their parent row index within
    /// `batch`: `(parent_row, results)`. Any per-row filtering is folded into
    /// the iterator by the op that produced it.
    pending: VecDeque<(usize, RowIter<'a, I>)>,
}

impl<'a, I: GatherItem> BatchedResultEmitter<'a, I> {
    /// Emitter for an arbitrary [`GatherItem`] binding (multi-column scans,
    /// unwound values, ...). Id-column operators use the [`new`](Self::new) /
    /// [`new_without_alias`](Self::new_without_alias) convenience constructors
    /// instead.
    pub(crate) const fn with_binding(binding: I::Binding) -> Self {
        Self {
            binding,
            batch: None,
            pending: VecDeque::new(),
        }
    }

    /// True when the pending queue is drained and the op must refill it from its
    /// child before the next [`emit`](Self::emit).
    pub(crate) fn needs_refill(&self) -> bool {
        self.pending.is_empty()
    }

    /// Number of queued per-row result iterators. Operators that queue several
    /// small per-row results before emitting (e.g. `UNWIND` of a list literal)
    /// use this to pack a batch's worth across rows while bounding queued work.
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Install the parent batch whose rows the queued iterators expand. Called
    /// once per refill, after pushing that batch's per-row iterators.
    pub(crate) fn set_batch(
        &mut self,
        batch: Batch<'a>,
    ) {
        self.batch = Some(batch);
    }

    /// The parent batch currently being expanded, if any. Operators that push
    /// one row's iterator at a time (to bound peak memory) read it back here to
    /// build the next row's view without keeping a second copy of the batch.
    pub(crate) const fn batch(&self) -> Option<&Batch<'a>> {
        self.batch.as_ref()
    }

    /// Drop all queued state. Used by correlated (Apply) plans that re-seed the
    /// scan with a new argument batch and must not replay stale rows.
    pub(crate) fn reset(&mut self) {
        self.pending.clear();
        self.batch = None;
    }

    /// Queue a parent row's result iterator. The op folds any per-row filter
    /// into `iter`. Use [`push_one`](Self::push_one) instead when the row yields
    /// a single result to skip the boxed-iterator allocation.
    pub(crate) fn push(
        &mut self,
        row: usize,
        iter: Box<dyn Iterator<Item = I> + 'a>,
    ) {
        self.pending.push_back((row, RowIter::Many(iter)));
    }

    /// Queue a parent row that yields exactly one result, stored inline without
    /// a heap allocation.
    pub(crate) fn push_one(
        &mut self,
        row: usize,
        item: I,
    ) {
        self.pending.push_back((row, RowIter::One(Some(item))));
    }

    /// Pack up to [`BATCH_SIZE`] `(parent row, result)` pairs from the pending
    /// queue into one columnar batch. `gather` replicates each parent row's
    /// columns (and correlation origin) once per matching result; the result
    /// columns are then attached via [`GatherItem::scatter`]. The per-row index
    /// vector is only needed when the parent carries columns *or* a correlation
    /// origin sidecar — a bare column-less (leaf) parent skips it and emits a
    /// standalone result batch. Returns `None` when the queue drained without
    /// yielding a result, in which case the caller refills.
    pub(crate) fn emit(&mut self) -> Option<Batch<'a>> {
        // Gather (rather than build a standalone batch) whenever the parent has
        // columns to replicate *or* per-row correlation origins to carry forward;
        // otherwise a 0-column correlated parent would drop its origins, and a
        // no-alias emitter (which never sets a column) would emit an empty batch.
        let should_expand_batch = self
            .batch
            .as_ref()
            .is_some_and(|b| b.num_columns() > 0 || b.has_origin_rows());
        let mut indices: Vec<usize> = if should_expand_batch {
            Vec::with_capacity(BATCH_SIZE)
        } else {
            Vec::new()
        };
        let mut items: Vec<I> = Vec::with_capacity(BATCH_SIZE);
        while items.len() < BATCH_SIZE {
            let Some((row, iter)) = self.pending.front_mut() else {
                break;
            };
            if let Some(item) = iter.next() {
                if should_expand_batch {
                    indices.push(*row);
                }
                items.push(item);
            } else {
                self.pending.pop_front();
            }
        }
        if items.is_empty() {
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
        I::scatter(items, &self.binding, &mut out);
        Some(out)
    }
}

/// Convenience constructors for single-id scans (`NodeId` / `RelationshipId`),
/// whose binding is just the optional alias of the bound column. Defined once
/// over every such item so `BatchedResultEmitter::new(..)` stays unambiguous at
/// call sites that pin the item type through the field annotation.
impl<'a, I> BatchedResultEmitter<'a, I>
where
    I: GatherItem<Binding = Option<u32>>,
{
    /// Emitter that binds the packed ids to `alias`.
    pub(crate) const fn new(alias: u32) -> Self {
        Self::with_binding(Some(alias))
    }

    /// Emitter that binds no column: the pushed ids only drive how many rows
    /// each input row yields, and the matched input rows are gathered forward
    /// unchanged.
    pub(crate) const fn new_without_alias() -> Self {
        Self::with_binding(None)
    }
}
