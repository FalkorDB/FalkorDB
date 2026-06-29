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
//! parameter, which knows how to accumulate packed results into typed column
//! *lanes* ([`push_into`](GatherItem::push_into)) and then install them on the
//! gathered batch ([`finish`](GatherItem::finish)). A single id
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
//!     gather parent columns + origin per result, then install the result
//!     columns (if any)
//! ```

use std::collections::VecDeque;

use crate::graph::graph::{NodeId, RelationshipId};
use crate::runtime::batch::{BATCH_SIZE, Batch, Column};
use crate::runtime::value::Value;

/// A per-row result that knows how to accumulate a packed batch of itself into
/// one or more output columns. [`Binding`](GatherItem::Binding) carries the
/// per-operator metadata (which alias slots to bind, and any layout flags).
///
/// Results are accumulated directly into typed column *lanes* as they are pulled
/// ([`push_into`](Self::push_into)) rather than collected into a `Vec<Self>` and
/// transposed afterward — so a multi-column item (edge endpoints, scored id)
/// never materializes an intermediate `Vec` of tuples just to split it back into
/// per-column `Vec`s. [`finish`](Self::finish) installs the completed lanes on
/// the gathered batch.
pub(crate) trait GatherItem: Sized {
    /// Per-operator binding metadata consumed by [`new_lanes`](Self::new_lanes),
    /// [`push_into`](Self::push_into) and [`finish`](Self::finish).
    type Binding;

    /// Typed column accumulators this item scatters into: one growable `Vec` per
    /// output column.
    type Lanes;

    /// Create empty lanes pre-sized for up to `cap` results. Columns the binding
    /// leaves unbound (an absent score, a self-loop `to`) are left at zero
    /// capacity so they never allocate.
    fn new_lanes(
        binding: &Self::Binding,
        cap: usize,
    ) -> Self::Lanes;

    /// Push one result's components into their lanes, applying any per-item
    /// layout (e.g. the edge transpose swap).
    fn push_into(
        self,
        binding: &Self::Binding,
        lanes: &mut Self::Lanes,
    );

    /// Install the accumulated lanes as columns on `out`, which has already been
    /// gathered to one row per result.
    fn finish(
        lanes: Self::Lanes,
        binding: &Self::Binding,
        out: &mut Batch,
    );
}

impl GatherItem for NodeId {
    /// `Some(alias)` binds the node-id column; `None` binds no column.
    type Binding = Option<u32>;
    type Lanes = Vec<NodeId>;

    fn new_lanes(
        _binding: &Self::Binding,
        cap: usize,
    ) -> Self::Lanes {
        Vec::with_capacity(cap)
    }

    fn push_into(
        self,
        _binding: &Self::Binding,
        lanes: &mut Self::Lanes,
    ) {
        lanes.push(self);
    }

    fn finish(
        lanes: Self::Lanes,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        if let Some(alias) = binding {
            out.set_column(*alias, Column::NodeIds(lanes));
        }
    }
}

impl GatherItem for RelationshipId {
    /// `Some(alias)` binds the relationship-id column; `None` binds no column.
    type Binding = Option<u32>;
    type Lanes = Vec<RelationshipId>;

    fn new_lanes(
        _binding: &Self::Binding,
        cap: usize,
    ) -> Self::Lanes {
        Vec::with_capacity(cap)
    }

    fn push_into(
        self,
        _binding: &Self::Binding,
        lanes: &mut Self::Lanes,
    ) {
        lanes.push(self);
    }

    fn finish(
        lanes: Self::Lanes,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        if let Some(alias) = binding {
            out.set_column(*alias, Column::RelIds(lanes));
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
    type Lanes = (Vec<NodeId>, Vec<f64>);

    fn new_lanes(
        binding: &Self::Binding,
        cap: usize,
    ) -> Self::Lanes {
        // Only allocate the score lane when a score column is bound.
        let scores = if binding.score.is_some() {
            Vec::with_capacity(cap)
        } else {
            Vec::new()
        };
        (Vec::with_capacity(cap), scores)
    }

    fn push_into(
        self,
        binding: &Self::Binding,
        lanes: &mut Self::Lanes,
    ) {
        let (id, score) = self;
        lanes.0.push(id);
        if binding.score.is_some() {
            lanes.1.push(score);
        }
    }

    fn finish(
        lanes: Self::Lanes,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        let (ids, scores) = lanes;
        out.set_column(binding.id, Column::NodeIds(ids));
        if let Some(score_alias) = binding.score {
            out.set_column(score_alias, Column::Floats(scores));
        }
    }
}

impl GatherItem for (RelationshipId, f64) {
    type Binding = ScoredColumn;
    type Lanes = (Vec<RelationshipId>, Vec<f64>);

    fn new_lanes(
        binding: &Self::Binding,
        cap: usize,
    ) -> Self::Lanes {
        let scores = if binding.score.is_some() {
            Vec::with_capacity(cap)
        } else {
            Vec::new()
        };
        (Vec::with_capacity(cap), scores)
    }

    fn push_into(
        self,
        binding: &Self::Binding,
        lanes: &mut Self::Lanes,
    ) {
        let (id, score) = self;
        lanes.0.push(id);
        if binding.score.is_some() {
            lanes.1.push(score);
        }
    }

    fn finish(
        lanes: Self::Lanes,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        let (ids, scores) = lanes;
        out.set_column(binding.id, Column::RelIds(ids));
        if let Some(score_alias) = binding.score {
            out.set_column(score_alias, Column::Floats(scores));
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

/// Column lanes for an edge scan: the two endpoint node columns plus the edge
/// column. `tos` stays empty (and unallocated) for a self-loop pattern whose
/// endpoints share one alias.
pub(crate) struct EdgeLanes {
    froms: Vec<NodeId>,
    tos: Vec<NodeId>,
    edges: Vec<RelationshipId>,
}

impl GatherItem for (NodeId, NodeId, RelationshipId) {
    type Binding = EdgeEndpoints;
    type Lanes = EdgeLanes;

    fn new_lanes(
        binding: &Self::Binding,
        cap: usize,
    ) -> Self::Lanes {
        EdgeLanes {
            froms: Vec::with_capacity(cap),
            // A self-loop pattern binds one alias via `from`, so we never build
            // the `to` column for it — leave its lane unallocated.
            tos: if binding.to.is_some() {
                Vec::with_capacity(cap)
            } else {
                Vec::new()
            },
            edges: Vec::with_capacity(cap),
        }
    }

    fn push_into(
        self,
        binding: &Self::Binding,
        lanes: &mut Self::Lanes,
    ) {
        let (src, dst, edge) = self;
        let (from_node, to_node) = if binding.transposed {
            (dst, src)
        } else {
            (src, dst)
        };
        lanes.froms.push(from_node);
        if binding.to.is_some() {
            lanes.tos.push(to_node);
        }
        lanes.edges.push(edge);
    }

    fn finish(
        lanes: Self::Lanes,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        out.set_column(binding.from, Column::NodeIds(lanes.froms));
        if let Some(to_alias) = binding.to {
            out.set_column(to_alias, Column::NodeIds(lanes.tos));
        }
        out.set_column(binding.edge, Column::RelIds(lanes.edges));
    }
}

impl GatherItem for Value {
    /// Alias of the bound variable. `set_column` upgrades the value column to
    /// the best lossless stored shape (ints/floats), so this preserves the
    /// column specialization a row builder would have produced.
    type Binding = u32;
    type Lanes = Vec<Value>;

    fn new_lanes(
        _binding: &Self::Binding,
        cap: usize,
    ) -> Self::Lanes {
        Vec::with_capacity(cap)
    }

    fn push_into(
        self,
        _binding: &Self::Binding,
        lanes: &mut Self::Lanes,
    ) {
        lanes.push(self);
    }

    fn finish(
        lanes: Self::Lanes,
        binding: &Self::Binding,
        out: &mut Batch,
    ) {
        out.set_column(*binding, Column::Values(lanes));
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

    /// Install the parent batch and queue one result iterator per active row in
    /// a single pass: the active-index walk borrows our owned batch internally,
    /// so the caller needs no index Vec and no re-borrow via [`batch`]. `f`
    /// builds each row's iterator from a view into the batch (or `None` to skip
    /// the row). Replaces the set_batch-then-loop-push idiom for scans.
    pub(crate) fn seed<F>(
        &mut self,
        batch: Batch<'a>,
        mut f: F,
    ) -> Result<(), String>
    where
        F: FnMut(&Batch<'a>, usize) -> Result<Option<Box<dyn Iterator<Item = I> + 'a>>, String>,
    {
        self.batch = Some(batch);
        let b = self.batch.as_ref().expect("just set above");
        for row in b.active_indices() {
            if let Some(iter) = f(b, row)? {
                self.pending.push_back((row, RowIter::Many(iter)));
            }
        }
        Ok(())
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
    /// columns are accumulated straight into typed lanes as each result is
    /// pulled ([`GatherItem::push_into`]) and installed via
    /// [`GatherItem::finish`]. The per-row index vector is only needed when the
    /// parent carries columns — a bare column-less (leaf) parent skips it and
    /// emits a standalone result batch. Returns `None` when the queue drained
    /// without yielding a result, in which case the caller refills.
    pub(crate) fn emit(&mut self) -> Option<Batch<'a>> {
        // Gather (rather than build a standalone batch) whenever the parent has
        // columns to replicate; otherwise a no-alias emitter (which never sets a
        // column) would emit an empty batch. Correlation origins never need a
        // separate check: they are only ever stamped by
        // `clone_active_rows_seq_origin`, which clones the outer batch's
        // columns, so a parent carrying origins always has at least one column.
        let should_expand_batch = self.batch.as_ref().is_some_and(|b| b.num_columns() > 0);
        let mut indices: Vec<usize> = if should_expand_batch {
            Vec::with_capacity(BATCH_SIZE)
        } else {
            Vec::new()
        };
        // Accumulate results straight into typed column lanes instead of a
        // `Vec<I>`: a multi-column item never materializes an intermediate
        // tuple `Vec` just to transpose it back into columns at the end.
        let mut lanes = I::new_lanes(&self.binding, BATCH_SIZE);
        let mut count = 0usize;
        while count < BATCH_SIZE {
            let Some((row, iter)) = self.pending.front_mut() else {
                break;
            };
            let row = *row;
            // Keep taking from this row's iterator while the batch has room.
            // When it runs dry, drop the entry and move to the next queued row;
            // if the batch fills mid-iterator, the partially-drained entry stays
            // at the front for the next `emit`.
            let mut drained = false;
            while count < BATCH_SIZE && !drained {
                if let Some(item) = iter.next() {
                    if should_expand_batch {
                        indices.push(row);
                    }
                    item.push_into(&self.binding, &mut lanes);
                    count += 1;
                } else {
                    drained = true;
                }
            }
            if drained {
                self.pending.pop_front();
            }
        }
        if count == 0 {
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
        I::finish(lanes, &self.binding, &mut out);
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
