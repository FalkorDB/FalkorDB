//! Shared batched-result emit: packs per-row expansions into columnar batches.
//!
//! Several operators have the same shape: pull a parent batch from the child,
//! produce a per-active-row iterator of results, then pack up to [`BATCH_SIZE`]
//! `(parent_row, result)` pairs into one columnar output batch. The operators
//! differ only in *how* each row's iterator is produced (and any per-row
//! filtering, which they fold into the produced iterator); the pack-and-gather
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
//!  child BatchOp ──► parent batch ──► seed()
//!                         │
//!            for each active parent row (on demand, via the cursor):
//!              op-specific closure ──► RowIter (One / Spread / Many)
//!                         │
//!              ┌──────────┴───────────┐
//!              │ pack ≤ BATCH_SIZE    │  emit_lazy(closure)
//!              │ (parent_row, result) │
//!              └──────────┬───────────┘
//!                         │
//!     gather parent columns + origin per result, then install the result
//!     columns (if any)
//! ```

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

/// What a single parent row yields: returned by the lazy
/// [`emit_lazy`](BatchedResultEmitter::emit_lazy) closure and held in
/// [`pending`](BatchedResultEmitter::pending) while it is drained. Implements
/// [`Iterator`], so the emitter drives it for lazy cross-row packing and eager
/// consumers (e.g. the list-comprehension decomposition) can `for value in
/// result` over the same `next()`.
///
/// `One` is a single value (the scalar-`UNWIND`/single-node-per-row case),
/// carried in an `Option` so it can be `take`n as the iterator drains; `Spread`
/// is a small, arity-bounded run of values from a list *literal*
/// (`UNWIND [a, b, c]`) held inline in a smallvec; `Many` boxes an iterator for
/// an unbounded or lazy source (a property list, a `range(..)`, a label/index
/// scan). At most one is ever live, since the emitter holds a single row's
/// results at a time.
pub(crate) enum RowIter<'a, I> {
    One(Option<I>),
    Spread(smallvec::IntoIter<[I; 4]>),
    Many(Box<dyn Iterator<Item = I> + 'a>),
}

impl<'a, I> RowIter<'a, I> {
    /// A row that yields exactly one result, queued inline without allocation.
    pub(crate) fn one(item: I) -> Self {
        Self::One(Some(item))
    }

    /// A row that yields a small, arity-bounded run of results (a list literal)
    /// held inline in a smallvec — no per-row heap allocation.
    pub(crate) fn spread(iter: smallvec::IntoIter<[I; 4]>) -> Self {
        Self::Spread(iter)
    }

    /// A row that yields several results, drained from a boxed iterator.
    pub(crate) fn many(iter: Box<dyn Iterator<Item = I> + 'a>) -> Self {
        Self::Many(iter)
    }
}

impl<I> Iterator for RowIter<'_, I> {
    type Item = I;

    #[inline]
    fn next(&mut self) -> Option<I> {
        match self {
            Self::Many(iter) => iter.next(),
            Self::Spread(iter) => iter.next(),
            Self::One(item) => item.take(),
        }
    }
}

/// Owns the parent batch being expanded plus the current row's result iterator,
/// and performs the shared pack-and-gather emit.
pub(crate) struct BatchedResultEmitter<'a, I: GatherItem> {
    /// Per-operator binding describing how packed results scatter into columns.
    binding: I::Binding,
    /// Parent batch currently being expanded. Emitted rows are produced by
    /// `gather`ing this batch, which replicates every carried-forward parent
    /// column (and correlation origin) once per result.
    batch: Option<Batch<'a>>,
    /// The current parent row's result iterator, keyed by its row index within
    /// `batch`: `(parent_row, results)`. Any per-row filtering is folded into
    /// the iterator by the op that produced it.
    ///
    /// Filled lazily: [`emit_lazy`](Self::emit_lazy) holds at most one row's
    /// results at a time — the current row's partially-drained iterator — and
    /// rebuilds the next on demand using [`cursor`](Self::cursor).
    pending: Option<(usize, RowIter<'a, I>)>,
    /// Next active-row position to generate from (an index into the batch's
    /// active rows, not a raw row index).
    cursor: usize,
    /// Upper bound on results packed into one output batch. Defaults to
    /// [`BATCH_SIZE`]; an operator fed by a downstream `Skip`/`Limit` lowers it
    /// via [`set_pack_ceiling`](Self::set_pack_ceiling) so a capped query
    /// produces a small first batch instead of eagerly packing a whole
    /// `BATCH_SIZE` worth of work.
    pack_ceiling: usize,
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
            pending: None,
            cursor: 0,
            pack_ceiling: BATCH_SIZE,
        }
    }

    /// Lower the per-batch packing ceiling below [`BATCH_SIZE`]. Used by `UNWIND`
    /// when a downstream `Skip`/`Limit` bounds how many rows are needed, so the
    /// first [`emit_lazy`](Self::emit_lazy) returns just enough rows rather than
    /// eagerly packing a whole batch.
    pub(crate) fn set_pack_ceiling(
        &mut self,
        cap: usize,
    ) {
        debug_assert!(
            (1..=BATCH_SIZE).contains(&cap),
            "pack ceiling must be within [1, BATCH_SIZE]"
        );
        self.pack_ceiling = cap;
    }

    /// Install a parent batch for the lazy expand path: installs the batch but
    /// builds **no** per-row iterators up front. Resets the active-row
    /// [`cursor`](Self::cursor) so the next [`emit_lazy`](Self::emit_lazy) starts
    /// at the first active row. Call this whenever [`emit_lazy`](Self::emit_lazy)
    /// returns `None` (the previous batch is exhausted, or none was installed).
    pub(crate) fn seed(
        &mut self,
        batch: Batch<'a>,
    ) {
        debug_assert!(
            self.pending.is_none(),
            "seed requires an empty pending slot"
        );
        self.batch = Some(batch);
        self.cursor = 0;
    }

    /// Allocate the per-emit accumulators for [`emit_lazy`](Self::emit_lazy):
    /// the `should_expand` flag (true when the
    /// parent has columns to replicate), the parent-row index buffer (only sized
    /// when expanding), and the typed result lanes.
    ///
    /// We gather (rather than build a standalone batch) whenever the parent has
    /// columns to replicate; otherwise a no-alias emitter (which never sets a
    /// column) would emit an empty batch. Correlation origins never need a
    /// separate check: they are only ever stamped by
    /// `clone_active_rows_seq_origin`, which clones the outer batch's columns, so
    /// a parent carrying origins always has at least one column. The result lanes
    /// accumulate straight into typed columns instead of a `Vec<I>`, so a
    /// multi-column item never materializes an intermediate tuple `Vec` just to
    /// transpose it back into columns at the end.
    fn start_batch(&self) -> (bool, Vec<usize>, I::Lanes) {
        let should_expand = self.batch.as_ref().is_some_and(|b| b.num_columns() > 0);
        let indices = if should_expand {
            Vec::with_capacity(self.pack_ceiling)
        } else {
            Vec::new()
        };
        let lanes = I::new_lanes(&self.binding, self.pack_ceiling);
        (should_expand, indices, lanes)
    }

    /// Drain the pending entry's iterator into `lanes` until the batch is full or
    /// the entry is exhausted (then clear it). A partially-drained entry is kept
    /// for the next call. Precondition: `pending` is `Some`.
    fn drain_pending_entry(
        &mut self,
        indices: &mut Vec<usize>,
        lanes: &mut I::Lanes,
        count: &mut usize,
        should_expand: bool,
    ) {
        let ceiling = self.pack_ceiling;
        let (row, iter) = self.pending.as_mut().expect("pending is set");
        let row = *row;
        let mut drained = false;
        while *count < ceiling && !drained {
            if let Some(item) = iter.next() {
                if should_expand {
                    indices.push(row);
                }
                item.push_into(&self.binding, lanes);
                *count += 1;
            } else {
                drained = true;
            }
        }
        if drained {
            self.pending = None;
        }
    }

    /// Gather the accumulated `lanes`/`indices` into the output batch, or `None`
    /// when nothing was packed.
    fn finish_batch(
        &self,
        indices: Vec<usize>,
        lanes: I::Lanes,
        count: usize,
        should_expand: bool,
    ) -> Option<Batch<'a>> {
        if count == 0 {
            return None;
        }
        let batch = self.batch.as_ref().expect("batch is set while emitting");
        let mut out = if should_expand {
            batch.gather(&indices)
        } else {
            Batch::new(0)
        };
        I::finish(lanes, &self.binding, &mut out);
        Some(out)
    }

    /// Advance the [`cursor`](Self::cursor) to the next active row and queue its
    /// results (built on demand by `f` as a [`RowIter`]), skipping rows for
    /// which `f` yields nothing (including an empty `Spread`). Returns `false`
    /// when the batch's active rows are exhausted (or no batch is installed),
    /// signalling the caller to refill via [`seed`](Self::seed).
    fn refill_from_cursor<F>(
        &mut self,
        f: &mut F,
    ) -> Result<bool, String>
    where
        F: FnMut(&Batch<'a>, usize) -> Result<Option<RowIter<'a, I>>, String>,
    {
        let Some(b) = self.batch.as_ref() else {
            return Ok(false);
        };
        let active_len = b.active_len();
        while self.cursor < active_len {
            let row = match b.selection() {
                Some(sel) => sel[self.cursor] as usize,
                None => self.cursor,
            };
            self.cursor += 1;
            match f(b, row)? {
                None => {}
                Some(RowIter::Spread(mut values)) => match values.len() {
                    // An empty list literal (`UNWIND []`) yields nothing for this
                    // row; keep scanning rather than queueing an empty iterator.
                    0 => {}
                    // A single-element literal (`UNWIND [x]`) takes the lean `One`
                    // path so it never pays the wider `Spread` payload's drain cost.
                    1 => {
                        self.pending = Some((row, RowIter::One(values.next())));
                        return Ok(true);
                    }
                    _ => {
                        self.pending = Some((row, RowIter::Spread(values)));
                        return Ok(true);
                    }
                },
                Some(result) => {
                    self.pending = Some((row, result));
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Packs up to the pack ceiling ([`BATCH_SIZE`] by default) into one columnar
    /// batch, walking the parent batch's active rows on demand via the
    /// [`cursor`](Self::cursor), building each row's results with `f` only when
    /// reached. `f` returns the row's [`RowIter`] (a single inline value, a
    /// small spread of inline values, or a boxed iterator), or `None` to skip the
    /// row. At most one row's results are ever live (held in `pending`), so a
    /// wide parent batch never materializes one iterator per active row. Returns
    /// `Ok(None)` once the active rows are exhausted (or no batch is installed
    /// yet), at which point the caller pulls the next child batch and installs it
    /// via [`seed`](Self::seed).
    pub(crate) fn emit_lazy<F>(
        &mut self,
        mut f: F,
    ) -> Result<Option<Batch<'a>>, String>
    where
        F: FnMut(&Batch<'a>, usize) -> Result<Option<RowIter<'a, I>>, String>,
    {
        let (should_expand, mut indices, mut lanes) = self.start_batch();
        let mut count = 0usize;
        while count < self.pack_ceiling {
            if self.pending.is_none() && !self.refill_from_cursor(&mut f)? {
                break;
            }
            self.drain_pending_entry(&mut indices, &mut lanes, &mut count, should_expand);
        }
        Ok(self.finish_batch(indices, lanes, count, should_expand))
    }

    /// Drop all queued state. Used by correlated (Apply) plans that re-seed the
    /// scan with a new argument batch and must not replay stale rows.
    pub(crate) fn reset(&mut self) {
        self.pending = None;
        self.batch = None;
        self.cursor = 0;
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
