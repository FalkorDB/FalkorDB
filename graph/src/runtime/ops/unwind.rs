//! Batch-mode unwind operator — expands a list expression into individual rows.
//!
//! Implements Cypher `UNWIND expr AS var`. For each active parent row, evaluates
//! the list expression and queues its values on the shared
//! [`BatchedResultEmitter`], which packs up to
//! [`BATCH_SIZE`](super::super::batch::BATCH_SIZE) values into one columnar
//! batch — replicating the parent columns once per value via `gather` rather
//! than cloning the parent row per element.
//!
//! ```text
//!  Input row {a: 1}
//!       │
//!  eval list expr ──► [10, 20, 30]
//!       │
//!  ┌────▼───────────┐
//!  │ {a:1, x:10}    │
//!  │ {a:1, x:20}    │
//!  │ {a:1, x:30}    │
//!  └────────────────┘
//! ```
//!
//! Each parent row's results are queued on the emitter as one of two slot
//! kinds: an inline `One` (the value lives in the queue's existing buffer — no
//! allocation) or a boxed `Many` iterator (one heap allocation). Single values
//! (`UNWIND n.id`) and list *literals* (`UNWIND [n.id, n.id+1]`, which `eval`
//! fuses into already-evaluated inline values rather than an `Arc<Value::List>`)
//! are bounded and fully materialized, so they drain into `One` slots and are
//! **packed across rows**: the loop keeps pulling input rows until a
//! [`BATCH_SIZE`](super::super::batch::BATCH_SIZE)-worth of values is queued,
//! then `gather`/`scatter` folds them into one dense columnar batch — collapsing
//! one tiny batch per input row into far fewer dense ones. A downstream `LIMIT`
//! lowers that packing ceiling (via `record_cap`) so `UNWIND ... RETURN x
//! LIMIT k` still produces a small first batch.
//!
//! An unbounded or large source — a lazy `range(1, 20000000)` or a list
//! materialized from a property/parameter — is instead boxed (`Many`) and
//! expanded **one row at a time**, so a single huge expansion can never pile up
//! in the queue. NULL values (and `UNWIND null`) produce no output rows.

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::{ExprEval, ValueIter};
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp, BatchRow},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

use super::batched_result_emitter::BatchedResultEmitter;

pub struct UnwindOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    list: &'a QueryExpr<Variable>,
    /// Holds the parent batch being expanded and the current row's value
    /// iterator, and performs the shared pack-and-gather emit.
    emitter: BatchedResultEmitter<'a, Value>,
    /// Active row indices of the batch currently held by the emitter.
    active: Vec<usize>,
    /// Next index into `active` whose list expression hasn't been expanded yet.
    current_pos: usize,
    /// Cross-row packing ceiling derived from a downstream `Skip`/`Limit`
    /// (`None` when unbounded). Bounds how many values are queued before each
    /// emit so `UNWIND ... RETURN x LIMIT k` produces a small first batch
    /// instead of eagerly packing a full `BATCH_SIZE` worth of work.
    pack_cap: usize,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> UnwindOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        list: &'a QueryExpr<Variable>,
        name: &'a Variable,
        record_cap: Option<usize>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        // Translate the downstream row budget into a packing ceiling. With no
        // limit (or one at/over a full batch) we pack a whole `BATCH_SIZE`; a
        // tighter limit caps the queue so the first `emit` returns just enough
        // rows (clamped to at least 1, since `LIMIT 0` still has to run the op).
        let pack_cap = match record_cap {
            Some(0) => 1,
            Some(cap) if cap < BATCH_SIZE => cap,
            _ => BATCH_SIZE,
        };
        Self {
            runtime,
            child,
            list,
            emitter: BatchedResultEmitter::with_binding(name.id),
            active: Vec::new(),
            current_pos: 0,
            pack_cap,
            idx,
        }
    }
}

impl<'a> Iterator for UnwindOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Emit a packed batch from whatever results are already queued
            // before refilling, so we hand one batch upstream per `next()` call.
            if !self.emitter.needs_refill()
                && let Some(out) = self.emitter.emit()
            {
                return Some(Ok(out));
            }

            // The pending queue is drained: refill it. Walk the active rows of
            // the current parent batch, queue each row's expansion (see the
            // per-shape dispatch below), and pull the next child batch once this
            // one is exhausted — then loop back up to `emit`.
            loop {
                if self.current_pos < self.active.len() {
                    let row_idx = self.active[self.current_pos];
                    self.current_pos += 1;

                    let iter = {
                        let batch = self
                            .emitter
                            .batch()
                            .expect("batch is set while active rows remain");
                        let view = BatchRow::new(batch, row_idx);
                        match ExprEval::from_runtime(self.runtime).eval_iter_expr(
                            self.list,
                            self.list.root().idx(),
                            Some(&view),
                        ) {
                            Ok(it) => it,
                            Err(e) => return Some(Err(e)),
                        }
                    };

                    // Queue this row's results by shape. The choice of slot
                    // (`One` vs boxed `Many`) decides both whether we allocate
                    // and whether these results can pack with their neighbours:
                    // `push_one` stores a value inline in the queue's buffer (no
                    // allocation) and adds one entry per value, while `push`
                    // boxes an iterator (one allocation) and adds a single entry
                    // that `emit` drains lazily. We therefore unpack bounded,
                    // already-evaluated results into `One` slots and only box
                    // unbounded/large ones.
                    match iter {
                        // `UNWIND null` / empty produces no rows. A NULL *inside*
                        // a list arrives via `Inline`/`List` and is emitted.
                        ValueIter::Empty | ValueIter::Once(None | Some(Value::Null)) => continue,
                        // Single value (e.g. `UNWIND n.id`): one allocation-free
                        // `One` slot that packs with its neighbours below.
                        ValueIter::Once(Some(val)) => self.emitter.push_one(row_idx, val),
                        ValueIter::Inline(vals) => {
                            // Fused list literal: the elements are already
                            // evaluated and usually bounded by the literal's
                            // arity, so drain them straight into allocation-free
                            // `One` slots that pack with their neighbours. Boxing
                            // would add a heap allocation per row *and* make
                            // `pending_len` count rows rather than values,
                            // blunting the value-precise packing threshold below.
                            //
                            // The exception is a literal larger than the
                            // remaining pack budget (e.g. a giant `UNWIND [..]`):
                            // pushing every element up front would blow past
                            // `pack_cap` before the check below runs. In that case
                            // box the iterator so `emit` streams it in
                            // `BATCH_SIZE` chunks, keeping the queue bounded.
                            let remaining =
                                self.pack_cap.saturating_sub(self.emitter.pending_len());
                            if vals.len() > remaining {
                                self.emitter
                                    .push(row_idx, Box::new(ValueIter::Inline(vals)));
                                break;
                            }
                            for val in vals {
                                self.emitter.push_one(row_idx, val);
                            }
                        }
                        other => {
                            // Lazy `range(..)` or a list materialized from a
                            // property/parameter: box the iterator and expand
                            // this row alone, so a large (or unbounded) expansion
                            // streams through `emit` instead of piling up in the
                            // queue. `break` stops refilling so it drains first.
                            self.emitter.push(row_idx, Box::new(other));
                            break;
                        }
                    }

                    // `pending_len` counts queued entries, not the values inside
                    // a boxed `Many` iterator. It equals the staged value count
                    // at this check because every arm that reaches here used
                    // `push_one` (one queue entry per value); the arms that box
                    // an iterator (`push`) all `break` first, so they never make
                    // this count under-report. Keep pulling rows and packing
                    // their values until we have a batch's worth (or the
                    // downstream `LIMIT`, whichever is smaller), so the next
                    // `emit` gathers many input rows into one dense columnar
                    // batch rather than one tiny batch per input row. The cap
                    // also bounds the queue so a huge literal can't balloon it.
                    if self.emitter.pending_len() >= self.pack_cap {
                        break;
                    }
                    continue;
                }

                // The current batch is fully queued: emit what we have, otherwise
                // pull the next child batch (only safe once pending has drained,
                // since the emitter holds a single parent batch at a time).
                if !self.emitter.needs_refill() {
                    break;
                }
                match self.child.next() {
                    Some(Ok(batch)) => {
                        self.active = batch.active_indices().collect();
                        self.current_pos = 0;
                        self.emitter.set_batch(batch);
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                }
            }
        }
    }
}
