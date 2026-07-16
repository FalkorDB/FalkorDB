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
//! Each active parent row's list expression is evaluated on demand into a
//! [`RowIter`](super::batched_result_emitter::RowIter) (via
//! [`eval_iter_expr`](crate::runtime::eval::ExprEval::eval_iter_expr)), and the
//! shared [`BatchedResultEmitter`] drives the expansion
//! **lazily** — building one row's iterator at a time via
//! [`emit_lazy`](BatchedResultEmitter::emit_lazy) — while packing up to
//! [`BATCH_SIZE`](super::super::batch::BATCH_SIZE) values *across* rows into one
//! dense columnar batch via `gather`, rather than emitting one tiny batch per
//! input row. A downstream `LIMIT` lowers that packing ceiling (via
//! `record_cap` → [`set_pack_ceiling`](BatchedResultEmitter::set_pack_ceiling))
//! so `UNWIND ... RETURN x LIMIT k` still produces a small first batch.
//!
//! Holding only the current row's iterator means even an unbounded or large
//! source — a lazy `range(1, 20000000)` or a list materialized from a
//! property/parameter — streams across batches without ever piling up. A scalar
//! `UNWIND null` (and an empty list) produces no output rows; a NULL *inside* a
//! list is emitted as a row.

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx};

use super::batched_result_emitter::BatchedResultEmitter;

pub struct UnwindOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    list: &'a QueryExpr<Variable>,
    /// Holds the parent batch being expanded and performs the shared
    /// pack-and-gather emit, building each active row's value iterator on
    /// demand. A downstream `Skip`/`Limit` lowers its pack ceiling.
    emitter: BatchedResultEmitter<'a, Value>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> UnwindOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        list: &'a QueryExpr<Variable>,
        name: &'a Variable,
        record_cap: Option<usize>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        // Translate the downstream row budget into a packing ceiling. With no
        // limit (or one at/over a full batch) we pack a whole `BATCH_SIZE`; a
        // tighter limit caps each batch so the first `emit_lazy` returns just
        // enough rows (clamped to at least 1, since `LIMIT 0` still runs the op).
        let mut emitter = BatchedResultEmitter::with_binding(name.id);
        emitter.apply_record_cap(record_cap);
        Self {
            runtime,
            child,
            list,
            emitter,
            idx,
        }
    }
}

impl<'a> Iterator for UnwindOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let runtime = self.runtime;
        let list = self.list;
        loop {
            // Each active parent row's list expression is evaluated on demand
            // into a [`RowIter`](super::batched_result_emitter::RowIter)
            // (see `ExprEval::eval_iter_expr`): a scalar
            // becomes one inline row, a small list literal a stack-held spread,
            // and a `range(..)`/property list a boxed lazy iterator — so only
            // the current row's iterator is ever held. `UNWIND null` / an empty
            // list yields no rows for that row (`Ok(None)`); a NULL *inside* a
            // list is emitted. The emitter packs up to the pack ceiling values
            // across rows into one gathered batch. When the batch is exhausted
            // (`Ok(None)` from `emit_lazy`), pull and seed the next child batch.
            match self.emitter.emit_lazy(|batch, row| {
                let view = BatchRow::new(batch, row);
                ExprEval::from_runtime(runtime).eval_iter_expr(&list.root(), Some(&view))
            }) {
                Ok(Some(out)) => return Some(Ok(out)),
                Ok(None) => match self.child.next() {
                    Some(Ok(batch)) => self.emitter.seed(batch),
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                },
                Err(e) => return Some(Err(e)),
            }
        }
    }
}
