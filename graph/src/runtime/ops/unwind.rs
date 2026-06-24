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
//! Rows are expanded **one at a time**: only the current row's (possibly lazy)
//! iterator is queued, so `UNWIND range(1, 20000000)` keeps just one lazy range
//! in flight and a per-row list property is materialized for a single row at a
//! time. The emitter pulls values in `BATCH_SIZE` chunks, preventing memory
//! blow-up. Non-list values are treated as single-element results; NULL values
//! produce no output rows.

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::{ExprEval, ValueIter};
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
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
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> UnwindOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        list: &'a QueryExpr<Variable>,
        name: &'a Variable,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            list,
            emitter: BatchedResultEmitter::with_binding(name.id),
            active: Vec::new(),
            current_pos: 0,
            idx,
        }
    }
}

impl<'a> Iterator for UnwindOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Emit ready values from the current row's iterator first.
            if !self.emitter.needs_refill()
                && let Some(out) = self.emitter.emit()
            {
                return Some(Ok(out));
            }

            // The current row is drained: queue the next row's expansion,
            // pulling a fresh child batch when the current one is exhausted.
            // Only one row's iterator is ever queued, so a lazy range or a
            // per-row list property stays bounded to a single row.
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

                    match iter {
                        // `UNWIND null` / empty produces no rows. A NULL *inside*
                        // a list arrives via `List` and is emitted normally.
                        ValueIter::Empty | ValueIter::Once(None | Some(Value::Null)) => continue,
                        ValueIter::Once(Some(val)) => {
                            self.emitter.push_one(row_idx, val);
                            break;
                        }
                        other => {
                            self.emitter.push(row_idx, Box::new(other));
                            break;
                        }
                    }
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
