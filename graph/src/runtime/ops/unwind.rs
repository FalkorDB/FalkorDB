//! Batch-mode unwind operator — expands a list expression into individual rows.
//!
//! Implements Cypher `UNWIND expr AS var`. For each active row in each input
//! batch, evaluates the list expression and expands it into individual rows.
//! Output rows are accumulated into batches of up to `BATCH_SIZE`.
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
//! Large lists are expanded lazily: the operator uses `ValueIter` (which can
//! be a lazy range iterator) and only materializes `Env` rows in `BATCH_SIZE`
//! chunks, preventing memory blow-up for queries like
//! `UNWIND range(1, 20000000)`.
//! Non-list values are treated as single-element results; NULL values
//! produce no output rows.

use std::collections::VecDeque;

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::{ExprEval, ValueIter};
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    pool::Pool,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

/// State for lazily expanding a value iterator across multiple `next()` calls.
struct IterExpansion<'a> {
    /// The lazy iterator being expanded.
    iter: ValueIter,
    /// The base env for each output row (cloned per element).
    base_env: Env<'a>,
}

impl<'a> IterExpansion<'a> {
    /// Drain up to `budget` elements into `out`.
    /// Returns `true` if the expansion is fully drained.
    fn drain(
        &mut self,
        out: &mut VecDeque<Env<'a>>,
        budget: usize,
        name: &Variable,
        pool: &'a Pool<Value>,
    ) -> bool {
        for _ in 0..budget {
            match self.iter.next() {
                Some(val) => {
                    let mut row = self.base_env.clone_pooled(pool);
                    row.insert(name, val);
                    out.push_back(row);
                }
                None => return true,
            }
        }
        false
    }
}

/// Evaluate the list expression for a given row. Returns either:
/// - An `IterExpansion` if the result is a non-empty list or lazy range
/// - A single `Env` pushed onto `pending` for scalar values
/// - Nothing for `Null`
fn eval_row<'a>(
    runtime: &'a Runtime<'a>,
    list: &QueryExpr<Variable>,
    name: &Variable,
    env: &Env<'a>,
    pending: &mut VecDeque<Env<'a>>,
) -> Result<Option<IterExpansion<'a>>, String> {
    let pool = runtime.env_pool;
    let eval = ExprEval::from_runtime(runtime);
    let iter = eval.eval_iter_expr(list, list.root().idx(), Some(env))?;

    match iter {
        ValueIter::Empty | ValueIter::Once(None | Some(Value::Null)) => Ok(None),
        ValueIter::Once(Some(val)) => {
            let mut out_row = env.clone_pooled(pool);
            out_row.insert(name, val);
            pending.push_back(out_row);
            Ok(None)
        }
        _ => Ok(Some(IterExpansion {
            iter,
            base_env: env.clone_pooled(pool),
        })),
    }
}

pub struct UnwindOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    list: &'a QueryExpr<Variable>,
    name: &'a Variable,
    pending: VecDeque<Env<'a>>,
    current_batch: Option<Batch<'a>>,
    current_pos: usize,
    /// Lazy expansion state for a large list.
    iter_expansion: Option<IterExpansion<'a>>,
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
            name,
            pending: VecDeque::new(),
            current_batch: None,
            current_pos: 0,
            iter_expansion: None,
            idx,
        }
    }
}

impl<'a> Iterator for UnwindOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut envs = Vec::with_capacity(BATCH_SIZE);

        // Drain leftover rows from previous call.
        super::drain_pending(&mut self.pending, &mut envs);

        loop {
            if envs.len() >= BATCH_SIZE {
                break;
            }

            // Continue draining a partially-expanded iterator.
            if let Some(ref mut exp) = self.iter_expansion {
                let budget = BATCH_SIZE - envs.len();
                let done = exp.drain(&mut self.pending, budget, self.name, self.runtime.env_pool);
                if done {
                    self.iter_expansion = None;
                }
                super::drain_pending(&mut self.pending, &mut envs);
                if envs.len() >= BATCH_SIZE || self.iter_expansion.is_some() {
                    break;
                }
                continue;
            }

            if self.current_batch.is_none() {
                match self.child.next() {
                    Some(Ok(b)) => {
                        self.current_batch = Some(b);
                        self.current_pos = 0;
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => break,
                }
            }

            {
                let batch = self.current_batch.as_ref().unwrap();
                let active: Vec<usize> = batch.active_indices().collect();

                while self.current_pos < active.len() {
                    let row_idx = active[self.current_pos];
                    self.current_pos += 1;
                    let env = batch.env_ref(row_idx);
                    match eval_row(self.runtime, self.list, self.name, env, &mut self.pending) {
                        Ok(Some(expansion)) => {
                            self.iter_expansion = Some(expansion);
                            break; // drain the expansion in the next loop iteration
                        }
                        Ok(None) => {}
                        Err(e) => return Some(Err(e)),
                    }

                    if self.pending.len() >= BATCH_SIZE {
                        break;
                    }
                }
            }

            // Drain iterator expansion outside the batch borrow scope.
            if let Some(ref mut exp) = self.iter_expansion {
                let budget = BATCH_SIZE.saturating_sub(self.pending.len());
                let done = exp.drain(&mut self.pending, budget, self.name, self.runtime.env_pool);
                if done {
                    self.iter_expansion = None;
                }
            }

            super::drain_pending(&mut self.pending, &mut envs);

            // Check if batch is exhausted.
            if self.iter_expansion.is_none()
                && let Some(ref batch) = self.current_batch
                && self.current_pos >= batch.active_len()
            {
                self.current_batch = None;
            }
        }

        if envs.is_empty() {
            None
        } else {
            Some(Ok(Batch::from_envs(envs)))
        }
    }
}
