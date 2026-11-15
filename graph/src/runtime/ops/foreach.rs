//! Batch-mode FOREACH operator — iterates over a list and executes a body sub-plan
//! for each element as a side effect.
//!
//! ```text
//!  Input row
//!       │
//!  eval list expression ──► [item1, item2, item3]
//!       │
//!  ┌────▼──────────────────────────────┐
//!  │ Build batch of loop envs:         │
//!  │   row + var=item1                 │
//!  │   row + var=item2                 │
//!  │   row + var=item3                 │
//!  │                                   │
//!  │ Execute body sub-plan (eager)     │
//!  │   side effects: Create/Set/Delete │
//!  │   results discarded               │
//!  └────┬──────────────────────────────┘
//!       │
//!  pass through original input row unchanged
//! ```
//!
//! For each active row in each input batch, evaluates the list expression and
//! collects all loop items, then executes the body sub-plan once with all items
//! as a batch (eager execution). This matches the C implementation where MERGE
//! inside FOREACH sees a consistent graph state for all iterations.
//! The original input row is passed through unchanged -- FOREACH is purely
//! a side-effect clause.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    row::{Row, RowView},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct ForEachOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    list: &'a QueryExpr<Variable>,
    var: &'a Variable,
    body_idx: NodeIdx<Dyn<IR>>,
    pending: VecDeque<Row>,
    current_batch: Option<Batch<'a>>,
    current_pos: usize,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> ForEachOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        list: &'a QueryExpr<Variable>,
        var: &'a Variable,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        // Body sub-plan is always the last child of the ForEach node.
        // If there are 2 children, child(0) = input, child(1) = body.
        // If there is 1 child, child(0) = body.
        let node = runtime.plan.node(idx);
        let body_idx = node.child(node.num_children() - 1).idx();
        Self {
            runtime,
            child,
            list,
            var,
            body_idx,
            pending: VecDeque::new(),
            current_batch: None,
            current_pos: 0,
            idx,
        }
    }

    /// Execute the body sub-plan eagerly for all loop items at once.
    /// Builds a multi-row batch of loop envs and passes them to the body
    /// as a single Argument batch. This ensures MERGE inside the body sees
    /// a consistent state for all iterations (matching C's eager behavior).
    fn execute_list(
        &self,
        items: Vec<Value>,
        env: &Row,
    ) -> Result<(), String> {
        if items.is_empty() {
            return Ok(());
        }

        let mut loop_envs: Vec<Row> = items
            .into_iter()
            .map(|item| {
                let mut loop_env = env.clone();
                loop_env.insert(self.var, item);
                loop_env
            })
            .collect();

        while !loop_envs.is_empty() {
            let mut builder = BatchBuilder::new();
            for e in loop_envs.drain(..BATCH_SIZE.min(loop_envs.len())) {
                builder.push_row(&e);
            }
            let batch = builder.finish();
            let mut body = self.runtime.run_batch(self.body_idx)?;
            body.set_argument_batch(batch);
            // Drain the body sub-plan completely — we only care about side effects.
            for result in body {
                result?;
            }
        }
        Ok(())
    }
}

impl<'a> Iterator for ForEachOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut builder = BatchBuilder::new();

        // Drain leftover rows from previous call.
        super::drain_pending(&mut self.pending, &mut builder);

        loop {
            if builder.len() >= BATCH_SIZE {
                break;
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
                    let env = BatchRow::new(batch, row_idx).to_owned_row();

                    // Evaluate the list expression for this row.
                    let list_value = match ExprEval::from_runtime(self.runtime).eval(
                        self.list,
                        self.list.root().idx(),
                        Some(&env),
                        None,
                    ) {
                        Ok(v) => v,
                        Err(e) => return Some(Err(e)),
                    };

                    match list_value {
                        Value::List(list) => {
                            let items: Vec<Value> = Arc::unwrap_or_clone(list).into();
                            if let Err(e) = self.execute_list(items, &env) {
                                return Some(Err(e));
                            }
                        }
                        Value::Null => {
                            // Null list produces no iterations — skip.
                        }
                        _ => {
                            return Some(Err(format!(
                                "Type mismatch: expected List but was {}",
                                list_value.name()
                            )));
                        }
                    }

                    // Pass through the original input row unchanged.
                    self.pending.push_back(env.clone());

                    if self.pending.len() >= BATCH_SIZE {
                        break;
                    }
                }
            }

            super::drain_pending(&mut self.pending, &mut builder);

            // Check if batch is exhausted.
            if let Some(ref batch) = self.current_batch
                && self.current_pos >= batch.active_len()
            {
                self.current_batch = None;
            }
        }

        if builder.is_empty() {
            None
        } else {
            Some(Ok(builder.finish()))
        }
    }
}
