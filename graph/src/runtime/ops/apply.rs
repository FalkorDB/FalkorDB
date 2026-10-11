//! Batch-mode apply operator — correlated sub-query execution.
//!
//! For each input batch, instantiates the right sub-plan once and passes all
//! active rows as a multi-row argument batch. Uses `origin_row` on output envs
//! to correlate results back to input rows. Handles Optional fallback (NULL-fill)
//! when the right child is an Optional node.
//!
//! ```text
//!  Batched mode (default):
//!
//!     Left child ──► input batch [row0, row1, row2]
//!                          │
//!                     stamp origin_row
//!                          │
//!                     ┌────▼────┐
//!                     │ Right   │  single sub-plan instance
//!                     │ sub-plan│  processes all rows at once
//!                     └────┬────┘
//!                          │
//!                  correlate by origin_row
//!                          │
//!                  ┌───────┴───────┐
//!                  │ merge(input,  │
//!                  │       output) │
//!                  └───────────────┘
//!
//!  Per-row mode (fallback for blocking sub-plans like Aggregate):
//!
//!     Left child ──► for each row:
//!                       create fresh sub-plan instance
//!                       pass single-row argument batch
//!                       drain results and merge
//! ```
//!
//! Falls back to per-row sub-plan execution when the sub-plan contains blocking
//! operators (Aggregate) that accumulate state across all rows.

use crate::parser::ast::Variable;
use crate::planner::{IR, subtree_contains};
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    row::{Row, RowView},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use std::collections::VecDeque;

use super::optional_order::FallbackOrder;

/// Active batched sub-plan for all rows from one input batch.
struct ActiveSubPlan<'a> {
    /// Compacted input batch (row `i` carries sequential origin `i`), kept so
    /// each sub-plan output batch can be merged back columnar via
    /// [`Batch::merge_over_input`] and unmatched origins can be gathered for the
    /// optional NULL fallback.
    input_ref: Batch<'a>,
    /// The single sub-plan iterator producing result batches for all input rows.
    subtree: BatchOp<'a>,
    /// Which origins matched, and the results held back to keep input order,
    /// for the optional NULL fallback. `None` for the non-optional path, which
    /// needs no fallback and so skips match tracking entirely.
    order: Option<FallbackOrder<'a>>,
}

/// The currently-executing row's sub-plan (per-row mode).
struct PendingApply<'a> {
    env: Row,
    subtree: BatchOp<'a>,
    had_result: bool,
    current_batch: Option<(Batch<'a>, usize)>,
}

/// Per-row mode state: the input batch being expanded one row at a time plus
/// the single live sub-plan. Sub-plans are created lazily as each row is
/// reached, so at most one instantiated operator tree is alive at a time
/// (instead of a queue holding one per input row).
struct PerRowState<'a> {
    /// Input batch whose active rows are processed sequentially.
    input: Batch<'a>,
    /// Position into the input's active rows for the next sub-plan.
    pos: usize,
    /// The currently-executing row's sub-plan.
    current: Option<PendingApply<'a>>,
}

pub struct ApplyOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    optional_vars: Option<Vec<Variable>>,
    child_idx: NodeIdx<Dyn<IR>>,
    /// Batched mode state (used when can_batch is true).
    active: Option<Box<ActiveSubPlan<'a>>>,
    /// Batched mode's output batches (merged sub-plan results and optional NULL
    /// fallbacks, in input order), emitted on the next `next()` calls without a
    /// row-by-row rebuild.
    pending_batches: VecDeque<Batch<'a>>,
    /// Per-row mode state (used when can_batch is false).
    per_row: Option<Box<PerRowState<'a>>>,
    can_batch: bool,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> ApplyOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let right_child_idx = runtime.plan.node(idx).child(1).idx();
        let right_data = runtime.plan.node(right_child_idx).data().clone();

        let (optional_vars, child_idx) = match right_data {
            IR::Optional(ref vars) => {
                let optional_child_idx = runtime.plan.node(right_child_idx).child(0).idx();
                (Some(vars.clone()), optional_child_idx)
            }
            _ => (None, right_child_idx),
        };

        let can_batch = !subtree_contains(&runtime.plan, child_idx, |ir| {
            matches!(
                ir,
                IR::Aggregate { .. }
                    | IR::CartesianProduct
                    | IR::Optional(_)
                    | IR::Apply
                    | IR::Merge { .. }
                    | IR::Union
                    | IR::Sort(_)
                    | IR::Limit(_)
                    | IR::Skip(_)
                    | IR::Distinct
            )
        });

        Self {
            runtime,
            child,
            optional_vars,
            child_idx,
            active: None,
            pending_batches: VecDeque::new(),
            per_row: None,
            can_batch,
            idx,
        }
    }

    // -----------------------------------------------------------------------
    // Batched mode helpers
    // -----------------------------------------------------------------------

    fn next_batched(&mut self) -> Option<Result<Batch<'a>, String>> {
        loop {
            // 1. Emit the buffered columnar output batch first.
            if let Some(out) = self.pending_batches.pop_front() {
                return Some(Ok(out));
            }

            // 2. Ensure an active sub-plan is running for the current input batch.
            if self.active.is_none() {
                let batch = match self.child.next() {
                    Some(Ok(b)) => b,
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                };
                // One compacted copy feeds the sub-plan as its argument batch;
                // an identical copy is retained to merge results back against.
                let input_ref = batch.clone_active_rows_seq_origin();
                let arg_batch = batch.clone_active_rows_seq_origin();
                let mut subtree = match self.runtime.run_batch(self.child_idx) {
                    Ok(s) => s,
                    Err(e) => return Some(Err(e)),
                };
                subtree.set_argument_batch(arg_batch);
                // The non-optional path needs no fallback, so it skips match
                // tracking.
                let order = self
                    .optional_vars
                    .is_some()
                    .then(|| FallbackOrder::new(input_ref.len()));
                self.active = Some(Box::new(ActiveSubPlan {
                    input_ref,
                    subtree,
                    order,
                }));
            }

            // 3. Drive the sub-plan one batch at a time, merging each result
            //    batch back against the input columnar.
            let plan = self.active.as_mut().unwrap();
            match plan.subtree.next() {
                Some(Ok(sub)) => {
                    let origins: Vec<usize> = sub
                        .active_indices()
                        .map(|r| sub.origin_row(r) as usize)
                        .collect();
                    if !origins.is_empty() {
                        let merged = sub.merge_over_input(&plan.input_ref, &origins);
                        match plan.order.as_mut() {
                            Some(order) => order.push(merged, &origins, &mut self.pending_batches),
                            None => self.pending_batches.push_back(merged),
                        }
                    }
                }
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    // Sub-plan exhausted: for the optional path, emit what was held
                    // back and a NULL fallback row for each origin that never
                    // matched, in input order. Then retire the sub-plan.
                    if let (Some(order), Some(vars)) = (plan.order.as_mut(), &self.optional_vars) {
                        order.finish(&plan.input_ref, vars, &mut self.pending_batches);
                    }
                    self.active = None;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Per-row mode helpers (fallback for sub-plans with Aggregate)
    // -----------------------------------------------------------------------

    fn next_per_row(&mut self) -> Option<Result<Batch<'a>, String>> {
        let mut builder = BatchBuilder::new();

        while builder.len() < BATCH_SIZE {
            // Drain the currently-executing row's sub-plan first.
            if let Some(st) = self.per_row.as_mut()
                && let Some(p) = st.current.as_mut()
            {
                if let Some((batch, pos)) = &mut p.current_batch {
                    let active: Vec<usize> = batch.active_indices().collect();
                    while *pos < active.len() && builder.len() < BATCH_SIZE {
                        let row = BatchRow::new(batch, active[*pos]).to_owned_row();
                        p.had_result = true;
                        let mut merged = p.env.clone();
                        merged.merge(&row);
                        builder.push_row(&merged);
                        *pos += 1;
                    }
                    if *pos >= active.len() {
                        p.current_batch = None;
                    } else {
                        // Builder full with a partially-drained batch: resume
                        // from here on the next call.
                        break;
                    }
                }
                match p.subtree.next() {
                    Some(Ok(sub_batch)) => {
                        p.current_batch = Some((sub_batch, 0));
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => {
                        if let Some(ref vars) = self.optional_vars
                            && !p.had_result
                        {
                            let mut fallback = p.env.clone();
                            for v in vars {
                                fallback.insert(v, Value::Null);
                            }
                            builder.push_row(&fallback);
                        }
                        st.current = None;
                        // Clear DISTINCT deduplication state between per-row
                        // subquery invocations so that each CALL {} execution
                        // starts with a fresh deduper.
                        self.runtime.value_dedupers.borrow_mut().clear();
                    }
                }
                continue;
            }

            // Start the next row's sub-plan. Created lazily, one at a time, so
            // only a single instantiated sub-plan is ever alive.
            if let Some(st) = self.per_row.as_mut() {
                if st.pos < st.input.active_len() {
                    let row = match st.input.selection() {
                        Some(sel) => sel[st.pos] as usize,
                        None => st.pos,
                    };
                    st.pos += 1;
                    let env = BatchRow::new(&st.input, row).to_owned_row();
                    let mut subtree = match self.runtime.run_batch(self.child_idx) {
                        Ok(iter) => iter,
                        Err(e) => return Some(Err(e)),
                    };
                    let mut arg_builder = BatchBuilder::new();
                    arg_builder.push_row(&env);
                    subtree.set_argument_batch(arg_builder.finish());
                    st.current = Some(PendingApply {
                        env,
                        subtree,
                        had_result: false,
                        current_batch: None,
                    });
                    continue;
                }
                // Input batch exhausted.
                self.per_row = None;
            }

            // Pull the next input batch.
            match self.child.next() {
                Some(Ok(b)) => {
                    self.per_row = Some(Box::new(PerRowState {
                        input: b,
                        pos: 0,
                        current: None,
                    }));
                }
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            }
        }

        if builder.is_empty() {
            None
        } else {
            Some(Ok(builder.finish()))
        }
    }
}

impl<'a> Iterator for ApplyOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.can_batch {
            self.next_batched()
        } else {
            self.next_per_row()
        }
    }
}
