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

use std::collections::{HashSet, VecDeque};

use crate::parser::ast::Variable;
use crate::planner::{IR, subtree_contains};
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    row::{Row, RowView},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

/// Active batched sub-plan for all rows from one input batch.
struct ActiveSubPlan<'a> {
    /// Saved input envs for merging with sub-plan output (indexed by origin_row).
    input_envs: Vec<Row>,
    /// The single sub-plan iterator producing result batches for all input rows.
    subtree: BatchOp<'a>,
    /// Tracks which origin_rows have produced at least one result (for optional fallback).
    matched_origins: HashSet<u32>,
    /// Partially consumed sub-batch and current index into its active rows.
    current_batch: Option<(Batch<'a>, usize)>,
    /// When `Some`, the subtree is exhausted; value is the next index into
    /// `input_envs` for optional fallback emission.
    fallback_idx: Option<usize>,
}

/// Per-row sub-plan state (used when batching is not possible).
struct PendingApply<'a> {
    env: Row,
    subtree: BatchOp<'a>,
    had_result: bool,
    current_batch: Option<(Batch<'a>, usize)>,
}

pub struct ApplyOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    optional_vars: Option<Vec<Variable>>,
    child_idx: NodeIdx<Dyn<IR>>,
    /// Batched mode state (used when can_batch is true).
    active: Option<Box<ActiveSubPlan<'a>>>,
    /// Per-row mode state (used when can_batch is false).
    pending: VecDeque<PendingApply<'a>>,
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
            pending: VecDeque::new(),
            can_batch,
            idx,
        }
    }

    // -----------------------------------------------------------------------
    // Batched mode helpers
    // -----------------------------------------------------------------------

    fn drain_active(
        &mut self,
        builder: &mut BatchBuilder,
    ) -> Result<(), String> {
        while builder.len() < BATCH_SIZE {
            let Some(ref mut plan) = self.active else {
                break;
            };

            // Drain partially consumed sub-batch first.
            if let Some((ref batch, ref mut pos)) = plan.current_batch {
                let active: Vec<usize> = batch.active_indices().collect();
                while *pos < active.len() && builder.len() < BATCH_SIZE {
                    let env = BatchRow::new(batch, active[*pos]).to_owned_row();
                    let origin = env.origin_row as usize;
                    plan.matched_origins.insert(env.origin_row);
                    let mut merged = plan.input_envs[origin].clone();
                    merged.merge(&env);
                    builder.push_row(&merged);
                    *pos += 1;
                }
                if *pos >= active.len() {
                    plan.current_batch = None;
                } else {
                    return Ok(());
                }
            }

            // Emit optional fallbacks for unmatched origins.
            if let Some(ref mut fb_idx) = plan.fallback_idx {
                if let Some(ref vars) = self.optional_vars {
                    while *fb_idx < plan.input_envs.len() {
                        let i = *fb_idx;
                        *fb_idx += 1;
                        if !plan.matched_origins.contains(&(i as u32)) {
                            let mut fallback = plan.input_envs[i].clone();
                            for v in vars {
                                fallback.insert(v, Value::Null);
                            }
                            builder.push_row(&fallback);
                            if builder.len() >= BATCH_SIZE {
                                return Ok(());
                            }
                        }
                    }
                }
                self.active = None;
                break;
            }

            // Pull next sub-batch from the subtree.
            match plan.subtree.next() {
                Some(Ok(sub_batch)) => {
                    plan.current_batch = Some((sub_batch, 0));
                }
                Some(Err(e)) => return Err(e),
                None => {
                    if self.optional_vars.is_some() {
                        plan.fallback_idx = Some(0);
                    } else {
                        self.active = None;
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn next_batched(&mut self) -> Option<Result<Batch<'a>, String>> {
        let mut builder = BatchBuilder::new();

        if let Err(e) = self.drain_active(&mut builder) {
            return Some(Err(e));
        }

        while builder.len() < BATCH_SIZE {
            if self.active.is_some() {
                if let Err(e) = self.drain_active(&mut builder) {
                    return Some(Err(e));
                }
                continue;
            }

            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            };

            let input_envs: Vec<Row> = batch
                .active_indices()
                .enumerate()
                .map(|(i, row)| {
                    let mut e = BatchRow::new(&batch, row).to_owned_row();
                    e.origin_row = i as u32;
                    e
                })
                .collect();

            let arg_batch = batch.clone_active_rows_seq_origin();

            let mut subtree = match self.runtime.run_batch(self.child_idx) {
                Ok(s) => s,
                Err(e) => return Some(Err(e)),
            };
            subtree.set_argument_batch(arg_batch);

            self.active = Some(Box::new(ActiveSubPlan {
                input_envs,
                subtree,
                matched_origins: HashSet::new(),
                current_batch: None,
                fallback_idx: None,
            }));

            if let Err(e) = self.drain_active(&mut builder) {
                return Some(Err(e));
            }
        }

        if builder.is_empty() {
            None
        } else {
            Some(Ok(builder.finish()))
        }
    }

    // -----------------------------------------------------------------------
    // Per-row mode helpers (fallback for sub-plans with Aggregate)
    // -----------------------------------------------------------------------

    fn drain_pending(
        &mut self,
        builder: &mut BatchBuilder,
    ) -> Result<(), String> {
        while builder.len() < BATCH_SIZE {
            let Some(p) = self.pending.front_mut() else {
                break;
            };

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
                    return Ok(());
                }
            }

            match p.subtree.next() {
                Some(Ok(sub_batch)) => {
                    p.current_batch = Some((sub_batch, 0));
                }
                Some(Err(e)) => return Err(e),
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
                    self.pending.pop_front();
                    // Clear DISTINCT deduplication state between per-row
                    // subquery invocations so that each CALL {} execution
                    // starts with a fresh deduper.
                    self.runtime.value_dedupers.borrow_mut().clear();
                }
            }
        }
        Ok(())
    }

    fn next_per_row(&mut self) -> Option<Result<Batch<'a>, String>> {
        let mut builder = BatchBuilder::new();

        if let Err(e) = self.drain_pending(&mut builder) {
            return Some(Err(e));
        }

        while builder.len() < BATCH_SIZE {
            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            };

            for row in batch.active_indices() {
                let env = BatchRow::new(&batch, row).to_owned_row();
                let mut subtree = match self.runtime.run_batch(self.child_idx) {
                    Ok(iter) => iter,
                    Err(e) => return Some(Err(e)),
                };
                let mut arg_builder = BatchBuilder::new();
                arg_builder.push_row(&env);
                subtree.set_argument_batch(arg_builder.finish());

                self.pending.push_back(PendingApply {
                    env,
                    subtree,
                    had_result: false,
                    current_batch: None,
                });
            }

            if let Err(e) = self.drain_pending(&mut builder) {
                return Some(Err(e));
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
