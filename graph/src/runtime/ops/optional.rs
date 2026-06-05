//! Batch-mode optional operator — implements OPTIONAL MATCH semantics.
//!
//! For each input batch, runs the sub-plan once with all active rows as a
//! multi-row argument batch. Uses `origin_row` on output envs to track which
//! input rows had results. For input rows with no results, emits a fallback
//! row with the specified variables set to NULL.
//!
//! ```text
//!  Input batch [row0, row1, row2]
//!       │
//!  stamp origin_row
//!       │
//!  ┌────▼─────────────┐
//!  │ Sub-plan          │  single instance, all rows at once
//!  └────┬─────────────┘
//!       │
//!  results with origin_row tags
//!       │
//!  ┌────▼──────────────────────────────────────────┐
//!  │ row0 matched? ──► yes: emit sub-plan results  │
//!  │ row1 matched? ──► no:  emit row1 + NULLs      │
//!  │ row2 matched? ──► yes: emit sub-plan results  │
//!  └───────────────────────────────────────────────┘
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
    /// Saved input envs for fallback (indexed by origin_row).
    input_envs: Vec<Row>,
    /// The single sub-plan iterator producing result batches for all input rows.
    subtree: BatchOp<'a>,
    /// Tracks which origin_rows have produced at least one result.
    matched_origins: HashSet<u32>,
    /// Partially consumed sub-batch and position within it.
    current_batch: Option<(Batch<'a>, usize)>,
    /// Index into `input_envs` for resuming fallback emission.
    fallback_index: Option<usize>,
}

/// Per-row sub-plan state (used when batching is not possible).
struct PendingOptional<'a> {
    env: Row,
    subtree: BatchOp<'a>,
    had_result: bool,
    current_batch: Option<(Batch<'a>, usize)>,
}

pub struct OptionalOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    vars: &'a [Variable],
    optional_child_idx: NodeIdx<Dyn<IR>>,
    /// Batched mode state.
    active: Option<Box<ActiveSubPlan<'a>>>,
    /// Per-row mode state.
    pending: VecDeque<PendingOptional<'a>>,
    can_batch: bool,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> OptionalOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        vars: &'a [Variable],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let optional_child_idx = if runtime.plan.node(idx).num_children() == 1 {
            runtime.plan.node(idx).child(0).idx()
        } else {
            runtime.plan.node(idx).child(1).idx()
        };

        let can_batch = !subtree_contains(&runtime.plan, optional_child_idx, |ir| {
            matches!(ir, IR::Aggregate { .. } | IR::CartesianProduct)
        });

        Self {
            runtime,
            child,
            vars,
            optional_child_idx,
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

            // Resume draining a partially consumed sub-batch.
            if let Some((ref batch, ref mut pos)) = plan.current_batch {
                let active: Vec<usize> = batch.active_indices().collect();
                while *pos < active.len() && builder.len() < BATCH_SIZE {
                    let env = BatchRow::new(batch, active[*pos]).to_owned_row();
                    plan.matched_origins.insert(env.origin_row);
                    builder.push_row(&env);
                    *pos += 1;
                }
                if *pos >= active.len() {
                    plan.current_batch = None;
                } else {
                    return Ok(());
                }
                continue;
            }

            // Resume emitting fallback rows for unmatched origins.
            if let Some(ref mut fb_idx) = plan.fallback_index {
                while *fb_idx < plan.input_envs.len() && builder.len() < BATCH_SIZE {
                    if !plan.matched_origins.contains(&(*fb_idx as u32)) {
                        let mut fallback = plan.input_envs[*fb_idx].clone();
                        for v in self.vars {
                            fallback.insert(v, Value::Null);
                        }
                        builder.push_row(&fallback);
                    }
                    *fb_idx += 1;
                }
                if *fb_idx >= plan.input_envs.len() {
                    self.active = None;
                    break;
                }
                return Ok(());
            }

            match plan.subtree.next() {
                Some(Ok(sub_batch)) => {
                    plan.current_batch = Some((sub_batch, 0));
                }
                Some(Err(e)) => return Err(e),
                None => {
                    plan.fallback_index = Some(0);
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

            let mut subtree = match self.runtime.run_batch(self.optional_child_idx) {
                Ok(s) => s,
                Err(e) => return Some(Err(e)),
            };
            subtree.set_argument_batch(arg_batch);

            self.active = Some(Box::new(ActiveSubPlan {
                input_envs,
                subtree,
                matched_origins: HashSet::new(),
                current_batch: None,
                fallback_index: None,
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
                    builder.push_row(&row);
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
                    if !p.had_result {
                        let mut fallback = p.env.clone();
                        for v in self.vars {
                            fallback.insert(v, Value::Null);
                        }
                        builder.push_row(&fallback);
                    }
                    self.pending.pop_front();
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
                let mut subtree = match self.runtime.run_batch(self.optional_child_idx) {
                    Ok(iter) => iter,
                    Err(e) => return Some(Err(e)),
                };
                let mut arg_builder = BatchBuilder::new();
                arg_builder.push_row(&env);
                subtree.set_argument_batch(arg_builder.finish());

                self.pending.push_back(PendingOptional {
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

impl<'a> Iterator for OptionalOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.can_batch {
            self.next_batched()
        } else {
            self.next_per_row()
        }
    }
}
