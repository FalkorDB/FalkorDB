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

use crate::parser::ast::{ExprIR, Variable};
use crate::planner::{IR, subtree_contains};
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp, Column},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef, Traversal, Traverser};

/// Active batched sub-plan for all rows from one input batch.
struct ActiveSubPlan<'a> {
    /// Compacted input batch (row `i` carries sequential origin `i`), kept so
    /// each sub-plan output batch can be merged back columnar via
    /// [`Batch::merge_over_input`] and unmatched origins can be gathered for the
    /// optional NULL fallback.
    input_ref: Batch<'a>,
    /// The single sub-plan iterator producing result batches for all input rows.
    subtree: BatchOp<'a>,
    /// Dense per-origin match flags (`matched[i]` == origin `i` produced a
    /// result), used for the optional NULL fallback. Empty for the non-optional
    /// path, which needs no fallback and so skips match tracking entirely.
    matched: Vec<bool>,
}

/// The currently-executing row's sub-plan (per-row mode).
struct PendingApply<'a> {
    /// The one input row this sub-plan runs for, kept to restore the columns
    /// the sub-plan does not bind and for the optional NULL fallback.
    env: Batch<'a>,
    subtree: BatchOp<'a>,
    had_result: bool,
}

/// Per-row mode state: the input batch being expanded one row at a time plus
/// the single live sub-plan. Sub-plans are created lazily as each row is
/// reached, so at most one instantiated operator tree is alive at a time
/// (instead of a queue holding one per input row).
struct PerRowState<'a> {
    /// Compacted input batch whose rows are processed sequentially.
    input: Batch<'a>,
    /// Position of the next row to start a sub-plan for.
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
    /// Batched mode's single buffered output batch (merged sub-plan results or
    /// the optional NULL fallback), emitted on the next `next()` without a
    /// row-by-row rebuild. The drive loop consumes it before producing another,
    /// so one slot suffices.
    pending_batch: Option<Batch<'a>>,
    /// Per-row mode state (used when can_batch is false).
    per_row: Option<Box<PerRowState<'a>>>,
    /// Per-row mode's merged output so far (one batch per sub-plan batch),
    /// concatenated into a single output batch once it holds a full batch.
    acc: Vec<Batch<'a>>,
    acc_rows: usize,
    /// Whether the sub-plan deletes, so a row whose imported entity an
    /// earlier row already deleted must be skipped (as C does).
    sub_plan_deletes: bool,
    /// A non-returning CALL {} (a keyless, empty Aggregate over the body):
    /// every input row passes through once, whatever the body yields.
    passthrough: bool,
    can_batch: bool,
    /// The `DISTINCT` aggregate arguments inside the sub-plan. Per-row mode
    /// resets their entries in [`Runtime::value_dedupers`] after each
    /// invocation; entries of enclosing aggregates must survive.
    sub_plan_distincts: Vec<NodeIdx<Dyn<ExprIR<Variable>>>>,
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

        // A sub-plan that writes is not a pure function of its argument row:
        // CALL {} runs once per outer row, so row N must see what rows
        // 0..N-1 wrote (and deleted). One batched instance would run every
        // row's reads before any row's writes (#2738).
        let can_batch = !subtree_contains(&runtime.plan, child_idx, |ir| {
            matches!(
                ir,
                IR::Create(_)
                    | IR::Set(_)
                    | IR::Remove(_)
                    | IR::Delete { .. }
                    | IR::ForEach { .. }
                    | IR::Commit
                    | IR::Aggregate { .. }
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

        let sub_plan_distincts = if can_batch {
            Vec::new()
        } else {
            sub_plan_distincts(runtime, child_idx)
        };

        Self {
            runtime,
            child,
            optional_vars,
            child_idx,
            active: None,
            pending_batch: None,
            per_row: None,
            acc: Vec::new(),
            acc_rows: 0,
            sub_plan_deletes: subtree_contains(&runtime.plan, child_idx, |ir| {
                matches!(ir, IR::Delete { .. })
            }),
            passthrough: matches!(
                runtime.plan.node(child_idx).data(),
                IR::Aggregate { keys, aggregations, .. } if keys.is_empty() && aggregations.is_empty()
            ),
            can_batch,
            sub_plan_distincts,
            idx,
        }
    }

    // -----------------------------------------------------------------------
    // Batched mode helpers
    // -----------------------------------------------------------------------

    fn next_batched(&mut self) -> Option<Result<Batch<'a>, String>> {
        loop {
            // 1. Emit the buffered columnar output batch first.
            if let Some(out) = self.pending_batch.take() {
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
                // tracking; the optional path uses a dense per-origin bitmap.
                let matched = if self.optional_vars.is_some() {
                    vec![false; input_ref.len()]
                } else {
                    Vec::new()
                };
                self.active = Some(Box::new(ActiveSubPlan {
                    input_ref,
                    subtree,
                    matched,
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
                        // Only the optional path needs match tracking.
                        if self.optional_vars.is_some() {
                            for &o in &origins {
                                plan.matched[o] = true;
                            }
                        }
                        let merged = sub.merge_over_input(&plan.input_ref, &origins);
                        self.pending_batch = Some(merged);
                    }
                }
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    // Sub-plan exhausted: emit an optional NULL fallback batch for
                    // the origins that never matched, then retire the sub-plan.
                    if let Some(ref vars) = self.optional_vars {
                        let unmatched: Vec<usize> = (0..plan.matched.len())
                            .filter(|&i| !plan.matched[i])
                            .collect();
                        if !unmatched.is_empty() {
                            let mut fb = plan.input_ref.gather(&unmatched);
                            for v in vars {
                                fb.set_column(
                                    v.id,
                                    Column::Values(vec![Value::Null; unmatched.len()]),
                                );
                            }
                            self.pending_batch = Some(fb);
                        }
                    }
                    self.active = None;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Per-row mode helpers (fallback for sub-plans with Aggregate)
    // -----------------------------------------------------------------------

    /// Whether any entity bound in `row` was deleted by an earlier invocation.
    /// C does not run the sub-query for such a row.
    fn row_has_deleted_entity(
        runtime: &Runtime<'_>,
        input: &Batch<'a>,
        row: usize,
    ) -> bool {
        let g = runtime.graph();
        let pending = runtime.pending.borrow();
        (0..input.num_columns() as u32).any(|id| {
            if !input.is_bound_at(id, row) {
                return false;
            }
            match input.value_at(id, row) {
                Some(Value::Node(n)) => g.is_node_deleted(n) || pending.is_node_deleted(n),
                Some(Value::Relationship(r)) => {
                    g.is_relationship_deleted(r) || pending.is_relationship_deleted(r)
                }
                _ => false,
            }
        })
    }

    fn flush_acc(&mut self) -> Batch<'a> {
        self.acc_rows = 0;
        if self.acc.len() == 1 {
            return self.acc.pop().unwrap();
        }
        let out = Batch::concat(&self.acc);
        self.acc.clear();
        out
    }

    fn push_acc(
        &mut self,
        batch: Batch<'a>,
    ) {
        self.acc_rows += batch.active_len();
        self.acc.push(batch);
    }

    fn next_per_row(&mut self) -> Option<Result<Batch<'a>, String>> {
        loop {
            if self.acc_rows >= BATCH_SIZE {
                return Some(Ok(self.flush_acc()));
            }

            if let Some(st) = self.per_row.as_mut() {
                // Drain the currently-executing row's sub-plan first.
                if let Some(p) = st.current.as_mut() {
                    match p.subtree.next() {
                        Some(Ok(sub)) => {
                            let n = sub.active_len();
                            if n > 0 {
                                p.had_result = true;
                                // Every sub-plan row correlates to the single
                                // input row, i.e. index 0.
                                let merged = sub.merge_over_input(&p.env, &vec![0usize; n]);
                                self.push_acc(merged);
                            }
                        }
                        Some(Err(e)) => return Some(Err(e)),
                        None => {
                            let done = st.current.take().unwrap();
                            if let Some(ref vars) = self.optional_vars
                                && !done.had_result
                            {
                                let mut fallback = done.env;
                                for v in vars {
                                    fallback.set_column(v.id, Column::Values(vec![Value::Null]));
                                }
                                self.push_acc(fallback);
                            }
                            // Reset the sub-plan's DISTINCT state so the next
                            // CALL {} invocation starts fresh. Only its own
                            // entries: the map is runtime-wide, and an
                            // enclosing `count(DISTINCT ..)` is still
                            // accumulating.
                            if !self.sub_plan_distincts.is_empty() {
                                self.runtime
                                    .value_dedupers
                                    .borrow_mut()
                                    .retain(|(idx, _), _| !self.sub_plan_distincts.contains(idx));
                            }
                        }
                    }
                    continue;
                }

                // Start the next row's sub-plan. Created lazily, one at a
                // time, so only a single instantiated sub-plan is ever alive.
                if st.pos < st.input.len() {
                    let row = st.pos;
                    st.pos += 1;
                    let env = st.input.gather(&[row]);
                    if self.sub_plan_deletes
                        && Self::row_has_deleted_entity(self.runtime, &st.input, row)
                    {
                        // No sub-plan ran, so there are no results; the
                        // optional fallback still applies, and a non-returning
                        // CALL {} still passes its input row through.
                        if self.passthrough {
                            self.push_acc(env);
                        } else if let Some(ref vars) = self.optional_vars {
                            let mut fallback = env;
                            for v in vars {
                                fallback.set_column(v.id, Column::Values(vec![Value::Null]));
                            }
                            self.push_acc(fallback);
                        }
                        continue;
                    }
                    let mut subtree = match self.runtime.run_batch(self.child_idx) {
                        Ok(iter) => iter,
                        Err(e) => return Some(Err(e)),
                    };
                    subtree.set_argument_batch(env.clone());
                    st.current = Some(PendingApply {
                        env,
                        subtree,
                        had_result: false,
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
                        input: b.into_compacted(),
                        pos: 0,
                        current: None,
                    }));
                }
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            }
        }

        if self.acc.is_empty() {
            None
        } else {
            Some(Ok(self.flush_acc()))
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

/// The `DISTINCT` nodes of every aggregate expression in the plan subtree
/// rooted at `root`: the keys, together with a group id, under which those
/// aggregates keep their state in [`Runtime::value_dedupers`].
fn sub_plan_distincts(
    runtime: &Runtime<'_>,
    root: NodeIdx<Dyn<IR>>,
) -> Vec<NodeIdx<Dyn<ExprIR<Variable>>>> {
    let mut out = Vec::new();
    for ir in runtime
        .plan
        .node(root)
        .walk_with(&mut Traversal.bfs().over_nodes())
    {
        if let IR::Aggregate { aggregations, .. } = ir.data() {
            for (_, tree) in aggregations {
                out.extend(
                    tree.root()
                        .walk_with(&mut Traversal.bfs().over_nodes())
                        .filter(|n| matches!(n.data(), ExprIR::Distinct))
                        .map(|n| n.idx()),
                );
            }
        }
    }
    out
}
