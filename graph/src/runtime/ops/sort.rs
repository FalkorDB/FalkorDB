//! Batch-mode sort operator — orders result rows by one or more expressions.
//!
//! This is a *blocking* operator: it consumes all batches from the child on
//! the first `next()` call, evaluates sort-key expressions for each
//! row, sorts in-memory using a stable sort with per-key ascending/descending
//! control, and then yields rows in sorted batches.
//!
//! ```text
//!  Child batches (all consumed on first call)
//!       │
//!       ▼
//!  ┌──────────────────────────────────┐
//!  │ Evaluate sort keys per row       │
//!  │ [(env, [(val, desc), ...])]      │
//!  └──────────────┬───────────────────┘
//!                 │
//!       stable sort (multi-key)
//!                 │
//!       ┌────────▼────────┐
//!       │ reversed Vec    │  stored reversed so pop() is O(1)
//!       │ yield BATCH_SIZE│
//!       │ at a time       │
//!       └─────────────────┘
//! ```
//!
//! When primary sort keys are equal, a deterministic tiebreaker compares
//! env values slot-by-slot.

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::{CompareValue, Value},
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use std::cmp::Ordering;

type SortItem<'a> = (Env<'a>, Vec<(Value, bool)>);

pub struct SortOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Option<Box<BatchOp<'a>>>,
    trees: &'a [(QueryExpr<Variable>, bool)],
    /// Sorted results stored in reverse order so we can pop from the end in O(1).
    results: Vec<Env<'a>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// When set, only the top `limit` rows (after skip) are needed.
    /// Allows truncation after sorting to avoid excess work.
    limit: Option<usize>,
    /// Number of rows to skip before the limit applies.
    skip: usize,
}

impl<'a> SortOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        trees: &'a [(QueryExpr<Variable>, bool)],
        idx: NodeIdx<Dyn<IR>>,
        limit: Option<usize>,
        skip: usize,
    ) -> Self {
        Self {
            runtime,
            child: Some(child),
            trees,
            results: Vec::new(),
            idx,
            limit,
            skip,
        }
    }
}

impl<'a> Iterator for SortOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Consume all input on first call.
        if let Some(child) = self.child.take() {
            let mut items: Vec<SortItem<'a>> = Vec::new();
            for batch_result in child {
                let batch = match batch_result {
                    Ok(b) => b,
                    Err(e) => return Some(Err(e)),
                };
                for env in batch.active_env_iter() {
                    let sort_keys = match self
                        .trees
                        .iter()
                        .map(|(tree, desc)| {
                            Ok((
                                ExprEval::from_runtime(self.runtime).eval(
                                    tree,
                                    tree.root().idx(),
                                    Some(env),
                                    None,
                                )?,
                                *desc,
                            ))
                        })
                        .collect::<Result<Vec<_>, String>>()
                    {
                        Ok(keys) => keys,
                        Err(e) => return Some(Err(e)),
                    };
                    items.push((env.clone_pooled(self.runtime.env_pool), sort_keys));
                }
            }

            items.sort_by(|(env_a, a), (env_b, b)| {
                let primary =
                    a.iter()
                        .zip(b)
                        .fold(Ordering::Equal, |acc, ((a, desc_a), (b, _))| {
                            if acc != Ordering::Equal {
                                return acc;
                            }
                            let (ordering, _) = a.compare_value(b);
                            if *desc_a {
                                ordering.reverse()
                            } else {
                                ordering
                            }
                        });
                if primary != Ordering::Equal {
                    return primary;
                }
                // Tiebreaker: compare env values slot-by-slot for deterministic
                // ordering when primary sort keys are equal.
                let len = env_a.len().min(env_b.len());
                for i in 0..len {
                    if let (Some(va), Some(vb)) =
                        (env_a.get_by_id(i as u32), env_b.get_by_id(i as u32))
                    {
                        let (ord, _) = va.compare_value(vb);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                    }
                }
                env_a.len().cmp(&env_b.len())
            });

            // Reverse so we can pop from the end in O(1) while preserving
            // sorted order, and drop sort keys immediately (no longer needed).
            let mut sorted: Vec<Env<'a>> = items.into_iter().map(|(env, _keys)| env).collect();

            // When limit is known, truncate to limit + skip to avoid keeping
            // excess rows. The Skip and Limit operators above handle the actual
            // row-level trimming.
            if let Some(limit) = self.limit {
                let keep = limit + self.skip;
                sorted.truncate(keep);
            }

            sorted.reverse();
            self.results = sorted;
        }

        // Emit sorted rows in batches by popping from the end.
        if self.results.is_empty() {
            return None;
        }

        let n = BATCH_SIZE.min(self.results.len());
        let mut envs = self.results.split_off(self.results.len() - n);
        envs.reverse();

        Some(Ok(Batch::from_envs(envs)))
    }
}
