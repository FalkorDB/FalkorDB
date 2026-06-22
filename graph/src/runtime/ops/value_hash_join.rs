//! Batch-mode value hash join operator — equi-join via build/probe hash table.
//!
//! Replaces CartesianProduct + equality Filter with a hash join when the
//! optimizer detects an equality predicate between a left and right expression.
//!
//! ```text
//!  Phase 1: BUILD (materialize right sub-plan into hash table)
//!
//!     Right child ──► for each row: hash(rhs_expr) ──► HashMap<hash, Vec<(key, envs)>>
//!
//!  Phase 2: PROBE (stream left rows, look up matches)
//!
//!     Left child ──► for each row: hash(lhs_expr) ──► probe table
//!                                                        │
//!                          ┌──────────────────────────────┘
//!                          │  for each matching right env:
//!                          │    merged = left_env + right_env
//!                          ▼
//!                     output batches
//! ```
//!
//! The hash table uses chaining for collision resolution: each bucket stores
//! a `Vec<(Value, BuildSlot)>` where exact key equality is checked during
//! probe. NULL keys are skipped on both sides (Cypher NULL != NULL semantics).

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    eval::ExprEval,
    row::{Row, RowView},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

/// A lightweight reference to one retained right-side row: which batch in
/// `right_batches`, and which row within it. The build phase stores these
/// instead of owned `Row`s, so only rows that actually match are materialised
/// (on demand) during probe.
#[derive(Clone, Copy)]
pub(crate) struct RightRowRef {
    batch: u32,
    row: u32,
}

/// One hash bucket's worth of build-side rows that share the same key value,
/// kept as positions so probe can re-check exact key equality after a hash hit.
pub(crate) type BuildSlot = Vec<RightRowRef>;

/// The build/probe hash table: `hash(key) -> [(key, matching right rows)]`.
/// The per-bucket `Vec` resolves hash collisions; exact key equality is
/// re-checked on probe (see `BuildSlot`).
pub(crate) type JoinHashTable = HashMap<u64, Vec<(Value, BuildSlot)>>;

/// Hash a join key with the same hasher used on both the build and probe sides.
fn hash_value(value: &Value) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

pub struct ValueHashJoinOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    pub(crate) right: Box<BatchOp<'a>>,
    pub(crate) lhs_exp: &'a QueryExpr<Variable>,
    pub(crate) rhs_exp: &'a QueryExpr<Variable>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// Build/probe hash table; `None` until the right side has been consumed.
    pub(crate) hash_table: Option<JoinHashTable>,
    /// The right sub-plan's batches, retained so probe can gather matched rows
    /// by `RightRowRef` position without the build phase materialising an owned
    /// `Row` per right row.
    pub(crate) right_batches: Vec<Batch<'a>>,
    /// Current block of left-side rows being probed (columnar).
    pub(crate) left_batch: Option<Batch<'a>>,
    /// Current position within `left_batch`.
    pub(crate) left_pos: usize,
    /// Current position within the matched right envs for the current left row.
    pub(crate) right_match_envs: Vec<Row>,
    pub(crate) right_match_pos: usize,
}

impl<'a> ValueHashJoinOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        right: Box<BatchOp<'a>>,
        lhs_exp: &'a QueryExpr<Variable>,
        rhs_exp: &'a QueryExpr<Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            right,
            lhs_exp,
            rhs_exp,
            idx,
            hash_table: None,
            right_batches: Vec::new(),
            left_batch: None,
            left_pos: 0,
            right_match_envs: Vec::new(),
            right_match_pos: 0,
        }
    }

    /// Build the probe hash table from the right sub-plan. Each retained batch
    /// keeps its rows in place; the table stores `RightRowRef` positions into
    /// `right_batches` rather than owned `Row`s, so the build side allocates
    /// nothing per row and only matched rows are materialised during probe.
    fn build_hash_table(&mut self) -> Result<JoinHashTable, String> {
        let eval = ExprEval::from_runtime(self.runtime);
        let rhs_idx = self.rhs_exp.root().idx();
        let mut table = JoinHashTable::new();

        for result in self.right.by_ref() {
            let batch = result?;
            // The index this batch will occupy in `right_batches` once pushed
            // below; refs point at its rows by that position.
            let batch_ref = self.right_batches.len() as u32;
            for row in batch.active_indices() {
                let view = BatchRow::new(&batch, row);
                let key = eval.eval(self.rhs_exp, rhs_idx, Some(&view), None)?;
                if matches!(key, Value::Null) {
                    continue; // NULL never joins (Cypher NULL != NULL).
                }
                let slot = RightRowRef {
                    batch: batch_ref,
                    row: row as u32,
                };
                // Group refs under their key within the bucket, re-using the
                // existing key entry on a hash collision or repeated key.
                let bucket = table.entry(hash_value(&key)).or_default();
                match bucket.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, refs)) => refs.push(slot),
                    None => bucket.push((key, vec![slot])),
                }
            }
            self.right_batches.push(batch);
        }

        Ok(table)
    }

    /// Populate `right_match_envs` with the build-side rows whose key equals
    /// `key`, preserving build insertion order, and reset `right_match_pos`.
    /// Matched rows are gathered from `right_batches` by position (only matched
    /// rows are materialised).
    fn fill_matches(
        &mut self,
        key: &Value,
    ) {
        self.right_match_envs.clear();
        self.right_match_pos = 0;
        let mut matches: BuildSlot = Vec::new();
        if let Some(bucket) = self.hash_table.as_ref().unwrap().get(&hash_value(key)) {
            for (k, refs) in bucket {
                if k == key {
                    matches.extend_from_slice(refs);
                }
            }
        }
        for slot in matches {
            let env = BatchRow::new(&self.right_batches[slot.batch as usize], slot.row as usize)
                .to_owned_row();
            self.right_match_envs.push(env);
        }
    }
}

impl<'a> Iterator for ValueHashJoinOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Lazy materialization of right side
        if self.hash_table.is_none() {
            match self.build_hash_table() {
                Ok(table) => {
                    if table.is_empty() {
                        return None;
                    }
                    self.hash_table = Some(table);
                }
                Err(e) => return Some(Err(e)),
            }
        }

        let mut builder = BatchBuilder::new();

        loop {
            let left_len = self.left_batch.as_ref().map_or(0, Batch::len);

            // Drain remaining matches from current left row
            while builder.len() < BATCH_SIZE && self.right_match_pos < self.right_match_envs.len() {
                let mut merged =
                    BatchRow::new(self.left_batch.as_ref().unwrap(), self.left_pos).to_owned_row();
                merged.merge(&self.right_match_envs[self.right_match_pos]);
                builder.push_row(&merged);
                self.right_match_pos += 1;
            }

            if self.right_match_pos >= self.right_match_envs.len() {
                // Move to next left row
                if self.right_match_pos > 0 || self.left_pos > 0 {
                    self.left_pos += 1;
                }
                self.right_match_envs.clear();
                self.right_match_pos = 0;
            }

            if builder.len() >= BATCH_SIZE {
                return Some(Ok(builder.finish()));
            }

            // Process more left rows
            while self.left_pos < left_len {
                // Inline probe to avoid borrow conflict with self.right_match_envs
                let eval = ExprEval::from_runtime(self.runtime);
                let lhs_idx = self.lhs_exp.root().idx();
                let key = {
                    let left_row = BatchRow::new(self.left_batch.as_ref().unwrap(), self.left_pos);
                    match eval.eval(self.lhs_exp, lhs_idx, Some(&left_row), None) {
                        Ok(k) => k,
                        Err(e) => return Some(Err(e)),
                    }
                };
                if matches!(key, Value::Null) {
                    self.left_pos += 1;
                    continue;
                }
                self.fill_matches(&key);
                if self.right_match_envs.is_empty() {
                    self.left_pos += 1;
                    continue;
                }
                // Now drain from right_match
                while builder.len() < BATCH_SIZE
                    && self.right_match_pos < self.right_match_envs.len()
                {
                    let mut merged =
                        BatchRow::new(self.left_batch.as_ref().unwrap(), self.left_pos)
                            .to_owned_row();
                    merged.merge(&self.right_match_envs[self.right_match_pos]);
                    builder.push_row(&merged);
                    self.right_match_pos += 1;
                }
                if self.right_match_pos >= self.right_match_envs.len() {
                    self.left_pos += 1;
                    self.right_match_envs.clear();
                    self.right_match_pos = 0;
                }
                if builder.len() >= BATCH_SIZE {
                    return Some(Ok(builder.finish()));
                }
            }

            // Need more left rows
            self.left_batch = None;
            self.left_pos = 0;

            match self.child.next() {
                Some(Ok(batch)) => {
                    self.left_batch = Some(batch.into_compacted());
                }
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    if builder.is_empty() {
                        return None;
                    }
                    return Some(Ok(builder.finish()));
                }
            }
        }
    }
}
