//! Batch-mode value hash join operator — equi-join via build/probe hash table.
//!
//! Replaces CartesianProduct + equality Filter with a hash join when the
//! optimizer detects an equality predicate between a left and right expression.
//!
//! ```text
//!  Phase 1: BUILD (consume right sub-plan into the probe table)
//!
//!     Right child ──► eval_batch(rhs_expr) per batch ──► one key per row
//!                          │   all-integer keys ─► Int   table:  i64       ─► [RightRowRef]
//!                          │   otherwise        ─► Value table:  hash(key) ─► [(key, [RightRowRef])]
//!                          ▼
//!                     (right rows are NOT copied — only RightRowRef positions are kept)
//!
//!  Phase 2: PROBE (stream left rows, look up matches)
//!
//!     Left child ──► eval(lhs_expr) per row ──► probe the table
//!                          │  for each matching RightRowRef:
//!                          │    materialize right row, merged = left_row + right_row
//!                          ▼
//!                     output batches
//! ```
//!
//! The table has two representations (see [`JoinHashTable`]): an `i64`-keyed
//! fast path when every build key is integer-valued, and a general `Value`
//! table (hash-to-bucket, then exact `Value` equality) for everything else.
//! NULL keys are skipped on both sides (Cypher NULL != NULL semantics).

use std::collections::HashMap;

use ahash::RandomState;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use smallvec::{SmallVec, smallvec};

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow, Column},
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
///
/// Inline capacity 1, because the overwhelmingly common case is a key that
/// appears once. A `Vec` here is a heap allocation per *distinct key* to hold a
/// single 8-byte position, and joins on near-unique keys are exactly the shape
/// that produces one per row. Measured on a 10,000-row build side: 1,646,215
/// bytes allocated when every key was distinct, against 12,423 when the same
/// rows carried only five distinct keys. Duplicate keys spill to the heap as
/// before.
pub(crate) type BuildSlot = SmallVec<[RightRowRef; 1]>;

/// General-path build/probe table: `seeded_hash(Value) -> [(key, right rows)]`.
/// The per-bucket `Vec` resolves hash collisions; exact `Value` equality is
/// re-checked on probe (see [`BuildSlot`]).
pub(crate) type ValueTable = FxHashMap<u64, Vec<(Value, BuildSlot)>>;

/// The build/probe hash table, in one of two representations.
pub(crate) enum JoinHashTable {
    /// Fast path used when every build-side key is integer-valued. Keyed
    /// directly on `i64` with the seeded hasher (hashbrown resolves
    /// collisions), so the build side stores no boxed `Value` per key, hashes
    /// 8 bytes instead of the `Value` enum, and compares keys with a plain
    /// `i64` equality instead of `Value::compare_value` — none of which leave a
    /// `Value` to drop. Whole-valued probe floats (Cypher treats `30.0 = 30`)
    /// are converted to `i64` to probe; see [`key_as_i64`].
    Int(HashMap<i64, BuildSlot, RandomState>),
    /// General path: any non-integer key (string, non-whole float, …) on the
    /// build side promotes the table here, where keys keep their `Value` form.
    Value(ValueTable),
}

impl JoinHashTable {
    fn is_empty(&self) -> bool {
        match self {
            Self::Int(table) => table.is_empty(),
            Self::Value(table) => table.is_empty(),
        }
    }
}

/// Process-global random seed for join-key hashing.
///
/// A *randomized* seed (chosen once per process) makes the hash output
/// unpredictable, so a client who controls join-key values can't precompute
/// many distinct keys that all collide into a single bucket — the classic
/// hash-flooding / algorithmic-complexity DoS that would turn an O(n) join into
/// O(n²) chain scans. `aHash` stays close to `FxHash` speed (it uses AES
/// intrinsics where available) while resisting that attack. The same seed is
/// shared by the build and probe sides, so equal keys still hash identically.
static JOIN_HASH_SEED: Lazy<RandomState> = Lazy::new(RandomState::new);

/// Hash a join key with the shared, seeded hasher (see [`JOIN_HASH_SEED`]).
/// Buckets re-check exact `Value` equality, so the hash only needs to spread
/// keys; the random seed is what keeps that spread non-adversarial.
fn hash_value(value: &Value) -> u64 {
    JOIN_HASH_SEED.hash_one(value)
}

/// Map a key to the `i64` the [`JoinHashTable::Int`] fast path is keyed on,
/// honouring `Value`'s numeric equality (`Int(n)` and the whole float `n.0`
/// compare equal and hash identically, so they must share a key). Integers map
/// directly; a float maps only if it round-trips through `i64` exactly (a whole
/// number, in range — matching the `Value` `Hash`/`compare_value` rules);
/// anything else (string, non-whole float, NaN, …) can't equal an integer key,
/// so it returns `None`.
fn key_as_i64(key: &Value) -> Option<i64> {
    match key {
        Value::Int(n) => Some(*n),
        Value::Float(f) => {
            let n = *f as i64;
            (n as f64 == *f).then_some(n)
        }
        _ => None,
    }
}

/// Insert one build row into the general (`Value`) table, grouping rows that
/// share a key under a single bucket entry (re-using it on a hash collision or
/// a repeated key).
fn insert_value(
    table: &mut ValueTable,
    key: Value,
    slot: RightRowRef,
) {
    let bucket = table.entry(hash_value(&key)).or_default();
    match bucket.iter_mut().find(|(k, _)| *k == key) {
        Some((_, refs)) => refs.push(slot),
        None => bucket.push((key, smallvec![slot])),
    }
}

/// Move every entry of the integer fast-path table into a fresh general table
/// (as `Value::Int` keys) so the build can continue once a non-integer key
/// forces the general representation. Equality is preserved: `Value::Int(n)`
/// matches exactly the same probe keys the raw `i64 n` did (`Int(n)` and the
/// whole float `n.0` hash identically and compare equal).
fn promote_int_table(int_table: &mut HashMap<i64, BuildSlot, RandomState>) -> ValueTable {
    let mut table = ValueTable::default();
    for (n, refs) in int_table.drain() {
        let key = Value::Int(n);
        table.entry(hash_value(&key)).or_default().push((key, refs));
    }
    table
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
        // Build on the integer fast path; the first non-integer-valued key
        // promotes the entries gathered so far into the general `Value` table,
        // and the rest of the build continues there.
        let mut int_table: HashMap<i64, BuildSlot, RandomState> =
            HashMap::with_hasher((*JOIN_HASH_SEED).clone());
        let mut value_table: Option<ValueTable> = None;

        for result in self.right.by_ref() {
            let batch = result?;
            // The index this batch will occupy in `right_batches` once pushed
            // below; refs point at its rows by that position.
            let batch_ref = self.right_batches.len() as u32;

            // Bulk-evaluate the build key for the whole batch in one shot. For
            // the common `b.attr` shape this is a single columnar attribute
            // fetch yielding a primitive `Column::Ints`, which feeds the integer
            // fast-path table straight from the `i64` slice with no per-row
            // `Value` created or dropped; other shapes fall back to per-row eval
            // inside `eval_batch`. The column is lossless, so routing it below is
            // equivalent to evaluating each key individually.
            let active: Vec<usize> = batch.active_indices().collect();
            let (column, nulls) = eval.eval_batch(self.rhs_exp, &batch, &active)?;

            for (i, &row) in active.iter().enumerate() {
                if nulls.is_null(i) {
                    continue; // NULL never joins (Cypher NULL != NULL).
                }
                let slot = RightRowRef {
                    batch: batch_ref,
                    row: row as u32,
                };
                match &mut value_table {
                    // General path already active: re-materialize the key value.
                    Some(table) => insert_value(table, column.get(i), slot),
                    // All-integer column: key directly on the `i64` — the
                    // build side's hot path (no `Value` box / hash / drop).
                    None => {
                        if let Column::Ints(ints) = &column {
                            int_table.entry(ints[i]).or_default().push(slot);
                        } else {
                            // Heterogeneous keys: honour `Value` numeric equality
                            // (`30.0 == 30`) via `key_as_i64`, otherwise promote the
                            // accumulated integer entries to the general table.
                            let key = column.get(i);
                            if let Some(n) = key_as_i64(&key) {
                                int_table.entry(n).or_default().push(slot);
                            } else {
                                let mut table = promote_int_table(&mut int_table);
                                insert_value(&mut table, key, slot);
                                value_table = Some(table);
                            }
                        }
                    }
                }
            }
            self.right_batches.push(batch);
        }

        Ok(value_table.map_or_else(|| JoinHashTable::Int(int_table), JoinHashTable::Value))
    }

    /// Populate `right_match_envs` with the build-side rows whose key equals
    /// `key`, preserving build insertion order, and reset `right_match_pos`.
    /// Build groups all rows for a given key under a single entry, so at most
    /// one entry matches; its `RightRowRef`s are materialised straight into
    /// `right_match_envs` from `right_batches` with no intermediate ref copy.
    fn fill_matches(
        &mut self,
        key: &Value,
    ) {
        self.right_match_envs.clear();
        self.right_match_pos = 0;
        let refs = match self.hash_table.as_ref().unwrap() {
            // Integer fast path: only integer-valued probe keys can match an
            // all-integer build side; everything else short-circuits.
            JoinHashTable::Int(table) => {
                let Some(n) = key_as_i64(key) else {
                    return;
                };
                table.get(&n)
            }
            // General path: hash to the bucket, then re-check exact equality.
            JoinHashTable::Value(table) => {
                let Some(bucket) = table.get(&hash_value(key)) else {
                    return;
                };
                bucket.iter().find(|(k, _)| k == key).map(|(_, refs)| refs)
            }
        };
        let Some(refs) = refs else {
            return;
        };
        for slot in refs {
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

            // Only advance when this block actually drained a non-empty match
            // set for the current left row. If `right_match_envs` is already
            // empty, a prior `next()` call finished this left row exactly at a
            // BATCH_SIZE boundary (advancing `left_pos` and clearing the
            // matches before returning); advancing again here would skip the
            // following left row.
            if !self.right_match_envs.is_empty()
                && self.right_match_pos >= self.right_match_envs.len()
            {
                self.left_pos += 1;
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
