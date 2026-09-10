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
//!
//! Keys carrying a NULL cannot join, because Cypher `=` is three-valued and a
//! NULL operand makes the predicate UNKNOWN rather than TRUE. A key like
//! `[1, null]` is not itself NULL, yet `[1, null] = [1, null]` is still NULL,
//! so [`keys_match`] rejects any comparison the `Value` layer flagged as
//! inconclusive, and the build side additionally drops such keys outright via
//! [`contains_null`] so they never reach the table.

use std::cmp::Ordering;
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
    value::{CompareValue, DisjointOrNull, Value},
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

/// True when `value` is NULL or hides a NULL anywhere inside a list, path or
/// map.
///
/// Such a key can never join. Cypher `=` is three-valued, and
/// `Value::compare_value` propagates that faithfully: comparing `[1, null]`
/// with `[1, null]` yields `(Ordering::Equal, DisjointOrNull::ComparedNull)` —
/// "equal as far as the non-NULL elements go, but the answer is UNKNOWN". A
/// NULL nested at any depth bubbles the flag all the way out through
/// `compare_list`/`compare_map`, so the result is never a genuine match.
///
/// This runs on the *build* side only, once per key as it is materialised:
/// such a key would otherwise occupy a bucket that nothing can ever hit, and
/// every probe landing on that bucket would pay to compare against it. The
/// probe side deliberately does not repeat the deep scan — [`keys_match`]
/// already rejects these keys for free, so scanning there would only tax the
/// far more common null-free container key. The integer fast path skips the
/// test entirely, since an `i64` cannot hide a NULL.
fn contains_null(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::List(items) | Value::Path(items) => items.iter().any(contains_null),
        Value::Map(entries) => entries.values().any(contains_null),
        _ => false,
    }
}

/// Cypher `=` between two join keys: a match only when they compare equal *and*
/// the comparison was conclusive.
///
/// `Value`'s `PartialEq` is `compare_value(..).0 == Ordering::Equal`, which
/// throws the [`DisjointOrNull`] flag away and so collapses UNKNOWN into TRUE:
/// `[1, null] == [1, null]` reports `true` even though Cypher says the
/// predicate is NULL. Keeping the flag is what makes the join agree with the
/// `Filter` it replaces. `DisjointOrNull::Disjoint` is rejected for the same
/// reason — `compare_map` reports `(Equal, Disjoint)` for maps whose values
/// have incomparable types, which is FALSE, not TRUE.
///
/// This costs nothing over `PartialEq`: the flag is already computed and
/// returned by the same `compare_value` call, so the inner bucket scan just
/// stops discarding it.
fn keys_match(
    a: &Value,
    b: &Value,
) -> bool {
    let (ordering, disjoint_or_null) = a.compare_value(b);
    ordering == Ordering::Equal && matches!(disjoint_or_null, DisjointOrNull::None)
}

/// Map a key to the `i64` the [`JoinHashTable::Int`] fast path is keyed on,
/// honouring `Value`'s numeric equality (`Int(n)` and the whole float `n.0`
/// compare equal and hash identically, so they must share a key). Integers map
/// directly; a float maps only if it round-trips through `i64` exactly (a whole
/// number, in range — matching the `Value` `Hash`/`compare_value` rules);
/// anything else (string, non-whole float, NaN, …) can't equal an integer key,
/// so it returns `None`. Only scalar numerics are accepted, so a key that
/// [`contains_null`] rejects can never reach the integer table.
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
/// a repeated key). Grouping uses the same [`keys_match`] test as probe, so two
/// keys share an entry exactly when a probe key matching one must also match
/// the other.
fn insert_value(
    table: &mut ValueTable,
    key: Value,
    slot: RightRowRef,
) {
    let bucket = table.entry(hash_value(&key)).or_default();
    match bucket.iter_mut().find(|(k, _)| keys_match(k, &key)) {
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
                    Some(table) => {
                        let key = column.get(i);
                        if contains_null(&key) {
                            continue;
                        }
                        insert_value(table, key, slot);
                    }
                    // All-integer column: key directly on the `i64` — the
                    // build side's hot path (no `Value` box / hash / drop, and
                    // no null scan, since an `i64` cannot hide a NULL).
                    None => {
                        if let Column::Ints(ints) = &column {
                            int_table.entry(ints[i]).or_default().push(slot);
                        } else {
                            // Heterogeneous keys: honour `Value` numeric equality
                            // (`30.0 == 30`) via `key_as_i64`, otherwise promote the
                            // accumulated integer entries to the general table.
                            let key = column.get(i);
                            // A NULL nested inside a container key (`[1, null]`)
                            // is not caught by `nulls` above, and can never
                            // compare equal under three-valued logic — drop it
                            // here rather than seed a bucket nothing can hit.
                            if contains_null(&key) {
                                continue;
                            }
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
            // General path: hash to the bucket, then re-check exact equality
            // under three-valued logic (`PartialEq` would report a match for
            // an UNKNOWN comparison; see [`keys_match`]).
            JoinHashTable::Value(table) => {
                let Some(bucket) = table.get(&hash_value(key)) else {
                    return;
                };
                bucket
                    .iter()
                    .find(|(k, _)| keys_match(k, key))
                    .map(|(_, refs)| refs)
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
                // Cheap top-level early-out. A key with a NULL *nested* inside
                // a container is not caught here on purpose: `keys_match`
                // already rejects it, and paying for a deep scan on every
                // probe row would tax the far more common null-free container
                // key. The build side does scan deeply — once per build row —
                // so no such key is ever in the table to be matched against.
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use thin_vec::thin_vec;

    use super::{contains_null, key_as_i64, keys_match};
    use crate::runtime::{ordermap::OrderMap, value::Value};

    fn list(items: Vec<Value>) -> Value {
        Value::List(Arc::new(items.into_iter().collect()))
    }

    fn map(entries: Vec<(&str, Value)>) -> Value {
        Value::Map(Arc::new(OrderMap::from_vec(
            entries
                .into_iter()
                .map(|(k, v)| (Arc::new(k.to_string()), v))
                .collect(),
        )))
    }

    fn string(s: &str) -> Value {
        Value::String(Arc::new(s.to_string()))
    }

    #[test]
    fn contains_null_sees_through_containers() {
        // Scalars: only NULL itself.
        assert!(contains_null(&Value::Null));
        assert!(!contains_null(&Value::Int(1)));
        assert!(!contains_null(&Value::Float(1.5)));
        assert!(!contains_null(&Value::Bool(true)));
        assert!(!contains_null(&string("x")));

        // Lists, at any depth.
        assert!(contains_null(&list(vec![Value::Int(1), Value::Null])));
        assert!(contains_null(&list(vec![list(vec![Value::Null])])));
        assert!(!contains_null(&list(vec![Value::Int(1), Value::Int(2)])));
        assert!(!contains_null(&list(vec![])));

        // Maps, including nested inside a list.
        assert!(contains_null(&map(vec![("a", Value::Null)])));
        assert!(contains_null(&list(vec![map(vec![("a", Value::Null)])])));
        assert!(!contains_null(&map(vec![("a", Value::Int(1))])));

        // Paths share the list representation.
        assert!(!contains_null(&Value::Path(Arc::new(thin_vec![]))));
    }

    #[test]
    fn keys_match_requires_a_conclusive_equality() {
        // This is the regression under test: `PartialEq` drops the
        // `DisjointOrNull` flag, so it reports `[1, null] == [1, null]`,
        // which turned an UNKNOWN predicate into a join match.
        let null_list = list(vec![Value::Int(1), Value::Null]);
        assert!(null_list == null_list, "PartialEq is deliberately laxer");
        assert!(!keys_match(&null_list, &null_list));

        // A NULL on only one side is no match either.
        assert!(!keys_match(
            &null_list,
            &list(vec![Value::Int(1), Value::Int(2)])
        ));
        assert!(!keys_match(&Value::Null, &Value::Null));
        assert!(!keys_match(&Value::Null, &Value::Int(1)));

        // Maps whose values are of incomparable types compare FALSE, not TRUE.
        assert!(!keys_match(
            &map(vec![("a", Value::Int(1))]),
            &map(vec![("a", string("1"))])
        ));
        assert!(!keys_match(
            &map(vec![("a", Value::Null)]),
            &map(vec![("a", Value::Null)])
        ));
    }

    #[test]
    fn keys_match_still_accepts_legitimate_joins() {
        // Scalars, including Cypher's cross-type numeric equality.
        assert!(keys_match(&Value::Int(30), &Value::Int(30)));
        assert!(keys_match(&Value::Int(30), &Value::Float(30.0)));
        assert!(!keys_match(&Value::Int(30), &Value::Int(31)));
        assert!(keys_match(&string("a"), &string("a")));
        assert!(!keys_match(&string("a"), &string("b")));

        // NULL-free containers.
        let pair = list(vec![Value::Int(1), Value::Int(2)]);
        assert!(keys_match(&pair, &pair));
        assert!(!keys_match(&pair, &list(vec![Value::Int(1)])));
        assert!(keys_match(&list(vec![]), &list(vec![])));
        let m = map(vec![("a", Value::Int(1)), ("b", string("x"))]);
        assert!(keys_match(&m, &m));
        assert!(!keys_match(&m, &map(vec![("a", Value::Int(1))])));
    }

    #[test]
    fn integer_fast_path_never_sees_a_null_bearing_key() {
        // The `Int` table can only ever be keyed by a scalar numeric, so a
        // key that `contains_null` cannot reach it via `key_as_i64`.
        assert_eq!(key_as_i64(&Value::Int(7)), Some(7));
        assert_eq!(key_as_i64(&Value::Float(7.0)), Some(7));
        assert_eq!(key_as_i64(&Value::Null), None);
        assert_eq!(key_as_i64(&list(vec![Value::Int(1), Value::Null])), None);
        assert_eq!(key_as_i64(&map(vec![("a", Value::Null)])), None);
    }
}
