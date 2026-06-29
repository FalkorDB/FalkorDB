//! Batch-mode sort operator — orders result rows by one or more expressions.
//!
//! This is a *blocking* operator. It has two regimes, picked on the first
//! `next()` call:
//!
//! * **Full sort** (no `LIMIT`, or a very large one): every child batch is
//!   concatenated columnar via [`Batch::concat`] (no per-row `Row`), the sort
//!   keys are evaluated once per row, and the rows are ordered by sorting
//!   contiguous `(primary key, row index)` pairs — equal primary keys break by
//!   the remaining keys, then full content, then arrival. Rows are emitted
//!   `BATCH_SIZE` at a time by gathering from the combined buffer.
//!
//! * **Top-`k`** (`ORDER BY … LIMIT k`, with `k + skip` small): a streaming
//!   bounded max-heap of capacity `k + skip` keeps only the rows that can
//!   survive the limit, so the whole input is never buffered or fully sorted.
//!   This is `O(N·log k)` time and `O(k)` memory instead of `O(N·log N)` +
//!   `O(N)`.
//!
//! ```text
//!  Child batches
//!       │
//!       ▼
//!  ┌──────────────────────────────┐     ┌──────────────────────────────┐
//!  │ Full sort:                   │ or  │ Top-k:                       │
//!  │   Batch::concat → keys →     │     │   bounded max-heap of k+skip │
//!  │   sort (key,idx) pairs       │     │   (evict the worst row)      │
//!  └──────────────┬───────────────┘     └──────────────┬───────────────┘
//!                 │                                     │
//!                 └──────────────┬──────────────────────┘
//!                                ▼
//!                   emit BATCH_SIZE rows at a time
//! ```
//!
//! When primary sort keys are equal, both regimes apply the *same* deterministic
//! tiebreaker: compare every bound slot position-by-position, then fall back to
//! arrival order. Sharing the tiebreaker means the top-k heap retains the same
//! rows, in the same order, as a full sort followed by truncation — which some
//! queries rely on (e.g. comparing a result against the same query with a
//! reversed traversal pattern, tests/flow/test_social).

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::row::Row;
use crate::runtime::row::RowView;
use crate::runtime::{
    batch::{
        BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow, ValueKinds, floats_from_values,
        ints_from_values,
    },
    runtime::Runtime,
    value::{CompareValue, Value},
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Upper bound on `limit + skip` for which the streaming top-k heap is used.
/// Beyond this the columnar full sort (then truncate) wins, because the heap's
/// per-surviving-row materialization no longer pays off against bulk gather.
const TOP_K_HEAP_MAX: usize = 4 * BATCH_SIZE;

/// One sort key paired with its descending flag, ordered so a plain
/// lexicographic comparison of a key vector matches the requested direction.
struct OrderedKey {
    value: Value,
    desc: bool,
}

impl OrderedKey {
    /// Compares two keys in *final output order*: the underlying `Value` total
    /// order, with the result reversed for `DESC`. Folding the direction in here
    /// lets every layer above (the key-vector and heap comparisons) use plain
    /// ascending comparisons without re-checking ASC/DESC.
    fn cmp_key(
        &self,
        other: &Self,
    ) -> Ordering {
        let (ordering, _) = self.value.compare_value(&other.value);
        if self.desc {
            ordering.reverse()
        } else {
            ordering
        }
    }
}

impl PartialEq for OrderedKey {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.cmp_key(other) == Ordering::Equal
    }
}
impl Eq for OrderedKey {}
impl PartialOrd for OrderedKey {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedKey {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        self.cmp_key(other)
    }
}

/// Deterministic content tiebreaker for two materialised rows, mirroring
/// [`Batch::compare_rows_at`] used by the full-sort path: compare every slot
/// position-by-position, treating an absent/out-of-range slot as `Null` (so two
/// such slots are equal). `get_by_id` returns the stored value even for a
/// value-present-but-unbound slot, matching how the full sort compares a
/// `value_only` column's values. Borrows the stored `Value`s (no clone).
///
/// This is what makes the top-k heap pick the *same* surviving rows, in the
/// *same* order, as a full sort followed by truncation: when the ORDER BY keys
/// tie, both paths fall back to the full row content and then arrival order.
fn compare_row_content(
    a: &Row,
    b: &Row,
) -> Ordering {
    let n = a.len().max(b.len());
    for id in 0..n {
        let ordering = match (a.get_by_id(id as u32), b.get_by_id(id as u32)) {
            (Some(va), Some(vb)) => va.compare_value(vb).0,
            (Some(va), None) => va.compare_value(&Value::Null).0,
            (None, Some(vb)) => Value::Null.compare_value(vb).0,
            (None, None) => Ordering::Equal,
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

/// A candidate row held by the bounded top-k heap. Ordered so the max-heap's
/// root is the row that sorts *last* — i.e. the next one to evict once the heap
/// is full. Ties on the sort keys break by full row *content* and finally by
/// `seq` (arrival order), so the heap reproduces the exact total order of the
/// full-sort path while staying stable.
struct HeapEntry {
    keys: SmallVec<[OrderedKey; 2]>,
    seq: u64,
    row: Row,
}

impl PartialEq for HeapEntry {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        // Lexicographic over the key vector (multi-key ORDER BY: first key, then
        // ties broken by the next), then by full row content, then by arrival
        // `seq` so equal rows keep a stable order. This is the final output
        // order, so the max-heap's greatest element is the row that sorts *last*
        // (the eviction candidate). The content tiebreaker matches the full-sort
        // path's `compare_rows_at` loop, so both regimes order ties identically
        // (required by queries that compare a result against a reordered
        // pattern, e.g. tests/flow/test_social).
        self.keys
            .as_slice()
            .cmp(other.keys.as_slice())
            .then_with(|| compare_row_content(&self.row, &other.row))
            .then(self.seq.cmp(&other.seq))
    }
}

/// One materialised sort-key column for the full-sort path, specialised by type
/// so the comparator can compare raw scalars and skip the `Value` enum dispatch.
///
/// A key is `Ints`/`Floats` only when *every* row evaluated that key to the same
/// primitive type; any null, mismatch, or non-primitive value (and any mixed
/// `Int`/`Float` column) keeps the whole key on the `Values` path. That keeps
/// `compare_at` byte-for-byte identical to `Value::compare_value`: an all-`Int`
/// column always hits the `(Int, Int) => a.cmp(b)` arm, and an all-`Float`
/// column always hits `(Float, Float) => compare_floats`, which is exactly
/// `partial_cmp(...).unwrap_or(Less)`.
enum KeyColumn {
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Values(Vec<Value>),
}

impl KeyColumn {
    /// Specialises a key column to `Ints`/`Floats` when homogeneous, else keeps
    /// the boxed `Values` (consuming them, no clone). A null (or any non-numeric
    /// value) keeps the whole key on the `Values` path so `compare_at` stays
    /// byte-for-byte identical to `Value::compare_value`.
    fn classify(values: Vec<Value>) -> Self {
        let kinds = ValueKinds::scan(&values);
        if kinds.all_int_no_null() {
            Self::Ints(ints_from_values(values))
        } else if kinds.all_float_no_null() {
            Self::Floats(floats_from_values(values))
        } else {
            Self::Values(values)
        }
    }

    /// Compares rows `a` and `b` of this key column in ascending value order
    /// (the caller applies the `DESC` reversal). Mirrors `Value::compare_value`
    /// for the specialised types.
    #[inline]
    fn compare_at(
        &self,
        a: usize,
        b: usize,
    ) -> Ordering {
        match self {
            Self::Ints(v) => v[a].cmp(&v[b]),
            Self::Floats(v) => v[a].partial_cmp(&v[b]).unwrap_or(Ordering::Less),
            Self::Values(v) => v[a].compare_value(&v[b]).0,
        }
    }
}

pub struct SortOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Option<Box<BatchOp<'a>>>,
    trees: &'a [(QueryExpr<Variable>, bool)],
    /// The materialised, ordered output buffer (built on the first `next`), the
    /// row order to emit, and a cursor into that order.
    sorted: Option<Batch<'a>>,
    order: Vec<usize>,
    pos: usize,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// When set, only the top `limit` rows (after skip) are needed.
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
            sorted: None,
            order: Vec::new(),
            pos: 0,
            idx,
            limit,
            skip,
        }
    }

    /// Full sort: concatenate every child batch columnar, evaluate the sort keys
    /// once per row, and stable-sort the row indices. When `truncate` is set
    /// (a large `LIMIT`), drop the indices the Skip/Limit operators above will
    /// never consume.
    fn build_full_sort(
        runtime: &'a Runtime<'a>,
        trees: &'a [(QueryExpr<Variable>, bool)],
        child: Box<BatchOp<'a>>,
        truncate: Option<usize>,
    ) -> Result<(Batch<'a>, Vec<usize>), String> {
        let mut batches: Vec<Batch<'a>> = Vec::new();
        for batch_result in child {
            batches.push(batch_result?);
        }
        let combined = Batch::concat(&batches);
        drop(batches);
        let total = combined.len();
        if total == 0 {
            return Ok((combined, Vec::new()));
        }

        // Evaluate the sort keys once per row through a borrowed columnar view
        // (so `rand()` and other expression keys still work) with no per-row
        // `Row` allocation. Each key goes into its own column so it can be
        // specialised to a typed scalar vector (`KeyColumn`), letting the
        // comparator skip the `Value` enum dispatch for primitive keys.
        let num_keys = trees.len();
        let mut key_cols: Vec<Vec<Value>> =
            (0..num_keys).map(|_| Vec::with_capacity(total)).collect();
        for row in 0..total {
            let view = BatchRow::new(&combined, row);
            for (k, (tree, _desc)) in trees.iter().enumerate() {
                let value = ExprEval::from_runtime(runtime).eval(
                    tree,
                    tree.root().idx(),
                    Some(&view),
                    None,
                )?;
                key_cols[k].push(value);
            }
        }
        let typed_keys: Vec<KeyColumn> = key_cols.into_iter().map(KeyColumn::classify).collect();

        let num_columns = combined.num_columns();
        // Comparator for rows that tie on the *primary* key: the remaining keys
        // (already direction-folded), then a position-by-position content
        // compare, then arrival order. Columns unbound in every row read back as
        // `Null` on both sides and so never move the order; `compare_rows_at`
        // borrows the stored values instead of cloning. Only invoked on a
        // primary-key tie, so the random-access content scan rarely fires.
        let tiebreak = |a: usize, b: usize| -> Ordering {
            for (k, (_tree, desc)) in trees.iter().enumerate().skip(1) {
                let ordering = typed_keys[k].compare_at(a, b);
                if ordering != Ordering::Equal {
                    return if *desc { ordering.reverse() } else { ordering };
                }
            }
            for id in 0..num_columns {
                let ordering = combined.compare_rows_at(id as u32, a, b);
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            a.cmp(&b)
        };
        // Pack the primary key inline so the sort scans contiguous `(key, idx)`
        // pairs instead of indices that random-gather the key column — the same
        // total order, far fewer cache misses. The `Values` primary key keeps
        // the index sort, which already borrows through the comparator.
        let primary_desc = trees[0].1;
        let mut order: Vec<usize> = match &typed_keys[0] {
            KeyColumn::Ints(keys) => {
                let mut pairs: Vec<(i64, u32)> = (0..total).map(|i| (keys[i], i as u32)).collect();
                pairs.sort_unstable_by(|&(ka, ia), &(kb, ib)| {
                    let primary = if primary_desc {
                        kb.cmp(&ka)
                    } else {
                        ka.cmp(&kb)
                    };
                    primary.then_with(|| tiebreak(ia as usize, ib as usize))
                });
                pairs.into_iter().map(|(_, i)| i as usize).collect()
            }
            KeyColumn::Floats(keys) => {
                let mut pairs: Vec<(f64, u32)> = (0..total).map(|i| (keys[i], i as u32)).collect();
                pairs.sort_unstable_by(|&(ka, ia), &(kb, ib)| {
                    let primary = ka.partial_cmp(&kb).unwrap_or(Ordering::Less);
                    let primary = if primary_desc {
                        primary.reverse()
                    } else {
                        primary
                    };
                    primary.then_with(|| tiebreak(ia as usize, ib as usize))
                });
                pairs.into_iter().map(|(_, i)| i as usize).collect()
            }
            KeyColumn::Values(_) => {
                let mut order: Vec<usize> = (0..total).collect();
                order.sort_by(|&a, &b| {
                    let ordering = typed_keys[0].compare_at(a, b);
                    let primary = if primary_desc {
                        ordering.reverse()
                    } else {
                        ordering
                    };
                    primary.then_with(|| tiebreak(a, b))
                });
                order
            }
        };

        if let Some(cap) = truncate {
            order.truncate(cap);
        }

        Ok((combined, order))
    }

    /// Streaming top-k: keep only the `cap = limit + skip` smallest rows in a
    /// bounded max-heap, evicting the current worst when a better row arrives.
    /// The whole input is never buffered nor fully sorted; only the rows that
    /// enter the heap are materialised.
    fn build_top_k(
        runtime: &'a Runtime<'a>,
        trees: &'a [(QueryExpr<Variable>, bool)],
        child: Box<BatchOp<'a>>,
        cap: usize,
    ) -> Result<(Batch<'a>, Vec<usize>), String> {
        if cap == 0 {
            // `ORDER BY ... LIMIT 0` (or `SKIP n LIMIT 0`) yields no rows, but
            // the child must still be drained so its side effects fire and any
            // error surfaces — the full-sort path always consumes the child, so
            // match that here instead of returning early.
            for batch_result in child {
                batch_result?;
            }
            return Ok((Batch::new(0), Vec::new()));
        }
        let num_keys = trees.len();
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(cap);
        let mut seq: u64 = 0;
        for batch_result in child {
            let batch = batch_result?;
            for row in batch.active_indices() {
                let view = BatchRow::new(&batch, row);
                // Evaluate the keys first; the row is only materialised if it
                // actually survives the limit (the common case rejects it).
                let mut keys: SmallVec<[OrderedKey; 2]> = SmallVec::with_capacity(num_keys);
                for (tree, desc) in trees {
                    let value = ExprEval::from_runtime(runtime).eval(
                        tree,
                        tree.root().idx(),
                        Some(&view),
                        None,
                    )?;
                    keys.push(OrderedKey { value, desc: *desc });
                }
                let cur_seq = seq;
                seq += 1;

                if heap.len() < cap {
                    let row = view.to_owned_row();
                    heap.push(HeapEntry {
                        keys,
                        seq: cur_seq,
                        row,
                    });
                } else {
                    // The heap is a max-heap, so `peek()` is the worst survivor
                    // (the row that sorts last). Replace it only when the new row
                    // sorts strictly before it under the same total order as
                    // `HeapEntry`: keys, then full row content, then `seq`.
                    //
                    // A *strict* key comparison decides most rows without ever
                    // materialising them, so the common reject path stays
                    // allocation-free. Only a key *tie* needs the content
                    // tiebreaker, which requires the candidate row, so it is
                    // materialised lazily — exactly as the full-sort path falls
                    // back from keys to `compare_rows_at`. Matching that fallback
                    // is what makes the heap retain the *same* rows in the *same*
                    // order as a full sort + truncate.
                    //
                    // On a full content tie the later arrival (larger `seq`)
                    // loses, so the earliest-arriving rows are kept — a stable
                    // top-k.
                    let (wins, prematerialized) = {
                        let key_ord = {
                            let worst = heap.peek().expect("heap is full, so non-empty");
                            keys.as_slice().cmp(worst.keys.as_slice())
                        };
                        match key_ord {
                            Ordering::Less => (true, None),
                            Ordering::Greater => (false, None),
                            Ordering::Equal => {
                                let row = view.to_owned_row();
                                let worst = heap.peek().expect("heap is full, so non-empty");
                                let wins = compare_row_content(&row, &worst.row)
                                    .then(cur_seq.cmp(&worst.seq))
                                    == Ordering::Less;
                                (wins, Some(row))
                            }
                        }
                    };
                    if wins {
                        let row = prematerialized.unwrap_or_else(|| view.to_owned_row());
                        heap.pop();
                        heap.push(HeapEntry {
                            keys,
                            seq: cur_seq,
                            row,
                        });
                    }
                }
            }
        }

        // `into_sorted_vec` drains the max-heap in ascending `HeapEntry` order,
        // which is exactly the requested output order; rebuild a dense batch.
        let mut builder = BatchBuilder::new();
        for entry in heap.into_sorted_vec() {
            builder.push_row(&entry.row);
        }
        let combined = builder.finish();
        let total = combined.len();
        Ok((combined, (0..total).collect()))
    }
}

impl<'a> Iterator for SortOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Consume and sort all input on the first call.
        if let Some(child) = self.child.take() {
            let result = match self.limit {
                Some(limit) => {
                    let cap = limit.saturating_add(self.skip);
                    if cap <= TOP_K_HEAP_MAX {
                        Self::build_top_k(self.runtime, self.trees, child, cap)
                    } else {
                        Self::build_full_sort(self.runtime, self.trees, child, Some(cap))
                    }
                }
                None => Self::build_full_sort(self.runtime, self.trees, child, None),
            };
            match result {
                Ok((combined, order)) => {
                    self.sorted = Some(combined);
                    self.order = order;
                    self.pos = 0;
                }
                Err(e) => return Some(Err(e)),
            }
        }

        // Emit the sorted rows BATCH_SIZE at a time by gathering from the
        // combined buffer in sorted order.
        if self.pos >= self.order.len() {
            return None;
        }
        let end = (self.pos + BATCH_SIZE).min(self.order.len());
        let out = {
            let combined = self.sorted.as_ref().unwrap();
            combined.gather(&self.order[self.pos..end])
        };
        self.pos = end;
        Some(Ok(out))
    }
}
