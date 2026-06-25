//! Batch-mode sort operator — orders result rows by one or more expressions.
//!
//! This is a *blocking* operator. It has two regimes, picked on the first
//! `next()` call:
//!
//! * **Full sort** (no `LIMIT`, or a very large one): every child batch is
//!   concatenated columnar via [`Batch::concat`] (no per-row `Row`), the sort
//!   keys are evaluated once per row, the row *indices* are stable-sorted, and
//!   rows are emitted `BATCH_SIZE` at a time by gathering from the combined
//!   buffer.
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
//!  │   stable-sort indices        │     │   (evict the worst row)      │
//!  └──────────────┬───────────────┘     └──────────────┬───────────────┘
//!                 │                                     │
//!                 └──────────────┬──────────────────────┘
//!                                ▼
//!                   emit BATCH_SIZE rows at a time
//! ```
//!
//! When primary sort keys are equal, the full-sort path applies a deterministic
//! tiebreaker comparing every bound slot position-by-position and finally the
//! original row index. The top-k path breaks ties by arrival order (a stable
//! heap); for rows tied at the limit boundary, which of them is retained is
//! unspecified by Cypher, so either tiebreak is correct.

use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::row::Row;
use crate::runtime::row::RowView;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
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

/// A candidate row held by the bounded top-k heap. Ordered so the max-heap's
/// root is the row that sorts *last* — i.e. the next one to evict once the heap
/// is full. Ties on the sort keys break by `seq` (arrival order) so the heap is
/// stable.
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
        // ties broken by the next), then by arrival `seq` so equal-keyed rows
        // keep a stable order. This is the final output order, so the max-heap's
        // greatest element is the row that sorts *last* (the eviction candidate).
        self.keys
            .as_slice()
            .cmp(other.keys.as_slice())
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
    /// the boxed `Values` (consuming them, no clone).
    fn classify(values: Vec<Value>) -> Self {
        let mut all_int = true;
        let mut all_float = true;
        for v in &values {
            match v {
                Value::Int(_) => all_float = false,
                Value::Float(_) => all_int = false,
                _ => {
                    all_int = false;
                    all_float = false;
                    break;
                }
            }
        }
        if values.is_empty() {
            Self::Values(values)
        } else if all_int {
            Self::Ints(
                values
                    .into_iter()
                    .map(|v| match v {
                        Value::Int(i) => i,
                        _ => unreachable!("column proven all-Int"),
                    })
                    .collect(),
            )
        } else if all_float {
            Self::Floats(
                values
                    .into_iter()
                    .map(|v| match v {
                        Value::Float(f) => f,
                        _ => unreachable!("column proven all-Float"),
                    })
                    .collect(),
            )
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
        let mut order: Vec<usize> = (0..total).collect();
        order.sort_by(|&a, &b| {
            // Lexicographic multi-key comparison over the typed key columns.
            for (k, (_tree, desc)) in trees.iter().enumerate() {
                let ordering = typed_keys[k].compare_at(a, b);
                if ordering != Ordering::Equal {
                    return if *desc { ordering.reverse() } else { ordering };
                }
            }
            // Deterministic tiebreaker: compare bound slots position-by-position.
            // Columns unbound in every row read back as `Null` on both sides and
            // so never change the ordering. `compare_rows_at` borrows stored
            // values instead of cloning them.
            for id in 0..num_columns {
                let ordering = combined.compare_rows_at(id as u32, a, b);
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            // Final total-order fallback: rows still equal here are identical in
            // every compared column. `sort_by` is unstable, so break remaining
            // ties by original row index to keep output deterministic.
            a.cmp(&b)
        });

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
                    // sorts strictly before it, i.e. its `(keys, seq)` is smaller
                    // — exactly `HeapEntry`'s order, but compared here without
                    // building a `HeapEntry`, so the row is materialised
                    // (`to_owned_row`, which clones values) only when it actually
                    // wins a slot. The common reject path stays allocation-free.
                    //
                    // The `Equal` arm is effectively always false: rows arrive in
                    // increasing `seq`, so `cur_seq` exceeds every `seq` already
                    // in the heap. On a key tie the new (later) row is therefore
                    // rejected, which keeps the *earliest*-arriving rows — a
                    // stable top-k.
                    let smaller = {
                        let worst = heap.peek().expect("heap is full, so non-empty");
                        match keys.as_slice().cmp(worst.keys.as_slice()) {
                            Ordering::Less => true,
                            Ordering::Greater => false,
                            Ordering::Equal => cur_seq < worst.seq,
                        }
                    };
                    if smaller {
                        let row = view.to_owned_row();
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
