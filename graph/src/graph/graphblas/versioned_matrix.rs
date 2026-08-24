//! Copy-on-write sparse matrix with MVCC delta tracking.
//!
//! This module provides [`VersionedMatrix`], which wraps a base [`Matrix`] with
//! two delta matrices to track pending additions and deletions. This is the
//! building block for snapshot isolation: readers see the committed base state
//! while writers accumulate changes in separate delta matrices.
//!
//! ## Internal Structure
//!
//! ```text
//!   VersionedMatrix
//!     |
//!     |-- m   Cow<Matrix>   Base matrix (committed / shared with readers)
//!     |-- dp  Cow<Matrix>   Delta-plus  (pending additions)
//!     |-- dm  Cow<Matrix>   Delta-minus (pending deletions)
//!
//!   Effective state = (m UNION dp) MINUS dm
//! ```
//!
//! Each inner matrix is wrapped in [`Cow`] (copy-on-write). When a new version
//! is created via [`Dup`], the `Cow` clones share the underlying `Arc<Matrix>`
//! until a mutation triggers a deep copy.
//!
//! ## Read Path
//!
//! ```text
//!   get(i, j):
//!     m has (i,j)?
//!       yes --> dm has (i,j)?  -->  yes: None (deleted)
//!                                   no:  Some(true)
//!       no  --> dp has (i,j)?  -->  yes: Some(true)
//!                                   no:  None
//! ```
//!
//! ## Write Path
//!
//! ```text
//!   set(i, j):
//!     m has (i,j)?
//!       yes --> remove (i,j) from dm   (un-delete)
//!       no  --> add (i,j) to dp        (new addition)
//!
//!   remove(i, j):
//!     m has (i,j)?
//!       yes --> add (i,j) to dm        (mark deleted)
//!       no  --> remove (i,j) from dp   (undo pending add)
//! ```
//!
//! UINT64-valued layers (e.g. `Tensor`'s forward adjacency, which owns its
//! own `m`/`dp`/`dm` triple) additionally support in-place *value* updates of
//! a committed entry: the new value is written to `dp`, *shadowing* the live
//! `m` entry (no `dm` mask — `dm` marks pure deletions only, so `dp ∩ dm = ∅`
//! and `dm ⊆ m`). [`Iter`]'s sorted merge yields the live `dp` value for
//! shadowed pairs and skips the stale `m` entry.
//!
//! ## Flush
//!
//! [`flush`](VersionedMatrix::flush) merges a delta into the base matrix (`dp`
//! via element-wise add, `dm` via masked removal) and clears it. It executes
//! only the folds already latched by the policy documented on
//! [`WRITE_FOLD_K`], never deciding one itself.
//!
//! ## Iterator
//!
//! [`Iter`] performs a three-way sorted merge over the `m`, `dp`, and `dm`
//! iterators (GraphBLAS row iterators yield ascending `(row, col)` order):
//! `m` entries matched by the `dm` lookahead are dropped, `dp` entries are
//! interleaved in order (winning over a shadowed `m` entry at the same
//! position), producing the effective state — sorted by `(row, col)` —
//! without materializing a merged matrix or issuing per-entry point lookups.

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::{
    GxB_Print_Level,
    matrix::{self, Dup, Matrix},
    serialization::{Decode, Encode, Reader, Writer},
};
use crate::graph::{
    cow::Cow,
    graphblas::matrix::{BoolExtract, IterExtract, MatrixType},
};

/// A delta layer is folded into the base once its size justifies a base
/// rewrite. Every write transaction pays `O(|delta|)` just to touch a shared
/// delta, while a fold costs a fixed `F` amortized over the entries the
/// transaction contributes (`tx_added`). With transactions of `tx_added`
/// entries accumulating to a fold point `D`, writing `D` entries costs
/// `Σᵏ (i·tx_added)·w + F ≈ w·D²/(2·tx_added) + F`, i.e. per entry
/// `w·D/(2·tx_added) + F/D`: a rewrite tax that grows with `D` against a fold
/// bill that shrinks with it. The two balance at
///
/// ```text
/// D* = sqrt(2 · (F/w) · tx_added)
/// ```
///
/// which is why this is a sqrt rule and not an LSM-style ratio (`D ≥ base/K`).
/// A ratio is right when each entry is rewritten `O(1)` times between merges;
/// here every transaction re-touches the *whole* delta, so the accumulated tax
/// is quadratic in `D` and depends on `tx_added`. Dropping `tx_added` gets both
/// ends wrong — at `tx_added = 1` on a big graph the delta grows until each
/// write pays milliseconds, and at a bulk `tx_added` it folds every
/// transaction. A flat absolute limit fails for the same reason, and measures
/// worse than the sqrt rule on both per-write cost and worst-case stall.
///
/// `F/w` is measured rather than modelled; `fold_cost_bench` derives the two
/// constants below on the target machine, and its output is what to re-read if
/// they ever need retuning. Two properties of that measurement shape the rule:
///
/// * `F`, the fold, is dominated by a fixed cost and is independent of `nrows`
///   — an empty fold costs the same at nrows = 65k and at 16.7M. So `F` is
///   effectively constant above a modest base and the balance point is *flat*
///   in the base rather than growing like `sqrt(base)`. (An earlier revision
///   used `base.nvals() + base.nrows()` on the theory that a fold rewrites the
///   row-pointer structure. The measurement refutes it: the `+16%` instructions
///   that motivated the `nrows` term came from fold *frequency*, which the
///   measured `F` now fixes directly.)
/// * `w` differs by two orders of magnitude between the two paths, which is
///   what the write/read split below encodes. A write transaction nobody reads
///   pays only the COW dup — memcpy speed, and duping a *pending* matrix costs
///   the same as an assembled one, so `dup` does not assemble. A transaction
///   whose delta is also materialized pays the pending-tuple merge, which is
///   `O(|delta|)` no matter how little the transaction added.
///
/// Since `D* ∝ 1/sqrt(w)`, the write path's balance point sits well above the
/// read path's. Reads keep paying for a lingering delta on every access, so
/// [`should_fold_read`] — the policy for `wait`, which only runs when something
/// reads the matrix — folds at the tighter point, promptly bounding the delta
/// once a workload turns read-heavy.
///
/// A delta comparable to the base always folds (`2·|delta| ≥ base_nvals`): the
/// fold then costs the same order as one delta touch, and without it a one-shot
/// bulk transaction (whose huge `tx_added` defeats the sqrt term) can leave a
/// base-sized delta taxing every later transaction. This is also what bounds
/// the delta on a pure-write stream, where the write path deliberately lets it
/// run large — an absolute cap there is not free, costing multiples of bulk
/// write throughput to save a one-time assembly for the first reader after the
/// burst.
const WRITE_FOLD_K: u64 = 20_500_000;

/// `2 · F / w_merge` — the read path's `D*² / tx_added`, putting the balance
/// point at `sqrt(READ_FOLD_K · tx_added)` entries.
const READ_FOLD_K: u64 = 82_000;

/// Deltas below this never fold: the fold would cost more than the tax it
/// removes. It sits essentially on the read-path balance point for a
/// single-entity transaction (`sqrt(READ_FOLD_K) ≈ 286`).
const MIN_FOLD_DELTA: u64 = 256;

/// Write-path fold policy, evaluated in `dup` (version creation). Balances the
/// COW dup tax against the fold; see the module docs above for the model and
/// the measurements behind [`WRITE_FOLD_K`].
pub(super) fn should_fold(
    delta_nvals: u64,
    tx_added: u64,
    base_nvals: u64,
) -> bool {
    fold_balance(delta_nvals, tx_added, base_nvals, WRITE_FOLD_K)
}

/// Read-path fold policy, evaluated in `wait`. Same model as
/// [`should_fold`], but the transaction pays the `O(|delta|)` pending-tuple
/// merge on top of the dup, so `w` is 250x larger and the balance point
/// `sqrt(250) ≈ 16x` tighter. See [`READ_FOLD_K`].
pub(super) fn should_fold_read(
    delta_nvals: u64,
    tx_added: u64,
    base_nvals: u64,
) -> bool {
    fold_balance(delta_nvals, tx_added, base_nvals, READ_FOLD_K)
}

/// `|delta| ≥ sqrt(k · tx_added)`, or a delta comparable to the base.
fn fold_balance(
    delta_nvals: u64,
    tx_added: u64,
    base_nvals: u64,
    k: u64,
) -> bool {
    // The `MIN_FOLD_DELTA` floor is applied once, here: `delta_dominates_base`
    // re-checks it for its own callers, and repeating it in the disjunction
    // would suggest the two arms had different floors.
    tx_added > 0
        && delta_nvals >= MIN_FOLD_DELTA
        && (delta_nvals.saturating_mul(2) >= base_nvals
            || delta_nvals.saturating_mul(delta_nvals) >= k.saturating_mul(tx_added))
}

/// The escape hatch of [`fold_balance`] on its own: a delta comparable to the
/// base folds whatever the balance point says, because the fold then costs the
/// same order as a single delta touch — and because the delta is holding as
/// much memory as the base it shadows. Evaluated on the approximate counters,
/// so it never forces a delta's pending tuples to merge.
pub(super) fn delta_dominates_base(
    delta_nvals: u64,
    base_nvals: u64,
) -> bool {
    delta_nvals >= MIN_FOLD_DELTA && delta_nvals.saturating_mul(2) >= base_nvals
}

/// One delta layer plus the bookkeeping the fold policy reads about it.
///
/// Both deltas need the same three pieces of state, so they live next to the
/// layer they describe rather than as `dp_`/`dm_`-prefixed pairs on the owner:
/// every method that touches a delta would otherwise spell its bookkeeping out
/// twice.
///
/// [`Deref`] gives read access to the layer (`nvals`, `get`, `wait`, `iter`),
/// but there is deliberately **no** `DerefMut`: a mutation that does not move
/// `count` leaves the fold policy reading a delta size that never happened, so
/// writes go through the counted [`Self::insert`] / [`Self::erase`], or through
/// [`Self::layer_mut`] for callers that resync the count themselves.
/// Bits in [`RowFilter`]'s bitmap. 4096 bits is 512 bytes, allocated only once
/// a delta actually holds something, and enough that the deltas the fold policy
/// permits stay well below saturation.
const ROW_FILTER_BITS: usize = 32768;
const ROW_FILTER_WORDS: usize = ROW_FILTER_BITS / 64;

/// Which rows a delta layer may hold entries in.
///
/// Readers merge three layers per scan, and the existing short-circuit asks
/// whether a layer is empty *in total*. That is the wrong question for a narrow
/// scan: one pending entry anywhere makes every single-row read attach and seek
/// a GraphBLAS iterator that yields nothing (#2430). This answers the narrower
/// question — may this layer hold row `r`? — without touching GraphBLAS.
///
/// **The only safety property that matters is that it never says "no" when the
/// answer is yes.** [`Self::Unknown`] is therefore the default for anything not
/// explicitly tracked: a bulk write, a transpose, any future path that reaches
/// [`Delta::layer_mut`]. Being wrong in the conservative direction costs the
/// scan it would have done anyway.
#[derive(Clone)]
enum RowFilter {
    /// Nothing has been added since the last clear, so no row is present. The
    /// state a fresh or just-folded delta is in, and the one that costs nothing.
    Empty,
    /// A row is present only if its hash bit is set. False positives are
    /// expected (hash collisions, and entries erased since); false negatives
    /// are not possible.
    Bits(Box<[u64; ROW_FILTER_WORDS]>),
    /// A bulk write moved entries this cannot track. Every row may be present.
    Unknown,
}

impl RowFilter {
    /// Fibonacci hashing: rows here are `compound_key(src, dst)` values whose
    /// low bits are a raw destination id, so the multiply is what spreads them.
    const fn slot(row: u64) -> (usize, u64) {
        let h = (row.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - 15)) as usize;
        (h / 64, 1u64 << (h % 64))
    }

    fn add(
        &mut self,
        row: u64,
    ) {
        match self {
            Self::Unknown => {}
            Self::Empty => {
                let mut bits = Box::new([0u64; ROW_FILTER_WORDS]);
                let (w, b) = Self::slot(row);
                bits[w] |= b;
                *self = Self::Bits(bits);
            }
            Self::Bits(bits) => {
                let (w, b) = Self::slot(row);
                bits[w] |= b;
            }
        }
    }

    /// Whether any row in `min..=max` may be present. Wide ranges answer `true`
    /// without testing: a scan that broad wants the delta anyway, and the point
    /// of this is the single-row case.
    fn may_hold(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> bool {
        match self {
            Self::Unknown => true,
            Self::Empty => false,
            Self::Bits(bits) => {
                let Some(span) = max_row.checked_sub(min_row) else {
                    return false;
                };
                if span >= 64 {
                    return true;
                }
                (min_row..=max_row).any(|r| {
                    let (w, b) = Self::slot(r);
                    bits[w] & b != 0
                })
            }
        }
    }
}

pub(super) struct Delta<T> {
    layer: Cow<Matrix<T>>,
    /// Approximate `layer.nvals()`, maintained by the mutation methods without
    /// GraphBLAS calls. `GrB_Matrix_nvals` on a delta forces its pending
    /// tuples to merge (an `O(|delta|)` sort/merge), so the fold policy reads
    /// this counter instead and [`Self::resync`] pins it to the exact value
    /// whenever something materializes the layer anyway.
    ///
    /// Atomic only because the read path latches fold decisions from `&self`.
    count: AtomicU64,
    /// `count` when this MVCC version was created ([`Self::new_version`]), or
    /// after the last fold: `count - tx_nvals` is what the current transaction
    /// has added — the batch size the fold policy weighs the fold cost
    /// against.
    tx_nvals: u64,
    /// Fold decision, latched here by `dup` / `wait` / `fold_*` and executed
    /// later by `flush` — which runs *before* the next mutation, the ideal
    /// moment, since folding there replaces the COW dup of both the delta and
    /// the base. The flag carries the decision across that gap (and across
    /// versions, via `new_version`).
    fold: AtomicBool,
    /// Which rows this layer may hold, so a single-row scan can skip it
    /// entirely. Conservative by construction; see [`RowFilter`].
    rows: RowFilter,
}

// Manual `Clone` so it holds for every `T` without a `T: Clone` bound (see the
// note on `Matrix`'s manual `Clone`). Shares the layer's GraphBLAS handle; use
// [`Delta::new_version`] to start a new MVCC version instead.
impl<T> Clone for Delta<T> {
    fn clone(&self) -> Self {
        self.relayer(self.layer.clone())
    }
}

impl<T> Deref for Delta<T> {
    type Target = Matrix<T>;

    fn deref(&self) -> &Matrix<T> {
        &self.layer
    }
}

impl<T> Delta<T> {
    /// Wrap a layer whose size is exactly known — a fresh empty matrix, or one
    /// just decoded. Pins it hypersparse, as every delta is.
    pub(super) fn new(layer: Matrix<T>) -> Self {
        let rows = if layer.nvals() == 0 {
            RowFilter::Empty
        } else {
            // A decoded layer arrives with entries this never saw inserted.
            RowFilter::Unknown
        };
        Self {
            count: AtomicU64::new(layer.nvals()),
            layer: Cow::new(layer.into_hyper()),
            tx_nvals: 0,
            fold: AtomicBool::new(false),
            rows,
        }
    }

    /// This delta, transposed: same bookkeeping over the transposed layer, since
    /// a transpose moves no entries.
    pub(super) fn transposed(&self) -> Self {
        let mut t = self.relayer(Cow::new(self.transpose().into_hyper()));
        // A transpose turns rows into columns, so the filter describes the
        // wrong axis and no cheap correction exists.
        t.rows = RowFilter::Unknown;
        t
    }

    /// This delta's bookkeeping over a different layer of the same size (a
    /// clone, a transpose).
    fn relayer(
        &self,
        layer: Cow<Matrix<T>>,
    ) -> Self {
        Self {
            layer,
            count: AtomicU64::new(self.count()),
            tx_nvals: self.tx_nvals,
            fold: AtomicBool::new(self.fold.load(Ordering::Relaxed)),
            // Same entries at the same coordinates; `transposed` overrides.
            rows: self.rows.clone(),
        }
    }

    /// Next MVCC version: the layer stays COW-shared, and the count so far
    /// becomes the new transaction's baseline, so `tx_added` measures only what
    /// that transaction goes on to add.
    pub(super) fn new_version(
        &self,
        fold: bool,
    ) -> Self {
        let count = self.count();
        Self {
            layer: self.layer.new_version(),
            count: AtomicU64::new(count),
            tx_nvals: count,
            fold: AtomicBool::new(fold),
            rows: self.rows.clone(),
        }
    }

    /// The approximate entry count the fold policy reads.
    pub(super) fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Materialize the layer and pin `count` to its exact size, bounding the
    /// drift the approximate counter has accumulated.
    pub(super) fn resync(&self) {
        self.layer.wait();
        self.count.store(self.layer.nvals(), Ordering::Relaxed);
    }

    /// Latch a fold decision. Latching is monotone within a version — a
    /// decision already taken is never revoked — so a `false` decision is
    /// simply nothing to record.
    pub(super) fn latch(
        &self,
        decision: bool,
    ) {
        if decision {
            self.fold.store(true, Ordering::Relaxed);
        }
    }

    /// What `policy` ([`should_fold`] or [`should_fold_read`]) decides for this
    /// layer against `base`, or-ed with any decision already latched. Reads the
    /// approximate counter — exact after a [`Self::resync`]. Pure: `dup`
    /// decides for the *next* version without disturbing this one.
    pub(super) fn fold_decision(
        &self,
        policy: fn(u64, u64, u64) -> bool,
        base: u64,
    ) -> bool {
        let count = self.count();
        self.fold.load(Ordering::Relaxed)
            || policy(count, count.saturating_sub(self.tx_nvals), base)
    }

    /// Whether a fold is currently latched — i.e. whether a `flush` armed now
    /// would have work to do on this layer.
    pub(super) fn folding(&self) -> bool {
        self.fold.load(Ordering::Relaxed)
    }

    /// Consume the latched decision: true when this layer must fold *and* has
    /// something to fold.
    pub(super) fn take_fold(&mut self) -> bool {
        self.fold.swap(false, Ordering::Relaxed) && self.layer.nvals() > 0
    }

    /// Swap in a fresh empty layer and zero the bookkeeping — after a fold, or
    /// at a dimension change. Emptying through the `Cow` instead would
    /// deep-copy a still-shared delta just to clear it.
    pub(super) fn clear(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) where
        T: MatrixType,
    {
        self.layer.replace(T::new_matrix(nrows, ncols).into_hyper());
        *self.count.get_mut() = 0;
        self.tx_nvals = 0;
        *self.fold.get_mut() = false;
        // The layer is empty again, which is the one state that can be asserted
        // rather than approximated.
        self.rows = RowFilter::Empty;
    }

    /// Swap in a layer holding the same entries at different dimensions (a
    /// grow), keeping the bookkeeping: a re-emit changes neither the entry count
    /// nor whether a fold is due. Named to line up with `m.replace`, which the
    /// grow path calls alongside it.
    pub(super) fn replace(
        &mut self,
        layer: Matrix<T>,
    ) {
        self.layer.replace(layer.into_hyper());
    }

    /// Resize the layer in place. A resize moves no entries between layers, so
    /// the bookkeeping stands.
    pub(super) fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) {
        // A resize moves no entry to a different row, so the filter stands —
        // and it must be restored explicitly, since `layer_mut` invalidates.
        let rows = std::mem::replace(&mut self.rows, RowFilter::Unknown);
        self.layer_mut().resize(nrows, ncols);
        self.rows = rows;
    }

    /// The layer, mutably, *without* counting the mutation. For bulk callers
    /// that resync `count` themselves (`remove_mask`) or that cannot change the
    /// entry count at all (`resize`).
    pub(super) fn layer_mut(&mut self) -> &mut Matrix<T> {
        // Every path that adds entries reaches the layer through here, so
        // invalidating by default is what makes the filter safe: a caller that
        // knows better says so (`insert`, `resize`), and one that says nothing
        // costs a scan rather than a wrong answer.
        self.rows = RowFilter::Unknown;
        &mut self.layer
    }

    /// The layer, mutably, recording that row `i` now holds an entry. For the
    /// counted single-entry inserts, which are the paths worth tracking.
    fn layer_mut_row(
        &mut self,
        i: u64,
    ) -> &mut Matrix<T> {
        self.rows.add(i);
        &mut self.layer
    }

    /// Whether this layer may hold an entry in rows `min_row..=max_row`. Only
    /// ever over-reports; see [`RowFilter`].
    pub(super) fn may_hold_rows(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> bool {
        self.rows.may_hold(min_row, max_row)
    }

    /// Drop `(i, j)` and un-count it. Saturates: the caller may not know
    /// whether the entry was there, and probing a shared, possibly-pending
    /// delta to find out is the mutation this whole counter exists to avoid.
    pub(super) fn erase(
        &mut self,
        i: u64,
        j: u64,
    ) {
        // A removal only shrinks the true row set, so the filter is still a
        // superset — `layer_mut_row` keeps it that way without invalidating.
        self.layer_mut_row(i).remove(i, j);
        let count = self.count.get_mut();
        *count = count.saturating_sub(1);
    }
}

// `Matrix::set` is element-typed (`impl Matrix<bool>` / `impl Matrix<u64>`),
// so the counted insert is too.
impl Delta<bool> {
    /// Add `(i, j)` and count it. A `bool` layer carries only a pattern — a
    /// stored `false` reads as absent to the valued masks these layers are used
    /// with — so there is no value to pass.
    pub(super) fn insert(
        &mut self,
        i: u64,
        j: u64,
    ) {
        self.layer_mut_row(i).set(i, j, true);
        *self.count.get_mut() += 1;
    }

    /// `self<mask> = mask ∩ base`: tombstone every committed entry the mask
    /// selects, then resync. Keeps the resync with the bulk write that
    /// invalidates the counter rather than leaving it to the caller.
    pub(super) fn tombstone_masked<TV>(
        &mut self,
        mask: &Matrix<bool>,
        base: &Matrix<TV>,
    ) {
        self.layer_mut()
            .element_wise_multiply(Some(mask), Some(mask), Some(base), None);
        self.resync();
    }
}

impl<T> Delta<T> {
    /// `self &= ¬mask`, then resync — the counterpart of
    /// [`Delta::tombstone_masked`] for the pending-add layer.
    pub(super) fn remove_all(
        &mut self,
        mask: &Matrix<bool>,
    ) {
        self.layer_mut().remove_all(mask);
        self.resync();
    }
}

impl Delta<u64> {
    /// Write `value` at `(i, j)` and count it.
    ///
    /// On a valued layer this doubles as the in-place *update* of a committed
    /// entry — the write *shadows* the live `m` entry with no `dm` mask (see the
    /// module docs) — so the count may overshoot by one per shadowed pair until
    /// the next [`Delta::resync`]. That is the drift the counter is allowed:
    /// the fold policy only needs the delta's order of magnitude, and probing
    /// the layer to tell an update from an insert would materialize it.
    pub(super) fn insert(
        &mut self,
        i: u64,
        j: u64,
        value: u64,
    ) {
        // No row filter here: `VersionedMatrix::iter` — the only thing that
        // reads one — exists for `bool` matrices alone, so tracking rows on a
        // valued delta would be a per-insert cost nothing can spend. `layer_mut`
        // pins the filter at `Unknown`, which is the answer that is always safe.
        self.layer_mut().set(i, j, value);
        *self.count.get_mut() += 1;
    }
}

/// A matrix with MVCC delta tracking for snapshot isolation.
///
/// Wraps a base matrix with separate matrices for tracking additions
/// and deletions, enabling concurrent reads during writes.
///
/// The type parameter `T` tags the element type of the *valued* layers (base
/// `m` and delta-plus `dp`): `bool` for pure structure/presence, `u64` for
/// valued matrices such as inline edge ids. The delta-minus `dm` is always a
/// `bool` deletion mask. `T` defaults to `bool`.
pub struct VersionedMatrix<T> {
    /// Base committed matrix
    m: Cow<Matrix<T>>,
    /// Delta-plus: edges added in current transaction
    dp: Delta<T>,
    /// Delta-minus: edges removed in current transaction (always a bool mask)
    dm: Delta<bool>,
    /// Whether the fold decisions latched on the deltas are executable *now*.
    /// Not derivable from those flags: [`wait`](Self::wait) deliberately
    /// latches a decision without arming it, so this is the bit that separates
    /// "decided" from "run it at the next mutation".
    needs_flush: AtomicBool,
}

// Manual `Clone` so it holds for every `V` without a `V: Clone` bound (see the
// note on `Matrix`'s manual `Clone`).
impl<T> Clone for VersionedMatrix<T> {
    fn clone(&self) -> Self {
        Self {
            m: self.m.clone(),
            dp: self.dp.clone(),
            dm: self.dm.clone(),
            needs_flush: AtomicBool::new(self.needs_flush.load(Ordering::Relaxed)),
        }
    }
}

unsafe impl<T> Send for VersionedMatrix<T> {}
unsafe impl<T> Sync for VersionedMatrix<T> {}

impl<T> VersionedMatrix<T> {
    /// Base committed matrix. Element values are typed by `T`; structural
    /// consumers (`ANY_PAIR` mxm, masks) may use it regardless of `T`.
    #[must_use]
    pub fn m(&self) -> &Matrix<T> {
        &self.m
    }

    /// Delta-plus matrix (pending additions), element values typed by `T`.
    #[must_use]
    pub fn dp(&self) -> &Matrix<T> {
        &self.dp
    }

    /// Delta-minus deletion mask (always `bool`).
    #[must_use]
    pub fn dm(&self) -> &Matrix<bool> {
        &self.dm
    }

    #[must_use]
    pub fn nrows(&self) -> u64 {
        self.m.nrows()
    }

    #[must_use]
    pub fn ncols(&self) -> u64 {
        self.m.ncols()
    }

    /// The committed base `m` never holds real pending work: it is only
    /// mutated by `flush` (which waits) and `resize` (GrB_Matrix_resize waits
    /// internally), so it is never waited here. `GxB_WILL_WAIT` can still
    /// report true on `m` after a grow-resize — the hyper hash was freed —
    /// but GraphBLAS rebuilds it on demand.
    pub fn wait(&self) {
        // nvals only changes through ops that mark the matrix pending, so the
        // flag can only be stale when there is pending work — skip the nvals
        // FFI calls on the hot read path otherwise.
        if self.dp.is_synced() && self.dm.is_synced() {
            return;
        }
        // `resync` materializes each delta and pins its approximate counter to
        // the exact count, bounding any drift it accumulated — so the policy
        // below weighs exact sizes rather than the counters `dup` has to
        // settle for.
        self.dp.resync();
        self.dm.resync();
        let base = self.m.nvals();
        self.dp.latch(self.dp.fold_decision(should_fold_read, base));
        self.dm.latch(self.dm.fold_decision(should_fold_read, base));
        // Deliberately NOT setting needs_flush: executing a fold mid-tx is
        // pathological — a create+delete transaction folds its own pending
        // adds into the base right before deleting them, leaving the base
        // full of stale entries and dm full of tombstones that later
        // transactions erode one zombie at a time (measured 18-63x on small
        // creates after `write 1m`). The latched decision is carried into
        // the next version by `dup`, which does set needs_flush, so the fold
        // runs before the next transaction's first mutation instead.
    }

    /// Materialize only the committed base `m`.
    ///
    /// Read paths access `m` raw (no lock, no wait): `m()` hands the base to
    /// structural consumers, and [`Self::wait`] reads `m.nvals()` directly.
    /// A GrB call on a pending matrix finishes that work internally — a
    /// mutation — so a published snapshot whose base is pending lets two
    /// lock-free readers corrupt GrB state. dp/dm don't need this: every
    /// read of them goes through the mutex-guarded [`Matrix::wait`] first.
    /// Called at MVCC commit; a no-op (one atomic load) when `m` is synced.
    pub fn wait_base(&self) {
        self.m.wait();
    }

    /// Wait on all three internal matrices (m, dp, dm).
    /// Used for fork safety — ensures no GrB internal locks are held.
    pub fn wait_all(&self) {
        self.m.wait();
        self.dp.wait();
        self.dm.wait();
    }

    /// Returns true if every internal matrix has no pending GraphBLAS
    /// operations — i.e. wait_all was effective.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        self.m.is_synced() && self.dp.is_synced() && self.dm.is_synced()
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.m.memory_usage() + self.dp.memory_usage() + self.dm.memory_usage()
    }

    /// Materialize the effective structure as a `bool` matrix: `(m - dm) ∪ dp`,
    /// values discarded. Works for both bool and valued (uint64) bases — only
    /// structure is preserved, which is all the structure-only consumers
    /// (traversal, relationship-matrix building) need.
    #[must_use]
    pub fn extract(&self) -> Matrix<bool> {
        self.wait();
        let mut m = Matrix::<bool>::new(self.m.nrows(), self.m.ncols());
        m.set_pattern(None, &*self.m, None);
        if self.dm.nvals() > 0 {
            m.remove_all(&self.dm);
        }
        if self.dp.nvals() > 0 {
            m.set_pattern(None, &*self.dp, None);
        }
        m
    }

    /// Effective entry count: `|m| + |dp| − |dm|`.
    ///
    /// Relies on the invariants maintained by [`Self::set`] /
    /// [`Self::remove`]: `dm ⊆ m` and `dp ∩ (m ∖ dm) = ∅` (bool matrices
    /// keep the stricter `dp ∩ m = ∅`; `u64` matrices mask any in-place
    /// update in `dm`).
    #[must_use]
    pub fn nvals(&self) -> u64 {
        self.wait();
        self.m.nvals() + self.dp.nvals() - self.dm.nvals()
    }

    pub fn print(
        &self,
        level: GxB_Print_Level,
    ) {
        self.m.print(level);
        self.dp.print(level);
        self.dm.print(level);
    }
}

impl VersionedMatrix<bool> {
    #[must_use]
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> Iter {
        self.wait();
        Iter::<BoolExtract>::new(self, min_row, max_row)
    }

    pub fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) {
        if nrows < self.m.nrows() || ncols < self.m.ncols() {
            // Shrinking can drop entries; keep the straightforward path.
            self.flush();
            // GrB_Matrix_resize waits internally, so `m` holds no real
            // pending work afterwards — waiting here would rebuild the freed
            // hyper hash on every capacity change (measured 1.4-5.7x write
            // regressions).
            self.m.resize(nrows, ncols);
            // A resize moves no entries between layers, so the delta counters
            // stand (a shrink that drops entries only adds to the drift
            // `resync` bounds).
            self.dp.resize(nrows, ncols);
            self.dm.resize(nrows, ncols);
            return;
        }
        // Growing: the base is COW-shared with the committed snapshot, so an
        // in-place resize deep-copies the whole matrix first, and any
        // lingering delta later pays another O(|m|) fold pass (measured
        // 2.4-9.8 ms spikes per capacity grow on bulk creates). Instead
        // rebuild the base at the target dims in one pass, folding both
        // deltas in: merge (m \ dm) ∪ dp into a fresh matrix.
        //
        // The layers may still be shared with the committed snapshot AND
        // carry pending GraphBLAS work (commit does not wait). Iterating
        // materializes pending work — a mutation — so it must go through the
        // lock-coordinated wait() first or it races concurrent readers
        // (observed as GrB_INVALID_OBJECT / heap corruption under stress).
        self.wait_all();
        // With nothing to fold there is no merge to do, only a copy of `m` at
        // the new dims, which `grown` does as a `dup` plus a `GrB_Matrix_resize`
        // — 0.47 ms against the tuple round-trip's 3.7 ms at 262k entries and
        // 0.81 ms against 9.0 ms at 1m (`grow_cost_rebuild_vs_resize`). This is
        // the common shape, since a grow typically follows a commit that already
        // folded.
        if self.dp.nvals() == 0 && self.dm.nvals() == 0 {
            let new_m = self.m.grown(nrows, ncols);
            new_m.wait();
            self.m.replace(new_m);
            // The counters are approximate and may have drifted; both deltas
            // are provably empty here, so `clear` pins them to the truth
            // rather than leaving a phantom delta driving the fold policy.
            self.clear_deltas(nrows, ncols);
            return;
        }
        // Streamed with row iterators rather than `extract_tuples` per layer:
        // all three layers yield `(row, col)` in row-major order, which is
        // already the sorted order `build` wants, so the merge needs no
        // per-layer tuple arrays at all — only the single output pair. That
        // takes peak extra memory from `2·(|m| + |dp| + |dm|) + 2·n` u64s down
        // to `2·n`, roughly halving it on a base-dominated matrix.
        let n = (self.m.nvals() + self.dp.nvals()) as usize;
        let mut ri = Vec::with_capacity(n);
        let mut rj = Vec::with_capacity(n);
        let mut base = self.m.iter(0, u64::MAX).peekable();
        let mut adds = self.dp.iter(0, u64::MAX).peekable();
        let mut tombs = self.dm.iter(0, u64::MAX).peekable();
        loop {
            // Next surviving base entry: skip any the tombstone stream covers.
            // `dm ⊆ m` and both are sorted, so `tombs` only ever advances
            // toward the current base key and never rewinds.
            let next_base = loop {
                let Some(&bk) = base.peek() else { break None };
                while tombs.peek().is_some_and(|&t| t < bk) {
                    tombs.next();
                }
                if tombs.peek() == Some(&bk) {
                    tombs.next();
                    base.next();
                    continue;
                }
                break Some(bk);
            };
            // `dp ∩ m = ∅` for bool layers, so the two streams never collide
            // on a key and a plain two-way merge emits the union in order.
            let take_add = match (next_base, adds.peek()) {
                (None, None) => break,
                (None, Some(_)) => true,
                (Some(_), None) => false,
                (Some(bk), Some(&ak)) => ak < bk,
            };
            let (r, c) = if take_add {
                adds.next().unwrap_or_else(|| unreachable!())
            } else {
                base.next().unwrap_or_else(|| unreachable!())
            };
            ri.push(r);
            rj.push(c);
        }
        drop((base, adds, tombs));
        let mut new_m = Matrix::<bool>::new(nrows, ncols);
        new_m.build(&ri, &rj);
        new_m.wait();
        self.m.replace(new_m);
        // Both deltas are folded into the new base above.
        self.clear_deltas(nrows, ncols);
    }

    /// Swap in fresh empty deltas at the given dims and drop all their
    /// bookkeeping — for the grow paths, which have just folded (or proved
    /// empty) both layers.
    fn clear_deltas(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) {
        self.dp.clear(nrows, ncols);
        self.dm.clear(nrows, ncols);
        self.needs_flush.store(false, Ordering::Relaxed);
    }

    /// Mark `(i, j)` deleted, or undo a pending add.
    ///
    /// Reads only the committed base — never the deltas. `dp ∩ m = ∅` is what
    /// makes that sound: a pair live in `m` cannot also sit in `dp`, so the
    /// `m` branch needs no `dp` probe. The invariant is covered by
    /// `delta_invariants_hold_across_mutation_sequences` rather than a
    /// `debug_assert`, which would have to materialize the (possibly shared,
    /// possibly pending) `dp` to read it — making debug builds wait where
    /// release does not.
    pub fn remove(
        &mut self,
        i: u64,
        j: u64,
    ) {
        self.flush();
        if self.m.get(i, j).is_some() {
            self.dm.insert(i, j);
        } else {
            self.dp.erase(i, j);
        }
    }

    /// Bulk-remove all entries matching a mask matrix.
    ///
    /// Equivalent to calling `remove(i, j)` for every entry `(i, j)` in `mask`,
    /// but executes in two GraphBLAS bulk operations instead of N individual calls:
    /// - Entries in base `m` matching `mask` are marked deleted in `dm`
    /// - Entries in delta-plus `dp` matching `mask` are removed from `dp`
    pub fn remove_mask(
        &mut self,
        mask: &Matrix<bool>,
    ) {
        self.flush();
        // eWiseMult below reads `m`, which may be shared with the committed
        // snapshot and carry pending work; reading materializes it, so wait
        // first under the readers' lock.
        self.m.wait();
        // dm<mask> = mask ∩ m: mark deleted every committed entry that `mask`
        // selects. eWiseMult computes over the pattern intersection, so the
        // cost scales with the smaller operand (the mask), not with |m|.
        // Existing dm entries survive: outside the mask they are untouched,
        // and inside the mask dm ⊆ m guarantees they are in the intersection.
        self.dm.tombstone_masked(mask, &self.m);
        // dp &= ~mask: remove entries from dp that exist in mask
        self.dp.remove_all(mask);
    }

    #[must_use]
    pub fn get(
        &self,
        i: u64,
        j: u64,
    ) -> Option<bool> {
        self.wait();
        self.m.get(i, j).map_or_else(
            || self.dp.get(i, j),
            |value| {
                if self.dm.get(i, j).is_some() {
                    None
                } else {
                    Some(value)
                }
            },
        )
    }

    /// Add `(i, j)`, or undo a pending delete.
    ///
    /// Like [`Self::remove`], reads only the committed base: `dp ∩ m = ∅`
    /// covers the `m` branch and `dm ⊆ m` the other. Both are covered by
    /// `delta_invariants_hold_across_mutation_sequences`, not by
    /// `debug_assert`s — see [`Self::remove`] for why probing the deltas here
    /// would diverge debug from release.
    pub fn set(
        &mut self,
        i: u64,
        j: u64,
        value: bool,
    ) {
        self.flush();
        if self.m.get(i, j).is_some() {
            self.dm.erase(i, j);
        } else {
            debug_assert!(value, "bool layers store presence only");
            self.dp.insert(i, j);
        }
    }

    #[must_use]
    pub fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        Self {
            m: Cow::new(Matrix::<bool>::new(nrows, ncols)),
            dp: Delta::new(Matrix::<bool>::new(nrows, ncols)),
            dm: Delta::new(Matrix::<bool>::new(nrows, ncols)),
            needs_flush: AtomicBool::new(false),
        }
    }

    /// Wrap an owned `Matrix` as a `VersionedMatrix` with empty delta-plus /
    /// delta-minus.  Used when callers materialize a merged matrix and then
    /// want to expose it through the versioned-matrix iter API without the
    /// dup overhead of re-building inside the versioned wrapper.
    #[must_use]
    pub fn from_matrix(m: Matrix<bool>) -> Self {
        // Freshly merged matrices (e.g. `set_pattern` unions) may carry
        // pending GraphBLAS work; the base slot is required to be synced
        // (`wait` debug-asserts `!m.pending()`).
        m.wait();
        let nrows = m.nrows();
        let ncols = m.ncols();
        Self {
            m: Cow::new(m),
            dp: Delta::new(Matrix::<bool>::new(nrows, ncols)),
            dm: Delta::new(Matrix::<bool>::new(nrows, ncols)),
            needs_flush: AtomicBool::new(false),
        }
    }

    pub fn flush(&mut self) {
        if self.needs_flush.load(Ordering::Relaxed) {
            // The layers may be shared with the committed snapshot and carry
            // pending work; nvals/eWiseAdd/select below materialize it (a
            // mutation), so wait first under the readers' lock.
            self.wait_all();
            let fold_dp = self.dp.take_fold();
            let fold_dm = self.dm.take_fold();
            if fold_dp || fold_dm {
                let nrows = self.m.nrows();
                let ncols = self.m.ncols();
                // Always build the folded base into a fresh matrix and swap
                // it in. When `m` is shared with the committed snapshot an
                // in-place fold would deep-copy it (a full O(|m|) memcpy)
                // first; when it is not, GraphBLAS materializes the eWiseAdd
                // result in a temporary anyway, so the fresh build costs the
                // same — and under full MVCC the base is always shared.
                let mut new_m = Matrix::<bool>::new(nrows, ncols);
                match (fold_dp, fold_dm) {
                    // new_m<!dm, replace> = m ∪ dp: REPLACE drops the dm
                    // entries, the complemented mask lets everything else
                    // through (dp ∩ dm = ∅, so no pending add is lost).
                    (true, true) => new_m.element_wise_add(
                        Some(&self.dm),
                        Some(&self.m),
                        Some(&*self.dp),
                        Some(matrix::Descriptor::RC),
                    ),
                    (true, false) => {
                        new_m.element_wise_add(None, Some(&self.m), Some(&*self.dp), None);
                    }
                    // new_m<!dm, replace> = m
                    (false, true) => new_m.select(&self.dm, &self.m),
                    (false, false) => unreachable!(),
                }
                new_m.wait();
                self.m.replace(new_m);
                if fold_dp {
                    self.dp.clear(nrows, ncols);
                }
                if fold_dm {
                    self.dm.clear(nrows, ncols);
                }
            }
            self.needs_flush.store(false, Ordering::Relaxed);
        }
    }

    /// Latch the fold decision from the current (materialized) delta sizes
    /// and execute it immediately. Only safe once a transaction has finished
    /// mutating — e.g. the end of a GRAPH.BULK command — where the mid-tx
    /// pathology `wait` guards against (folding a transaction's own pending
    /// adds right before it deletes them) cannot occur. Without this,
    /// the decision latched by `wait` only runs at the *next* version's
    /// first mutation (via `dup` → `flush`), so the final bulk command's
    /// deltas would stay unfolded.
    pub fn fold_latched(&mut self) {
        self.wait();
        if *self.dp.fold.get_mut() || *self.dm.fold.get_mut() {
            self.needs_flush.store(true, Ordering::Relaxed);
            self.flush();
        }
    }

    /// Fold a delta that has grown comparable to the base
    /// ([`delta_dominates_base`]), at MVCC commit: after the transaction's
    /// last mutation, while the writer still holds the write lock and before
    /// the new version is published.
    ///
    /// Deferring *this* fold to the next mutation's `flush` — where the fold
    /// usefully replaces a COW dup — keeps both the base and a base-sized
    /// delta resident until some later transaction happens to touch the same
    /// matrix, which for a delete-everything is unbounded: `MATCH (n) DELETE
    /// n` over 250k nodes / 500k edges left the relationship tensor holding
    /// `m = 499_998` alongside `dm = 499_998` (GRAPH.MEMORY USAGE 41 MB after
    /// the delete against 25 MB before it).
    ///
    /// Sub-hatch deltas stay lazy. The sqrt balance point is a throughput
    /// decision whose whole value is landing on the `dup`, and those deltas
    /// are by definition small next to the base they shadow.
    pub fn fold_oversized(&mut self) {
        let base = self.m.nvals();
        // Only the escape hatch arms the flush here. Latching alone must not:
        // a decision `wait` left latched is deliberately deferred to the next
        // version, and executing it here would be the mid-tx fold that
        // deferral exists to prevent.
        let oversized_dp = delta_dominates_base(self.dp.count(), base);
        let oversized_dm = delta_dominates_base(self.dm.count(), base);
        if oversized_dp || oversized_dm {
            self.dp.latch(oversized_dp);
            self.dm.latch(oversized_dm);
            self.needs_flush.store(true, Ordering::Relaxed);
            self.flush();
        }
    }

    /// Set multiple entries, checking dm emptiness once upfront.
    ///
    /// If dm is empty, uses a batched path that skips per-entry dm handling;
    /// otherwise falls back to the full `set` path. Both paths keep the
    /// invariant `dp ∩ m = ∅`: an entry already committed in `m` must not be
    /// re-added to `dp` (a fold or grow-resize may have moved it there since
    /// the caller last saw it), or a later `remove` finds it in both and the
    /// `nvals`/iter arithmetic double-counts it.
    ///
    /// `NEW` asserts the caller's entries are never live in the committed base
    /// (fresh entity ids: a reclaimed id's stale base entry always has a `dm`
    /// tombstone, which routes to the safe per-entry path here). It skips the
    /// per-entry `m` lookup that otherwise keeps `dp ∩ m = ∅`; callers whose
    /// entries can already be committed (e.g. the adjacency matrix, where two
    /// edges share one pair) must pass `false`. Probing `m` is safe either way
    /// (unlike a delta, `m` is never pending — see [`Self::wait`]), so the
    /// skipped check survives as a `debug_assert` rather than being dropped for
    /// the reasons [`Self::remove`] documents.
    pub fn set_all<const NEW: bool>(
        &mut self,
        entries: impl Iterator<Item = (u64, u64)>,
    ) {
        self.flush();
        // `flush` is a no-op unless a fold was latched, so it guarantees
        // nothing on return. The nvals below reads the shared dm raw, and
        // nvals on a pending matrix merges its pending tuples internally (a
        // mutation), racing concurrent readers on the committed snapshot.
        // Materialize through the mutex-guarded wait first — an atomic load
        // when already synced.
        self.dm.wait();
        if self.dm.nvals() == 0 {
            for (i, j) in entries {
                if NEW {
                    debug_assert!(
                        self.m.get(i, j).is_none(),
                        "set_all::<true> on ({i}, {j}), which is live in the committed base"
                    );
                } else if self.m.get(i, j).is_some() {
                    continue;
                }
                self.dp.insert(i, j);
            }
        } else {
            for (i, j) in entries {
                self.set(i, j, true);
            }
        }
    }
}

impl<T> Dup<Self> for VersionedMatrix<T> {
    /// `dup` creates the next write version — the one reliable pre-mutation
    /// hook in write-only workloads (nothing calls `wait` there). The
    /// just-finished transaction's contribution (`*_count - *_tx_nvals`)
    /// predicts the next one's batch size, so the fold decision is made here
    /// and latched for the new version's `flush` to execute before its first
    /// mutation — folding there also replaces the COW dup of delta and base.
    ///
    /// Delta sizes come from the approximate counters: reading `nvals` here
    /// would force each mutated delta's pending tuples to merge, an
    /// `O(|delta|)` tax on every small write transaction (measured +35%
    /// instructions on single-node creates). `m` is never pending, so its
    /// `nvals` is a cheap field read.
    fn dup(&self) -> Self {
        let base = self.m.nvals();
        let fold_dp = self.dp.fold_decision(should_fold, base);
        let fold_dm = self.dm.fold_decision(should_fold, base);
        Self {
            m: self.m.new_version(),
            dp: self.dp.new_version(fold_dp),
            dm: self.dm.new_version(fold_dm),
            needs_flush: AtomicBool::new(fold_dp || fold_dm),
        }
    }
}

impl VersionedMatrix<bool> {
    /// Transposes the matrix.
    ///
    /// # Returns
    /// A new matrix that is the transpose of the original.
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            m: Cow::new(self.m.transpose()),
            // A transpose moves no entries, so each delta keeps its
            // bookkeeping verbatim over the transposed layer.
            dp: self.dp.transposed(),
            dm: self.dm.transposed(),
            needs_flush: AtomicBool::new(self.needs_flush.load(Ordering::Relaxed)),
        }
    }
}

impl<V> Encode<19> for VersionedMatrix<V> {
    fn encode(
        &self,
        w: &mut dyn Writer,
    ) {
        self.m.encode(w);
        self.dp.encode(w);
        self.dm.encode(w);
    }
}

impl<V> Decode<19> for VersionedMatrix<V> {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        let m = Matrix::<V>::decode(r)?;
        let dp = Matrix::<V>::decode(r)?;
        let dm = Matrix::<bool>::decode(r)?;
        // Decoded deltas have no owning transaction; `tx_nvals` starts at 0, so
        // the fold policy treats the whole delta as freshly added and sees it
        // on the first flush.
        let base = m.nvals();
        let dp = Delta::new(dp);
        let dm = Delta::new(dm);
        dp.latch(dp.fold_decision(should_fold, base));
        dm.latch(dm.fold_decision(should_fold, base));
        let needs_flush = dp.folding() || dm.folding();
        Ok(Self {
            m: Cow::new(m),
            dp,
            dm,
            needs_flush: AtomicBool::new(needs_flush),
        })
    }
}

pub struct Iter<E: IterExtract = BoolExtract> {
    mit: matrix::Iter<E>,
    /// Lookahead on `mit` (buffered while waiting for the merge to pick it).
    m_next: Option<E::Item>,
    /// Delta-plus iterator. Lazily left `None` when `dp` is empty (the common
    /// read-only hot path on a freshly loaded graph) so we skip allocating and
    /// freeing a `GxB_Iterator` that would never yield anything. `dp` is a
    /// stable read snapshot for the life of this iterator, so once `None` it
    /// stays `None` across `seek` calls.
    dpit: Option<matrix::Iter<E>>,
    /// Lookahead on `dpit`.
    dp_next: Option<E::Item>,
    /// Delta-minus iterator, `None` when `dm` is empty (same rationale as
    /// `dpit`).
    dmit: Option<matrix::Iter<BoolExtract>>,
    /// Lookahead on `dmit`.
    dm_next: Option<(u64, u64)>,
}

unsafe impl<E: IterExtract> Send for Iter<E> {}
unsafe impl<E: IterExtract> Sync for Iter<E> {}

impl<E: IterExtract> Iter<E> {
    /// Streams the effective content `(m ∖ dm) ∪ dp` as a sorted merge of the
    /// three layer iterators. Valid for a `VersionedMatrix` of any element
    /// type when `E = BoolExtract` (only the sparsity pattern is read).
    fn new<V>(
        vm: &VersionedMatrix<V>,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        // A layer whose row filter rules this range out is left *detached*: no
        // `GxB_Iterator` is allocated, attached or seeked for it, and it yields
        // nothing. It is still carried, so a later `seek` to a range the filter
        // does not rule out attaches it and reads normally — the skip is an
        // optimisation, not a promise the caller has to keep.
        Self::from_layers_detaching(
            &vm.m,
            &vm.dp,
            !vm.dp.may_hold_rows(min_row, max_row),
            &vm.dm,
            !vm.dm.may_hold_rows(min_row, max_row),
            min_row,
            max_row,
        )
    }

    /// Build the effective-content iterator directly from the three delta
    /// layers, for owners that manage `m`/`dp`/`dm` themselves (e.g.
    /// `Tensor`'s forward adjacency). Callers must have waited `dp`/`dm`.
    /// When `dp` shadows a committed `m` pair (an in-place update with no
    /// `dm` mask), the merge yields the live `dp` value and skips the `m`
    /// entry.
    pub(crate) fn from_layers<V>(
        m: &Matrix<V>,
        dp: &Matrix<V>,
        dm: &Matrix<bool>,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        Self::from_layers_detaching(m, dp, false, dm, false, min_row, max_row)
    }

    /// As [`Self::from_layers`], but a layer flagged `detach` is one the caller
    /// has established holds nothing in `min_row..=max_row`. It yields nothing
    /// now and attaches on the first [`Self::seek`], so the stream is identical
    /// either way and only the cost differs.
    fn from_layers_detaching<V>(
        m: &Matrix<V>,
        dp: &Matrix<V>,
        detach_dp: bool,
        dm: &Matrix<bool>,
        detach_dm: bool,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        let mut dmit = if dm.nvals() == 0 {
            None
        } else if detach_dm {
            Some(matrix::Iter::<BoolExtract>::detached(dm))
        } else {
            Some(matrix::Iter::<BoolExtract>::new(dm, min_row, max_row))
        };
        let dm_next = dmit.as_mut().and_then(Iterator::next);
        Self {
            mit: matrix::Iter::new(m, min_row, max_row),
            m_next: None,
            dpit: if dp.nvals() == 0 {
                None
            } else if detach_dp {
                Some(matrix::Iter::detached(dp))
            } else {
                Some(matrix::Iter::new(dp, min_row, max_row))
            },
            dp_next: None,
            dmit,
            dm_next,
        }
    }

    /// Re-seek the inner GraphBLAS iterators to a new row range without
    /// re-allocating them. Hot-loop callers (e.g. `CondTraverseOp` and
    /// `ExpandInto` looking up edges by `(src, dst)`) use this to amortize
    /// the per-pair iterator allocation.
    pub fn seek(
        &mut self,
        min_row: u64,
        max_row: u64,
    ) {
        self.mit.seek(min_row, max_row);
        self.m_next = None;
        if let Some(dpit) = &mut self.dpit {
            dpit.seek(min_row, max_row);
        }
        self.dp_next = None;
        if let Some(dmit) = &mut self.dmit {
            dmit.seek(min_row, max_row);
            self.dm_next = dmit.next();
        }
    }
}

/// Three-way sorted merge over the layer iterators. GraphBLAS row iterators
/// yield entries in ascending `(row, col)` order, so the output is sorted by
/// `(row, col)` too. `m` entries masked by `dm` are dropped; when `dp` holds
/// the same pair as `m` (shadow), the live `dp` item wins.
impl<E: IterExtract> Iterator for Iter<E> {
    type Item = E::Item;

    fn next(&mut self) -> Option<E::Item> {
        if self.m_next.is_none() {
            self.m_next = self.mit.next();
        }
        // Skip m entries deleted by dm (both streams ascending, dm ⊆ m).
        while let Some(m) = &self.m_next {
            let mp = E::pos(m);
            while let Some(dm) = self.dm_next {
                if dm < mp {
                    self.dm_next = self.dmit.as_mut().and_then(Iterator::next);
                } else {
                    break;
                }
            }
            if self.dm_next == Some(mp) {
                self.dm_next = self.dmit.as_mut().and_then(Iterator::next);
                self.m_next = self.mit.next();
            } else {
                break;
            }
        }
        if self.dp_next.is_none() {
            self.dp_next = self.dpit.as_mut().and_then(Iterator::next);
        }
        match (&self.m_next, &self.dp_next) {
            (Some(m), Some(dp)) => {
                let mp = E::pos(m);
                let dpp = E::pos(dp);
                if dpp <= mp {
                    if dpp == mp {
                        self.m_next = None; // shadowed; dp yields the live value
                    }
                    self.dp_next.take()
                } else {
                    self.m_next.take()
                }
            }
            (Some(_), None) => self.m_next.take(),
            (None, _) => self.dp_next.take(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::super::matrix::{Dup, Matrix};
    use super::super::test_init::ensure_init;
    use super::{
        MIN_FOLD_DELTA, READ_FOLD_K, VersionedMatrix, WRITE_FOLD_K, should_fold, should_fold_read,
    };

    /// Smallest delta that satisfies the sqrt rule, i.e. `ceil(sqrt(k · tx))`.
    fn threshold(
        k: u64,
        tx_added: u64,
    ) -> u64 {
        let target = k * tx_added;
        (1..).find(|d| d * d >= target).unwrap()
    }

    /// A base big enough that the `2·|delta| ≥ base_nvals` escape hatch never
    /// fires, so these cases exercise the sqrt rule alone.
    const HUGE_BASE: u64 = u64::MAX / 4;

    #[test]
    fn read_path_balance_point_is_flat_in_base_size() {
        assert_eq!(threshold(READ_FOLD_K, 1), 287);
        // The measured fold cost is independent of the base, so the same
        // threshold must hold from a 1m-entry base to a 100m-entry one.
        for base in [1_000_000, 10_000_000, 100_000_000, HUGE_BASE] {
            assert!(!should_fold_read(286, 1, base), "base {base}");
            assert!(should_fold_read(287, 1, base), "base {base}");
        }
    }

    #[test]
    fn write_path_is_16x_looser_than_read_path() {
        // w_dup is 250x cheaper than w_merge, and D* scales as 1/sqrt(w), so
        // the write path folds at sqrt(250) ~= 15.8x the read path's delta.
        assert_eq!(threshold(WRITE_FOLD_K, 1), 4_528);
        assert_eq!(threshold(WRITE_FOLD_K, 1) / threshold(READ_FOLD_K, 1), 15);
        assert!(!should_fold(4_527, 1, HUGE_BASE));
        assert!(should_fold(4_528, 1, HUGE_BASE));
    }

    #[test]
    fn balance_point_grows_as_sqrt_of_transaction_size() {
        // 100x the transaction size buys only a ~10x bigger delta.
        assert_eq!(threshold(READ_FOLD_K, 1), 287);
        assert_eq!(threshold(READ_FOLD_K, 100), 2_864);
        for tx_added in [1_u64, 10, 100, 1_000] {
            let d = threshold(READ_FOLD_K, tx_added);
            assert!(
                !should_fold_read(d - 1, tx_added, HUGE_BASE),
                "tx {tx_added}"
            );
            assert!(should_fold_read(d, tx_added, HUGE_BASE), "tx {tx_added}");
        }
    }

    #[test]
    fn delta_comparable_to_base_always_folds() {
        // The escape hatch: a one-shot bulk transaction's huge `tx_added`
        // defeats the sqrt term, so without this a base-sized delta would tax
        // every later transaction.
        assert!(should_fold(512, u64::MAX, 1_024));
        assert!(should_fold_read(512, u64::MAX, 1_024));
    }

    #[test]
    fn tiny_deltas_and_read_only_transactions_never_fold() {
        assert!(!should_fold(MIN_FOLD_DELTA - 1, 1, 0));
        assert!(!should_fold_read(MIN_FOLD_DELTA - 1, 1, 0));
        // `tx_added == 0` is a transaction that added nothing to this layer.
        assert!(!should_fold(u64::MAX, 0, 1_024));
        assert!(!should_fold_read(u64::MAX, 0, 1_024));
    }

    const DIM: u64 = 512;

    /// Check the two delta invariants `set`/`remove` rely on, plus the derived
    /// state that breaks when they don't hold.
    ///
    /// `set`/`remove` branch on the committed base alone and never probe the
    /// deltas, which is only sound while:
    ///
    /// * `dp ∩ m = ∅` — nothing lives in both the base and the pending adds,
    ///   so the `m` branch cannot be shadowing a `dp` entry it fails to clear;
    /// * `dm ⊆ m` — a tombstone only ever masks a committed entry, so the
    ///   `dp` branch cannot be adding a pair that a tombstone still hides.
    ///
    /// Together they are what makes `nvals`'s `|m| + |dp| − |dm|` arithmetic
    /// and `Iter`'s three-way merge correct, so both are asserted too: a
    /// violation that the raw layer checks somehow miss still surfaces as a
    /// wrong count or a wrong effective state.
    fn assert_invariants(
        v: &VersionedMatrix<bool>,
        model: &BTreeSet<(u64, u64)>,
    ) {
        // Reading the layers raw materializes them; the production write paths
        // deliberately avoid doing so, which is why this lives in a test.
        v.wait_all();
        for (i, j) in v.dp().iter(0, u64::MAX) {
            assert!(
                v.m().get(i, j).is_none(),
                "dp ∩ m ≠ ∅ at ({i}, {j}): the `m` branch of set/remove would miss the dp entry"
            );
            assert!(v.dm().get(i, j).is_none(), "dp ∩ dm ≠ ∅ at ({i}, {j})");
        }
        for (i, j) in v.dm().iter(0, u64::MAX) {
            assert!(
                v.m().get(i, j).is_some(),
                "dm ⊄ m at ({i}, {j}): a tombstone with no committed entry to mask"
            );
        }
        assert_eq!(
            v.nvals(),
            model.len() as u64,
            "|m| + |dp| - |dm| disagrees with the effective entry count"
        );
        let effective: BTreeSet<(u64, u64)> = v.iter(0, u64::MAX).collect();
        assert_eq!(&effective, model, "effective state diverged from the model");
    }

    /// Deterministic LCG — the sequence must be reproducible so a failure is
    /// replayable.
    fn next_rand(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    /// Drive `set`/`remove`/`set_all`/`remove_mask` through every state
    /// transition — including the ones only reachable after a fold has moved
    /// pending adds into the committed base — and check the invariants after
    /// each step.
    ///
    /// This replaces the `debug_assert!`s that used to sit in the `set`/
    /// `remove` branches. Those could only be evaluated by probing `dp`/`dm`,
    /// and probing a possibly-shared, possibly-pending delta materializes it —
    /// a mutation racing lock-free readers. Guarding them with
    /// `#[cfg(debug_assertions)] if is_shared() { wait() }` made debug builds
    /// wait where release builds don't, so the write path being tested was not
    /// the write path being shipped.
    #[test]
    fn delta_invariants_hold_across_mutation_sequences() {
        ensure_init();
        let mut v = VersionedMatrix::<bool>::new(DIM, DIM);
        let mut model = BTreeSet::new();
        let mut rng = 0x5eed_1234_u64;
        // A keyspace far smaller than the matrix, so set-after-set,
        // remove-after-remove and set-after-remove on the same pair are the
        // common case rather than a rarity.
        let key = |r: u64| ((r % 24) * 7, (r / 24 % 24) * 11);

        for step in 0..4_000 {
            match next_rand(&mut rng) % 16 {
                // Bulk add: `set_all` routes through `set` once `dm` is
                // non-empty and takes its own fast path otherwise, so both
                // arms need covering.
                0 => {
                    let batch: Vec<(u64, u64)> =
                        (0..16).map(|_| key(next_rand(&mut rng))).collect();
                    v.set_all::<false>(batch.iter().copied());
                    model.extend(batch);
                }
                // Bulk delete through the two-GraphBLAS-op mask path.
                1 => {
                    let batch: BTreeSet<(u64, u64)> =
                        (0..16).map(|_| key(next_rand(&mut rng))).collect();
                    let rows: Vec<u64> = batch.iter().map(|&(i, _)| i).collect();
                    let cols: Vec<u64> = batch.iter().map(|&(_, j)| j).collect();
                    let mut mask = Matrix::<bool>::new(DIM, DIM);
                    mask.build(&rows, &cols);
                    mask.wait();
                    v.remove_mask(&mask);
                    for k in &batch {
                        model.remove(k);
                    }
                }
                // Version boundary: latches the fold decision, which the next
                // mutation's `flush` executes — moving `dp` into `m` and
                // clearing `dm`, the transition the `m` branches exist for.
                2 => v = v.dup(),
                // Read path: resyncs the counters and latches the read-path
                // fold decision.
                3 => {
                    v.wait();
                }
                4 => v.fold_oversized(),
                // Deletes, biased to roughly a third of single-entry ops so
                // the base keeps growing and `dm` keeps getting exercised.
                5..=8 => {
                    let k = key(next_rand(&mut rng));
                    v.remove(k.0, k.1);
                    model.remove(&k);
                }
                _ => {
                    let k = key(next_rand(&mut rng));
                    v.set(k.0, k.1, true);
                    model.insert(k);
                }
            }
            // The full check materializes the layers, so run it on a stride
            // rather than every step; the boundary steps are covered by the
            // final check below.
            if step % 37 == 0 {
                assert_invariants(&v, &model);
            }
        }
        assert_invariants(&v, &model);
        // The sequence has to have actually reached the post-fold states, or
        // it only proved the invariants for a base-empty matrix.
        assert!(
            v.m().nvals() > 0,
            "no fold ever happened: the `m` branches of set/remove were never taken"
        );
    }

    /// The tightest form of the same thing: an entry that is added, folded into
    /// the committed base, deleted and re-added must round-trip through the
    /// `dm` branches without ever landing in `dp` alongside its `m` entry.
    ///
    /// Reaching a fold needs a delta above `MIN_FOLD_DELTA` (`delta_dominates_
    /// base` then fires against the empty base), so the probe pair rides along
    /// with enough filler to trip the policy.
    #[test]
    fn folded_entry_deleted_and_re_added_stays_out_of_dp() {
        ensure_init();
        let filler = 4 * MIN_FOLD_DELTA;
        let mut v = VersionedMatrix::<bool>::new(DIM, DIM);
        v.set_all::<false>((0..filler).map(|i| (i % DIM, (i / DIM + 1) % DIM)));
        let probe = (7, 11);
        v.set(probe.0, probe.1, true);

        // New version + a mutation: the latched fold runs in `flush`. The
        // trigger pair must be outside the filler (cols 1 and 2) so the final
        // `nvals` check counts it once.
        let mut v = v.dup();
        v.set(300, 301, true);
        v.wait_all();
        assert!(
            v.m().get(probe.0, probe.1).is_some(),
            "the fold did not move the probe into the committed base"
        );
        assert!(v.dp().get(probe.0, probe.1).is_none());

        // Delete: must become a tombstone, not a `dp` removal.
        v.remove(probe.0, probe.1);
        v.wait_all();
        assert!(v.dm().get(probe.0, probe.1).is_some(), "no tombstone");
        assert!(v.m().get(probe.0, probe.1).is_some(), "base entry vanished");
        assert!(v.get(probe.0, probe.1).is_none(), "deleted entry readable");

        // Re-add: must clear the tombstone, not push a duplicate into `dp` —
        // which would leave `dp ∩ m ≠ ∅` and double-count in `nvals`.
        v.set(probe.0, probe.1, true);
        v.wait_all();
        assert!(
            v.dm().get(probe.0, probe.1).is_none(),
            "tombstone survived the re-add"
        );
        assert!(
            v.dp().get(probe.0, probe.1).is_none(),
            "re-add duplicated the committed entry into dp"
        );
        assert_eq!(v.get(probe.0, probe.1), Some(true));
        assert_eq!(v.nvals(), filler + 2, "nvals double-counted the re-add");
    }

    /// The row filter lets a single-row scan skip a delta layer entirely, which
    /// is only sound if it never rules out a row the layer holds. These pin the
    /// two halves of that: the skip happens, and it is never wrong.
    ///
    /// This one is the wrong half — a scan of a row the delta *does* hold must
    /// still see it, whether the entry is the only one in the delta or one of
    /// many.
    #[test]
    fn row_filter_never_hides_a_pending_entry() {
        ensure_init();
        let mut v = VersionedMatrix::<bool>::new(1 << 12, 1 << 12);
        for r in (0..2048).step_by(2) {
            v.set(r, r + 1, true);
        }
        let mut v = v.dup();
        v.flush();
        v.wait_all();
        // Pending entries on rows that are *not* in the committed base, so the
        // only way to read them is through `dp`.
        let pending: Vec<u64> = vec![1, 3, 777, 2047];
        for &r in &pending {
            v.set(r, r, true);
        }
        v.wait_all();
        for &r in &pending {
            let got: Vec<_> = v.iter(r, r).collect();
            assert_eq!(got, vec![(r, r)], "row {r}'s pending entry was skipped");
        }
        // And every committed row still reads, with the delta non-empty.
        for r in (0..2048).step_by(2) {
            let got: Vec<_> = v.iter(r, r).collect();
            let want = if pending.contains(&r) {
                vec![(r, r), (r, r + 1)]
            } else {
                vec![(r, r + 1)]
            };
            assert_eq!(got, want, "row {r} misread with a non-empty delta");
        }
    }

    /// **The hazard the design has to answer.** A skipped layer is skipped for
    /// *a range*, but `Iter::seek` re-points an existing iterator at a different
    /// range — so a layer dropped because the first range missed it would
    /// silently swallow entries in every later one. Detaching rather than
    /// dropping is what makes that safe, and this is the test that says so:
    /// build the iterator on a row the delta does not hold, then seek to one it
    /// does.
    #[test]
    fn seek_after_a_skipped_layer_still_reads_the_delta() {
        ensure_init();
        let mut v = VersionedMatrix::<bool>::new(1 << 12, 1 << 12);
        v.set(10, 10, true);
        let mut v = v.dup();
        v.flush();
        v.wait_all();
        v.set(900, 901, true);
        v.wait_all();

        // Row 10 is committed and holds no delta entry, so this iterator is
        // built with `dp` detached.
        let mut it = v.iter(10, 10);
        assert_eq!(it.by_ref().collect::<Vec<_>>(), vec![(10, 10)]);

        // Re-seeking must attach it rather than keep answering "empty".
        it.seek(900, 900);
        assert_eq!(
            it.collect::<Vec<_>>(),
            vec![(900, 901)],
            "a re-seeked iterator lost the pending entry"
        );
    }

    /// The filter is only worth having if the resizes a bulk create performs
    /// constantly leave it usable — degrading to `Unknown` would be safe and
    /// would silently switch the optimisation off. Both directions are checked,
    /// because they take different paths and end in different states.
    #[test]
    fn row_filter_stays_exact_across_both_resize_paths() {
        ensure_init();
        let mut v = VersionedMatrix::<bool>::new(4096, 4096);
        v.set(5, 6, true);
        v.wait_all();
        assert!(v.dp.may_hold_rows(5, 5), "the row it just took");
        assert!(!v.dp.may_hold_rows(7, 7), "a row it never took");

        // Shrink keeps the delta and resizes it in place, so the filter has to
        // survive `Delta::resize` — which reaches the layer through the same
        // `layer_mut` that invalidates by default.
        v.resize(2048, 2048);
        v.wait_all();
        assert!(v.dp.may_hold_rows(5, 5), "shrink lost the entry's row");
        assert!(
            !v.dp.may_hold_rows(7, 7),
            "shrink degraded the filter to Unknown"
        );
        assert_eq!(v.iter(5, 5).collect::<Vec<_>>(), vec![(5, 6)]);

        // Growing with a non-empty delta rebuilds the base with the delta
        // folded in, so afterwards nothing is pending and no row is claimed —
        // and the entry must still read, now out of the base.
        v.resize(8192, 8192);
        v.wait_all();
        assert_eq!(v.dp().nvals(), 0, "grow did not fold the delta");
        assert!(
            !v.dp.may_hold_rows(5, 5),
            "a folded delta still claims a row"
        );
        assert_eq!(v.iter(5, 5).collect::<Vec<_>>(), vec![(5, 6)]);
    }
}
