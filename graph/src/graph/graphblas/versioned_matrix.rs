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
//! When delta matrices exceed 10,000 entries, [`flush`](VersionedMatrix::flush)
//! merges them into the base matrix (`dp` via element-wise add, `dm` via
//! masked removal) and clears the deltas.
//!
//! ## Iterator
//!
//! [`Iter`] performs a three-way sorted merge over the `m`, `dp`, and `dm`
//! iterators (GraphBLAS row iterators yield ascending `(row, col)` order):
//! `m` entries matched by the `dm` lookahead are dropped, `dp` entries are
//! interleaved in order (winning over a shadowed `m` entry at the same
//! position), producing the effective state — sorted by `(row, col)` —
//! without materializing a merged matrix or issuing per-entry point lookups.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::{
    GxB_Print_Level,
    matrix::{self, Dup, Matrix},
    serialization::{Decode, Encode, Reader, Writer},
};
use crate::graph::{
    cow::Cow,
    graphblas::matrix::{BoolExtract, IterExtract},
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
/// transaction. A flat absolute limit was measured against the sqrt rule and
/// rejected for the same reason: on the read path at `tx_added = 1` a 4k limit
/// costs 7.2x the per-write cost (102.5 µs vs 14.3 µs) and 14x the worst-case
/// stall (206.5 µs vs ~14 µs).
///
/// `F/w` is measured rather than modelled — see `fold_cost_bench`:
///
/// * `F`, the fold, is ~82% fixed cost — `1690 µs + 0.34 ns · nvals` — and
///   independent of `nrows`: an empty fold costs 2.35 µs at nrows = 65k, 1M
///   and 16.7M alike. So `F ≈ 2050 µs` for any base above ~1m entries and the
///   balance point is *flat* in the base rather than growing like
///   `sqrt(base)`. (An earlier revision used `base.nvals() + base.nrows()`
///   on the theory that a fold rewrites the row-pointer structure. The
///   measurement refutes it: the `+16%` instructions that motivated the
///   `nrows` term came from fold *frequency*, which the measured `F` now
///   fixes directly.)
/// * `w` differs by **250x** between the two paths, which is what the
///   write/read split below encodes. A write transaction nobody reads pays
///   only the COW dup: `w_dup ≈ 0.2 ns/entry`, memcpy speed — and duping a
///   *pending* matrix costs the same as an assembled one, so `dup` does not
///   assemble. A transaction whose delta is also materialized pays the
///   pending-tuple merge, `w_merge ≈ 50 ns/entry`. That merge is
///   `O(|delta|)` no matter how little the transaction added: adding 1 entry
///   to a 16k delta and adding 100 both measure ~900 µs.
///
/// Since `D* ∝ 1/sqrt(w)`, the write path's balance point sits
/// `sqrt(250) ≈ 16x` above the read path's. Reads keep paying for a lingering
/// delta on every access, so [`should_fold_read`] — the policy for `wait`,
/// which only runs when something reads the matrix — folds at the tighter
/// point, promptly bounding the delta once a workload turns read-heavy.
///
/// A delta comparable to the base always folds (`2·|delta| ≥ base_nvals`): the
/// fold then costs the same order as one delta touch, and without it a one-shot
/// bulk transaction (whose huge `tx_added` defeats the sqrt term) can leave a
/// base-sized delta taxing every later transaction. This is also what bounds
/// the delta on a pure-write stream, where the write path deliberately lets it
/// run large — an absolute cap there is not free: capping at 64k costs 3.5x
/// bulk write throughput at `tx_added = 10k`, and capping at 16k costs 13.8x,
/// against a one-time ~3 ms assembly for the first reader after the burst.
const WRITE_FOLD_K: u64 = 20_500_000;

/// `2 · F / w_merge` — the read path's `D*² / tx_added`. `w_merge ≈ 50 ns`
/// per entry, 250x the dup cost, putting the balance point at
/// `286 · sqrt(tx_added)` entries.
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
    tx_added > 0
        && delta_nvals >= MIN_FOLD_DELTA
        && (delta_dominates_base(delta_nvals, base_nvals)
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
    dp: Cow<Matrix<T>>,
    /// Delta-minus: edges removed in current transaction (always a bool mask)
    dm: Cow<Matrix<bool>>,
    /// Approximate `dp.nvals()`, maintained by the mutation methods without
    /// GraphBLAS calls. `GrB_Matrix_nvals` on a delta forces its pending
    /// tuples to merge (an `O(|delta|)` sort/merge), so the fold policy reads
    /// these counters instead and [`wait`](Self::wait) resyncs them to the
    /// exact values whenever it materializes the deltas anyway.
    dp_count: AtomicU64,
    /// Same as `dp_count`, for `dm`.
    dm_count: AtomicU64,
    /// `dp_count` when this MVCC version was created (`dup`), or after the
    /// last fold: `dp_count - dp_tx_nvals` is what the current transaction
    /// has added — the batch size the fold policy weighs the fold cost
    /// against.
    dp_tx_nvals: u64,
    /// Same as `dp_tx_nvals`, for `dm`.
    dm_tx_nvals: u64,
    /// Fold decisions are made in `dup` (version creation, pre-mutation) and
    /// `wait` (post-mutation, when the transaction's contribution is known)
    /// but executed by `flush`, which runs *before* the next mutation — the
    /// ideal moment, since folding there replaces the COW dup of both the
    /// delta and the base. The flags latch the decision across that gap (and
    /// across versions, via `dup`).
    fold_dp: AtomicBool,
    fold_dm: AtomicBool,
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
            dp_count: AtomicU64::new(self.dp_count.load(Ordering::Relaxed)),
            dm_count: AtomicU64::new(self.dm_count.load(Ordering::Relaxed)),
            dp_tx_nvals: self.dp_tx_nvals,
            dm_tx_nvals: self.dm_tx_nvals,
            fold_dp: AtomicBool::new(self.fold_dp.load(Ordering::Relaxed)),
            fold_dm: AtomicBool::new(self.fold_dm.load(Ordering::Relaxed)),
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
        self.dp.wait();
        self.dm.wait();
        let base = self.m.nvals();
        let dp_nvals = self.dp.nvals();
        let dm_nvals = self.dm.nvals();
        // The deltas are materialized now — resync the approximate counters
        // to the exact counts, bounding any drift they accumulated.
        self.dp_count.store(dp_nvals, Ordering::Relaxed);
        self.dm_count.store(dm_nvals, Ordering::Relaxed);
        let fold_dp = self.fold_dp.load(Ordering::Relaxed)
            || should_fold_read(dp_nvals, dp_nvals.saturating_sub(self.dp_tx_nvals), base);
        let fold_dm = self.fold_dm.load(Ordering::Relaxed)
            || should_fold_read(dm_nvals, dm_nvals.saturating_sub(self.dm_tx_nvals), base);
        self.fold_dp.store(fold_dp, Ordering::Relaxed);
        self.fold_dm.store(fold_dm, Ordering::Relaxed);
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
        // Deltas are folded in; swap in fresh empty ones at the new dims
        // (resizing through the Cow would deep-copy a still-shared delta).
        self.dp
            .replace(Matrix::<bool>::new(nrows, ncols).into_hyper());
        self.dm
            .replace(Matrix::<bool>::new(nrows, ncols).into_hyper());
        *self.dp_count.get_mut() = 0;
        *self.dm_count.get_mut() = 0;
        self.dp_tx_nvals = 0;
        self.dm_tx_nvals = 0;
        self.fold_dp.store(false, Ordering::Relaxed);
        self.fold_dm.store(false, Ordering::Relaxed);
        self.needs_flush.store(false, Ordering::Relaxed);
    }

    pub fn remove(
        &mut self,
        i: u64,
        j: u64,
    ) {
        self.flush();
        // See `set`: the debug_assert reads dp raw; only wait while shared.
        #[cfg(debug_assertions)]
        if self.dp.is_shared() {
            self.dp.wait();
        }
        if self.m.get(i, j).is_some() {
            debug_assert!(self.dp.get(i, j).is_none());
            self.dm.set(i, j, true);
            *self.dm_count.get_mut() += 1;
        } else {
            self.dp.remove(i, j);
            let dp_count = self.dp_count.get_mut();
            *dp_count = dp_count.saturating_sub(1);
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
        self.dm
            .element_wise_multiply(Some(mask), Some(mask), Some(&*self.m), None);
        // dp &= ~mask: remove entries from dp that exist in mask
        self.dp.remove_all(mask);
        // Resync the approximate counters to the exact counts. `nvals` would
        // report them correctly on its own — `GrB_Matrix_nvals` completes any
        // pending work internally — but the eWiseMult/remove_all above leave
        // the wrapper's `has_pending` flag set, so without waiting first the
        // flag stays a lie: `is_synced()` keeps reporting false and every
        // later read path pays a redundant `wait()`. Waiting here is a single
        // early-returning atomic load per layer once already synced.
        self.dm.wait();
        self.dp.wait();
        *self.dm_count.get_mut() = self.dm.nvals();
        *self.dp_count.get_mut() = self.dp.nvals();
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

    pub fn set(
        &mut self,
        i: u64,
        j: u64,
        value: bool,
    ) {
        self.flush();
        // The debug_asserts below read the deltas raw; a get on a pending
        // matrix finishes that work internally (a mutation), racing
        // concurrent readers when the layer is still shared with the
        // committed snapshot. Only wait while shared — once dup'd the layer
        // is writer-local and raw gets are single-threaded, and waiting
        // unconditionally re-merges pending tuples per element (quadratic
        // bulk mutation in debug builds).
        #[cfg(debug_assertions)]
        {
            if self.dp.is_shared() {
                self.dp.wait();
            }
            if self.dm.is_shared() {
                self.dm.wait();
            }
        }
        if self.m.get(i, j).is_some() {
            debug_assert!(self.dp.get(i, j).is_none());
            self.dm.remove(i, j);
            let dm_count = self.dm_count.get_mut();
            *dm_count = dm_count.saturating_sub(1);
        } else {
            debug_assert!(self.dm.get(i, j).is_none());
            self.dp.set(i, j, value);
            *self.dp_count.get_mut() += 1;
        }
    }

    #[must_use]
    pub fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        Self {
            m: Cow::new(Matrix::<bool>::new(nrows, ncols)),
            dp: Cow::new(Matrix::<bool>::new(nrows, ncols).into_hyper()),
            dm: Cow::new(Matrix::<bool>::new(nrows, ncols).into_hyper()),
            dp_count: AtomicU64::new(0),
            dm_count: AtomicU64::new(0),
            dp_tx_nvals: 0,
            dm_tx_nvals: 0,
            fold_dp: AtomicBool::new(false),
            fold_dm: AtomicBool::new(false),
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
            dp: Cow::new(Matrix::<bool>::new(nrows, ncols).into_hyper()),
            dm: Cow::new(Matrix::<bool>::new(nrows, ncols).into_hyper()),
            dp_count: AtomicU64::new(0),
            dm_count: AtomicU64::new(0),
            dp_tx_nvals: 0,
            dm_tx_nvals: 0,
            fold_dp: AtomicBool::new(false),
            fold_dm: AtomicBool::new(false),
            needs_flush: AtomicBool::new(false),
        }
    }

    pub fn flush(&mut self) {
        if self.needs_flush.load(Ordering::Relaxed) {
            // The layers may be shared with the committed snapshot and carry
            // pending work; nvals/eWiseAdd/select below materialize it (a
            // mutation), so wait first under the readers' lock.
            self.wait_all();
            let fold_dp = self.fold_dp.swap(false, Ordering::Relaxed) && self.dp.nvals() > 0;
            let fold_dm = self.fold_dm.swap(false, Ordering::Relaxed) && self.dm.nvals() > 0;
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
                // Clearing through the Cow would deep-copy a still-shared
                // delta just to empty it; swap in a fresh empty matrix.
                if fold_dp {
                    self.dp
                        .replace(Matrix::<bool>::new(nrows, ncols).into_hyper());
                    *self.dp_count.get_mut() = 0;
                    self.dp_tx_nvals = 0;
                }
                if fold_dm {
                    self.dm
                        .replace(Matrix::<bool>::new(nrows, ncols).into_hyper());
                    *self.dm_count.get_mut() = 0;
                    self.dm_tx_nvals = 0;
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
        if self.fold_dp.load(Ordering::Relaxed) || self.fold_dm.load(Ordering::Relaxed) {
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
        let fold_dp = delta_dominates_base(self.dp_count.load(Ordering::Relaxed), base);
        let fold_dm = delta_dominates_base(self.dm_count.load(Ordering::Relaxed), base);
        if fold_dp || fold_dm {
            self.fold_dp.fetch_or(fold_dp, Ordering::Relaxed);
            self.fold_dm.fetch_or(fold_dm, Ordering::Relaxed);
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
    pub fn set_all(
        &mut self,
        entries: impl Iterator<Item = (u64, u64)>,
    ) {
        self.flush();
        // The nvals below reads the shared dm raw; nvals on a pending matrix
        // merges its pending tuples internally (a mutation), racing
        // concurrent readers on the committed snapshot. Materialize through
        // the mutex-guarded wait first — an atomic load when already synced.
        self.dm.wait();
        if self.dm.nvals() == 0 {
            let mut n = 0u64;
            for (i, j) in entries {
                if self.m.get(i, j).is_none() {
                    self.dp.set(i, j, true);
                    n += 1;
                }
            }
            *self.dp_count.get_mut() += n;
        } else {
            for (i, j) in entries {
                self.set(i, j, true);
            }
        }
    }

    /// [`Self::set_all`] for entries the caller guarantees are new — never
    /// live in the committed base (fresh entity ids: a reclaimed id's stale
    /// base entry always has a `dm` tombstone, which routes to the safe
    /// per-entry path here). Skips the per-entry base lookup the general
    /// path needs to keep `dp ∩ m = ∅`; entries with committed pairs (e.g.
    /// the adjacency matrix, where two edges share one pair) must use
    /// [`Self::set_all`] instead.
    pub fn set_all_new(
        &mut self,
        entries: impl Iterator<Item = (u64, u64)>,
    ) {
        self.flush();
        // See `set_all`: the nvals below reads the shared dm raw.
        self.dm.wait();
        if self.dm.nvals() == 0 {
            let mut n = 0u64;
            for (i, j) in entries {
                debug_assert!(self.m.get(i, j).is_none());
                self.dp.set(i, j, true);
                n += 1;
            }
            *self.dp_count.get_mut() += n;
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
        let dp_count = self.dp_count.load(Ordering::Relaxed);
        let dm_count = self.dm_count.load(Ordering::Relaxed);
        let fold_dp = self.fold_dp.load(Ordering::Relaxed)
            || should_fold(dp_count, dp_count.saturating_sub(self.dp_tx_nvals), base);
        let fold_dm = self.fold_dm.load(Ordering::Relaxed)
            || should_fold(dm_count, dm_count.saturating_sub(self.dm_tx_nvals), base);
        Self {
            m: self.m.new_version(),
            dp: self.dp.new_version(),
            dm: self.dm.new_version(),
            dp_count: AtomicU64::new(dp_count),
            dm_count: AtomicU64::new(dm_count),
            dp_tx_nvals: dp_count,
            dm_tx_nvals: dm_count,
            fold_dp: AtomicBool::new(fold_dp),
            fold_dm: AtomicBool::new(fold_dm),
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
            dp: Cow::new(self.dp.transpose().into_hyper()),
            dm: Cow::new(self.dm.transpose().into_hyper()),
            dp_count: AtomicU64::new(self.dp_count.load(Ordering::Relaxed)),
            dm_count: AtomicU64::new(self.dm_count.load(Ordering::Relaxed)),
            dp_tx_nvals: self.dp_tx_nvals,
            dm_tx_nvals: self.dm_tx_nvals,
            fold_dp: AtomicBool::new(self.fold_dp.load(Ordering::Relaxed)),
            fold_dm: AtomicBool::new(self.fold_dm.load(Ordering::Relaxed)),
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
        // Decoded deltas have no owning transaction; treat the whole delta as
        // freshly added so the fold policy sees it on the first flush.
        let fold_dp = should_fold(dp.nvals(), dp.nvals(), m.nvals());
        let fold_dm = should_fold(dm.nvals(), dm.nvals(), m.nvals());
        Ok(Self {
            dp_count: AtomicU64::new(dp.nvals()),
            dm_count: AtomicU64::new(dm.nvals()),
            m: Cow::new(m),
            dp: Cow::new(dp.into_hyper()),
            dm: Cow::new(dm.into_hyper()),
            dp_tx_nvals: 0,
            dm_tx_nvals: 0,
            fold_dp: AtomicBool::new(fold_dp),
            fold_dm: AtomicBool::new(fold_dm),
            needs_flush: AtomicBool::new(fold_dp || fold_dm),
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
        Self::from_layers(&vm.m, &vm.dp, &vm.dm, min_row, max_row)
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
        dm: &Cow<Matrix<bool>>,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        let mut dmit = if dm.nvals() == 0 {
            None
        } else {
            Some(matrix::Iter::<BoolExtract>::new(&**dm, min_row, max_row))
        };
        let dm_next = dmit.as_mut().and_then(Iterator::next);
        Self {
            mit: matrix::Iter::new(m, min_row, max_row),
            m_next: None,
            dpit: if dp.nvals() == 0 {
                None
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
    use super::{MIN_FOLD_DELTA, READ_FOLD_K, WRITE_FOLD_K, should_fold, should_fold_read};

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
}
