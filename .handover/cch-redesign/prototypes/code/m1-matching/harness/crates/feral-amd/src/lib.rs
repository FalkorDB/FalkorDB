//! Approximate Minimum Degree (AMD) fill-reducing ordering.
//!
//! Standalone implementation of the in-place quotient-graph AMD
//! algorithm (Amestoy, Davis & Duff 1996, 2004). See the crate
//! README and `dev/plans/ordering-amd-upgrade.md` for scope and
//! references.
//!
//! Slice B is complete: mass elimination (Commit 9) and
//! supervariable detection (Commit 10) are both live, so the
//! ordering matches SuiteSparse / faer on the full oracle
//! fixture suite.
//!
//! The public surface conforms to the FERAL ordering-crate
//! contract (`dev/plans/ordering-crate-contract.md`). `CscPattern`,
//! `OrderingStats`, `OrderingError`, and `CONTRACT_VERSION` are
//! re-exported from `feral-ordering-core`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod stats;

pub use feral_ordering_core::{CscPattern, OrderingError, OrderingStats, CONTRACT_VERSION};
pub use stats::AmdStats;

use feral_ordering_core::quotient_graph::{
    finalize_permutation, order, run_elimination, MinDegree, Workspace, WorkspaceOptions,
};
use feral_ordering_core::Idx;
use std::time::Instant;

/// Tunable parameters for AMD ordering.
///
/// Defaults match faer / SuiteSparse: `aggressive = true`,
/// `dense_alpha = 10.0`.
#[derive(Debug, Clone)]
pub struct AmdOptions {
    /// Enable aggressive element absorption in the Pass-2 degree
    /// loop (faer `amd.rs:404-407`).
    pub aggressive: bool,
    /// Dense-row threshold multiplier. A variable with initial
    /// degree exceeding `min(max(16, floor(dense_alpha * sqrt(n))), n)`
    /// is deferred to the end of the ordering — the `max(16)` floor is
    /// applied before the `min(n)` cap, matching faer `amd.rs:173-179`
    /// / SuiteSparse AMD (the order matters: it guarantees the
    /// threshold is `<= n`). A negative value uses a raw threshold of
    /// `n - 2` in place of `dense_alpha * sqrt(n)`, with the same
    /// `max(16)`/`min(n)` clamps; for `n >= 18` that is exactly
    /// `n - 2`, suppressing deferral for all but true hubs of degree
    /// `n - 1`.
    pub dense_alpha: f64,
}

impl Default for AmdOptions {
    fn default() -> Self {
        Self {
            aggressive: true,
            dense_alpha: 10.0,
        }
    }
}

/// Compute a fill-reducing AMD ordering.
///
/// Returns a permutation `perm` (new-to-old) such that factoring
/// `P·A·Pᵀ` with `P[k] = perm[k]` produces less fill than the
/// natural ordering. The input must be the full symmetric pattern
/// (both halves present).
pub fn amd_order(pattern: &CscPattern<'_>) -> Result<Vec<Idx>, OrderingError> {
    amd_order_opts(pattern, &AmdOptions::default()).map(|(perm, _)| perm)
}

/// Compute an AMD ordering and return the crate-specific diagnostic
/// counters.
///
/// See [`amd_order`] and [`AmdStats`]. Callers that also need the
/// shared [`OrderingStats`] (wall time, fill estimate) should use
/// [`amd_order_full`] instead.
pub fn amd_order_with_stats(
    pattern: &CscPattern<'_>,
) -> Result<(Vec<Idx>, AmdStats), OrderingError> {
    amd_order_opts(pattern, &AmdOptions::default())
}

/// Compute an AMD ordering with explicit options.
///
/// Returns `(perm, amd_stats)`. See [`amd_order_full`] for the
/// contract-conforming three-tuple return.
pub fn amd_order_opts(
    pattern: &CscPattern<'_>,
    opts: &AmdOptions,
) -> Result<(Vec<Idx>, AmdStats), OrderingError> {
    amd_order_full(pattern, opts).map(|(perm, _, amd_stats)| (perm, amd_stats))
}

/// Contract-conforming ordering producer.
///
/// Signature matches the shape every FERAL ordering crate must
/// expose per `dev/plans/ordering-crate-contract.md`: input is a
/// full-symmetric [`CscPattern`] and options; output is a
/// three-tuple of `(perm, OrderingStats, crate-stats)`, with
/// errors in [`OrderingError`].
///
/// `OrderingStats.time_us` is the wall-clock time of this call.
/// `fill_estimate` and `flop_estimate` are left as `None` for AMD —
/// the per-crate [`AmdStats`] carries `ndiv` / `nms_lu` / `nms_ldl`
/// flop counters that may be surfaced here in a future revision
/// without bumping the contract.
pub fn amd_order_full(
    pattern: &CscPattern<'_>,
    opts: &AmdOptions,
) -> Result<(Vec<Idx>, OrderingStats, AmdStats), OrderingError> {
    let t0 = Instant::now();
    let ws_opts = WorkspaceOptions {
        dense_alpha: opts.dense_alpha,
    };
    let (perm, diag) = order::<MinDegree>(pattern, &ws_opts, opts.aggressive)?;
    let amd_stats = AmdStats {
        ncmpa: diag.ncmpa,
        n_clear_flag: 0,
        n_mass_elim: diag.n_mass_elim,
        n_supervar_merge: diag.n_supervar_merge,
        n_dense_deferred: diag.ndense.max(0) as u32,
        ndiv: diag.flops.ndiv.max(0.0) as u64,
        nms_lu: diag.flops.nms_lu.max(0.0) as u64,
        nms_ldl: diag.flops.nms_ldl.max(0.0) as u64,
    };
    let ordering_stats = OrderingStats {
        time_us: t0.elapsed().as_micros() as u64,
        fill_estimate: None,
        flop_estimate: None,
    };
    Ok((perm, ordering_stats, amd_stats))
}

/// Per-sub-stage wall-clock breakdown of one AMD call.
///
/// Returned by [`amd_order_substages`] alongside the permutation.
/// All fields are wall-clock microseconds for that single call.
/// Sum is approximately equal to the total time reported by
/// [`amd_order_full`]'s [`OrderingStats::time_us`], modulo a few
/// hundred ns of `Instant::now()` overhead.
///
/// Used by the small-n diagnostic probe
/// `feral::bin::diag_amd_substages` to attribute the per-call AMD
/// cost between workspace allocation, the main elimination loop,
/// and permutation finalisation. See
/// `dev/plans/phase-2.13-tail-diagnostic.md` step 5.
#[derive(Debug, Clone, Copy, Default)]
pub struct AmdSubstages {
    /// Time spent in `AmdWorkspace::new`: input ingest, vector
    /// allocations, initial degree lists, dense-row deferral.
    pub workspace_new_us: u64,
    /// Time spent in the main pivot/eliminate/finalize loop
    /// (`select_pivot` + `create_element` + `finalize_step`).
    pub run_elimination_us: u64,
    /// Time spent in the assembly-tree postorder and final
    /// permutation emission.
    pub finalize_permutation_us: u64,
}

/// Compute an AMD ordering and report a per-sub-stage timing
/// breakdown.
///
/// Behaves identically to [`amd_order_full`] but additionally
/// returns an [`AmdSubstages`] split of the wall-clock time across
/// `workspace::new`, `run_elimination`, and `finalize_permutation`.
/// Diagnostic-only — production callers should keep using
/// [`amd_order`] / [`amd_order_full`]. The `Instant::now()` calls
/// add at most ~100 ns vs the un-profiled path, but the API surface
/// is intentionally separate to avoid polluting the stable
/// contract-conforming function.
pub fn amd_order_substages(
    pattern: &CscPattern<'_>,
    opts: &AmdOptions,
) -> Result<(Vec<Idx>, AmdSubstages), OrderingError> {
    let t = Instant::now();
    let ws_opts = WorkspaceOptions {
        dense_alpha: opts.dense_alpha,
    };
    let mut ws = Workspace::new(pattern, &ws_opts)?;
    let workspace_new_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    let _flops = run_elimination(&mut ws, opts.aggressive)?;
    let run_elimination_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    let perm = finalize_permutation(&mut ws);
    let finalize_permutation_us = t.elapsed().as_micros() as u64;

    Ok((
        perm,
        AmdSubstages {
            workspace_new_us,
            run_elimination_us,
            finalize_permutation_us,
        },
    ))
}
