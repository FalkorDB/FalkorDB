//! Shared quotient-graph machinery for AMD-family bottom-up
//! orderings.
//!
//! This module hosts the workspace, elimination loop, and assembly-
//! tree postorder used by `feral-amd` and (planned) `feral-amf`.
//! Both orderings share the quotient-graph data structures
//! (`PE / IW / LEN / NV / ELEN`), the standard / aggressive element
//! absorption logic, the mass-elimination fast path, the
//! supervariable hash bucket detection, and the inline garbage
//! collector. They differ only in the *selection metric* —
//! approximate degree (AMD) vs approximate fill (AMF) — which is
//! abstracted behind the [`Metric`] trait. Phase A shipped the trait
//! plus the AMD-specialised [`MinDegree`] impl; Phase B.2 added
//! [`MinFill`] driving the parallel `run_elimination_amf` /
//! `create_element_amf` / `select_pivot_amf` / `finalize_step_amf`
//! family in `algo.rs`. The duplicated inner loops trade LoC for a
//! zero-risk AMD bit-parity contract.
//!
//! Reference: Amestoy, Davis, Duff (1996) "An approximate minimum
//! degree ordering algorithm," SIAM J. Matrix Analysis 17:886-905;
//! Amestoy (1999) habilitation thesis (AMF metric).

#![allow(dead_code)]
// Quotient-graph internals (Workspace fields, StepFlops fields, etc.)
// are pub because the planned `feral-amf` crate will read them
// directly. They are deliberately not part of the locked
// ordering-crate contract; see CONTRACT_VERSION.
#![allow(missing_docs)]

mod algo;
mod metric;
mod workspace;

pub use algo::{
    create_element, create_element_amf, finalize_permutation, finalize_step, finalize_step_amf,
    run_elimination, run_elimination_amf, select_pivot, select_pivot_amf, StepFlops,
};
pub use metric::{Metric, MinDegree, MinFill};
pub use workspace::{clear_flag, flip, Workspace, NONE};

use crate::Idx;
use crate::{CscPattern, OrderingError};

/// Tunable parameters for the shared quotient-graph workspace.
///
/// Only the workspace-relevant parameters live here. Crate-specific
/// knobs (e.g. `aggressive` for the elimination loop) are passed
/// directly to the relevant entry point.
#[derive(Debug, Clone)]
pub struct WorkspaceOptions {
    /// Dense-row threshold multiplier (Davis 1996 §5). A variable
    /// with initial degree exceeding
    /// `min(max(16, floor(dense_alpha * sqrt(n))), n)` is deferred to
    /// the end of the ordering — the `max(16)` floor is applied before
    /// the `min(n)` cap, matching faer `amd.rs:173-179`. A negative
    /// value uses a raw threshold of `n - 2` with the same clamps; for
    /// `n >= 18` that is exactly `n - 2`, suppressing deferral for
    /// everything but true hubs of degree `n - 1`.
    pub dense_alpha: f64,
}

impl Default for WorkspaceOptions {
    fn default() -> Self {
        Self { dense_alpha: 10.0 }
    }
}

/// Diagnostic counters extracted from a completed [`Workspace`].
///
/// Surfaced by [`order`] alongside the permutation so callers can
/// build crate-specific stats structs without re-borrowing the
/// workspace internals.
#[derive(Debug, Clone, Copy, Default)]
pub struct OrderDiagnostics {
    pub ncmpa: u32,
    pub n_mass_elim: u32,
    pub n_supervar_merge: u32,
    pub ndense: Idx,
    pub flops: StepFlops,
}

/// Run a metric-driven AMD-family ordering on a full-symmetric
/// pattern, returning the permutation plus diagnostic counters.
///
/// Equivalent to:
///
/// ```ignore
/// let mut ws = Workspace::new(pattern, opts)?;
/// let flops = M::run_elimination(&mut ws, aggressive)?;
/// let perm = finalize_permutation(&mut ws);
/// ```
///
/// `M` selects the metric (and, transitively, the elimination loop).
/// AMD uses [`MinDegree`]; the planned AMF crate will pass `MinFill`.
pub fn order<M: Metric>(
    pattern: &CscPattern<'_>,
    opts: &WorkspaceOptions,
    aggressive: bool,
) -> Result<(Vec<Idx>, OrderDiagnostics), OrderingError> {
    let n_buckets = M::n_buckets(pattern.n);
    let mut ws = Workspace::new_with_n_buckets(pattern, opts, n_buckets)?;
    let flops = M::run_elimination(&mut ws, aggressive)?;
    let diag = OrderDiagnostics {
        ncmpa: ws.ncmpa,
        n_mass_elim: ws.n_mass_elim,
        n_supervar_merge: ws.n_supervar_merge,
        ndense: ws.ndense,
        flops,
    };
    let perm = finalize_permutation(&mut ws);
    Ok((perm, diag))
}
