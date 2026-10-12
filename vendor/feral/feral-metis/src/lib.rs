//! Multilevel nested-dissection fill-reducing ordering.
//!
//! Clean-room Rust implementation of the algorithm described in
//! Karypis & Kumar, "A Fast and High Quality Multilevel Scheme for
//! Partitioning Irregular Graphs" (SIAM J. Sci. Comput., 1998), and
//! George, "Nested Dissection of a Regular Finite Element Mesh"
//! (SIAM J. Numer. Anal., 1973).
//!
//! The public surface conforms to the FERAL ordering-crate contract
//! (`dev/plans/ordering-crate-contract.md`): `CscPattern`,
//! `OrderingStats`, `OrderingError`, and `CONTRACT_VERSION` are
//! re-exported from `feral-ordering-core`.
//!
//! **Status: M1–M7 complete.** `metis_order_full` coarsens the graph
//! (SHEM + 2-hop), picks the best of `niparts` initial bisections
//! scored on their post-FM cut, uncoarsens with FM refinement, turns
//! the final edge bisection into a node separator via min vertex
//! cover (König's theorem), and recursively orders the two sides —
//! handing off to AMD on subgraphs no larger than
//! `nd_to_amd_switch`. M8 (integration into the main solver) is
//! tracked separately in `dev/plans/ordering-metis.md`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use feral_ordering_core::Idx;

// Modules are exercised only by `metis_order_full` once all
// milestones land; until then, dead-code lint is suppressed at the
// module root for internal helpers.
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod coarsen;
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod fm_refine;
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod graph;
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod initial_partition;
mod node_nd;
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod rng;
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod separator;

/// Crate-internal infrastructure exposed for sibling ordering
/// crates (notably `feral-scotch`) that share the multilevel
/// coarsening, initial-bisection, and FM-refinement plumbing.
///
/// **Not part of the stable public API.** No semver guarantees on
/// signatures inside `internals`; consumers re-export it at their
/// own risk. This module exists solely so feral-scotch does not
/// have to clone the multilevel framework.
#[doc(hidden)]
pub mod internals {
    pub use crate::coarsen;
    pub use crate::fm_refine;
    pub use crate::graph;
    pub use crate::initial_partition;
    pub use crate::rng;
    pub use crate::separator;
}

pub use feral_ordering_core::{CscPattern, OrderingError, OrderingStats, CONTRACT_VERSION};

/// Tunable parameters for METIS nested-dissection ordering.
///
/// Defaults mirror METIS 5.2.0's `METIS_NodeND` defaults as documented
/// in `dev/plans/ordering-metis.md` audit (MUMPS uses stock METIS
/// defaults for KKT problems: `METIS_OPTION_NUMBERING = 1`, all other
/// options at library default).
#[derive(Debug, Clone)]
pub struct MetisOptions {
    /// Deterministic RNG seed. Defaults to 1. Two runs with the same
    /// seed on the same input must produce the same permutation.
    pub seed: u64,
    /// Number of initial-bisection trials at the coarsest level
    /// (METIS 5.2.0 default: 7). Each trial alternates GGP and random
    /// BFS and is scored on its post-FM cut.
    pub niparts: u32,
    /// Stop coarsening when the graph has fewer than this many
    /// vertices (METIS 5.2.0 default: 120).
    pub coarsen_floor: u32,
    /// Switch from recursive ND to AMD on uncoarsened subproblems of
    /// at most this many vertices (METIS 5.2.0 default: 200).
    pub nd_to_amd_switch: u32,
    /// Reduction-ratio threshold below which SHEM falls back to
    /// 2-hop matching (METIS 5.2.0 default: 0.85).
    pub two_hop_ratio_threshold: f64,
    /// Maximum partition imbalance factor (`ufactor` in METIS terms,
    /// encoded as a fraction here). METIS 5.2.0 uses 200, which
    /// corresponds to 1.20 load balance tolerance; expressed as the
    /// fractional deviation 0.20.
    pub max_imbalance: f64,
    /// Number of FM passes at each uncoarsening level (METIS 5.2.0
    /// default: 10).
    pub fm_passes: u32,
    /// Carry the **node separator** through uncoarsening and refine it
    /// with FM at every level (METIS's `Refine2WayNode` /
    /// `FM_2WayNodeRefine1Sided`), instead of refining the edge cut
    /// through uncoarsening and converting to a separator once at the
    /// finest level.
    ///
    /// **Default: `true`.** The two paths share coarsening and initial
    /// bisection; they differ only in what the uncoarsening loop
    /// optimises. Measured against MA57's bundled real METIS as the
    /// oracle (`dev/research/feral-metis-node-separator-fm-2026-09-17.md`),
    /// on the six matrices in this repo large enough for nested
    /// dissection to engage, `node_refine` cuts the elimination flop
    /// count by 2.4-3.5x on collocation KKTs. It is **not a uniform
    /// win**: measured in wall-clock on grid Laplacians (min over 3
    /// runs of 5 pairs each, non-overlapping), a 40^3 grid factors
    /// 12.8% *faster* while a 300x300 grid factors 5.8% **slower**
    /// despite 10.3% less fill — one more case of fill not predicting
    /// speed. On every other matrix measured (seven real IPM families
    /// up to n=607,500, and 37 of 38 parity matrices) the permutation
    /// is bit-identical, so there is nothing to win or lose.
    ///
    /// Set to `false` to recover the pre-2026-09-17 edge-cut
    /// behaviour.
    pub node_refine: bool,
    /// Pull near-dense columns out of the ND graph before recursive
    /// bisection and append them at the *end* of the returned
    /// permutation.
    ///
    /// **Default: `false`.** The technique was implemented to mimic
    /// what we believed MUMPS's `ICNTL(6)` and SSIDS did, but expert
    /// review of the MUMPS and SPRAL sources (2026-04-27) found:
    /// (a) `ICNTL(6)` is MC64 matching, not dense-row removal;
    /// (b) MUMPS handles dense rows *inside* its AMD/AMF
    /// (`MUMPS_QAMD` in `ana_orderings.F:5226+` with the `THRESM`
    /// parameter and `HEAD(N)` quasi-dense list); and
    /// (c) SSIDS does not special-case dense rows at all — it relies
    /// on METIS placing them in the top separator and supernodal
    /// amalgamation collapsing the resulting chain into one dense
    /// BLAS-3 root frontal. Neither solver pre-strips the graph.
    /// Empirically, on ORBIT2_0000 (n=4795, one column of off-degree
    /// 1794) Fix A *increased* `nnz_L` from 1.54M to 2.25M because
    /// removing the dense column destroys the structural signal that
    /// makes it the natural top separator. The opt-in path is kept
    /// for diagnostic experimentation; the correct fix lives in
    /// `feral-amd` (a QAMD-style deferral, future work).
    ///
    /// References (kept for the opt-in code path):
    /// - Davis & Hager, "Dynamic supernodes in sparse Cholesky
    ///   update/downdate and triangular solves" (2009), §3.2.
    /// - Davis (1996) AMD paper, §5 ("dense rows / `Alpha` parameter").
    /// - MUMPS source: `ana_orderings.F:5226-5650` (QAMD).
    pub dense_quotient_enabled: bool,
    /// Override the off-diagonal-degree threshold above which a column
    /// is treated as quasi-dense.
    ///
    /// When `None` (the default) the threshold is computed as
    /// `max(40, ceil(10 * sqrt(n)))` per Davis & Hager / AMD §5. Set
    /// to `Some(usize::MAX)` to effectively disable the quotient
    /// without flipping `dense_quotient_enabled` (useful for
    /// regression sweeps).
    pub dense_quotient_threshold: Option<usize>,
}

impl Default for MetisOptions {
    fn default() -> Self {
        Self {
            seed: 1,
            niparts: 7,
            coarsen_floor: 120,
            nd_to_amd_switch: 200,
            two_hop_ratio_threshold: 0.85,
            max_imbalance: 0.20,
            fm_passes: 10,
            node_refine: true,
            dense_quotient_enabled: false,
            dense_quotient_threshold: None,
        }
    }
}

/// Crate-specific diagnostic counters for METIS nested dissection.
///
/// Populated per call to [`metis_order_full`]. Callers that only need
/// the permutation should use [`metis_order`]; callers that need the
/// shared [`OrderingStats`] (wall time) should use
/// [`metis_order_full`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MetisStats {
    /// Number of coarsening levels built.
    pub n_levels: u32,
    /// Number of top-level connected components encountered.
    pub n_components: u32,
    /// Number of vertices assigned to a separator at any level.
    pub n_separator_vertices: u32,
    /// Number of FM passes executed across all levels.
    pub n_fm_passes: u32,
    /// Number of times SHEM fell through to the 2-hop matching path.
    pub n_two_hop_fallbacks: u32,
    /// Number of subgraphs handed off to the AMD leaf solver (when
    /// `nd_to_amd_switch` triggers).
    pub n_amd_leaf_calls: u32,
}

/// Compute a fill-reducing METIS nested-dissection ordering.
///
/// Thin wrapper over [`metis_order_full`] that discards the
/// diagnostic stats. Returns a permutation `perm` (new-to-old).
pub fn metis_order(pattern: &CscPattern<'_>) -> Result<Vec<Idx>, OrderingError> {
    metis_order_full(pattern, &MetisOptions::default()).map(|(perm, _, _)| perm)
}

/// Contract-conforming ordering producer.
///
/// Signature matches the shape every FERAL ordering crate must expose
/// per `dev/plans/ordering-crate-contract.md`: input is a
/// full-symmetric [`CscPattern`] and options; output is a three-tuple
/// of `(perm, OrderingStats, crate-stats)`, with errors in
/// [`OrderingError`].
///
/// `OrderingStats.time_us` is the wall-clock time of this call.
/// `fill_estimate` and `flop_estimate` stay `None` — METIS does not
/// produce them at the ordering boundary; they belong to a downstream
/// symbolic analysis.
///
/// Runs the M1–M7 pipeline: coarsen, initial bisection, FM, separator
/// construction, and recursive nested dissection with an AMD leaf
/// fallback for subgraphs of at most `nd_to_amd_switch` vertices.
pub fn metis_order_full(
    pattern: &CscPattern<'_>,
    opts: &MetisOptions,
) -> Result<(Vec<Idx>, OrderingStats, MetisStats), OrderingError> {
    if pattern.col_ptr.len() != pattern.n + 1 {
        return Err(OrderingError::MalformedInput);
    }
    let t0 = std::time::Instant::now();
    let mut stats = MetisStats::default();

    // Fix A — quasi-dense column quotient.
    //
    // Pull columns with off-diagonal degree above the
    // `dense_quotient_threshold` (default `max(40, 10*sqrt(n))`) out
    // of the ND input graph, run M1–M7 ND on the *sparse-induced*
    // subgraph, and append the dense columns at the end of the
    // returned permutation. This was originally modelled on a belief
    // that HSL_MC68 / MUMPS ICNTL(6) / SSIDS pre-strip dense rows, but
    // a 2026-04-27 audit of the MUMPS and SPRAL sources found that
    // belief wrong: ICNTL(6) is MC64 matching, MUMPS defers dense rows
    // inside QAMD, and SSIDS does not special-case them — neither
    // pre-strips the graph. See `MetisOptions::dense_quotient_enabled`
    // for the full finding. The path is kept opt-in (default off) for
    // diagnostic use only.
    let (sparse_pat_storage, dense_cols, sparse_to_orig) =
        if opts.dense_quotient_enabled && pattern.n > 0 {
            split_dense_columns(pattern, opts)?
        } else {
            (None, Vec::new(), Vec::new())
        };

    let perm = if let Some((cp, ri, sub_n)) = sparse_pat_storage.as_ref().map(|s| {
        let (cp, ri, sub_n) = s;
        (cp.as_slice(), ri.as_slice(), *sub_n)
    }) {
        // Run ND on the sparse-induced subgraph.
        let sub_pat = CscPattern::new(sub_n, cp, ri).ok_or(OrderingError::MalformedInput)?;
        let sub_perm = node_nd::nd_order(&sub_pat, opts, &mut stats)?;
        // Lift sub-perm back to original indices and append dense
        // columns at the end (in descending degree order — Davis &
        // Hager 2009 §3.2 ordering choice; ties broken by ascending
        // original index).
        let mut perm: Vec<Idx> = Vec::with_capacity(pattern.n);
        for &local in &sub_perm {
            let idx = local as usize;
            if idx >= sparse_to_orig.len() {
                return Err(OrderingError::Internal(
                    "dense-quotient: subgraph perm index out of range",
                ));
            }
            perm.push(sparse_to_orig[idx]);
        }
        for &c in &dense_cols {
            perm.push(c);
        }
        if perm.len() != pattern.n {
            return Err(OrderingError::Internal(
                "dense-quotient: assembled perm has wrong length",
            ));
        }
        perm
    } else {
        node_nd::nd_order(pattern, opts, &mut stats)?
    };

    let ordering_stats = OrderingStats {
        time_us: t0.elapsed().as_micros() as u64,
        fill_estimate: None,
        flop_estimate: None,
    };
    Ok((perm, ordering_stats, stats))
}

/// Resolve the dense-column threshold for an `n`-vertex graph.
///
/// `max(40, ceil(10 * sqrt(n)))` per Davis & Hager 2009 §3.2 and
/// MUMPS `ICNTL(6)` defaults. Honours the caller's override when
/// `opts.dense_quotient_threshold` is `Some(_)`.
fn resolve_dense_threshold(n: usize, opts: &MetisOptions) -> usize {
    if let Some(t) = opts.dense_quotient_threshold {
        return t;
    }
    let computed = (10.0 * (n as f64).sqrt()).ceil() as usize;
    computed.max(40)
}

/// Partition `pattern`'s columns into "dense" and "sparse" sets using
/// off-diagonal degree, and produce the CSC pattern of the
/// sparse-induced subgraph.
///
/// Returns:
/// - `Some((col_ptr, row_idx, sub_n))` carrying the induced
///   sub-pattern, plus the dense column list (in descending degree
///   order) and the `sparse_local → original` mapping. When the
///   dense set is empty, returns `(None, Vec::new(), Vec::new())` so
///   the caller can fast-path to the original pattern.
type DenseSplit = (Option<(Vec<Idx>, Vec<Idx>, usize)>, Vec<Idx>, Vec<Idx>);
fn split_dense_columns(
    pattern: &CscPattern<'_>,
    opts: &MetisOptions,
) -> Result<DenseSplit, OrderingError> {
    let n = pattern.n;
    let thresh = resolve_dense_threshold(n, opts);

    // Off-diagonal degree per column. The pattern is full-symmetric
    // with the diagonal optionally present; we count entries `r != c`.
    let mut deg: Vec<usize> = vec![0; n];
    for (c, d) in deg.iter_mut().enumerate() {
        let lo = pattern.col_ptr[c] as usize;
        let hi = pattern.col_ptr[c + 1] as usize;
        if hi < lo || hi > pattern.row_idx.len() {
            return Err(OrderingError::MalformedInput);
        }
        let mut acc: usize = 0;
        for k in lo..hi {
            let r = pattern.row_idx[k] as usize;
            if r != c {
                acc += 1;
            }
        }
        *d = acc;
    }

    // Collect dense columns.
    let mut dense: Vec<Idx> = (0..n)
        .filter(|&c| deg[c] > thresh)
        .map(|c| c as Idx)
        .collect();

    // No-op fast path: dense set empty.
    if dense.is_empty() {
        return Ok((None, Vec::new(), Vec::new()));
    }

    // Sort dense columns by *descending* degree, ties by ascending
    // original index — Davis & Hager 2009 §3.2: "eliminate the densest
    // last".
    dense.sort_by(|&a, &b| {
        deg[b as usize]
            .cmp(&deg[a as usize])
            .then_with(|| a.cmp(&b))
    });

    // Build the local-id maps for the sparse subgraph.
    //
    // `sparse_to_orig[local] = original`
    // `orig_to_local[original] = sparse local id, or -1 if dense`.
    let mut is_dense = vec![false; n];
    for &c in &dense {
        is_dense[c as usize] = true;
    }
    let mut sparse_to_orig: Vec<Idx> = Vec::with_capacity(n - dense.len());
    let mut orig_to_local: Vec<Idx> = vec![-1; n];
    for c in 0..n {
        if !is_dense[c] {
            orig_to_local[c] = sparse_to_orig.len() as Idx;
            sparse_to_orig.push(c as Idx);
        }
    }
    let sub_n = sparse_to_orig.len();

    // Build the induced CSC pattern. Re-include the diagonal entry so
    // downstream consumers (Graph::from_csc_pattern, AMD leaf) see a
    // well-formed pattern. Row indices stay sorted because we walk
    // each original column in ascending row order.
    let mut col_ptr: Vec<Idx> = Vec::with_capacity(sub_n + 1);
    let mut row_idx: Vec<Idx> = Vec::new();
    col_ptr.push(0);
    for &orig in &sparse_to_orig {
        let c = orig as usize;
        let lo = pattern.col_ptr[c] as usize;
        let hi = pattern.col_ptr[c + 1] as usize;
        let mut diag_inserted = false;
        let local_c = orig_to_local[c];
        for k in lo..hi {
            let r = pattern.row_idx[k] as usize;
            if r == c {
                // Diagonal handled below; skip here so we control its
                // placement (input may or may not carry the diagonal).
                continue;
            }
            let lr = orig_to_local[r];
            if lr < 0 {
                // Edge crosses into the dense set — drop it from the
                // sparse-induced subgraph; the dense column carries
                // that coupling and is eliminated at the end.
                continue;
            }
            if !diag_inserted && lr > local_c {
                row_idx.push(local_c);
                diag_inserted = true;
            }
            row_idx.push(lr);
        }
        if !diag_inserted {
            row_idx.push(local_c);
        }
        col_ptr.push(row_idx.len() as Idx);
    }

    Ok((Some((col_ptr, row_idx, sub_n)), dense, sparse_to_orig))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trivial_pattern() -> (Vec<Idx>, Vec<Idx>) {
        // Diagonal n=3: col_ptr=[0,1,2,3], row_idx=[0,1,2]
        (vec![0, 1, 2, 3], vec![0, 1, 2])
    }

    #[test]
    fn options_defaults_match_metis_5_2_0() {
        let o = MetisOptions::default();
        assert_eq!(o.niparts, 7);
        assert_eq!(o.coarsen_floor, 120);
        assert_eq!(o.nd_to_amd_switch, 200);
        assert_eq!(o.seed, 1);
    }

    #[test]
    fn stats_default_is_zeros() {
        let s = MetisStats::default();
        assert_eq!(s.n_levels, 0);
        assert_eq!(s.n_components, 0);
        assert_eq!(s.n_separator_vertices, 0);
        assert_eq!(s.n_fm_passes, 0);
        assert_eq!(s.n_two_hop_fallbacks, 0);
        assert_eq!(s.n_amd_leaf_calls, 0);
    }

    #[test]
    fn diagonal_pattern_yields_permutation() {
        let (cp, ri) = trivial_pattern();
        let pat = CscPattern::new(3, &cp, &ri).unwrap();
        let (perm, ostats, _mstats) = metis_order_full(&pat, &MetisOptions::default()).expect("ok");
        assert_eq!(perm.len(), 3);
        let mut seen = [false; 3];
        for &p in &perm {
            assert!((0..3).contains(&p));
            seen[p as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
        // time_us is populated; fill/flop remain None.
        assert!(ostats.fill_estimate.is_none());
        assert!(ostats.flop_estimate.is_none());
    }

    #[test]
    fn convenience_wrapper_returns_permutation() {
        let (cp, ri) = trivial_pattern();
        let pat = CscPattern::new(3, &cp, &ri).unwrap();
        let perm = metis_order(&pat).expect("ok");
        assert_eq!(perm.len(), 3);
    }

    #[test]
    fn contract_version_matches_core() {
        assert_eq!(CONTRACT_VERSION, feral_ordering_core::CONTRACT_VERSION);
    }
}
