//! Selection-metric trait abstracting AMD vs AMF differences.
//!
//! The shared quotient-graph machinery (`Workspace`, `select_pivot`,
//! `create_element`, `finalize_step`, hash-based supervariable
//! detection, mass elimination, aggressive absorption) is identical
//! across AMD and AMF. The metrics differ in:
//!
//! 1. **Initial score** seeded from each row's adjacency length.
//!    AMD: identity. AMF: identity (both = `len`).
//! 2. **Bucket array length.** AMD: `n` (degrees `0..n` indexable;
//!    `select_pivot` scans `while deg < n`). AMF: `2 * n + 2` because
//!    the quantized RMF can exceed `n`.
//! 3. **Bucket index for a score.** AMD: identity. AMF: identity for
//!    `s ≤ n`, then coarse stride `PAS = max(n / 8, 1)` above.
//! 4. **Pivot selection within a bucket.** AMD: head only.
//!    AMF: linear scan when the bucket is in the coarse region.
//! 5. **Score on supervariable merge.** AMD: no-op (only `nv[i]`
//!    accumulates). AMF: `score[i] = max(score[i], score[j])`.
//! 6. **Score finalisation** at the end of Pass-2: AMD's loose-degree
//!    formula `min(deg_prev, scan2_deg) + degme - nvi` clamped at
//!    `nleft - nvi`; AMF's quantized RMF (Amestoy 1999 thesis).
//!
//! The trait below covers (1)–(5). Site (6) is metric-specific in the
//! Pass-2 inner loop and is reached via the `run_elimination`
//! dispatch — each metric impl wires its own concrete loop in
//! `algo.rs` (AMD: `run_elimination`, AMF: `run_elimination_amf`).
//! The trait stays light: keeping the inner loops as parallel
//! concrete functions trades ~300 LoC duplication for zero risk to
//! the AMD bit-parity contract.
//!
//! Reference: `dev/research/amf-clean-room.md` Section 6.

use super::algo::{run_elimination as run_elimination_amd, run_elimination_amf, StepFlops};
use super::workspace::Workspace;
use crate::Idx;
use crate::OrderingError;

/// Selection metric for an AMD-family bottom-up ordering.
///
/// All methods are zero-overhead `#[inline(always)]` no-ops or
/// identity functions in the AMD case; AMF (`MinFill`) provides
/// non-trivial implementations. The trait is consumed at the
/// `run_elimination` dispatch point and at the bucket-allocation
/// dispatch in [`crate::quotient_graph::order`]; the metric-specific
/// inner-loop sites are inlined into the concrete `run_elimination_*`
/// functions in `algo.rs`.
pub trait Metric {
    /// Bucket key produced by the selection metric. AMD uses `Idx`
    /// (the running degree); AMF will also use `Idx` (quantized RMF).
    type Score: Copy + Ord + Default;

    /// Length of the bucket head array `Workspace::head`. AMD: `n`
    /// (indexed up to `n - 1` by `select_pivot`'s `while deg < n`).
    /// AMF: `2 * n + 2`.
    fn n_buckets(n: usize) -> usize;

    /// Initial score for a freshly-loaded variable with adjacency
    /// length `len`. AMD and AMF both seed `len`.
    fn init_score(len: Idx) -> Self::Score;

    /// Bucket index for the given score. AMD: identity. AMF: identity
    /// for `s ≤ n`, coarse-stride above.
    fn bucket(score: Self::Score, n: usize) -> usize;

    /// Whether `idx` falls in the "coarse" bucket region — i.e.
    /// `select_pivot` must linear-scan the bucket chain to pick the
    /// minimum-score entry, rather than just taking the head. AMD
    /// always returns `false`; AMF returns `idx > n`.
    fn coarse_bucket(idx: usize, n: usize) -> bool;

    /// Update `parent`'s score on supervariable merge of `child` into
    /// `parent`. AMD: no-op. AMF: `*parent = max(*parent, child)`.
    fn merge_supervariable(parent: &mut Self::Score, child: Self::Score);

    /// Run the metric's elimination loop on a freshly initialised
    /// `Workspace`. Returns the accumulated flop counters.
    ///
    /// MinDegree dispatches to `run_elimination` (the AMD-specific
    /// loop); MinFill dispatches to `run_elimination_amf`.
    fn run_elimination(ws: &mut Workspace, aggressive: bool) -> Result<StepFlops, OrderingError>;
}

/// Minimum-degree metric — the AMD selection rule of Amestoy, Davis,
/// Duff (1996).
///
/// Score is the running degree. Bucket index is the score itself.
/// All buckets are "fine" (head-only pivot selection). Supervariable
/// merge does not update the score (AMD tracks degree only via
/// `nv[i]` and the per-iteration Pass-2 monotone cap).
#[derive(Debug, Clone, Copy, Default)]
pub struct MinDegree;

impl Metric for MinDegree {
    type Score = Idx;

    #[inline(always)]
    fn n_buckets(n: usize) -> usize {
        n
    }

    #[inline(always)]
    fn init_score(len: Idx) -> Idx {
        len
    }

    #[inline(always)]
    fn bucket(score: Idx, _n: usize) -> usize {
        score as usize
    }

    #[inline(always)]
    fn coarse_bucket(_idx: usize, _n: usize) -> bool {
        false
    }

    #[inline(always)]
    fn merge_supervariable(_parent: &mut Idx, _child: Idx) {
        // AMD does not maintain a per-supervariable score; degree
        // bookkeeping flows entirely through `nv[i]` and the
        // re-insertion loop's loose-degree formula.
    }

    #[inline(always)]
    fn run_elimination(ws: &mut Workspace, aggressive: bool) -> Result<StepFlops, OrderingError> {
        run_elimination_amd(ws, aggressive)
    }
}

/// Approximate Minimum Fill metric (HAMF4) — Amestoy 1999 thesis.
///
/// AMF selects the next pivot to minimise the *fill* introduced by
/// the elimination, rather than the candidate's degree. On bipartite-
/// KKT graphs with a few "hub" rows AMF can be 47× better than AMD on
/// final `nnz_L` (see `dev/research/amf-clean-room.md` Section 1).
///
/// Score is a quantized `RMF = DEG*(DEG-1+2*DEGME) - WF(i)` value
/// stored in `Idx`. Buckets up to and including `NORIG = n` are one
/// bucket per integer score; above `NORIG` the buckets quantize with
/// stride `PAS = max(n / 8, 1)` and the head must be linear-scanned
/// to pick the minimum-RMF entry. Supervariable absorption merges
/// the per-supervariable WF with `max`.
///
/// **Inner loop**: [`MinFill::run_elimination`] dispatches to
/// `run_elimination_amf` (Phase B.2 of `dev/plans/amf-clean-room.md`).
/// The lazy WF(e) cache, three-accumulator Pass-2, supervariable
/// max-merge of `wf`, saturated/regular RMF branch, and coarse-bucket
/// linear scan all live in `algo.rs`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MinFill;

impl Metric for MinFill {
    type Score = Idx;

    /// AMF needs `2 * n + 2` slots: `0..=NORIG` for one-per-score
    /// fine buckets, `NORIG+1..=NBBUCK` (`NBBUCK = 2 * n`) for
    /// coarse-stride buckets, and one halo slot at `NBBUCK + 1`
    /// reserved for V1 boundary variables (inert in our use case;
    /// see Section 11 of `dev/research/amf-clean-room.md`).
    #[inline(always)]
    fn n_buckets(n: usize) -> usize {
        2 * n + 2
    }

    /// Seed the per-supervariable score with the row's adjacency
    /// length `len(i)`. Same seeding as AMD; the AMF metric only
    /// diverges once the elimination loop starts producing elements.
    #[inline(always)]
    fn init_score(len: Idx) -> Idx {
        len
    }

    /// Quantize `score` into a bucket index.
    ///
    /// `0..=n` are fine buckets (one per integer score). Above `n`
    /// the buckets are coarse with stride `PAS = max(n / 8, 1)`,
    /// capped at `NBBUCK = 2 * n`. Negative scores (which should not
    /// occur in a well-formed AMF run, but we defend against
    /// truncation underflow on `RMF / (NVI + 1)`) clamp to bucket 0.
    ///
    /// Reference: `ana_orderings.F:4954-5017` and
    /// `dev/research/amf-clean-room.md` Section 4.
    #[inline]
    fn bucket(score: Idx, n: usize) -> usize {
        if score <= 0 {
            return 0;
        }
        let s = score as usize;
        if s <= n {
            return s;
        }
        let pas = (n / 8).max(1);
        let nbbuck = 2 * n;
        let coarse = (s - n) / pas + n;
        coarse.min(nbbuck)
    }

    /// Coarse buckets are those above `NORIG = n`. `select_pivot`
    /// must walk the bucket chain and pick the entry with the
    /// smallest *exact* score (`ana_orderings.F:4392-4418`).
    #[inline(always)]
    fn coarse_bucket(idx: usize, n: usize) -> bool {
        idx > n
    }

    /// On supervariable merge `j → i`, update the surviving anchor's
    /// score with `max(WF(i), WF(j))` (`ana_orderings.F:4920`).
    #[inline(always)]
    fn merge_supervariable(parent: &mut Idx, child: Idx) {
        if child > *parent {
            *parent = child;
        }
    }

    fn run_elimination(ws: &mut Workspace, aggressive: bool) -> Result<StepFlops, OrderingError> {
        run_elimination_amf(ws, aggressive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_degree_n_buckets_matches_workspace_alloc() {
        // Workspace::new allocates head of length `n`; MinDegree
        // must agree so the AMD code path indexes the right region.
        for n in [0usize, 1, 5, 100, 10_000] {
            assert_eq!(MinDegree::n_buckets(n), n);
        }
    }

    #[test]
    fn min_degree_init_score_is_identity() {
        for len in [0 as Idx, 1, 17, 1024] {
            assert_eq!(MinDegree::init_score(len), len);
        }
    }

    #[test]
    fn min_degree_bucket_is_identity() {
        for s in [0 as Idx, 1, 7, 100] {
            assert_eq!(MinDegree::bucket(s, 200), s as usize);
        }
    }

    #[test]
    fn min_degree_no_coarse_buckets() {
        for n in [10usize, 100, 10_000] {
            for idx in [0usize, 1, n / 2, n - 1] {
                assert!(!MinDegree::coarse_bucket(idx, n));
            }
        }
    }

    #[test]
    fn min_degree_merge_does_not_touch_parent() {
        let mut parent: Idx = 42;
        MinDegree::merge_supervariable(&mut parent, 7);
        assert_eq!(parent, 42, "AMD merge is a true no-op on the score");
    }

    #[test]
    fn min_fill_n_buckets_is_2n_plus_2() {
        // NBBUCK = 2*n, plus the +1 head index plus the V1 halo slot.
        for n in [0usize, 1, 5, 100, 10_000] {
            assert_eq!(MinFill::n_buckets(n), 2 * n + 2);
        }
    }

    #[test]
    fn min_fill_init_score_is_len() {
        for len in [0 as Idx, 1, 17, 1024] {
            assert_eq!(MinFill::init_score(len), len);
        }
    }

    /// Fine bucket region: scores `0..=n` map identity.
    #[test]
    fn min_fill_bucket_fine_region_is_identity() {
        let n = 100usize;
        for s in [0 as Idx, 1, 50, 99, 100] {
            assert_eq!(MinFill::bucket(s, n), s as usize);
        }
    }

    /// Coarse bucket region: scores above `n` quantize with
    /// stride `PAS = max(n/8, 1)`.
    #[test]
    fn min_fill_bucket_coarse_region_quantizes_with_pas() {
        let n = 100usize;
        // PAS = 100 / 8 = 12.
        // bucket(101) = (101 - 100) / 12 + 100 = 0 + 100 = 100.
        // bucket(112) = (112 - 100) / 12 + 100 = 1 + 100 = 101.
        // bucket(113) = (113 - 100) / 12 + 100 = 1 + 100 = 101 (same coarse bin).
        // bucket(124) = (124 - 100) / 12 + 100 = 2 + 100 = 102.
        assert_eq!(MinFill::bucket(101, n), 100);
        assert_eq!(MinFill::bucket(112, n), 101);
        assert_eq!(MinFill::bucket(113, n), 101);
        assert_eq!(MinFill::bucket(124, n), 102);
    }

    /// Very large scores cap at `NBBUCK = 2 * n`.
    #[test]
    fn min_fill_bucket_caps_at_nbbuck() {
        let n = 100usize;
        let nbbuck = 2 * n;
        // bucket(1_000_000) saturates to NBBUCK.
        assert_eq!(MinFill::bucket(1_000_000, n), nbbuck);
        assert_eq!(MinFill::bucket(Idx::MAX, n), nbbuck);
    }

    /// Small `n` falls through to `PAS = 1` so coarse buckets are
    /// effectively per-integer above `n`.
    #[test]
    fn min_fill_bucket_pas_is_at_least_one() {
        // n = 4, PAS = max(0, 1) = 1.
        // bucket(5, 4) = (5 - 4) / 1 + 4 = 5.
        // bucket(6, 4) = (6 - 4) / 1 + 4 = 6.
        // bucket(8, 4) = (8 - 4) / 1 + 4 = 8 = NBBUCK; cap.
        // bucket(9, 4) caps at NBBUCK = 8.
        assert_eq!(MinFill::bucket(5, 4), 5);
        assert_eq!(MinFill::bucket(6, 4), 6);
        assert_eq!(MinFill::bucket(8, 4), 8);
        assert_eq!(MinFill::bucket(9, 4), 8);
    }

    /// Negative or zero scores clamp to bucket 0 (defensive — the
    /// AMF math should never produce them after the `RMF / (NVI + 1)`
    /// division but the saturated-RMF branch can underflow on tiny
    /// problems).
    #[test]
    fn min_fill_bucket_clamps_nonpositive() {
        assert_eq!(MinFill::bucket(0, 100), 0);
        assert_eq!(MinFill::bucket(-1, 100), 0);
        assert_eq!(MinFill::bucket(Idx::MIN, 100), 0);
    }

    /// Coarse bucket region is exactly `idx > n`.
    #[test]
    fn min_fill_coarse_bucket_threshold() {
        let n = 100usize;
        for idx in [0usize, 50, 99, 100] {
            assert!(!MinFill::coarse_bucket(idx, n), "{idx} <= n is fine");
        }
        for idx in [101usize, 150, 200, 201] {
            assert!(MinFill::coarse_bucket(idx, n), "{idx} > n is coarse");
        }
    }

    /// Supervariable merge takes the max — the larger of the two
    /// fill estimates becomes the merged score.
    #[test]
    fn min_fill_merge_takes_max() {
        let mut parent: Idx = 10;
        MinFill::merge_supervariable(&mut parent, 25);
        assert_eq!(parent, 25, "child larger ⇒ adopt child");

        let mut parent: Idx = 100;
        MinFill::merge_supervariable(&mut parent, 7);
        assert_eq!(parent, 100, "child smaller ⇒ keep parent");

        let mut parent: Idx = 42;
        MinFill::merge_supervariable(&mut parent, 42);
        assert_eq!(parent, 42, "equal ⇒ unchanged");
    }

    /// MinFill's elimination now runs the real AMF inner loop on a
    /// workspace allocated with the AMF bucket count `2 * n + 2`.
    /// Smoke test on a 2-variable pattern: must succeed and reach
    /// `nel == n`.
    #[test]
    fn min_fill_run_elimination_completes() {
        use crate::quotient_graph::WorkspaceOptions;
        use crate::CscPattern;
        let cp = [0 as Idx, 1, 2];
        let ri = [0 as Idx, 1];
        let p = CscPattern::new(2, &cp, &ri).unwrap();
        let mut ws =
            Workspace::new_with_n_buckets(&p, &WorkspaceOptions::default(), MinFill::n_buckets(2))
                .unwrap();
        MinFill::run_elimination(&mut ws, true).expect("AMF loop runs");
        assert_eq!(ws.nel, ws.n);
    }
}
