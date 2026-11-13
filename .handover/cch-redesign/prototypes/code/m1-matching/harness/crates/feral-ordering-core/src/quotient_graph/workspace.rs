//! Quotient-graph workspace and initialization, shared by AMD-family
//! bottom-up orderings.
//!
//! All arrays follow the faer / SuiteSparse naming convention and use
//! signed `Idx` so we can reserve negative values for sentinels via
//! [`flip`]: `flip(x) = -2 - x`. The input pattern is also `Idx`-
//! indexed (per the ordering-crate contract), so workspace ingestion
//! converts to `usize` only for Rust slice addressing.
//!
//! Builds the workspace arrays from a full-symmetric CSC pattern,
//! runs the two initialization fast paths (zero-degree
//! pre-elimination and dense-deferred bucket), and seats the
//! remaining variables into degree-indexed linked lists ready for
//! the elimination loop.
//!
//! Migrated from `feral-amd` in 2026-04-27 to host the shared
//! machinery for the planned `feral-amf` crate; the AMD-vs-AMF
//! delta lives entirely in the elimination metric (see
//! `dev/research/amf-clean-room.md`).
//
// Items are consumed by the elimination loop in subsequent commits
// (Commit 4 onwards). Until then several fields and helpers are
// intentionally unused.
#![allow(dead_code)]

use super::WorkspaceOptions;
use crate::Idx;
use crate::{CscPattern, OrderingError};

/// Sentinel for "no index" in the `Idx` arrays.
pub const NONE: Idx = -1;

/// Sentinel encoding used by the quotient graph: `flip(x) = -2 - x`.
/// Used to mark absorbed elements (`pe[e] < 0` ⇒ `flip(parent)`) and
/// as a tag on `elen` for freshly eliminated zero-degree variables.
#[inline(always)]
pub fn flip(x: Idx) -> Idx {
    -2 - x
}

/// Reset the generation counter `wflg` without visiting `iw`.
///
/// Called when `wflg` would overflow the `wbig` ceiling, or when it
/// drops below 2 (which should not happen in practice, but matches
/// SuiteSparse AMD's defensive check). All nonzero entries in `w`
/// are clamped to `1`, and `2` is returned as the fresh counter.
///
/// Reference: `amd.rs:130-143`.
#[inline]
pub fn clear_flag(wflg: Idx, wbig: Idx, w: &mut [Idx]) -> Idx {
    if wflg < 2 || wflg >= wbig {
        for x in w.iter_mut() {
            if *x != 0 {
                *x = 1;
            }
        }
        return 2;
    }
    wflg
}

/// In-memory workspace for one AMD run.
///
/// The fields mirror faer's `amd_2` locals. Ownership of every buffer
/// is held here so the elimination loop can borrow them concurrently
/// through split borrows without reallocation.
#[derive(Debug)]
pub struct Workspace {
    pub n: usize,
    pub iwlen: usize,
    pub pfree: usize,
    pub iw: Vec<Idx>,

    pub pe: Vec<Idx>,
    pub len: Vec<Idx>,
    pub nv: Vec<Idx>,
    pub elen: Vec<Idx>,
    pub degree: Vec<Idx>,
    pub w: Vec<Idx>,
    pub head: Vec<Idx>,
    pub next: Vec<Idx>,
    pub last: Vec<Idx>,
    /// Per-supervariable / per-element fill-score scratch used by the
    /// AMF inner loop only. Length `n`. Ignored by the AMD path
    /// (`feral-amd` never reads or writes it). For variable indices
    /// `i`, holds the running quantized RMF score; for element indices
    /// `e`, holds the lazily-cached `dext * (2*deg(e) - dext - 1)`
    /// surface contribution (sentinel `0` = "first touch this iter").
    ///
    /// `i64` (not upstream's `i32`): the un-quantized surface contribution has
    /// both factors `O(n)`, so it reaches ~`n^2` and overflows `i32`
    /// for `n` ≳ 46k before being consumed as `f64` in the RMF score
    /// (O1, `dev/research/repo-review-2026-06-09.md`). MUMPS computes
    /// the RMF in DBLE for the same reason. The post-quantization RMF
    /// score stored here later is bounded by `Idx::MAX - 1`.
    pub wf: Vec<i64>,

    /// Generation counter for the mark array `w`.
    pub wflg: Idx,
    /// Overflow ceiling for `wflg`: `Idx::MAX - n`.
    pub wbig: Idx,
    /// Largest element size encountered so far — used by supervariable
    /// detection (Slice B) to bump `wflg` safely.
    pub lemax: Idx,
    /// Lower bound on the next pivot's degree. Monotone non-decreasing.
    pub mindeg: usize,
    /// Number of garbage-collection compactions so far.
    pub ncmpa: u32,
    /// Supervariables eliminated so far (pivoted OR dense-deferred).
    pub nel: usize,
    /// Dense-deferred supervariable count.
    pub ndense: Idx,
    /// Variables folded into a concurrent pivot by mass elimination.
    pub n_mass_elim: u32,
    /// Supervariable merges detected during indistinguishable-variable
    /// consolidation.
    pub n_supervar_merge: u32,
}

impl Workspace {
    /// Build a workspace from a full-symmetric CSC pattern and run
    /// initialization. On return, all variables have been classified
    /// into one of three buckets:
    ///
    /// 1. **Zero-degree** (`deg == 0`) — pre-eliminated. `pe[i] = NONE`,
    ///    `elen[i] = flip(1)`, `w[i] = 0`, `nel` incremented.
    /// 2. **Dense-deferred** (`deg > dense`) — moved to the dense tail.
    ///    `pe[i] = NONE`, `nv[i] = 0`, `elen[i] = NONE`, `nel` incremented.
    /// 3. **Live** — inserted LIFO into the degree-indexed linked list
    ///    headed by `head[deg]`, threaded through `next`/`last`.
    ///
    /// `pattern` must be the full-symmetric graph (both halves). The
    /// diagonal is ignored if present.
    pub fn new(
        pattern: &CscPattern<'_>,
        opts: &WorkspaceOptions,
    ) -> Result<Workspace, OrderingError> {
        Self::new_with_n_buckets(pattern, opts, pattern.n)
    }

    /// Variant of [`Workspace::new`] that allocates `head` with the
    /// caller-supplied bucket count. Used by AMF, where the quantized
    /// fill score can exceed `n` and the head array must extend up to
    /// `NBBUCK + 1 = 2 * n + 1`. AMD always passes `pattern.n`, which
    /// makes this byte-equivalent to [`Workspace::new`].
    ///
    /// `n_buckets` must be at least `n` so the init insertion at
    /// `head[deg]` (with `deg ≤ dense ≤ n`) is in range.
    pub fn new_with_n_buckets(
        pattern: &CscPattern<'_>,
        opts: &WorkspaceOptions,
        n_buckets: usize,
    ) -> Result<Workspace, OrderingError> {
        let n = pattern.n;
        debug_assert!(n_buckets >= n, "n_buckets must cover deg ∈ [0, n)");

        // Idx addressing requires n < Idx::MAX. The algorithm also
        // stores `pfree` as Idx via `pe[i]`, so iwlen must fit.
        if n >= Idx::MAX as usize {
            return Err(OrderingError::IndexOverflow);
        }

        // Count off-diagonal entries per column.
        let mut len: Vec<Idx> = vec![0; n];
        let mut nzaat: usize = 0;
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            let j_i32 = j as Idx;
            let start = pattern.col_ptr[j] as usize;
            let end = pattern.col_ptr[j + 1] as usize;
            let mut cnt: usize = 0;
            for &r in &pattern.row_idx[start..end] {
                if r != j_i32 {
                    cnt += 1;
                }
            }
            len[j] = cnt as Idx;
            nzaat += cnt;
        }

        // iwlen = nzaat + nzaat/5 + n  (plan A1 / faer amd.rs:921-924).
        let iwlen = nzaat
            .checked_add(nzaat / 5)
            .and_then(|s| s.checked_add(n))
            .ok_or(OrderingError::IndexOverflow)?;
        if iwlen > Idx::MAX as usize {
            return Err(OrderingError::IndexOverflow);
        }
        // iw needs at least one slot even when n==0 so `pfree` is
        // addressable; we allocate exactly iwlen.
        let mut iw: Vec<Idx> = vec![0; iwlen];

        // pe[j] = start of j's adjacency list in iw; fill iw with
        // off-diagonals, in the order they appear in the CSC pattern.
        let mut pe: Vec<Idx> = vec![0; n];
        let mut pfree: usize = 0;
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            pe[j] = pfree as Idx;
            let j_i32 = j as Idx;
            let start = pattern.col_ptr[j] as usize;
            let end = pattern.col_ptr[j + 1] as usize;
            for &r in &pattern.row_idx[start..end] {
                if r != j_i32 {
                    iw[pfree] = r;
                    pfree += 1;
                }
            }
        }
        debug_assert_eq!(pfree, nzaat);

        // Dense threshold. alpha < 0 disables dense deferral.
        let dense = if opts.dense_alpha < 0.0 {
            n.saturating_sub(2)
        } else {
            (opts.dense_alpha * (n as f64).sqrt()) as usize
        };
        let dense = dense.max(16).min(n);

        // Fixed-value arrays.
        let mut nv: Vec<Idx> = vec![1; n];
        let mut elen: Vec<Idx> = vec![0; n];
        let mut w: Vec<Idx> = vec![1; n];
        let degree: Vec<Idx> = len.clone();
        let mut head: Vec<Idx> = vec![NONE; n_buckets];
        let mut next: Vec<Idx> = vec![NONE; n];
        let mut last: Vec<Idx> = vec![NONE; n];
        let mut wf: Vec<i64> = vec![0; n];

        let wbig = Idx::MAX - n as Idx;
        let wflg = 0; // clear_flag will lift to 2 on first use.

        let mut nel: usize = 0;
        let mut ndense: Idx = 0;

        // Classify each variable.
        for i in 0..n {
            let deg = degree[i] as usize;
            if deg == 0 {
                // Zero-degree fast path — pre-eliminated.
                elen[i] = flip(1);
                nel += 1;
                pe[i] = NONE;
                w[i] = 0;
            } else if deg > dense {
                // Dense-deferred fast path.
                ndense += 1;
                nv[i] = 0;
                elen[i] = NONE;
                pe[i] = NONE;
                nel += 1;
            } else {
                // LIFO head-insert at head[deg]. The AMF metric's
                // `bucket(deg, n)` is identity for `deg ≤ n`, so the
                // index `deg` is the right slot for both AMD and AMF
                // at init time. Seed the AMF fill score so the AMF
                // path can compute `bucket(wf[i], n)` consistently.
                let inext = head[deg];
                if inext != NONE {
                    last[inext as usize] = i as Idx;
                }
                next[i] = inext;
                head[deg] = i as Idx;
                wf[i] = deg as i64;
            }
        }

        Ok(Workspace {
            n,
            iwlen,
            pfree,
            iw,
            pe,
            len,
            nv,
            elen,
            degree,
            w,
            head,
            next,
            last,
            wf,
            wflg,
            wbig,
            lemax: 0,
            mindeg: 0,
            ncmpa: 0,
            nel,
            ndense,
            n_mass_elim: 0,
            n_supervar_merge: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pat<'a>(n: usize, cp: &'a [Idx], ri: &'a [Idx]) -> CscPattern<'a> {
        CscPattern::new(n, cp, ri).expect("valid test pattern")
    }

    #[test]
    fn flip_involution() {
        for x in [-100 as Idx, -1, 0, 1, 17, 1000] {
            assert_eq!(flip(flip(x)), x);
        }
    }

    #[test]
    fn clear_flag_resets_on_overflow() {
        let mut w = [0, 3, 5, 7, 0];
        // wflg >= wbig triggers reset.
        let wflg = clear_flag(100, 100, &mut w);
        assert_eq!(wflg, 2);
        assert_eq!(w, [0, 1, 1, 1, 0]);
    }

    #[test]
    fn clear_flag_passthrough() {
        let mut w = [1, 2, 3];
        let wflg = clear_flag(5, 100, &mut w);
        assert_eq!(wflg, 5);
        assert_eq!(w, [1, 2, 3]);
    }

    /// Diagonal 4x4: every variable has degree 0. All four are
    /// pre-eliminated by the zero-degree fast path; none are
    /// inserted into any degree list.
    #[test]
    fn diag_4_zero_degree_fast_path() {
        let cp = [0, 1, 2, 3, 4];
        let ri = [0, 1, 2, 3];
        let p = pat(4, &cp, &ri);
        let ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();

        assert_eq!(ws.n, 4);
        assert_eq!(ws.nel, 4, "all four pre-eliminated");
        assert_eq!(ws.ndense, 0, "no dense deferral");
        for i in 0..4 {
            assert_eq!(ws.elen[i], flip(1));
            assert_eq!(ws.pe[i], NONE);
            assert_eq!(ws.w[i], 0);
            assert_eq!(ws.degree[i], 0);
        }
        for d in 0..4 {
            assert_eq!(ws.head[d], NONE, "degree {d} bucket empty");
        }
    }

    /// Tridiagonal 5x5 full-symmetric: var 0 has deg 1, 4 has deg 1,
    /// interior vars have deg 2. No dense deferral (max deg 2 < 16).
    /// All five enter degree lists.
    #[test]
    fn tridiag_5_populates_degree_lists() {
        // Full-symmetric tridiag of size 5.
        // column j contains diagonal + off-diagonals j-1 and j+1.
        let cp = [0, 2, 5, 8, 11, 13];
        let ri = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let p = pat(5, &cp, &ri);
        let ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();

        assert_eq!(ws.n, 5);
        assert_eq!(ws.nel, 0, "no fast-path eliminations");
        assert_eq!(ws.ndense, 0);
        assert_eq!(ws.degree, vec![1, 2, 2, 2, 1]);
        assert_eq!(ws.len, vec![1, 2, 2, 2, 1]);

        // LIFO insertion: the last variable inserted into deg list d
        // is at head[d]. For deg=1, vars 0 and 4 hit the bucket;
        // last in is 4. For deg=2, vars 1,2,3; last in is 3.
        assert_eq!(ws.head[1], 4);
        assert_eq!(ws.next[4], 0);
        assert_eq!(ws.next[0], NONE);
        assert_eq!(ws.last[0], 4);

        assert_eq!(ws.head[2], 3);
        assert_eq!(ws.next[3], 2);
        assert_eq!(ws.next[2], 1);
        assert_eq!(ws.next[1], NONE);
        assert_eq!(ws.last[1], 2);
        assert_eq!(ws.last[2], 3);
        assert_eq!(ws.last[3], NONE);

        // iwlen = nzaat + nzaat/5 + n.
        // nzaat = sum(len) = 1+2+2+2+1 = 8.
        // iwlen = 8 + 1 + 5 = 14.
        assert_eq!(ws.iwlen, 14);
        assert_eq!(ws.pfree, 8);
    }

    /// Arrow(5): hub at 0 has deg 4, spokes have deg 1. Dense
    /// threshold for n=5 is min(n, max(16, 10*sqrt(5))) = 5; deg 4
    /// < 5, so nothing is deferred. Hub enters head[4].
    #[test]
    fn arrow_5_all_live() {
        let cp = [0, 5, 7, 9, 11, 13];
        let ri = [0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4];
        let p = pat(5, &cp, &ri);
        let ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();

        assert_eq!(ws.degree, vec![4, 1, 1, 1, 1]);
        assert_eq!(ws.nel, 0);
        assert_eq!(ws.ndense, 0);
        assert_eq!(ws.head[4], 0, "hub at deg-4 bucket");
        assert_eq!(ws.head[1], 4, "last spoke inserted");
    }

    /// Arrow(200): hub has deg 199. Dense threshold = max(16,
    /// floor(10*sqrt(200))) = max(16, 141) = 141, min(141, 200) =
    /// 141. 199 > 141 so the hub is deferred. Spokes (deg 1) live.
    #[test]
    fn arrow_200_hub_deferred() {
        let n = 200usize;
        let mut cp: Vec<Idx> = Vec::with_capacity(n + 1);
        let mut ri: Vec<Idx> = Vec::new();
        cp.push(0);
        // col 0: diagonal + all spokes 1..n
        ri.push(0);
        for r in 1..n {
            ri.push(r as Idx);
        }
        cp.push(ri.len() as Idx);
        // cols 1..n: diagonal + hub
        for j in 1..n {
            ri.push(0);
            ri.push(j as Idx);
            cp.push(ri.len() as Idx);
        }
        let p = pat(n, &cp, &ri);
        let ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();

        assert_eq!(ws.degree[0], (n - 1) as Idx);
        assert_eq!(ws.nel, 1, "hub only");
        assert_eq!(ws.ndense, 1);
        assert_eq!(ws.nv[0], 0, "hub marked deferred");
        assert_eq!(ws.pe[0], NONE);
        assert_eq!(ws.elen[0], NONE);
        // Spokes all land in head[1] — LIFO, last in is n-1.
        assert_eq!(ws.head[1], (n - 1) as Idx);
    }

    /// With `dense_alpha < 0`, the threshold is set to `n - 2`
    /// (faer amd.rs:173-177). A variable with degree `n - 1` (only
    /// possible for a true hub) is still deferred; everything else
    /// stays live. This matches SuiteSparse AMD semantics exactly.
    #[test]
    fn dense_alpha_negative_uses_n_minus_2() {
        // Band(20, 5): max degree = 10, well under n - 2 = 18.
        let n = 20usize;
        let b = 5usize;
        let mut cp: Vec<Idx> = vec![0];
        let mut ri: Vec<Idx> = Vec::new();
        for j in 0..n {
            let lo = j.saturating_sub(b);
            let hi = (j + b + 1).min(n);
            for r in lo..hi {
                ri.push(r as Idx);
            }
            cp.push(ri.len() as Idx);
        }
        let p = pat(n, &cp, &ri);
        let opts = WorkspaceOptions { dense_alpha: -1.0 };
        let ws = Workspace::new(&p, &opts).unwrap();
        assert_eq!(ws.ndense, 0, "nothing deferred below n - 2");
        assert_eq!(ws.nel, 0);
    }

    #[test]
    fn empty_pattern_ok() {
        let cp = [0 as Idx];
        let ri: [Idx; 0] = [];
        let p = pat(0, &cp, &ri);
        let ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        assert_eq!(ws.n, 0);
        assert_eq!(ws.nel, 0);
        assert_eq!(ws.iwlen, 0);
    }

    /// Diagonal entries in the input are ignored when computing
    /// adjacency lists — only off-diagonal neighbors contribute to
    /// `len`, `degree`, and `iw`.
    #[test]
    fn diagonal_entries_skipped() {
        // 3x3 with diagonal only.
        let cp = [0, 1, 2, 3];
        let ri = [0, 1, 2];
        let p = pat(3, &cp, &ri);
        let ws = Workspace::new(&p, &WorkspaceOptions::default()).unwrap();
        assert_eq!(ws.len, vec![0, 0, 0]);
        assert_eq!(ws.pfree, 0);
        assert_eq!(ws.nel, 3);
    }
}
