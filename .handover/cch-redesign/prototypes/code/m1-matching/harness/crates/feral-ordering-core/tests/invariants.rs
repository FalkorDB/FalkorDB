//! Functional invariants of the quotient-graph ordering driver,
//! parameterised over the [`Metric`] trait.
//!
//! Per `dev/plans/amf-clean-room.md` Phase B deliverable 5: every
//! `Metric` impl that successfully drives `order` must produce a
//! permutation of `0..n`. These tests pin that contract so Phase
//! B.2's MinFill inner loop has a passing oracle to drive against.
//!
//! Phase B.2 has wired up the AMF inner loop, so every `Metric` arm
//! now exercises the real `run_elimination`. The five fixtures hold
//! both `MinDegree` and `MinFill` to the same generic invariant
//! (output is a permutation of `0..n`).

use feral_ordering_core::quotient_graph::{order, Metric, MinDegree, MinFill, WorkspaceOptions};
use feral_ordering_core::CscPattern;

// ---------------------------------------------------------------------
// Fixture builders. Each returns owned (col_ptr, row_idx) so the test
// can borrow them into a `CscPattern` without lifetime juggling.
// ---------------------------------------------------------------------

/// 3×3 arrowhead with hub at row 0:
///
/// ```text
/// * * *
/// * *
/// * . *
/// ```
///
/// Used by both AMD and AMF unit tests; AMF must defer the hub to
/// the end of the elimination order (Section 9 of the research
/// note's small-matrix derivation).
fn arrow_3() -> (Vec<i64>, Vec<i64>) {
    // Full-symmetric, including diagonal entries (which the workspace
    // ingest filters out).
    let col_ptr = vec![0, 3, 5, 7];
    let row_idx = vec![0, 1, 2, 0, 1, 0, 2];
    (col_ptr, row_idx)
}

/// 5×5 dual-arrowhead with two hubs at rows 0 and 4 connected by
/// the spine 1-2-3:
///
/// ```text
/// 0 - 1 - 2 - 3 - 4
/// |               |
/// +---- (none) ---+
/// ```
///
/// Hub structure: row 0 connects to {1, 2, 3, 4}; row 4 connects to
/// {0, 1, 2, 3}; rows 1-3 each connect to {0, 4}. Both hubs share
/// the same fill cost — AMF should not eliminate either before the
/// inner spine variables are processed.
fn dual_arrow_5() -> (Vec<i64>, Vec<i64>) {
    // Adjacencies (excluding diagonal):
    // 0: 1 2 3 4    (hub left)
    // 1: 0 4
    // 2: 0 4
    // 3: 0 4
    // 4: 0 1 2 3    (hub right)
    let col_ptr = vec![0, 5, 8, 11, 14, 19];
    let row_idx = vec![
        0, 1, 2, 3, 4, // col 0 (incl. self)
        0, 1, 4, // col 1
        0, 2, 4, // col 2
        0, 3, 4, // col 3
        0, 1, 2, 3, 4, // col 4
    ];
    (col_ptr, row_idx)
}

/// Tridiagonal `n×n` with adjacencies `i ~ i±1`.
fn tridiag(n: usize) -> (Vec<i64>, Vec<i64>) {
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for j in 0..n {
        if j > 0 {
            row_idx.push((j - 1) as i64);
        }
        row_idx.push(j as i64);
        if j + 1 < n {
            row_idx.push((j + 1) as i64);
        }
        col_ptr.push(row_idx.len() as i64);
    }
    (col_ptr, row_idx)
}

/// Banded matrix of width `b` (off-diagonal half-bandwidth).
fn banded(n: usize, b: usize) -> (Vec<i64>, Vec<i64>) {
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for j in 0..n {
        let lo = j.saturating_sub(b);
        let hi = (j + b + 1).min(n);
        for r in lo..hi {
            row_idx.push(r as i64);
        }
        col_ptr.push(row_idx.len() as i64);
    }
    (col_ptr, row_idx)
}

// ---------------------------------------------------------------------
// Generic invariant assertions. Each takes the perm output and the
// matrix dimension; they hold for *any* successful `Metric`.
// ---------------------------------------------------------------------

fn assert_is_permutation(perm: &[i64], n: usize) {
    assert_eq!(perm.len(), n, "perm length must equal n");
    let mut seen = vec![false; n];
    for &p in perm {
        assert!(
            p >= 0 && (p as usize) < n,
            "perm entry {p} out of range [0, {n})"
        );
        assert!(!seen[p as usize], "duplicate perm entry {p}");
        seen[p as usize] = true;
    }
    assert!(
        seen.iter().all(|&b| b),
        "perm did not cover every index in 0..{n}"
    );
}

/// Run `order::<M>` on `pattern` with default options, expecting
/// success, and assert the permutation invariant.
fn run_and_check_perm<M: Metric>(pattern: &CscPattern<'_>) {
    let opts = WorkspaceOptions::default();
    let (perm, _diag) = order::<M>(pattern, &opts, true).expect("ordering call must succeed");
    assert_is_permutation(&perm, pattern.n);
}

// ---------------------------------------------------------------------
// Per-fixture parameterised tests. Each fixture is run through both
// MinDegree and MinFill arms; both must produce a valid permutation.
// ---------------------------------------------------------------------

#[test]
fn mindegree_arrow_3_perm() {
    let (cp, ri) = arrow_3();
    let p = CscPattern::new(3, &cp, &ri).expect("valid pattern");
    run_and_check_perm::<MinDegree>(&p);
}

#[test]
fn minfill_arrow_3_perm() {
    let (cp, ri) = arrow_3();
    let p = CscPattern::new(3, &cp, &ri).expect("valid pattern");
    run_and_check_perm::<MinFill>(&p);
}

#[test]
fn mindegree_dual_arrow_5_perm() {
    let (cp, ri) = dual_arrow_5();
    let p = CscPattern::new(5, &cp, &ri).expect("valid pattern");
    run_and_check_perm::<MinDegree>(&p);
}

#[test]
fn minfill_dual_arrow_5_perm() {
    let (cp, ri) = dual_arrow_5();
    let p = CscPattern::new(5, &cp, &ri).expect("valid pattern");
    run_and_check_perm::<MinFill>(&p);
}

#[test]
fn mindegree_tridiag_10_perm() {
    let (cp, ri) = tridiag(10);
    let p = CscPattern::new(10, &cp, &ri).expect("valid pattern");
    run_and_check_perm::<MinDegree>(&p);
}

#[test]
fn minfill_tridiag_10_perm() {
    let (cp, ri) = tridiag(10);
    let p = CscPattern::new(10, &cp, &ri).expect("valid pattern");
    run_and_check_perm::<MinFill>(&p);
}

#[test]
fn mindegree_banded_20_3_perm() {
    let (cp, ri) = banded(20, 3);
    let p = CscPattern::new(20, &cp, &ri).expect("valid pattern");
    run_and_check_perm::<MinDegree>(&p);
}

#[test]
fn minfill_banded_20_3_perm() {
    let (cp, ri) = banded(20, 3);
    let p = CscPattern::new(20, &cp, &ri).expect("valid pattern");
    run_and_check_perm::<MinFill>(&p);
}

#[test]
fn mindegree_empty_pattern_perm() {
    let cp = [0i64];
    let ri: [i64; 0] = [];
    let p = CscPattern::new(0, &cp, &ri).expect("empty pattern");
    let opts = WorkspaceOptions::default();
    let (perm, _) = order::<MinDegree>(&p, &opts, true).expect("n=0 succeeds");
    assert!(perm.is_empty());
}

#[test]
fn minfill_empty_pattern_perm() {
    let cp = [0i64];
    let ri: [i64; 0] = [];
    let p = CscPattern::new(0, &cp, &ri).expect("empty pattern");
    let opts = WorkspaceOptions::default();
    let (perm, _) = order::<MinFill>(&p, &opts, true).expect("n=0 succeeds");
    assert!(perm.is_empty());
}
