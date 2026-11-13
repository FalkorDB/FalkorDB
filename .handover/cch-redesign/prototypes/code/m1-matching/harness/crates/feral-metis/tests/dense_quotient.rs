//! Tests for Fix A — quasi-dense column quotient inside
//! `feral-metis::metis_order_full`.
//!
//! Oracle: research note `dev/research/orbit2-cluster-regression.md`,
//! §6 ("Fix A"). The technique is published in Davis & Hager (2009)
//! §3.2 and AMD §5; see the docstring on
//! `MetisOptions::dense_quotient_enabled`.
//!
//! These tests do NOT touch the ORBIT2_0000 fixture — that is an
//! integration measurement that lives in the bench / diag harness.
//! Here we only verify:
//!
//! 1. A synthetic pattern with one near-dense column places that
//!    column at the LAST position(s) of the returned permutation.
//! 2. A pattern with no near-dense column produces the same
//!    permutation as the no-Fix-A baseline (byte-for-byte).
//! 3. The permutation is a valid bijection on `[0, n)` in both cases.
//! 4. Disabling the quotient reverts to the legacy path.

use feral_metis::{metis_order_full, MetisOptions};
use feral_ordering_core::CscPattern;
use std::collections::BTreeSet;

/// Build a full-symmetric CSC pattern from an `(i, j)` triple list
/// (lower OR upper triangle, both directions are inserted, plus the
/// diagonal). Row indices within each column are sorted ascending.
fn csc_from_triples(n: usize, triples: &[(usize, usize)]) -> (Vec<i64>, Vec<i64>) {
    let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
    for i in 0..n {
        set.insert((i, i));
    }
    for &(i, j) in triples {
        set.insert((i, j));
        set.insert((j, i));
    }
    let mut cols: Vec<Vec<i64>> = vec![Vec::new(); n];
    for &(r, c) in &set {
        cols[c].push(r as i64);
    }
    for col in &mut cols {
        col.sort();
    }
    let mut col_ptr: Vec<i64> = vec![0];
    let mut row_idx: Vec<i64> = Vec::new();
    for col in &cols {
        for &r in col {
            row_idx.push(r);
        }
        col_ptr.push(row_idx.len() as i64);
    }
    (col_ptr, row_idx)
}

fn assert_is_permutation(perm: &[i64], n: usize) {
    assert_eq!(perm.len(), n, "perm has wrong length");
    let mut seen = vec![false; n];
    for &p in perm {
        let p = p as usize;
        assert!(p < n, "perm value {} out of range for n={}", p, n);
        assert!(!seen[p], "duplicate perm value {}", p);
        seen[p] = true;
    }
    assert!(seen.iter().all(|&b| b), "perm is not surjective");
}

/// Banded sparse pattern: tridiagonal-ish (each row connects to
/// `r-1`, `r+1`, plus `r-band` and `r+band` so the graph is well
/// connected without any near-dense column).
fn banded_triples(n: usize, band: usize) -> Vec<(usize, usize)> {
    let mut t = Vec::new();
    for r in 0..n {
        if r + 1 < n {
            t.push((r, r + 1));
        }
        if r + band < n {
            t.push((r, r + band));
        }
    }
    t
}

#[test]
fn dense_column_lands_at_end_of_perm() {
    // n = 1000, band = 7 → max off-degree from band ≈ 4.
    // Threshold for n=1000 is max(40, ceil(10*sqrt(1000))) = 317.
    // Inject one column (id 500) coupled to 800 other rows. That is
    // well above 317, so it must be quotiented out and land at the
    // LAST position of the returned permutation.
    const N: usize = 1000;
    const DENSE_COL: usize = 500;
    const DENSE_DEG: usize = 800;

    let mut t = banded_triples(N, 7);
    let mut added: BTreeSet<usize> = BTreeSet::new();
    let mut step: usize = 1;
    while added.len() < DENSE_DEG {
        let r = (DENSE_COL + step) % N;
        if r != DENSE_COL && !added.contains(&r) {
            t.push((DENSE_COL, r));
            added.insert(r);
        }
        step += 1;
        if step > 2 * N {
            break;
        }
    }
    assert_eq!(added.len(), DENSE_DEG, "test setup failed");

    let (cp, ri) = csc_from_triples(N, &t);
    let pat = CscPattern::new(N, &cp, &ri).expect("valid CSC");

    // Fix A is OFF by default per the 2026-04-27 expert review (see
    // `MetisOptions::dense_quotient_enabled` doc comment). Enable it
    // explicitly to exercise the opt-in code path.
    let opts = MetisOptions {
        dense_quotient_enabled: true,
        ..Default::default()
    };

    let (perm, _ostats, _mstats) = metis_order_full(&pat, &opts).expect("ordering ok");
    assert_is_permutation(&perm, N);

    // The dense column must land at the end. Only one column is
    // dense, so it is the very last entry.
    let last = *perm.last().expect("non-empty perm");
    assert_eq!(
        last as usize, DENSE_COL,
        "dense column {} expected at perm.last(), got {}",
        DENSE_COL, last
    );
}

#[test]
fn no_dense_column_matches_legacy_baseline() {
    // Banded pattern, no near-dense column. With Fix A enabled the
    // quotient set is empty, so `metis_order_full` must produce the
    // exact same permutation as with Fix A disabled.
    const N: usize = 600;
    let t = banded_triples(N, 11);
    let (cp, ri) = csc_from_triples(N, &t);
    let pat = CscPattern::new(N, &cp, &ri).expect("valid CSC");

    let opts_off = MetisOptions {
        dense_quotient_enabled: false,
        ..Default::default()
    };
    let opts_on = MetisOptions {
        dense_quotient_enabled: true,
        ..Default::default()
    };

    let (perm_off, _, _) = metis_order_full(&pat, &opts_off).expect("ok");
    let (perm_on, _, _) = metis_order_full(&pat, &opts_on).expect("ok");

    assert_is_permutation(&perm_off, N);
    assert_is_permutation(&perm_on, N);
    assert_eq!(
        perm_off, perm_on,
        "dense quotient must be a no-op when no column exceeds threshold"
    );
}

#[test]
fn disabling_quotient_keeps_dense_column_in_interior_permutation() {
    // Same dense pattern as the first test, but with the quotient
    // disabled. Intent: prove the new option actually controls the
    // behaviour. We only assert that the produced perm is a valid
    // permutation; we do NOT assert where the dense column lands
    // because that depends on ND splits and varies by seed.
    const N: usize = 1000;
    const DENSE_COL: usize = 500;
    const DENSE_DEG: usize = 800;

    let mut t = banded_triples(N, 7);
    let mut added: BTreeSet<usize> = BTreeSet::new();
    let mut step: usize = 1;
    while added.len() < DENSE_DEG {
        let r = (DENSE_COL + step) % N;
        if r != DENSE_COL && !added.contains(&r) {
            t.push((DENSE_COL, r));
            added.insert(r);
        }
        step += 1;
        if step > 2 * N {
            break;
        }
    }
    let (cp, ri) = csc_from_triples(N, &t);
    let pat = CscPattern::new(N, &cp, &ri).expect("valid CSC");

    let opts_off = MetisOptions {
        dense_quotient_enabled: false,
        ..Default::default()
    };

    let (perm_off, _, _) = metis_order_full(&pat, &opts_off).expect("ok");
    assert_is_permutation(&perm_off, N);

    // With quotient OFF, the dense column is not forced to the end.
    // It MAY end up there by chance, but in practice the ND
    // separator pulls it in earlier. We at least sanity-check that
    // the permutation is a valid bijection (already done above).
    //
    // To make the contrast visible, also run with quotient ON and
    // confirm the column lands exactly at the last position.
    let opts_on = MetisOptions {
        dense_quotient_enabled: true,
        ..Default::default()
    };
    let (perm_on, _, _) = metis_order_full(&pat, &opts_on).expect("ok");
    assert_is_permutation(&perm_on, N);
    assert_eq!(perm_on.last().copied().unwrap_or(-1) as usize, DENSE_COL);
}

#[test]
fn threshold_override_promotes_lower_degree_column() {
    // With the default threshold for n=200 (= max(40, 142) = 142),
    // a column of degree 50 is NOT dense. If the caller forces a
    // threshold of 30, that same column becomes dense and must land
    // at the end. Validates `dense_quotient_threshold` plumbing.
    const N: usize = 200;
    let mut t = banded_triples(N, 5);
    const DENSE_COL: usize = 73;
    let mut added: BTreeSet<usize> = BTreeSet::new();
    let mut step: usize = 1;
    while added.len() < 50 {
        let r = (DENSE_COL + step) % N;
        if r != DENSE_COL && !added.contains(&r) {
            t.push((DENSE_COL, r));
            added.insert(r);
        }
        step += 1;
        if step > 2 * N {
            break;
        }
    }

    let (cp, ri) = csc_from_triples(N, &t);
    let pat = CscPattern::new(N, &cp, &ri).expect("valid CSC");

    // Default threshold: 50 < max(40, 142) = 142 → not dense, no
    // forced placement. We assert only validity here.
    let opts_default = MetisOptions::default();
    let (perm_default, _, _) = metis_order_full(&pat, &opts_default).expect("ok");
    assert_is_permutation(&perm_default, N);

    // Forced threshold 30: column degree 50 > 30 → dense, must be
    // last. Quotient is OFF by default (post-2026-04-27 expert review),
    // so opt in explicitly.
    let opts_forced = MetisOptions {
        dense_quotient_enabled: true,
        dense_quotient_threshold: Some(30),
        ..Default::default()
    };
    let (perm_forced, _, _) = metis_order_full(&pat, &opts_forced).expect("ok");
    assert_is_permutation(&perm_forced, N);
    assert_eq!(
        perm_forced.last().copied().unwrap_or(-1) as usize,
        DENSE_COL
    );
}
