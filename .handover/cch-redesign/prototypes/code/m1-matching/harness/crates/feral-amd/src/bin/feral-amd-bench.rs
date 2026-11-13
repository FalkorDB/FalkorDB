//! `feral-amd-bench`: run AMD on a small built-in fixture suite and
//! report wall-clock time + stats. Intentionally minimal — a full
//! benchmark harness with larger matrices and comparison against
//! oracle lnz counts is deferred.

use std::time::Instant;

use feral_amd::{amd_order_with_stats, CscPattern};

fn arrow(n: usize) -> (Vec<i64>, Vec<i64>) {
    let mut cp: Vec<i64> = vec![0];
    let mut ri: Vec<i64> = Vec::new();
    ri.push(0);
    for r in 1..n {
        ri.push(r as i64);
    }
    cp.push(ri.len() as i64);
    for j in 1..n {
        ri.push(0);
        ri.push(j as i64);
        cp.push(ri.len() as i64);
    }
    (cp, ri)
}

fn band(n: usize, b: usize) -> (Vec<i64>, Vec<i64>) {
    let mut cp: Vec<i64> = vec![0];
    let mut ri: Vec<i64> = Vec::new();
    for j in 0..n {
        let lo = j.saturating_sub(b);
        let hi = (j + b + 1).min(n);
        for r in lo..hi {
            ri.push(r as i64);
        }
        cp.push(ri.len() as i64);
    }
    (cp, ri)
}

fn grid(m: usize, n: usize) -> (Vec<i64>, Vec<i64>) {
    use std::collections::BTreeSet;
    let total = m * n;
    let idx = |r: usize, c: usize| r * n + c;
    let mut cp: Vec<i64> = vec![0];
    let mut ri: Vec<i64> = Vec::new();
    for c in 0..total {
        let r0 = c / n;
        let c0 = c % n;
        let mut neigh: BTreeSet<usize> = BTreeSet::new();
        neigh.insert(c);
        if r0 > 0 {
            neigh.insert(idx(r0 - 1, c0));
        }
        if r0 + 1 < m {
            neigh.insert(idx(r0 + 1, c0));
        }
        if c0 > 0 {
            neigh.insert(idx(r0, c0 - 1));
        }
        if c0 + 1 < n {
            neigh.insert(idx(r0, c0 + 1));
        }
        for &r in &neigh {
            ri.push(r as i64);
        }
        cp.push(ri.len() as i64);
    }
    (cp, ri)
}

fn run(label: &str, n: usize, cp: &[i64], ri: &[i64]) {
    let pattern = match CscPattern::new(n, cp, ri) {
        Some(p) => p,
        None => {
            eprintln!("{:20} MALFORMED", label);
            return;
        }
    };
    let t0 = Instant::now();
    let res = amd_order_with_stats(&pattern);
    let dt = t0.elapsed();
    match res {
        Ok((_perm, stats)) => println!(
            "{:20} n={:6} nnz={:7} ncmpa={:3} ndense={:3} ndiv={:10} time={:.3}ms",
            label,
            n,
            pattern.nnz(),
            stats.ncmpa,
            stats.n_dense_deferred,
            stats.ndiv,
            dt.as_secs_f64() * 1e3
        ),
        Err(e) => println!("{:20} ERROR {:?}", label, e),
    }
}

fn main() {
    let (cp, ri) = arrow(200);
    run("arrow(200)", 200, &cp, &ri);
    let (cp, ri) = band(100, 5);
    run("band(100, 5)", 100, &cp, &ri);
    let (cp, ri) = band(500, 10);
    run("band(500, 10)", 500, &cp, &ri);
    let (cp, ri) = grid(30, 30);
    run("grid(30x30)", 30 * 30, &cp, &ri);
    let (cp, ri) = grid(60, 60);
    run("grid(60x60)", 60 * 60, &cp, &ri);
}
