//! Measurements behind the fold policy in [`super::versioned_matrix`].
//!
//! The policy folds a delta at `|delta| ≈ sqrt(2·(F/w)·tx_added)`, which
//! follows from one assumption: **every write transaction pays `O(|delta|)`**,
//! because it COW-dups the delta and merges its pending tuples. These benches
//! measure `F` and `w` so the constants stop being guesses. What they found:
//!
//!   1. `dup` of an assembled delta — `w_dup ≈ 0.2 ns/entry`, memcpy speed.
//!      This is the whole tax on the write path, and it is what
//!      `WRITE_FOLD_K` is derived from.
//!   2. `dup` of a delta holding *pending* tuples costs the same as an
//!      assembled one, so `dup` does not assemble. The `wait` that does
//!      assemble is `w_merge ≈ 50 ns/entry` — 250x dearer, and it tracks
//!      `|delta|` rather than what the transaction added. That 250x gap is
//!      why the read path folds `sqrt(250) ≈ 16x` tighter than the write path.
//!   3. The fold itself, vs base nvals *and* nrows. It is flat in nrows —
//!      2.0-2.5 µs for an empty base at nrows = 65k, 1m and 16.7m alike —
//!      which is what retired the `base_cost = nvals + nrows` term an earlier
//!      policy carried. Against nvals it is strongly *sub*-linear: ~2 µs
//!      empty, 707 µs at 16k (43 ns/entry), 2174 µs at 1m (2 ns/entry). Read
//!      that as two format regimes rather than one line — a delta at 1.5%
//!      density is hypersparse, one at 95% is sparse/bitmap, and eWiseAdd
//!      vectorizes only in the latter. Fitting a line across the top gives a
//!      "~1.7 ms fixed cost", which is an artifact of that curve, not a real
//!      constant term. What matters for the policy is `F` at the operating
//!      point, and across 262k -> 1m it moves only 1.2x for 4x the entries —
//!      flat enough that the balance point does not need to track the base.
//!
//! Caveat on absolute numbers: taken on macOS, and `w` is cache-bound, so
//! expect different values in CI. Re-measure in the toolchain image before
//! re-tuning. Note the numbers above were taken *with* the vendored PreJIT
//! kernels compiled in (`GRAPHBLAS_LIB_DIR` pointed at a local
//! `graphblas.sh` build) — vendoring them did not move the fold's shape, so
//! the sub-linearity is a property of eWiseAdd, not of kernel fallback.
//!
//! Run with:
//!   cargo test --release -p graph fold_cost -- --ignored --nocapture

use std::time::Instant;

use super::matrix::{Dup, Matrix};
use super::test_init::ensure_init;

/// Big-graph shape: square, high capacity, so `nrows` dominates `nvals` for
/// sparse matrices exactly as it does on a real graph's label/adjacency
/// matrices.
///
/// Must stay `>=` the largest sample size below, or [`scatter`] wraps and the
/// matrix holds fewer entries than the row is labelled with. At the previous
/// `1_000_000` the `1_048_576` sample produced only 1,000,000 unique pairs:
/// `rows` wraps at `CAP`, and because `7 * 1_000_000 ≡ 0 (mod 1_000_000)` the
/// `cols` stride collided on the same period, so `i` and `i + CAP` generated
/// an identical coordinate pair.
///
/// Deliberately the *smallest* power of two that clears the largest sample,
/// not something roomier: `CAP` sets the density of every matrix under test
/// (`nvals / CAP`), and density is what GraphBLAS keys its sparse/hypersparse
/// format choice on. Raising it to `2_097_152` halved the density and moved
/// `fold_cost_fold_vs_base` by up to 1.9x at the top end while the
/// capacity-only floor below stayed flat — i.e. it changed what was being
/// measured, not just the labels. Being a power of two also keeps the `* 7`
/// column stride coprime with `CAP`, so `cols` is a bijection.
const CAP: u64 = 1_048_576;

/// Spread entries over distinct rows so the matrix is genuinely sparse rather
/// than a dense block, which is what a real delta looks like.
///
/// `debug_assert`s uniqueness: a silently-deduplicated build would measure a
/// smaller matrix than the caller asked for and quietly mis-calibrate the
/// fold policy's constants.
fn scatter(n: u64) -> (Vec<u64>, Vec<u64>) {
    assert!(n <= CAP, "sample {n} exceeds CAP {CAP}; scatter would wrap");
    let stride = (CAP / n.max(1)).max(1);
    let rows: Vec<u64> = (0..n).map(|i| (i * stride) % CAP).collect();
    let cols: Vec<u64> = (0..n).map(|i| (i * 7 + 3) % CAP).collect();
    debug_assert_eq!(
        rows.iter()
            .zip(&cols)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        n as usize,
        "scatter({n}) generated duplicate coordinates"
    );
    (rows, cols)
}

fn assembled(n: u64) -> Matrix<bool> {
    let mut m = Matrix::<bool>::new(CAP, CAP);
    let (rows, cols) = scatter(n);
    m.build(&rows, &cols);
    m.wait();
    m
}

fn pending(n: u64) -> Matrix<bool> {
    let mut m = Matrix::<bool>::new(CAP, CAP);
    let (rows, cols) = scatter(n);
    for i in 0..rows.len() {
        m.set(rows[i], cols[i], true);
    }
    m // deliberately NOT waited: entries sit in the pending list
}

fn time_us<F: FnMut()>(
    reps: u32,
    mut f: F,
) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e6 / f64::from(reps)
}

#[test]
#[ignore = "measurement, not a correctness check"]
fn fold_cost_dup_vs_nvals() {
    ensure_init();
    println!("\n=== dup cost vs delta size (the per-transaction tax) ===");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}",
        "nvals", "assembled us", "pending us", "wait us"
    );
    for &n in &[256u64, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576] {
        let reps = if n <= 16_384 { 200 } else { 20 };

        let a = assembled(n);
        let dup_assembled = time_us(reps, || {
            let d = a.dup();
            std::hint::black_box(&d);
        });

        // Rebuild per rep: dup of a pending matrix may assemble it, which
        // would make a reused matrix cheap on every rep after the first.
        let mut dup_pending_total = 0.0;
        let mut wait_total = 0.0;
        let pending_reps = if n <= 16_384 { 20 } else { 5 };
        for _ in 0..pending_reps {
            let p = pending(n);
            let t = Instant::now();
            let d = p.dup();
            dup_pending_total += t.elapsed().as_secs_f64() * 1e6;
            std::hint::black_box(&d);

            let p2 = pending(n);
            let t = Instant::now();
            p2.wait();
            wait_total += t.elapsed().as_secs_f64() * 1e6;
        }

        println!(
            "{:>10}  {:>12.1}  {:>12.1}  {:>12.1}",
            n,
            dup_assembled,
            dup_pending_total / f64::from(pending_reps),
            wait_total / f64::from(pending_reps),
        );
    }
    println!("\nper-entry cost (us per 1k entries), assembled dup:");
    for &n in &[1_024u64, 16_384, 262_144, 1_048_576] {
        let a = assembled(n);
        let us = time_us(if n <= 16_384 { 200 } else { 20 }, || {
            let d = a.dup();
            std::hint::black_box(&d);
        });
        println!("{:>10}  {:>10.3} us/1k", n, us / (n as f64 / 1000.0));
    }
}

#[test]
#[ignore = "measurement, not a correctness check"]
fn fold_cost_fold_vs_base() {
    ensure_init();
    println!("\n=== fold cost (eWiseAdd base<-delta) vs base size ===");
    println!(
        "{:>12}  {:>10}  {:>12}  {:>14}",
        "base nvals", "delta", "fold us", "us per 1k base"
    );
    for &b in &[1_024u64, 16_384, 262_144, 1_048_576] {
        for &d in &[256u64, 4_096, 65_536] {
            if d > b {
                continue;
            }
            let delta = assembled(d);
            let reps = if b <= 16_384 { 50 } else { 10 };
            let base = assembled(b);
            let mut total = 0.0;
            for _ in 0..reps {
                // Mirror VersionedMatrix::flush: build the folded base into a
                // fresh matrix (under MVCC the base is always shared, so an
                // in-place fold would deep-copy it first).
                let t = Instant::now();
                let mut new_m = Matrix::<bool>::new(CAP, CAP);
                new_m.element_wise_add(None, Some(&base), Some(&delta), None);
                new_m.wait();
                total += t.elapsed().as_secs_f64() * 1e6;
                std::hint::black_box(&new_m);
            }
            let us = total / f64::from(reps);
            println!(
                "{:>12}  {:>10}  {:>12.1}  {:>14.3}",
                b,
                d,
                us,
                us / (b as f64 / 1000.0)
            );
        }
    }
}

#[test]
#[ignore = "measurement, not a correctness check"]
fn fold_cost_empty_matrix_floor() {
    ensure_init();
    // What does an empty-but-high-capacity matrix cost to dup and fold?
    //
    // This is the measurement that *refuted* the `O(nrows)` term an earlier
    // policy carried as `base_cost = nvals + nrows`: dup and fold both come
    // out flat across nrows = 1k .. 16.7m, so a fold does not pay for the
    // base's row-pointer structure and `should_fold` keys on nvals alone.
    // Kept as a regression check on that conclusion — if these columns ever
    // start scaling with nrows, the policy constants need revisiting.
    println!("\n=== capacity-only cost (nvals = 0) ===");
    println!("{:>12}  {:>10}  {:>10}", "nrows", "dup us", "fold us");
    for &cap in &[1_024u64, 65_536, 1_048_576, 16_777_216] {
        let empty = Matrix::<bool>::new(cap, cap);
        let dup_us = time_us(50, || {
            let d = empty.dup();
            std::hint::black_box(&d);
        });
        let mut delta = Matrix::<bool>::new(cap, cap);
        delta.build(&[0, cap / 2], &[1, cap / 2]);
        delta.wait();
        let base = Matrix::<bool>::new(cap, cap);
        let mut total = 0.0;
        for _ in 0..20 {
            let t = Instant::now();
            let mut new_m = Matrix::<bool>::new(cap, cap);
            new_m.element_wise_add(None, Some(&base), Some(&delta), None);
            new_m.wait();
            total += t.elapsed().as_secs_f64() * 1e6;
            std::hint::black_box(&new_m);
        }
        println!("{:>12}  {:>10.2}  {:>10.2}", cap, dup_us, total / 20.0);
    }
}

#[test]
#[ignore = "measurement, not a correctness check"]
fn fold_cost_write_cycle() {
    ensure_init();
    // The actual per-transaction tax the policy models as `O(|delta|)`: take a
    // delta that already holds D entries, COW-dup it (new MVCC version), add
    // `t` more, then materialize. If this is flat in D the sqrt term is
    // unnecessary; if it grows with D the term is justified.
    println!("\n=== write-cycle cost vs existing delta size ===");
    println!(
        "{:>10}  {:>4}  {:>10}  {:>10}  {:>10}  {:>10}",
        "delta D", "t", "dup us", "set us", "wait us", "total us"
    );
    for &d in &[0u64, 1_024, 4_096, 16_384, 65_536, 262_144] {
        for &t in &[1u64, 100] {
            let reps: u32 = if d <= 16_384 { 30 } else { 10 };
            let (mut dup_us, mut set_us, mut wait_us) = (0.0, 0.0, 0.0);
            for r in 0..reps {
                let base_delta = assembled(d);

                let t0 = Instant::now();
                let mut v = base_delta.dup();
                dup_us += t0.elapsed().as_secs_f64() * 1e6;

                let t1 = Instant::now();
                for k in 0..t {
                    // Rows past the existing delta so these are new entries.
                    let row = (d + k + u64::from(r) * 1_000) % CAP;
                    v.set(row, (row * 13 + 1) % CAP, true);
                }
                set_us += t1.elapsed().as_secs_f64() * 1e6;

                let t2 = Instant::now();
                v.wait();
                wait_us += t2.elapsed().as_secs_f64() * 1e6;
                std::hint::black_box(&v);
            }
            let r = f64::from(reps);
            println!(
                "{:>10}  {:>4}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.1}",
                d,
                t,
                dup_us / r,
                set_us / r,
                wait_us / r,
                (dup_us + set_us + wait_us) / r
            );
        }
    }
}

fn assembled_u64(n: u64) -> Matrix<u64> {
    let mut m = Matrix::<u64>::new(CAP, CAP);
    let (rows, cols) = scatter(n);
    let vals: Vec<u64> = (0..n).collect();
    m.build(&rows, &cols, &vals);
    m.wait();
    m
}

/// `Tensor::resize`'s current grow strategy, inlined so the bench measures it
/// without depending on tensor internals.
fn rebuild(
    src: &Matrix<u64>,
    nrows: u64,
    ncols: u64,
) -> Matrix<u64> {
    let n = src.nvals() as usize;
    let mut rows = Vec::with_capacity(n);
    let mut cols = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(n);
    for (r, c, v) in src.iter(0, u64::MAX) {
        rows.push(r);
        cols.push(c);
        vals.push(v);
    }
    let mut dst = Matrix::<u64>::new(nrows, ncols);
    dst.build(&rows, &cols, &vals);
    dst.wait();
    dst
}

/// Grow-resize strategies for a COW-shared layer, which cannot be resized in
/// place. All three produce the same matrix at larger dims; they differ in how
/// much work they throw away getting there.
///
/// * `rebuild` — row-iterate the source and `GrB_Matrix_build` into a fresh
///   matrix at the target dims. Today's `Tensor::resize` grow path.
/// * `concat` — `GxB_Matrix_concat` the source into the top-left of a fresh
///   matrix, padded with empty tiles. One bulk block copy, no tuple arrays.
/// * `dup+resize` — the baseline the current code replaced: deep-copy at the
///   old dims (what `Cow::deref_mut` does on a shared layer), then
///   `GrB_Matrix_resize`. The trailing `wait` is charged because
///   `GrB_Matrix_resize` leaves the wrapper's `has_pending` set, so the next
///   reader pays it.
///
/// Growth factor is 1.14x, matching the capacity-grow pattern that produced
/// the original 2.4-9.8 ms spikes (100,000 nodes -> 114,688 capacity).
///
/// What it found (macOS, two runs): concat beats rebuild by ~7x at 1m entries
/// (1.2 ms vs 7.6-8.4 ms) and ~3x at 262k, they are par at 16k, and rebuild
/// wins below that — hence the empty-delta short-circuits at the call sites.
///
/// **`dup+resize` measured faster still** — 0.8 ms at 1m, and 0.7 µs on a
/// hypersparse 1k-entry delta where both others cost 20 µs — and
/// `grow_cost_post_grow_usage` found no deferred penalty: iteration, point
/// lookups, a follow-up mutation and `memory_usage` are identical across all
/// three. It is *not* adopted here, and the reason is not the copy cost the
/// original commit message gives (both strategies copy). It is that
/// `GrB_Matrix_resize` frees the hyper hash and leaves the wrapper's
/// `has_pending` set, so the layer needs a `wait` that rebuilds it — the effect
/// behind the "1.4-5.7x write regressions" note on the shrink path, and behind
/// `tensor::tests::resize_leaves_base_materialized`. Anyone revisiting this
/// should measure that end to end rather than trusting the microbench: the
/// `dup+resize` column here charges the `wait` but not what the *next*
/// transaction pays for a rebuilt hash.
#[test]
#[ignore = "measurement, not a correctness check"]
fn grow_cost_concat_vs_rebuild() {
    ensure_init();
    println!("\n=== grow-resize cost by strategy (uint64 layer, 1.14x dims) ===");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}  {:>10}",
        "nvals", "rebuild us", "concat us", "dup+resize us", "concat/rebuild"
    );
    let (nrows, ncols) = (CAP + CAP * 14 / 100, CAP + CAP * 14 / 100);
    for &n in &[0u64, 1_024, 16_384, 262_144, 1_048_576] {
        let reps: u32 = if n <= 16_384 { 50 } else { 10 };
        let src = assembled_u64(n);

        let rebuild_us = time_us(reps, || {
            let g = rebuild(&src, nrows, ncols);
            std::hint::black_box(&g);
        });
        let concat_us = time_us(reps, || {
            let g = src.grown(nrows, ncols);
            g.wait();
            std::hint::black_box(&g);
        });
        // dup per rep: resize mutates, so a reused copy would be measured
        // already-grown on every rep after the first.
        let mut dup_resize_us = 0.0;
        for _ in 0..reps {
            let t = Instant::now();
            let mut g = src.dup();
            g.resize(nrows, ncols);
            g.wait();
            dup_resize_us += t.elapsed().as_secs_f64() * 1e6;
            std::hint::black_box(&g);
        }
        dup_resize_us /= f64::from(reps);

        println!(
            "{:>10}  {:>12.1}  {:>12.1}  {:>12.1}  {:>10.2}",
            n,
            rebuild_us,
            concat_us,
            dup_resize_us,
            concat_us / rebuild_us.max(f64::EPSILON)
        );
    }
}

/// What the grown matrix costs to *use*, and how much memory it holds.
///
/// The grow itself is only half the question: `dup+resize` keeps the source's
/// internal representation (and drops its hyper hash, which the next lookup
/// rebuilds), while `rebuild`/`concat` hand back a freshly assembled matrix. A
/// strategy that wins the copy and loses the follow-up is not a win — the
/// follow-up is what every read after a capacity grow pays.
#[test]
#[ignore = "measurement, not a correctness check"]
fn grow_cost_post_grow_usage() {
    ensure_init();
    println!("\n=== cost of using the grown matrix, and its footprint ===");
    println!(
        "{:>10}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}",
        "nvals", "strategy", "iter us", "probe us", "set+wait us", "mem MB"
    );
    let (nrows, ncols) = (CAP + CAP * 14 / 100, CAP + CAP * 14 / 100);
    for &n in &[16_384u64, 262_144, 1_048_576] {
        let src = assembled_u64(n);
        let (probe_rows, probe_cols) = scatter(n);
        for strategy in ["rebuild", "concat", "dup+resize"] {
            let reps: u32 = if n <= 16_384 { 20 } else { 5 };
            let (mut iter_us, mut probe_us, mut set_us) = (0.0, 0.0, 0.0);
            let mut mem = 0usize;
            for _ in 0..reps {
                let mut g = match strategy {
                    "rebuild" => rebuild(&src, nrows, ncols),
                    "concat" => {
                        let g = src.grown(nrows, ncols);
                        g.wait();
                        g
                    }
                    _ => {
                        let mut g = src.dup();
                        g.resize(nrows, ncols);
                        g.wait();
                        g
                    }
                };

                let t = Instant::now();
                let mut seen = 0u64;
                for (_, _, v) in g.iter(0, u64::MAX) {
                    seen += v;
                }
                iter_us += t.elapsed().as_secs_f64() * 1e6;
                std::hint::black_box(seen);

                // Point lookups: what a hypersparse layer needs the hyper hash
                // for, and what `resize` freeing it would show up in.
                let t = Instant::now();
                let mut hits = 0u32;
                for k in (0..probe_rows.len()).step_by(probe_rows.len() / 1_000 + 1) {
                    hits += u32::from(g.contains(probe_rows[k], probe_cols[k]));
                }
                probe_us += t.elapsed().as_secs_f64() * 1e6;
                std::hint::black_box(hits);

                // One more mutation, materialized: the next transaction.
                let t = Instant::now();
                g.set(nrows - 1, ncols - 1, 7);
                g.wait();
                set_us += t.elapsed().as_secs_f64() * 1e6;

                mem = g.memory_usage();
            }
            let r = f64::from(reps);
            println!(
                "{:>10}  {:>12}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.2}",
                n,
                strategy,
                iter_us / r,
                probe_us / r,
                set_us / r,
                mem as f64 / (1024.0 * 1024.0),
            );
        }
    }
}
