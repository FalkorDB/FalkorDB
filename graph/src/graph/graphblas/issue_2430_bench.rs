//! Diagnosis for FalkorDB issue #2430: the two-state multi-edge point read.
//!
//! A bound multi-edge point read lands in one of two cost states — about 4.35k
//! or about 5.5k instructions per pair — selected *non-monotonically* by graph
//! size, crossing at least twice between 1,000 and 121,000 populated pairs.
//! Three hypotheses are refuted in the issue: it is not the sentinel charge
//! scaling with `|me|` (flat across a 41x change), not walking more ids (`k` is
//! pinned at 2), and not a lingering delta a read latches but never flushes.
//!
//! Two candidates remain, and this file is the cheap way to separate them.
//!
//! 1. **A GraphBLAS storage-format or capacity threshold.** Matrix dimensions
//!    step with node capacity, and sparse/hypersparse selection depends on the
//!    ratio of populated vectors to dimension, so the same logical read can land
//!    on a different kernel either side of a switch. [`issue_2430_sparsity`]
//!    prints the format of every matrix on the read path across the sizes where
//!    the engine-level cost is known to flip.
//!
//! 2. **Something outside edge storage entirely.** The boundary measurements say
//!    a sentinel read is a *flat* cost — 2,684 instructions at `k = 2` and the
//!    same at `k = 16`, and flat across `|me|`. If the two states do not
//!    reproduce with no query pipeline around the read, the defect is not in the
//!    tensor and the issue belongs to whoever owns the pipeline.
//!    [`issue_2430_boundary`] is that reproduction.
//!
//! Run with:
//!   cargo test --release -p graph issue_2430 -- --ignored --nocapture --test-threads=1

use std::time::Instant;

use super::instr::read_instr;
use super::tensor::Tensor;
use super::test_init::ensure_init;

/// The issue's fixture, and the detail that matters: **rows are held at
/// `ROWS`** while the pair count grows, so a bigger graph means *longer rows*,
/// not more of them. An earlier version of this file gave every pair its own row
/// and so held row length at 1 at every size — which is why it saw a flat cost
/// and wrongly cleared edge storage. `GrB_Matrix_extractElement` searches within
/// a row, so row length is exactly the variable a point read can be sensitive
/// to.
///
/// The first `multi` pairs of row 0.. are two-edge and are the ones read.
const ROWS: u64 = 1_000;

fn built(
    pairs: u64,
    multi: u64,
) -> Tensor {
    let mut t = Tensor::new(ROWS + 2, ROWS + 2);
    let (mut srcs, mut dsts, mut ids) = (Vec::new(), Vec::new(), Vec::new());
    let mut next = 0u64;
    for i in 0..pairs {
        // spread over ROWS rows: row length grows as `pairs` does
        let (s, d) = if i < multi {
            (i % ROWS, (i + 1) % ROWS)
        } else {
            (i % ROWS, (i * 7 + 3) % ROWS)
        };
        let k = if i < multi { 2 } else { 1 };
        for _ in 0..k {
            srcs.push(s);
            dsts.push(d);
            ids.push(next);
            next += 1;
        }
    }
    t.set_all_from_slices(&srcs, &dsts, &ids);
    let mut t = t.dup();
    t.flush();
    t.wait();
    t
}

const PROBES: u64 = 1_000;
const REPS: u64 = 20;

fn probe_cost(t: &Tensor) -> (f64, f64) {
    // warm
    for i in 0..PROBES {
        std::hint::black_box(t.get(i % ROWS, (i + 1) % ROWS).count());
    }
    let i0 = read_instr();
    let t0 = Instant::now();
    let mut acc = 0usize;
    for _ in 0..REPS {
        for i in 0..PROBES {
            acc += t.get(i % ROWS, (i + 1) % ROWS).count();
        }
    }
    let el = t0.elapsed();
    let i1 = read_instr();
    std::hint::black_box(acc);
    let n = (PROBES * REPS) as f64;
    (
        match (i0, i1) {
            (Some(a), Some(b)) => (b - a) as f64 / n,
            _ => f64::NAN,
        },
        el.as_secs_f64() * 1e9 / n,
    )
}

/// **Candidate 1**: print every read-path matrix's storage format across the
/// sizes where the engine-level cost is known to flip, alongside the cost.
///
/// If a format column changes exactly where the cost changes, that is the cause.
/// If every format is constant while the cost flips, candidate 1 is refuted and
/// the answer is candidate 2.
#[test]
#[ignore]
fn issue_2430_sparsity() {
    ensure_init();
    println!("\n=== #2430: read cost vs storage format (1,000 two-edge pairs read throughout) ===",);
    println!(
        "{:>10}  {:>10}  {:>9}  {:>12}  {:>12}  {:>12}  {:>12}",
        "pairs", "instr/pair", "ns/pair", "m", "dp", "dm", "me(base)"
    );
    for pairs in [1_000u64, 11_000, 41_000, 88_000, 121_000, 160_000] {
        let t = built(pairs, PROBES.min(pairs));
        let (instr, ns) = probe_cost(&t);
        println!(
            "{:>10}  {:>10.1}  {:>9.1}  {:>12}  {:>12}  {:>12}  {:>12}",
            pairs,
            instr,
            ns,
            t.fwd_m().sparsity_status(),
            t.fwd_dp().sparsity_status(),
            t.fwd_dm().sparsity_status(),
            t.edge_versioned().m().sparsity_status(),
        );
    }
}

/// **Candidate 2**: the same read with no query pipeline around it, at the same
/// sizes. The boundary says a sentinel read is flat; if it is flat here too
/// while the engine-level number flips, the defect is outside edge storage.
#[test]
#[ignore]
fn issue_2430_boundary() {
    ensure_init();
    println!("\n=== #2430: the same read at the data-structure boundary ===");
    println!("{:>10}  {:>12}  {:>10}", "pairs", "instr/pair", "ns/pair");
    for pairs in [1_000u64, 11_000, 41_000, 88_000, 121_000, 160_000] {
        let t = built(pairs, PROBES.min(pairs));
        let mut best = f64::MAX;
        let mut bns = f64::MAX;
        for _ in 0..3 {
            let (i, n) = probe_cost(&t);
            best = best.min(i);
            bns = bns.min(n);
        }
        println!("{:>10}  {:>12.1}  {:>10.1}", pairs, best, bns);
    }
}

/// **Candidate 3**: the build path, not the content.
///
/// The engine-level step tracks the read tensor's own growth — with filler edges
/// on a *different* relationship type the step disappears — yet a synthetic
/// tensor with the same logical content only drifts. The remaining difference is
/// how the two were built: the engine accumulates through many transactions,
/// each a `dup` and a fold, where [`built`] loads one batch and flushes once.
///
/// This builds the identical tensor both ways and reads it the same way. If the
/// incremental one shows the step, the state that matters is a property of the
/// matrix's internal layout after repeated folds, not of what it holds.
fn built_incremental(
    pairs: u64,
    multi: u64,
) -> Tensor {
    let mut t = Tensor::new(ROWS + 2, ROWS + 2);
    let mut next = 0u64;
    let batch = 20_000u64;
    let mut lo = 0u64;
    while lo < pairs {
        let hi = (lo + batch).min(pairs);
        let (mut srcs, mut dsts, mut ids) = (Vec::new(), Vec::new(), Vec::new());
        for i in lo..hi {
            let (s, d) = if i < multi {
                (i % ROWS, (i + 1) % ROWS)
            } else {
                (i % ROWS, (i * 7 + 3) % ROWS)
            };
            let k = if i < multi { 2 } else { 1 };
            for _ in 0..k {
                srcs.push(s);
                dsts.push(d);
                ids.push(next);
                next += 1;
            }
        }
        t.set_all_from_slices(&srcs, &dsts, &ids);
        // one transaction per batch, exactly as the engine does it
        let mut next_t = t.dup();
        next_t.flush();
        next_t.wait();
        t = next_t;
        lo = hi;
    }
    t
}

#[test]
#[ignore]
fn issue_2430_build_path() {
    ensure_init();
    println!("\n=== #2430: one batch against many, same logical tensor ===");
    println!(
        "{:>10}  {:>14}  {:>14}  {:>8}  {:>14}",
        "pairs", "one batch", "incremental", "delta", "after wait"
    );
    for pairs in [1_000u64, 11_000, 41_000, 88_000, 121_000] {
        let one = built(pairs, PROBES.min(pairs));
        let mut many = built_incremental(pairs, PROBES.min(pairs));
        assert_eq!(
            one.edge_count(),
            many.edge_count(),
            "the two build paths must produce the same tensor"
        );
        let (i1, _) = probe_cost(&one);
        let (i2, _) = probe_cost(&many);
        let fmt = |t: &Tensor| {
            format!(
                "m={} dp={} dm={} me={} me_dp={} me_dm={}",
                t.fwd_m().nvals(),
                t.fwd_dp().nvals(),
                t.fwd_dm().nvals(),
                t.edge_versioned().m().nvals(),
                t.edge_versioned().dp().nvals(),
                t.edge_versioned().dm().nvals(),
            )
        };
        // The candidate: `Matrix::wait` short-circuits on `has_pending`, which
        // tracks pending *tuples*. A hypersparse matrix can separately be
        // missing its hyper-hash, and then every row lookup is a binary search
        // over nvec instead of a hash probe. Force the materialize and re-read.
        // The candidate the numbers point at: `me`'s delta-plus is non-empty in
        // exactly the high-state rows. Fold it and re-read.
        many.fold_me_for_test();
        let (i3, _) = probe_cost(&many);
        println!(
            "{:>10}  {:>14.1}  {:>14.1}  {:>8.1}  {:>14.1}   one[{}]",
            pairs,
            i1,
            i2,
            i2 - i1,
            i3,
            format!("one {}  many {}", fmt(&one), fmt(&many))
        );
    }
}

/// **The causal test.** Everything above is correlation: the high state lands
/// exactly on the rows where `me`'s delta-plus is non-empty, at a magnitude that
/// does not vary with how big that delta is. This forces the variable directly.
///
/// Take the single-batch tensor, which reads in the low state with `me.dp`
/// empty. Add **one** edge to a pair the probes never touch — enough to put a
/// single entry in `me.dp` and nothing else — and re-read the same pairs. If the
/// read jumps by the same ~1,300, a non-empty `me.dp` is the cause and its size
/// is irrelevant: what costs is having to consult the layer at all.
#[test]
#[ignore]
fn issue_2430_one_pending_id() {
    ensure_init();
    println!("\n=== #2430: one pending id in `me` against none ===");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>9}  {:>10}",
        "pairs", "me.dp = 0", "me.dp = 1", "delta", "me base"
    );
    for pairs in [11_000u64, 41_000, 121_000] {
        let t = built(pairs, PROBES.min(pairs));
        assert_eq!(
            t.edge_versioned().dp().nvals(),
            0,
            "fixture should start folded"
        );
        let (before, _) = probe_cost(&t);

        // Promote one *filler* pair — a single-edge pair the probes never read.
        // Promotion is what writes into `me`, so this is the smallest edit that
        // leaves `me.dp` non-empty: a single-edge pair gains a second id and both
        // land in the delta.
        let mut dirty = t.dup();
        let i = pairs - 1;
        let (fs, fd) = (i % ROWS, (i * 7 + 3) % ROWS);
        dirty.set_all_from_slices(&[fs], &[fd], &[u32::MAX as u64]);
        dirty.wait();
        let pending = dirty.edge_versioned().dp().nvals();
        let (after, _) = probe_cost(&dirty);

        println!(
            "{:>10}  {:>12.1}  {:>12.1}  {:>9.1}  {:>10}   (me.dp = {pending})",
            pairs,
            before,
            after,
            after - before,
            dirty.edge_versioned().m().nvals(),
        );
    }
}
