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

use super::tensor::Tensor;
use super::test_init::ensure_init;

/// Instruction counter, same `proc_pid_rusage` backend as the other benches.
#[cfg(target_os = "macos")]
fn read_instr() -> Option<u64> {
    use std::os::raw::{c_int, c_void};
    unsafe extern "C" {
        fn proc_pid_rusage(
            pid: c_int,
            flavor: c_int,
            buffer: *mut c_void,
        ) -> c_int;
        fn getpid() -> c_int;
    }
    let mut buf = [0u8; 512];
    let rc = unsafe { proc_pid_rusage(unsafe { getpid() }, 4, buf.as_mut_ptr().cast()) };
    if rc != 0 {
        return None;
    }
    let off = 16 + 29 * 8;
    Some(u64::from_le_bytes(buf[off..off + 8].try_into().unwrap()))
}

#[cfg(not(target_os = "macos"))]
fn read_instr() -> Option<u64> {
    None
}

/// `pairs` populated adjacency rows, of which the first `multi` are two-edge.
/// Mirrors the fixture the issue's table was taken on: the *read* is always the
/// same 1,000 two-edge pairs; only how much else the graph holds varies.
fn built(
    pairs: u64,
    multi: u64,
) -> Tensor {
    let n = pairs + 2;
    let mut t = Tensor::new(n, n);
    let (mut srcs, mut dsts, mut ids) = (Vec::new(), Vec::new(), Vec::new());
    let mut next = 0u64;
    for i in 0..pairs {
        let k = if i < multi { 2 } else { 1 };
        for _ in 0..k {
            srcs.push(i);
            dsts.push(i + 1);
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
        std::hint::black_box(t.get(i, i + 1).count());
    }
    let i0 = read_instr();
    let t0 = Instant::now();
    let mut acc = 0usize;
    for _ in 0..REPS {
        for i in 0..PROBES {
            acc += t.get(i, i + 1).count();
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
