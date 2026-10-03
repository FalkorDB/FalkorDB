//! Is demotion hysteresis worth it? (`docs/papers/OPEN_WORK.md` item 4.)
//!
//! Demotion is eager: the moment a pair drops to one edge its surviving id
//! returns inline and its `me` row empties. A workload that oscillates across
//! the one-to-two boundary therefore pays a promotion *and* a demotion per
//! cycle. Hysteresis — keeping the pair in `me` until the transaction ends —
//! would trade a little space for stability, and it is not obvious which wins.
//!
//! This measures the thing the policy decision actually turns on, without
//! implementing the policy: **what one oscillation costs today**, split into the
//! part hysteresis would remove and the part it would not.
//!
//! An oscillation is `+1 edge` (promote) then `−1 edge` (demote) on the same
//! pair. Under hysteresis, a pair that re-promotes inside the same transaction
//! pays neither: the row is still there. So the saving hysteresis can offer is
//! bounded above by the promote+demote pair measured here, and the cost it adds
//! is an `me` row outliving its need — `promote_ids_bytes` below prices that.
//!
//! Deliberately *not* measured here: the non-oscillating workloads hysteresis
//! must not slow down. Those are the whole benchmark suite, and the right way to
//! check them is `bench measure` before and after a real implementation.
//!
//! Run with:
//!   cargo test --release -p graph oscillation -- --ignored --nocapture --test-threads=1

use std::time::Instant;

use super::instr::read_instr;
use super::tensor::Tensor;
use super::test_init::ensure_init;

const N: u64 = 50_000;

/// `N` single-edge pairs on the diagonal, committed.
fn base() -> Tensor {
    let mut t = Tensor::new(N + 2, N + 2);
    let (srcs, dsts, ids): (Vec<_>, Vec<_>, Vec<_>) = (0..N).map(|i| (i, i + 1, i)).fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut a, mut b, mut c), (s, d, e)| {
            a.push(s);
            b.push(d);
            c.push(e);
            (a, b, c)
        },
    );
    t.set_all_from_slices(&srcs, &dsts, &ids);
    let mut t = t.dup();
    t.flush();
    t.wait();
    t
}

fn measure<F: FnMut()>(
    ops: u64,
    mut f: F,
) -> (f64, f64) {
    let i0 = read_instr();
    let t0 = Instant::now();
    f();
    let el = t0.elapsed();
    let i1 = read_instr();
    (
        match (i0, i1) {
            (Some(a), Some(b)) => (b - a) as f64 / ops as f64,
            _ => f64::NAN,
        },
        el.as_secs_f64() * 1e9 / ops as f64,
    )
}

/// The oscillation cycle, priced. Each row is per pair.
#[test]
#[ignore]
fn oscillation_cost() {
    ensure_init();
    println!("\n=== one promote/demote oscillation, {N} pairs, per pair ===");
    println!("{:>44}  {:>12}  {:>10}", "phase", "instr/pair", "ns/pair");

    let second: Vec<u64> = (0..N).map(|i| N + i).collect();
    let srcs: Vec<u64> = (0..N).collect();
    let dsts: Vec<u64> = (0..N).map(|i| i + 1).collect();

    for rep in 0..3 {
        let mut t = base();

        // promote: every pair gains a second edge
        let (pi, pn) = measure(N, || {
            t.set_all_from_slices(&srcs, &dsts, &second);
        });
        t.wait();
        assert_eq!(
            t.edge_versioned_block_0().nvals(),
            2 * N,
            "promotion did not fill me"
        );

        // demote: every pair loses it again
        let rels: Vec<(u64, u64, u64)> = (0..N).map(|i| (second[i as usize], i, i + 1)).collect();
        let (di, dn) = measure(N, || {
            t.remove_all(&rels);
        });
        t.wait();
        assert_eq!(
            t.edge_versioned_block_0().nvals(),
            0,
            "demotion did not empty me"
        );

        println!("{:>44}  {:>12.1}  {:>10.1}", "promote (+1 edge)", pi, pn);
        println!("{:>44}  {:>12.1}  {:>10.1}", "demote  (-1 edge)", di, dn);
        println!(
            "{:>44}  {:>12.1}  {:>10.1}",
            "== full oscillation",
            pi + di,
            pn + dn
        );
        if rep == 0 {
            println!("{:>44}", "-- repeats --");
        }
    }
}

/// The control: the same two calls on pairs that do *not* cross the boundary.
/// Hysteresis cannot remove this part, so the difference is the ceiling on what
/// it could save.
#[test]
#[ignore]
fn oscillation_control() {
    ensure_init();
    println!("\n=== control: same calls, no transition ({N} pairs, per pair) ===");
    println!("{:>44}  {:>12}  {:>10}", "phase", "instr/pair", "ns/pair");

    let srcs: Vec<u64> = (0..N).collect();
    let dsts: Vec<u64> = (0..N).map(|i| i + 1).collect();

    for _ in 0..3 {
        // start already multi-edge, so +1/-1 never crosses the boundary
        let mut t = Tensor::new(N + 2, N + 2);
        let (mut s2, mut d2, mut i2) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..N {
            for j in 0..2 {
                s2.push(i);
                d2.push(i + 1);
                i2.push(i * 2 + j);
            }
        }
        t.set_all_from_slices(&s2, &d2, &i2);
        let mut t = t.dup();
        t.flush();
        t.wait();

        let third: Vec<u64> = (0..N).map(|i| 2 * N + i).collect();
        let (pi, pn) = measure(N, || {
            t.set_all_from_slices(&srcs, &dsts, &third);
        });
        t.wait();
        let rels: Vec<(u64, u64, u64)> = (0..N).map(|i| (third[i as usize], i, i + 1)).collect();
        let (di, dn) = measure(N, || {
            t.remove_all(&rels);
        });
        t.wait();
        assert_eq!(
            t.edge_versioned_block_0().nvals(),
            2 * N,
            "control changed state"
        );

        println!(
            "{:>44}  {:>12.1}  {:>10.1}",
            "add 3rd edge (no transition)", pi, pn
        );
        println!(
            "{:>44}  {:>12.1}  {:>10.1}",
            "del 3rd edge (no transition)", di, dn
        );
        println!(
            "{:>44}  {:>12.1}  {:>10.1}",
            "== full cycle",
            pi + di,
            pn + dn
        );
    }
}

/// What hysteresis would cost: an `me` row held past its need. Prices the row
/// that a deferred demotion keeps alive.
#[test]
#[ignore]
fn oscillation_row_bytes() {
    ensure_init();
    println!("\n=== the space hysteresis would hold ({N} pairs) ===");
    let mut t = base();
    let inline = t.memory_usage();
    let second: Vec<u64> = (0..N).map(|i| N + i).collect();
    let srcs: Vec<u64> = (0..N).collect();
    let dsts: Vec<u64> = (0..N).map(|i| i + 1).collect();
    t.set_all_from_slices(&srcs, &dsts, &second);
    t.wait();
    let promoted = t.memory_usage();
    println!(
        "  all-inline {inline} B, all-promoted {promoted} B, \
         delta {} B over {N} pairs = {:.1} B/pair",
        promoted - inline,
        (promoted - inline) as f64 / N as f64
    );
    println!(
        "  a deferred demotion holds that per oscillating pair until commit; \
         eager demotion returns it immediately."
    );
}
