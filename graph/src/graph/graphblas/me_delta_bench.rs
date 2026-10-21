//! #2430: a multi-edge point read costs ~1,200 extra instructions whenever
//! `me`'s delta-plus is non-empty, however little it holds.
//!
//! The issue reports point-read cost occupying two discrete states, selected by
//! graph size in a way that is not monotonic in size. The cause is that `me` is
//! a three-layer [`VersionedMatrix`] and reading a row builds a merge over all
//! three; the layer-empty short-circuit in `Iter::from_layers` is *global* — it
//! asks whether the delta holds anything at all, not whether it holds anything
//! in the row being read. So one pending identifier anywhere in `me` puts every
//! multi-edge read in the high state, and which state a graph is in is decided
//! by where the fold policy last fired relative to the final write, which is not
//! a function of size.
//!
//! [`me_delta_step`] is the repro: the same tensor, the same rows, read with
//! `me.dp` empty and then with a single pending identifier on a pair the probes
//! never touch. Before the row filter that step was **+1,275 instructions per
//! read**; with it, +102.
//!
//! [`me_delta_attach_vs_seek`] decomposes the step, and is why the fix skips the
//! `GxB_Iterator` rather than skipping the merge: on this fixture a fresh
//! iterator per read costs 2,310 where one re-seeked costs 638, so attaching is
//! most of what a narrow scan pays, and a delta that adds a second attach is
//! most of what #2430 reports. **That decomposition also names a larger cost
//! this fix does not address**: `Tensor::get` builds a fresh merge per call, so
//! every multi-edge point read pays ~1,670 instructions of attach whether or not
//! a delta exists. Removing that means giving the hot callers a reusable cursor,
//! which is an API change and a separate piece of work.
//!
//! Run with:
//!   cargo test --release -p graph me_delta -- --ignored --nocapture --test-threads=1

use super::instr::read_instr;
use super::tensor::Tensor;
use super::test_init::ensure_init;

/// Pairs in the fixture. Large enough that the per-read cost dominates the loop
/// and small enough to stay in cache, so the step is not confounded by misses.
const PAIRS: u64 = 100_000;

/// Reads per measured block.
const READS: u64 = 200_000;

fn per_op<F: FnMut()>(
    ops: u64,
    mut f: F,
) -> f64 {
    let a = read_instr();
    f();
    let b = read_instr();
    match (a, b) {
        (Some(a), Some(b)) => (b - a) as f64 / ops as f64,
        _ => f64::NAN,
    }
}

/// `PAIRS` two-edge pairs on the diagonal, committed: `me` holds `2 * PAIRS`
/// identifiers in its base and nothing in either delta.
fn committed_multi() -> Tensor {
    let mut t = Tensor::new(PAIRS + 1, PAIRS + 1);
    let (mut s, mut d, mut i) = (Vec::new(), Vec::new(), Vec::new());
    for p in 0..PAIRS {
        for e in 0..2 {
            s.push(p);
            d.push(p);
            i.push(p * 2 + e);
        }
    }
    t.set_all_from_slices(&s, &d, &i);
    let mut t = t.dup();
    t.flush();
    t.wait();
    assert_eq!(
        t.edge_versioned_block_0().dp().nvals(),
        0,
        "fixture starts clean"
    );
    t
}

/// **The repro.** One pending identifier in `me.dp`, on a pair no probe reads,
/// and every multi-edge read gets dearer.
#[test]
#[ignore]
fn me_delta_step() {
    ensure_init();
    let mut t = committed_multi();
    println!("\n=== multi-edge point read, {PAIRS} pairs x 2 edges ===");
    println!("{:>34}  {:>10}  {:>12}", "state", "|me.dp|", "instr/read");

    let probe = |t: &Tensor| {
        per_op(READS, || {
            let mut acc = 0u64;
            for r in 0..READS {
                let p = r % PAIRS;
                for id in t.get(p, p) {
                    acc = acc.wrapping_add(id);
                }
            }
            std::hint::black_box(acc);
        })
    };

    for _ in 0..3 {
        println!(
            "{:>34}  {:>10}  {:>12.1}",
            "committed, delta empty",
            t.edge_versioned_block_0().dp().nvals(),
            probe(&t)
        );
    }

    // One extra identifier on the *last* pair. The probes above read pairs
    // `0..PAIRS` in order and every one of them is already multi-edge, so this
    // changes what any of them holds by nothing — only whether `me.dp` is empty.
    t.set_all_from_slices(&[PAIRS - 1], &[PAIRS - 1], &[u64::from(u32::MAX)]);
    t.wait();
    assert_eq!(
        t.edge_versioned_block_0().dp().nvals(),
        1,
        "expected exactly one pending id"
    );

    for _ in 0..3 {
        println!(
            "{:>34}  {:>10}  {:>12.1}",
            "one pending id in me.dp",
            t.edge_versioned_block_0().dp().nvals(),
            probe(&t)
        );
    }
}

/// Where the step goes. `Tensor::get` builds a fresh three-layer merge per call,
/// so a non-empty delta adds one `GxB_Iterator` attach *and* one seek per read.
/// The fix is different depending on which dominates: a cheap "is this row in
/// the delta at all" test removes both, but reusing an attached iterator removes
/// only the attach.
#[test]
#[ignore]
fn me_delta_attach_vs_seek() {
    ensure_init();
    println!("\n=== where a non-empty me.dp spends its instructions ===");
    println!("{:>34}  {:>10}  {:>12}", "operation", "|me.dp|", "instr/op");

    let mut t = committed_multi();
    for round in 0..2 {
        let me = t.edge_versioned_block_0();
        let n = me.dp().nvals();

        for _ in 0..3 {
            let c = per_op(READS, || {
                for r in 0..READS {
                    let (_, k) = super::tensor::compound_key(r % PAIRS, r % PAIRS);
                    std::hint::black_box(me.iter(k, k).count());
                }
            });
            println!("{:>34}  {:>10}  {:>12.1}", "fresh me.iter per read", n, c);
        }

        for _ in 0..3 {
            let mut it = me.iter(0, 0);
            let c = per_op(READS, || {
                for r in 0..READS {
                    let (_, k) = super::tensor::compound_key(r % PAIRS, r % PAIRS);
                    it.seek(k, k);
                    std::hint::black_box(it.by_ref().count());
                }
            });
            println!("{:>34}  {:>10}  {:>12.1}", "one iterator, re-seeked", n, c);
        }

        if round == 0 {
            t.set_all_from_slices(&[PAIRS - 1], &[PAIRS - 1], &[u64::from(u32::MAX)]);
            t.wait();
        }
    }
}
