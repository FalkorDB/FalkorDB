//! Where node degree spends its instructions.
//!
//! The C engine answers degree in time proportional to the number of *pairs* in
//! the row: `Tensor_RowDegree` scans the row's cells and adds
//! `GrB_Vector_nvals` of any container it finds, which is a stored length. This
//! engine has no degree entry point at all — `Graph::get_node_outdegree` counts
//! `Tensor::iter(id, id, false)`, which materialises every identifier into a
//! `Vec` and throws them away.
//!
//! Measured at the data-structure boundary the gap was about 2x at `k = 1` and
//! 4x at `k = 2`. This decomposes it, because the two obvious explanations imply
//! different fixes: if the cost is *materialising identifiers* then not
//! materialising them is the fix, and if it is *attaching a `GxB_Iterator` per
//! call* then no amount of counting-instead-of-collecting will move it.
//!
//! **It is mostly the second, and that was not the expectation.** At `k = 1`
//! counting the identifiers costs 113 instructions of the 2,065; the row scan
//! underneath costs 1,952, of which 1,654 is getting an iterator — the same scan
//! through one that is re-seeked costs 298. So the dominant term is per-call
//! iterator setup, and a degree that merely stops collecting identifiers cannot
//! move much of it.
//!
//! `row_degree` / `col_degree` take what that leaves reachable, and no more:
//! they drop the per-pair `Vec`, and share one `me` cursor across a row instead
//! of attaching per multi-edge pair. That is worth 4% at `k = 1` and 12% at
//! `k = 2`, and rows (3) and (4) below show what it does not reach — the
//! per-call attach, which a cursor shared within one call cannot amortise across
//! calls. Removing that means not attaching, i.e. a reusable cursor threaded
//! through the callers, which is a larger API change than this one.
//!
//! Run with:
//!   cargo test --release -p graph degree_ -- --ignored --nocapture --test-threads=1

use super::instr::read_instr;
use super::tensor::Tensor;
use super::test_init::ensure_init;

const PAIRS: u64 = 100_000;
const OPS: u64 = 200_000;

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

/// `PAIRS` pairs on the diagonal, `k` edges each, committed.
fn built(k: u64) -> Tensor {
    let mut t = Tensor::new(PAIRS + 1, PAIRS + 1);
    let (mut s, mut d, mut i) = (Vec::new(), Vec::new(), Vec::new());
    for p in 0..PAIRS {
        for e in 0..k {
            s.push(p);
            d.push(p);
            i.push(p * k + e);
        }
    }
    t.set_all_from_slices(&s, &d, &i);
    let mut t = t.dup();
    t.flush();
    t.wait();
    t
}

#[test]
#[ignore]
fn degree_decomposition() {
    ensure_init();
    for k in [1u64, 2] {
        let t = built(k);
        println!("\n=== degree, {PAIRS} pairs x {k} edge(s) ===");
        println!("{:>44}  {:>12}", "operation", "instr/op");

        // (0) the dedicated entry points
        for _ in 0..3 {
            let c = per_op(OPS, || {
                let mut n = 0u64;
                for r in 0..OPS {
                    n += t.row_degree(r % PAIRS);
                }
                std::hint::black_box(n);
            });
            println!("{:>44}  {:>12.1}", "row_degree", c);
        }
        for _ in 0..3 {
            let c = per_op(OPS, || {
                let mut n = 0u64;
                for r in 0..OPS {
                    n += t.col_degree(r % PAIRS);
                }
                std::hint::black_box(n);
            });
            println!("{:>44}  {:>12.1}", "col_degree", c);
        }

        // (1) what the engine did before
        for _ in 0..3 {
            let c = per_op(OPS, || {
                let mut n = 0usize;
                for r in 0..OPS {
                    n += t.iter(r % PAIRS, r % PAIRS, false).count();
                }
                std::hint::black_box(n);
            });
            println!(
                "{:>44}  {:>12.1}",
                "was: Tensor::iter(r,r,false).count()", c
            );
        }

        // (2) the same, transposed — the in-degree path
        for _ in 0..3 {
            let c = per_op(OPS, || {
                let mut n = 0usize;
                for r in 0..OPS {
                    n += t.iter(r % PAIRS, r % PAIRS, true).count();
                }
                std::hint::black_box(n);
            });
            println!("{:>44}  {:>12.1}", "was: transposed", c);
        }

        // (3) the forward row scan alone, counting cells rather than edges.
        // This is the floor any degree implementation pays: it must learn which
        // pairs the row holds. The difference from (1) is everything spent on
        // turning cells into identifiers.
        for _ in 0..3 {
            let c = per_op(OPS, || {
                let mut n = 0usize;
                for r in 0..OPS {
                    n += t.fwd_iter_for_test(r % PAIRS, r % PAIRS).count();
                }
                std::hint::black_box(n);
            });
            println!("{:>44}  {:>12.1}", "floor: forward row scan, cells only", c);
        }

        // (4) the same forward row scan through *one* iterator, re-seeked. The
        // difference from (3) is the per-call `GxB_Iterator` attach and free,
        // which no amount of counting-instead-of-collecting can remove.
        for _ in 0..3 {
            let mut it = t.fwd_iter_for_test(0, 0);
            let c = per_op(OPS, || {
                let mut n = 0usize;
                for r in 0..OPS {
                    it.seek(r % PAIRS, r % PAIRS);
                    n += it.by_ref().count();
                }
                std::hint::black_box(n);
            });
            println!("{:>44}  {:>12.1}", "forward row scan, one iterator", c);
        }

        // (5) and the `me` side, likewise: one iterator re-seeked per row. At
        // `k = 1` this reads nothing and prices the wasted attach alone. The
        // fixture is on the diagonal below 2^30, so every pair is in block 0.
        for _ in 0..3 {
            let me = t.edge_versioned_block_0();
            let mut it = me.iter(0, 0);
            let c = per_op(OPS, || {
                let mut n = 0usize;
                for r in 0..OPS {
                    let (_, key) = super::tensor::compound_key(r % PAIRS, r % PAIRS);
                    it.seek(key, key);
                    n += it.by_ref().count();
                }
                std::hint::black_box(n);
            });
            println!("{:>44}  {:>12.1}", "me row scan, one iterator", c);
        }
    }
}
