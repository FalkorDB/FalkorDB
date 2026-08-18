//! What blocking the `me` key costs, and where it starts to matter.
//!
//! `compound_key` splits a pair across a *block* and a row within it, so `me` is
//! one matrix per live block instead of one matrix. Every maintenance path —
//! flush, wait, fold, sync checks, sizing — folds over the live blocks, which
//! turns per-transaction work that was constant into work proportional to how
//! many blocks a graph has touched.
//!
//! Two things worth knowing, and this measures both.
//!
//! [`block_scaling_zero_cost`] is the case that decides whether the change is
//! acceptable at all: a graph inside block `(0, 0)` — every graph up to 2^30
//! nodes per axis, which is every graph — must pay nothing measurable for a
//! generality it never uses.
//!
//! [`block_scaling_crossover`] is the case the design trades away. It spreads
//! the same number of multi-edge pairs over an increasing number of blocks and
//! reports per-transaction cost against block count, so the slope is visible
//! rather than asserted.
//!
//! Run with:
//!   cargo test --release -p graph block_scaling -- --ignored --nocapture --test-threads=1

use super::instr::read_instr;
use super::tensor::{BLOCK_SHIFT, Tensor};
use super::test_init::ensure_init;

/// Multi-edge pairs in every fixture, held constant so block count is the only
/// variable.
const PAIRS: u64 = 20_000;

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

/// `PAIRS` two-edge pairs spread evenly over `blocks` blocks along the source
/// axis. `blocks == 1` is block `(0, 0)` alone, i.e. today's shape.
fn spread(blocks: u64) -> Tensor {
    let span = 1u64 << BLOCK_SHIFT;
    let mut t = Tensor::new(blocks * span + PAIRS + 2, PAIRS + 2);
    let (mut s, mut d, mut i) = (Vec::new(), Vec::new(), Vec::new());
    for p in 0..PAIRS {
        let src = (p % blocks) * span + p;
        for e in 0..2 {
            s.push(src);
            d.push(p);
            i.push(p * 2 + e);
        }
    }
    t.set_all_from_slices(&s, &d, &i);
    let mut t = t.dup();
    t.flush();
    t.wait();
    t
}

/// The case that has to be free: a graph that never leaves block `(0, 0)`.
///
/// Reported per *transaction*, not per pair, because that is the quantity the
/// block loop makes proportional to block count.
#[test]
#[ignore]
fn block_scaling_zero_cost() {
    ensure_init();
    println!("\n=== block (0,0) only, {PAIRS} multi-edge pairs ===");
    println!("{:>34}  {:>14}", "operation", "instr/tx");
    let mut t = spread(1);
    for _ in 0..3 {
        let c = per_op(1, || {
            t.wait();
            t.flush();
        });
        println!("{:>34}  {:>14.1}", "wait + flush (one transaction)", c);
    }
    for _ in 0..3 {
        let c = per_op(1, || {
            std::hint::black_box(t.edge_count());
        });
        println!("{:>34}  {:>14.1}", "edge_count", c);
    }
    for _ in 0..3 {
        let c = per_op(1, || {
            std::hint::black_box(t.memory_usage());
        });
        println!("{:>34}  {:>14.1}", "memory_usage", c);
    }
}

/// The case the design trades away: the same pairs spread thinner and thinner.
#[test]
#[ignore]
fn block_scaling_crossover() {
    ensure_init();
    println!("\n=== {PAIRS} multi-edge pairs spread over N blocks ===");
    println!(
        "{:>8}  {:>14}  {:>14}  {:>14}",
        "blocks", "wait+flush", "edge_count", "memory_usage"
    );
    for blocks in [1u64, 2, 4, 8, 16, 32, 64] {
        let mut t = spread(blocks);
        let mut wf = f64::MAX;
        let mut ec = f64::MAX;
        let mut mu = f64::MAX;
        for _ in 0..3 {
            wf = wf.min(per_op(1, || {
                t.wait();
                t.flush();
            }));
            ec = ec.min(per_op(1, || {
                std::hint::black_box(t.edge_count());
            }));
            mu = mu.min(per_op(1, || {
                std::hint::black_box(t.memory_usage());
            }));
        }
        println!("{blocks:>8}  {wf:>14.1}  {ec:>14.1}  {mu:>14.1}");
        assert_eq!(
            t.edge_count(),
            2 * PAIRS,
            "fixture lost edges at {blocks} blocks"
        );
        assert_eq!(
            t.multi_pairs(),
            PAIRS,
            "fixture lost pairs at {blocks} blocks"
        );
    }
}

/// Does shrinking `me`'s declared dimensions buy anything?
///
/// GraphBLAS 10 chooses 32- or 64-bit index arrays per matrix from its
/// dimensions, so a matrix declared `GrB_INDEX_MAX` square stores 64-bit row
/// and column indices even when it holds four entries. `me`'s columns are edge
/// ids and its rows are compound keys, and the two are independent: the column
/// index array holds one entry per stored id and is the large one, while the
/// hyperlist holds one per multi-edge pair.
///
/// This prices each choice against the same content, and reports the index
/// widths GraphBLAS actually picked so the numbers are attributable rather than
/// inferred.
#[test]
#[ignore]
fn block_scaling_index_width() {
    use super::matrix::Matrix;

    ensure_init();
    const IDS: u64 = 200_000;
    const ROWS: u64 = 20_000;
    let max = super::tensor::GrB_INDEX_MAX;
    println!(
        "global 32-bit hint set: {:?}",
        Matrix::<bool>::global_hint_32bit_for_test()
    );

    println!(
        "\n=== `me` shape vs bytes, {ROWS} rows x {} ids each ===",
        IDS / ROWS
    );
    println!(
        "{:>26}  {:>12}  {:>10}  {:>10}  {:>12}",
        "declared dims", "bytes", "row bits", "col bits", "B/id"
    );

    let u32max = u64::from(u32::MAX);
    for (label, nrows, ncols, hint) in [
        ("2^60 x 2^60 (today)", max, max, false),
        // 2^32-1 is not enough: GraphBLAS needs the dimension itself to fit a
        // `u32`, and 2^32 does not. The cutoff is 2^31.
        ("(2^32-1) x (2^32-1)", u32max, u32max, false),
        ("2^31 x 2^31", 1u64 << 31, 1u64 << 31, false),
        // Rows and columns are chosen independently. Columns are edge ids and
        // hold one array entry per stored id; rows hold one per multi-edge
        // pair — which is why only the column half is worth narrowing, and why
        // `me` is built at the shape on the next line.
        ("2^60 rows x 2^31 cols", max, 1u64 << 31, false),
        ("2^31 rows x 2^60 cols", 1u64 << 31, max, false),
        // The hint is accepted (`GrB_SUCCESS`) and changes nothing: the
        // declared dimensions are the only lever. Kept so the next person does
        // not spend the afternoon re-discovering it.
        ("2^60 x 2^60, hint 32", max, max, true),
    ] {
        let mut m = Matrix::<bool>::new(nrows, ncols).into_hyper();
        let status = if hint {
            format!("{:?}", m.hint_32bit_indices_for_test())
        } else {
            "-".to_string()
        };
        for i in 0..IDS {
            m.set(i % ROWS, i, true);
        }
        m.wait();
        let bytes = m.memory_usage();
        let (rb, cb) = m.integer_bits_for_test();
        println!(
            "{label:>26}  {bytes:>12}  {rb:>10}  {cb:>10}  {:>12.2}  set={status}",
            bytes as f64 / IDS as f64
        );
    }
}
