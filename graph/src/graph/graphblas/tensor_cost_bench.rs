//! Measurements of the [`super::tensor::Tensor`] edge-storage design, taken at
//! the data-structure boundary rather than through the query engine.
//!
//! Every other number about this design comes from driving a whole engine
//! (`GRAPH.QUERY` through parser, planner, runtime), where the tensor is one
//! component among many and a difference is therefore only weakly attributable
//! to edge storage. These benches call the `Tensor` API directly, so what they
//! report is the storage design and nothing else.
//!
//! The design under measurement — call it **inline-first with lazy overflow** —
//! keeps the forward adjacency `UINT64` and stores a single-edge pair's lone id
//! *inline* in the matrix cell. The auxiliary id matrix `me` stays empty until a
//! pair gains a second edge, at which point the pair is **promoted**: all of its
//! ids move to `me` and a `MULTI_EDGE` sentinel takes the cell. Removal
//! **demotes** symmetrically. Two alternatives frame it:
//!
//! * *container-per-cell* — the cell holds a tagged pointer to a heap container
//!   once a pair has ≥ 2 edges (what the C implementation does). Not measurable
//!   from this crate.
//! * *always-materialised auxiliary matrix* — the adjacency stays `BOOL` and
//!   **every** id lives in `me`, so there is no promotion and no sentinel, but a
//!   simple graph pays a second matrix, a second entry per edge, and a second
//!   lookup on every read. This design does not exist in any implementation, so
//!   nothing below measures it; [`tensor_cost_space`] measures the *components*
//!   its space bill would be made of and labels the sum a model.
//!
//! ## What they found
//!
//! Taken on macOS/arm64, release build, and quoted per the geometry each bench
//! states. The C figures alongside come from a standalone harness against the C
//! implementation's `Tensor` (`Tensor_SetElements` / `TensorIterator_*` /
//! `Tensor_RemoveElements`), run on the *same* fixture — 200,000 pairs on the
//! diagonal — which is what [`tensor_cost_c_comparable`] exists to match.
//!
//! | quantity (diagonal fixture, per op)  | inline-first | container-per-cell |
//! |--------------------------------------|-------------:|-------------------:|
//! | memory, 1 edge/pair                  |  25.0 B/edge |        24.0 B/edge |
//! | memory, 2 edges/pair                 |  36.8 B/edge |       148.0 B/edge |
//! | point read + all ids, 1 edge/pair    |    566 instr |         553 instr |
//! | point read + all ids, 2 edges/pair   |  2,502 instr |         751 instr |
//! | full scan /edge, 1 edge/pair         |    163 instr |         149 instr |
//! | full scan /edge, 2 edges/pair        |    250 instr |         186 instr |
//! | promote (add 2nd edge) /pair         |  2,735 instr |      11,548 instr |
//! | demote (drop 1 of 2 ids) /pair       |  50.2M instr |       2,828 instr |
//! | build 2 edges/pair: incremental÷batch |       1.58x |             1.10x |
//!
//! Every row is a direct measurement of a whole operation; spread across reps is
//! under 0.1% except the multi-edge point reads, which moved 2.7% between whole
//! runs (2,502 and 2,570). Two notes on how to read the transition rows:
//!
//! *The transition cannot be isolated by differencing on the container side.*
//! The right control for a promotion is an insert that adds an edge to a pair
//! which is *already* multi — same edges, same pairs, no state change. Here that
//! control costs 1,394 against the promotion's 2,968 (block geometry), so
//! promotion ≈ 1,574 instr/pair, and the diagonal fixture agrees at
//! 2,735 − 1,164 = 1,571. In the C implementation the same control costs
//! **more** than the transition (23,099 vs 11,548) because appending an id to an
//! existing `GrB_Vector` re-materialises it; the difference comes out negative,
//! and so does the analogous engine-level control on both engines. That is why
//! the table compares whole operations rather than differences — and it is a
//! result in its own right: differencing is only a valid isolation when the
//! non-transitioning control is genuinely cheaper.
//!
//! *Batching avoids part of the transition, not all of it.* Building the same
//! 2-edges-per-pair tensor incrementally costs 1.58x the single-batch build here
//! (4,700 vs 2,978 instr/pair) and 1.10x on the C side (12,809 vs 11,606), since
//! `set_all_from_slices` resolves within-batch duplicates through its own map and
//! never probes the committed matrix. The two implementations' ratios are not
//! measured on the same geometry (block here, diagonal there), so compare the
//! ratios, not the absolute per-pair figures.
//!
//! Read that as four separate results:
//!
//!   1. **Space is where inline-first wins, and it wins on the multi-edge case
//!      too.** At one edge per pair the two are level (25.0 vs 24.0 B/edge) —
//!      inline-first pays 8 bytes for the `UINT64` cell where a `BOOL`
//!      adjacency would pay 4, and gets the id storage for free. At two edges
//!      per pair it is 4.0x smaller, because a container costs 272 bytes per
//!      *pair* (measured on the C side) against 24.3 bytes per *id* in `me`.
//!      The auxiliary matrix's marginal cost is ~8-9 B/id on both engines
//!      (measured at the engine level over k = 2..8); the whole difference is
//!      the fixed per-pair overhead.
//!   2. **Reads are where it loses.** A sentinel read is 4.4x an inline read
//!      (2,502 vs 566), while the C container is only 1.4x (751 vs 553) — a
//!      tagged pointer is one dereference, whereas reading `me` is a second
//!      GraphBLAS point lookup into a hypersparse matrix of `ME_DIM` rows.
//!      This is the cost the always-materialised design would pay on
//!      *every* pair, and it is the strongest argument against it.
//!   3. **Promotion is 4.2x cheaper here** (2,735 vs 11,548), because a
//!      promotion is two `me` writes rather than a `GrB_Vector_new` +
//!      `GrB_wait` of a 2^60-dimension vector — the C harness measured that
//!      container's new/set/materialise/free at 11,466 in isolation, i.e. ~99%
//!      of its whole promotion.
//!   4. **Bulk demotion is the outlier and it is a defect, not a design
//!      property.** [`tensor_cost_demote`] pins it down: the per-pair cost
//!      tracks the *batch size* (51.5k instr/pair at 100 pairs, 13.6M at
//!      100,000) and is flat in `|me|`, because `remove_all`'s per-edge path
//!      calls `me.remove` then `me.iter`, and the iter re-materialises the
//!      pending write the previous edge left. C's equivalent is flat at 2,827.
//!      Nothing about inline-first requires this; it is where to look first if
//!      multi-edge deletes ever matter.
//!
//! Caveat on the cross-implementation column: separate processes, separate
//! allocators, and the C harness ran `OMP_NUM_THREADS=1` while GraphBLAS here
//! uses its default thread count. That is why the comparison is stated in
//! instructions and on one shared fixture, and why the bulk-build rows are the
//! ones to trust least.
//!
//! Run with:
//!   cargo test --release -p graph tensor_cost -- --ignored --nocapture --test-threads=1
//!
//! `--test-threads=1` is not optional when instruction counts are wanted:
//! `proc_pid_rusage` is a whole-process counter, so a second bench running
//! concurrently on another libtest thread lands in these numbers.
//!
//! Instruction counts are macOS-only (`proc_pid_rusage`); elsewhere the `instr`
//! columns print `-` and only the microsecond columns are meaningful. Wall
//! clock is reported alongside but is the weaker metric — it drifts with
//! machine load, the instruction count does not.

use std::time::Instant;

use super::instr::read_instr;
use super::matrix::Matrix;
use super::tensor::{GrB_INDEX_MAX, ME_DIM, ME_NARROW_NCOLS, Tensor, compound_key};
use super::test_init::ensure_init;
use super::versioned_matrix::VersionedMatrix;

// --- process instruction counter ---------------------------------------------

/// Per-operation cost of one measured block.
struct Cost {
    instr: Option<f64>,
    us: f64,
}

impl Cost {
    fn fmt_instr(&self) -> String {
        self.instr
            .map_or_else(|| "-".to_string(), |i| format!("{i:.1}"))
    }
}

/// Run `f` once and divide by `ops`, the number of primitive operations it
/// performed.
///
/// The two `read_instr` syscalls sit outside the timed region's per-op
/// accounting only in the sense that they are amortised: every call site below
/// keeps `ops` ≥ 10⁵, so the syscall pair contributes < 0.1 instr/op.
/// [`tensor_cost_point_read`] prints an empty-loop row so that floor is visible
/// rather than assumed.
fn measure<F: FnMut()>(
    ops: u64,
    mut f: F,
) -> Cost {
    let i0 = read_instr();
    let t = Instant::now();
    f();
    let us = t.elapsed().as_secs_f64() * 1e6;
    let i1 = read_instr();
    let n = ops as f64;
    Cost {
        instr: i0.zip(i1).map(|(a, b)| (b - a) as f64 / n),
        us: us / n,
    }
}

// --- fixtures ----------------------------------------------------------------

/// Pair count used by every bench here. 100k pairs is large enough that the
/// per-call fixed costs of `set_all_from_slices`/`remove_all` amortise away and
/// small enough that a 16-ids-per-pair fixture still fits comfortably.
const PAIRS: u64 = 100_000;

/// Node-id space. `PAIRS` = 100 sources x 1000 destinations, so the adjacency
/// is a 100x1000 block inside a 1024x1024 matrix — sparse, and with more than
/// one entry per row, which is what a real adjacency looks like.
const DIM: u64 = 1_024;

/// `i` -> `(src, dst)`, filling destinations before sources.
fn pair(i: u64) -> (u64, u64) {
    (i / 1_000, i % 1_000)
}

/// A committed tensor where pairs `[0, multi)` carry `k` edges each and pairs
/// `[multi, PAIRS)` carry exactly one.
///
/// Built in a single `set_all_from_slices` batch and then committed with
/// `dup` + `flush`, so the state under measurement is the *committed base* `m`
/// (plus `me`), not a pile of pending deltas — that is the state a read query
/// actually meets, and the fold policy means deltas would otherwise dominate
/// whatever a point read touched.
fn built(
    multi: u64,
    k: u64,
) -> Tensor {
    let mut t = Tensor::new(DIM, DIM);
    let (mut srcs, mut dsts, mut ids) = (Vec::new(), Vec::new(), Vec::new());
    let mut next_id = 0u64;
    for i in 0..PAIRS {
        let (s, d) = pair(i);
        let n = if i < multi { k } else { 1 };
        for _ in 0..n {
            srcs.push(s);
            dsts.push(d);
            ids.push(next_id);
            next_id += 1;
        }
    }
    t.set_all_from_slices(&srcs, &dsts, &ids);
    let mut t = t.dup();
    t.flush();
    t.wait();
    assert_eq!(
        t.fwd_m().nvals(),
        PAIRS,
        "fixture not committed: the {PAIRS} pairs did not fold into the base"
    );
    assert_eq!(
        t.edge_versioned_block_0().nvals(),
        multi * k,
        "fixture's `me` does not hold exactly the multi pairs' ids"
    );
    t
}

/// Coordinates of the first `n` pairs at or after `from`, as a probe list.
fn probes(
    from: u64,
    n: u64,
) -> Vec<(u64, u64)> {
    (from..from + n).map(pair).collect()
}

// --- geometry-matched cross-implementation table -----------------------------

/// Pair count and geometry of the standalone C harness this table is meant to
/// be read against (`scratchpad/tensor_bench/tensor_bench.c`): 200,000 pairs on
/// the **diagonal**, one pair per row.
const XN: u64 = 200_000;

/// A committed diagonal tensor of `n` pairs, `k` edges each. [`built_diag`] is
/// this at `n = XN`; the sweep in [`tensor_cost_cold_cache`] needs other sizes.
fn built_n(
    n: u64,
    k: u64,
) -> Tensor {
    let mut t = Tensor::new(n + 1, n + 1);
    let (mut srcs, mut dsts, mut ids) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..k {
            srcs.push(i);
            dsts.push(i);
            ids.push(i * k + j);
        }
    }
    t.set_all_from_slices(&srcs, &dsts, &ids);
    let mut t = t.dup();
    t.flush();
    t.wait();
    assert_eq!(t.fwd_m().nvals(), n, "diagonal fixture not committed");
    t
}

/// A committed diagonal tensor: pairs `(i, i)` for `i < XN`, `k` edges each.
fn built_diag(k: u64) -> Tensor {
    let mut t = Tensor::new(XN + 1, XN + 1);
    let (mut srcs, mut dsts, mut ids) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..XN {
        for j in 0..k {
            srcs.push(i);
            dsts.push(i);
            ids.push(i * k + j);
        }
    }
    t.set_all_from_slices(&srcs, &dsts, &ids);
    let mut t = t.dup();
    t.flush();
    t.wait();
    assert_eq!(t.fwd_m().nvals(), XN, "diagonal fixture not committed");
    assert_eq!(
        t.edge_versioned_block_0().nvals(),
        if k == 1 { 0 } else { XN * k }
    );
    t
}

/// The one table that can be put beside the C implementation's numbers.
///
/// Every other bench in this file uses a 100x1000 block of pairs, which is what
/// a real adjacency looks like but is **not** what the standalone C harness
/// measured. Matrix geometry is not a free parameter here: per-row overhead is
/// charged once per row, so a fixture with one pair per row and a fixture with
/// 1,000 pairs per row report different bytes per edge for the same design.
/// This test therefore rebuilds the C harness's exact fixture — 200,000 pairs
/// on the diagonal, `k` edges each — so the two sets of numbers differ only in
/// the implementation.
///
/// Held fixed against the C harness: pair count (200,000), pairs per row (1),
/// ids per pair, and the operations measured. Not held fixed, and therefore
/// caveats on any comparison: the two are separate processes with separate
/// allocators, and the C side ran `OMP_NUM_THREADS=1` while GraphBLAS here uses
/// its default thread count — which matters for the bulk build rows, not for the
/// point reads (the C harness reported both threadings and only its
/// one-at-a-time insert moved).
#[test]
#[ignore = "measurement, not a correctness check"]
fn tensor_cost_c_comparable() {
    ensure_init();
    const REPS: u32 = 3;
    const READS: u64 = 1_000_000;

    for k in [1u64, 2, 4, 8, 16] {
        let t = built_diag(k);
        let edges = t.edge_count();
        println!(
            "\n=== diagonal fixture, {XN} pairs x {k} edge(s) = {edges} edges (C-comparable) ===",
        );
        println!(
            "tensor memory_usage = {} bytes ({:.1} B/edge, {:.1} B/pair); \
             of which me = {} bytes over {} ids ({:.1} B/id)",
            t.memory_usage(),
            t.memory_usage() as f64 / edges as f64,
            t.memory_usage() as f64 / XN as f64,
            t.edge_versioned_block_0().memory_usage(),
            t.edge_versioned_block_0().nvals(),
            if t.edge_versioned_block_0().nvals() == 0 {
                0.0
            } else {
                t.edge_versioned_block_0().memory_usage() as f64
                    / t.edge_versioned_block_0().nvals() as f64
            }
        );
        println!("{:>34}  {:>12}  {:>10}", "operation", "instr/op", "ns/op");
        for _ in 0..REPS {
            // Point read, first id only: the same logical work either way.
            let c = measure(READS, || {
                let mut acc = 0u64;
                for i in 0..READS {
                    let p = i % XN;
                    acc = acc.wrapping_add(t.get(p, p).next().unwrap_or(0));
                }
                std::hint::black_box(acc);
            });
            println!(
                "{:>34}  {:>12}  {:>10.1}",
                "point read + first id",
                c.fmt_instr(),
                c.us * 1_000.0
            );
        }
        for _ in 0..REPS {
            // Point read draining every id: 1 id at k=1, k ids at k>1.
            let c = measure(READS, || {
                let mut acc = 0u64;
                for i in 0..READS {
                    let p = i % XN;
                    for id in t.get(p, p) {
                        acc = acc.wrapping_add(id);
                    }
                }
                std::hint::black_box(acc);
            });
            println!(
                "{:>34}  {:>12}  {:>10.1}",
                "point read + all ids",
                c.fmt_instr(),
                c.us * 1_000.0
            );
        }
        for _ in 0..REPS {
            let a = measure(edges, || {
                let mut n = 0u64;
                for (_, _, id) in t.iter_edges() {
                    n = n.wrapping_add(id);
                }
                std::hint::black_box(n);
            });
            let b = measure(edges, || {
                let mut n = 0u64;
                for (_, _, id) in t.iter(0, u64::MAX, false) {
                    n = n.wrapping_add(id);
                }
                std::hint::black_box(n);
            });
            println!(
                "{:>34}  {:>12}  {:>10.1}",
                "full iteration /edge (iter_edges)",
                a.fmt_instr(),
                a.us * 1_000.0
            );
            println!(
                "{:>34}  {:>12}  {:>10.1}",
                "full iteration /edge (pair-order)",
                b.fmt_instr(),
                b.us * 1_000.0
            );
        }
        // Transposed iteration. `mt` carries structure only, so each incoming
        // pair costs a forward `eff_get` to recover its ids — the deliberate
        // price of not storing every id twice. This is the row the paper had
        // listed as unmeasured.
        for _ in 0..REPS {
            let c = measure(edges, || {
                let mut n = 0u64;
                for (_, _, id) in t.iter(0, u64::MAX, true) {
                    n = n.wrapping_add(id);
                }
                std::hint::black_box(n);
            });
            println!(
                "{:>34}  {:>12}  {:>10.1}",
                "full iteration /edge (transposed)",
                c.fmt_instr(),
                c.us * 1_000.0
            );
        }
        // Point reads in a scattered order rather than ascending. Every read
        // figure above walks pairs sequentially and warm, which flatters any
        // design whose next lookup is adjacent to the last; this reads the same
        // pairs through an order-`XN` multiplicative stride, so consecutive
        // probes land far apart in the index arrays without changing the set of
        // pairs read or their count.
        for _ in 0..REPS {
            let c = measure(READS, || {
                let mut acc = 0u64;
                let mut p = 1u64;
                for _ in 0..READS {
                    // 48271 is a primitive root mod 2^31-1 (MINSTD); reduced
                    // into range it visits pairs in a scattered, repeatable order.
                    p = p.wrapping_mul(48_271) % 2_147_483_647;
                    let q = p % XN;
                    acc = acc.wrapping_add(t.get(q, q).next().unwrap_or(0));
                }
                std::hint::black_box(acc);
            });
            println!(
                "{:>34}  {:>12}  {:>10.1}",
                "point read + first id (scattered)",
                c.fmt_instr(),
                c.us * 1_000.0
            );
        }
    }

    // Promotion and demotion on the same diagonal fixture, per pair, through
    // the bulk entry points. `add-2nd` promotes all XN committed single-edge
    // pairs; the control adds a 3rd edge to XN already-promoted pairs, so it
    // inserts the same number of edges into the same number of pairs with no
    // state change. `del-2nd` drops one id from all XN 2-edge pairs.
    println!("\n=== diagonal fixture, transitions per pair ===");
    println!(
        "{:>34}  {:>12}  {:>10}",
        "operation", "instr/pair", "ns/pair"
    );
    let diag: Vec<u64> = (0..XN).collect();
    for _ in 0..REPS {
        {
            let mut t = built_diag(2).dup();
            let ids: Vec<u64> = (2 * XN..3 * XN).collect();
            let c = measure(XN, || t.set_all_from_slices(&diag, &diag, &ids));
            t.wait();
            assert_eq!(t.edge_versioned_block_0().nvals(), 3 * XN);
            println!(
                "{:>34}  {:>12}  {:>10.1}",
                "add-3rd (control)",
                c.fmt_instr(),
                c.us * 1_000.0
            );
        }
        {
            let mut t = built_diag(1).dup();
            let ids: Vec<u64> = (XN..2 * XN).collect();
            let c = measure(XN, || t.set_all_from_slices(&diag, &diag, &ids));
            t.wait();
            assert_eq!(t.edge_versioned_block_0().nvals(), 2 * XN);
            println!(
                "{:>34}  {:>12}  {:>10.1}",
                "add-2nd (promote)",
                c.fmt_instr(),
                c.us * 1_000.0
            );
        }
    }
    // One rep only: this row costs ~8 minutes and the three reps of an earlier
    // run agreed to 0.05% (50,162,153 / 50,169,520 / 50,185,016 instr/pair).
    {
        let mut t = built_diag(2).dup();
        let rels: Vec<(u64, u64, u64)> = (0..XN).map(|i| (2 * i + 1, i, i)).collect();
        let c = measure(XN, || {
            std::hint::black_box(t.remove_all(&rels));
        });
        t.wait();
        assert_eq!(t.edge_versioned_block_0().nvals(), 0);
        println!(
            "{:>34}  {:>12}  {:>10.1}",
            "del-2nd (demote, batch of XN)",
            c.fmt_instr(),
            c.us * 1_000.0
        );
    }
}

// --- point reads --------------------------------------------------------------

/// Point-read cost of an inline pair vs a sentinel (promoted) pair.
///
/// Both probe sets are read out of **the same tensor**, so the forward base
/// matrix, its dimensions, its density and its GraphBLAS storage format are
/// identical for the two columns; the only difference is whether the cell holds
/// an edge id or the `MULTI_EDGE` sentinel that sends the read on to `me`. That
/// makes the difference between the columns the cost of the second lookup —
/// which is the per-read tax the always-materialised design would pay on
/// *every* pair, since under it no pair is ever inline.
///
/// Two read shapes are reported because they answer different questions:
/// `first` takes one id (`EdgeIds::next`), which is the same amount of *logical*
/// work on both sides and therefore isolates the lookup; `all` drains every id,
/// which is what a traversal does and necessarily returns `k` ids for a
/// multi-edge pair against 1 for an inline one.
#[test]
#[ignore = "measurement, not a correctness check"]
fn tensor_cost_point_read() {
    ensure_init();
    // Half the pairs promoted to 2 edges, half left inline.
    let half = PAIRS / 2;
    let t = built(half, 2);
    let sentinel = probes(0, half);
    let inline = probes(half, half);
    assert_eq!(t.get(sentinel[0].0, sentinel[0].1).count(), 2);
    assert_eq!(t.get(inline[0].0, inline[0].1).count(), 1);

    const REPS: u32 = 3;
    println!("\n=== point read, one tensor, {half} sentinel + {half} inline pairs ===");
    println!(
        "{:>10}  {:>6}  {:>12}  {:>10}",
        "cell", "shape", "instr/read", "us/read"
    );

    // Loop floor: same iteration, no tensor call. Subtract this from the rows
    // below to get the tensor's own cost.
    for _ in 0..REPS {
        let c = measure(half, || {
            let mut acc = 0u64;
            for &(s, d) in &inline {
                acc = acc.wrapping_add(s ^ d);
            }
            std::hint::black_box(acc);
        });
        println!(
            "{:>10}  {:>6}  {:>12}  {:>10.4}",
            "(none)",
            "loop",
            c.fmt_instr(),
            c.us
        );
    }

    for (label, set) in [("inline", &inline), ("sentinel", &sentinel)] {
        for shape in ["first", "all"] {
            for _ in 0..REPS {
                let c = measure(set.len() as u64, || {
                    let mut acc = 0u64;
                    for &(s, d) in set {
                        if shape == "first" {
                            acc = acc.wrapping_add(t.get(s, d).next().unwrap_or(0));
                        } else {
                            for id in t.get(s, d) {
                                acc = acc.wrapping_add(id);
                            }
                        }
                    }
                    std::hint::black_box(acc);
                });
                println!(
                    "{:>10}  {:>6}  {:>12}  {:>10.4}",
                    label,
                    shape,
                    c.fmt_instr(),
                    c.us
                );
            }
        }
    }
}

// --- the transition ----------------------------------------------------------

/// What **promotion** costs, isolated against a control that does the same
/// amount of surrounding work without changing state.
///
/// This is a cost the always-materialised design would not pay at all — it has
/// no state to change — so it is part of the price of the lazy `me` that keeps a
/// simple graph out of the auxiliary matrix entirely.
///
/// Three measurements, all per *pair*, all through the bulk entry point the
/// engine uses:
///
/// | row                  | what it does                                        |
/// |----------------------|-----------------------------------------------------|
/// | `add-3rd (control)`  | one `set_all_from_slices` adding a 3rd edge to `PAIRS` committed 2-edge pairs. Every pair is already promoted, so nothing transitions. |
/// | `add-2nd (promote)`  | one `set_all_from_slices` adding a 2nd edge to `PAIRS` committed single-edge pairs. Promotes every one. |
/// | `add-1st (fresh)`    | one `set_all_from_slices` adding a 1st edge to `PAIRS` pairs in an unused source band. No promotion, but it grows the pair count. |
/// | `batch-2 (per pair)` | one `set_all_from_slices` building `PAIRS` pairs with 2 edges each from empty, both edges in the same batch. |
///
/// Promote cost = `add-2nd` − `add-3rd`. The control matters: the entry point
/// does per-call work (`flush`, `wait`, hash-map build) that has nothing to do
/// with the transition. `add-3rd` is the right control and `add-1st` is not:
/// adding an edge to a *fresh* pair grows the adjacency, and at the engine level
/// that growth was measured costing more than a promotion (the fresh-pair
/// control made the C engine's promotion come out negative). `add-3rd` inserts
/// into an existing pair whose auxiliary row already exists.
///
/// The control's `me` already holds `2 * PAIRS` ids when it is measured, while
/// the promote pass starts from an empty `me` — a bias *against* the control, so
/// `add-2nd` − `add-3rd` is a lower bound on the transition cost.
///
/// `batch-2` is the row that says what batching saves. `set_all_from_slices`
/// resolves within-batch duplicates through its `batch` map and promotes the
/// pending inline slot in place, so a pair built multi-edge in one batch never
/// probes the committed matrix for an existing id and never writes a `dp`
/// shadow — the transition an incrementally-built pair pays for.
#[test]
#[ignore = "measurement, not a correctness check"]
fn tensor_cost_promote() {
    ensure_init();
    const REPS: u32 = 3;
    println!("\n=== promotion, per pair, {PAIRS} pairs per call ===");
    println!(
        "{:>20}  {:>12}  {:>10}",
        "operation", "instr/pair", "us/pair"
    );

    let coords: Vec<(u64, u64)> = probes(0, PAIRS);
    let srcs: Vec<u64> = coords.iter().map(|c| c.0).collect();
    let dsts: Vec<u64> = coords.iter().map(|c| c.1).collect();

    for _ in 0..REPS {
        // add-3rd: every pair is already promoted, so nothing transitions.
        {
            let base = built(PAIRS, 2);
            let mut t = base.dup();
            let ids: Vec<u64> = (2 * PAIRS..3 * PAIRS).collect();
            let c = measure(PAIRS, || t.set_all_from_slices(&srcs, &dsts, &ids));
            t.wait();
            assert_eq!(
                t.edge_versioned_block_0().nvals(),
                3 * PAIRS,
                "the 3rd edge of every pair did not land in `me`"
            );
            assert_eq!(t.fwd_dp().nvals(), 0, "control wrote a `dp` shadow");
            println!(
                "{:>20}  {:>12}  {:>10.4}",
                "add-3rd (control)",
                c.fmt_instr(),
                c.us
            );
        }
        // add-1st: fresh pairs. Reported for contrast only — it grows the pair
        // count, which is why it is not the control. The fixture occupies
        // sources 0..99; sources 100..199 are pairs the tensor has never seen
        // and are still inside `DIM`. (Shifting the *destination* instead would
        // leave the 1024-column matrix, which GraphBLAS rejects and which then
        // measures a failing write rather than an insert.)
        {
            let base = built(0, 1);
            let mut t = base.dup();
            let fresh: Vec<u64> = srcs.iter().map(|s| s + 100).collect();
            let ids: Vec<u64> = (PAIRS..2 * PAIRS).collect();
            let c = measure(PAIRS, || t.set_all_from_slices(&fresh, &dsts, &ids));
            t.wait();
            assert_eq!(
                t.edge_versioned_block_0().nvals(),
                0,
                "control promoted something"
            );
            assert_eq!(
                t.fwd_dp().nvals(),
                PAIRS,
                "control's inserts did not land in `dp`"
            );
            println!(
                "{:>20}  {:>12}  {:>10.4}",
                "add-1st (fresh)",
                c.fmt_instr(),
                c.us
            );
        }
        // add-2nd: promote every committed single-edge pair.
        {
            let base = built(0, 1);
            let mut t = base.dup();
            let ids: Vec<u64> = (PAIRS..2 * PAIRS).collect();
            let c = measure(PAIRS, || t.set_all_from_slices(&srcs, &dsts, &ids));
            t.wait();
            assert_eq!(
                t.edge_versioned_block_0().nvals(),
                2 * PAIRS,
                "promotion did not move both ids of every pair into `me`"
            );
            assert_eq!(
                t.fwd_dp().nvals(),
                PAIRS,
                "promotion did not shadow every pair with a sentinel"
            );
            println!(
                "{:>20}  {:>12}  {:>10.4}",
                "add-2nd (promote)",
                c.fmt_instr(),
                c.us
            );
        }
        // batch-2: both edges of every pair in one batch.
        {
            let mut t = Tensor::new(DIM, DIM);
            let mut b_srcs = Vec::with_capacity(2 * PAIRS as usize);
            let mut b_dsts = Vec::with_capacity(2 * PAIRS as usize);
            let mut b_ids = Vec::with_capacity(2 * PAIRS as usize);
            for (i, &(s, d)) in coords.iter().enumerate() {
                for j in 0..2u64 {
                    b_srcs.push(s);
                    b_dsts.push(d);
                    b_ids.push(i as u64 * 2 + j);
                }
            }
            let c = measure(PAIRS, || t.set_all_from_slices(&b_srcs, &b_dsts, &b_ids));
            t.wait();
            assert_eq!(t.edge_versioned_block_0().nvals(), 2 * PAIRS);
            println!(
                "{:>20}  {:>12}  {:>10.4}",
                "batch-2 (per pair)",
                c.fmt_instr(),
                c.us
            );
        }
    }
}

/// What **demotion** costs, and why it cannot be quoted as a single constant.
///
/// `remove_all`'s bulk fast path only applies when the tensor has no multi-edge
/// pair at all; the moment one exists every deletion in the call goes edge by
/// edge, and each edge's `eff_get` / `me.iter` re-materialises deltas that the
/// previous edge dirtied. The per-pair cost therefore **grows with the batch
/// size**, so this bench sweeps the batch instead of reporting one figure.
///
/// Two columns, both forced down the same per-edge path so they differ only in
/// the transition:
///
/// * `del-only` — delete the only edge of `n` single-edge pairs, on a tensor
///   holding one extra 2-edge pair purely to disable the fast path. No
///   demotion, and it never touches `me`, so it depends on `n` alone.
/// * `del-2nd` — delete one edge of `n` 2-edge pairs. Demotes every one.
///
/// Demotion cost = `del-2nd` − `del-only` **at the same `n`**, reported as the
/// `demote` column. Two things are varied independently, because the first
/// version of this bench found the cost tracking both:
///
/// * `n`, the batch size — the per-edge path re-materialises deltas that the
///   previous edge dirtied, so both columns grow with `n`.
/// * `|me|`, the total number of ids in the auxiliary matrix, set by how many of
///   the `PAIRS` pairs are promoted. `me.remove` followed by `me.iter` forces
///   `me` to materialise once per demoted pair, so the demotion column tracks
///   `|me|` even at a fixed, tiny batch.
///
/// Consequently there is **no single demotion constant** to quote: the smallest
/// `(n, |me|)` cell is the closest thing to an isolated per-transition cost, and
/// the rest of the table measures the two super-linearities.
#[test]
#[ignore = "measurement, not a correctness check"]
fn tensor_cost_demote() {
    ensure_init();
    println!("\n=== deletion on the per-edge path, per pair ===");
    println!(
        "{:>10}  {:>10}  {:>16}  {:>16}  {:>16}  {:>12}",
        "|me| ids", "batch n", "del-only instr", "del-2nd instr", "demote (diff)", "del-2nd us"
    );

    let coords: Vec<(u64, u64)> = probes(0, PAIRS);
    let srcs: Vec<u64> = coords.iter().map(|c| c.0).collect();
    let dsts: Vec<u64> = coords.iter().map(|c| c.1).collect();

    // The control is independent of `|me|` (it touches `m`/`dp`/`dm`/`mt` only),
    // so it is measured once per batch size and reused down the `|me|` sweep.
    let mut control = |n: u64| -> Cost {
        let mut t = Tensor::new(DIM, DIM);
        let mut s2 = srcs.clone();
        let mut d2 = dsts.clone();
        let mut i2: Vec<u64> = (0..PAIRS).collect();
        for extra in 0..2u64 {
            s2.push(DIM - 1);
            d2.push(DIM - 1);
            i2.push(PAIRS + extra);
        }
        t.set_all_from_slices(&s2, &d2, &i2);
        let mut t = t.dup();
        t.flush();
        t.wait();
        assert!(t.has_multi_edge(), "fast path not disabled");
        let rels: Vec<(u64, u64, u64)> = (0..n)
            .map(|i| {
                let (s, d) = pair(i);
                (i, s, d)
            })
            .collect();
        measure(n, || {
            std::hint::black_box(t.remove_all(&rels));
        })
    };

    // The floor `measure` itself imposes: two `proc_pid_rusage` calls. It is
    // amortised to nothing at n >= 100 but is a real part of the n = 1 and
    // n = 10 rows, so it is printed rather than assumed away.
    let floor = measure(1, || {});
    println!("(measurement floor, 1 op: {} instr)", floor.fmt_instr());

    for n in [1u64, 10, 100, 1_000, 10_000, 100_000] {
        let reps: u32 = if n >= 10_000 { 1 } else { 3 };
        for _ in 0..reps {
            let ctl = control(n);
            for multi in [1_000u64, 10_000, 100_000] {
                if multi < n {
                    continue;
                }
                // Ids are laid out 2i, 2i+1 for each promoted pair; drop the
                // second of each of the first `n`.
                let base = built(multi, 2);
                let mut t = base.dup();
                let rels: Vec<(u64, u64, u64)> = (0..n)
                    .map(|i| {
                        let (s, d) = pair(i);
                        (2 * i + 1, s, d)
                    })
                    .collect();
                let dem = measure(n, || {
                    std::hint::black_box(t.remove_all(&rels));
                });
                t.wait();
                assert_eq!(
                    t.edge_versioned_block_0().nvals(),
                    2 * (multi - n),
                    "demotion left ids in `me` for the pairs it touched"
                );
                let diff = dem
                    .instr
                    .zip(ctl.instr)
                    .map_or_else(|| "-".to_string(), |(a, b)| format!("{:.1}", a - b));
                println!(
                    "{:>10}  {:>10}  {:>16}  {:>16}  {:>16}  {:>12.4}",
                    2 * multi,
                    n,
                    ctl.fmt_instr(),
                    dem.fmt_instr(),
                    diff,
                    dem.us
                );
            }
        }
    }
}

// --- iteration ---------------------------------------------------------------

/// Full-scan cost per edge, all-inline vs all-promoted vs mixed.
///
/// `iter_edges` streams the inline ids in one pass and then drains `me`
/// wholesale, which is the cheap shape a single-edge graph gets for free;
/// `iter(.., false)` is the pair-ordered iterator the runtime traverses with,
/// which buffers a promoted pair's `me` row as it reaches it. Both are reported
/// because the second is what a query pays and the first is what the storage
/// design costs at its best.
///
/// Held fixed: pair count (`PAIRS`) and matrix dimensions. Varied: how many of
/// those pairs are promoted and how many ids each carries — so edge count
/// varies with the row and the metric is per *edge*, not per call.
#[test]
#[ignore = "measurement, not a correctness check"]
fn tensor_cost_iteration() {
    ensure_init();
    const REPS: u32 = 3;
    println!("\n=== full scan, per edge, {PAIRS} pairs held fixed ===");
    println!(
        "{:>18}  {:>10}  {:>16}  {:>12}  {:>16}  {:>12}",
        "fixture", "edges", "iter_edges instr", "us", "iter(fwd) instr", "us"
    );
    for (label, multi, k) in [
        ("all inline", 0, 1),
        ("half promoted", PAIRS / 2, 2),
        ("all promoted x2", PAIRS, 2),
        ("all promoted x4", PAIRS, 4),
    ] {
        let t = built(multi, k);
        let edges = t.edge_count();
        for _ in 0..REPS {
            let a = measure(edges, || {
                let mut n = 0u64;
                for (_, _, id) in t.iter_edges() {
                    n = n.wrapping_add(id);
                }
                std::hint::black_box(n);
            });
            let b = measure(edges, || {
                let mut n = 0u64;
                for (_, _, id) in t.iter(0, u64::MAX, false) {
                    n = n.wrapping_add(id);
                }
                std::hint::black_box(n);
            });
            println!(
                "{:>18}  {:>10}  {:>16}  {:>12.4}  {:>16}  {:>12.4}",
                label,
                edges,
                a.fmt_instr(),
                a.us,
                b.fmt_instr(),
                b.us
            );
        }
    }
}

// --- space -------------------------------------------------------------------

/// Bytes the auxiliary id matrix costs per id it holds, which is the quantity
/// the always-materialised design's space penalty is made of.
///
/// Three groups:
///
/// 1. **`me` under the real tensor**, pair count held fixed at `PAIRS` and ids
///    per pair varied over 2, 4, 8, 16. Reports `me`'s own `memory_usage` and
///    bytes/id. This is a *measurement* of this design's auxiliary matrix.
/// 2. **A standalone `me`-shaped matrix holding one id per pair.** This is the
///    geometry the always-materialised design would have — `ME_DIM` rows by
///    `ME_NARROW_NCOLS` columns, the shape #2579 gave `me`, one
///    `(compound_key(src,dst).1, edge_id)` entry per pair, so one
///    entry per row, the least amortised regime there is. Constructed directly
///    rather than through a tensor, because no tensor state has one id per `me`
///    row: promotion moves at least two. It is a measurement of the
///    *structure*, and the row it feeds is labelled a model, not a measurement
///    of a design.
/// 3. **Adjacency at both widths**: `UINT64` (what this design needs to hold an
///    inline id) against `BOOL` (what would suffice if every id lived in the
///    auxiliary matrix), same `PAIRS` entries, same dimensions.
///
/// The closing model line adds (2) and (3) to state the always-materialised
/// design's footprint for a simple graph. It is an **extrapolation from
/// measured components**, and it assumes exactly this: that such an
/// implementation would reuse these same GraphBLAS matrices at these same
/// dimensions with these same key encodings, and that its adjacency would be
/// `BOOL`. It does not account for any structure that implementation might add
/// or drop elsewhere.
#[test]
#[ignore = "measurement, not a correctness check"]
fn tensor_cost_space() {
    ensure_init();
    println!("\n=== auxiliary-matrix space, {PAIRS} pairs held fixed ===");
    println!(
        "{:>10}  {:>10}  {:>14}  {:>12}  {:>16}",
        "ids/pair", "ids in me", "me bytes", "bytes/id", "tensor bytes"
    );
    for k in [1u64, 2, 4, 8, 16] {
        let t = built(if k == 1 { 0 } else { PAIRS }, k);
        let me = t.edge_versioned_block_0().nvals();
        let me_bytes = t.edge_versioned_block_0().memory_usage();
        let per = if me == 0 {
            f64::NAN
        } else {
            me_bytes as f64 / me as f64
        };
        println!(
            "{:>10}  {:>10}  {:>14}  {:>12.2}  {:>16}",
            k,
            me,
            me_bytes,
            per,
            t.memory_usage()
        );
    }

    // One id per `me` row: the always-materialised geometry.
    let mut aux = Matrix::<bool>::new(ME_DIM, ME_NARROW_NCOLS);
    let keys: Vec<u64> = (0..PAIRS)
        .map(|i| {
            let (s, d) = pair(i);
            // Every fixture pair is within `BLOCK_SHIFT` bits per axis, so the
            // block is `ME_BLOCK_0` and the row half is the whole key.
            compound_key(s, d).1
        })
        .collect();
    let ids: Vec<u64> = (0..PAIRS).collect();
    aux.build(&keys, &ids);
    aux.wait();
    let aux_vm = VersionedMatrix::from_matrix(aux);
    let aux_bytes = aux_vm.memory_usage();

    // Adjacency at both value widths, same entries and dimensions.
    let coords: Vec<(u64, u64)> = probes(0, PAIRS);
    let srcs: Vec<u64> = coords.iter().map(|c| c.0).collect();
    let dsts: Vec<u64> = coords.iter().map(|c| c.1).collect();
    let mut adj_u64 = Matrix::<u64>::new(DIM, DIM);
    adj_u64.build(&srcs, &dsts, &ids);
    adj_u64.wait();
    let mut adj_bool = Matrix::<bool>::new(DIM, DIM);
    adj_bool.build(&srcs, &dsts);
    adj_bool.wait();

    println!("\n=== components, {PAIRS} entries each ===");
    println!(
        "{:>34}  {:>14}  {:>12}",
        "component", "bytes", "bytes/entry"
    );
    for (label, bytes) in [
        ("aux matrix, 1 id/row (me geometry)", aux_bytes),
        ("adjacency UINT64 (inline id)", adj_u64.memory_usage()),
        ("adjacency BOOL (no inline id)", adj_bool.memory_usage()),
    ] {
        println!(
            "{:>34}  {:>14}  {:>12.2}",
            label,
            bytes,
            bytes as f64 / PAIRS as f64
        );
    }

    // The empty auxiliary matrix this design leaves behind on a simple graph.
    let simple = built(0, 1);
    println!(
        "\nsimple graph (1 edge/pair): tensor {} bytes, of which `me` {} bytes ({} ids)",
        simple.memory_usage(),
        simple.edge_versioned_block_0().memory_usage(),
        simple.edge_versioned_block_0().nvals()
    );
    let model = adj_bool.memory_usage() + aux_bytes;
    let measured_inline = adj_u64.memory_usage() + simple.edge_versioned_block_0().memory_usage();
    println!(
        "MODEL (not a measurement): always-materialised simple graph = BOOL adjacency {} + aux {} = {} bytes; \
         inline-first equivalent = UINT64 adjacency {} + empty me {} = {} bytes; ratio {:.2}x",
        adj_bool.memory_usage(),
        aux_bytes,
        model,
        adj_u64.memory_usage(),
        simple.edge_versioned_block_0().memory_usage(),
        measured_inline,
        model as f64 / measured_inline as f64,
    );
    // Guard the fixture the model rests on: a simple graph really does leave
    // `me` empty, so its whole auxiliary cost is what the model adds back.
    assert_eq!(simple.edge_versioned_block_0().nvals(), 0);
}

/// The entry points the paper listed as unmeasured on the Rust side, plus the
/// cold-cache case the scattered-order test explicitly did *not* answer.
///
/// Degrees have no dedicated method here: `Graph::get_node_outdegree` counts a
/// one-row iteration, and the in-degree the transposed one, so that is what is
/// measured. The C side calls `Tensor_RowDegree` / `Tensor_ColDegree`.
#[test]
#[ignore]
fn tensor_cost_entry_points() {
    ensure_init();
    const REPS: u32 = 3;
    const OPS: u64 = 200_000;

    for k in [1u64, 2] {
        let t = built_diag(k);
        println!("\n=== entry points, {XN} pairs x {k} edge(s) ===");
        println!("{:>38}  {:>12}  {:>10}", "operation", "instr/op", "ns/op");

        for _ in 0..REPS {
            let c = measure(OPS, || {
                let mut n = 0usize;
                for i in 0..OPS {
                    n += t.iter(i % XN, i % XN, false).count();
                }
                std::hint::black_box(n);
            });
            println!(
                "{:>38}  {:>12}  {:>10.1}",
                "row degree (one-row iter)",
                c.fmt_instr(),
                c.us * 1_000.0
            );
        }
        for _ in 0..REPS {
            let c = measure(OPS, || {
                let mut n = 0usize;
                for i in 0..OPS {
                    n += t.iter(i % XN, i % XN, true).count();
                }
                std::hint::black_box(n);
            });
            println!(
                "{:>38}  {:>12}  {:>10.1}",
                "col degree (transposed one-row)",
                c.fmt_instr(),
                c.us * 1_000.0
            );
        }
    }

    // The flat removal path: every pair single-edge, so `remove_all` takes the
    // three-bulk-op fast path rather than the per-edge one.
    println!("\n=== bulk removal, {XN} single-edge pairs ===");
    println!("{:>38}  {:>12}  {:>10}", "operation", "instr/op", "ns/op");
    for _ in 0..REPS {
        let mut t = built_diag(1);
        let rels: Vec<(u64, u64, u64)> = (0..XN).map(|i| (i, i, i)).collect();
        let c = measure(XN, || {
            t.remove_all(&rels);
        });
        println!(
            "{:>38}  {:>12}  {:>10.1}",
            "remove_all, flat path (per pair)",
            c.fmt_instr(),
            c.us * 1_000.0
        );
    }

    // The bulk insert path, against C's `batch insert n new inline pairs`.
    println!("\n=== bulk insert, {XN} new inline pairs ===");
    println!("{:>38}  {:>12}  {:>10}", "operation", "instr/op", "ns/op");
    let srcs: Vec<u64> = (0..XN).collect();
    let ids: Vec<u64> = (0..XN).collect();
    for _ in 0..REPS {
        let mut t = Tensor::new(XN + 1, XN + 1);
        let c = measure(XN, || {
            t.set_all_from_slices(&srcs, &srcs, &ids);
        });
        t.wait();
        assert_eq!(t.fwd_m().nvals() + t.fwd_dp().nvals(), XN);
        println!(
            "{:>38}  {:>12}  {:>10.1}",
            "batch insert, inline (per edge)",
            c.fmt_instr(),
            c.us * 1_000.0
        );
    }
}

/// **Cold cache.** Every read measured elsewhere here is warm: `XN` pairs is a
/// working set small enough to sit in cache, and the scattered-order test
/// changed the probe *order* without changing that. So it found instruction
/// counts almost unmoved and wall clock only mildly up — which is the right
/// answer to the question it asked, and not an answer about residency.
///
/// This asks about residency, by sweeping the working set from far inside the
/// cache to far outside it and probing in a scrambled order at every size. Below
/// the cache the reads hit; above it they miss, and nothing about the code path
/// has changed in between.
///
/// The instruction column is reported and is, by construction, the *wrong*
/// metric here: a cache miss retires no extra instruction. Its flatness is the
/// control — it shows the sweep changes residency and not work. The nanoseconds
/// are the measurement.
///
/// An eviction-sweep design was tried first and abandoned: streaming a buffer
/// large enough to evict costs milliseconds, and subtracting milliseconds to
/// recover a ~100 ns read is not a measurement.
#[test]
#[ignore]
fn tensor_cost_cold_cache() {
    ensure_init();
    const REPS: u32 = 3;
    const PROBES: u64 = 200_000;
    // 10k pairs is inside L2; 8M is far past any last level on this class of
    // machine. The interesting region is somewhere in between and the sweep
    // does not assume where.
    const SIZES: [u64; 6] = [10_000, 100_000, 500_000, 2_000_000, 4_000_000, 8_000_000];

    for k in [1u64, 2] {
        println!("\n=== working-set sweep, {k} edge(s) per pair, scrambled probe order ===");
        println!(
            "{:>12}  {:>10}  {:>12}  {:>10}",
            "pairs", "MB", "instr/op", "ns/op"
        );
        for n in SIZES {
            let t = built_n(n, k);
            let mb = t.memory_usage() as f64 / (1 << 20) as f64;
            // an odd multiplier coprime to `n` walks every row in an order with
            // no useful locality, without needing a stored permutation (which
            // would itself dominate the working set)
            let step = 0x9E37_79B9_7F4A_7C15u64;
            let mut best = f64::MAX;
            let mut instr = None;
            for _ in 0..REPS {
                let c = measure(PROBES, || {
                    let mut acc = 0u64;
                    let mut x = 1u64;
                    for _ in 0..PROBES {
                        x = x.wrapping_mul(step).wrapping_add(1);
                        let p = (x >> 32) % n;
                        // consume every id, matching the C harness's
                        // `TensorIterator_ScanEntry` drain
                        for id in t.get(p, p) {
                            acc = acc.wrapping_add(id);
                        }
                    }
                    std::hint::black_box(acc);
                });
                if c.us < best {
                    best = c.us;
                    instr = c.instr;
                }
            }
            println!(
                "{:>12}  {:>10.1}  {:>12}  {:>10.1}",
                n,
                mb,
                instr.map_or_else(|| "-".to_string(), |i| format!("{i:.1}")),
                best * 1_000.0
            );
        }
    }
}
