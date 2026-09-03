//! What an `IdList` costs to build, write and read, per id shape.
//!
//! The encoding ladder this replaced had four encodings and a forced-encoding
//! harness to compare them on identical input. There is nothing to force any
//! more: a list is a sequence of segments, the segments *are* the encoding, and
//! the only choice left — whether an ascending run collapses into a bitmap — is
//! arithmetic made during `push`. So the axis is the **shape of the ids**, not a
//! menu of encodings.
//!
//! Four shapes, each of which the representation treats differently:
//!
//! | shape       | what produces it                        | segments |
//! |-------------|-----------------------------------------|----------|
//! | consecutive | every bulk create, every delete-by-label | 1        |
//! | gapped      | deletes and filtered matches             | collapses to a bitmap |
//! | shuffled    | a hash-ordered scan                      | one per id |
//! | duplicates  | edge endpoints out of a supernode        | one per id |
//!
//! Where the work lands matters: encoding runs on the write thread holding the
//! GIL, decoding on the replica's main thread. Neither is a background cost.
//!
//! ```bash
//! cargo bench -p graph --bench effects_blocks
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use graph::effects::v3::{IdList, Reader, read_ids};
use roaring::RoaringTreemap;

/// The four shapes at one size.
///
/// `shuffled` and `duplicates` both use a fixed multiplicative hash rather than
/// a cycle: a cycle is perfectly periodic, which flatters anything that spots
/// runs in a way real id order does not.
fn shapes(n: usize) -> Vec<(&'static str, Vec<u64>)> {
    let n64 = n as u64;
    vec![
        ("consecutive", (0..n64).collect()),
        ("gapped", (0..n64).map(|i| i * 3).collect()),
        (
            "shuffled",
            (0..n64)
                .map(|i| i.wrapping_mul(0x9E37_7997_9F4A_7C15) % n64)
                .collect(),
        ),
        (
            // Three sources, as a supernode's edges produce. No two adjacent
            // ids continue each other, so this is one segment per id and the
            // shape with no compact form at all — the cost of dropping the
            // dictionary encoding, priced.
            "duplicates",
            (0..n as usize)
                .map(|i| [4_000_000_000_u64, 7, 2_100_000_000][i % 3])
                .collect(),
        ),
    ]
}

/// Building the list — the axis the representation actually moves, and the only
/// place the collapse decision is made.
///
/// Every other group here starts from a finished `IdList`, so none of them can
/// see what `push` costs or what it allocates. A consecutive run never
/// allocates; a shuffled one allocates a segment per id.
fn construct(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_list/construct");
    for n in [1_000_usize, 100_000, 1_000_000] {
        g.throughput(Throughput::Elements(n as u64));
        for (name, src) in shapes(n) {
            g.bench_with_input(BenchmarkId::new(name, n), &src, |b, src| {
                b.iter(|| {
                    let mut list = IdList::new();
                    for &id in black_box(src) {
                        list.push(id);
                    }
                    black_box(list.len())
                });
            });
        }
    }
    g.finish();
}

/// Writing a finished list. No decision is taken here — the segments are the
/// encoding — so this prices the write itself.
fn encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_list/encode");
    let n = 10_000_usize;
    g.throughput(Throughput::Elements(n as u64));
    for (name, src) in shapes(n) {
        let ids = IdList::from(src.as_slice());
        g.bench_with_input(BenchmarkId::from_parameter(name), &ids, |b, ids| {
            b.iter(|| {
                let mut buf = Vec::new();
                black_box(ids).encode(&mut buf);
                black_box(buf.len())
            });
        });
    }
    g.finish();
}

/// Reading one back, on the replica's main thread.
fn decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_list/decode");
    let n = 10_000_usize;
    g.throughput(Throughput::Elements(n as u64));
    for (name, src) in shapes(n) {
        let mut buf = Vec::new();
        IdList::from(src.as_slice()).encode(&mut buf);
        g.bench_with_input(BenchmarkId::from_parameter(name), &buf, |b, buf| {
            b.iter(|| {
                let mut r = Reader::new(black_box(buf));
                black_box(read_ids(&mut r, n as u32).unwrap().len())
            });
        });
    }
    g.finish();
}

/// The apply path's bitmap, built per segment against per id.
///
/// `apply_record` needs a `RoaringTreemap` to verify which ids it may create.
/// Collecting one id at a time is a `BTreeMap` lookup each; the segments are
/// already the runs roaring wants, so `to_roaring` states each one with a single
/// `insert_range`. This is what that is worth.
fn to_roaring(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_list/to_roaring");
    let n = 100_000_usize;
    g.throughput(Throughput::Elements(n as u64));
    for (name, src) in shapes(n) {
        let ids = IdList::from(src.as_slice());
        g.bench_with_input(BenchmarkId::new("per_segment", name), &ids, |b, ids| {
            b.iter(|| black_box(black_box(ids).to_roaring().len()))
        });
        g.bench_with_input(BenchmarkId::new("per_id", name), &ids, |b, ids| {
            b.iter(|| {
                let m: RoaringTreemap = black_box(ids).iter().collect();
                black_box(m.len())
            });
        });
    }
    g.finish();
}

criterion_group!(benches, construct, encode, decode, to_roaring);
criterion_main!(benches);
