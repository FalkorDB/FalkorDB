//! What the `IdList` dictionary costs in CPU, against the plain form.
//!
//! Size already favours the dictionary by 2x-8x uncompressed, and the encoder
//! picks by size. The open question this answers is the other half: whether
//! building the dictionary (a rank per row) and resolving it (a lookup per row)
//! costs more than the bytes it saves are worth.
//!
//! It matters where the work lands. Encoding runs on the write thread while it
//! holds the GIL, and decoding runs on the replica's main thread — neither is a
//! background cost.
//!
//! Both paths are measured on **identical input** via `write_id_list_forced`, so
//! the comparison is the encoding rather than the data. `endpoints(rows,
//! distinct)` models edge endpoints: `distinct` sources drawn pseudo-randomly,
//! which is the shape a supernode produces.
//!
//! ```bash
//! cargo bench -p graph --bench effects_blocks
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use graph::effects::v3::{BlockEncoding, IdList, Reader, read_ids, write_forced};

/// `rows` endpoints drawn from `distinct` sources inside a graph of `nodes`
/// nodes.
///
/// The draw is a fixed multiplicative hash rather than a cycle: cycling makes
/// the rank array perfectly periodic, which flatters the dictionary in a way
/// real endpoint order does not. `nodes` matters because FalkorDB allocates ids
/// densely from zero, so graph size sets the largest id — and that is what
/// decides whether narrowing alone already captures the saving.
fn endpoints(
    rows: usize,
    distinct: usize,
    nodes: u64,
) -> Vec<u64> {
    (0..rows)
        .map(|i| {
            let h = (i as u64).wrapping_mul(0x9E37_7997_9F4A_7C15);
            (h % distinct as u64) * (nodes / distinct as u64).max(1)
        })
        .collect()
}

const ENCODINGS: [(&str, BlockEncoding); 2] = [
    ("raw", BlockEncoding::Plain),
    ("dictionary", BlockEncoding::Compressed),
];

/// The shapes the selection table shows a real winner for.
const SHAPES: [(u64, usize); 4] = [
    (10_000, 100),
    (10_000, 10_000),
    (5_000_000, 100),
    (5_000_000, 10_000),
];

/// Building the list, which is the axis the representation actually moves.
///
/// Every other group here starts from a finished `IdList`, so none of them can
/// see what `push` costs or what it allocates — and that is where a range wins
/// and where a bitmap rung would have to pay for itself.
///
/// The three shapes are the ones the ladder distinguishes: consecutive is what
/// every allocator hands out, gapped is what deletes and filtered matches leave
/// behind, and shuffled is the case no bitmap can hold.
fn construct(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_list/construct");
    for n in [1_000_usize, 100_000, 1_000_000] {
        let consecutive: Vec<u64> = (0..n as u64).collect();
        let gapped: Vec<u64> = (0..n as u64).map(|i| i * 3).collect();
        let shuffled: Vec<u64> = (0..n as u64)
            .map(|i| i.wrapping_mul(0x9E37_7997_9F4A_7C15) % n as u64)
            .collect();

        g.throughput(Throughput::Elements(n as u64));
        for (name, src) in [
            ("consecutive", &consecutive),
            ("gapped", &gapped),
            ("shuffled", &shuffled),
        ] {
            g.bench_with_input(BenchmarkId::new(name, n), src, |b, src| {
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

fn encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_list/encode");
    for (nodes, distinct) in SHAPES {
        let ids = IdList::from(endpoints(10_000, distinct, nodes).as_slice());
        let label = format!("{nodes}nodes/{distinct}distinct");
        g.throughput(Throughput::Elements(10_000));
        for (name, enc) in ENCODINGS {
            g.bench_with_input(BenchmarkId::new(name, &label), &ids, |b, ids| {
                b.iter(|| {
                    let mut buf = Vec::new();
                    write_forced(&mut buf, black_box(ids), enc);
                    black_box(buf.len())
                });
            });
        }
    }
    g.finish();
}

fn decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_list/decode");
    for (nodes, distinct) in SHAPES {
        let ids = IdList::from(endpoints(10_000, distinct, nodes).as_slice());
        let label = format!("{nodes}nodes/{distinct}distinct");
        g.throughput(Throughput::Elements(10_000));
        for (name, enc) in ENCODINGS {
            let mut buf = Vec::new();
            write_forced(&mut buf, &ids, enc);
            g.bench_with_input(BenchmarkId::new(name, &label), &buf, |b, buf| {
                b.iter(|| {
                    let mut r = Reader::new(black_box(buf));
                    black_box(read_ids(&mut r, 10_000).unwrap().len())
                });
            });
        }
    }
    g.finish();
}

/// `IdSet` on the shape `CREATE_NODE` actually produces: consecutive ids, which
/// roaring collapses to a single run. This is a different regime from `IdList`
/// — the saving is three orders of magnitude, not a percentage — so it is worth
/// pricing separately rather than assuming the IdList conclusion carries.
fn id_range(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_range");
    let ids: IdList = (0..10_000).collect();
    g.throughput(Throughput::Elements(10_000));

    g.bench_function("encode/consecutive", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            black_box(&ids).encode(&mut buf);
            black_box(buf.len())
        });
    });

    let mut buf = Vec::new();
    ids.encode(&mut buf);
    let bytes = buf.len();
    g.bench_function("decode/consecutive", |b| {
        b.iter(|| {
            let mut r = Reader::new(black_box(&buf));
            black_box(read_ids(&mut r, 10_000).unwrap().len())
        });
    });
    println!("    [id_range: 10,000 consecutive ids -> {bytes} B, vs 20,002 plain]");
    g.finish();
}

/// What the shipped rule costs, as opposed to each encoding in isolation.
/// This is the number that matters: it includes the duplication probe, and the
/// dictionary build only where the probe lets it happen.
fn selection(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_list/selection");
    for (nodes, distinct) in SHAPES {
        let ids = IdList::from(endpoints(10_000, distinct, nodes).as_slice());
        g.throughput(Throughput::Elements(10_000));
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("{nodes}nodes/{distinct}distinct")),
            &ids,
            |b, ids| {
                b.iter(|| {
                    let mut buf = Vec::new();
                    black_box(ids).encode(&mut buf);
                    black_box(buf.len())
                });
            },
        );
    }
    g.finish();
}

/// Each encoding against the alternatives that were legal on the same input.
///
/// `write_id_list_forced` is what makes this a comparison of encodings rather
/// than of data — every row below encodes the identical id list. Only legal
/// combinations are measured: `Range` is defined solely on a consecutive run
/// and `Sorted` only on a strictly ascending one, so forcing either elsewhere
/// would measure a buffer that cannot be produced.
fn ladder(c: &mut Criterion) {
    /// A shape, and the encodings that are legal on it.
    type Case = (&'static str, IdList, Vec<(&'static str, BlockEncoding)>);

    let cases: Vec<Case> = vec![
        (
            "consecutive/10k",
            (0..10_000).collect::<IdList>(),
            vec![
                ("range", BlockEncoding::Range),
                ("sorted", BlockEncoding::Sorted),
                ("plain", BlockEncoding::Plain),
            ],
        ),
        (
            "consecutive/1M",
            (0..1_000_000).collect::<IdList>(),
            vec![
                ("range", BlockEncoding::Range),
                ("sorted", BlockEncoding::Sorted),
                ("plain", BlockEncoding::Plain),
            ],
        ),
        (
            "gapped/10k",
            (0..10_000).map(|i| i * 2).collect::<IdList>(),
            vec![
                ("sorted", BlockEncoding::Sorted),
                ("plain", BlockEncoding::Plain),
            ],
        ),
        (
            "supernode/10k",
            IdList::from(endpoints(10_000, 100, 5_000_000).as_slice()),
            vec![
                ("dictionary", BlockEncoding::Compressed),
                ("plain", BlockEncoding::Plain),
            ],
        ),
    ];

    println!("\n    bytes per encoding");
    for (name, ids, encs) in &cases {
        let mut line = format!("      {name:<18}");
        for (label, enc) in encs {
            let mut buf = Vec::new();
            write_forced(&mut buf, ids, *enc);
            line.push_str(&format!(" {label}={} B", buf.len()));
        }
        let mut chosen = Vec::new();
        ids.encode(&mut chosen);
        line.push_str(&format!("  -> chose enc {}", chosen[0]));
        println!("{line}");
    }
    println!();

    for (name, ids, encs) in &cases {
        let mut g = c.benchmark_group(format!("ladder/{name}"));
        g.throughput(Throughput::Elements(ids.len() as u64));
        for (label, enc) in encs {
            g.bench_with_input(BenchmarkId::new("encode", label), ids, |b, ids| {
                b.iter(|| {
                    let mut buf = Vec::new();
                    write_forced(&mut buf, black_box(ids), *enc);
                    black_box(buf.len())
                });
            });
            let mut buf = Vec::new();
            write_forced(&mut buf, ids, *enc);
            let n = ids.len() as u32;
            g.bench_with_input(BenchmarkId::new("decode", label), &buf, |b, buf| {
                b.iter(|| {
                    let mut r = Reader::new(black_box(buf));
                    black_box(read_ids(&mut r, n).unwrap().len())
                });
            });
        }
        g.finish();
    }
}

criterion_group!(
    benches, construct, encode, decode, id_range, selection, ladder
);
criterion_main!(benches);
