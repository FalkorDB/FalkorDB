//! Encoding, decoding and compressing a v3 payload, across five orders of
//! magnitude.
//!
//! The query modelled throughout is the motivating one:
//!
//! ```cypher
//! UNWIND range(0, N-1) AS i CREATE (:Person {name: 'n' + i, age: i % 80})
//! ```
//!
//! One shape, so this is a single record however many nodes — the case the
//! format was designed for, and its best case rather than a typical one. A
//! query whose shapes all differ lands at one record each, which
//! `the_motivating_query_end_to_end` pins.
//!
//!
//! ```bash
//! cargo bench -p graph --bench effects_versions
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use atomic_refcell::AtomicRefCell;
use rustc_hash::FxHashMap;

use graph::{
    effects::{EffectEncode, v3},
    graph::graph::{Graph, LabelId, NodeId},
    runtime::{orderset::OrderSet, pending::Pending, value::Value},
};

/// Stage one created node the way the runtime's ops do.
///
/// A bench is a separate compilation unit, so it cannot reach the crate's
/// test-only staging trait — and it should not: going through `Pending`'s real
/// API is what makes the staged shape the same one a query produces.
fn stage_created_node(
    p: &mut Pending,
    id: u64,
    labels: &[u64],
    attrs: &[(u16, Value)],
) {
    p.created_nodes(&[NodeId::from(id)]);
    if !labels.is_empty() {
        let set: OrderSet<LabelId> = labels.iter().map(|&l| LabelId(l as usize)).collect();
        p.set_node_labels(NodeId::from(id), &set);
    }
    p.set_node_attributes(NodeId::from(id), attrs.to_vec())
        .expect("bench attrs are valid node properties");
}

/// Node counts spanning the range worth caring about: a single write, a small
/// batch, and up to a bulk load.
const SCALES: [usize; 6] = [1, 100, 1_000, 10_000, 100_000, 1_000_000];

/// Criterion's default 100 samples is far too many once a single iteration
/// encodes a million records.
fn samples_for(n: usize) -> usize {
    match n {
        0..=10_000 => 50,
        0..=100_000 => 20,
        _ => 10,
    }
}

/// `GxB_init` is process-wide and may only run once.
fn ensure_graphblas() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    unsafe extern "C" {
        fn malloc(size: usize) -> *mut std::ffi::c_void;
        fn calloc(
            n: usize,
            size: usize,
        ) -> *mut std::ffi::c_void;
        fn realloc(
            p: *mut std::ffi::c_void,
            size: usize,
        ) -> *mut std::ffi::c_void;
        fn free(p: *mut std::ffi::c_void);
    }
    INIT.call_once(|| {
        graph::graph::graphblas::matrix::init(
            Some(malloc),
            Some(calloc),
            Some(realloc),
            Some(free),
        )
        .expect("GraphBLAS must initialize");
    });
}

fn fixture(n: usize) -> (AtomicRefCell<Graph>, Pending) {
    ensure_graphblas();
    let g = AtomicRefCell::new(Graph::new(64, 64, 0, 0, "bench"));
    {
        let mut graph = g.borrow_mut();
        graph.get_label_id_mut("Person");
        graph.add_node_attribute_name("name");
        graph.add_node_attribute_name("age");
    }
    let mut p = Pending::default();
    p.set_schema_baseline(&g);
    for id in 0..n as u64 {
        stage_created_node(
            &mut p,
            id,
            &[0],
            &[
                (0, Value::String(std::sync::Arc::new(format!("n{id}")))),
                (1, Value::Int((id % 80) as i64)),
            ],
        );
    }
    (g, p)
}

/// Payload size is exact, so it is printed once rather than sampled.
///
/// The bytes-per-node column is what to watch: one shape means one record
/// however many nodes, so it should fall as `n` grows and the record's fixed
/// cost is amortised. For the record, v2 encoded this query at ~26 bytes per
/// node at every scale, because it emitted one record each.
fn report_sizes() {
    println!("\n  nodes         bytes   B/node   zstd-1   zstd B/node");
    for n in SCALES {
        let (g, p) = fixture(n);
        let mut buf = Vec::new();
        v3::emit::build_effects_buffer(&p, &g, &mut buf);
        let mut z = buf.clone();
        v3::maybe_compress(&mut z, 1);
        println!(
            "  {n:>9} {:>13} {:>7.2} {:>8} {:>13.2}",
            buf.len(),
            buf.len() as f64 / n as f64,
            z.len(),
            z.len() as f64 / n as f64,
        );
    }
    println!();
}

fn encode(c: &mut Criterion) {
    report_sizes();
    let mut group = c.benchmark_group("encode");
    for n in SCALES {
        let (g, p) = fixture(n);
        group.sample_size(samples_for(n));
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("v3", n), &n, |b, _| {
            b.iter(|| {
                let mut buf = Vec::new();
                v3::emit::build_effects_buffer(black_box(&p), &g, &mut buf);
                black_box(buf.len())
            });
        });
        // What compression adds on top of the v3 encode it follows.
        group.bench_with_input(BenchmarkId::new("v3+zstd-1", n), &n, |b, _| {
            b.iter(|| {
                let mut buf = Vec::new();
                v3::emit::build_effects_buffer(black_box(&p), &g, &mut buf);
                v3::maybe_compress(&mut buf, 1);
                black_box(buf.len())
            });
        });
    }
    group.finish();
}

fn decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    for n in SCALES {
        let (g, p) = fixture(n);
        let mut plain = Vec::new();
        v3::emit::build_effects_buffer(&p, &g, &mut plain);
        let mut compressed = plain.clone();
        v3::maybe_compress(&mut compressed, 1);

        group.sample_size(samples_for(n));
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("v3", n), &n, |b, _| {
            b.iter(|| black_box(v3::read_buffer(black_box(&plain)).unwrap().len()));
        });
        group.bench_with_input(BenchmarkId::new("v3+zstd-1", n), &n, |b, _| {
            b.iter(|| black_box(v3::read_buffer(black_box(&compressed)).unwrap().len()));
        });
    }
    group.finish();
}

/// Materializing every record before touching the graph, against decoding one
/// at a time.
///
/// Two shapes, because they answer different questions. **One shape** is the
/// motivating query: a single record however many nodes it covers, so streaming
/// can only win the cost of one `Vec`. **Every node its own shape** is the
/// pathological end — N records live at once under the old scheme — and is
/// where holding them all should show.
fn stream_vs_materialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_shape");
    for n in [1_000_usize, 10_000, 100_000] {
        let one = {
            let (g, p) = fixture(n);
            let mut b = Vec::new();
            v3::emit::build_effects_buffer(&p, &g, &mut b);
            b
        };
        // One record per node: distinct attribute ids make distinct shapes.
        let many = {
            ensure_graphblas();
            let g = AtomicRefCell::new(Graph::new(64, 64, 0, 0, "bench"));
            {
                let mut graph = g.borrow_mut();
                graph.get_label_id_mut("Person");
                for i in 0..n {
                    graph.add_node_attribute_name(&format!("p{i}"));
                }
            }
            let mut p = Pending::default();
            p.set_schema_baseline(&g);
            for id in 0..n as u64 {
                stage_created_node(&mut p, id, &[0], &[(id as u16, Value::Int(id as i64))]);
            }
            let mut b = Vec::new();
            v3::emit::build_effects_buffer(&p, &g, &mut b);
            b
        };

        group.sample_size(samples_for(n));
        for (shape, buf) in [("1shape", &one), ("Nshapes", &many)] {
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("materialize/{shape}"), n),
                buf,
                |b, buf| {
                    b.iter(|| black_box(v3::read_buffer(black_box(buf)).unwrap().len()));
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("stream/{shape}"), n),
                buf,
                |b, buf| {
                    b.iter(|| {
                        let payload = v3::open_payload(black_box(buf)).unwrap();
                        let mut count = 0_usize;
                        for rec in payload.records() {
                            black_box(rec.unwrap());
                            count += 1;
                        }
                        black_box(count)
                    });
                },
            );
        }
    }
    group.finish();
}

/// The emit side of the same question: streaming records into the buffer
/// against collecting them first. Same two payload shapes as the decode case.
fn encode_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_shape");
    for n in [1_000_usize, 10_000, 100_000] {
        let one = fixture(n);
        let many = {
            ensure_graphblas();
            let g = AtomicRefCell::new(Graph::new(64, 64, 0, 0, "bench"));
            {
                let mut graph = g.borrow_mut();
                graph.get_label_id_mut("Person");
                for i in 0..n {
                    graph.add_node_attribute_name(&format!("p{i}"));
                }
            }
            let mut p = Pending::default();
            p.set_schema_baseline(&g);
            for id in 0..n as u64 {
                stage_created_node(&mut p, id, &[0], &[(id as u16, Value::Int(id as i64))]);
            }
            (g, p)
        };

        group.sample_size(samples_for(n));
        for (shape, (g, p)) in [("1shape", &one), ("Nshapes", &many)] {
            group.throughput(Throughput::Elements(n as u64));
            // What the code did before: build every record, then encode them.
            group.bench_with_input(
                BenchmarkId::new(format!("collect/{shape}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        let mut buf = v3::new_buffer();
                        for record in v3::emit::digest(black_box(p), g) {
                            record.encode(&mut buf);
                        }
                        black_box(buf.len())
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("stream/{shape}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        let mut buf = Vec::new();
                        v3::emit::build_effects_buffer(black_box(p), g, &mut buf);
                        black_box(buf.len())
                    });
                },
            );
        }
    }
    group.finish();
}

/// Where the v3 writer's time goes, at 100,000 nodes of one shape.
///
/// Measured by isolating the two stages that are reachable through the public
/// codec — the id block and the attribute block — and taking the rest of the
/// whole-encode time as the grouping and row-gathering that produced them. The
/// remainder is arrived at by subtraction and labelled as such.
fn writer_breakdown(c: &mut Criterion) {
    const N: usize = 100_000;
    let (g, p) = fixture(N);
    let ids: Vec<u64> = (0..N as u64).collect();
    // The rows a single-shape CREATE of this query produces: a string and an int
    // per node, row-major.
    let rows: Vec<Value> = (0..N as u64)
        .flat_map(|i| {
            [
                Value::String(std::sync::Arc::new(format!("n{i}"))),
                Value::Int((i % 80) as i64),
            ]
        })
        .collect();

    let mut group = c.benchmark_group("writer_breakdown");
    group.sample_size(20);

    group.bench_function("whole_encode", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            v3::emit::build_effects_buffer(black_box(&p), &g, &mut buf);
            black_box(buf.len())
        });
    });
    // The roaring bitmap: build, run-optimize, size, serialize.
    group.bench_function("id_block", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            v3::IdList::from(ids.as_slice()).encode(&mut buf);
            black_box(buf.len())
        });
    });
    // Every SIValue written out.
    group.bench_function("attr_block", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            v3::write_attr_ids(&mut buf, &[0, 1]);
            v3::write_attr_values(&mut buf, 2, black_box(&rows));
            black_box(buf.len())
        });
    });
    // Cloning the values out of Pending into row-major order is part of the
    // remainder; this is the clone alone, without the encoding.
    group.bench_function("row_clone_only", |b| {
        b.iter(|| {
            let cloned: Vec<Value> = black_box(&rows).clone();
            black_box(cloned.len())
        });
    });
    // The remainder above is the grouping plus row-gathering. This is the
    // grouping's shape-key loop on its own, reproduced from
    // `digest_created_nodes`, to confirm it is where the remainder goes rather
    // than inferring it: a Vec<i32> and a Vec<u16> are allocated per node
    // purely to serve as the hash key.
    group.bench_function("shape_key_loop", |b| {
        let labels_src: Vec<u64> = vec![0];
        let attrs_src: Vec<(u16, Value)> = vec![(0, Value::Int(0)), (1, Value::Int(0))];
        b.iter(|| {
            let mut groups: FxHashMap<(Vec<i32>, Vec<u16>), Vec<u64>> = FxHashMap::default();
            for id in 0..N as u64 {
                let mut labels: Vec<i32> =
                    black_box(&labels_src).iter().map(|&v| v as i32).collect();
                labels.sort_unstable();
                labels.dedup();
                let attr_ids: Vec<u16> = black_box(&attrs_src).iter().map(|(a, _)| *a).collect();
                groups.entry((labels, attr_ids)).or_default().push(id);
            }
            black_box(groups.len())
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    encode,
    decode,
    stream_vs_materialize,
    encode_shape,
    writer_breakdown
);
criterion_main!(benches);
