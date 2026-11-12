//! Correctness of [`CchInput::build`] against an oracle that shares no code
//! with it.
//!
//! The oracle walks `Tensor::iter_edges` — the tensor's own iterator path,
//! which resolves the `m`/`dp`/`dm` delta layers and the `me` multi-edge rows
//! by iteration rather than by matrix algebra — and folds the weights with
//! plain Rust `f64::min`. So a bug in the GraphBLAS construction (a wrong
//! descriptor, a missed delta layer, a sentinel left in) shows up as a
//! disagreement rather than as two copies of the same mistake.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::thread;

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;

use super::weight_ops::DEFAULT_WEIGHT;
use crate::graph::graph::{Graph, TypeId};
use crate::graph::graphblas::test_init;
use crate::index::falkordb::cch::{Cch, CchError, CchIndexDef, CchInput};
use crate::index::index_docs::IndexDocs;
use crate::runtime::value::Value;

/// `(src, dst)` -> cheapest weight, and the undirected skeleton as a set of
/// ordered pairs. The two things `CchInput` carries, in a form a test can
/// compare.
type Decoded = (BTreeMap<(u64, u64), f64>, BTreeSet<(u64, u64)>);

/// A graph under construction, plus the bookkeeping the tests need to talk
/// about it.
struct Builder {
    g: Graph,
    weight_attr: u16,
}

impl Builder {
    fn new(
        name: &str,
        nodes: u64,
    ) -> Self {
        test_init::ensure_init();
        let mut g = Graph::new(nodes.max(1), 16, 1, 0, name);
        // A fresh graph's id spaces open their first batch themselves.
        let ids: RoaringTreemap = g
            .node_id_space()
            .reserve(nodes as usize, &RoaringTreemap::new())
            .expect("reserved")
            .into_iter()
            .collect();
        assert_eq!(ids, (0..nodes).collect(), "a fresh graph hands out 0..n");
        g.create_nodes(&ids).expect("creatable");
        let weight_attr = g.get_or_create_rel_attr_id(&Arc::new(String::from("w")));
        Self { g, weight_attr }
    }

    /// Add edges of one relationship type. `edges` is `(src, dst, weight)`,
    /// with `None` meaning "no weight attribute on this edge".
    fn add(
        &mut self,
        rel_type: &str,
        edges: &[(u64, u64, Option<Value>)],
    ) -> Vec<u64> {
        let name = Arc::new(String::from(rel_type));
        // Each call is its own batch, as each query is; rolling also verifies
        // the batch before it left a possible id space.
        self.g.roll_id_batches().expect("a consistent space");
        let ids: Vec<u64> = self
            .g
            .relationship_id_space()
            .reserve(edges.len(), &RoaringTreemap::new())
            .expect("reserved");
        let srcs: Vec<u64> = edges.iter().map(|e| e.0).collect();
        let dsts: Vec<u64> = edges.iter().map(|e| e.1).collect();
        self.g
            .create_relationships_bulk(&name, &srcs, &dsts, &ids, &mut IndexDocs::default())
            .expect("creatable");

        let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        for (&id, (_, _, w)) in ids.iter().zip(edges) {
            if let Some(w) = w {
                attrs.insert(id, vec![(self.weight_attr, w.clone())]);
            }
        }
        if !attrs.is_empty() {
            let mut docs = IndexDocs::default();
            self.g
                .set_relationships_attributes(&attrs, &mut docs)
                .expect("set weights");
        }
        ids
    }

    fn def(
        &self,
        rel_types: &[&str],
    ) -> CchIndexDef {
        let ids: Vec<TypeId> = rel_types
            .iter()
            .map(|t| self.g.get_type_id(t).expect("relationship type exists"))
            .collect();
        CchIndexDef::new(ids, self.weight_attr)
    }

    fn build(
        &self,
        rel_types: &[&str],
    ) -> Result<CchInput, CchError> {
        CchInput::build(&self.g, &self.def(rel_types))
    }
}

/// Read a built [`CchInput`] back into comparable sets.
///
/// Also checks the shape both CSRs promise: the last row offset is the entry
/// count, and every row's column ids strictly ascend.
fn decode(input: &CchInput) -> Decoded {
    for (name, csr) in [
        ("weights", &input.weights.csr),
        ("skeleton", &input.skeleton),
    ] {
        assert_eq!(
            csr.row_offsets[csr.node_count],
            csr.neighbors.len() as u64,
            "{name}: the last row offset is not the entry count"
        );
        for u in 0..csr.node_count {
            let row = csr.row(u);
            assert!(
                row.windows(2).all(|p| p[0] < p[1]),
                "{name} row {u} is not strictly ascending: {row:?}"
            );
        }
    }
    let mut weights = BTreeMap::new();
    for u in 0..input.weights.csr.node_count {
        for k in input.weights.csr.row_offsets[u]..input.weights.csr.row_offsets[u + 1] {
            let k = k as usize;
            #[allow(clippy::cast_possible_truncation)]
            let prev = weights.insert(
                (u as u64, input.weights.csr.neighbors[k]),
                input.weights.values[k],
            );
            assert!(prev.is_none(), "duplicate weight entry in row {u}");
        }
    }
    let mut skeleton = BTreeSet::new();
    for u in 0..input.skeleton.node_count {
        for &v in input.skeleton.row(u) {
            #[allow(clippy::cast_possible_truncation)]
            skeleton.insert((u as u64, v));
        }
    }
    (weights, skeleton)
}

/// The oracle: the same definition, computed by walking the tensors.
fn oracle(
    g: &Graph,
    def: &CchIndexDef,
) -> Decoded {
    let attrs = g.relationship_attrs();
    let mut weights: BTreeMap<(u64, u64), f64> = BTreeMap::new();
    for t in def.rel_types() {
        for (src, dst, edge) in g.relationship_tensors()[t.0].iter_edges() {
            if src == dst {
                continue; // a self-loop is never on a shortest path
            }
            let w = match attrs.get_attr_by_idx(edge, def.weight_attr) {
                Some(Value::Float(f)) => f,
                #[allow(clippy::cast_precision_loss)]
                Some(Value::Int(k)) => k as f64,
                _ => DEFAULT_WEIGHT,
            };
            #[allow(clippy::cast_possible_truncation)]
            weights
                .entry((src, dst))
                .and_modify(|cur| *cur = cur.min(w))
                .or_insert(w);
        }
    }
    let mut skeleton = BTreeSet::new();
    for &(u, v) in weights.keys() {
        skeleton.insert((u, v));
        skeleton.insert((v, u));
    }
    (weights, skeleton)
}

fn assert_matches_oracle(
    b: &Builder,
    rel_types: &[&str],
) -> CchInput {
    let input = b.build(rel_types).expect("build succeeds");
    let (got_w, got_s) = decode(&input);
    let (want_w, want_s) = oracle(&b.g, &b.def(rel_types));
    assert_eq!(got_w, want_w, "weights disagree with the oracle");
    assert_eq!(got_s, want_s, "skeleton disagrees with the oracle");

    // The invariants Phase 1 and Phase 2 are allowed to rely on.
    assert_eq!(input.n, b.g.node_cap() as usize);
    assert_eq!(input.skeleton.node_count, input.n);
    assert_eq!(input.weights.csr.node_count, input.n);
    assert_eq!(
        input.weights.csr.neighbors.len(),
        input.weights.values.len()
    );
    for &(u, v) in &got_s {
        assert_ne!(u, v, "skeleton has a self-loop");
        assert!(
            got_s.contains(&(v, u)),
            "skeleton is not symmetric at {u},{v}"
        );
    }
    for (&(u, v), &w) in &got_w {
        assert_ne!(u, v, "weights have a self-loop");
        assert!(
            got_s.contains(&(u, v)),
            "weighted arc {u}->{v} has no skeleton edge"
        );
        assert!(w >= 0.0, "weight {w} is not >= 0");
    }
    input
}

fn weight(value: f64) -> Option<Value> {
    Some(Value::Float(value))
}

#[test]
fn missing_or_non_numeric_weights_default_to_one() {
    let mut b = Builder::new("cch_in_default", 6);
    b.add(
        "R",
        &[
            (0, 1, None),
            (1, 2, Some(Value::String(Arc::new(String::from("heavy"))))),
            (2, 3, Some(Value::Bool(true))),
            (3, 4, Some(Value::Int(6))),
            // A pair whose cheapest edge is the one with no weight at all.
            (4, 5, weight(9.0)),
            (4, 5, None),
        ],
    );
    let input = assert_matches_oracle(&b, &["R"]);
    let (w, _) = decode(&input);
    // Literal 1.0, not `DEFAULT_WEIGHT`: the oracle uses the constant too, so
    // asserting against it would let a changed default pass unnoticed.
    assert_eq!(w[&(0, 1)], 1.0);
    assert_eq!(w[&(1, 2)], 1.0, "a string weight is not numeric");
    assert_eq!(w[&(2, 3)], 1.0, "a bool weight is not numeric");
    assert_eq!(w[&(3, 4)], 6.0, "an integer weight is used as-is");
    assert_eq!(
        w[&(4, 5)],
        1.0,
        "the unweighted parallel edge is the cheaper"
    );
}

#[test]
fn a_graph_with_no_edges_builds_an_empty_input() {
    let mut b = Builder::new("cch_in_empty", 8);
    // The type has to exist for the definition to resolve, so make one and
    // give it nothing.
    b.add("R", &[]);
    let input = b.build(&["R"]).expect("an edgeless graph is not an error");
    assert_eq!(input.n, b.g.node_cap() as usize);
    assert!(input.skeleton.neighbors.is_empty());
    assert!(input.weights.csr.neighbors.is_empty() && input.weights.values.is_empty());
    assert_eq!(input.skeleton.row_offsets.len(), input.n + 1);
    assert!(input.skeleton.row_offsets.iter().all(|&o| o == 0));

    // And a type whose every edge (a multi-edge pair) is deleted: the deltas
    // are not empty, the effective graph is.
    let ids = b.add("S", &[(0, 1, weight(1.0)), (0, 1, weight(2.0))]);
    b.g.flush_for_bulk();
    let mut docs = IndexDocs::default();
    b.g.delete_relationships(&ids.iter().copied().collect(), &mut docs)
        .expect("delete succeeds");
    let input = b.build(&["S"]).expect("an edgeless graph is not an error");
    assert!(input.skeleton.neighbors.is_empty() && input.weights.values.is_empty());
}

/// A bad weight (negative, NaN, `-inf`, a negative integer) is refused wherever it
/// sits, not only when it is the cheapest edge of its pair: `GrB_MIN_FP64` drops NaN
/// (it behaves like `fmin`), so the per-pair minimum alone would hide it. The places:
/// a single edge; a parallel edge (`me` path); a second relationship type (cross-type
/// merge); a single self-loop; and the second edge of a multi-edge self-loop, which
/// only the multi-edge pass scores. A self-loop is dropped from the output, but its
/// weight is checked per edge, where it is scored: pinned so the choice is deliberate.
#[test]
fn a_bad_weight_is_refused_wherever_it_sits() {
    type Place = fn(&mut Builder, Option<Value>) -> &'static [&'static str];
    let places: [(&str, Place); 5] = [
        ("single", |b, bad| {
            b.add("R", &[(0, 1, weight(1.0)), (1, 2, bad)]);
            &["R"]
        }),
        ("parallel", |b, bad| {
            b.add("R", &[(0, 1, weight(2.0)), (0, 1, bad)]);
            &["R"]
        }),
        ("second_type", |b, bad| {
            b.add("R", &[(0, 1, weight(2.0))]);
            b.add("S", &[(0, 1, bad)]);
            &["R", "S"]
        }),
        ("self_loop", |b, bad| {
            b.add("R", &[(0, 1, weight(1.0)), (2, 2, bad)]);
            &["R"]
        }),
        ("multi_self_loop", |b, bad| {
            b.add(
                "R",
                &[(0, 1, weight(1.0)), (2, 2, weight(1.0)), (2, 2, bad)],
            );
            &["R"]
        }),
    ];
    let values = [
        ("neg", weight(-1.0)),
        ("nan", weight(f64::NAN)),
        ("neginf", weight(f64::NEG_INFINITY)),
        ("negint", Some(Value::Int(-3))),
    ];
    for (place, add) in places {
        for (value, bad) in &values {
            let mut b = Builder::new(&format!("cch_in_bad_{place}_{value}"), 4);
            let rel_types = add(&mut b, bad.clone());
            assert_eq!(
                b.build(rel_types).err(),
                Some(CchError::InvalidWeight),
                "{value} on {place}"
            );
        }
    }
}

/// The check on the merged weights, on its own: `-inf` is how the operators
/// report a bad weight, and NaN left on a pair means a multi-edge pair the
/// multi-edge pass never scored, which no graph built through the tensor API
/// can produce, so it is checked here directly. `+inf` (a closed edge) and a
/// sum that overflows are accepted.
#[test]
fn the_weight_check_reads_both_ends() {
    use super::check_weights;
    use crate::graph::graphblas::matrix::Matrix;
    test_init::ensure_init();
    let mut w = Matrix::<f64>::new(4, 4);
    assert_eq!(check_weights(&w), Ok(()), "empty");
    w.set(0, 1, 0.0);
    w.set(1, 2, f64::MAX);
    w.set(1, 3, f64::MAX);
    assert_eq!(check_weights(&w), Ok(()), "both ends of the finite range");
    w.set(2, 1, f64::INFINITY);
    assert_eq!(check_weights(&w), Ok(()), "a closed edge");
    w.set(2, 3, f64::NAN);
    assert!(matches!(check_weights(&w), Err(CchError::Internal(_))));
    w.set(3, 0, f64::NEG_INFINITY);
    assert_eq!(check_weights(&w), Err(CchError::InvalidWeight));
}

/// `+inf` is a closed edge: a weight like any other, kept as is, on a single
/// edge, on every edge of a multi-edge pair (scored by the multi-edge pass, so
/// a pair left at `+inf` is not the unscored-pair bug), across two types, and on
/// a self-loop (dropped as any self-loop is). A finite parallel edge or a finite
/// edge of a second type wins its pair. `-0.0` is `>= 0`, so it is legal too.
#[test]
fn a_plus_inf_weight_is_a_closed_edge_and_is_kept() {
    let inf = f64::INFINITY;
    let mut b = Builder::new("cch_in_closed", 8);
    b.add(
        "R",
        &[
            (0, 1, weight(inf)), // single, closed
            (1, 2, weight(inf)), // multi, every edge closed
            (1, 2, weight(inf)),
            (2, 3, weight(inf)), // multi, a finite edge wins
            (2, 3, weight(4.0)),
            (3, 4, weight(inf)), // closed here, finite on S
            (5, 5, weight(inf)), // closed self-loop
            (6, 6, weight(inf)), // closed multi-edge self-loop
            (6, 6, weight(inf)),
            (6, 7, weight(-0.0)), // -0.0 weighs 0
        ],
    );
    b.add("S", &[(3, 4, weight(2.5)), (4, 5, weight(inf))]);
    let input = assert_matches_oracle(&b, &["R", "S"]);
    let (w, _) = decode(&input);
    assert_eq!(w[&(0, 1)], inf);
    assert_eq!(w[&(1, 2)], inf);
    assert_eq!(w[&(2, 3)], 4.0);
    assert_eq!(w[&(3, 4)], 2.5);
    assert_eq!(w[&(4, 5)], inf);
    assert_eq!(w[&(6, 7)], 0.0);
    assert!(!w.contains_key(&(5, 5)) && !w.contains_key(&(6, 6)));

    // the index over it: closed edges are never on a path
    let cch = Cch::build(&input).expect("+inf is legal");
    assert_eq!(cch.query(0, 1), None);
    assert_eq!(cch.query(1, 3), None);
    assert_eq!(cch.query(2, 4), Some((6.5, vec![2, 3, 4])));
}

/// Builds on several threads at once, sharing the process-wide operators,
/// agree with the oracle.
#[test]
fn concurrent_builds_match_the_oracle() {
    let graphs: Vec<Builder> = (0..4)
        .map(|k| {
            let mut b = Builder::new(&format!("cch_in_shared_ops_{k}"), 8);
            #[allow(clippy::cast_precision_loss)]
            b.add(
                "R",
                &[
                    (0, 1, weight(k as f64)),
                    (0, 1, weight(9.0)),
                    (1, 2, weight(2.0)),
                    (3, 3, weight(1.0)),
                ],
            );
            b
        })
        .collect();
    thread::scope(|s| {
        for b in &graphs {
            s.spawn(move || {
                let input = assert_matches_oracle(b, &["R"]);
                assert_eq!(decode(&input).0.len(), 2);
            });
        }
    });
}

/// The definition's bounds: the first relationship type id past the graph's types
/// is refused, and so is the first attribute id past its property keys; the last
/// attribute id is accepted (an edge without it weighs the default).
#[test]
fn unknown_definitions_are_refused_at_their_bounds() {
    let mut b = Builder::new("cch_in_bounds", 4);
    b.add("R", &[(0, 1, weight(2.0))]);
    let real = b.g.get_type_id("R").unwrap();
    let type_count = b.g.get_types().len();
    assert_eq!(
        CchInput::build(
            &b.g,
            &CchIndexDef::new(vec![TypeId(type_count)], b.weight_attr)
        )
        .err(),
        Some(CchError::UnknownRelType),
        "the type id one past the last"
    );
    let count = u16::try_from(b.g.property_key_count()).expect("few property keys");
    assert_eq!(
        CchInput::build(&b.g, &CchIndexDef::new(vec![real], count)).err(),
        Some(CchError::UnknownAttribute),
        "the attribute id one past the last"
    );
    assert!(
        CchInput::build(&b.g, &CchIndexDef::new(vec![real], count - 1)).is_ok(),
        "the last attribute id"
    );
}

/// The broad one: several relationship types, parallel edges, self-loops,
/// missing and non-numeric weights, deletions, and uncommitted deltas, all at
/// once. A generator seeded per case, so a failure is replayable from the
/// seed printed in the assertion.
#[test]
fn random_graphs_match_the_oracle() {
    for seed in 1u64..=12 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut next = move || rng.random::<u64>();
        let nodes = 8 + (next() % 24);
        let mut b = Builder::new(&format!("cch_in_rand_{seed}"), nodes);
        let types = ["A", "B", "C"];
        let mut all_ids = Vec::new();
        for t in types {
            let count = (next() % 40) as usize;
            let edges: Vec<(u64, u64, Option<Value>)> = (0..count)
                .map(|_| {
                    let src = next() % nodes;
                    let dst = next() % nodes;
                    let w = match next() % 5 {
                        0 => None,
                        1 => Some(Value::String(Arc::new(String::from("x")))),
                        #[allow(clippy::cast_precision_loss)]
                        2 => Some(Value::Int((next() % 20) as i64)),
                        _ => {
                            #[allow(clippy::cast_precision_loss)]
                            let v = (next() % 1000) as f64 / 10.0;
                            Some(Value::Float(v))
                        }
                    };
                    (src, dst, w)
                })
                .collect();
            all_ids.extend(b.add(t, &edges));
        }
        b.g.flush_for_bulk();

        // Delete roughly a tenth of the edges, leaving the rest in `dm`.
        let mut del = RoaringTreemap::new();
        for &id in &all_ids {
            if next() % 10 == 0 {
                del.insert(id);
            }
        }
        if !del.is_empty() {
            let mut docs = IndexDocs::default();
            b.g.delete_relationships(&del, &mut docs).expect("delete");
        }
        // And add a few more, uncommitted, so `dp` is populated too.
        let extra: Vec<(u64, u64, Option<Value>)> = (0..(next() % 8))
            .map(|_| {
                #[allow(clippy::cast_precision_loss)]
                let w = (next() % 100) as f64;
                (next() % nodes, next() % nodes, Some(Value::Float(w)))
            })
            .collect();
        b.add("A", &extra);

        for subset in [&["A"][..], &["A", "B"][..], &["A", "B", "C"][..]] {
            let input = b.build(subset).unwrap_or_else(|e| {
                panic!("seed {seed}, types {subset:?}: build failed with {e:?}")
            });
            let (got_w, got_s) = decode(&input);
            let (want_w, want_s) = oracle(&b.g, &b.def(subset));
            assert_eq!(got_w, want_w, "seed {seed}, types {subset:?}: weights");
            assert_eq!(got_s, want_s, "seed {seed}, types {subset:?}: skeleton");
        }
    }
}

/// Exercise the committed base and the `dm` deletion mask.
///
/// A delta never folds into `m` below `MIN_FOLD_DELTA` (256) entries
/// (`versioned_matrix.rs`), so every small test above leaves the base empty
/// and every deletion takes the `dp.remove` branch. Nothing then reads `dm`, and the
/// complemented `dm` mask of the effective matrices goes untested. This test
/// is big enough to cross that threshold, and asserts it did.
#[test]
fn deletions_against_the_committed_base_use_the_dm_mask() {
    const EDGES: u64 = 10_050;
    // Enough nodes that every (src, dst) is distinct, so every edge is an
    // inline forward entry. Sized well past the fold floor; the assertions
    // below are what prove the base and `dm` were reached.
    let nodes = 20_011;
    let mut b = Builder::new("cch_in_dm", nodes);
    let edges: Vec<(u64, u64, Option<Value>)> = (0..EDGES)
        .map(|k| {
            let src = (k * 7) % nodes;
            let dst = (k * 13 + 1) % nodes;
            #[allow(clippy::cast_precision_loss)]
            (src, dst, Some(Value::Float((k % 97) as f64 + 1.0)))
        })
        .collect();
    let mut ids = b.add("R", &edges);
    // A few genuine multi-edge pairs on top, so deletions also reach `me`.
    ids.extend(b.add("R", &[(0, 1, weight(0.5)), (7, 14, weight(0.25))]));
    b.g.flush_for_bulk();

    let t = |b: &Builder| {
        let ty = b.g.get_type_id("R").unwrap().0;
        let t = &b.g.relationship_tensors()[ty];
        (t.fwd_m().nvals(), t.fwd_dm().nvals())
    };
    let (base, _) = t(&b);
    assert!(base > 0, "the test needs a non-empty committed base");

    // Delete every 100th edge: these live in `m`, so they can only be hidden
    // by `dm`.
    let mut del = RoaringTreemap::new();
    for (i, &id) in ids.iter().enumerate() {
        if i % 100 == 0 {
            del.insert(id);
        }
    }
    let mut docs = IndexDocs::default();
    b.g.delete_relationships(&del, &mut docs).expect("delete");
    let (_, dm) = t(&b);
    assert!(
        dm > 0,
        "the deletions did not reach `dm`, so this test proves nothing"
    );
    assert_matches_oracle(&b, &["R"]);

    // Pending additions on top of the pending deletions: the effective matrix
    // then masks `m` and merges `dp` in the same call.
    b.add("R", &[(2, 3, weight(0.75)), (5, 9, weight(9.0))]);
    let ty = b.g.get_type_id("R").unwrap().0;
    let fwd_dp = b.g.relationship_tensors()[ty].fwd_dp().nvals();
    assert!(
        fwd_dp > 0 && t(&b).1 > 0,
        "`dp` and `dm` must both hold entries, or this proves nothing"
    );
    assert_matches_oracle(&b, &["R"]);
}

/// The `me` counterpart of the test above: committed multi-edge ids hidden by
/// `me`'s own `dm`. `me` is a `VersionedMatrix` under the same fold floor, so
/// this needs enough committed multi-edge ids to fold (asserted, not assumed),
/// and a deletion that leaves each touched pair still multi-edge (three edges per
/// pair, one deleted) so it lands in `me.dm` rather than demoting the pair.
#[test]
fn deletions_inside_committed_multi_edge_pairs_use_me_dm() {
    const PAIRS: u64 = 3_400;
    let mut b = Builder::new("cch_in_me_dm", PAIRS + 1);
    let mut edges = Vec::new();
    for k in 0..PAIRS {
        // The cheapest of the three is the one that gets deleted below.
        #[allow(clippy::cast_precision_loss)]
        let base = (k % 50) as f64 + 2.0;
        edges.push((k, k + 1, weight(base - 1.0)));
        edges.push((k, k + 1, weight(base)));
        edges.push((k, k + 1, weight(base + 1.0)));
    }
    let ids = b.add("R", &edges);
    b.g.flush_for_bulk();

    let me_layers = |b: &Builder| {
        let ty = b.g.get_type_id("R").unwrap().0;
        b.g.relationship_tensors()[ty]
            .edge_versioned_all()
            .fold((0, 0), |(m, dm), (_, me)| {
                (m + me.m().nvals(), dm + me.dm().nvals())
            })
    };
    assert!(
        me_layers(&b).0 > 0,
        "the test needs a non-empty committed `me`"
    );

    let mut del = RoaringTreemap::new();
    for k in (0..PAIRS as usize).step_by(10) {
        del.insert(ids[3 * k]);
    }
    let mut docs = IndexDocs::default();
    b.g.delete_relationships(&del, &mut docs).expect("delete");
    assert!(
        me_layers(&b).1 > 0,
        "the deletions did not reach `me.dm`, so this test proves nothing"
    );

    let input = assert_matches_oracle(&b, &["R"]);
    // Pair 0 lost its 1.0 edge; 2.0 is now its cheapest.
    assert_eq!(decode(&input).0[&(0, 1)], 2.0);

    // A fourth, cheaper edge on two committed multi-edge pairs, pending: new
    // ids in `me.dp` while `me.dm` still hides the deleted ones.
    b.add("R", &[(0, 1, weight(0.5)), (11, 12, weight(0.25))]);
    let ty = b.g.get_type_id("R").unwrap().0;
    let me_dp: u64 = b.g.relationship_tensors()[ty]
        .edge_versioned_all()
        .map(|(_, me)| me.dp().nvals())
        .sum();
    assert!(
        me_dp > 0 && me_layers(&b).1 > 0,
        "`me.dp` and `me.dm` must both hold entries, or this proves nothing"
    );
    let input = assert_matches_oracle(&b, &["R"]);
    assert_eq!(decode(&input).0[&(0, 1)], 0.5);
}

/// Blocks past [`ME_BLOCK_0`] need node ids at or above 2^30, which only a graph
/// that size reaches. This test reaches them without one: a bare [`Tensor`] at
/// that dimension is hypersparse, so it costs its entries, not its dimension.
mod blocks {
    use std::collections::BTreeMap;

    use rustc_hash::FxHashMap;

    use super::super::check_weights;
    use super::super::weight_ops::{WeightContext, WeightOps};
    use crate::graph::attribute_store::AttributeStore;
    use crate::graph::graphblas::tensor::Tensor;
    use crate::graph::graphblas::test_init;
    use crate::index::falkordb::cch::MAX_NODES;
    use crate::runtime::value::Value;

    /// One block per axis: 2^30, written out rather than derived from
    /// `BLOCK_SHIFT` so a change to the shift cannot move the fixtures with it.
    const B: u64 = 1 << 30;

    /// The weight passes over a real [`Tensor`] whose multi-edge pairs sit in all
    /// four blocks, against `Tensor::iter_edges` folded with `f64::min`. Four of
    /// the pairs share one row key and differ only in block, so a pass that
    /// ignored the block would put their weights on the wrong pair.
    ///
    /// No `CchInput::build`: its CSR offsets are dense in `n`, and `n` here is
    /// 2^31 - 1. Run on pending ids (`dp`) and again after a flush (`m`).
    #[test]
    fn multi_edge_pairs_past_two_to_the_thirty_land_on_their_own_pair() {
        test_init::ensure_init();
        let n = 2 * B - 1;
        assert!(n <= MAX_NODES as u64, "CCH accepts it");
        let top = n - 1;
        // Three edges per multi-edge pair, cheapest at a different position
        // each time; plus single-edge pairs, which take the inline pass. The
        // named pairs are the interesting ones; the 100 per block after them
        // only push every `me` block past the fold floor (256 entries), so the
        // second check below really reads the committed base.
        let mut multi = vec![(1, 2), (B + 1, 2), (1, B + 2), (B + 1, B + 2), (top, B)];
        for (bs, bd) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            multi.extend((0..100).map(|i| (bs * B + 10 + i, bd * B + 20 + i)));
        }
        let single = [(B + 5, 3), (4, top)];
        let (mut srcs, mut dsts, mut weights) = (vec![], vec![], vec![]);
        for (k, &(s, d)) in multi.iter().enumerate() {
            for j in 0..3u64 {
                srcs.push(s);
                dsts.push(d);
                weights.push(10.0 * (k as f64 + 1.0) + ((j + k as u64) % 3) as f64);
            }
        }
        for (k, &(s, d)) in single.iter().enumerate() {
            srcs.push(s);
            dsts.push(d);
            weights.push(100.0 + k as f64);
        }
        let ids: Vec<u64> = (0..srcs.len() as u64).collect();

        let mut attrs = AttributeStore::new();
        let rows: FxHashMap<u64, Vec<(u16, Value)>> = ids
            .iter()
            .map(|&id| (id, vec![(0u16, Value::Float(weights[id as usize]))]))
            .collect();
        attrs.insert_attrs(&rows).expect("weights stored");
        let ctx = WeightContext::new(&attrs, 0);

        let mut t = Tensor::new(n, n);
        t.set_all_from_slices(&srcs, &dsts, &ids);
        assert_eq!(t.multi_pairs(), multi.len() as u64);

        let check = |t: &Tensor, layer: &str| {
            let mut want: BTreeMap<(u64, u64), f64> = BTreeMap::new();
            for (s, d, id) in t.iter_edges() {
                let w = weights[id as usize];
                want.entry((s, d))
                    .and_modify(|m| *m = m.min(w))
                    .or_insert(w);
            }
            assert_eq!(want.len(), multi.len() + single.len(), "{layer}: oracle");

            let w = WeightOps::get().tensor_weights(t, &ctx);
            check_weights(&w).expect("valid weights");
            assert_eq!(w.nvals(), want.len() as u64, "{layer}: arc count");
            for (&(s, d), &min) in &want {
                assert_eq!(w.get(s, d), Some(min), "{layer}: ({s}, {d})");
            }
        };
        check(&t, "pending (dp)");

        t.wait(); // latches the fold decisions `dup` arms and `flush` runs
        let mut t = t.dup();
        t.flush();
        t.wait();
        assert_eq!(t.edge_versioned_all().count(), 4);
        for (block, me) in t.edge_versioned_all() {
            assert!(
                me.m().nvals() > 0 && me.dp().nvals() == 0,
                "block {block:?} did not fold into `m`, so this check proves nothing"
            );
        }
        check(&t, "committed (m)");
    }
}
