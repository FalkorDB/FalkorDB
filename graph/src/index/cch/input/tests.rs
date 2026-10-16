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
use std::{ptr, thread};

use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;

use super::DEFAULT_WEIGHT;
use crate::graph::graph::{EdgeDirection, Graph, NodeId, TypeId};
use crate::graph::graphblas::tensor::Tensor;
use crate::graph::graphblas::test_init;
use crate::index::cch::{CchError, CchIndexDef, CchInput, MAX_NODES};
use crate::index::index_docs::IndexDocs;
use crate::runtime::value::Value;

/// `(src, dst)` -> cheapest weight, and the undirected skeleton as a set of
/// ordered pairs. The two things `CchInput` carries, in a form a test can
/// compare.
type Decoded = (BTreeMap<(u32, u32), f64>, BTreeSet<(u32, u32)>);

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
            .create_relationships_bulk(&name, &srcs, &dsts, &ids)
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
fn decode(input: &CchInput) -> Decoded {
    let mut weights = BTreeMap::new();
    for u in 0..input.weights.csr.n {
        for k in input.weights.csr.p[u]..input.weights.csr.p[u + 1] {
            let k = k as usize;
            #[allow(clippy::cast_possible_truncation)]
            let prev = weights.insert((u as u32, input.weights.csr.i[k]), input.weights.w[k]);
            assert!(prev.is_none(), "duplicate weight entry in row {u}");
        }
    }
    let mut skeleton = BTreeSet::new();
    for u in 0..input.skeleton.n {
        let row = input.skeleton.row(u);
        assert!(
            row.windows(2).all(|p| p[0] < p[1]),
            "skeleton row {u} is not strictly ascending: {row:?}"
        );
        for &v in row {
            #[allow(clippy::cast_possible_truncation)]
            skeleton.insert((u as u32, v));
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
    let mut weights: BTreeMap<(u32, u32), f64> = BTreeMap::new();
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
                .entry((src as u32, dst as u32))
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
    assert_eq!(input.skeleton.n, input.n);
    assert_eq!(input.weights.csr.n, input.n);
    assert_eq!(input.weights.csr.i.len(), input.weights.w.len());
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
        assert!(
            w.is_finite() && w >= 0.0,
            "weight {w} is not finite and >= 0"
        );
    }
    input
}

fn f(x: f64) -> Option<Value> {
    Some(Value::Float(x))
}

#[test]
fn single_type_single_edges() {
    let mut b = Builder::new("cch_in_simple", 6);
    b.add(
        "R",
        &[
            (0, 1, f(2.0)),
            (1, 2, f(3.0)),
            (2, 0, f(4.0)),
            (3, 4, f(0.0)),
        ],
    );
    let input = assert_matches_oracle(&b, &["R"]);
    let (w, s) = decode(&input);
    assert_eq!(w[&(0, 1)], 2.0);
    // A zero weight is a real entry, not an absence, in both outputs.
    assert_eq!(w[&(3, 4)], 0.0);
    assert!(s.contains(&(4, 3)) && s.contains(&(3, 4)));
    assert!(!w.contains_key(&(1, 0)), "weights stay directed");
}

/// All weights equal (here: no edge carries the attribute, so every weight is
/// the 1.0 default), the case where GraphBLAS could store `W` iso, one value
/// for every entry. The input still holds one weight per entry.
#[test]
fn equal_weights_give_one_weight_per_entry() {
    let mut b = Builder::new("cch_in_equal", 5);
    b.add(
        "R",
        &[(0, 1, None), (1, 2, None), (3, 2, None), (4, 0, None)],
    );
    let input = assert_matches_oracle(&b, &["R"]);
    assert_eq!(input.weights.w, vec![1.0; 4]);
    assert_eq!(input.weights.csr.i.len(), 4);
}

#[test]
fn parallel_edges_collapse_to_the_cheapest() {
    let mut b = Builder::new("cch_in_parallel", 4);
    b.add(
        "R",
        &[
            (0, 1, f(5.0)),
            (0, 1, f(2.0)),
            (0, 1, f(9.0)),
            (1, 2, f(1.0)),
        ],
    );
    assert!(
        b.g.relationship_tensors()[b.g.get_type_id("R").unwrap().0].has_multi_edge(),
        "the test needs a genuine multi-edge pair, or it proves nothing"
    );
    let input = assert_matches_oracle(&b, &["R"]);
    assert_eq!(decode(&input).0[&(0, 1)], 2.0);
}

#[test]
fn cheapest_wins_across_relationship_types() {
    let mut b = Builder::new("cch_in_multitype", 4);
    b.add("A", &[(0, 1, f(7.0)), (1, 2, f(1.0))]);
    b.add("B", &[(0, 1, f(3.0)), (2, 3, f(8.0))]);
    b.add("C", &[(0, 1, f(100.0))]);

    let input = assert_matches_oracle(&b, &["A", "B"]);
    let (w, _) = decode(&input);
    assert_eq!(w[&(0, 1)], 3.0, "B's cheaper edge must win");
    assert!(w.contains_key(&(2, 3)), "B-only arcs are included");

    // An un-indexed type must not contribute, even though it is cheaper.
    let only_a = assert_matches_oracle(&b, &["A"]);
    assert_eq!(decode(&only_a).0[&(0, 1)], 7.0);
    assert!(!decode(&only_a).0.contains_key(&(2, 3)));
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
            (4, 5, f(9.0)),
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
    assert_eq!(
        DEFAULT_WEIGHT, 1.0,
        "C's `_edge_weight` defaults to SI_LongVal(1)"
    );
}

#[test]
fn self_loops_are_dropped_from_both_sides() {
    let mut b = Builder::new("cch_in_selfloop", 4);
    b.add("R", &[(0, 0, f(1.0)), (1, 1, f(2.0)), (0, 1, f(3.0))]);
    let input = assert_matches_oracle(&b, &["R"]);
    let (w, s) = decode(&input);
    assert!(!w.contains_key(&(0, 0)) && !w.contains_key(&(1, 1)));
    assert!(!s.contains(&(0, 0)) && !s.contains(&(1, 1)));
    assert_eq!(w[&(0, 1)], 3.0);
}

#[test]
fn a_graph_with_no_edges_builds_an_empty_input() {
    let mut b = Builder::new("cch_in_empty", 8);
    // The type has to exist for the definition to resolve, so make one and
    // give it nothing.
    b.add("R", &[]);
    let input = b.build(&["R"]).expect("an edgeless graph is not an error");
    assert_eq!(input.n, b.g.node_cap() as usize);
    assert!(input.skeleton.i.is_empty());
    assert!(input.weights.csr.i.is_empty() && input.weights.w.is_empty());
    assert_eq!(input.skeleton.p.len(), input.n + 1);
    assert!(input.skeleton.p.iter().all(|&o| o == 0));
}

#[test]
fn deleted_edges_and_pending_deltas_are_reflected() {
    let mut b = Builder::new("cch_in_deltas", 6);
    let ids = b.add(
        "R",
        &[
            (0, 1, f(2.0)),
            (1, 2, f(3.0)),
            (2, 3, f(4.0)),
            (0, 1, f(1.0)),
        ],
    );
    // Four edges are below the fold floor, so this leaves them in `dp`: what
    // follows exercises the pending-delta paths. The committed base and `dm`
    // are the job of the two large tests at the end of this file.
    b.g.flush_for_bulk();
    let before = assert_matches_oracle(&b, &["R"]);
    assert_eq!(decode(&before).0[&(0, 1)], 1.0);

    // Delete the cheaper of the two parallel 0->1 edges and the whole 2->3
    // pair, then add a new arc — all uncommitted.
    let mut rels = RoaringTreemap::new();
    rels.insert(ids[3]);
    rels.insert(ids[2]);
    let mut docs = IndexDocs::default();
    b.g.delete_relationships(&rels, &mut docs)
        .expect("delete succeeds");
    b.add("R", &[(3, 4, f(5.0))]);

    let after = assert_matches_oracle(&b, &["R"]);
    let (w, s) = decode(&after);
    assert_eq!(
        w[&(0, 1)],
        2.0,
        "the surviving parallel edge sets the weight"
    );
    assert!(!w.contains_key(&(2, 3)), "a deleted pair leaves no arc");
    assert!(!s.contains(&(2, 3)) && !s.contains(&(3, 2)));
    assert_eq!(w[&(3, 4)], 5.0, "an uncommitted arc is included");

    // And again once those deltas are folded into the base.
    b.g.flush_for_bulk();
    let committed = assert_matches_oracle(&b, &["R"]);
    assert_eq!(decode(&committed).0, w);
}

/// A node deletion leaves its id in the hierarchy as an isolated vertex: node
/// ids *are* CCH ids, so `n` never shrinks and the freed id is simply never
/// on a path.
///
/// Note the ordering this depends on. `Graph::delete_nodes` does **not** touch
/// the relationship tensors — `Pending::commit` calls it and *then* cascades
/// the incident edges through `delete_implicit_edges` (`runtime/pending.rs`).
/// So the graph is only edge-consistent once both have run, and an input built
/// between them would still carry the dead node's arcs. That is a constraint on
/// where the maintenance hook may fire, not something this module can fix.
#[test]
fn a_deleted_node_becomes_an_isolated_vertex() {
    let mut b = Builder::new("cch_in_delnode", 6);
    let ids = b.add("R", &[(0, 1, f(2.0)), (1, 2, f(3.0)), (2, 3, f(4.0))]);
    b.g.flush_for_bulk();

    // Delete node 1 the way a committed write does: the node, then its edges.
    let mut del = RoaringTreemap::new();
    del.insert(1);
    let mut docs = IndexDocs::default();
    b.g.delete_nodes(&del, &mut docs).expect("delete node");
    let mut dead_edges = RoaringTreemap::new();
    dead_edges.insert(ids[0]);
    dead_edges.insert(ids[1]);
    let mut edocs = IndexDocs::default();
    b.g.delete_relationships(&dead_edges, &mut edocs)
        .expect("cascade the incident edges");

    let input = assert_matches_oracle(&b, &["R"]);
    let (w, s) = decode(&input);
    assert!(
        w.keys().all(|&(u, v)| u != 1 && v != 1),
        "node 1's arcs survived its deletion: {w:?}"
    );
    assert!(s.iter().all(|&(u, v)| u != 1 && v != 1));
    assert_eq!(w[&(2, 3)], 4.0, "untouched arcs are unaffected");
    // The id is still a vertex of the hierarchy, just an isolated one.
    assert_eq!(input.n, b.g.node_cap() as usize);
}

#[test]
fn bad_weights_are_refused_not_crashed() {
    for (name, bad) in [
        ("cch_in_neg", f(-1.0)),
        ("cch_in_nan", f(f64::NAN)),
        ("cch_in_inf", f(f64::INFINITY)),
        ("cch_in_neginf", f(f64::NEG_INFINITY)),
        ("cch_in_negint", Some(Value::Int(-3))),
    ] {
        let mut b = Builder::new(name, 4);
        b.add("R", &[(0, 1, f(1.0)), (1, 2, bad)]);
        assert_eq!(
            b.build(&["R"]).err(),
            Some(CchError::InvalidWeight),
            "{name}: a bad weight must be refused"
        );
    }
}

/// A bad weight must be refused wherever it sits, not only when it is the cheapest
/// edge of its pair. The per-pair minimum would otherwise hide it: `GrB_MIN_FP64`
/// drops NaN (it behaves like `fmin`), and any finite weight beats `+inf`. So every
/// case here puts the bad weight on an edge that *loses* the minimum, on a
/// parallel edge (`me` path) and on a second relationship type (cross-type merge).
#[test]
fn a_bad_weight_hidden_behind_a_cheaper_edge_is_still_refused() {
    for (i, bad) in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -5.0]
        .into_iter()
        .enumerate()
    {
        let mut parallel = Builder::new(&format!("cch_in_hidden_par_{i}"), 4);
        parallel.add("R", &[(0, 1, f(2.0)), (0, 1, f(bad))]);
        assert_eq!(
            parallel.build(&["R"]).err(),
            Some(CchError::InvalidWeight),
            "{bad} on a parallel edge"
        );

        let mut multi = Builder::new(&format!("cch_in_hidden_type_{i}"), 4);
        multi.add("R", &[(0, 1, f(2.0))]);
        multi.add("S", &[(0, 1, f(bad))]);
        assert_eq!(
            multi.build(&["R", "S"]).err(),
            Some(CchError::InvalidWeight),
            "{bad} on a second relationship type"
        );
    }
}

/// A bad weight on a self-loop is refused too, even though the self-loop itself
/// is dropped from the output: weights are checked per edge, where they are
/// scored, and a multi-edge self-loop cannot be told apart there without the
/// host-side key split. Pinned so the choice is deliberate.
#[test]
fn a_bad_weight_on_a_self_loop_is_refused() {
    let mut b = Builder::new("cch_in_bad_selfloop", 4);
    b.add("R", &[(0, 1, f(1.0)), (2, 2, f(-1.0))]);
    assert_eq!(b.build(&["R"]).err(), Some(CchError::InvalidWeight));
}

/// The same on a multi-edge self-loop, where the bad weight is the dearer of
/// the two parallel edges and so is scored by the multi-edge pass alone.
#[test]
fn a_bad_weight_on_a_multi_edge_self_loop_is_refused() {
    for (i, bad) in [f64::NAN, f64::INFINITY, -1.0].into_iter().enumerate() {
        let mut b = Builder::new(&format!("cch_in_bad_multi_selfloop_{i}"), 4);
        b.add("R", &[(0, 1, f(1.0)), (2, 2, f(1.0)), (2, 2, f(bad))]);
        assert_eq!(
            b.build(&["R"]).err(),
            Some(CchError::InvalidWeight),
            "{bad}"
        );
    }
}

/// The check on the merged weights, on its own: `-inf` is how the operators
/// report a bad weight, and `+inf` left on a pair means a multi-edge pair the
/// multi-edge pass never scored, which no graph built through the tensor API
/// can produce, so it is checked here directly.
#[test]
fn the_weight_check_reads_both_ends() {
    use super::check_weights;
    use crate::graph::graphblas::matrix::Matrix;
    test_init::ensure_init();
    let mut w = Matrix::<f64>::new(4, 4);
    assert_eq!(check_weights(&w), Ok(()), "empty");
    w.set(0, 1, 0.0);
    w.set(1, 2, f64::MAX);
    assert_eq!(check_weights(&w), Ok(()), "both ends of the valid range");
    w.set(2, 3, f64::INFINITY);
    assert!(matches!(check_weights(&w), Err(CchError::Internal(_))));
    w.set(3, 0, f64::NEG_INFINITY);
    assert_eq!(check_weights(&w), Err(CchError::InvalidWeight));
}

/// The operators are created once and shared: two builds see the same
/// handles, and builds on several threads at once (the first of which may be
/// the one creating them) agree with the oracle.
#[test]
fn the_weight_operators_are_created_once_and_shared() {
    use super::WeightOps;
    let graphs: Vec<Builder> = (0..4)
        .map(|k| {
            let mut b = Builder::new(&format!("cch_in_shared_ops_{k}"), 8);
            #[allow(clippy::cast_precision_loss)]
            b.add(
                "R",
                &[
                    (0, 1, f(k as f64)),
                    (0, 1, f(9.0)),
                    (1, 2, f(2.0)),
                    (3, 3, f(1.0)),
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
    let (a, b) = (WeightOps::get(), WeightOps::get());
    assert!(ptr::eq(a, b));
    assert_eq!(a.inline_op, b.inline_op);
}

#[test]
fn unknown_definitions_are_refused() {
    let mut b = Builder::new("cch_in_unknown", 4);
    b.add("R", &[(0, 1, f(1.0))]);
    let real = b.g.get_type_id("R").unwrap();

    assert_eq!(
        CchInput::build(&b.g, &CchIndexDef::new(vec![TypeId(99)], b.weight_attr)).err(),
        Some(CchError::UnknownRelType)
    );
    assert_eq!(
        CchInput::build(&b.g, &CchIndexDef::new(vec![real], 250)).err(),
        Some(CchError::UnknownAttribute)
    );
}

/// One past the `MAX_NODES` (2^31 - 1) ceiling is refused before anything is
/// allocated. The at-limit side is deliberately not built here: the contract's
/// CSR row pointers are `n + 1` `u64`s, so an input at the ceiling is 16 GiB
/// per array.
#[test]
fn more_than_max_nodes_is_refused() {
    test_init::ensure_init();
    let make = |cap: u64, name: &str| {
        let mut g = Graph::new(cap, 16, 1, 0, name);
        let rel = Arc::new(String::from("R"));
        g.create_relationships_bulk(&rel, &[], &[], &[])
            .expect("creatable");
        let attr = g.get_or_create_rel_attr_id(&Arc::new(String::from("w")));
        let def = CchIndexDef::new(vec![g.get_type_id("R").unwrap()], attr);
        assert_eq!(
            g.node_cap(),
            cap,
            "the test relies on node_cap == requested"
        );
        CchInput::build(&g, &def)
    };
    assert_eq!(
        MAX_NODES,
        i32::MAX as usize,
        "DECISIONS: the ceiling is the ordering backend's, 2^31 - 1"
    );
    assert_eq!(
        make(MAX_NODES as u64 + 1, "cch_in_toolarge").err(),
        Some(CchError::GraphTooLarge)
    );
    assert_eq!(
        make(1 << 32, "cch_in_u32").err(),
        Some(CchError::GraphTooLarge)
    );
}

/// The broad one: several relationship types, parallel edges, self-loops,
/// missing and non-numeric weights, deletions, and uncommitted deltas, all at
/// once. A deterministic LCG rather than a crate, so a failure is replayable
/// from the seed printed in the assertion.
#[test]
fn random_graphs_match_the_oracle() {
    for seed in 1u64..=12 {
        let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
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
    ids.extend(b.add("R", &[(0, 1, f(0.5)), (7, 14, f(0.25))]));
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
    b.add("R", &[(2, 3, f(0.75)), (5, 9, f(9.0))]);
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
        edges.push((k, k + 1, f(base - 1.0)));
        edges.push((k, k + 1, f(base)));
        edges.push((k, k + 1, f(base + 1.0)));
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
    b.add("R", &[(0, 1, f(0.5)), (11, 12, f(0.25))]);
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
/// that size reaches. These tests reach them without one: the key split is a
/// pure function, and a bare [`Tensor`] at that dimension is hypersparse, so it
/// costs its entries, not its dimension.
mod blocks {
    use std::collections::BTreeMap;

    use rustc_hash::FxHashMap;

    use super::super::{CchWeightCtx, WeightOps, block_pair_coords, check_weights};
    use crate::graph::attribute_store::AttributeStore;
    use crate::graph::graphblas::matrix::Matrix;
    use crate::graph::graphblas::tensor::{BLOCK_SHIFT, ME_BLOCK_0, MeBlock, Tensor, compound_key};
    use crate::graph::graphblas::test_init;
    use crate::index::cch::MAX_NODES;
    use crate::runtime::value::Value;

    /// One block per axis: 2^30, written out rather than derived from
    /// `BLOCK_SHIFT` so a change to the shift cannot move the fixtures with it.
    const B: u64 = 1 << 30;

    #[test]
    fn the_fixture_constant_is_the_block_width() {
        assert_eq!(B, 1 << BLOCK_SHIFT);
    }

    /// Pairs in each of the four blocks a graph under `MAX_NODES` can have,
    /// grouped through the tensor's own `compound_key` and split back.
    #[test]
    fn every_reachable_block_splits_back_to_its_pairs() {
        let top = MAX_NODES as u64 - 1; // the largest node id CCH accepts
        assert_eq!(top, 2 * B - 2);
        let pairs = [
            (0, 1),
            (1, 0),
            (12_345, 678),
            (B - 1, B - 1),
            (B, 0),
            (B, B - 1),
            (top, 7),
            (0, B),
            (5, top),
            (B - 1, B),
            (B, B),
            (B + 3, B + 9),
            (top, top),
        ];
        let mut by_block: BTreeMap<MeBlock, (Vec<u64>, Vec<(u64, u64)>)> = BTreeMap::new();
        for &(s, d) in &pairs {
            let (block, row) = compound_key(s, d);
            let e = by_block.entry(block).or_default();
            e.0.push(row);
            e.1.push((s, d));
        }
        assert_eq!(
            by_block.keys().copied().collect::<Vec<_>>(),
            vec![ME_BLOCK_0, (0, 1), (1, 0), (1, 1)],
            "the fixture must cover all four blocks"
        );
        for (block, (rows, want)) in by_block {
            let (src, dst) = block_pair_coords(block, &rows);
            let got: Vec<(u64, u64)> = src.into_iter().zip(dst).collect();
            assert_eq!(got, want, "block {block:?}");
        }
    }

    /// The edges of a block's key space, with the expected pairs written out by
    /// hand: the first and last key, and the keys either side of where the
    /// source half begins.
    #[test]
    fn boundary_keys_split_exactly() {
        let keys = [0, 1, B - 1, B, B * B - 1];
        let offsets = [(0, 0), (0, 1), (0, B - 1), (1, 0), (B - 1, B - 1)];
        for block in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let base = (block.0 * B, block.1 * B);
            let (src, dst) = block_pair_coords(block, &keys);
            let got: Vec<(u64, u64)> = src.into_iter().zip(dst).collect();
            let want: Vec<(u64, u64)> = offsets
                .iter()
                .map(|&(s, d)| (base.0 + s, base.1 + d))
                .collect();
            assert_eq!(got, want, "block {block:?}");
        }
        assert_eq!(block_pair_coords((1, 1), &[]), (vec![], vec![]));
    }

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
        let n = MAX_NODES as u64;
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
        let ctx = CchWeightCtx {
            attrs: std::ptr::from_ref(&attrs),
            attr_idx: 0,
        };

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

            let mut w = Matrix::<f64>::new(n, n);
            WeightOps::get().accumulate(&mut w, t, &ctx, n);
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

/// The rule the path procedure used before [`super::cheapest_edge`]: scan
/// `u`'s outgoing edges of `types`, keep those ending at `v`, and take the
/// first strict minimum of the numeric weight (1.0 when missing). The oracle
/// for the per-pair lookup, which must pick the same edge, ties included.
fn scan_cheapest(
    g: &Graph,
    types: &[Arc<String>],
    attr: u16,
    u: u64,
    v: u64,
) -> Option<(u64, f64)> {
    let mut best: Option<(u64, f64)> = None;
    for (_, dst, eid) in
        g.get_node_relationships_by_type(NodeId::from(u), types, EdgeDirection::Outgoing)
    {
        if u64::from(dst) != v {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let w = match g.get_relationship_attribute_by_idx(eid, attr) {
            Some(Value::Int(i)) => i as f64,
            Some(Value::Float(f)) => f,
            _ => 1.0,
        };
        if best.is_none_or(|(_, b)| w < b) {
            best = Some((u64::from(eid), w));
        }
    }
    best
}

/// [`super::cheapest_edge`] against [`scan_cheapest`] on every ordered pair of
/// `0..n`, for `types` in the given order and reversed (the order breaks
/// ties). Returns how many pairs had a tie for the minimum across or within
/// types, so a caller can check the fixture really exercised ties.
fn assert_cheapest_matches_scan(
    b: &Builder,
    types: &[&str],
    n: u64,
    layer: &str,
) -> usize {
    let mut ties = 0;
    for order in [types.to_vec(), types.iter().rev().copied().collect()] {
        let names: Vec<Arc<String>> = order.iter().map(|t| Arc::new(String::from(*t))).collect();
        let tensors: Vec<&Tensor> = names
            .iter()
            .map(|t| b.g.get_relationship_matrix(t).expect("type exists"))
            .collect();
        let attrs = b.g.relationship_attrs();
        for u in 0..n {
            for v in 0..n {
                let got = super::cheapest_edge(tensors.iter().copied(), attrs, b.weight_attr, u, v);
                let want = scan_cheapest(&b.g, &names, b.weight_attr, u, v);
                assert_eq!(got, want, "{layer}, order {order:?}: ({u}, {v})");
                if let Some((_, w)) = want {
                    let tied = tensors
                        .iter()
                        .flat_map(|t| t.get(u, v))
                        .filter(|&id| super::edge_weight(attrs, b.weight_attr, id) == w)
                        .count();
                    ties += usize::from(tied > 1);
                }
            }
        }
    }
    ties
}

/// The path procedure's per-hop lookup picks exactly the edge the old
/// outgoing-edge scan picked: random parallel edges on three types (so most
/// pairs are multi-edge pairs read from `me`), weights from a small set with
/// missing and integer ones (so ties are common), on the committed base,
/// then with uncommitted deletions (`dm`, `me`'s `dm`) and additions
/// (`dp`, promotions) on top, then folded again.
#[test]
fn cheapest_edge_picks_what_the_outgoing_scan_picked() {
    const N: u64 = 30;
    let types = ["A", "B", "C"];
    let mut b = Builder::new("cch_in_cheapest", N);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    };
    let weight = |k: u64| match k % 6 {
        0 => None,
        1 => Some(Value::Int(1)),
        2 => f(1.0),
        3 => f(2.0),
        4 => Some(Value::Int(2)),
        _ => f(0.5),
    };
    let mut random_edges = |count: usize| -> Vec<(u64, u64, Option<Value>)> {
        (0..count)
            .map(|_| (next() % N, next() % N, weight(next())))
            .collect()
    };
    let mut ids = Vec::new();
    for t in types {
        let edges = random_edges(600);
        ids.extend(b.add(t, &edges));
    }
    b.g.flush_for_bulk();
    let tensor = |b: &Builder, t: &str| -> (u64, u64, u64, bool) {
        let x =
            b.g.get_relationship_matrix(&Arc::new(String::from(t)))
                .unwrap();
        (
            x.fwd_m().nvals(),
            x.fwd_dp().nvals(),
            x.fwd_dm().nvals(),
            x.has_multi_edge(),
        )
    };
    for t in types {
        let (m, dp, _, multi) = tensor(&b, t);
        assert!(
            m > 0 && dp == 0 && multi,
            "{t}: committed base with multi-edge pairs"
        );
    }
    let ties = assert_cheapest_matches_scan(&b, &types, N, "committed");
    assert!(
        ties > 100,
        "only {ties} tied pairs: the fixture does not test ties"
    );

    let mut del = RoaringTreemap::new();
    for (i, &id) in ids.iter().enumerate() {
        if i % 7 == 0 {
            del.insert(id);
        }
    }
    let mut docs = IndexDocs::default();
    b.g.delete_relationships(&del, &mut docs).expect("delete");
    for t in types {
        let edges = random_edges(40);
        b.add(t, &edges);
    }
    for t in types {
        let (_, dp, dm, _) = tensor(&b, t);
        assert!(dp > 0 && dm > 0, "{t}: uncommitted additions and deletions");
        let x =
            b.g.get_relationship_matrix(&Arc::new(String::from(t)))
                .unwrap();
        assert!(
            x.edge_versioned_all().any(|(_, me)| me.dm().nvals() > 0),
            "{t}: deletions inside committed multi-edge pairs"
        );
    }
    assert_cheapest_matches_scan(&b, &types, N, "pending");

    b.g.flush_for_bulk();
    assert_cheapest_matches_scan(&b, &types, N, "folded");
}

/// A tensor narrower than the input's dimension (`n` is the widest tensor in
/// play) is scored through a resized matrix of the pass's own, in both forms
/// of the effective matrix: the weights land on their pairs, and the tensor
/// keeps its shape.
mod narrow_tensor {
    use rustc_hash::FxHashMap;

    use super::super::{CchWeightCtx, WeightOps, check_weights};
    use crate::graph::attribute_store::AttributeStore;
    use crate::graph::graphblas::matrix::Matrix;
    use crate::graph::graphblas::tensor::Tensor;
    use crate::graph::graphblas::test_init;
    use crate::runtime::value::Value;

    #[test]
    fn a_narrower_tensor_is_resized_on_a_matrix_of_its_own() {
        test_init::ensure_init();
        // Enough edges that `flush` folds them into the committed base.
        const SMALL: u64 = 1_000;
        let n = 2 * SMALL;
        let mut attrs = AttributeStore::new();
        let rows: FxHashMap<u64, Vec<(u16, Value)>> = (0..SMALL)
            .map(|id| (id, vec![(0u16, Value::Float(id as f64 + 0.5))]))
            .collect();
        attrs.insert_attrs(&rows).expect("weights stored");
        let ctx = CchWeightCtx {
            attrs: std::ptr::from_ref(&attrs),
            attr_idx: 0,
        };
        let check = |t: &Tensor, layer: &str| {
            let mut w = Matrix::<f64>::new(n, n);
            WeightOps::get().accumulate(&mut w, t, &ctx, n);
            check_weights(&w).expect("valid weights");
            let edges: Vec<_> = t.iter_edges().collect();
            assert_eq!(w.nvals(), edges.len() as u64, "{layer}: arc count");
            for (s, d, id) in edges {
                assert_eq!(w.get(s, d), Some(id as f64 + 0.5), "{layer}: ({s}, {d})");
            }
            let base = t.fwd_m();
            assert_eq!(
                (base.nrows(), base.ncols()),
                (SMALL, SMALL),
                "{layer}: the tensor's base was resized"
            );
        };

        let mut t = Tensor::new(SMALL, SMALL);
        let srcs: Vec<u64> = (0..SMALL - 1).collect();
        let dsts: Vec<u64> = (1..SMALL).collect();
        t.set_all_from_slices(&srcs, &dsts, &srcs);
        t.wait();
        let mut t = t.dup();
        t.flush();
        assert!(t.fwd_m().nvals() == SMALL - 1 && t.fwd_dp().nvals() == 0);
        check(&t, "committed (m)");

        t.set_all_from_slices(&[SMALL - 1], &[0], &[SMALL - 1]);
        t.wait();
        assert!(t.fwd_dp().nvals() == 1);
        check(&t, "pending (dp)");
    }
}

/// Not a check: what `CchInput::build` costs per node of capacity, for reading
/// the practical ceiling under `MAX_NODES`. Builds a graph of capacity
/// `CCH_MEM_N` holding 1,000 edges, and calls `CchInput::build` only when
/// `CCH_MEM_BUILD=1`, so peak RSS with and without it isolates the build. Run
/// under `/usr/bin/time -l` (macOS) or `-v` (Linux):
///
/// ```text
/// CCH_MEM_N=67108864 CCH_MEM_BUILD=1 /usr/bin/time -l <test binary> \
///     cch_input_memory_per_node --ignored --exact
/// ```
#[test]
#[ignore = "memory measurement, not a check"]
fn cch_input_memory_per_node() {
    let n: u64 = std::env::var("CCH_MEM_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 20);
    let build = std::env::var("CCH_MEM_BUILD").is_ok_and(|v| v == "1");
    test_init::ensure_init();
    let mut g = Graph::new(n, 16, 1, 0, "cch_in_mem");
    assert_eq!(g.node_cap(), n);
    let rel = Arc::new(String::from("R"));
    let ids = g
        .relationship_id_space()
        .reserve(1_000, &RoaringTreemap::new())
        .expect("reserved");
    let srcs: Vec<u64> = (0..1_000).map(|k| (k * 7_919) % n).collect();
    let dsts: Vec<u64> = (0..1_000).map(|k| (k * 104_729 + 1) % n).collect();
    let mut nodes: RoaringTreemap = srcs.iter().chain(&dsts).copied().collect();
    nodes.insert(n - 1);
    g.create_nodes(&nodes).expect("creatable");
    g.create_relationships_bulk(&rel, &srcs, &dsts, &ids)
        .expect("creatable");
    let attr = g.get_or_create_rel_attr_id(&Arc::new(String::from("w")));
    let def = CchIndexDef::new(vec![g.get_type_id("R").unwrap()], attr);
    if build {
        let input = CchInput::build(&g, &def).expect("builds");
        eprintln!(
            "n {n}: skeleton nnz {}, weights nnz {}",
            input.skeleton.i.len(),
            input.weights.w.len()
        );
    }
}
