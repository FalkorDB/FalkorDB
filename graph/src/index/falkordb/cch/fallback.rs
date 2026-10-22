//! The answer `db.idx.cch.query` gives while an index's hierarchy cannot: a plain
//! Dijkstra over the live graph, with the index's own weight rule.
//!
//! A CCH index answers from its hierarchy. While that hierarchy is not there to read — being
//! built, or failed (both arrive with the background builder) — the query does not error: it
//! searches the graph directly, the way a native column falls back to a label scan while it is
//! not ready. The answer is the same shortest path weight the hierarchy would give, because
//! both read every edge through the same rule:
//!
//! - an edge's weight is [`edge_weight`]: its numeric weight attribute, or `1.0` when that is
//!   missing or not a number;
//! - `+inf` is a closed edge, never traversed;
//! - between two nodes the cheapest edge of any indexed type counts (parallel edges and several
//!   types alike), and only the indexed types are read;
//! - edges are directed, `src -> dst`, as `CchInput::build` stores them;
//! - a self-loop is never on a shortest path and is skipped.
//!
//! Negative, `-inf` and NaN weights cannot exist on an indexed type (the write-time check
//! refuses them while an index exists); meeting one anyway is an error, as building over it is.
//!
//! What may differ from the hierarchy is the path on a tie: two routes of equal weight, and the
//! hierarchy picks by its elimination order, which a search without the hierarchy cannot see.
//! This search is deterministic on its own (see [`shortest_path`]). The weight is equal, as a
//! real number; as `f64` the two add the same edge weights in a different order, so they are
//! bit-equal when every partial sum is exact (integer weights below 2^53, for instance) and can
//! otherwise differ in the last bits.
//!
//! Forward only, not bidirectional: the tensors keep edge ids on the forward matrix alone, so a
//! backward search pays a point lookup per incoming edge. On the California road network
//! (1.25M nodes, 3.06M edges, random pairs) that made the bidirectional search slower: 264 ms
//! against 222 ms at the median, 555 against 409 ms at p95.
//!
//! Host code, not GraphBLAS: a point-to-point search needs a min-heap, settling the nearest
//! unsettled node one at a time and stopping as soon as the target is settled, and no
//! GraphBLAS operation keeps one. The GraphBLAS single-source search
//! (`LAGr_SingleSourceShortestPath`) instead relaxes in buckets until every distance is
//! final, and returns no path: 0.7-2.3 s per query on the same network, depending on its
//! delta.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

use super::input::edge_weight;
use super::{CchError, CchIndexDef};
use crate::graph::attribute_store::AttributeStore;
use crate::graph::graph::Graph;
use crate::graph::graphblas::tensor;

/// Pops between two calls of the caller's poll (query timeout, memory cap): rare enough that
/// the poll costs nothing, often enough that a runaway search stops promptly. The same
/// interval `algo.SPpaths` uses.
const POLL_INTERVAL: u64 = 1024;

/// What the search knows about a node it has reached.
#[derive(Clone, Copy)]
struct Label {
    dist: f64,
    /// The node this one was reached from; [`NO_PRED`] at the source.
    pred: u64,
    /// Popped with its final distance.
    settled: bool,
}

const NO_PRED: u64 = u64::MAX;

/// A heap entry: `BinaryHeap` pops the greatest, so the order is reversed, nearest first and
/// then lowest node id, which fixes the order an equal-distance frontier is settled in.
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    dist: f64,
    node: u64,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The cheapest `src -> dst` route over the types and weight attribute of `def`, read from the
/// live graph: `(weight, node ids from src to dst)`, the shape [`Cch::query`] answers in, or
/// `None` when `dst` cannot be reached. `src == dst` is the single-node path of weight `0.0`.
///
/// Dijkstra from `src`, stopping when `dst` is settled. Each settled node's outgoing edges are
/// read through one iterator per indexed type, re-seeked to the node's row, so a row costs no
/// new iterator (a multi-edge pair still reads its ids through a fresh iterator over its `me`
/// row, `tensor::Iter`). Deterministic for a given graph version: equal distances settle lowest
/// node id first, and a label is replaced only by a strictly cheaper one. The route is a
/// shortest-path tree branch, so it never repeats a node.
///
/// `poll` is called every [`POLL_INTERVAL`] settled nodes; an error from it (a timeout, the
/// memory cap) ends the search with that error.
///
/// # Errors
/// - `poll`'s error.
/// - [`CchError::InvalidWeight`]'s message for a negative or NaN weight on an indexed edge,
///   which the write-time check should have refused.
/// - An unknown relationship type in `def`.
///
/// [`Cch::query`]: super::Cch::query
pub(crate) fn shortest_path(
    g: &Graph,
    def: &CchIndexDef,
    src: u64,
    dst: u64,
    poll: &mut dyn FnMut() -> Result<(), String>,
) -> Result<Option<(f64, Vec<u64>)>, String> {
    if src == dst {
        return Ok(Some((0.0, vec![src])));
    }
    let all = g.relationship_tensors();
    let mut rows = def
        .rel_types()
        .iter()
        .map(|t| {
            all.get(t.0)
                .map(|t| t.iter(src, src, false))
                .ok_or_else(|| format!("db.idx.cch.query: {}", CchError::UnknownRelType))
        })
        .collect::<Result<Vec<tensor::Iter<'_>>, _>>()?;
    let attrs = g.relationship_attrs();
    let attr = def.weight_attr;

    let mut labels: FxHashMap<u64, Label> = FxHashMap::default();
    labels.insert(
        src,
        Label {
            dist: 0.0,
            pred: NO_PRED,
            settled: false,
        },
    );
    let mut heap = BinaryHeap::new();
    heap.push(Candidate {
        dist: 0.0,
        node: src,
    });
    let mut pops: u64 = 0;

    // A host loop over tensor iterators, not GraphBLAS, on purpose: a point-to-point search
    // stops when `dst` is settled and recovers its path. `LAGr_SingleSourceShortestPath` settles
    // every node and returns no path; on California its best p50 was 711 ms against 222 ms here.
    while let Some(Candidate { dist: d, node: u }) = heap.pop() {
        let label = labels.get_mut(&u).expect("a queued node has a label");
        if label.settled {
            continue; // a stale entry
        }
        label.settled = true;
        if u == dst {
            return Ok(Some((d, route(&labels, dst))));
        }

        pops += 1;
        if pops.is_multiple_of(POLL_INTERVAL) {
            poll()?;
        }

        for it in &mut rows {
            it.seek(u, u);
            for (_, v, id) in it.by_ref() {
                if v == u {
                    continue;
                }
                let w = weight(attrs, attr, id)?;
                if w == f64::INFINITY {
                    continue; // a closed edge
                }
                let nd = d + w;
                let label = labels.entry(v).or_insert(Label {
                    dist: f64::INFINITY,
                    pred: NO_PRED,
                    settled: false,
                });
                if label.settled || nd >= label.dist {
                    continue;
                }
                label.dist = nd;
                label.pred = u;
                heap.push(Candidate { dist: nd, node: v });
            }
        }
    }
    Ok(None)
}

/// The nodes from the source to `to`, following predecessors back and reversing.
fn route(
    labels: &FxHashMap<u64, Label>,
    to: u64,
) -> Vec<u64> {
    let mut out = vec![to];
    let mut cur = to;
    while let Some(&Label { pred, .. }) = labels.get(&cur)
        && pred != NO_PRED
    {
        out.push(pred);
        cur = pred;
    }
    out.reverse();
    out
}

/// One edge's weight by the index's rule; a weight the index refuses is an error here too.
#[inline]
fn weight(
    attrs: &AttributeStore,
    attr: u16,
    id: u64,
) -> Result<f64, String> {
    let w = edge_weight(attrs, attr, id);
    // NaN fails the comparison; `+inf` and `-0.0` pass it.
    if w >= 0.0 {
        Ok(w)
    } else {
        Err(format!("db.idx.cch.query: {}", CchError::InvalidWeight))
    }
}

#[cfg(test)]
mod tests {
    use super::shortest_path;
    use crate::graph::graph::Graph;
    use crate::graph::graphblas::matrix::Matrix;
    use crate::index::falkordb::cch::CchIndexDef;
    use crate::index::falkordb::cch::testkit::{
        Decoys, INDEXED, RealEdge, assert_path_simple, assert_path_valid, dijkstra, grid,
        materialize, node_count, random_sparse, real_graph, reweight,
    };

    #[allow(clippy::unnecessary_wraps)]
    fn no_poll() -> Result<(), String> {
        Ok(())
    }

    fn search(
        g: &Graph,
        def: &CchIndexDef,
        s: u64,
        t: u64,
    ) -> Option<(f64, Vec<u64>)> {
        shortest_path(g, def, s, t, &mut no_poll).expect("valid weights")
    }

    /// A real graph of `edges` and the index over its `T0` type alone.
    fn on_t0(
        n: usize,
        edges: &[RealEdge],
        name: &str,
    ) -> (Graph, CchIndexDef) {
        let (g, attr) = real_graph(n, edges, name);
        let t0 = g.get_type_id(INDEXED[0]).expect("T0 exists");
        (g, CchIndexDef::new(vec![t0], attr))
    }

    /// Every ordered pair, three answers side by side: the fallback search over the real
    /// graph, the hierarchy built over the same graph, and the kit's Dijkstra over `oracle`,
    /// the cheapest edge per pair on the indexed types. Weights are integers (or `0`, or
    /// `+inf`), so equal means bit-equal. Returns how many pairs the search and the hierarchy
    /// answered with different, equally cheap paths.
    fn assert_three_way(
        mut g: Graph,
        def: &CchIndexDef,
        oracle: &Matrix<f64>,
        name: &str,
    ) -> usize {
        g.build_cch_index_now(def.clone()).expect("create");
        let cch = g.cch_index(def).expect("held").cch().expect("built");
        let n = node_count(oracle) as u64;
        let mut ties = 0;
        for s in 0..n {
            let truth = dijkstra(oracle, s);
            for t in 0..n {
                let found = search(&g, def, s, t);
                let by_cch = cch.query(s, t);
                match (&found, &by_cch) {
                    (Some((w, nodes)), Some((cw, cnodes))) => {
                        assert_eq!(*w, truth[t as usize], "{name} {s}->{t} oracle");
                        assert_eq!(w, cw, "{name} {s}->{t} hierarchy");
                        assert_path_valid(oracle, s, t, *w, nodes);
                        assert_path_simple(nodes);
                        ties += usize::from(nodes != cnodes);
                    }
                    (None, None) => {
                        assert!(truth[t as usize].is_infinite(), "{name} {s}->{t} missed");
                    }
                    _ => panic!("{name} {s}->{t}: search {found:?}, hierarchy {by_cch:?}"),
                }
            }
        }
        ties
    }

    /// Random graphs with closed edges, zero weights (and so zero-weight cycles) and pairs
    /// that cannot reach each other, materialized with every decoy: heavier parallel edges
    /// on both indexed types, closed ones, ties across the types, weight-0 edges of an
    /// unindexed type and self-loops.
    #[test]
    fn answers_like_the_hierarchy_and_the_oracle_on_random_graphs() {
        let all = Decoys {
            heavier: 2,
            closed: true,
            unindexed: true,
            self_loops: true,
            ties: true,
        };
        let mut ties = 0;
        for seed in 0..6u64 {
            let oracle = reweight(&random_sparse(28, 100, 6, seed), |i, w| {
                if i % 7 == 3 {
                    f64::INFINITY
                } else if i % 5 == 1 {
                    0.0
                } else {
                    w
                }
            });
            let name = format!("cch_fb_random_{seed}");
            let m = materialize(&oracle, all, &name);
            ties += assert_three_way(m.graph, &m.def, &oracle, &name);
        }
        // Not a requirement, a record: equal-weight routes do come up here.
        eprintln!("pairs answered by a different, equally cheap path: {ties}");
    }

    /// The caller's poll runs during a long search, and its error ends the search.
    #[test]
    fn the_poll_can_stop_a_long_search() {
        let base = grid(50, 50);
        let m = materialize(&base, Decoys::default(), "cch_fb_poll");
        let (g, def) = (&m.graph, &m.def);
        let last = node_count(&base) as u64 - 1;
        let mut calls = 0;
        let mut stop = || {
            calls += 1;
            Err(String::from("Query timed out"))
        };
        assert_eq!(
            shortest_path(g, def, 0, last, &mut stop),
            Err(String::from("Query timed out"))
        );
        assert_eq!(calls, 1);
        assert!(
            search(g, def, 0, last).is_some(),
            "without the poll it answers"
        );
    }

    /// A weight the index refuses is an error here too. Reachable only on a graph with no
    /// index over the type (the write-time check refuses it otherwise), as here.
    #[test]
    fn a_negative_weight_is_an_error() {
        let edges = [
            (0, 1, Some(1.0), INDEXED[0]),
            (1, 2, Some(-1.0), INDEXED[0]),
        ];
        let (g, def) = on_t0(3, &edges, "cch_fb_negative");
        let err = shortest_path(&g, &def, 0, 2, &mut no_poll).unwrap_err();
        assert!(err.contains("negative or NaN"), "{err}");
    }
}
