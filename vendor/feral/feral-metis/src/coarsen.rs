//! Multilevel coarsening.
//!
//! Sorted Heavy-Edge Matching (SHEM): iterate vertices in ascending
//! degree order (ties broken by a seeded random permutation) and, for
//! each unmatched vertex, pair it with its unmatched neighbor of
//! maximum edge weight (ties broken by lower vertex id). Visiting
//! low-degree vertices first — the "sorted" in SHEM — keeps a leaf
//! from being stranded when its sole neighbor is claimed by a
//! higher-degree vertex. If the reduction ratio falls below the
//! configured threshold (default 0.85 per METIS 5.2.0), a simple
//! 2-hop fallback pairs unmatched vertices that share a common
//! neighbor. This keeps power-law-degree KKT graphs from stalling
//! the coarsening pipeline.
//!
//! Contraction then folds the fine graph into a coarse graph:
//! parallel edges between the same coarse endpoints are **summed**,
//! self-loops (edges inside a merged pair) are **dropped**, vertex
//! weights accumulate.
//!
//! References:
//! - Karypis & Kumar, "A Fast and High Quality Multilevel Scheme",
//!   §3.1 (matching) and §3.2 (graph contraction).
//! - METIS 5.2.0 `libmetis/coarsen.c` (`Match_SHEM`, `Match_2Hop`,
//!   `CreateCoarseGraph`).

use crate::graph::Graph;
use crate::rng::SplitMix;
use crate::MetisOptions;
use feral_ordering_core::Idx;

const UNMATCHED: Idx = -1;

/// Output of one coarsening level.
#[derive(Debug)]
pub struct CoarseGraph {
    /// Coarser graph.
    pub graph: Graph,
    /// Map from fine vertex index to coarse vertex index. Length
    /// `fine.nvtxs`.
    pub cmap: Vec<Idx>,
}

/// Coarsening-pass diagnostics (accumulated into `MetisStats`).
#[derive(Debug, Default)]
pub struct CoarsenCounters {
    pub n_two_hop_fallbacks: u32,
}

/// Coarsen a single level. Returns the coarse graph plus a fine-to-
/// coarse vertex map. Always produces a valid result; when matching
/// is sparse the coarse graph can be nearly as large as the fine
/// graph (caller is responsible for detecting stalled coarsening).
pub fn coarsen_level(
    fine: &Graph,
    rng: &mut SplitMix,
    two_hop_threshold: f64,
    counters: &mut CoarsenCounters,
) -> CoarseGraph {
    let n = fine.nvtxs as usize;
    let mut match_: Vec<Idx> = vec![UNMATCHED; n];
    let mut cmap: Vec<Idx> = vec![-1; n];

    // --- Pass 1: SHEM ---
    // Sorted Heavy-Edge Matching: visit vertices in ascending degree
    // order, breaking ties by the seeded random permutation. METIS
    // Match_SHEM bucket-sorts the random permutation by degree so that
    // low-degree vertices — which have the fewest matching options —
    // are matched before their few neighbors are claimed by a
    // higher-degree vertex. Plain shuffle order (HEM) strands those
    // leaves as self-matches, inflating the coarse graph on the
    // irregular / power-law inputs SHEM is meant for. `sort_by_key` is
    // stable, so the random shuffle survives as the within-degree
    // tie-break, preserving seed determinism. [O7]
    let mut order: Vec<Idx> = (0..fine.nvtxs).collect();
    rng.shuffle(&mut order);
    order.sort_by_key(|&v| fine.xadj[v as usize + 1] - fine.xadj[v as usize]);

    let mut cnvtxs: Idx = 0;
    for &v in &order {
        let vu = v as usize;
        if match_[vu] != UNMATCHED {
            continue;
        }
        let lo = fine.xadj[vu] as usize;
        let hi = fine.xadj[vu + 1] as usize;
        let mut best: Idx = -1;
        let mut best_w: Idx = -1;
        for k in lo..hi {
            let u = fine.adjncy[k];
            if match_[u as usize] != UNMATCHED {
                continue;
            }
            let w = fine.adjwgt[k];
            // Tie-break: strictly greater weight wins; on ties pick
            // the lower vertex id (matches METIS SHEM).
            if w > best_w || (w == best_w && (best < 0 || u < best)) {
                best = u;
                best_w = w;
            }
        }
        if best >= 0 {
            match_[vu] = best;
            match_[best as usize] = v;
            cmap[vu] = cnvtxs;
            cmap[best as usize] = cnvtxs;
        } else {
            // No unmatched neighbor — self-match for now, may be
            // revisited by 2-hop.
            match_[vu] = v;
            cmap[vu] = cnvtxs;
        }
        cnvtxs += 1;
    }

    // --- Pass 2: 2-hop fallback, only if reduction is poor ---
    let reduction_ratio = cnvtxs as f64 / fine.nvtxs.max(1) as f64;
    if reduction_ratio > two_hop_threshold && fine.nvtxs >= 4 {
        counters.n_two_hop_fallbacks += 1;
        cnvtxs = two_hop_pass(fine, &mut match_, &mut cmap);
    }

    // --- Contract ---
    let graph = contract(fine, &cmap, cnvtxs);
    CoarseGraph { graph, cmap }
}

/// Run `coarsen_level` in a loop until the graph is small enough or
/// coarsening stalls (reduction ratio < 5%). Returns the hierarchy
/// of coarse graphs, finest first.
pub fn coarsen(
    fine: &Graph,
    opts: &MetisOptions,
    rng: &mut SplitMix,
    counters: &mut CoarsenCounters,
) -> Vec<CoarseGraph> {
    let mut levels: Vec<CoarseGraph> = Vec::new();
    let mut prev_nvtxs = fine.nvtxs;
    let mut cur: &Graph = fine;
    loop {
        if cur.nvtxs <= opts.coarsen_floor as Idx {
            break;
        }
        let level = coarsen_level(cur, rng, opts.two_hop_ratio_threshold, counters);
        let new_nvtxs = level.graph.nvtxs;
        if new_nvtxs == 0 || new_nvtxs as f64 > 0.95 * prev_nvtxs as f64 {
            // Stalled: this level made <5% progress, so stop. Keep it
            // only if it actually shrank, independent of whether earlier
            // levels exist. The old `!levels.is_empty()` gate both
            // discarded a *first* level that genuinely shrank (returning
            // an empty hierarchy) and pushed a zero-progress later level
            // (breaking the strictly-decreasing-nvtxs invariant). [O8]
            if new_nvtxs > 0 && new_nvtxs < prev_nvtxs {
                levels.push(level);
            }
            break;
        }
        prev_nvtxs = new_nvtxs;
        levels.push(level);
        cur = &levels.last().expect("just pushed").graph;
    }
    levels
}

/// 2-hop matching pass: revisit self-matched vertices and pair them
/// with another self-matched vertex that shares a common neighbor.
/// Rebuilds `cmap` and returns the new `cnvtxs`.
///
/// `first_free[mid]` is where the scan of `mid`'s neighbours resumes:
/// a vertex once matched stays matched, so the entries before it can
/// never be a partner again. Without it a hub's adjacency is rescanned
/// from the start for each of its neighbours, quadratic in its degree
/// (a star of 200k leaves took 11 s); with it the pass is linear. The
/// partner chosen is the same.
fn two_hop_pass(fine: &Graph, match_: &mut [Idx], cmap: &mut [Idx]) -> Idx {
    let mut first_free: Vec<usize> = fine.xadj[..fine.nvtxs as usize]
        .iter()
        .map(|&x| x as usize)
        .collect();
    for v in 0..fine.nvtxs {
        let vu = v as usize;
        if match_[vu] != v {
            // Already paired in SHEM or a previous 2-hop step.
            continue;
        }
        // Look for a 2-hop partner: neighbor of a neighbor that is
        // currently self-matched.
        let lo = fine.xadj[vu] as usize;
        let hi = fine.xadj[vu + 1] as usize;
        let mut partner: Idx = -1;
        'outer: for k in lo..hi {
            let mid = fine.adjncy[k];
            let mhi = fine.xadj[(mid as usize) + 1] as usize;
            let mut mlo = first_free[mid as usize];
            while mlo < mhi && match_[fine.adjncy[mlo] as usize] != fine.adjncy[mlo] {
                mlo += 1;
            }
            first_free[mid as usize] = mlo;
            for &w in &fine.adjncy[mlo..mhi] {
                if w == v {
                    continue;
                }
                if match_[w as usize] == w {
                    partner = w;
                    break 'outer;
                }
            }
        }
        if partner >= 0 {
            match_[vu] = partner;
            match_[partner as usize] = v;
        }
    }

    // Rebuild cmap contiguous from the (possibly updated) match_.
    cmap.iter_mut().for_each(|c| *c = -1);
    let mut cnvtxs: Idx = 0;
    for v in 0..fine.nvtxs {
        let vu = v as usize;
        if cmap[vu] >= 0 {
            continue;
        }
        cmap[vu] = cnvtxs;
        let u = match_[vu];
        if u != v {
            cmap[u as usize] = cnvtxs;
        }
        cnvtxs += 1;
    }
    cnvtxs
}

/// Build the coarse graph from a fine graph and a fine-to-coarse map.
fn contract(fine: &Graph, cmap: &[Idx], cnvtxs: Idx) -> Graph {
    let cn = cnvtxs as usize;
    // Accumulate vertex weights.
    let mut vwgt: Vec<Idx> = vec![0; cn];
    for v in 0..fine.nvtxs as usize {
        vwgt[cmap[v] as usize] = vwgt[cmap[v] as usize].saturating_add(fine.vwgt[v]);
    }
    // Group fine vertices by coarse id to avoid rescans.
    let mut head: Vec<Idx> = vec![-1; cn];
    let mut next: Vec<Idx> = vec![-1; fine.nvtxs as usize];
    for v in 0..fine.nvtxs as usize {
        let c = cmap[v] as usize;
        next[v] = head[c];
        head[c] = v as Idx;
    }
    // Contract one coarse vertex at a time with an edge-weight marker.
    let mut marker: Vec<Idx> = vec![-1; cn];
    let mut weight_to: Vec<Idx> = vec![0; cn];
    let mut xadj: Vec<Idx> = Vec::with_capacity(cn + 1);
    let mut adjncy: Vec<Idx> = Vec::with_capacity(fine.adjncy.len());
    let mut adjwgt: Vec<Idx> = Vec::with_capacity(fine.adjncy.len());
    xadj.push(0);
    let mut touched: Vec<Idx> = Vec::new();
    for (c, &head_c) in head.iter().enumerate().take(cn) {
        touched.clear();
        let mut v = head_c;
        while v >= 0 {
            let vu = v as usize;
            let lo = fine.xadj[vu] as usize;
            let hi = fine.xadj[vu + 1] as usize;
            for k in lo..hi {
                let nbr = fine.adjncy[k];
                let cn2 = cmap[nbr as usize];
                if cn2 == c as Idx {
                    // self-loop after contraction — drop
                    continue;
                }
                let cu = cn2 as usize;
                if marker[cu] != c as Idx {
                    marker[cu] = c as Idx;
                    weight_to[cu] = fine.adjwgt[k];
                    touched.push(cn2);
                } else {
                    weight_to[cu] = weight_to[cu].saturating_add(fine.adjwgt[k]);
                }
            }
            v = next[vu];
        }
        for &tgt in &touched {
            adjncy.push(tgt);
            adjwgt.push(weight_to[tgt as usize]);
        }
        xadj.push(adjncy.len() as Idx);
    }
    Graph {
        nvtxs: cnvtxs,
        xadj,
        adjncy,
        vwgt,
        adjwgt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use feral_ordering_core::CscPattern;
    use std::collections::BTreeSet;

    fn csc_from_triples(n: usize, triples: &[(usize, usize)]) -> (Vec<Idx>, Vec<Idx>) {
        let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
        for &(i, j) in triples {
            set.insert((i, j));
            set.insert((j, i));
        }
        let mut cols: Vec<Vec<Idx>> = vec![Vec::new(); n];
        for &(r, c) in &set {
            cols[c].push(r as Idx);
        }
        for col in &mut cols {
            col.sort();
        }
        let mut col_ptr: Vec<Idx> = vec![0];
        let mut row_idx: Vec<Idx> = Vec::new();
        for col in &cols {
            for &r in col {
                row_idx.push(r);
            }
            col_ptr.push(row_idx.len() as Idx);
        }
        (col_ptr, row_idx)
    }

    fn grid(m: usize, n: usize) -> Graph {
        let idx = |r: usize, c: usize| r * n + c;
        let total = m * n;
        let mut t = Vec::new();
        for r in 0..m {
            for c in 0..n {
                let k = idx(r, c);
                t.push((k, k));
                if r + 1 < m {
                    t.push((k, idx(r + 1, c)));
                }
                if c + 1 < n {
                    t.push((k, idx(r, c + 1)));
                }
            }
        }
        let (cp, ri) = csc_from_triples(total, &t);
        let pat = CscPattern::new(total, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    fn tridiag(n: usize) -> Graph {
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i));
            if i + 1 < n {
                t.push((i, i + 1));
            }
        }
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    fn assert_valid_coarse(fine: &Graph, cg: &CoarseGraph) {
        assert_eq!(cg.cmap.len(), fine.nvtxs as usize);
        for &c in &cg.cmap {
            assert!(c >= 0 && c < cg.graph.nvtxs, "cmap entry OOB: {}", c);
        }
        // Vertex weight conservation.
        let fine_total: i64 = fine.vwgt.iter().copied().sum();
        let coarse_total: i64 = cg.graph.vwgt.iter().copied().sum();
        assert_eq!(fine_total, coarse_total, "vwgt conservation");
        // No self-loops in coarse graph.
        for v in 0..cg.graph.nvtxs {
            let lo = cg.graph.xadj[v as usize] as usize;
            let hi = cg.graph.xadj[(v + 1) as usize] as usize;
            for &u in &cg.graph.adjncy[lo..hi] {
                assert_ne!(u, v, "self-loop in coarse graph at {}", v);
            }
        }
        // Structural symmetry.
        for v in 0..cg.graph.nvtxs {
            let lo = cg.graph.xadj[v as usize] as usize;
            let hi = cg.graph.xadj[(v + 1) as usize] as usize;
            for &u in &cg.graph.adjncy[lo..hi] {
                let ulo = cg.graph.xadj[u as usize] as usize;
                let uhi = cg.graph.xadj[(u + 1) as usize] as usize;
                assert!(
                    cg.graph.adjncy[ulo..uhi].contains(&v),
                    "asymmetric coarse edge {} -> {}",
                    v,
                    u
                );
            }
        }
    }

    #[test]
    fn coarsen_grid_8x8_halves_vertices() {
        let g = grid(8, 8);
        let mut rng = SplitMix::new(1);
        let mut ctr = CoarsenCounters::default();
        let cg = coarsen_level(&g, &mut rng, 0.85, &mut ctr);
        assert_valid_coarse(&g, &cg);
        // On a 2D grid SHEM should pair ~half the vertices.
        assert!(
            cg.graph.nvtxs < g.nvtxs,
            "coarse graph must shrink (was {}, now {})",
            g.nvtxs,
            cg.graph.nvtxs
        );
        assert!(
            cg.graph.nvtxs as f64 <= 0.70 * g.nvtxs as f64,
            "reduction ratio on 8x8 grid should be <= 0.70, got {}/{}",
            cg.graph.nvtxs,
            g.nvtxs
        );
    }

    #[test]
    fn shem_matches_low_degree_before_hub() {
        // Chorded path 0-1-2-3 plus the extra edge 0-2. Degrees:
        // 0:2, 1:2, 2:3 (hub), 3:1 (leaf). Plain heavy-edge matching
        // in shuffle order (seed 1 visits the hub 2 first) pairs
        // 2<->0, stranding both the leaf 3 and vertex 1 as
        // self-matches -> 3 coarse vertices. Sorted heavy-edge
        // matching (SHEM) visits the degree-1 leaf 3 first, pairing
        // 3<->2, then 0<->1 -> 2 coarse vertices. Ascending-degree
        // visitation is the defining property of METIS Match_SHEM
        // (Karypis & Kumar Sec. 3.1); plain shuffle order is HEM,
        // not the advertised SHEM. [O7]
        let t = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 2),
        ];
        let (cp, ri) = csc_from_triples(4, &t);
        let pat = CscPattern::new(4, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let mut rng = SplitMix::new(1);
        let mut ctr = CoarsenCounters::default();
        let cg = coarsen_level(&g, &mut rng, 0.85, &mut ctr);
        assert_valid_coarse(&g, &cg);
        assert_eq!(
            cg.graph.nvtxs, 2,
            "SHEM must match the degree-1 leaf with the hub and pair \
             the remaining two vertices for 2 coarse vertices; got {}",
            cg.graph.nvtxs
        );
    }

    #[test]
    fn coarsen_tridiag_10() {
        let g = tridiag(10);
        let mut rng = SplitMix::new(1);
        let mut ctr = CoarsenCounters::default();
        let cg = coarsen_level(&g, &mut rng, 0.85, &mut ctr);
        assert_valid_coarse(&g, &cg);
        assert!(cg.graph.nvtxs <= 6);
    }

    #[test]
    fn coarsen_is_deterministic_with_seed() {
        let g = grid(6, 6);
        let mut r1 = SplitMix::new(42);
        let mut r2 = SplitMix::new(42);
        let mut c1 = CoarsenCounters::default();
        let mut c2 = CoarsenCounters::default();
        let a = coarsen_level(&g, &mut r1, 0.85, &mut c1);
        let b = coarsen_level(&g, &mut r2, 0.85, &mut c2);
        assert_eq!(a.cmap, b.cmap);
        assert_eq!(a.graph.xadj, b.graph.xadj);
        assert_eq!(a.graph.adjncy, b.graph.adjncy);
    }

    #[test]
    fn coarsen_hierarchy_shrinks_monotonically() {
        let g = grid(12, 12);
        let opts = MetisOptions {
            coarsen_floor: 20,
            ..Default::default()
        };
        let mut rng = SplitMix::new(1);
        let mut ctr = CoarsenCounters::default();
        let levels = coarsen(&g, &opts, &mut rng, &mut ctr);
        assert!(!levels.is_empty(), "must produce at least one level");
        let mut prev = g.nvtxs;
        for (i, lvl) in levels.iter().enumerate() {
            assert!(
                lvl.graph.nvtxs < prev,
                "level {} did not shrink: {} >= {}",
                i,
                lvl.graph.nvtxs,
                prev
            );
            prev = lvl.graph.nvtxs;
        }
        // Coarsest level is at or below floor (the loop stops on
        // the first level whose vertex count dips below floor).
        assert!(
            prev <= opts.coarsen_floor as Idx
                || levels.last().expect("non-empty").graph.nvtxs < g.nvtxs
        );
    }

    #[test]
    fn stall_keeps_first_level_that_shrank() {
        // Star K_{1,24}: center 0 with 24 degree-1 leaves. SHEM matches
        // exactly one leaf to the center, leaving 23 self-matched
        // leaves, so the first level coarsens 25 -> 24 -- a real shrink
        // that still trips the <5% "stall" branch. The two-hop fallback
        // is disabled (threshold > 1) so nothing rescues the leaves and
        // the stall branch is the only exit. The level genuinely shrank
        // and must be kept; the old code discarded it because `levels`
        // was still empty, returning an empty hierarchy. [O8]
        let mut t: Vec<(usize, usize)> = vec![(0, 0)];
        for l in 1..=24usize {
            t.push((l, l));
            t.push((0, l));
        }
        let (cp, ri) = csc_from_triples(25, &t);
        let pat = CscPattern::new(25, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let opts = MetisOptions {
            coarsen_floor: 4,
            two_hop_ratio_threshold: 10.0,
            ..Default::default()
        };
        let mut rng = SplitMix::new(1);
        let mut ctr = CoarsenCounters::default();
        let levels = coarsen(&g, &opts, &mut rng, &mut ctr);
        assert!(
            !levels.is_empty(),
            "a first level that shrank 25->24 must be kept, not discarded"
        );
        assert_eq!(levels.last().expect("non-empty").graph.nvtxs, 24);
    }

    #[test]
    fn contract_sums_parallel_edges() {
        // Two triangles sharing an edge: 0-1-2, 3-1-2. After
        // matching 0<->3, coarse edges {{0,3}, 1} and {{0,3}, 2}
        // should each have weight 2 (two fine edges collapse).
        // Build the pattern manually.
        let n = 4;
        let t = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (0, 1),
            (0, 2),
            (1, 2),
            (3, 1),
            (3, 2),
        ];
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let fine = Graph::from_csc_pattern(&pat).unwrap();
        // Force a matching 0<->3 by custom cmap.
        let cmap: Vec<Idx> = vec![0, 1, 2, 0];
        let coarse = contract(&fine, &cmap, 3);
        // Coarse vertex 0 = {0,3}: weight 2 to coarse vertex 1 (via
        // 0-1 and 3-1), weight 2 to coarse vertex 2.
        let mut got: Vec<(Idx, Idx)> = Vec::new();
        let lo = coarse.xadj[0] as usize;
        let hi = coarse.xadj[1] as usize;
        for k in lo..hi {
            got.push((coarse.adjncy[k], coarse.adjwgt[k]));
        }
        got.sort();
        assert_eq!(got, vec![(1, 2), (2, 2)]);
        // Symmetry.
        for v in 0..coarse.nvtxs {
            let lo = coarse.xadj[v as usize] as usize;
            let hi = coarse.xadj[(v + 1) as usize] as usize;
            for k in lo..hi {
                let u = coarse.adjncy[k];
                let w = coarse.adjwgt[k];
                let ulo = coarse.xadj[u as usize] as usize;
                let uhi = coarse.xadj[(u + 1) as usize] as usize;
                let mut ok = false;
                for kk in ulo..uhi {
                    if coarse.adjncy[kk] == v {
                        assert_eq!(coarse.adjwgt[kk], w, "edge-weight symmetry");
                        ok = true;
                        break;
                    }
                }
                assert!(ok, "edge {}->{} has no reverse", v, u);
            }
        }
    }
}
