//! Initial bisection at the coarsest level.
//!
//! Implements two simple schemes matched to METIS 5.2.0's
//! `Init2WayPartition`:
//!
//! - **GGP** (Greedy Graph Growing): seed one vertex in part 0,
//!   repeatedly pull in the boundary vertex whose inclusion minimises
//!   the added cut. Stops when part-0 weight reaches half the total.
//! - **Random BFS**: seed one vertex, assign vertices to part 0 in
//!   BFS order until the half-weight mark is reached.
//!
//! Each call runs one trial. The multi-trial driver (with post-FM
//! scoring) lives in `node_nd.rs` where the FM module is already in
//! scope.

use crate::graph::Graph;
use crate::rng::SplitMix;
use feral_ordering_core::Idx;

pub const PART_A: u8 = 0;
pub const PART_B: u8 = 1;

/// Compute the edge-cut of a bisection.
pub fn cut_size(graph: &Graph, labels: &[u8]) -> Idx {
    debug_assert_eq!(labels.len(), graph.nvtxs as usize);
    let mut cut: i64 = 0;
    for v in 0..graph.nvtxs as usize {
        let lv = labels[v];
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            if u > v && labels[u] != lv {
                cut += graph.adjwgt[k];
            }
        }
    }
    cut as Idx
}

/// Total vertex weight of part `p`.
pub fn part_weight(graph: &Graph, labels: &[u8], p: u8) -> i64 {
    let mut s: i64 = 0;
    for (v, &l) in labels.iter().enumerate() {
        if l == p {
            s += graph.vwgt[v];
        }
    }
    s
}

/// GGP trial. Grows part 0 from a random seed until its weight
/// reaches `target_a`. All unvisited vertices go to part 1.
pub fn initial_bisect_ggp(graph: &Graph, rng: &mut SplitMix, target_a: i64) -> Vec<u8> {
    let n = graph.nvtxs as usize;
    let mut labels: Vec<u8> = vec![PART_B; n];
    if n == 0 {
        return labels;
    }
    let seed = rng.gen_range(n as u64) as usize;
    labels[seed] = PART_A;
    let mut a_weight: i64 = graph.vwgt[seed];

    // GGP gain of pulling a boundary vertex v (still in B) into A is
    //   (edges to A) - (edges to B),
    // i.e. the reduction in cut. Since every neighbour of an in-B
    // vertex is in A or B, edges_to_B(v) = wtot[v] - edges_to_A(v), so
    // the gain equals 2*edges_to_A(v) - wtot[v]. We accumulate the
    // edges-to-A term in `gain` and apply the `- wtot` correction in
    // the selection scan below; selecting on edges-to-A alone would
    // bias toward high-degree vertices and inflate the cut.
    //
    // No full PQ: pick the max-gain vertex via a linear scan over the
    // boundary set. On coarsest graphs (n <= coarsen_floor = 120) this
    // is fine.
    let mut gain: Vec<Idx> = vec![0; n];
    let mut in_boundary: Vec<bool> = vec![false; n];
    let mut boundary: Vec<Idx> = Vec::new();

    // Per-vertex total incident edge weight (constant). Turns the
    // edges-to-A accumulator into the true GGP gain in the scan below.
    let mut wtot: Vec<Idx> = vec![0; n];
    for (v, wt) in wtot.iter_mut().enumerate() {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            *wt = wt.saturating_add(graph.adjwgt[k]);
        }
    }

    // Add v's still-in-B neighbours to the boundary and credit each
    // with the v–u edge weight on the A side.
    let push_neighbors = |v: usize,
                          graph: &Graph,
                          labels: &[u8],
                          gain: &mut [Idx],
                          in_boundary: &mut [bool],
                          boundary: &mut Vec<Idx>| {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            if labels[u] != PART_B {
                continue;
            }
            // v is in A, u is in B: edge (v,u) is one more edge on u's
            // A side.
            gain[u] = gain[u].saturating_add(graph.adjwgt[k]);
            if !in_boundary[u] {
                in_boundary[u] = true;
                boundary.push(u as Idx);
            }
        }
    };

    push_neighbors(
        seed,
        graph,
        &labels,
        &mut gain,
        &mut in_boundary,
        &mut boundary,
    );

    while a_weight < target_a && !boundary.is_empty() {
        // Pick best-gain boundary vertex.
        let mut best_i: usize = 0;
        let mut best_g: i64 = i64::MIN;
        for (i, &v) in boundary.iter().enumerate() {
            if labels[v as usize] != PART_B {
                continue;
            }
            // True GGP gain: edges_to_A - edges_to_B = 2*edges_to_A - wtot.
            let g = 2 * gain[v as usize] as i64 - wtot[v as usize] as i64;
            if g > best_g {
                best_g = g;
                best_i = i;
            }
        }
        let v = boundary.swap_remove(best_i);
        let vu = v as usize;
        if labels[vu] != PART_B {
            continue;
        }
        labels[vu] = PART_A;
        in_boundary[vu] = false;
        a_weight += graph.vwgt[vu];
        // Update neighbors' gains: v was in B, now in A; each
        // B-neighbor u of v sees +adjwgt(v,u) toward A.
        push_neighbors(
            vu,
            graph,
            &labels,
            &mut gain,
            &mut in_boundary,
            &mut boundary,
        );
    }
    labels
}

/// Random BFS trial. Grows part 0 in BFS order from a random seed
/// until its weight reaches `target_a`.
pub fn initial_bisect_bfs(graph: &Graph, rng: &mut SplitMix, target_a: i64) -> Vec<u8> {
    let n = graph.nvtxs as usize;
    let mut labels: Vec<u8> = vec![PART_B; n];
    if n == 0 {
        return labels;
    }
    let seed = rng.gen_range(n as u64) as usize;
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    queue.push_back(seed);
    let mut visited: Vec<bool> = vec![false; n];
    visited[seed] = true;
    let mut a_weight: i64 = 0;
    while let Some(v) = queue.pop_front() {
        if a_weight >= target_a {
            break;
        }
        labels[v] = PART_A;
        a_weight += graph.vwgt[v];
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            if !visited[u] {
                visited[u] = true;
                queue.push_back(u);
            }
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coarsen::{coarsen_level, CoarsenCounters};
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

    fn total_weight(graph: &Graph) -> i64 {
        graph.vwgt.iter().copied().sum()
    }

    #[test]
    fn cut_size_on_known_bisection() {
        let g = grid(2, 2); // four vertices, four edges
                            // Split into {0,1} and {2,3}. Cut edges: (0,2) and (1,3) → 2.
        let labels = vec![PART_A, PART_A, PART_B, PART_B];
        assert_eq!(cut_size(&g, &labels), 2);
        // All together → zero cut.
        let zero = vec![PART_A; 4];
        assert_eq!(cut_size(&g, &zero), 0);
    }

    #[test]
    fn ggp_produces_valid_bisection() {
        let g = grid(6, 6);
        let total = total_weight(&g);
        let target = total / 2;
        let mut rng = SplitMix::new(1);
        let labels = initial_bisect_ggp(&g, &mut rng, target);
        assert_eq!(labels.len(), g.nvtxs as usize);
        let a = part_weight(&g, &labels, PART_A);
        let b = part_weight(&g, &labels, PART_B);
        assert_eq!(a + b, total);
        assert!(
            a > 0 && b > 0,
            "both parts non-empty (got a={}, b={})",
            a,
            b
        );
        // GGP stops at the first vertex that tips a_weight past target,
        // so |a - target| is bounded by max vertex weight (1 here).
        assert!((a - target).abs() <= 1, "GGP weight balance: a={}", a);
    }

    #[test]
    fn bfs_produces_valid_bisection() {
        let g = grid(6, 6);
        let total = total_weight(&g);
        let target = total / 2;
        let mut rng = SplitMix::new(3);
        let labels = initial_bisect_bfs(&g, &mut rng, target);
        let a = part_weight(&g, &labels, PART_A);
        let b = part_weight(&g, &labels, PART_B);
        assert_eq!(a + b, total);
        assert!(a > 0 && b > 0);
        assert!((a - target).abs() <= 1, "BFS weight balance: a={}", a);
    }

    #[test]
    fn ggp_on_coarsened_graph_is_balanced() {
        // Exercise the coarsening + bisection sequence end-to-end
        // on a small grid. The coarse graph still has to accept a
        // balanced bisection.
        let g = grid(8, 8);
        let mut rng = SplitMix::new(5);
        let mut ctr = CoarsenCounters::default();
        let cg = coarsen_level(&g, &mut rng, 0.85, &mut ctr);
        let total = total_weight(&cg.graph);
        let target = total / 2;
        let labels = initial_bisect_ggp(&cg.graph, &mut rng, target);
        let a = part_weight(&cg.graph, &labels, PART_A);
        let b = part_weight(&cg.graph, &labels, PART_B);
        assert_eq!(a + b, total);
        assert!(a > 0 && b > 0);
    }

    #[test]
    fn ggp_is_deterministic() {
        let g = grid(5, 5);
        let total = total_weight(&g);
        let target = total / 2;
        let mut r1 = SplitMix::new(11);
        let mut r2 = SplitMix::new(11);
        let a = initial_bisect_ggp(&g, &mut r1, target);
        let b = initial_bisect_ggp(&g, &mut r2, target);
        assert_eq!(a, b);
    }

    #[test]
    fn ggp_uses_true_gain_not_edges_to_a() {
        // Weighted graph that separates the *true* GGP gain
        // (edges_to_A - edges_to_B) from edges-to-A alone. Selecting on
        // edges-to-A would pull in a high-degree vertex that inflates
        // the cut; the documented GGP gain must not.
        //
        //   index 0 = s  (seed; SplitMix seed 1 makes gen_range(7) == 0)
        //   index 1 = w : s--w weight 3                 (W[w] = 3)
        //   index 2 = u : s--u weight 5, u--b1..b4 wt 1  (W[u] = 9)
        //   index 3..6  = b1..b4 : each only u--bi weight 1
        //
        // After seeding s the boundary is {w, u}:
        //   edges_to_A:  w = 3, u = 5  -> max-edges-to-A picks u (wrong).
        //   true gain (2*edges_to_A - W): w = 6-3 = 3, u = 10-9 = 1
        //     -> GGP must pick w. Pulling u gives cut 7; w gives cut 5.
        let g = Graph {
            nvtxs: 7,
            xadj: vec![0, 2, 3, 8, 9, 10, 11, 12],
            adjncy: vec![1, 2, 0, 0, 3, 4, 5, 6, 2, 2, 2, 2],
            adjwgt: vec![3, 5, 3, 5, 1, 1, 1, 1, 1, 1, 1, 1],
            vwgt: vec![1; 7],
        };
        let mut rng = SplitMix::new(1);
        // target_a = 2: seed (weight 1) plus exactly one pulled vertex.
        let labels = initial_bisect_ggp(&g, &mut rng, 2);
        assert_eq!(labels[0], PART_A, "seed s must be in A");
        assert_eq!(
            labels[1], PART_A,
            "true GGP pulls low-degree w (gain 3) into A, not high-degree u (gain 1)"
        );
        assert_eq!(
            labels[2], PART_B,
            "high-degree u must stay in B under the true GGP gain"
        );
        assert_eq!(
            cut_size(&g, &labels),
            5,
            "correct GGP cut is 5, not the buggy 7"
        );
    }
}
