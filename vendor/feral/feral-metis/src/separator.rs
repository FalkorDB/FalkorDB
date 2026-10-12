//! Convert an edge bisection to a node separator.
//!
//! Given `labels[v] ∈ {PART_A, PART_B}`, we compute a vertex set
//! `S ⊆ boundary(A) ∪ boundary(B)` such that removing `S` disconnects
//! the remaining A-vertices from the remaining B-vertices. The
//! minimum-weight such set is the minimum vertex cover of the
//! bipartite graph between A-boundary and B-boundary induced by
//! crossing edges (König's theorem).
//!
//! Algorithm:
//!
//! 1. Identify the A-boundary and B-boundary (vertices on each side
//!    incident to a crossing edge).
//! 2. Build the bipartite crossing-edge graph.
//! 3. Compute a maximum matching with Kuhn's augmenting-path
//!    algorithm (O(V·E); sufficient at our graph sizes).
//! 4. Use König's construction to extract the minimum vertex cover
//!    from the matching, and relabel those vertices as `PART_SEP`.
//!
//! References:
//! - Karypis & Kumar, *A Fast and High Quality Multilevel Scheme for
//!   Partitioning Irregular Graphs*, 1998 (§5).
//! - METIS 5.2.0 `libmetis/separator.c::ConstructMinCoverSeparator`.

use crate::fm_refine::PART_SEP;
use crate::graph::Graph;
use crate::initial_partition::{PART_A, PART_B};
use feral_ordering_core::Idx;

/// Convert an edge bisection (labels ∈ {PART_A, PART_B}) to a node
/// separator (labels ∈ {PART_A, PART_B, PART_SEP}) in place.
///
/// Returns the number of separator vertices.
pub fn construct_separator(graph: &Graph, labels: &mut [u8]) -> usize {
    let n = graph.nvtxs as usize;
    debug_assert_eq!(labels.len(), n);

    // 1. Identify boundaries.
    let mut is_bnd_a: Vec<bool> = vec![false; n];
    let mut is_bnd_b: Vec<bool> = vec![false; n];
    for v in 0..n {
        let lv = labels[v];
        if lv != PART_A && lv != PART_B {
            continue;
        }
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            if labels[u] != lv && (labels[u] == PART_A || labels[u] == PART_B) {
                if lv == PART_A {
                    is_bnd_a[v] = true;
                } else {
                    is_bnd_b[v] = true;
                }
                break;
            }
        }
    }

    // 2. Collect A-boundary and B-boundary vertices and index them.
    //    a_idx[v] = slot in left bipartite side, or -1.
    //    b_idx[v] = slot in right bipartite side, or -1.
    let mut a_list: Vec<Idx> = Vec::new();
    let mut b_list: Vec<Idx> = Vec::new();
    let mut a_idx: Vec<Idx> = vec![-1; n];
    let mut b_idx: Vec<Idx> = vec![-1; n];
    for v in 0..n {
        if is_bnd_a[v] {
            a_idx[v] = a_list.len() as Idx;
            a_list.push(v as Idx);
        }
        if is_bnd_b[v] {
            b_idx[v] = b_list.len() as Idx;
            b_list.push(v as Idx);
        }
    }
    if a_list.is_empty() || b_list.is_empty() {
        return 0;
    }

    // 3. Build bipartite adjacency: for each A-boundary vertex, list its
    //    B-boundary neighbors (by slot).
    let na = a_list.len();
    let nb = b_list.len();
    let mut adj: Vec<Vec<Idx>> = vec![Vec::new(); na];
    for (ai, &v) in a_list.iter().enumerate() {
        let vu = v as usize;
        let lo = graph.xadj[vu] as usize;
        let hi = graph.xadj[vu + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            let bi = b_idx[u];
            if bi >= 0 {
                adj[ai].push(bi);
            }
        }
    }

    // 4. Kuhn's algorithm for maximum bipartite matching.
    let mut match_a: Vec<Idx> = vec![-1; na]; // match_a[ai] = bi or -1
    let mut match_b: Vec<Idx> = vec![-1; nb]; // match_b[bi] = ai or -1
    for ai in 0..na {
        let mut visited: Vec<bool> = vec![false; nb];
        try_augment(ai as Idx, &adj, &mut match_a, &mut match_b, &mut visited);
    }

    // 5. König's construction. Starting from unmatched A-vertices,
    //    alternately walk unmatched edges (A→B) and matched edges (B→A).
    //    Let Z be the set of reachable vertices.
    //    Min vertex cover = (A \ Z) ∪ (B ∩ Z).
    let mut in_z_a: Vec<bool> = vec![false; na];
    let mut in_z_b: Vec<bool> = vec![false; nb];
    let mut stack: Vec<Idx> = Vec::new();
    for ai in 0..na {
        if match_a[ai] < 0 {
            in_z_a[ai] = true;
            stack.push(ai as Idx);
        }
    }
    while let Some(ai) = stack.pop() {
        for &bi in &adj[ai as usize] {
            if !in_z_b[bi as usize] {
                in_z_b[bi as usize] = true;
                let m = match_b[bi as usize];
                if m >= 0 && !in_z_a[m as usize] {
                    in_z_a[m as usize] = true;
                    stack.push(m);
                }
            }
        }
    }

    // 6. Mark cover vertices as separator.
    let mut sep_count: usize = 0;
    for (ai, &v) in a_list.iter().enumerate() {
        if !in_z_a[ai] {
            labels[v as usize] = PART_SEP;
            sep_count += 1;
        }
    }
    for (bi, &v) in b_list.iter().enumerate() {
        if in_z_b[bi] {
            labels[v as usize] = PART_SEP;
            sep_count += 1;
        }
    }
    sep_count
}

/// Iterative augmenting-path search used by Kuhn's algorithm.
fn try_augment(
    start: Idx,
    adj: &[Vec<Idx>],
    match_a: &mut [Idx],
    match_b: &mut [Idx],
    visited: &mut [bool],
) -> bool {
    // Recursive form is idiomatic here; recursion depth is bounded by
    // min(|A|, |B|) which is bounded by the boundary size at the
    // coarsest graph (small).
    fn dfs(
        u: Idx,
        adj: &[Vec<Idx>],
        match_a: &mut [Idx],
        match_b: &mut [Idx],
        visited: &mut [bool],
    ) -> bool {
        for &v in &adj[u as usize] {
            if visited[v as usize] {
                continue;
            }
            visited[v as usize] = true;
            if match_b[v as usize] < 0 || dfs(match_b[v as usize], adj, match_a, match_b, visited) {
                match_a[u as usize] = v;
                match_b[v as usize] = u;
                return true;
            }
        }
        false
    }
    dfs(start, adj, match_a, match_b, visited)
}

/// Verify that labels form a valid node separator: no edge connects a
/// PART_A vertex directly to a PART_B vertex.
#[cfg(test)]
pub fn is_valid_separator(graph: &Graph, labels: &[u8]) -> bool {
    for v in 0..graph.nvtxs as usize {
        let lv = labels[v];
        if lv != PART_A && lv != PART_B {
            continue;
        }
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            let lu = labels[u];
            if (lv == PART_A && lu == PART_B) || (lv == PART_B && lu == PART_A) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initial_partition::{cut_size, initial_bisect_ggp, part_weight};
    use crate::rng::SplitMix;
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

    #[test]
    fn separator_disconnects_parts_on_grid() {
        let g = grid(6, 6);
        let total: i64 = g.vwgt.iter().copied().sum();
        let mut rng = SplitMix::new(21);
        let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
        let sep_count = construct_separator(&g, &mut labels);
        assert!(sep_count > 0, "expected non-trivial separator");
        assert!(is_valid_separator(&g, &labels));
    }

    #[test]
    fn separator_empty_if_no_cut() {
        let g = grid(3, 3);
        // All on one side → no crossing edges → no separator.
        let mut labels: Vec<u8> = vec![PART_A; 9];
        let sep_count = construct_separator(&g, &mut labels);
        assert_eq!(sep_count, 0);
        assert!(is_valid_separator(&g, &labels));
    }

    #[test]
    fn separator_on_4x4_grid_balanced() {
        let g = grid(4, 4);
        // Hand-built balanced cut: left 2 cols = A, right 2 cols = B.
        let mut labels: Vec<u8> = (0..16u8)
            .map(|k| if (k % 4) < 2 { PART_A } else { PART_B })
            .collect();
        let cut = cut_size(&g, &labels);
        assert!(cut > 0);
        let sep_count = construct_separator(&g, &mut labels);
        assert!(sep_count > 0);
        assert!(is_valid_separator(&g, &labels));
        // Remaining A and B must both be non-empty (we have some
        // vertices left on each side after carving out the cover).
        let a = part_weight(&g, &labels, PART_A);
        let b = part_weight(&g, &labels, PART_B);
        assert!(a > 0 && b > 0);
    }

    #[test]
    fn min_cover_no_larger_than_lighter_boundary() {
        // Kőnig guarantees |cover| = |max matching| ≤ min(|bnd_a|, |bnd_b|).
        // So a trivial "take the lighter boundary" upper bound must hold.
        let g = grid(8, 8);
        let total: i64 = g.vwgt.iter().copied().sum();
        let mut rng = SplitMix::new(33);
        let labels_init = initial_bisect_ggp(&g, &mut rng, total / 2);
        // Count boundaries.
        let n = g.nvtxs as usize;
        let mut bnd_a = 0usize;
        let mut bnd_b = 0usize;
        for v in 0..n {
            let lv = labels_init[v];
            let lo = g.xadj[v] as usize;
            let hi = g.xadj[v + 1] as usize;
            let mut is_bnd = false;
            for k in lo..hi {
                if labels_init[g.adjncy[k] as usize] != lv {
                    is_bnd = true;
                    break;
                }
            }
            if is_bnd {
                if lv == PART_A {
                    bnd_a += 1;
                } else {
                    bnd_b += 1;
                }
            }
        }
        let mut labels = labels_init.clone();
        let sep_count = construct_separator(&g, &mut labels);
        assert!(sep_count <= bnd_a.min(bnd_b).max(1));
    }
}
