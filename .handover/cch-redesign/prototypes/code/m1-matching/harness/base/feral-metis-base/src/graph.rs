//! CSR graph representation for the multilevel partitioning pipeline.
//!
//! Mirrors METIS 5.2.0's `graph_t` (see `libmetis/struct.h`) with
//! Rust-idiomatic `Vec` storage. Diagonal entries of the input
//! matrix (self-loops in the graph) are dropped — they carry no
//! information for fill-reducing ordering and would otherwise need
//! to be stripped in every downstream kernel.
//!
//! Indices are held as `Idx` end-to-end: it matches the crate
//! contract, the METIS reference, and keeps cache density high for
//! the partitioning kernels without touching `usize`-shaped
//! `Vec<T>` storage.

use feral_ordering_core::Idx;
use feral_ordering_core::{CscPattern, OrderingError};

/// CSR graph for partitioning.
///
/// All arrays are `Idx`-indexed. Adjacency lists are **not**
/// guaranteed sorted: METIS's kernels don't require sortedness, and
/// enforcing it would be a constant-factor cost in every coarsening
/// level. `from_csc_pattern` happens to emit them sorted because it
/// reads a CSC with sorted row indices, but callers must not rely
/// on this after coarsening.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices. Must equal `xadj.len() - 1` and fit in `Idx`.
    pub nvtxs: Idx,
    /// Adjacency offsets, length `nvtxs + 1`. `xadj[0] == 0`,
    /// non-decreasing. `xadj[nvtxs]` equals `adjncy.len()` and
    /// therefore `2 * |E|` for undirected graphs without self-loops.
    pub xadj: Vec<Idx>,
    /// Neighbor lists, length `2 * |E|`. Each undirected edge `{u,v}`
    /// appears twice (once in `u`'s list, once in `v`'s).
    pub adjncy: Vec<Idx>,
    /// Vertex weights, length `nvtxs`. Default: all 1.
    pub vwgt: Vec<Idx>,
    /// Edge weights, aligned with `adjncy`. Default: all 1.
    pub adjwgt: Vec<Idx>,
}

impl Graph {
    /// Build a `Graph` from a full-symmetric `CscPattern`.
    ///
    /// Diagonal entries are dropped. The input is assumed to be
    /// already structurally symmetric per the contract (debug-
    /// asserted elsewhere). Row indices within each CSC column are
    /// sorted ascending — `CscPattern::new` enforces this — so any
    /// duplicates are adjacent and a running "last seen" check drops
    /// all of them without needing a hash set.
    ///
    /// Complexity: `O(nnz)` time, `O(nnz + n)` space.
    pub fn from_csc_pattern(pattern: &CscPattern<'_>) -> Result<Self, OrderingError> {
        let n = pattern.n;
        if n > Idx::MAX as usize {
            return Err(OrderingError::IndexOverflow);
        }
        let nvtxs = n as Idx;
        let mut xadj: Vec<Idx> = Vec::with_capacity(n + 1);
        let mut adjncy: Vec<Idx> = Vec::with_capacity(pattern.row_idx.len());
        xadj.push(0);
        for j in 0..n {
            let lo = pattern.col_ptr[j] as usize;
            let hi = pattern.col_ptr[j + 1] as usize;
            if hi > pattern.row_idx.len() || lo > hi {
                return Err(OrderingError::MalformedInput);
            }
            let jj = j as Idx;
            let mut last: Idx = -1;
            for &r in &pattern.row_idx[lo..hi] {
                if r == jj {
                    // drop diagonal
                    continue;
                }
                if r < 0 || r >= nvtxs {
                    return Err(OrderingError::MalformedInput);
                }
                if r == last {
                    // drop duplicate; rows are sorted (CscPattern::new enforces
                    // it), so duplicates are adjacent and this catches them all
                    continue;
                }
                adjncy.push(r);
                last = r;
            }
            if adjncy.len() > Idx::MAX as usize {
                return Err(OrderingError::IndexOverflow);
            }
            xadj.push(adjncy.len() as Idx);
        }
        let vwgt = vec![1 as Idx; n];
        let adjwgt = vec![1 as Idx; adjncy.len()];
        Ok(Graph {
            nvtxs,
            xadj,
            adjncy,
            vwgt,
            adjwgt,
        })
    }

    /// Number of undirected edges (each stored twice in `adjncy`).
    pub fn nedges(&self) -> usize {
        self.adjncy.len() / 2
    }

    /// Degree of vertex `v`.
    pub fn degree(&self, v: Idx) -> Idx {
        self.xadj[(v + 1) as usize] - self.xadj[v as usize]
    }

    /// Borrow the adjacency slice of vertex `v`.
    pub fn neighbors(&self, v: Idx) -> &[Idx] {
        let lo = self.xadj[v as usize] as usize;
        let hi = self.xadj[(v + 1) as usize] as usize;
        &self.adjncy[lo..hi]
    }

    /// Borrow the edge-weight slice aligned with [`Self::neighbors`].
    pub fn edge_weights(&self, v: Idx) -> &[Idx] {
        let lo = self.xadj[v as usize] as usize;
        let hi = self.xadj[(v + 1) as usize] as usize;
        &self.adjwgt[lo..hi]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    // -- small pattern generators (full symmetric, diagonal included) --

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
        let mut col_ptr: Vec<Idx> = Vec::with_capacity(n + 1);
        col_ptr.push(0);
        let mut row_idx: Vec<Idx> = Vec::new();
        for col in &cols {
            for &r in col {
                row_idx.push(r);
            }
            col_ptr.push(row_idx.len() as Idx);
        }
        (col_ptr, row_idx)
    }

    fn diag(n: usize) -> (Vec<Idx>, Vec<Idx>) {
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i));
        }
        csc_from_triples(n, &t)
    }

    fn arrow(n: usize) -> (Vec<Idx>, Vec<Idx>) {
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i));
        }
        for i in 1..n {
            t.push((0, i));
        }
        csc_from_triples(n, &t)
    }

    fn tridiag(n: usize) -> (Vec<Idx>, Vec<Idx>) {
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i));
            if i + 1 < n {
                t.push((i, i + 1));
            }
        }
        csc_from_triples(n, &t)
    }

    fn grid_2d(m: usize, n: usize) -> (Vec<Idx>, Vec<Idx>) {
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
        csc_from_triples(total, &t)
    }

    // -- invariant helpers --

    fn assert_graph_invariants(g: &Graph) {
        assert_eq!(g.xadj.len(), g.nvtxs as usize + 1, "xadj length");
        assert_eq!(g.xadj[0], 0, "xadj[0]");
        assert_eq!(
            g.xadj[g.nvtxs as usize] as usize,
            g.adjncy.len(),
            "xadj[nvtxs] == adjncy.len()"
        );
        for i in 0..g.nvtxs as usize {
            assert!(g.xadj[i] <= g.xadj[i + 1], "xadj non-decreasing at {}", i);
        }
        assert_eq!(g.vwgt.len(), g.nvtxs as usize, "vwgt length");
        assert_eq!(g.adjwgt.len(), g.adjncy.len(), "adjwgt length");
        for (i, &r) in g.adjncy.iter().enumerate() {
            assert!(r >= 0 && r < g.nvtxs, "adjncy[{}]={} OOB", i, r);
        }
        // No self-loops.
        for v in 0..g.nvtxs {
            for &u in g.neighbors(v) {
                assert_ne!(u, v, "self-loop at {}", v);
            }
        }
        // Structural symmetry: every edge (u,v) has a matching (v,u).
        for v in 0..g.nvtxs {
            for &u in g.neighbors(v) {
                assert!(
                    g.neighbors(u).contains(&v),
                    "asymmetric edge: {} -> {} but not back",
                    v,
                    u
                );
            }
        }
        // adjncy length is 2 * |E| because every edge appears twice.
        assert_eq!(g.nedges() * 2, g.adjncy.len(), "edge-count parity");
    }

    // -- tests --

    #[test]
    fn diag_has_no_edges() {
        let (cp, ri) = diag(4);
        let pat = CscPattern::new(4, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_graph_invariants(&g);
        assert_eq!(g.nvtxs, 4);
        assert_eq!(g.nedges(), 0);
        for v in 0..4 {
            assert_eq!(g.degree(v), 0);
        }
    }

    #[test]
    fn arrow_5_degree_sequence() {
        // Arrow(5): hub 0 connects to {1,2,3,4}; leaves each connect only to 0.
        let (cp, ri) = arrow(5);
        let pat = CscPattern::new(5, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_graph_invariants(&g);
        assert_eq!(g.nvtxs, 5);
        assert_eq!(g.nedges(), 4);
        assert_eq!(g.degree(0), 4);
        for v in 1..5 {
            assert_eq!(g.degree(v), 1);
            assert_eq!(g.neighbors(v), &[0]);
        }
        let mut hub: Vec<Idx> = g.neighbors(0).to_vec();
        hub.sort();
        assert_eq!(hub, vec![1, 2, 3, 4]);
    }

    #[test]
    fn tridiag_10_degree_sequence() {
        // Tridiagonal(10): endpoints have degree 1, interiors degree 2.
        let (cp, ri) = tridiag(10);
        let pat = CscPattern::new(10, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_graph_invariants(&g);
        assert_eq!(g.nvtxs, 10);
        assert_eq!(g.nedges(), 9);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(9), 1);
        for v in 1..9 {
            assert_eq!(g.degree(v), 2);
        }
    }

    #[test]
    fn grid_3x3_structure() {
        // 3x3 grid: corners=2, edges=3, center=4. Edges = 12.
        let (cp, ri) = grid_2d(3, 3);
        let pat = CscPattern::new(9, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_graph_invariants(&g);
        assert_eq!(g.nvtxs, 9);
        assert_eq!(g.nedges(), 12);
        let mut by_degree = [0u32; 5];
        for v in 0..9 {
            by_degree[g.degree(v) as usize] += 1;
        }
        assert_eq!(by_degree[2], 4, "four corners with degree 2");
        assert_eq!(by_degree[3], 4, "four edge-centres with degree 3");
        assert_eq!(by_degree[4], 1, "one interior with degree 4");
    }

    #[test]
    fn weights_default_to_one() {
        let (cp, ri) = tridiag(5);
        let pat = CscPattern::new(5, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert!(g.vwgt.iter().all(|&w| w == 1));
        assert!(g.adjwgt.iter().all(|&w| w == 1));
    }

    #[test]
    fn empty_graph_is_legal() {
        // n=0: no columns. Graph has zero vertices and zero edges.
        let cp: Vec<Idx> = vec![0];
        let ri: Vec<Idx> = vec![];
        let pat = CscPattern::new(0, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_eq!(g.nvtxs, 0);
        assert_eq!(g.nedges(), 0);
        assert_eq!(g.xadj, vec![0]);
    }

    #[test]
    fn diagonal_entries_are_dropped() {
        // Diagonal-only pattern results in zero edges even though nnz > 0.
        let (cp, ri) = diag(3);
        assert_eq!(ri.len(), 3); // three diagonal entries
        let pat = CscPattern::new(3, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_eq!(g.adjncy.len(), 0);
    }
}
