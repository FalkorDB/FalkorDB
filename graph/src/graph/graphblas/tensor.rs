//! 3D sparse tensor for multi-edge relationship storage.
//!
//! This module provides [`Tensor`], which extends the adjacency matrix model
//! to support multiple edges of the same type between the same pair of nodes.
//! While a plain adjacency matrix can only record whether an edge exists,
//! the tensor stores individual edge IDs so that each edge can carry its own
//! properties.
//!
//! ## Internal Structure
//!
//! A tensor is composed of three [`VersionedMatrix`] instances:
//!
//! ```text
//!   Tensor
//!     |
//!     |-- m  (forward adjacency)      src --> dst  (boolean)
//!     |-- mt (backward adjacency)     dst --> src  (boolean)
//!     |-- me (edge ID storage)        compound_key --> edge_id  (boolean)
//!
//!   Forward matrix (m):        Backward matrix (mt):
//!     dst: 0  1  2               src: 0  1  2
//!   src 0 [ .  T  . ]         dst 0 [ .  .  . ]
//!       1 [ .  .  . ]             1 [ T  .  T ]
//!       2 [ .  T  . ]             2 [ .  .  . ]
//!
//!   Edges: 0->1 (id=5), 0->1 (id=9), 2->1 (id=7)
//! ```
//!
//! ## Compound Key Encoding
//!
//! The edge matrix `me` stores edge IDs using a compound row key that packs
//! both source and destination node IDs into a single u64:
//!
//! ```text
//!   row = (src << 32) | dst
//!
//!   Example: edge from node 3 to node 7
//!     row = (3 << 32) | 7 = 0x0000_0003_0000_0007
//!
//!   me[row, edge_id] = true
//! ```
//!
//! This encoding allows multiple edge IDs per (src, dst) pair by storing
//! each edge ID as a separate column in the same row.
//!
//! ## Iteration
//!
//! [`Iter`] walks the forward (or backward) adjacency matrix and, for each
//! (src, dst) pair found, looks up all edge IDs from `me`. It yields
//! `(src, dst, edge_id)` triples.
//!
//! ## Use Case
//!
//! In property graphs, multiple edges of the same type can connect two nodes.
//! For example, two "TRANSFERRED" relationships between the same bank accounts
//! with different amounts and dates.

use std::collections::HashMap;

use super::{
    matrix::{Dup, Matrix, New, Remove, Set, Size, Transpose},
    serialization::{Decode, Encode, Reader, Writer},
    vector::Vector,
    versioned_matrix::{self, VersionedMatrix},
};

/// Maximum GraphBLAS index value (2^60 - 1).
#[allow(non_upper_case_globals)]
pub const GrB_INDEX_MAX: u64 = (1u64 << 60) - 1;

/// Multi-edge storage supporting multiple edges between node pairs.
///
/// Maintains three matrices for efficient traversal in both directions
/// and edge ID lookup.
pub struct Tensor {
    /// Forward adjacency matrix (src → dst)
    m: VersionedMatrix,
    /// Transpose/backward adjacency (dst → src)
    mt: VersionedMatrix,
    /// Edge ID storage keyed by (src, dst) pair
    me: VersionedMatrix,
}

impl New for Tensor {
    fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        Self {
            m: VersionedMatrix::new(nrows, ncols),
            mt: VersionedMatrix::new(ncols, nrows),
            me: VersionedMatrix::new(GrB_INDEX_MAX, GrB_INDEX_MAX),
        }
    }
}

impl Tensor {
    #[must_use]
    pub fn get(
        &self,
        src: u64,
        dest: u64,
    ) -> versioned_matrix::Iter {
        debug_assert!(u32::try_from(src).is_ok() && u32::try_from(dest).is_ok());
        let row = src << 32 | dest;
        self.me.iter(row, row)
    }

    pub fn set(
        &mut self,
        src: u64,
        dest: u64,
        id: u64,
    ) {
        self.m.set(src, dest, true);
        self.mt.set(dest, src, true);
        self.me.set(src << 32 | dest, id, true);
    }

    /// Set multiple entries, checking dm emptiness once per sub-matrix.
    pub fn set_all(
        &mut self,
        entries: impl Iterator<Item = (u64, u64, u64)>,
    ) {
        // Collect entries since we need to iterate 3 times (once per sub-matrix)
        let entries: Vec<_> = entries.collect();
        self.m
            .set_all(entries.iter().map(|&(src, dst, _)| (src, dst)));
        self.mt
            .set_all(entries.iter().map(|&(src, dst, _)| (dst, src)));
        self.me
            .set_all(entries.iter().map(|&(src, dst, id)| (src << 32 | dst, id)));
    }

    pub fn remove_all(
        &mut self,
        rels: &[(u64, u64, u64)],
    ) {
        for (id, src, dest) in rels {
            self.me.remove(src << 32 | dest, *id);
        }
        for (_, src, dest) in rels {
            if self
                .me
                .iter(src << 32 | dest, src << 32 | dest)
                .next()
                .is_none()
            {
                self.m.remove(*src, *dest);
                self.mt.remove(*dest, *src);
            }
        }
    }

    /// Bulk-remove all edges whose `(src, dst)` pair appears in `mask`.
    ///
    /// `mask_t` must be the transpose of `mask`.
    /// This removes ALL edges at matching `(src, dst)` pairs — suitable for
    /// implicit (cascade) deletion where a node and all its edges are removed.
    pub fn clear_elements(
        &mut self,
        mask: &Matrix,
        mask_t: &Matrix,
    ) {
        // Build me_mask: me uses compound key (src<<32|dst) as row, edge_id as col
        let mut me_mask = Matrix::new(GrB_INDEX_MAX, GrB_INDEX_MAX);
        for (src, dst) in mask.iter(0, u64::MAX) {
            let compound = src << 32 | dst;
            for (_, edge_id) in self.me.iter(compound, compound) {
                me_mask.set(compound, edge_id, true);
            }
        }

        // Bulk-remove from edge ID matrix
        if me_mask.nvals() > 0 {
            self.me.remove_mask(&me_mask);
        }

        // Bulk-remove from forward and backward adjacency
        self.m.remove_mask(mask);
        self.mt.remove_mask(mask_t);
    }

    pub fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) {
        self.m.resize(nrows, ncols);
        self.mt.resize(ncols, nrows);
    }

    /// Rebuild the backward matrix as the transpose of the forward matrix.
    pub fn rebuild_backward(&mut self) {
        self.mt = self.m.transpose();
    }

    #[must_use]
    pub fn dup(&self) -> Self {
        Self {
            m: self.m.dup(),
            mt: self.mt.dup(),
            me: self.me.dup(),
        }
    }

    #[must_use]
    pub const fn matrix(&self) -> &VersionedMatrix {
        &self.m
    }

    /// Total number of edges (including multi-edges between the same node pair).
    #[must_use]
    pub fn edge_count(&self) -> u64 {
        self.me.nvals()
    }

    #[must_use]
    pub fn iter(
        &self,
        min_row: u64,
        max_row: u64,
        transpose: bool,
    ) -> Iter<'_> {
        Iter::new(self, min_row, max_row, transpose)
    }

    /// Whether this tensor has any (src, dst) pair with more than one edge.
    #[must_use]
    pub fn has_multi_edge(&self) -> bool {
        self.m.nvals() != self.me.nvals()
    }

    pub fn wait(&mut self) {
        self.m.wait();
        self.mt.wait();
        self.me.wait();
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.m.memory_usage() + self.mt.memory_usage() + self.me.memory_usage()
    }
}

/// MSB flag used by C FalkorDB to indicate multi-edge entries in the
/// UINT64 forward matrix.
const MSB_MASK: u64 = 1u64 << 63;

impl Encode<19> for Tensor {
    fn encode(
        &self,
        w: &mut dyn Writer,
    ) {
        // Build a UINT64 forward matrix for C compatibility.
        // Single-edge (src,dst): cell = edge_id
        // Multi-edge (src,dst): cell = edge_count | MSB_MASK
        let (m, dp) = self.m.extract_m_dp();

        let mut uint64_m = Matrix::new_uint64(m.nrows(), m.ncols());
        let mut uint64_dp = Matrix::new_uint64(dp.nrows(), dp.ncols());
        // Track multi-edge (src, dst) pairs per sub-matrix for tensor section
        let mut multi_edge_m: Vec<(u64, u64)> = Vec::new();
        let mut multi_edge_dp: Vec<(u64, u64)> = Vec::new();

        // Pre-build edge-ID map in a single pass over `me` to avoid creating
        // one GxB_Iterator per edge pair in the inner loop below.
        // me rows = compound_key (src<<32|dst), cols = edge_id.
        let mut edge_id_map: HashMap<u64, Vec<u64>> = HashMap::new();
        for (compound_key, edge_id) in self.me.iter(0, u64::MAX) {
            edge_id_map.entry(compound_key).or_default().push(edge_id);
        }

        for (matrix, uint64_matrix, multi_edges) in [
            (&m, &mut uint64_m, &mut multi_edge_m),
            (&dp, &mut uint64_dp, &mut multi_edge_dp),
        ] {
            for (src, dst) in matrix.iter(0, u64::MAX) {
                let compound_key = (src << 32) | dst;
                let edge_ids = edge_id_map
                    .get(&compound_key)
                    .map_or(&[][..], |v| v.as_slice());

                if edge_ids.len() == 1 {
                    // Single edge: store edge ID directly
                    uint64_matrix.set_uint64(src, dst, edge_ids[0]);
                } else {
                    // Multi-edge: store count with MSB set
                    uint64_matrix.set_uint64(src, dst, edge_ids.len() as u64 | MSB_MASK);
                    multi_edges.push((src, dst));
                }
            }
        }

        // Encode the UINT64 forward matrix (as a VersionedMatrix: m, dp, dm)
        let dm = Matrix::new_uint64(m.nrows(), m.ncols()); // empty delta-minus
        uint64_m.encode(w);
        uint64_dp.encode(w);
        dm.encode(w);

        let total = self.edge_count();
        w.write_unsigned(total);

        if total == 0 {
            return;
        }

        // Tensor section: only multi-edge pairs (rare in practice).
        // Re-use the already-built edge_id_map instead of calling me.iter() again.
        let mut v = Vector::<u64>::new(GrB_INDEX_MAX);
        for multi_edges in [&multi_edge_m, &multi_edge_dp] {
            w.write_unsigned(multi_edges.len() as u64);
            for &(src, dst) in multi_edges {
                let compound_key = (src << 32) | dst;
                v.clear();

                if let Some(ids) = edge_id_map.get(&compound_key) {
                    for (idx, &edge_id) in ids.iter().enumerate() {
                        v.set(idx as u64, edge_id);
                    }
                }

                w.write_unsigned(src);
                w.write_unsigned(dst);
                v.encode(w);
            }
        }
    }
}

impl Decode<19> for Tensor {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        let forward = VersionedMatrix::decode(r)?;
        let mut edges = VersionedMatrix::new(GrB_INDEX_MAX, GrB_INDEX_MAX);

        // C FalkorDB stores edge IDs as UINT64 values in the forward matrix.
        // Single-edge entries (MSB not set) hold the edge ID directly.
        // Multi-edge entries (MSB set) are stored in the tensor section below.
        // Iterate entries, extract single-edge IDs, and rebuild as BOOL.
        let forward = if forward.is_uint64() {
            let mut bool_forward = VersionedMatrix::new(forward.nrows(), forward.ncols());
            for (src, dst, value) in forward.uint64_iter() {
                bool_forward.set(src, dst, true);
                if value & MSB_MASK == 0 {
                    // Single-edge: value is the edge ID
                    let compound_key = (src << 32) | dst;
                    edges.set(compound_key, value, true);
                }
            }
            bool_forward
        } else {
            forward
        };

        let total_tensor_count = r.read_unsigned()?;
        if total_tensor_count > 0 {
            // TM tensors (base), then TDP tensors (delta-plus)
            for _ in 0..2 {
                let count = r.read_unsigned()?;
                for _ in 0..count {
                    let src = r.read_unsigned()?;
                    let dst = r.read_unsigned()?;
                    let v = Vector::<u64>::decode(r)?;
                    let compound_key = (src << 32) | dst;
                    for (_, edge_id) in v.iter() {
                        edges.set(compound_key, edge_id, true);
                    }
                }
            }
        }

        let backward = VersionedMatrix::new(0, 0);
        Ok(Self {
            m: forward,
            mt: backward,
            me: edges,
        })
    }
}

pub struct Iter<'a> {
    t: &'a Tensor,
    mit: versioned_matrix::Iter,
    vit: Option<versioned_matrix::Iter>,
    transpose: bool,
    src: u64,
    dest: u64,
}

impl<'a> Iter<'a> {
    fn new(
        t: &'a Tensor,
        min_row: u64,
        max_row: u64,
        transpose: bool,
    ) -> Self {
        Self {
            t,
            mit: if transpose {
                t.mt.iter(min_row, max_row)
            } else {
                t.m.iter(min_row, max_row)
            },
            vit: None,
            transpose,
            src: 0,
            dest: 0,
        }
    }
}

impl Iterator for Iter<'_> {
    type Item = (u64, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(vit) = &mut self.vit {
            if let Some((_, id)) = vit.next() {
                return Some((self.src, self.dest, id));
            }
            self.vit = None;
        }

        if let Some((src, dest)) = self.mit.next() {
            if self.transpose {
                self.src = dest;
                self.dest = src;
            } else {
                self.src = src;
                self.dest = dest;
            }
            let row = self.src << 32 | self.dest;
            self.vit = Some(self.t.me.iter(row, row));
            return self.next();
        }

        None
    }
}
