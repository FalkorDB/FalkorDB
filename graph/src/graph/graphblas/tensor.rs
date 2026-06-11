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

use rustc_hash::FxHashSet;

use super::{
    matrix::{Dup, Matrix, New, Remove, Set, Size, Transpose},
    serialization::{Decode, Encode, Reader, Writer},
    vector::Vector,
    versioned_matrix::{self, VersionedMatrix},
};

/// Maximum GraphBLAS index value (2^60 - 1).
#[allow(non_upper_case_globals)]
pub const GrB_INDEX_MAX: u64 = (1u64 << 60) - 1;

/// Pack a `(src, dst)` node-id pair into the compound row key used by the
/// edge-id matrix `me`.
///
/// The encoding `(src << 32) | dst` reserves 32 bits for each side, so both
/// values must fit in a `u32`. We check this unconditionally (not just under
/// `debug_assert!`) because silent truncation would corrupt the key and
/// conflate edges between different node pairs.
#[inline]
#[must_use]
pub fn compound_key(
    src: u64,
    dst: u64,
) -> u64 {
    assert!(
        u32::try_from(src).is_ok() && u32::try_from(dst).is_ok(),
        "Tensor compound key overflow: src={src}, dst={dst} (each must fit in u32)",
    );
    (src << 32) | dst
}

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
        let row = compound_key(src, dest);
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
        self.me.set(compound_key(src, dest), id, true);
    }

    /// Set entries from parallel slices, updating all three sub-matrices.
    pub fn set_all_from_slices(
        &mut self,
        srcs: &[u64],
        dsts: &[u64],
        ids: &[u64],
    ) {
        debug_assert_eq!(srcs.len(), dsts.len());
        debug_assert_eq!(srcs.len(), ids.len());
        if srcs.is_empty() {
            return;
        }
        self.m
            .set_all(srcs.iter().copied().zip(dsts.iter().copied()));
        self.mt
            .set_all(dsts.iter().copied().zip(srcs.iter().copied()));
        let compound_iter = srcs
            .iter()
            .zip(dsts.iter())
            .zip(ids.iter())
            .map(|((&s, &d), &id)| (compound_key(s, d), id));
        self.me.set_all(compound_iter);
    }

    /// Bulk-remove specific edges from this tensor.
    ///
    /// Each entry in `rels` is `(edge_id, src, dst)`.
    /// Returns the list of `(src, dst)` pairs that became completely empty
    /// in this tensor (no remaining edges of this type between those nodes).
    pub fn remove_all(
        &mut self,
        rels: &[(u64, u64, u64)],
    ) -> Vec<(u64, u64)> {
        if rels.is_empty() {
            return Vec::new();
        }

        // Build me_mask arrays for bulk GraphBLAS build
        let mut me_rows = Vec::with_capacity(rels.len());
        let mut me_cols = Vec::with_capacity(rels.len());
        for &(id, src, dest) in rels {
            me_rows.push(compound_key(src, dest));
            me_cols.push(id);
        }
        let mut me_mask = Matrix::new(GrB_INDEX_MAX, GrB_INDEX_MAX);
        me_mask.build_bool(&me_rows, &me_cols);

        // Check for multi-edges BEFORE removal
        let no_multi_edges = !self.has_multi_edge();

        // Single bulk GraphBLAS operation
        self.me.remove_mask(&me_mask);

        if no_multi_edges {
            // Non-multi-edge: each pair had exactly one edge, so all pairs
            // with a removed edge are now empty. Build m/mt masks directly
            // from rels without an intermediate FxHashSet.
            let nrows = self.m.nrows();
            let ncols = self.m.ncols();
            let mut m_rows = Vec::with_capacity(rels.len());
            let mut m_cols = Vec::with_capacity(rels.len());
            let mut mt_rows = Vec::with_capacity(rels.len());
            let mut mt_cols = Vec::with_capacity(rels.len());
            for &(_, src, dst) in rels {
                m_rows.push(src);
                m_cols.push(dst);
                mt_rows.push(dst);
                mt_cols.push(src);
            }
            let mut m_mask = Matrix::new(nrows, ncols);
            m_mask.build_bool(&m_rows, &m_cols);
            let mut mt_mask = Matrix::new(ncols, nrows);
            mt_mask.build_bool(&mt_rows, &mt_cols);
            self.m.remove_mask(&m_mask);
            self.mt.remove_mask(&mt_mask);
            return rels.iter().map(|&(_, src, dst)| (src, dst)).collect();
        }

        // Slow path for multi-edge tensors: need FxHashSet to dedup pairs.
        let mut pairs: FxHashSet<(u64, u64)> = FxHashSet::default();
        for &(_, src, dest) in rels {
            pairs.insert((src, dest));
        }
        let mut emptied = Vec::new();
        for (src, dst) in pairs {
            let key = compound_key(src, dst);
            if self.me.iter(key, key).next().is_none() {
                self.m.remove(src, dst);
                self.mt.remove(dst, src);
                emptied.push((src, dst));
            }
        }
        emptied
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

    /// Transposed/backward pair-level adjacency (dst → src).
    #[must_use]
    pub const fn matrix_t(&self) -> &VersionedMatrix {
        &self.mt
    }

    /// Iterate the edge-id matrix (`me`) keyed by `compound_key(src, dst)`.
    /// Hot-loop callers build a persistent iterator and `seek` to a specific
    /// (src, dst) per row, avoiding the per-pair iterator allocation that
    /// `get(src, dst)` does.
    #[must_use]
    pub fn edge_iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> versioned_matrix::Iter {
        self.me.iter(min_row, max_row)
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

    /// Wait on all matrices for fork safety (takes &self, not &mut self).
    pub fn wait_all(&self) {
        self.m.wait_all();
        self.mt.wait_all();
        self.me.wait_all();
    }

    /// Returns true if every internal matrix has no pending GraphBLAS
    /// operations queued.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        self.m.is_synced() && self.mt.is_synced() && self.me.is_synced()
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
    #[allow(clippy::similar_names)]
    fn encode(
        &self,
        w: &mut dyn Writer,
    ) {
        let nrows = self.m.nrows();
        let ncols = self.m.ncols();

        // Compute both nvals at once to avoid redundant wait()/nvals() FFI calls.
        let m_nvals = self.m.nvals();
        let me_nvals = self.me.nvals();
        let has_multi = m_nvals != me_nvals;
        let total = me_nvals;

        // Reusable empty UINT64 matrix — shared across the 2 empty slots.
        let empty = Matrix::new_uint64(nrows, ncols);

        let (multi_edge_m, multi_edge_dp, edge_id_map) = if has_multi {
            let (m, dp) = self.m.extract_m_dp();
            let (me_rows, me_cols) = self.me.extract_all_tuples();
            let mut edge_id_map: HashMap<u64, Vec<u64>> = HashMap::with_capacity(me_rows.len());
            for i in 0..me_rows.len() {
                edge_id_map.entry(me_rows[i]).or_default().push(me_cols[i]);
            }

            let mut multi_edge_m: Vec<(u64, u64)> = Vec::new();
            let mut multi_edge_dp: Vec<(u64, u64)> = Vec::new();

            for (matrix, multi_edges) in [(&m, &mut multi_edge_m), (&dp, &mut multi_edge_dp)] {
                let (rows, cols) = matrix.extract_tuples_bool();
                let mut u_rows = Vec::with_capacity(rows.len());
                let mut u_cols = Vec::with_capacity(rows.len());
                let mut u_vals = Vec::with_capacity(rows.len());

                for i in 0..rows.len() {
                    let src = rows[i];
                    let dst = cols[i];
                    let key = compound_key(src, dst);
                    let edge_ids = edge_id_map.get(&key).map_or(&[][..], |v| v.as_slice());

                    u_rows.push(src);
                    u_cols.push(dst);
                    if edge_ids.len() == 1 {
                        u_vals.push(edge_ids[0]);
                    } else {
                        u_vals.push(edge_ids.len() as u64 | MSB_MASK);
                        multi_edges.push((src, dst));
                    }
                }

                if u_rows.is_empty() {
                    empty.encode(w);
                } else {
                    let mut uint64_mat = Matrix::new_uint64(nrows, ncols);
                    uint64_mat.build_uint64(&u_rows, &u_cols, &u_vals);
                    uint64_mat.encode(w);
                }
            }

            empty.encode(w);

            (multi_edge_m, multi_edge_dp, Some(edge_id_map))
        } else {
            // Fast path: no multi-edges, no extract_m_dp() needed.
            let ((me_m_rows, me_m_cols), (me_dp_rows, me_dp_cols)) = self.me.extract_m_dp_tuples();

            // Encode uint64_m
            if me_m_rows.is_empty() {
                empty.encode(w);
            } else {
                let m_len = me_m_rows.len();
                let mut m_src = Vec::with_capacity(m_len);
                let mut m_dst = Vec::with_capacity(m_len);
                for &row in &me_m_rows {
                    m_src.push(row >> 32);
                    m_dst.push(row & 0xFFFF_FFFF);
                }
                let mut uint64_m = Matrix::new_uint64(nrows, ncols);
                uint64_m.build_uint64(&m_src, &m_dst, &me_m_cols);
                uint64_m.encode(w);
            }

            // Encode uint64_dp
            if me_dp_rows.is_empty() {
                empty.encode(w);
            } else {
                let dp_len = me_dp_rows.len();
                let mut dp_src = Vec::with_capacity(dp_len);
                let mut dp_dst = Vec::with_capacity(dp_len);
                for &row in &me_dp_rows {
                    dp_src.push(row >> 32);
                    dp_dst.push(row & 0xFFFF_FFFF);
                }
                let mut uint64_dp = Matrix::new_uint64(nrows, ncols);
                uint64_dp.build_uint64(&dp_src, &dp_dst, &me_dp_cols);
                uint64_dp.encode(w);
            }

            // Empty delta-minus
            empty.encode(w);

            (Vec::new(), Vec::new(), None)
        };

        w.write_unsigned(total);

        if total == 0 {
            return;
        }

        let mut v = Vector::<u64>::new(GrB_INDEX_MAX);
        for multi_edges in [&multi_edge_m, &multi_edge_dp] {
            w.write_unsigned(multi_edges.len() as u64);
            for &(src, dst) in multi_edges {
                let key = compound_key(src, dst);
                v.clear();

                if let Some(ref map) = edge_id_map
                    && let Some(ids) = map.get(&key)
                {
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
                    let key = compound_key(src, dst);
                    edges.set(key, value, true);
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
                    let key = compound_key(src, dst);
                    for (_, edge_id) in v.iter() {
                        edges.set(key, edge_id, true);
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
            let row = compound_key(self.src, self.dest);
            self.vit = Some(self.t.me.iter(row, row));
            return self.next();
        }

        None
    }
}
