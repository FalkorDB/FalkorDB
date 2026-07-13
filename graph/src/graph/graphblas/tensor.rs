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

use rustc_hash::FxHashSet;

use super::{
    matrix::{Dup, Matrix},
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

/// Edge storage for one relationship type, with inline edge ids.
///
/// The forward (`m`) and backward (`mt`) adjacency matrices are **UINT64** and
/// store the edge id of the (first) edge between a pair directly as the matrix
/// value — so a single-edge graph needs no separate edge-id matrix at all. The
/// `me` overflow matrix is **lazy**: it stays empty until a pair gains a second
/// edge, at which point the *additional* edge ids (2nd, 3rd, …) are stored there
/// keyed by `compound_key(src, dst)`. This keeps the common single-edge case at
/// ~12 bytes/edge instead of materializing a hypersparse edge-id matrix.
///
/// Invariants:
/// - `m[s, d]` / `mt[d, s]` hold the same edge id for the pair's first edge.
/// - `me` is non-empty iff some pair has more than one edge
///   (`has_multi_edge()` == `me.nvals() > 0`).
/// - total edges == `m.nvals() + me.nvals()` (first edges + overflow edges).
pub struct Tensor {
    /// Forward adjacency (src → dst), UINT64 value = first edge id.
    m: VersionedMatrix<u64>,
    /// Backward adjacency (dst → src), BOOL structure only. Edge ids are never
    /// stored here — they are recovered from `m` (and `me`) when iterating
    /// incoming edges, avoiding a redundant copy of every id.
    mt: VersionedMatrix<bool>,
    /// Overflow edge-id storage for multi-edges, keyed by `compound_key(src,
    /// dst)` → edge_id (BOOL). Empty unless a pair has more than one edge.
    me: VersionedMatrix<bool>,
}

impl Tensor {
    pub fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        Self {
            m: VersionedMatrix::<u64>::new(nrows, ncols),
            mt: VersionedMatrix::<bool>::new(ncols, nrows),
            me: VersionedMatrix::<bool>::new(GrB_INDEX_MAX, GrB_INDEX_MAX),
        }
    }

    /// Edge ids for the `(src, dest)` pair, in ascending edge-id order. The
    /// inline first-edge id from `m` is merged with any overflow ids from `me`
    /// (multi-edge pairs only) and sorted, matching the edge-id iteration order
    /// other engines expose. Returns an owned iterator (borrows nothing).
    #[must_use]
    pub fn get(
        &self,
        src: u64,
        dest: u64,
    ) -> std::vec::IntoIter<u64> {
        let mut ids: Vec<u64> = Vec::new();
        if let Some(first) = self.m.get(src, dest) {
            ids.push(first);
            if self.me.nvals() != 0 {
                let key = compound_key(src, dest);
                for (_, edge_id) in self.me.iter(key, key) {
                    ids.push(edge_id);
                }
                if ids.len() > 1 {
                    ids.sort_unstable();
                }
            }
        }
        ids.into_iter()
    }

    pub fn set(
        &mut self,
        src: u64,
        dest: u64,
        id: u64,
    ) {
        if self.m.get(src, dest).is_none() {
            // First edge for this pair: store inline in the forward/backward
            // adjacency.
            self.m.set(src, dest, id);
            self.mt.set(dest, src, true);
        } else {
            // Additional edge: pair already has a first edge → overflow to `me`.
            self.me.set(compound_key(src, dest), id, true);
        }
    }

    /// Set entries from parallel slices. The first edge of each pair lands
    /// inline in `m`/`mt`; any additional edges between an already-present pair
    /// overflow to `me`.
    ///
    /// Avoids the per-edge `get_uint64` (which would sync pending GraphBLAS work
    /// on every call, making bulk insertion quadratic): the set of pairs that
    /// already have an inline first edge is materialized once, then updated
    /// in-batch so the first occurrence of each pair is detected without
    /// touching GraphBLAS per edge.
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

        // Pairs that already have an inline first edge (committed or pending).
        let mut present: FxHashSet<(u64, u64)> = self.m.iter().map(|(s, d, _)| (s, d)).collect();

        let mut m_srcs: Vec<u64> = Vec::with_capacity(srcs.len());
        let mut m_dsts: Vec<u64> = Vec::with_capacity(srcs.len());
        let mut m_ids: Vec<u64> = Vec::with_capacity(srcs.len());
        for ((&s, &d), &id) in srcs.iter().zip(dsts.iter()).zip(ids.iter()) {
            if present.insert((s, d)) {
                // First edge for this pair → inline.
                m_srcs.push(s);
                m_dsts.push(d);
                m_ids.push(id);
            } else {
                // Additional edge → overflow.
                self.me.set(compound_key(s, d), id, true);
            }
        }

        self.m.set_all(
            m_srcs
                .iter()
                .zip(m_dsts.iter())
                .zip(m_ids.iter())
                .map(|((&s, &d), &id)| (s, d, id)),
        );
        self.mt
            .set_all(m_dsts.iter().zip(m_srcs.iter()).map(|(&d, &s)| (d, s)));
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

        // Fast path: no overflow edges exist, so every edge is the inline first
        // edge of its pair. Bulk-remove from the forward/backward adjacency in
        // two GraphBLAS ops; every touched pair becomes empty.
        if !self.has_multi_edge() {
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
            let mut m_mask = Matrix::<bool>::new(nrows, ncols);
            m_mask.build(&m_rows, &m_cols);
            let mut mt_mask = Matrix::<bool>::new(ncols, nrows);
            mt_mask.build(&mt_rows, &mt_cols);
            self.m.remove_mask(&m_mask);
            self.mt.remove_mask(&mt_mask);
            return rels.iter().map(|&(_, src, dst)| (src, dst)).collect();
        }

        // Slow path: some pairs have overflow edges. Handle per edge:
        //  - removing an overflow id: drop it from `me`; the pair survives.
        //  - removing the inline first-edge id: promote one overflow id into
        //    the inline slot if any remain, otherwise empty the pair.
        let mut emptied = Vec::new();
        for &(id, src, dst) in rels {
            let key = compound_key(src, dst);
            if self.m.get(src, dst) == Some(id) {
                // Removing the inline first edge.
                let promote = self.me.iter(key, key).next().map(|(_, eid)| eid);
                if let Some(eid) = promote {
                    self.me.remove(key, eid);
                    self.m.set(src, dst, eid);
                    // mt structure already has (dst, src); the pair survives.
                } else {
                    self.m.remove(src, dst);
                    self.mt.remove(dst, src);
                    emptied.push((src, dst));
                }
            } else {
                // Removing an overflow edge; the inline first edge is untouched.
                self.me.remove(key, id);
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
    ///
    /// `mt` is structure-only (`bool`). The forward matrix's *effective*
    /// structure (`(m − dm) ∪ dp`) is materialized first, then transposed into
    /// a clean base with empty deltas. Materializing (rather than transposing
    /// the three layers separately) keeps `mt` valid even when the uint64
    /// forward matrix carries a dp overlay (`dp ∩ m ≠ ∅`), which would break
    /// the bool disjointness invariants `mt` relies on.
    pub fn rebuild_backward(&mut self) {
        self.mt = VersionedMatrix::from_matrix(self.m.extract().transpose());
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
    pub const fn matrix(&self) -> &VersionedMatrix<u64> {
        &self.m
    }

    /// Transposed/backward pair-level adjacency (dst → src), structure only.
    #[must_use]
    pub const fn matrix_t(&self) -> &VersionedMatrix<bool> {
        &self.mt
    }

    /// Overflow multi-edge id storage (`me`), keyed by `compound_key(src,
    /// dst)` → edge id. Holds only the 2nd, 3rd, … edge of a pair; empty
    /// unless some pair has more than one edge.
    #[must_use]
    pub const fn edge_versioned(&self) -> &VersionedMatrix<bool> {
        &self.me
    }

    /// Iterate the edge-id matrix (`me`) keyed by `compound_key(src, dst)`.
    /// Total number of edges (inline first edges plus multi-edge overflow).
    #[must_use]
    pub fn edge_count(&self) -> u64 {
        self.m.nvals() + self.me.nvals()
    }

    /// Iterate every `(src, dst, edge_id)` triple in the tensor.
    ///
    /// Yields the inline first edge of every pair from the UINT64 forward
    /// matrix `m`, followed by any overflow (multi-edge) ids from `me`. On a
    /// single-edge graph `me` is empty, so this is a single streaming pass over
    /// `m` with no per-pair sub-iterator.
    pub fn iter_edges(&self) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
        let overflow: Box<dyn Iterator<Item = (u64, u64, u64)> + '_> = if self.me.nvals() != 0 {
            Box::new(
                self.me
                    .iter(0, GrB_INDEX_MAX)
                    .map(|(key, edge_id)| (key >> 32, key & 0xFFFF_FFFF, edge_id)),
            )
        } else {
            Box::new(std::iter::empty())
        };
        self.m.uint64_iter_range(0, u64::MAX).chain(overflow)
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
    /// Overflow ids live in `me`, so this is simply `me` being non-empty.
    #[must_use]
    pub fn has_multi_edge(&self) -> bool {
        self.me.nvals() != 0
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
        let has_multi = self.me.nvals() != 0;

        // Serialize the C-compatible UINT64 forward matrix from the effective
        // inline state. Single-edge pairs store the edge id directly; multi-edge
        // pairs store `(edge_count | MSB)` and push their full id list (inline
        // first edge + `me` overflow) into the tensor section below.
        let mut f_rows: Vec<u64> = Vec::new();
        let mut f_cols: Vec<u64> = Vec::new();
        let mut f_vals: Vec<u64> = Vec::new();
        let mut multi: Vec<(u64, u64, Vec<u64>)> = Vec::new();
        for (src, dst, first_id) in self.m.uint64_iter_range(0, u64::MAX) {
            f_rows.push(src);
            f_cols.push(dst);
            if has_multi {
                let key = compound_key(src, dst);
                let overflow: Vec<u64> = self.me.iter(key, key).map(|(_, id)| id).collect();
                if overflow.is_empty() {
                    f_vals.push(first_id);
                } else {
                    let mut ids = Vec::with_capacity(1 + overflow.len());
                    ids.push(first_id);
                    ids.extend(overflow);
                    ids.sort_unstable();
                    f_vals.push(ids.len() as u64 | MSB_MASK);
                    multi.push((src, dst, ids));
                }
            } else {
                f_vals.push(first_id);
            }
        }

        // Forward VersionedMatrix layout: base (effective), empty delta-plus,
        // empty delta-minus. Folding dp into the base keeps the on-disk form
        // canonical and matches what decode expects.
        let empty = Matrix::<u64>::new(nrows, ncols);
        if f_rows.is_empty() {
            empty.encode(w);
        } else {
            let mut fm = Matrix::<u64>::new(nrows, ncols);
            fm.build(&f_rows, &f_cols, &f_vals);
            fm.encode(w);
        }
        empty.encode(w); // delta-plus
        empty.encode(w); // delta-minus

        let total = self.m.nvals() + self.me.nvals();
        w.write_unsigned(total);
        if total == 0 {
            return;
        }

        // Tensor section: two groups (base TM, delta-plus TDP). All multi-edge
        // pairs live in the base group; the delta-plus group is empty since dp
        // was folded into the base above.
        let mut v = Vector::<u64>::new(GrB_INDEX_MAX);
        w.write_unsigned(multi.len() as u64);
        for (src, dst, ids) in &multi {
            v.clear();
            for (idx, &edge_id) in ids.iter().enumerate() {
                v.set(idx as u64, edge_id);
            }
            w.write_unsigned(*src);
            w.write_unsigned(*dst);
            v.encode(w);
        }
        w.write_unsigned(0); // empty delta-plus tensor group
    }
}

impl Decode<19> for Tensor {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        let forward = VersionedMatrix::<u64>::decode(r)?;
        let nrows = forward.nrows();
        let ncols = forward.ncols();

        // Inline representation: `m`/`mt` are UINT64 first-edge ids; `me` holds
        // multi-edge overflow. The on-disk forward matrix (C-compatible) stores
        // single-edge ids directly (MSB clear) and `(count | MSB)` for multi-edge
        // pairs, whose real id lists follow in the tensor section.
        let mut m = VersionedMatrix::<u64>::new(nrows, ncols);
        let mut me = VersionedMatrix::<bool>::new(GrB_INDEX_MAX, GrB_INDEX_MAX);

        for (src, dst, value) in forward.iter() {
            if value & MSB_MASK == 0 {
                // Single edge: value is the edge id; store inline.
                m.set(src, dst, value);
            }
            // Multi-edge (MSB set): ids are supplied by the tensor section.
        }

        let total_tensor_count = r.read_unsigned()?;
        if total_tensor_count > 0 {
            // Two groups: base (TM) then delta-plus (TDP). Each pair's id list is
            // `[first_inline, overflow...]`; the first lands in `m`, the rest in
            // `me`.
            for _ in 0..2 {
                let count = r.read_unsigned()?;
                for _ in 0..count {
                    let src = r.read_unsigned()?;
                    let dst = r.read_unsigned()?;
                    let v = Vector::<u64>::decode(r)?;
                    let key = compound_key(src, dst);
                    let mut first = true;
                    for (_, edge_id) in v.iter() {
                        if first {
                            m.set(src, dst, edge_id);
                            first = false;
                        } else {
                            me.set(key, edge_id, true);
                        }
                    }
                }
            }
        }

        // Backward matrix is rebuilt from `m` by the caller (`rebuild_backward`)
        // after decode, so leave it empty here.
        let backward = VersionedMatrix::<bool>::new(0, 0);
        Ok(Self {
            m,
            mt: backward,
            me,
        })
    }
}

/// Base adjacency iterator. Forward iteration streams inline first-edge ids
/// directly from `m`; backward iteration streams the BOOL structure of `mt`
/// (which carries no ids) and recovers each first-edge id from `m`.
enum BaseIter {
    Forward(versioned_matrix::UintIter),
    Backward(versioned_matrix::Iter),
}

pub struct Iter<'a> {
    t: &'a Tensor,
    base: BaseIter,
    /// Whether multi-edge overflow exists at all (skip `me` lookups if not).
    has_multi: bool,
    src: u64,
    dest: u64,
    /// Buffered, ascending edge ids for the current multi-edge pair.
    buf: Vec<u64>,
    buf_pos: usize,
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
            base: if transpose {
                BaseIter::Backward(t.mt.iter(min_row, max_row))
            } else {
                BaseIter::Forward(t.m.uint64_iter_range(min_row, max_row))
            },
            has_multi: t.me.nvals() != 0,
            src: 0,
            dest: 0,
            buf: Vec::new(),
            buf_pos: 0,
        }
    }
}

impl Iterator for Iter<'_> {
    type Item = (u64, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        // Drain buffered (sorted) overflow edges for the current pair.
        if self.buf_pos < self.buf.len() {
            let id = self.buf[self.buf_pos];
            self.buf_pos += 1;
            return Some((self.src, self.dest, id));
        }

        // Next base pair, oriented as (src, dest) with its inline first-edge id.
        // Forward carries the id inline; backward recovers it from `m`.
        let (src, dest, first_id) = match &mut self.base {
            BaseIter::Forward(it) => {
                let (row, col, id) = it.next()?;
                (row, col, id)
            }
            BaseIter::Backward(it) => {
                let (row, col) = it.next()?;
                let (src, dest) = (col, row);
                let id = self.t.m.get(src, dest).unwrap_or(0);
                (src, dest, id)
            }
        };
        self.src = src;
        self.dest = dest;

        if self.has_multi {
            // Merge the inline first edge with any overflow ids and yield them
            // in ascending edge-id order (matches other engines' iteration).
            let key = compound_key(self.src, self.dest);
            self.buf.clear();
            for (_, id) in self.t.me.iter(key, key) {
                self.buf.push(id);
            }
            if !self.buf.is_empty() {
                self.buf.push(first_id);
                self.buf.sort_unstable();
                let id = self.buf[0];
                self.buf_pos = 1;
                return Some((self.src, self.dest, id));
            }
        }
        Some((self.src, self.dest, first_id))
    }
}
