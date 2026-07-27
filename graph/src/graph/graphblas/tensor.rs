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
//! The forward adjacency is stored as three delta layers (the same
//! base/delta-plus/delta-minus model as [`VersionedMatrix`], owned directly by
//! the tensor so edge-id semantics stay local); the backward adjacency and
//! multi-edge storage are [`VersionedMatrix`] instances:
//!
//! ```text
//!   Tensor
//!     |
//!     |-- m/dp/dm (forward adjacency)  src --> dst  (UINT64 edge id or MULTI_EDGE)
//!     |-- mt (backward adjacency)      dst --> src  (boolean)
//!     |-- me (multi-edge storage)      compound_key --> edge_id  (boolean)
//!
//!   Forward matrix (m):        Backward matrix (mt):
//!     dst: 0  1  2               src: 0  1  2
//!   src 0 [ .  M  . ]         dst 0 [ .  .  . ]
//!       1 [ .  .  . ]             1 [ T  .  T ]
//!       2 [ .  7  . ]             2 [ .  .  . ]
//!
//!   Edges: 0->1 (ids 5 and 9, both in me; inline value M = MULTI_EDGE),
//!          2->1 (id=7, inline)
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
//! ## Delta-Layer Invariants
//!
//! The effective value of a pair is: **`dp` wins, else `m` unless its `dm`
//! bit is set.**
//!
//! - `dm ⊆ m`, and `dm` marks *pure deletions only*: `dp ∩ dm = ∅` (writing
//!   a pair clears its `dm` bit).
//! - `dp` may *shadow* `m` (in-place value update: promotion, demotion, or
//!   replace-in-one-transaction), carrying the live value; `dp ∩ m` may be
//!   non-empty. The sorted-merge iterator yields the live `dp` value for a
//!   shadowed pair and skips the stale `m` entry.
//! - *Cancel-to-clean*: restoring a pair's committed value drops its deltas
//!   — `dp` never holds a value equal to the pair's live `m` value (in
//!   particular `dp = M` never shadows `m = M`).
//! - A pair has ≥ 2 edges iff its effective inline value is the
//!   [`MULTI_EDGE`] sentinel `M`; then *all* its ids live in `me` and the
//!   pair is counted by `multi_count`. Otherwise its single id is inline and
//!   `me` has no row for it.
//! - `mt` mirrors the effective forward structure (BOOL, no ids).
//! - `edge_count = |m| + |dp| − |dm| − |dp ∩ m| − multi_count + |me|`.
//!
//! ## Per-Pair State Diagram
//!
//! All reachable layer states of one `(src, dst)` pair (example ids 1, 2, 3;
//! `x` = `dm` bit set; `·` = empty):
//!
//! ```text
//!   Uncommitted (pair absent from m):
//!     A  ·                          empty
//!     B  dp=1                       pending single edge
//!     C  dp=M          me={1,2}     pending multi pair
//!
//!   Committed single (m=1):
//!     D  m=1                        clean
//!     G  m=1 dm=x                   deleted
//!     H  m=1 dp=2                   value replaced   (dp shadows m)
//!     F  m=1 dp=M      me={1,2}     promoted         (dp shadows m)
//!
//!   Committed multi (m=M):
//!     E  m=M           me={1,2}     clean
//!     I  m=M dp=1      me={}        demoted          (dp shadows m)
//!     J  m=M dm=x      me={}        deleted
//!
//!   Transitions (add/del of edge ids; "cancel" = deltas removed because the
//!   committed value is restored):
//!
//!     A --add 1--> B --add 2--> C          C --del 2--> B --del 1--> A
//!
//!     D --add 2--> F                       F --del 2--> D   (cancel)
//!     D --del 1--> G                       G --add 1--> D   (cancel)
//!     G --add 2--> H                       H --del 2--> G
//!     G --add 2,3--> F                     F --del 1--> H
//!     H --add 3--> F  (me={2,3})           F --del 1,2--> G
//!
//!     E --del 2--> I                       I --add 3--> E   (cancel, me={1,3})
//!     E --del 1,2--> J                     J --add 3,4--> E (cancel, me={3,4})
//!     I --del 1--> J                       J --add 3--> I
//! ```
//!
//! ## Iteration
//!
//! [`Iter`] walks the effective forward (or backward) adjacency and yields
//! `(src, dst, edge_id)` triples. Single-edge pairs yield their inline id
//! directly; pairs whose inline value is [`MULTI_EDGE`] yield every id stored
//! in their `me` row.
//!
//! ## Use Case
//!
//! In property graphs, multiple edges of the same type can connect two nodes.
//! For example, two "TRANSFERRED" relationships between the same bank accounts
//! with different amounts and dates.

use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;

use crate::graph::{
    cow::Cow,
    graphblas::matrix::{BoolExtract, Uint64Extract},
};

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
/// The forward adjacency is **UINT64**: a single-edge pair stores its edge id
/// directly as the matrix value, so a single-edge graph needs no separate
/// edge-id matrix at all. It is kept as three MVCC delta layers owned by the
/// tensor: committed base `m`, pending additions `dp`, pending deletions `dm`.
/// The `me` multi-edge matrix is **lazy**: it stays empty until a pair gains a
/// second edge. At that point the pair is *promoted*: its inline value becomes
/// the [`MULTI_EDGE`] sentinel (`u64::MAX`) and **all** of its edge ids
/// (including the former inline one) move to `me`, keyed by
/// `compound_key(src, dst)`. When removals bring a pair back down to one edge
/// it is *demoted*: the surviving id returns inline and its `me` row empties.
/// This keeps the common single-edge case at ~12 bytes/edge instead of
/// materializing a hypersparse edge-id matrix.
///
/// Invariants and the full per-pair state diagram are documented in the
/// [module docs](self): `dp` wins over `m` unless dm-masked; `dm ⊆ m` with
/// `dp ∩ dm = ∅`; `dp` may shadow `m` with the live value; deltas cancel when
/// a pair's committed value is restored.
pub struct Tensor {
    /// Base committed matrix
    m: Cow<Matrix<u64>>,
    /// Delta-plus: edges added in current transaction
    dp: Cow<Matrix<u64>>,
    /// Delta-minus: edges removed in current transaction (always a bool mask)
    dm: Cow<Matrix<bool>>,
    /// Backward adjacency (dst → src), BOOL structure only. Edge ids are never
    /// stored here — they are recovered from `m` (and `me`) when iterating
    /// incoming edges, avoiding a redundant copy of every id.
    mt: VersionedMatrix<bool>,
    /// Multi-edge id storage, keyed by `compound_key(src, dst)` → edge_id
    /// (BOOL). Holds *all* ids of pairs with more than one edge; empty
    /// otherwise.
    me: VersionedMatrix<bool>,
    /// Number of pairs whose effective inline value is [`MULTI_EDGE`].
    multi_count: u64,
}

/// Sentinel stored as the inline forward value of a pair with more than one
/// edge; the pair's real edge ids all live in `me`. Real edge ids can never
/// collide with it (they are bounded by [`GrB_INDEX_MAX`]).
pub const MULTI_EDGE: u64 = u64::MAX;

/// Shallow clone sharing the underlying GraphBLAS handles (no data copy),
/// same semantics as `VersionedMatrix`'s `Clone`. Use [`Tensor::dup`] to
/// create a new MVCC version instead.
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            m: self.m.clone(),
            dp: self.dp.clone(),
            dm: self.dm.clone(),
            mt: self.mt.clone(),
            me: self.me.clone(),
            multi_count: self.multi_count,
        }
    }
}

impl Tensor {
    #[must_use]
    pub fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        Self {
            m: Cow::new(Matrix::<u64>::new(nrows, ncols)),
            dp: Cow::new(Matrix::<u64>::new(nrows, ncols).into_hyper()),
            dm: Cow::new(Matrix::<bool>::new(nrows, ncols).into_hyper()),
            mt: VersionedMatrix::<bool>::new(ncols, nrows),
            me: VersionedMatrix::<bool>::new(GrB_INDEX_MAX, GrB_INDEX_MAX),
            multi_count: 0,
        }
    }

    /// Wait pending GraphBLAS work on the forward delta layers. The committed
    /// base `m` is never written to inside a transaction, so it can't be
    /// pending (same invariant as `VersionedMatrix::wait`).
    fn wait_fwd(&self) {
        debug_assert!(!self.m.pending());
        self.dp.wait();
        self.dm.wait();
    }

    /// Effective first-edge id at `(src, dest)`: `dp` wins, then `m` unless
    /// masked by `dm`. Returns `None` if absent or deleted.
    fn eff_get(
        &self,
        src: u64,
        dest: u64,
    ) -> Option<u64> {
        self.wait_fwd();
        if let Some(v) = self.dp.get(src, dest) {
            return Some(v);
        }
        if self.dm.nvals() != 0 && self.dm.contains(src, dest) {
            return None;
        }
        self.m.get(src, dest)
    }

    /// Edge ids for the `(src, dest)` pair, in ascending edge-id order.
    /// Single-edge pairs answer straight from the inline value — no heap
    /// allocation; multi-edge pairs (inline value == [`MULTI_EDGE`]) read
    /// their id row from `me`, which GraphBLAS already yields in ascending
    /// column order. Returns an owned iterator (borrows nothing).
    #[must_use]
    pub fn get(
        &self,
        src: u64,
        dest: u64,
    ) -> EdgeIds {
        match self.eff_get(src, dest) {
            Some(MULTI_EDGE) => {
                let key = compound_key(src, dest);
                EdgeIds::Multi(self.me.iter(key, key))
            }
            inline => EdgeIds::Inline(inline.into_iter()),
        }
    }

    /// Set entries from parallel slices. The first edge of a pair lands
    /// inline in `dp`/`mt`; when a pair gains a second edge it is promoted:
    /// the inline value becomes [`MULTI_EDGE`] and *all* of its ids (the
    /// former inline one included) go to `me`.
    ///
    /// Split into a read phase and a write phase so per-edge `dp` lookups
    /// never observe pending GraphBLAS work (which would force a wait per
    /// element and make bulk insertion quadratic). `me` is written during the
    /// read phase — it is never read here, so its pending work is harmless.
    /// Pairs appearing more than once *within this batch* are caught by the
    /// `batch` map, which tracks the pending inline slot of each new pair so
    /// a later duplicate can retroactively promote it.
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

        self.wait_fwd();
        let dm_empty = self.dm.nvals() == 0;

        // Read phase: decide each edge's placement. `batch` maps a pair to
        // the index of its pending inline id in `m_ids`, or `usize::MAX` once
        // the pair is known multi-edge. `m_masked` (parallel to `m_ids`) holds
        // the committed `m` value of pairs needing delta reconciliation
        // (dm-masked, or dp-shadowed at promotion) so the write phase can
        // clear the mask and cancel to clean when the value matches.
        let mut batch: FxHashMap<(u64, u64), usize> = FxHashMap::default();
        let mut m_srcs: Vec<u64> = Vec::with_capacity(srcs.len());
        let mut m_dsts: Vec<u64> = Vec::with_capacity(srcs.len());
        let mut m_ids: Vec<u64> = Vec::with_capacity(srcs.len());
        let mut m_masked: Vec<Option<u64>> = Vec::with_capacity(srcs.len());
        for ((&s, &d), &id) in srcs.iter().zip(dsts.iter()).zip(ids.iter()) {
            let key = compound_key(s, d);
            match batch.entry((s, d)) {
                Entry::Occupied(mut e) => {
                    let idx = *e.get();
                    if idx != usize::MAX {
                        // Second edge of a pair new in this batch: promote the
                        // pending inline slot in place.
                        self.me.set(key, m_ids[idx], true);
                        m_ids[idx] = MULTI_EDGE;
                        self.multi_count += 1;
                        e.insert(usize::MAX);
                    }
                    self.me.set(key, id, true);
                }
                Entry::Vacant(e) => {
                    let masked = !dm_empty && self.dm.contains(s, d);
                    let from_dp = self.dp.get(s, d);
                    let cur = from_dp.or_else(|| if masked { None } else { self.m.get(s, d) });
                    match cur {
                        // Already multi-edge: just add the id to `me`.
                        Some(MULTI_EDGE) => {
                            self.me.set(key, id, true);
                            e.insert(usize::MAX);
                        }
                        // Present single edge: promote — move the existing
                        // inline id to `me` alongside the new one, and queue
                        // the sentinel for the inline slot.
                        Some(cur_id) => {
                            self.me.set(key, cur_id, true);
                            self.me.set(key, id, true);
                            self.multi_count += 1;
                            e.insert(usize::MAX);
                            m_srcs.push(s);
                            m_dsts.push(d);
                            m_ids.push(MULTI_EDGE);
                            // A dp-shadowed pair may have MULTI_EDGE committed
                            // (demoted from committed multi earlier in this
                            // transaction); record the committed value so the
                            // write phase cancels re-promotion back to clean
                            // instead of shadowing `m=[M]` with `dp=[M]`.
                            m_masked.push(if from_dp.is_some() {
                                self.m.get(s, d)
                            } else {
                                None
                            });
                        }
                        // First edge for this pair: inline.
                        None => {
                            e.insert(m_ids.len());
                            m_srcs.push(s);
                            m_dsts.push(d);
                            m_ids.push(id);
                            m_masked.push(if masked { self.m.get(s, d) } else { None });
                        }
                    }
                }
            }
        }

        // Write phase: inline values (real ids or promotion sentinels) land
        // in `dp`, shadowing any live committed entry (no `dm` mask). A pair
        // whose committed entry was dm-masked gets un-masked (`dp ∩ dm = ∅`);
        // if the new value equals the committed one, the deltas cancel and
        // the pair returns to the clean state.
        for (i, ((&s, &d), &id)) in m_srcs
            .iter()
            .zip(m_dsts.iter())
            .zip(m_ids.iter())
            .enumerate()
        {
            self.mt.set(d, s, true);
            if let Some(committed) = m_masked[i] {
                self.dm.remove(s, d);
                if committed == id {
                    // Cancel to clean: committed value restored. Drop any dp
                    // shadow (re-promotion of a demoted committed-multi pair;
                    // both removes are no-ops when the entry is absent).
                    self.dp.remove(s, d);
                    continue;
                }
            }
            self.dp.set(s, d, id);
        }
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

        // Fast path: no multi-edge pairs exist, so every edge is the inline
        // value of its pair. Bulk-update the delta layers in a few GraphBLAS
        // ops (same layer math as `VersionedMatrix::remove_mask`); every
        // touched pair becomes empty.
        if !self.has_multi_edge() {
            self.wait_fwd();
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
            // dm |= (m ∩ mask): mark every committed entry the mask selects as
            // deleted; dp &= ¬mask: drop pending adds (including the shadow
            // value of any in-place-updated pair, whose committed entry the dm
            // update just masked — keeping `dp ∩ dm = ∅`).
            self.dm
                .element_wise_add(Some(&m_mask), None, Some(&*self.m), None);
            self.dp.remove_all(&m_mask);
            self.mt.remove_mask(&mt_mask);
            return rels.iter().map(|&(_, src, dst)| (src, dst)).collect();
        }

        // Slow path: some pairs have multiple edges. Handle per edge:
        //  - single-edge pair: delete the pair.
        //  - multi-edge pair (inline == MULTI_EDGE): drop the id from `me`;
        //    when exactly one id remains, demote it back into the inline slot.
        let mut emptied = Vec::new();
        for &(id, src, dst) in rels {
            let key = compound_key(src, dst);
            match self.eff_get(src, dst) {
                Some(MULTI_EDGE) => {
                    self.me.remove(key, id);
                    let mut it = self.me.iter(key, key);
                    let survivor = it.next();
                    let still_multi = it.next().is_some();
                    drop(it);
                    if still_multi {
                        continue;
                    }
                    self.multi_count -= 1;
                    if let Some((_, last)) = survivor {
                        // Demote: the surviving id returns inline. If it *is*
                        // the committed value, the deltas cancel and the pair
                        // returns clean; otherwise `dp` shadows `m` with the
                        // live value (no `dm` mask).
                        self.me.remove(key, last);
                        if self.m.get(src, dst) == Some(last) {
                            self.dp.remove(src, dst);
                        } else {
                            self.dp.set(src, dst, last);
                        }
                        // mt structure already has (dst, src); the pair survives.
                    } else {
                        // All ids removed at once; the pair is gone.
                        self.dp.remove(src, dst);
                        if self.m.contains(src, dst) {
                            self.dm.set(src, dst, true);
                        }
                        self.mt.remove(dst, src);
                        emptied.push((src, dst));
                    }
                }
                Some(inline_id) if inline_id == id => {
                    self.dp.remove(src, dst);
                    if self.m.contains(src, dst) {
                        self.dm.set(src, dst, true);
                    }
                    self.mt.remove(dst, src);
                    emptied.push((src, dst));
                }
                // Unknown id or absent pair: nothing to remove.
                _ => {}
            }
        }
        emptied
    }

    pub fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) {
        self.wait_fwd();
        self.m.resize(nrows, ncols);
        self.dp.resize(nrows, ncols);
        self.dm.resize(nrows, ncols);
        self.mt.resize(ncols, nrows);
    }

    /// Merge oversized deltas into the committed base (same policy as
    /// [`VersionedMatrix::flush`]): when `dp` exceeds 10,000 entries it is
    /// folded into `m` via a value-preserving eWiseAdd (`SECOND`, so `dp`
    /// wins on shadowed pairs) and cleared; when `dm` exceeds 10,000 entries
    /// its masked entries are removed from `m` and it is cleared. `dp ∩ dm =
    /// ∅`, so the two merges are order-independent and all invariants are
    /// preserved. `mt` and `me` flush themselves.
    pub fn flush(&mut self) {
        self.wait_fwd();
        if self.dp.nvals() >= 10000 {
            self.m.element_wise_add_second(&self.dp);
            self.dp.clear();
        }
        if self.dm.nvals() >= 10000 {
            self.m.remove_all(&self.dm);
            self.dm.clear();
        }
        self.mt.flush();
        self.me.flush();
    }

    /// Materialize the effective forward structure as a `bool` matrix:
    /// `(m ∖ dm) ∪ dp`, values discarded. Shadowed pairs (`dp ∩ m`) collapse
    /// in the bool union; `dm ⊆ m` is disjoint from `dp`, so order is safe.
    #[must_use]
    pub fn extract(&self) -> Matrix<bool> {
        self.wait_fwd();
        let mut m = Matrix::<bool>::new(self.m.nrows(), self.m.ncols());
        m.element_wise_add(None, None, Some(&*self.m), None);
        if self.dm.nvals() > 0 {
            m.remove_all(&self.dm);
        }
        if self.dp.nvals() > 0 {
            m.element_wise_add(None, None, Some(&*self.dp), None);
        }
        m
    }

    /// Rebuild the backward matrix as the transpose of the forward matrix.
    ///
    /// `mt` is structure-only (`bool`). The forward matrix's *effective*
    /// structure (`(m ∖ dm) ∪ dp`) is materialized first, then transposed into
    /// a clean base with empty deltas. Materializing (rather than transposing
    /// the three layers separately) keeps `mt` valid even when the uint64
    /// forward matrix carries a shadowed in-place update (`dp ∩ m ≠ ∅`),
    /// which would break the bool disjointness invariants `mt` relies on.
    pub fn rebuild_backward(&mut self) {
        self.mt = VersionedMatrix::from_matrix(self.extract().transpose());
    }

    #[must_use]
    pub fn dup(&self) -> Self {
        Self {
            m: self.m.new_version(),
            dp: self.dp.new_version(),
            dm: self.dm.new_version(),
            mt: self.mt.dup(),
            me: self.me.dup(),
            multi_count: self.multi_count,
        }
    }

    /// Committed base of the forward adjacency (UINT64 first-edge ids).
    #[must_use]
    pub fn fwd_m(&self) -> &Matrix<u64> {
        &self.m
    }

    /// Pending additions to the forward adjacency (UINT64 first-edge ids).
    #[must_use]
    pub fn fwd_dp(&self) -> &Matrix<u64> {
        &self.dp
    }

    /// Pending-deletion mask over the forward base (BOOL, `dm ⊆ m`, pure
    /// deletions only: `dp ∩ dm = ∅`).
    #[must_use]
    pub fn fwd_dm(&self) -> &Matrix<bool> {
        &self.dm
    }

    /// Structure-only `(src, dst)` iterator over the effective forward
    /// adjacency `(m ∖ dm) ∪ dp`, ignoring the stored edge-id values.
    #[must_use]
    pub fn structural_iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> versioned_matrix::Iter<BoolExtract> {
        self.wait_fwd();
        versioned_matrix::Iter::<BoolExtract>::from_layers(
            &self.m, &self.dp, &self.dm, min_row, max_row,
        )
    }

    /// Effective `(src, dst, first_edge_id)` triples of the forward adjacency.
    fn fwd_iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> versioned_matrix::Iter<Uint64Extract> {
        self.wait_fwd();
        versioned_matrix::Iter::<Uint64Extract>::from_layers(
            &self.m, &self.dp, &self.dm, min_row, max_row,
        )
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

    /// Total number of edges. Each effective forward entry is one edge,
    /// except `MULTI_EDGE` sentinels (counted by `multi_count`), whose real
    /// ids all live in `me`. Effective nvals is `|m| + |dp| − |dm| − |dp ∩ m|`
    /// (`dp` may shadow `m`; `dm ⊆ m` is disjoint from `dp`).
    #[must_use]
    pub fn edge_count(&self) -> u64 {
        self.wait_fwd();
        let shadow = if self.dp.nvals() == 0 {
            0
        } else {
            self.dp.intersection_nvals(&self.m)
        };
        self.m.nvals() + self.dp.nvals() - self.dm.nvals() - shadow - self.multi_count
            + self.me.nvals()
    }

    /// Iterate every `(src, dst, edge_id)` triple in the tensor.
    ///
    /// Yields the inline single edge of every pair from the effective UINT64
    /// forward adjacency (skipping `MULTI_EDGE` sentinels), followed by all
    /// multi-edge ids from `me`. On a single-edge graph `me` is empty, so
    /// this is a single streaming pass with no per-pair sub-iterator.
    pub fn iter_edges(&self) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
        let multi: Box<dyn Iterator<Item = (u64, u64, u64)> + '_> = if self.me.nvals() == 0 {
            Box::new(std::iter::empty())
        } else {
            Box::new(
                self.me
                    .iter(0, GrB_INDEX_MAX)
                    .map(|(key, edge_id)| (key >> 32, key & 0xFFFF_FFFF, edge_id)),
            )
        };
        self.fwd_iter(0, u64::MAX)
            .filter(|&(_, _, id)| id != MULTI_EDGE)
            .chain(multi)
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
    /// All ids of such pairs live in `me`, so this is simply `me` being
    /// non-empty.
    #[must_use]
    pub fn has_multi_edge(&self) -> bool {
        self.me.nvals() != 0
    }

    pub fn wait(&self) {
        self.wait_fwd();
        self.mt.wait();
        self.me.wait();
    }

    /// Wait on all matrices for fork safety (takes &self, not &mut self).
    pub fn wait_all(&self) {
        self.m.wait();
        self.dp.wait();
        self.dm.wait();
        self.mt.wait_all();
        self.me.wait_all();
    }

    /// Returns true if every internal matrix has no pending GraphBLAS
    /// operations queued.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        self.m.is_synced()
            && self.dp.is_synced()
            && self.dm.is_synced()
            && self.mt.is_synced()
            && self.me.is_synced()
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.m.memory_usage()
            + self.dp.memory_usage()
            + self.dm.memory_usage()
            + self.mt.memory_usage()
            + self.me.memory_usage()
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

        // Serialize the C-compatible UINT64 forward matrix from the effective
        // inline state. Single-edge pairs store the edge id directly; multi-edge
        // pairs (inline value == MULTI_EDGE) store `(edge_count | MSB)` and push
        // their full id list from `me` into the tensor section below.
        let mut f_rows: Vec<u64> = Vec::new();
        let mut f_cols: Vec<u64> = Vec::new();
        let mut f_vals: Vec<u64> = Vec::new();
        let mut multi: Vec<(u64, u64, Vec<u64>)> = Vec::new();
        for (src, dst, inline) in self.fwd_iter(0, u64::MAX) {
            f_rows.push(src);
            f_cols.push(dst);
            if inline == MULTI_EDGE {
                let key = compound_key(src, dst);
                let ids: Vec<u64> = self.me.iter(key, key).map(|(_, id)| id).collect();
                f_vals.push(ids.len() as u64 | MSB_MASK);
                multi.push((src, dst, ids));
            } else {
                f_vals.push(inline);
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

        let total = self.edge_count();
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
        // On-disk forward layers (base, delta-plus, delta-minus). Encode folds
        // the deltas into the base, but merge defensively: `(fwd_m ∖ fwd_dm) ∪
        // fwd_dp`.
        let fwd_m = Matrix::<u64>::decode(r)?;
        let fwd_dp = Matrix::<u64>::decode(r)?;
        let fwd_dm = Matrix::<bool>::decode(r)?;
        let nrows = fwd_m.nrows();
        let ncols = fwd_m.ncols();

        // Inline representation: `m` is UINT64 inline edge ids with MULTI_EDGE
        // sentinels; `me` holds all ids of multi-edge pairs. The on-disk forward
        // matrix (C-compatible) stores single-edge ids directly (MSB clear) and
        // `(count | MSB)` for multi-edge pairs, whose real id lists follow in
        // the tensor section.
        let mut m = Matrix::<u64>::new(nrows, ncols);
        let mut me = VersionedMatrix::<bool>::new(GrB_INDEX_MAX, GrB_INDEX_MAX);
        let mut multi_count: u64 = 0;

        let dm_empty = fwd_dm.nvals() == 0;
        for (src, dst, value) in fwd_m.iter(0, u64::MAX) {
            if !dm_empty && fwd_dm.contains(src, dst) {
                continue; // deleted (any live replacement id lives in fwd_dp)
            }
            if value & MSB_MASK == 0 {
                m.set(src, dst, value);
            } else {
                // Multi-edge: ids are supplied by the tensor section.
                m.set(src, dst, MULTI_EDGE);
            }
        }
        for (src, dst, value) in fwd_dp.iter(0, u64::MAX) {
            if value & MSB_MASK == 0 {
                m.set(src, dst, value);
            } else {
                m.set(src, dst, MULTI_EDGE);
            }
        }

        let total_tensor_count = r.read_unsigned()?;
        if total_tensor_count > 0 {
            // Two groups: base (TM) then delta-plus (TDP). Every id of a
            // multi-edge pair lands in `me`; the inline value stays MULTI_EDGE.
            for _ in 0..2 {
                let count = r.read_unsigned()?;
                multi_count += count;
                for _ in 0..count {
                    let src = r.read_unsigned()?;
                    let dst = r.read_unsigned()?;
                    let v = Vector::<u64>::decode(r)?;
                    let key = compound_key(src, dst);
                    for (_, edge_id) in v.iter() {
                        me.set(key, edge_id, true);
                    }
                }
            }
        }

        // The freshly built base must be materialized: the committed base is
        // never pending inside a transaction (see `wait_fwd`).
        m.wait();
        // Backward matrix is rebuilt by the caller (`rebuild_backward`) after
        // decode, so leave it empty here.
        Ok(Self {
            m: Cow::new(m),
            dp: Cow::new(Matrix::<u64>::new(nrows, ncols).into_hyper()),
            dm: Cow::new(Matrix::<bool>::new(nrows, ncols).into_hyper()),
            mt: VersionedMatrix::<bool>::new(0, 0),
            me,
            multi_count,
        })
    }
}

/// Owned edge-id iterator for one `(src, dest)` pair (see [`Tensor::get`]).
/// The common single-edge case is allocation-free; multi-edge pairs stream
/// their `me` row live (the inner iterator keeps the underlying GraphBLAS
/// matrix alive via its own `Arc`, so no borrow of the tensor is held).
pub enum EdgeIds {
    Inline(std::option::IntoIter<u64>),
    Multi(versioned_matrix::Iter<BoolExtract>),
}

impl Iterator for EdgeIds {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        match self {
            Self::Inline(it) => it.next(),
            Self::Multi(it) => it.next().map(|(_, edge_id)| edge_id),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Inline(it) => it.size_hint(),
            Self::Multi(it) => it.size_hint(),
        }
    }
}

/// Base adjacency iterator. Forward iteration streams inline values directly
/// from the effective forward adjacency; backward iteration streams the BOOL
/// structure of `mt` (which carries no ids) and recovers each inline value
/// via `eff_get`.
enum BaseIter {
    Forward(versioned_matrix::Iter<Uint64Extract>),
    Backward(versioned_matrix::Iter<BoolExtract>),
}

pub struct Iter<'a> {
    t: &'a Tensor,
    base: BaseIter,
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
                BaseIter::Forward(t.fwd_iter(min_row, max_row))
            },
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
        // Drain buffered (ascending) edge ids for the current multi-edge pair.
        if self.buf_pos < self.buf.len() {
            let id = self.buf[self.buf_pos];
            self.buf_pos += 1;
            return Some((self.src, self.dest, id));
        }

        // Next base pair, oriented as (src, dest) with its inline value.
        // Forward carries the value inline; backward recovers it via eff_get.
        let (src, dest, inline) = match &mut self.base {
            BaseIter::Forward(it) => {
                let (row, col, id) = it.next()?;
                (row, col, id)
            }
            BaseIter::Backward(it) => {
                let (row, col) = it.next()?;
                let (src, dest) = (col, row);
                let id = self.t.eff_get(src, dest).unwrap_or(0);
                (src, dest, id)
            }
        };
        self.src = src;
        self.dest = dest;

        if inline == MULTI_EDGE {
            // Multi-edge pair: all ids live in `me`, already in ascending
            // column (edge-id) order.
            let key = compound_key(self.src, self.dest);
            self.buf.clear();
            for (_, id) in self.t.me.iter(key, key) {
                self.buf.push(id);
            }
            let id = self.buf[0];
            self.buf_pos = 1;
            return Some((self.src, self.dest, id));
        }
        Some((self.src, self.dest, inline))
    }
}
