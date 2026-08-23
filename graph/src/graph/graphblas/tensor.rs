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
//!   pair is one of `multi_pairs()`. Otherwise its single id is inline and
//!   `me` has no row for it.
//! - `mt` mirrors the effective forward structure (BOOL, no ids).
//! - `edge_count = |m| + |dp| − |dm| − |dp ∩ m| − multi_pairs() + |me|`,
//!   where `multi_pairs()` is read from `me`'s row structure, not tracked.
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
use smallvec::{SmallVec, smallvec};
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::graph::{
    cow::Cow,
    graphblas::matrix::{BoolExtract, Uint64Extract},
};

use super::{
    matrix::{Descriptor, Dup, Matrix},
    serialization::{Decode, Encode, Reader, Writer},
    vector::Vector,
    versioned_matrix::{
        self, Delta, VersionedMatrix, delta_dominates_base, should_fold, should_fold_read,
    },
};

/// Maximum GraphBLAS index value (2^60 - 1).
#[allow(non_upper_case_globals)]
pub const GrB_INDEX_MAX: u64 = (1u64 << 60) - 1;

/// Row dimension of every `me` block: one more than the largest key
/// [`compound_key`] can produce, because a matrix of `n` rows indexes `0..n-1`.
///
/// `compound_key` emits exactly `GrB_INDEX_MAX` when both endpoint halves are
/// all-ones, so declaring `GrB_INDEX_MAX` rows is one short and that write is
/// dropped — silently, in release, which is the very failure this module exists
/// to remove. Pinned by `the_top_row_key_of_a_block_is_writable`.
const ME_DIM: u64 = GrB_INDEX_MAX + 1;

/// Column count `me` is created with: the widest declaration for which
/// GraphBLAS still stores column indices in 32 bits rather than 64.
///
/// Columns are edge ids, and the column-index array holds one entry per stored
/// id — the largest array in the structure. Measured on 200k ids: 6.44 B/id
/// here against 11.69 at `GrB_INDEX_MAX`, so the declaration alone cost 45%.
///
/// The cutoff is `2^31`, not `2^32`: the dimension itself must fit a `u32`. A
/// tensor whose edge ids reach it widens back ([`Tensor::widen_me_for_id`]) and
/// pays the old price, so the narrow default costs only the check.
///
/// Rows stay at [`ME_DIM`]. Narrowing them saves a further 0.40 B/id — row
/// arrays hold one entry per multi-edge *pair*, not per id — and would need
/// [`BLOCK_SHIFT`] at 15, i.e. a block per 32,768 nodes per axis.
const ME_NARROW_NCOLS: u64 = 1 << 31;

/// Bits of each node id carried in the compound row key. Everything above them
/// selects a *block* instead (see [`compound_key`]).
///
/// Two halves must fit one GraphBLAS index, so `2 * BLOCK_SHIFT <= 60`; 30 is
/// the maximum, and therefore the fewest blocks. This is not a limit on node
/// ids — it is where one block's key space ends and the next begins.
pub const BLOCK_SHIFT: u32 = 30;
const BLOCK_MASK: u64 = (1u64 << BLOCK_SHIFT) - 1;

/// Which `me` matrix a pair's edge ids live in — the high bits of each
/// endpoint, per [`compound_key`].
///
/// [`ME_BLOCK_0`] for every graph under 2^[`BLOCK_SHIFT`] nodes per axis, which
/// is every graph in practice, and [`Tensor`] keeps its blocks in a `SmallVec`
/// sized for exactly one: a graph inside that block pays a compare against the
/// first (and only) entry, with no hashing and nothing on the heap.
pub type MeBlock = (u64, u64);

/// The block every pair within [`BLOCK_SHIFT`] bits per axis lands in, and the
/// one [`Tensor`] always holds at index 0 of its block list.
pub const ME_BLOCK_0: MeBlock = (0, 0);

/// Split a `(src, dst)` node-id pair into the `me` block that holds its edge
/// ids and the row key within that block.
///
/// Each endpoint is cut in two: the low [`BLOCK_SHIFT`] bits pack into the row
/// key, the high bits become the block coordinate.
///
/// ```text
/// src  [ src_hi ............ ][ src_lo : 30 ]
/// dst  [ dst_hi ............ ][ dst_lo : 30 ]
///             │                      │
///             ▼                      ▼
///  block (src_hi, dst_hi)    row [ src_lo : 30 | dst_lo : 30 ]  <- 60 bits, always
/// ```
///
/// Equivalently: a tiling of the `src x dst` id space into 2^30 x 2^30 tiles,
/// where the block names the tile and the row names the cell inside it.
///
/// Because the key is built only from masked halves it is in range for
/// [`ME_DIM`] whatever the input, so there is no bound to check and nothing to
/// panic on. Ids too large for one tile move to another rather than truncate.
///
/// It replaces `(src << 32) | dst`, which collided above `dst = 2^32` and went
/// out of range above `src = 2^28` — where the write was silently dropped,
/// leaving a pair tagged [`MULTI_EDGE`] over an empty row, i.e. Invariant
/// *promotion completeness* broken and edges vanishing on read. The guard that
/// was there checked `2^32` and so never fired. See
/// `promotion_survives_node_ids_past_the_old_key_width`.
#[inline]
#[must_use]
pub fn compound_key(
    src: u64,
    dst: u64,
) -> (MeBlock, u64) {
    (
        (src >> BLOCK_SHIFT, dst >> BLOCK_SHIFT),
        ((src & BLOCK_MASK) << BLOCK_SHIFT) | (dst & BLOCK_MASK),
    )
}

/// Rebuild `(src, dst)` from a block and a row key — the inverse of
/// [`compound_key`], which iteration needs to learn which pair an `me` entry
/// belongs to.
///
/// Each endpoint is reassembled from its high half (the block) and its low half
/// (the row key). Exact for every pair [`compound_key`] accepts; injective only
/// for `row < ME_DIM`, which holds for any key it produced.
#[inline]
#[must_use]
pub fn compound_key_inverse(
    block: MeBlock,
    row: u64,
) -> (u64, u64) {
    (
        (block.0 << BLOCK_SHIFT) | (row >> BLOCK_SHIFT),
        (block.1 << BLOCK_SHIFT) | (row & BLOCK_MASK),
    )
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
    dp: Delta<u64>,
    /// Delta-minus: edges removed in current transaction (always a bool mask)
    dm: Delta<bool>,
    /// Backward adjacency (dst → src), BOOL structure only. Edge ids are never
    /// stored here — they are recovered from `m` (and `me`) when iterating
    /// incoming edges, avoiding a redundant copy of every id.
    mt: VersionedMatrix<bool>,
    /// Multi-edge id storage, one matrix per live block, keyed within a block by
    /// the row half of `compound_key(src, dst)` → edge_id (BOOL). Holds *all*
    /// ids of pairs with more than one edge; empty otherwise.
    ///
    /// [`ME_BLOCK_0`] is always entry 0, and a second entry appears only if a
    /// pair ever lands outside it — i.e. only above 2^30 nodes on one axis, so
    /// never in practice. Sizing the `SmallVec` for one keeps that case free:
    /// the list is inline, lookup is a compare against the first entry, and an
    /// MVCC version clones it without touching the heap. A map would hash on
    /// every access and allocate on every version, to hold one entry.
    me: SmallVec<[(MeBlock, VersionedMatrix<bool>); 1]>,
    /// Whether the fold decisions latched on `dp`/`dm` are executable now; see
    /// `VersionedMatrix`'s field of the same name.
    needs_flush: AtomicBool,
}

/// Sentinel stored as the inline forward value of a pair with more than one
/// edge; the pair's real edge ids all live in `me`. Real edge ids can never
/// collide with it (they are bounded by [`GrB_INDEX_MAX`]).
pub const MULTI_EDGE: u64 = u64::MAX;

/// What [`Tensor::remove_all`]'s read phase has worked out about one
/// `(src, dst)` pair: the state it is in after replaying the batch's removals
/// against the state the layers still hold, and hence what the write phase owes
/// it. Named after the per-pair states in the [module docs](self).
enum PairPlan {
    /// Effective inline value is [`MULTI_EDGE`]: the ids still left in the
    /// pair's `me` row, ascending. At least two — reaching one demotes the pair.
    Multi(Vec<u64>),
    /// One edge left, held inline. `demoted` marks the id as having arrived
    /// there by a demotion in this batch, which the write phase must still put
    /// in the inline slot; otherwise the pair was already single and untouched.
    Single { id: u64, demoted: bool },
    /// Emptied by this batch: its last edge was removed.
    Emptied,
    /// No edges when the batch reached it (absent, or deleted earlier).
    Absent,
}

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
            needs_flush: AtomicBool::new(self.needs_flush.load(Ordering::Relaxed)),
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
            dp: Delta::new(Matrix::<u64>::new(nrows, ncols)),
            dm: Delta::new(Matrix::<bool>::new(nrows, ncols)),
            mt: VersionedMatrix::<bool>::new(ncols, nrows),
            me: smallvec![(
                ME_BLOCK_0,
                VersionedMatrix::<bool>::new(ME_DIM, ME_NARROW_NCOLS)
            )],
            needs_flush: AtomicBool::new(false),
        }
    }

    /// The `me` matrix holding `block`, or `None` if no pair has ever landed in
    /// it. Reads take this: an absent block holds no ids, which is the same
    /// answer an empty matrix would give without allocating one.
    ///
    /// A scan rather than a lookup because the list is one entry long for every
    /// real graph — [`ME_BLOCK_0`] is entry 0 — so this is a compare and a
    /// return, where hashing would be a hash and a probe to reach the same
    /// place. It is linear in blocks alone, and 64 of those is a node-id span
    /// of ~69 billion.
    #[inline]
    fn me_block(
        &self,
        block: MeBlock,
    ) -> Option<&VersionedMatrix<bool>> {
        self.me.iter().find(|(b, _)| *b == block).map(|(_, me)| me)
    }

    /// The `me` matrix holding `block`, creating it if this is the first pair to
    /// reach it. Writes take this.
    ///
    /// Every block is [`ME_DIM`] rows like the original single matrix, and
    /// hypersparse, so an empty one costs the fixed matrix header rather than
    /// anything proportional to its dimensions.
    fn me_block_mut(
        &mut self,
        block: MeBlock,
    ) -> &mut VersionedMatrix<bool> {
        let at = match self.me.iter().position(|(b, _)| *b == block) {
            Some(at) => at,
            None => {
                // A new block is created at whatever width the tensor already
                // has, so one that has widened does not go on creating narrow
                // blocks behind it and re-widening them one at a time.
                let ncols = self.me[0].1.ncols();
                self.me
                    .push((block, VersionedMatrix::<bool>::new(ME_DIM, ncols)));
                self.me.len() - 1
            }
        };
        &mut self.me[at].1
    }

    /// Widen every `me` block's column space if `max_id` will not fit.
    ///
    /// Called once per batch with the batch's largest edge id, so the common
    /// path is one comparison against [`ME_BLOCK_0`], whose width every later
    /// block inherits. Widening is one-way and rare: edge ids come from a
    /// counter, so a tensor crosses `ME_NARROW_NCOLS` at most once.
    fn widen_me_for_id(
        &mut self,
        max_id: u64,
    ) {
        if max_id < self.me[0].1.ncols() {
            return;
        }
        for (_, me) in &mut self.me {
            me.resize(ME_DIM, ME_DIM);
        }
    }

    /// Wait pending GraphBLAS work on the forward delta layers. The committed
    /// base `m` never holds real pending work: it is only mutated by `flush`
    /// (which waits) and `resize` (GrB_Matrix_resize waits internally), so it
    /// is never waited here (same invariant as `VersionedMatrix::wait`). Note
    /// `GxB_WILL_WAIT` can still report true on `m` after a grow-resize — the
    /// hyper hash was freed — but GraphBLAS rebuilds it on demand.
    pub(crate) fn wait_fwd(&self) {
        // nvals only changes through ops that mark the matrix pending, so the
        // flag can only be stale when there is pending work — skip the nvals
        // FFI calls on the hot read path otherwise.
        if self.dp.is_synced() && self.dm.is_synced() {
            return;
        }
        // `resync` materializes each delta and pins its approximate counter to
        // the exact count, so the policy below weighs exact sizes.
        self.dp.resync();
        self.dm.resync();
        let base = self.m.nvals();
        self.dp.latch(self.dp.fold_decision(should_fold_read, base));
        self.dm.latch(self.dm.fold_decision(should_fold_read, base));
        // Not setting needs_flush: see `VersionedMatrix::wait` — mid-tx fold
        // execution is pathological for create+delete transactions; `dup`
        // carries the latched decision into the next version instead.
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
                let (block, key) = compound_key(src, dest);
                // An absent block holds no identifiers. Promotion completeness
                // says that cannot happen for a pair reading as MULTI_EDGE, so
                // this arm is unreachable rather than a fallback — but an empty
                // iterator is the honest answer if it ever is reached, and it is
                // what an empty matrix would have returned anyway.
                self.me_block(block).map_or_else(
                    || EdgeIds::Inline(None.into_iter()),
                    |me| EdgeIds::Multi(me.iter(key, key)),
                )
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

        // `me` is created with a narrow column space so GraphBLAS stores 32-bit
        // column indices; an id past it widens every block first, once, before
        // any of them is written.
        if let Some(&max_id) = ids.iter().max() {
            self.widen_me_for_id(max_id);
        }

        self.flush();
        // `flush` no-ops unless a fold was latched; materialize the deltas
        // explicitly (single atomic load each when already synced) so the
        // read phase's per-edge lookups never trigger implicit GraphBLAS
        // waits. `m` is never pending (see `wait_fwd`).
        self.dp.wait();
        self.dm.wait();
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
            let (block, key) = compound_key(s, d);
            match batch.entry((s, d)) {
                Entry::Occupied(mut e) => {
                    let idx = *e.get();
                    if idx != usize::MAX {
                        // Second edge of a pair new in this batch: promote the
                        // pending inline slot in place.
                        let me = self.me_block_mut(block);
                        me.set(key, m_ids[idx], true);
                        me.set(key, id, true);
                        m_ids[idx] = MULTI_EDGE;
                        e.insert(usize::MAX);
                    } else {
                        self.me_block_mut(block).set(key, id, true);
                    }
                }
                Entry::Vacant(e) => {
                    let masked = !dm_empty && self.dm.contains(s, d);
                    let from_dp = self.dp.get(s, d);
                    let cur = from_dp.or_else(|| if masked { None } else { self.m.get(s, d) });
                    match cur {
                        // Already multi-edge: just add the id to `me`.
                        Some(MULTI_EDGE) => {
                            self.me_block_mut(block).set(key, id, true);
                            e.insert(usize::MAX);
                        }
                        // Present single edge: promote — move the existing
                        // inline id to `me` alongside the new one, and queue
                        // the sentinel for the inline slot.
                        Some(cur_id) => {
                            let me = self.me_block_mut(block);
                            me.set(key, cur_id, true);
                            me.set(key, id, true);
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
                self.dm.erase(s, d);
                if committed == id {
                    // Cancel to clean: committed value restored. Drop any dp
                    // shadow (re-promotion of a demoted committed-multi pair;
                    // both erases are no-ops when the entry is absent).
                    self.dp.erase(s, d);
                    continue;
                }
            }
            self.dp.insert(s, d, id);
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
        self.flush();

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
            // dm<mask> = mask ∩ m: mark every committed entry the mask selects
            // as deleted (eWiseMult's PAIR never reads m's u64 values — an
            // eWiseAdd copy would typecast edge id 0 to `false`, which valued
            // masks then skip); dp &= ¬mask: drop pending adds (including the
            // shadow value of any in-place-updated pair, whose committed entry
            // the dm update just masked — keeping `dp ∩ dm = ∅`).
            self.dm.tombstone_masked(&m_mask, &self.m);
            self.dp.remove_all(&m_mask);
            self.mt.remove_mask(&mt_mask);
            return rels.iter().map(|&(_, src, dst)| (src, dst)).collect();
        }

        // Slow path: some pairs have multiple edges. Handle per edge:
        //  - single-edge pair: delete the pair.
        //  - multi-edge pair (inline == MULTI_EDGE): drop the id from `me`;
        //    when exactly one id remains, demote it back into the inline slot.
        //
        // Split into a read phase and a write phase, for the same reason as
        // `set_all_from_slices`. Mutating a layer *and* reading it back in the
        // same iteration is what makes a batch quadratic: a `set`/`remove`
        // leaves GraphBLAS tuples pending, and the next iteration's read has to
        // materialize them — an `O(|layer|)` merge (plus an OpenMP fan-out) per
        // edge. Measured on a 1,000-pair demote batch: 9.9 ms of the 13.2 ms
        // spent in this function went to the `me.iter` right after `me.remove`,
        // and another 2.9 ms to the `eff_get` right after the `dp` write.
        //
        // So the read phase reads `dp`/`dm`/`me` and writes none of them: it
        // replays each touched pair's transitions in `plans` and buffers the
        // `me` removals. The write phase then writes without reading those
        // layers back (probing only the committed bases, which never carry
        // pending work).
        self.wait_fwd();
        let mut plans: FxHashMap<(u64, u64), PairPlan> = FxHashMap::default();
        // `me` entries to drop, in discovery order.
        let mut me_del: Vec<(MeBlock, u64, u64)> = Vec::new();
        let mut emptied = Vec::new();
        for &(id, src, dst) in rels {
            let (block, key) = compound_key(src, dst);
            let plan = match plans.entry((src, dst)) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => e.insert(match self.eff_get(src, dst) {
                    // All of a MULTI pair's ids live in `me`; read the row once
                    // and take every removal for this pair out of that list
                    // instead of re-reading a row we are about to dirty.
                    Some(MULTI_EDGE) => {
                        let ids: Vec<u64> = self
                            .me_block(block)
                            .map(|me| me.iter(key, key).map(|(_, id)| id).collect())
                            .unwrap_or_default();
                        debug_assert!(
                            ids.windows(2).all(|w| w[0] < w[1]),
                            "`me` row ({src}, {dst}) not ascending; the search below needs it"
                        );
                        PairPlan::Multi(ids)
                    }
                    Some(inline) => PairPlan::Single {
                        id: inline,
                        demoted: false,
                    },
                    None => PairPlan::Absent,
                }),
            };
            match plan {
                PairPlan::Multi(ids) => {
                    // Unknown id (another pair's, or already removed by an
                    // earlier duplicate in this batch): nothing to remove.
                    let Ok(pos) = ids.binary_search(&id) else {
                        continue;
                    };
                    ids.remove(pos);
                    me_del.push((block, key, id));
                    match ids.len() {
                        // A MULTI pair keeps *all* of its ids in `me` and has
                        // at least two of them, so removing one always leaves
                        // a survivor and this pair cannot empty here — a
                        // single-edge pair holds its id inline and is handled
                        // by the arm below. Machine-checked as
                        // `removeOne_survivor` in `proofs/tensor`, which is
                        // also what retired the "all ids removed at once"
                        // branch this replaced.
                        0 => unreachable!("MULTI pair ({src}, {dst}) held one id in `me`"),
                        // Down to one edge: the pair demotes. Its `me` row
                        // empties and the survivor goes back inline (in the
                        // write phase); a later removal of that survivor now
                        // sees a single-edge pair, exactly as if the demotion
                        // had already been written.
                        1 => {
                            let last = ids[0];
                            me_del.push((block, key, last));
                            *plan = PairPlan::Single {
                                id: last,
                                demoted: true,
                            };
                        }
                        _ => {}
                    }
                }
                PairPlan::Single { id: inline, .. } if *inline == id => {
                    *plan = PairPlan::Emptied;
                    emptied.push((src, dst));
                }
                // Unknown id, or a pair already emptied / never there.
                PairPlan::Single { .. } | PairPlan::Emptied | PairPlan::Absent => {}
            }
        }

        // Write phase: `me` first, then the forward/backward layers. Every
        // probe here reads a committed base (`m`, and `me`/`mt`'s own bases,
        // which `VersionedMatrix::remove` is careful to be the only thing it
        // reads) — never a layer this phase writes, so no write materializes
        // another write's pending tuples. `dp` erases are batched ahead of
        // `dp` inserts for the same reason: a `GrB_Matrix_removeElement` that
        // finds nothing may finish the matrix to be sure.
        for &(block, key, id) in &me_del {
            self.me_block_mut(block).remove(key, id);
        }
        let mut dp_set: Vec<(u64, u64, u64)> = Vec::new();
        for (&(src, dst), plan) in &plans {
            match plan {
                PairPlan::Emptied => {
                    self.dp.erase(src, dst);
                    if self.m.contains(src, dst) {
                        self.dm.insert(src, dst);
                    }
                    self.mt.remove(dst, src);
                }
                // Demote: the surviving id returns inline. If it *is* the
                // committed value, the deltas cancel and the pair returns
                // clean; otherwise `dp` shadows `m` with the live value (no
                // `dm` mask). `mt` already holds (dst, src): the pair survives.
                PairPlan::Single { id, demoted: true } => {
                    if self.m.get(src, dst) == Some(*id) {
                        self.dp.erase(src, dst);
                    } else {
                        dp_set.push((src, dst, *id));
                    }
                }
                // Untouched: still multi, or a single-edge pair whose inline
                // id this batch never named.
                PairPlan::Multi(_) | PairPlan::Single { demoted: false, .. } | PairPlan::Absent => {
                }
            }
        }
        for &(src, dst, id) in &dp_set {
            self.dp.insert(src, dst, id);
        }
        emptied
    }

    pub fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) {
        if nrows < self.m.nrows() || ncols < self.m.ncols() {
            // Shrinking can drop entries; keep the straightforward path.
            self.flush();
            // GrB_Matrix_resize waits internally, so `m` holds no real pending
            // work afterwards — waiting here would rebuild the freed hyper hash
            // on every capacity grow (measured 1.4-5.7x write regressions).
            self.m.resize(nrows, ncols);
            // A resize moves no entries between layers, so the delta counters
            // stand (a shrink that drops entries only adds to the drift
            // `resync` bounds).
            self.dp.resize(nrows, ncols);
            self.dm.resize(nrows, ncols);
            self.mt.resize(ncols, nrows);
            return;
        }
        // Growing: the base is always COW-shared with the committed snapshot,
        // so resizing through the Cow would deep-copy the full matrix at the
        // old dims and then rewrite it. Instead re-emit each layer at the
        // target dims and swap it in; contents (and therefore all delta
        // invariants, counters and fold latches) are unchanged.
        //
        // The layers may be shared with the committed snapshot AND carry
        // pending work (commit does not wait). Any GraphBLAS call on a pending
        // matrix materializes it — a mutation — so wait first under the
        // readers' lock or this races concurrent readers (GrB_INVALID_OBJECT /
        // heap corruption under stress).
        self.m.wait();
        self.dp.wait();
        self.dm.wait();
        // `grown` re-emits each layer at the target dims as a `dup` plus a
        // `GrB_Matrix_resize`. Measured against the row-iterate +
        // `GrB_Matrix_build` rebuild it replaced (`grow_cost_rebuild_vs_resize`,
        // uint64 layer, 1.14x dims): 0.47 ms against 3.7 ms at 262k entries and
        // 0.81 ms against 9.0 ms at 1m, and cheaper than the `GxB_Matrix_concat`
        // formulation that sat here before at every size measured. An empty
        // delta still skips it entirely rather than growing a matrix with
        // nothing in it.
        let new_m = self.m.grown(nrows, ncols);
        new_m.wait();
        self.m.replace(new_m);
        let new_dp = if self.dp.nvals() > 0 {
            self.dp.grown(nrows, ncols)
        } else {
            Matrix::<u64>::new(nrows, ncols)
        };
        self.dp.replace(new_dp);
        let new_dm = if self.dm.nvals() > 0 {
            self.dm.grown(nrows, ncols)
        } else {
            Matrix::<bool>::new(nrows, ncols)
        };
        self.dm.replace(new_dm);
        self.mt.resize(ncols, nrows);
    }

    /// Merge oversized deltas into the committed base (same fold policy as
    /// [`VersionedMatrix::flush`], see `should_fold`): `dp` is folded into
    /// `m` via a value-preserving eWiseAdd (`SECOND`, so `dp` wins on
    /// shadowed pairs) and cleared; `dm`'s masked entries are removed from
    /// `m` and it is cleared. `dp ∩ dm = ∅`, so the two merges are
    /// order-independent and all invariants are preserved. `mt` and `me`
    /// flush themselves.
    ///
    /// Gated on the `needs_flush` flag, which is set by `dup` (carrying the
    /// decision `wait_fwd` latched) or by `fold_latched`, so calling this on
    /// every write is a single atomic load in the common case.
    pub fn flush(&mut self) {
        if self.needs_flush.load(Ordering::Relaxed) {
            // The layers may be shared with the committed snapshot and carry
            // pending work; nvals/eWiseAdd/select below materialize it (a
            // mutation), so wait first under the readers' lock.
            self.m.wait();
            self.dp.wait();
            self.dm.wait();
            let fold_dp = self.dp.take_fold();
            let fold_dm = self.dm.take_fold();
            if fold_dp || fold_dm {
                let nrows = self.m.nrows();
                let ncols = self.m.ncols();
                // Always build the folded base into a fresh matrix and swap
                // it in. When `m` is shared with the committed snapshot an
                // in-place fold would deep-copy it (a full O(|m|) memcpy)
                // first; when it is not, GraphBLAS materializes the eWiseAdd
                // result in a temporary anyway, so the fresh build costs the
                // same — and under full MVCC the base is always shared.
                let mut new_m = Matrix::<u64>::new(nrows, ncols);
                match (fold_dp, fold_dm) {
                    // new_m<!dm, replace> = m ⊕ dp (dp wins on shadowed
                    // pairs); dp ∩ dm = ∅, so no pending add is lost.
                    (true, true) => new_m.element_wise_add(
                        Some(&self.dm),
                        Some(&self.m),
                        Some(&*self.dp),
                        Some(Descriptor::RC),
                    ),
                    (true, false) => {
                        new_m.element_wise_add(None, Some(&self.m), Some(&*self.dp), None);
                    }
                    // new_m<!dm, replace> = m
                    (false, true) => new_m.select(&self.dm, &self.m),
                    (false, false) => unreachable!(),
                }
                new_m.wait();
                self.m.replace(new_m);
                if fold_dp {
                    self.dp.clear(nrows, ncols);
                }
                if fold_dm {
                    self.dm.clear(nrows, ncols);
                }
            }
            self.needs_flush.store(false, Ordering::Relaxed);
        }
        self.mt.flush();
        for (_, me) in &mut self.me {
            me.flush();
        }
    }

    /// Latch the fold decision from the current (materialized) delta sizes
    /// and execute it immediately, on the forward layers and on `mt`/`me`.
    /// Only safe once a transaction has finished mutating — e.g. the end of
    /// a GRAPH.BULK command; see [`VersionedMatrix::fold_latched`].
    pub fn fold_latched(&mut self) {
        self.wait_fwd();
        if self.dp.folding() || self.dm.folding() {
            self.needs_flush.store(true, Ordering::Relaxed);
            self.flush();
        }
        self.mt.fold_latched();
        for (_, me) in &mut self.me {
            me.fold_latched();
        }
    }

    /// Fold forward-layer deltas that have grown comparable to the base, plus
    /// the backward and multi-edge matrices' own; see
    /// [`VersionedMatrix::fold_oversized`] for why commit is where this
    /// happens. A delete-everything is the case that needs it: every deleted
    /// edge leaves a `dm` tombstone shadowing a base entry, so the tensor
    /// holds both copies until the next transaction touching it.
    pub fn fold_oversized(&mut self) {
        let base = self.m.nvals();
        // Only the escape hatch arms the flush; a decision `wait_fwd` left
        // latched stays deferred (see `VersionedMatrix::fold_oversized`).
        let oversized_dp = delta_dominates_base(self.dp.count(), base);
        let oversized_dm = delta_dominates_base(self.dm.count(), base);
        if oversized_dp || oversized_dm {
            self.dp.latch(oversized_dp);
            self.dm.latch(oversized_dm);
            self.needs_flush.store(true, Ordering::Relaxed);
            self.flush();
        }
        self.mt.fold_oversized();
        for (_, me) in &mut self.me {
            me.fold_oversized();
        }
    }

    /// Materialize the effective forward structure as a `bool` matrix:
    /// `(m ∖ dm) ∪ dp`, values discarded. Shadowed pairs (`dp ∩ m`) collapse
    /// in the bool union; `dm ⊆ m` is disjoint from `dp`, so order is safe.
    #[must_use]
    pub fn extract(&self) -> Matrix<bool> {
        self.wait_fwd();
        let mut m = Matrix::<bool>::new(self.m.nrows(), self.m.ncols());
        m.set_pattern(None, &*self.m, None);
        if self.dm.nvals() > 0 {
            m.remove_all(&self.dm);
        }
        if self.dp.nvals() > 0 {
            m.set_pattern(None, &*self.dp, None);
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

    /// See `VersionedMatrix::dup`: version creation is the pre-mutation hook
    /// where the fold decision is made (using the finished transaction's
    /// contribution as the next one's predictor) and latched for `flush`.
    /// Delta sizes come from the approximate counters — reading `nvals` here
    /// would force each mutated delta's pending tuples to merge on every
    /// write transaction.
    #[must_use]
    pub fn dup(&self) -> Self {
        let base = self.m.nvals();
        let fold_dp = self.dp.fold_decision(should_fold, base);
        let fold_dm = self.dm.fold_decision(should_fold, base);
        Self {
            m: self.m.new_version(),
            dp: self.dp.new_version(fold_dp),
            dm: self.dm.new_version(fold_dm),
            mt: self.mt.dup(),
            me: self.me.iter().map(|(b, me)| (*b, me.dup())).collect(),
            needs_flush: AtomicBool::new(fold_dp || fold_dm),
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

    /// Multi-edge id storage for block [`ME_BLOCK_0`], keyed by the row half of
    /// `compound_key(src, dst)` → edge id. Holds all ids of pairs with more
    /// than one edge; empty unless some pair has one.
    ///
    /// Callers that must see *every* id — rather than every id of a graph
    /// within 2^30 nodes per axis — want [`Self::edge_versioned_all`].
    #[must_use]
    pub fn edge_versioned_block_0(&self) -> &VersionedMatrix<bool> {
        &self.me[0].1
    }

    /// Every `me` matrix with the block it holds, so a caller doing its own
    /// GraphBLAS work over the multi-edge ids can cover all of them and map row
    /// keys back to pairs with [`compound_key_inverse`].
    pub fn edge_versioned_all(&self) -> impl Iterator<Item = (MeBlock, &VersionedMatrix<bool>)> {
        self.me.iter().map(|(b, me)| (*b, me))
    }

    /// Total number of edges. Each effective forward entry is one edge,
    /// except `MULTI_EDGE` sentinels (see [`Self::multi_pairs`]), whose real
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
        self.m.nvals() + self.dp.nvals() - self.dm.nvals() - shadow - self.multi_pairs()
            + self.me.iter().map(|(_, me)| me.nvals()).sum::<u64>()
    }

    /// Iterate every `(src, dst, edge_id)` triple in the tensor.
    ///
    /// Yields the inline single edge of every pair from the effective UINT64
    /// forward adjacency (skipping `MULTI_EDGE` sentinels), followed by all
    /// multi-edge ids from `me`. On a single-edge graph `me` is empty, so
    /// this is a single streaming pass with no per-pair sub-iterator.
    pub fn iter_edges(&self) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
        let multi: Box<dyn Iterator<Item = (u64, u64, u64)> + '_> = if self.has_multi_edge() {
            Box::new(self.edge_versioned_all().flat_map(|(block, me)| {
                me.iter(0, GrB_INDEX_MAX).map(move |(key, edge_id)| {
                    let (src, dst) = compound_key_inverse(block, key);
                    (src, dst, edge_id)
                })
            }))
        } else {
            Box::new(std::iter::empty())
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

    /// How many pairs are multi-edge, computed from `me` rather than tracked.
    ///
    /// Promotion completeness makes this a structural property of `me`:
    /// `ε(p) = MULTI_EDGE` exactly when row `κ(p)` of `me` is non-empty. A
    /// hypersparse matrix stores its non-empty vectors and nothing else, so once
    /// `me` is assembled the answer is a metadata read rather than a count.
    ///
    /// While `me` carries live deltas the effective row set spans three layers
    /// and has to be walked, which is `O(multi)` — never `O(|E|)`. That is the
    /// price of not maintaining a counter, and the reason this is called from
    /// [`Self::edge_count`] (statistics, planning, two graph algorithms) and not
    /// from anything per-row.
    #[cfg(test)]
    fn eff_get_for_test(
        &self,
        src: u64,
        dst: u64,
    ) -> Option<u64> {
        self.eff_get(src, dst)
    }

    #[cfg(test)]
    fn fwd_me_nvals_for_test(&self) -> u64 {
        self.me.iter().map(|(_, me)| me.nvals()).sum()
    }

    #[must_use]
    pub fn multi_pairs(&self) -> u64 {
        // Blocks partition the pairs — a pair's identifiers live in exactly one
        // block — so the per-block counts simply add. Rows are counted within a
        // block, never across, which is why this is a sum and not a merge.
        self.me.iter().map(|(_, me)| Self::multi_pairs_in(me)).sum()
    }

    /// [`Self::multi_pairs`] for one block.
    fn multi_pairs_in(me: &VersionedMatrix<bool>) -> u64 {
        // A block with no multi-edge pair — which is every block of the
        // dominant single-edge case — answers from `nvals` without waiting it
        // or attaching an iterator to it.
        if me.nvals() == 0 {
            return 0;
        }
        me.wait();
        if me.dp().nvals() == 0
            && me.dm().nvals() == 0
            && let Some(kount) = me.m().hyper_vector_count()
        {
            return kount;
        }
        let mut rows = 0u64;
        let mut last: Option<u64> = None;
        for (key, _) in me.iter(0, u64::MAX) {
            if last != Some(key) {
                rows += 1;
                last = Some(key);
            }
        }
        rows
    }

    /// Whether this tensor has any (src, dst) pair with more than one edge.
    /// All ids of such pairs live in `me`, so this is simply `me` being
    /// non-empty.
    #[must_use]
    pub fn has_multi_edge(&self) -> bool {
        self.me.iter().any(|(_, me)| me.nvals() != 0)
    }

    pub fn wait(&self) {
        self.wait_fwd();
        self.mt.wait();
        for (_, me) in &self.me {
            me.wait();
        }
    }

    /// Materialize only the committed base layers (`m`, `mt.m`, `me.m`).
    /// See [`VersionedMatrix::wait_base`] for why bases must be synced at
    /// MVCC commit while dp/dm may stay lazy.
    pub fn wait_base(&self) {
        self.m.wait();
        self.mt.wait_base();
        for (_, me) in &self.me {
            me.wait_base();
        }
    }

    /// Wait on all matrices for fork safety (takes &self, not &mut self).
    pub fn wait_all(&self) {
        self.m.wait();
        self.dp.wait();
        self.dm.wait();
        self.mt.wait_all();
        for (_, me) in &self.me {
            me.wait_all();
        }
    }

    /// Returns true if every internal matrix has no pending GraphBLAS
    /// operations queued.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        self.m.is_synced()
            && self.dp.is_synced()
            && self.dm.is_synced()
            && self.mt.is_synced()
            && self.me.iter().all(|(_, me)| me.is_synced())
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.m.memory_usage()
            + self.dp.memory_usage()
            + self.dm.memory_usage()
            + self.mt.memory_usage()
            + self
                .me
                .iter()
                .map(|(_, me)| me.memory_usage())
                .sum::<usize>()
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
                let (block, key) = compound_key(src, dst);
                let ids: Vec<u64> = self
                    .me_block(block)
                    .map(|me| me.iter(key, key).map(|(_, id)| id).collect())
                    .unwrap_or_default();
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
        // was folded into the base above. The scratch vector is allocated only
        // when there is a multi-edge pair to write — a GraphBLAS vector per
        // tensor is pure overhead on the single-edge graphs that dominate.
        w.write_unsigned(multi.len() as u64);
        if !multi.is_empty() {
            // C stores a pair's id list as a BOOL vector *indexed by edge id*
            // (`GrB_Vector_setElement_BOOL(V, true, edge_id)`), so the ids are
            // the indices, not the values. Writing them as values at positions
            // 0..n-1 made C read the positions back as edge ids — the ids only
            // looked right when they happened to be 0..n-1.
            let mut v = Vector::<bool>::new(GrB_INDEX_MAX);
            for (src, dst, ids) in &multi {
                v.clear();
                for &edge_id in ids {
                    v.set(edge_id, true);
                }
                w.write_unsigned(*src);
                w.write_unsigned(*dst);
                v.encode_blob(w);
            }
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
        let mut me: SmallVec<[(MeBlock, VersionedMatrix<bool>); 1]> = smallvec![(
            ME_BLOCK_0,
            VersionedMatrix::<bool>::new(ME_DIM, ME_NARROW_NCOLS)
        )];
        // Widened lazily, as in `set_all_from_slices`: a blob whose ids all fit
        // decodes into the narrow form and keeps the 32-bit column indices.
        let mut me_ncols = ME_NARROW_NCOLS;

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
                for _ in 0..count {
                    let src = r.read_unsigned()?;
                    let dst = r.read_unsigned()?;
                    // BOOL vector indexed by edge id (see `encode`): the id is
                    // the index. Reading the *values* instead collapsed every
                    // pair written by C to the single edge id 1.
                    let v = Vector::<bool>::decode_blob(r)?;
                    // The blob stores endpoints, not row keys, so the on-disk
                    // form says nothing about how they are blocked and needs no
                    // version bump for this change: re-keying happens here.
                    let (block, key) = compound_key(src, dst);
                    let at = match me.iter().position(|(b, _)| *b == block) {
                        Some(at) => at,
                        None => {
                            me.push((block, VersionedMatrix::<bool>::new(ME_DIM, me_ncols)));
                            me.len() - 1
                        }
                    };
                    // Streamed rather than collected: the ids are only needed
                    // one at a time, and the width check they used to be
                    // gathered for is a compare that fires at most once in the
                    // life of the tensor.
                    for edge_id in v.iter() {
                        if edge_id >= me_ncols {
                            me_ncols = ME_DIM;
                            for (_, b) in &mut me {
                                b.resize(ME_DIM, ME_DIM);
                            }
                        }
                        me[at].1.set(key, edge_id, true);
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
            dp: Delta::new(Matrix::<u64>::new(nrows, ncols)),
            dm: Delta::new(Matrix::<bool>::new(nrows, ncols)),
            mt: VersionedMatrix::<bool>::new(0, 0),
            me,
            needs_flush: AtomicBool::new(false),
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
        loop {
            let (src, dest, inline) = match &mut self.base {
                BaseIter::Forward(it) => {
                    let (row, col, id) = it.next()?;
                    (row, col, id)
                }
                BaseIter::Backward(it) => {
                    let (row, col) = it.next()?;
                    let (src, dest) = (col, row);
                    // `mt` mirrors the effective forward structure, so a pair
                    // reached through it always has a forward inline value.
                    // Machine-checked as `iterBwd_eff_get_isSome` in
                    // `proofs/tensor`. This replaced an `unwrap_or(0)`, which was
                    // worse than unreachable: 0 is a *valid* edge id, so the
                    // fallback would have quietly emitted a fabricated edge
                    // rather than failed.
                    let Some(id) = self.t.eff_get(src, dest) else {
                        unreachable!("mt holds ({src}, {dest}) but the forward matrix does not")
                    };
                    (src, dest, id)
                }
            };
            self.src = src;
            self.dest = dest;

            if inline == MULTI_EDGE {
                // Multi-edge pair: all ids live in `me`, already in ascending
                // column (edge-id) order.
                let (block, key) = compound_key(self.src, self.dest);
                self.buf.clear();
                if let Some(me) = self.t.me_block(block) {
                    for (_, id) in me.iter(key, key) {
                        self.buf.push(id);
                    }
                }
                // Promotion completeness says a MULTI_EDGE pair always has ids in
                // `me`. If one does not — a corrupt or partially decoded tensor —
                // skip the pair rather than panicking on a read path, which is what
                // `get` already does for a missing block.
                let Some(&id) = self.buf.first() else {
                    debug_assert!(false, "MULTI_EDGE pair ({src}, {dest}) has no ids in `me`");
                    continue;
                };
                self.buf_pos = 1;
                return Some((self.src, self.dest, id));
            }
            return Some((self.src, self.dest, inline));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_init::ensure_init;
    use super::*;

    /// Edge id 0 must survive the bool round-trips in the delta layers: the
    /// bulk `remove_all` path derives `dm` from the committed u64 matrix, and
    /// `extract` derives a bool pattern from it. Any op that *typecasts* the
    /// u64 edge id instead of reading only the sparsity pattern turns id 0
    /// into a `false` entry, which valued masks then treat as absent — the
    /// deletion is silently lost.
    /// The bulk loader puts a pair's duplicate edges in ONE batch, which takes
    /// `set_all_from_slices`' retro-promotion path rather than the plain promote.
    #[test]
    fn multi_pairs_after_within_batch_duplicates() {
        ensure_init();
        for &(pairs, dup) in &[(64u64, 2u64), (1_000, 2), (1_000, 4)] {
            let n = pairs + 1;
            let mut t = Tensor::new(n, n);
            // one batch, each pair repeated `dup` times consecutively
            let mut srcs = Vec::new();
            let mut dsts = Vec::new();
            let mut ids = Vec::new();
            let mut next_id = 0u64;
            for i in 0..pairs {
                for _ in 0..dup {
                    srcs.push(i);
                    dsts.push(i + 1);
                    ids.push(next_id);
                    next_id += 1;
                }
            }
            t.set_all_from_slices(&srcs, &dsts, &ids);
            t.wait_fwd();
            let sentinels = (0..pairs)
                .filter(|&i| t.eff_get_for_test(i, i + 1) == Some(MULTI_EDGE))
                .count() as u64;
            let derived = t.multi_pairs();
            let edges: u64 = (0..pairs).map(|i| t.get(i, i + 1).count() as u64).sum();
            println!(
                "one batch: pairs={pairs:>5} dup={dup}  sentinels={sentinels:>5} \
                 derived={derived:>5} edge_count={:>7} true_edges={edges:>7}",
                t.edge_count()
            );
            assert_eq!(
                derived, sentinels,
                "multi_pairs disagrees (within-batch dups)"
            );
            assert_eq!(
                t.edge_count(),
                edges,
                "edge_count disagrees with a full scan"
            );
        }
    }

    /// Ground truth for `multi_pairs`: count the forward cells whose effective
    /// value is the sentinel, which is what the quantity means, and compare
    /// against the derivation from `me`'s row structure.
    #[test]
    fn multi_pairs_matches_the_sentinel_count() {
        ensure_init();
        for &(pairs, dup) in &[(64u64, 2u64), (1_000, 2), (1_000, 3), (5_000, 2)] {
            let n = pairs + 1;
            let mut t = Tensor::new(n, n);
            let srcs: Vec<u64> = (0..pairs).collect();
            let dsts: Vec<u64> = (0..pairs).map(|i| i + 1).collect();
            // `dup` edges on every pair, inserted as separate batches
            for round in 0..dup {
                let ids: Vec<u64> = (0..pairs).map(|i| round * pairs + i).collect();
                t.set_all_from_slices(&srcs, &dsts, &ids);
            }
            // ground truth: effective forward cells holding MULTI_EDGE
            t.wait_fwd();
            let mut sentinels = 0u64;
            for (src, dst) in (0..pairs).map(|i| (i, i + 1)) {
                if t.eff_get_for_test(src, dst) == Some(MULTI_EDGE) {
                    sentinels += 1;
                }
            }
            let derived = t.multi_pairs();
            let edges: u64 = (0..pairs).map(|i| t.get(i, i + 1).count() as u64).sum();
            println!(
                "pairs={pairs:>5} dup={dup}  sentinels={sentinels:>5} \
                 derived={derived:>5} me_nvals={:>6} edge_count={:>7} true_edges={edges:>7}",
                t.fwd_me_nvals_for_test(),
                t.edge_count()
            );
            assert_eq!(
                derived, sentinels,
                "multi_pairs disagrees with the sentinel count"
            );
            assert_eq!(
                t.edge_count(),
                edges,
                "edge_count disagrees with a full scan"
            );
        }
    }

    #[test]
    fn bulk_remove_and_extract_edge_id_zero() {
        ensure_init();
        const N: u64 = 10_000;
        let mut t = Tensor::new(N + 1, N + 1);
        let srcs: Vec<u64> = (0..N).collect();
        let dsts: Vec<u64> = (0..N).map(|i| i + 1).collect();
        let ids: Vec<u64> = (0..N).collect();
        t.set_all_from_slices(&srcs, &dsts, &ids);
        // Fold the pending adds into the committed base so edge id 0 lives
        // in the u64 base matrix. The fold decision is latched at version
        // creation (`dup`) and executed by the next `flush`, which folds via
        // a (possibly pending) eWiseAdd — materialize before the probe.
        let mut t = t.dup();
        t.flush();
        t.fwd_m().wait();
        assert!(t.fwd_m().contains(0, 1), "edge id 0 not folded into base");

        // Bulk-delete edge id 0 (and a nonzero control) via the fast path.
        t.remove_all(&[(0, 0, 1), (5, 5, 6)]);

        assert!(t.get(0, 1).next().is_none(), "edge id 0 still readable");
        assert!(t.get(5, 6).next().is_none(), "edge id 5 still readable");

        let ex = t.extract();
        ex.wait();
        assert!(ex.contains(1, 2), "unrelated live pair (1,2) disappeared");
        assert!(!ex.contains(5, 6), "control pair (5,6) not deleted");
        assert!(
            !ex.contains(0, 1),
            "deleted pair (0,1) still present in extract: edge id 0 was typecast to false in dm"
        );
    }

    /// `resize` must leave the committed base materialized.
    ///
    /// `wait_fwd` asserts `!self.m.pending()` — the base is only written by a
    /// fold, which materializes. `resize` also writes it, and
    /// `GrB_Matrix_resize` can leave pending work, so afterwards every read
    /// path that calls `wait_fwd` panicked. Release compiles the assert out,
    /// so only debug builds saw it.
    ///
    /// Found by `bench/coverage.sh` (a debug build): with a graph loaded from
    /// RDB — which is what makes relationship tensors exist to be resized —
    /// `UNWIND range(1, 100000) AS i CREATE (t:Tmp {x: i}) WITH t DELETE t`
    /// grew node capacity to 114,688 and killed the server via
    /// `Pending::commit` -> `delete_implicit_edges` -> `Tensor::iter`.
    ///
    /// The growth must be large enough to change GraphBLAS's internal
    /// representation: growing 10,001 -> 10,501 leaves nothing pending and the
    /// bug hides.
    #[test]
    fn resize_leaves_base_materialized() {
        ensure_init();
        const N: u64 = 10_000;
        let mut t = Tensor::new(N + 1, N + 1);
        let srcs: Vec<u64> = (0..N).collect();
        let dsts: Vec<u64> = (0..N).map(|i| i + 1).collect();
        let ids: Vec<u64> = (0..N).collect();
        t.set_all_from_slices(&srcs, &dsts, &ids);
        t.flush();
        t.m.wait();

        t.resize(114_688, 114_688);

        assert!(
            !t.m.pending(),
            "resize left the committed base pending; the next wait_fwd panics"
        );
        // Exercise the assert the way the read paths reach it.
        t.wait_fwd();
        // And a read must still see the data.
        assert!(t.get(0, 1).next().is_some(), "edge lost across resize");
    }

    /// Deleting everything must not leave the base *and* a base-sized `dm`
    /// resident. The fold is latched by the delete, but `flush` only runs on
    /// the next mutation of this same tensor — which for a delete-everything
    /// may never come — so `fold_oversized` (MVCC commit) has to apply it.
    /// Measured before the fix: `MATCH (n) DELETE n` over 250k nodes / 500k
    /// edges reported 41 MB of graph memory against 25 MB before the delete.
    #[test]
    fn deleting_everything_folds_the_tombstones_away() {
        ensure_init();
        const N: u64 = 10_000;
        let mut t = Tensor::new(N + 1, N + 1);
        let srcs: Vec<u64> = (0..N).collect();
        let dsts: Vec<u64> = (0..N).map(|i| i + 1).collect();
        let ids: Vec<u64> = (0..N).collect();
        t.set_all_from_slices(&srcs, &dsts, &ids);

        // Next version: the adds fold into the base, as they do in the graph
        // when a later transaction mutates the tensor.
        let mut t = t.dup();
        t.flush();
        t.wait_fwd();
        assert_eq!(t.fwd_m().nvals(), N, "adds did not fold into the base");

        // Delete every edge, then commit.
        let mut t = t.dup();
        t.remove_all(&(0..N).map(|i| (i, i, i + 1)).collect::<Vec<_>>());
        t.fold_oversized();

        t.wait_fwd();
        assert_eq!(t.fwd_m().nvals(), 0, "base kept its deleted entries");
        assert_eq!(t.fwd_dm().nvals(), 0, "tombstones kept alongside the base");
        assert!(t.get(0, 1).next().is_none(), "deleted edge still readable");
    }

    /// Enough entries for the fold policy's [`MIN_FOLD_DELTA`] floor, so a
    /// `fold_oversized` really does move the deltas into the bases.
    const FOLDABLE: u64 = 512;

    /// A tensor whose every pair `(i, i + 1)` has two committed edges,
    /// `2i` and `2i + 1`, with `me` and the `MULTI_EDGE` sentinels in the base.
    fn committed_pairs(n: u64) -> Tensor {
        let mut t = Tensor::new(n + 1, n + 1);
        let srcs: Vec<u64> = (0..n).flat_map(|i| [i, i]).collect();
        let dsts: Vec<u64> = (0..n).flat_map(|i| [i + 1, i + 1]).collect();
        let ids: Vec<u64> = (0..n).flat_map(|i| [2 * i, 2 * i + 1]).collect();
        t.set_all_from_slices(&srcs, &dsts, &ids);
        // Commit: fold the pending adds into the bases, so the batch below
        // demotes *committed* multi pairs (state E of the module's diagram).
        t.fold_oversized();
        t.wait();
        assert_eq!(t.fwd_m().nvals(), n, "sentinels not folded into the base");
        t.dup()
    }

    /// One batch that demotes many multi-edge pairs must leave every survivor
    /// inline and every pair still there. Regression test for the read-after-
    /// write in the per-edge slow path (issue #2429): each demotion wrote `me`
    /// and `dp` and the next edge read them straight back, so GraphBLAS had to
    /// materialize the tuples just queued — 95x C per demoted pair, growing
    /// with the batch. Buffering those writes has to replay the same state
    /// machine the old loop got for free by re-reading its own writes.
    #[test]
    fn batch_demote_leaves_every_survivor_inline() {
        ensure_init();
        const N: u64 = FOLDABLE;
        let mut t = committed_pairs(N);

        // Delete the odd id of every pair: each demotes 2 -> 1 edges.
        let emptied = t.remove_all(&(0..N).map(|i| (2 * i + 1, i, i + 1)).collect::<Vec<_>>());

        assert!(emptied.is_empty(), "demoted pairs reported as emptied");
        assert_eq!(t.edge_count(), N, "edge count after demoting every pair");
        assert_eq!(t.multi_pairs(), 0, "a demoted pair still counts as multi");
        assert_eq!(
            t.fwd_me_nvals_for_test(),
            0,
            "`me` still holds ids of demoted pairs"
        );
        for i in 0..N {
            assert_eq!(
                t.get(i, i + 1).collect::<Vec<_>>(),
                vec![2 * i],
                "pair ({i}, {}) lost its surviving edge",
                i + 1
            );
        }
    }

    /// A pair created *and* promoted inside one batch. The second edge cannot
    /// read the first from anywhere — the inline slot is still only queued in
    /// `m_ids`, and the layers say nothing about the pair — so it has to reach
    /// back into the batch, move that pending id into `me` alongside itself and
    /// turn the queued slot into the sentinel. A third edge then takes the
    /// already-multi path instead, on the same pair, in the same batch.
    #[test]
    fn one_batch_promotes_a_pair_it_created() {
        ensure_init();
        let mut t = Tensor::new(8, 8);
        t.set_all_from_slices(&[0, 0, 2, 2, 2], &[1, 1, 3, 3, 3], &[10, 11, 20, 21, 22]);
        t.flush();
        t.wait();

        assert_eq!(
            t.get(0, 1).collect::<Vec<_>>(),
            vec![10, 11],
            "a pair promoted inside its own batch lost an id"
        );
        assert_eq!(
            t.get(2, 3).collect::<Vec<_>>(),
            vec![20, 21, 22],
            "the third edge of a pair promoted inside its own batch"
        );
        assert_eq!(t.edge_count(), 5, "edge count after in-batch promotion");
        assert_eq!(t.multi_pairs(), 2, "both pairs should read as multi");
    }

    /// The case that makes the in-batch promotion more than a duplicate-key
    /// convenience: a committed multi pair emptied *earlier in the same
    /// transaction* is, to the read phase, a pair with no value at all. Adding
    /// two edges back to it therefore goes through the created-in-this-batch
    /// branch rather than the committed-value one, and its `m_masked` entry has
    /// to cancel the deletion instead of shadowing it.
    #[test]
    fn a_pair_emptied_this_transaction_promotes_again_from_scratch() {
        ensure_init();
        let mut t = committed_pairs(FOLDABLE);
        let emptied = t.remove_all(&[(0, 0, 1), (1, 0, 1)]);
        assert_eq!(emptied, vec![(0, 1)], "pair (0, 1) not reported emptied");
        assert!(t.get(0, 1).next().is_none(), "emptied pair still readable");

        t.set_all_from_slices(&[0, 0], &[1, 1], &[2 * FOLDABLE, 2 * FOLDABLE + 1]);
        t.flush();
        t.wait();

        assert_eq!(
            t.get(0, 1).collect::<Vec<_>>(),
            vec![2 * FOLDABLE, 2 * FOLDABLE + 1],
            "re-added pair lost an id"
        );
        assert_eq!(
            t.edge_count(),
            2 * FOLDABLE,
            "edge count after emptying and re-promoting one pair"
        );
        assert_eq!(t.multi_pairs(), FOLDABLE, "re-added pair not counted multi");
    }

    /// The same batch may take a pair all the way down: the demotion is only
    /// buffered, so the edge that removes the survivor has to see the pair as
    /// single — not as the multi pair the layers still say it is. Foreign and
    /// repeated ids in the batch must change nothing.
    #[test]
    fn batch_can_demote_and_then_empty_the_same_pair() {
        ensure_init();
        const N: u64 = FOLDABLE;
        let mut t = committed_pairs(N);

        let mut rels: Vec<(u64, u64, u64)> = Vec::new();
        for i in 0..N {
            rels.push((2 * i + 1, i, i + 1)); // demotes the pair
            rels.push((2 * i + 1, i, i + 1)); // repeat: already gone
            rels.push((7 * N + i, i, i + 1)); // id of no edge of this pair
            rels.push((2 * i, i, i + 1)); // removes the survivor: pair empties
            rels.push((2 * i, i, i + 1)); // repeat: pair already empty
        }
        let mut emptied = t.remove_all(&rels);
        emptied.sort_unstable();

        assert_eq!(
            emptied,
            (0..N).map(|i| (i, i + 1)).collect::<Vec<_>>(),
            "every pair should be reported emptied exactly once"
        );
        assert_eq!(t.edge_count(), 0, "edges left after removing all of them");
        assert_eq!(t.multi_pairs(), 0, "a demoted pair still counts as multi");
        assert_eq!(
            t.fwd_me_nvals_for_test(),
            0,
            "`me` still holds ids of emptied pairs"
        );
        t.mt.wait();
        assert_eq!(
            t.mt.extract().nvals(),
            0,
            "backward adjacency kept the pairs"
        );
        for i in 0..N {
            assert!(
                t.get(i, i + 1).next().is_none(),
                "pair ({i}, {}) still readable",
                i + 1
            );
        }
    }

    /// Demoting back to the committed value must cancel the deltas to clean,
    /// not leave `dp` shadowing `m` with the value `m` already holds (state F
    /// -> D of the module's diagram). The write phase decides this from the
    /// committed base, which the buffered `dp` writes must not disturb.
    #[test]
    fn demoting_to_the_committed_value_cancels_to_clean() {
        ensure_init();
        const N: u64 = FOLDABLE;
        let mut t = Tensor::new(N + 1, N + 1);
        let srcs: Vec<u64> = (0..N).collect();
        let dsts: Vec<u64> = (0..N).map(|i| i + 1).collect();
        let ids: Vec<u64> = (0..N).map(|i| i + 1).collect();
        t.set_all_from_slices(&srcs, &dsts, &ids);
        t.fold_oversized();
        t.wait();
        assert_eq!(t.fwd_m().get(0, 1), Some(1), "single edge not committed");

        // Promote in this version (m = 1, dp = M, me = {1, 9}), then remove the
        // new edge again: the survivor is the committed 1.
        let mut t = t.dup();
        t.set_all_from_slices(&[0], &[1], &[9]);
        let emptied = t.remove_all(&[(9, 0, 1)]);

        assert!(emptied.is_empty(), "surviving pair reported as emptied");
        t.wait();
        assert_eq!(
            t.fwd_dp().nvals(),
            0,
            "dp still shadows m with m's own value"
        );
        assert_eq!(t.fwd_dm().nvals(), 0, "demotion left a tombstone");
        assert_eq!(
            t.fwd_me_nvals_for_test(),
            0,
            "`me` not emptied by the demotion"
        );
        assert_eq!(t.edge_count(), N, "edge count after promote + demote");
        assert_eq!(
            t.get(0, 1).collect::<Vec<_>>(),
            vec![1],
            "committed edge lost"
        );
    }

    /// **The bug this replaced.** `me` is `GrB_INDEX_MAX` square and the key
    /// was `(src << 32) | dst`, so `src = 2^28` put the row out of the matrix's
    /// range. `Matrix::set` checks the GraphBLAS status under `debug_assert!`
    /// only, so a release build dropped the write and left the pair tagged
    /// `MULTI_EDGE` over an empty `me` row: `get` returned nothing and
    /// `edge_count` disagreed with both the inserts and the reads. The guard
    /// that existed checked `u32`, four bits above the bound that mattered, and
    /// so never fired.
    ///
    /// Runs well past the old cliff and past `2^32`, the ceiling the guard
    /// nominally enforced, since blocks remove the bound rather than move it.
    #[test]
    fn promotion_survives_node_ids_past_the_old_key_width() {
        ensure_init();
        for shift in [27u32, 28, 29, 31, 32, 33, 40] {
            let src = 1u64 << shift;
            let dst = (1u64 << shift) + 5;
            let n = dst + 2;
            let mut t = Tensor::new(n, n);
            t.set_all_from_slices(&[src, src], &[dst, dst], &[10, 11]);
            let mut t = t.dup();
            t.flush();
            t.wait();

            let ids: Vec<u64> = t.get(src, dst).collect();
            assert_eq!(ids, vec![10, 11], "ids lost at src = 2^{shift}");
            assert_eq!(t.edge_count(), 2, "edge_count wrong at src = 2^{shift}");
            assert_eq!(t.multi_pairs(), 1, "multi_pairs wrong at src = 2^{shift}");
            assert!(
                t.has_multi_edge(),
                "has_multi_edge wrong at src = 2^{shift}"
            );

            // `iter_edges` recovers the pair from the block and row key, so it
            // is the check that `compound_key_inverse` really inverts.
            let mut seen: Vec<(u64, u64, u64)> = t.iter_edges().collect();
            seen.sort_unstable();
            assert_eq!(
                seen,
                vec![(src, dst, 10), (src, dst, 11)],
                "iter_edges wrong at src = 2^{shift}"
            );

            // And demotion still finds the ids it has to remove.
            t.remove_all(&[(10, src, dst)]);
            t.wait();
            assert_eq!(
                t.get(src, dst).collect::<Vec<_>>(),
                vec![11],
                "demotion lost the survivor at src = 2^{shift}"
            );
            assert_eq!(t.edge_count(), 1, "edge_count wrong after demotion");
        }
    }

    /// Pairs in different blocks must not collide, and the blocks must stay
    /// separable. Endpoints chosen so several distinct pairs share a row key
    /// while differing in block — the exact aliasing a single packed key would
    /// have produced.
    #[test]
    fn blocks_do_not_alias() {
        ensure_init();
        let b = 1u64 << BLOCK_SHIFT;
        let pairs = [(1u64, 2u64), (b + 1, 2), (1, b + 2), (b + 1, b + 2)];
        let n = 2 * b + 8;
        let mut t = Tensor::new(n, n);
        for (i, &(s, d)) in pairs.iter().enumerate() {
            let base = 100 * i as u64;
            t.set_all_from_slices(&[s, s], &[d, d], &[base, base + 1]);
        }
        let mut t = t.dup();
        t.flush();
        t.wait();

        // All four share the row key of (1, 2) and differ only in block.
        let rows: Vec<u64> = pairs.iter().map(|&(s, d)| compound_key(s, d).1).collect();
        assert!(
            rows.windows(2).all(|w| w[0] == w[1]),
            "fixture is not exercising aliasing: rows {rows:?} differ"
        );

        for (i, &(s, d)) in pairs.iter().enumerate() {
            let base = 100 * i as u64;
            assert_eq!(
                t.get(s, d).collect::<Vec<_>>(),
                vec![base, base + 1],
                "pair ({s}, {d}) read another block's ids"
            );
        }
        assert_eq!(t.edge_count(), 8);
        assert_eq!(t.multi_pairs(), 4);
    }

    /// The smallest thing that is both a [`Writer`] and a [`Reader`]: the
    /// encoder's calls, replayed to the decoder in order. The crate's real
    /// byte-level writers live in the host crate and are not reachable here,
    /// and the round trip under test is the *re-keying*, not the framing.
    #[derive(Default)]
    struct Tape {
        ops: std::collections::VecDeque<TapeOp>,
    }

    enum TapeOp {
        Unsigned(u64),
        Signed(i64),
        Double(f64),
        Buffer(Vec<u8>),
    }

    impl super::super::serialization::Writer for Tape {
        fn write_unsigned(
            &mut self,
            val: u64,
        ) {
            self.ops.push_back(TapeOp::Unsigned(val));
        }
        fn write_signed(
            &mut self,
            val: i64,
        ) {
            self.ops.push_back(TapeOp::Signed(val));
        }
        fn write_double(
            &mut self,
            val: f64,
        ) {
            self.ops.push_back(TapeOp::Double(val));
        }
        fn write_buffer(
            &mut self,
            data: &[u8],
        ) {
            self.ops.push_back(TapeOp::Buffer(data.to_vec()));
        }
    }

    impl super::super::serialization::Reader for Tape {
        fn read_unsigned(&mut self) -> Result<u64, String> {
            match self.ops.pop_front() {
                Some(TapeOp::Unsigned(v)) => Ok(v),
                _ => Err("tape: expected unsigned".to_string()),
            }
        }
        fn read_signed(&mut self) -> Result<i64, String> {
            match self.ops.pop_front() {
                Some(TapeOp::Signed(v)) => Ok(v),
                _ => Err("tape: expected signed".to_string()),
            }
        }
        fn read_double(&mut self) -> Result<f64, String> {
            match self.ops.pop_front() {
                Some(TapeOp::Double(v)) => Ok(v),
                _ => Err("tape: expected double".to_string()),
            }
        }
        fn read_buffer(&mut self) -> Result<Vec<u8>, String> {
            match self.ops.pop_front() {
                Some(TapeOp::Buffer(v)) => Ok(v),
                _ => Err("tape: expected buffer".to_string()),
            }
        }
    }

    /// The round trip has to survive blocks. The blob stores endpoints rather
    /// than row keys, so no format change was needed — this is the test that
    /// says so, by decoding a multi-block tensor and reading every pair back.
    #[test]
    fn encode_decode_round_trips_across_blocks() {
        ensure_init();
        let b = 1u64 << BLOCK_SHIFT;
        let pairs = [(1u64, 2u64), (b + 7, 9), (3, b + 4), (b + 5, b + 6)];
        let n = 2 * b + 16;
        let mut t = Tensor::new(n, n);
        for (i, &(s, d)) in pairs.iter().enumerate() {
            let base = 1000 * i as u64;
            t.set_all_from_slices(&[s, s, s], &[d, d, d], &[base, base + 1, base + 2]);
        }
        // one single-edge pair too, so the inline path is covered
        t.set_all_from_slices(&[b + 11], &[12], &[9999]);
        let mut t = t.dup();
        t.flush();
        t.wait();

        let mut tape = Tape::default();
        t.encode(&mut tape);
        let mut back = Tensor::decode(&mut tape).expect("decode");
        back.rebuild_backward();
        back.wait();

        for (i, &(s, d)) in pairs.iter().enumerate() {
            let base = 1000 * i as u64;
            assert_eq!(
                back.get(s, d).collect::<Vec<_>>(),
                vec![base, base + 1, base + 2],
                "pair ({s}, {d}) did not survive the round trip"
            );
        }
        assert_eq!(back.get(b + 11, 12).collect::<Vec<_>>(), vec![9999]);
        assert_eq!(back.edge_count(), t.edge_count());
        assert_eq!(back.multi_pairs(), t.multi_pairs());
    }

    /// `me` is created with a narrow column space so GraphBLAS stores its
    /// column indices — one per stored edge id, the largest array here — in 32
    /// bits. An id past that width has to widen it rather than be dropped, and
    /// the ids already stored have to survive the widening.
    #[test]
    fn me_widens_for_edge_ids_past_the_narrow_column_space() {
        ensure_init();
        let narrow = super::ME_NARROW_NCOLS;
        let mut t = Tensor::new(16, 16);
        t.set_all_from_slices(&[1, 1], &[2, 2], &[7, 8]);
        let mut t = t.dup();
        t.flush();
        t.wait();
        assert_eq!(
            t.edge_versioned_block_0().ncols(),
            narrow,
            "a tensor with small ids should still be narrow"
        );
        assert_eq!(t.get(1, 2).collect::<Vec<_>>(), vec![7, 8]);

        // An id at the boundary widens it, and nothing already stored is lost.
        t.set_all_from_slices(&[1], &[2], &[narrow]);
        t.wait();
        assert_eq!(
            t.edge_versioned_block_0().ncols(),
            super::ME_DIM,
            "an id at the narrow width should have widened `me`"
        );
        assert_eq!(t.get(1, 2).collect::<Vec<_>>(), vec![7, 8, narrow]);
        assert_eq!(t.edge_count(), 3);

        // And a block created *after* widening must be wide too, or its ids
        // would be the ones dropped.
        let b = 1u64 << BLOCK_SHIFT;
        let mut t2 = Tensor::new(4 * b, 4 * b);
        t2.set_all_from_slices(&[1], &[2], &[narrow + 1]);
        t2.set_all_from_slices(&[b + 1, b + 1], &[2, 2], &[narrow + 2, narrow + 3]);
        t2.wait();
        assert_eq!(
            t2.get(b + 1, 2).collect::<Vec<_>>(),
            vec![narrow + 2, narrow + 3],
            "a block created after widening dropped its ids"
        );
    }

    /// The top row key is the one [`compound_key`] can produce with both
    /// endpoint halves all-ones, and it is exactly `GrB_INDEX_MAX`. A matrix
    /// declared that many rows accepts one fewer, so this pair's writes were
    /// silently dropped — the original bug, at the new boundary. Caught in
    /// review; the earlier tests all used `src = 2^k`, which never lands here.
    #[test]
    fn the_top_row_key_of_a_block_is_writable() {
        ensure_init();
        let m = (1u64 << BLOCK_SHIFT) - 1;
        assert_eq!(
            compound_key(m, m).1,
            GrB_INDEX_MAX,
            "fixture is not exercising the top row key"
        );

        // In block (0,0), and in a higher block, since blocks are created by a
        // different path than the tensor's own field.
        let b = 1u64 << BLOCK_SHIFT;
        for (src, dst) in [(m, m), (b + m, b + m), (b + m, m), (m, b + m)] {
            let mut t = Tensor::new(2 * b + 2, 2 * b + 2);
            t.set_all_from_slices(&[src, src], &[dst, dst], &[10, 11]);
            let mut t = t.dup();
            t.flush();
            t.wait();
            assert_eq!(
                t.get(src, dst).collect::<Vec<_>>(),
                vec![10, 11],
                "ids dropped at the top row key of ({src}, {dst})"
            );
            assert_eq!(t.edge_count(), 2, "edge_count wrong at ({src}, {dst})");
            assert_eq!(t.multi_pairs(), 1);
            let mut seen: Vec<_> = t.iter_edges().collect();
            seen.sort_unstable();
            assert_eq!(seen, vec![(src, dst, 10), (src, dst, 11)]);
        }
    }
}
