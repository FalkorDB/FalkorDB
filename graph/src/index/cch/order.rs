//! Phase 1: nested-dissection ordering and triangulation (#3208).
//!
//! Phase 1 never looks at a weight. It reads the undirected skeleton of the graph
//! ([`CchInput::skeleton`](super::CchInput::skeleton)) and produces the *metric-independent*
//! half of the hierarchy, which later customizations reuse without rerunning any of this:
//!
//! 1. **Elimination order** ([`Ordering`]) — a nested-dissection permutation. Nested
//!    dissection splits the graph with small node separators, recurses into each side, and
//!    assigns the deepest pieces the lowest ranks so every separator lands above everything
//!    it separates. That placement is what keeps shortcuts local instead of spreading.
//! 2. **Elimination game** ([`Topology::build`]) — eliminate nodes in rank order, joining
//!    each eliminated node's not-yet-eliminated neighbours into a clique. The result is the
//!    *chordal supergraph*: every arc, original or shortcut, that any Phase 2 could ever
//!    need to write a weight into.
//!
//! Ported from the C implementation's `CCH_EliminationOrder` + `CCH_ChordalTriangulation`
//! (`src/algorithms/cch.c`), with two deliberate differences:
//!
//! * the ordering backend is the pure-Rust `feral-metis` crate rather than a C METIS
//!   binding, hidden behind the [`Ordering`] trait so the two are interchangeable;
//! * the chordal supergraph is one flat CSR pair instead of C's array-of-arrays, which is
//!   possible because child absorption only ever reads *lower* ranks — already finished and
//!   contiguous in the flat buffer by the time their parent is processed.

// Phase 1 lands on its own, before anything outside this file calls it: `Cch` grows its
// topology field in the customization change (#3209), and the tests below are the only caller until then.
#![allow(dead_code)]

use feral_metis::{CscPattern, MetisOptions, OrderingError, metis_order};

use std::ffi::CString;

use super::{CchError, Csr};
use crate::index::redisearch::redis::RedisModule_Log;

/// Sentinel for "no such rank" in [`Topology`]'s scratch arrays. Node ids are capped at
/// `i32::MAX` (see [`Ordering`]), so no real rank can collide with it.
const NONE: u32 = u32::MAX;

/// Why an elimination order could not be produced.
///
/// Deliberately not [`CchError`]: the ordering backend is swappable, so its failure modes
/// stay abstract here and are mapped at the single call site in [`Topology::build_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderError {
    /// Beyond what this backend can index. `feral-metis` is `i32`-indexed, so its ceiling
    /// is `i32::MAX` nodes and `i32::MAX` skeleton entries
    /// ([`MAX_SKELETON_ENTRIES`](super::MAX_SKELETON_ENTRIES): twice the number of connected
    /// node pairs, so the pair count is what a large graph runs into first) — below the
    /// contract's `u32::MAX` node limit. A 64-bit backend would raise it without any caller
    /// changing.
    TooLarge,
    /// The backend rejected a pattern this module built, or failed its own self-check.
    /// Unreachable for a skeleton that meets [`CchInput`](super::CchInput)'s contract; it
    /// means a bug here or in the backend, never bad user input.
    Backend(&'static str),
}

/// A pluggable elimination-order backend.
///
/// Implementors take the undirected, self-loop-free skeleton and return `perm`, where
/// `perm[rank]` is the node id eliminated at that rank — "new-to-old", the same convention
/// `feral-metis` and METIS itself use. The result must be a permutation of `0..skeleton.n`;
/// [`Topology::build_with`] verifies that rather than trusting it.
///
/// **Any** permutation is a *correct* elimination order: the ordering only decides how many
/// fill-in arcs the elimination game creates, i.e. how big and how fast the hierarchy is.
/// That is why [`IdentityOrdering`] is a usable test double.
///
/// (Named for the elimination order, not [`std::cmp::Ordering`]; this module never uses the
/// latter.)
pub trait Ordering {
    /// Produce `perm[rank] = node_id` for `skeleton`.
    ///
    /// # Errors
    /// [`OrderError::TooLarge`] when the graph exceeds the backend's index width,
    /// [`OrderError::Backend`] for any other backend failure.
    fn order(
        &self,
        skeleton: &Csr,
    ) -> Result<Vec<u32>, OrderError>;
}

/// Nested dissection via the pure-Rust `feral-metis` crate.
///
/// Uses `MetisOptions::default()`, whose seed is `1` — so the same skeleton always yields
/// the same order, which is what makes a rebuilt index byte-identical to the original.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetisOrdering;

/// `perm[rank] = rank`: eliminate nodes in node-id order.
///
/// Correct but makes no attempt to reduce fill: a deterministic, hand-checkable order for
/// tests. Deliberately *not* a fallback for a failing backend — that is a hard error
/// ([`CchError::Internal`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityOrdering;

impl Ordering for IdentityOrdering {
    fn order(
        &self,
        skeleton: &Csr,
    ) -> Result<Vec<u32>, OrderError> {
        if skeleton.n > i32::MAX as usize {
            return Err(OrderError::TooLarge);
        }
        Ok((0..skeleton.n as u32).collect())
    }
}

impl Ordering for MetisOrdering {
    fn order(
        &self,
        skeleton: &Csr,
    ) -> Result<Vec<u32>, OrderError> {
        let n = skeleton.n;
        if n == 0 {
            // `metis_order` handles this, but short-circuiting keeps the empty graph a
            // pure no-op rather than a backend round trip.
            return Ok(Vec::new());
        }
        if n > i32::MAX as usize || skeleton.i.len() > i32::MAX as usize {
            return Err(OrderError::TooLarge);
        }

        let comps = Components::of(skeleton);
        if comps.count() == 1 {
            let (col_ptr, row_idx) =
                csc_pattern(skeleton, 0..n as u32, |r| r).ok_or(OrderError::TooLarge)?;
            let perm = metis_perm(n, &col_ptr, &row_idx)?;
            return perm
                .into_iter()
                .map(|p| u32::try_from(p).map_err(|_| NEGATIVE_ID))
                .collect();
        }

        // `feral-metis` splits components itself, but extracts each one with an O(n) scan
        // of the whole graph, which is quadratic in the component count — and the skeleton
        // is sized by `node_cap`, so every unused id slot past the live nodes is a
        // component of its own (~300k of them on a 1.25M-node road network: 3 s of
        // ordering became 180 s). So the components are split here, in linear time:
        //
        // * up to two nodes: node-id order, no backend call. No order of one or two nodes
        //   creates fill-in, and node-id order is what the backend returns for them.
        // * up to the backend's AMD switch: one backend call each. The backend hands such a
        //   component straight to AMD, which never touches its RNG.
        // * larger: all of them in a single backend call, in component order. Only these
        //   draw from the backend's RNG, and one call threads it through them exactly as
        //   it would through the whole graph.
        //
        // The backend numbers components by lowest node id, as `Components` does, and
        // concatenates their orders; so the result is byte-identical to one call on the
        // whole skeleton. The large-component call is still quadratic in its own
        // component count, but every such component has over `amd_switch` nodes.
        //
        // `amd_switch` is read from the backend rather than copied, so a crate bump that
        // moves it cannot silently break that equivalence.
        let amd_switch = MetisOptions::default().nd_to_amd_switch as usize;
        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        let mut large_members: Vec<u32> = Vec::new();
        for members in comps.iter().filter(|m| m.len() > amd_switch) {
            let base = large_members.len() as u32;
            append_columns(
                skeleton,
                members.iter().copied(),
                |r| base + comps.local[r as usize],
                &mut col_ptr,
                &mut row_idx,
            )
            .ok_or(OrderError::TooLarge)?;
            large_members.extend_from_slice(members);
        }
        let mut large_perm = if large_members.is_empty() {
            Vec::new()
        } else {
            metis_perm(large_members.len(), &col_ptr, &row_idx)?
        }
        .into_iter();

        let mut perm = Vec::with_capacity(n);
        for members in comps.iter() {
            let (local_perm, members) = if members.len() <= 2 {
                perm.extend_from_slice(members);
                continue;
            } else if members.len() <= amd_switch {
                col_ptr.clear();
                col_ptr.push(0);
                row_idx.clear();
                append_columns(
                    skeleton,
                    members.iter().copied(),
                    |r| comps.local[r as usize],
                    &mut col_ptr,
                    &mut row_idx,
                )
                .ok_or(OrderError::TooLarge)?;
                (metis_perm(members.len(), &col_ptr, &row_idx)?, members)
            } else {
                // The large call ordered this component's nodes next, as one block.
                let block: Vec<i32> = large_perm.by_ref().take(members.len()).collect();
                (block, large_members.as_slice())
            };
            for p in local_perm {
                let local = usize::try_from(p).map_err(|_| NEGATIVE_ID)?;
                perm.push(*members.get(local).ok_or(OUT_OF_RANGE)?);
            }
        }
        if perm.len() != n {
            return Err(OrderError::Backend("perm has the wrong length"));
        }
        Ok(perm)
    }
}

const OUT_OF_RANGE: OrderError = OrderError::Backend("node id out of range in perm");
const NEGATIVE_ID: OrderError = OrderError::Backend("negative node id in perm");

/// Run the backend on one `n`-node CSC pattern, mapping its errors onto [`OrderError`].
fn metis_perm(
    n: usize,
    col_ptr: &[i32],
    row_idx: &[i32],
) -> Result<Vec<i32>, OrderError> {
    let pattern = CscPattern::new(n, col_ptr, row_idx)
        .ok_or(OrderError::Backend("skeleton rejected by CscPattern::new"))?;
    metis_order(&pattern).map_err(|e| match e {
        OrderingError::IndexOverflow => OrderError::TooLarge,
        OrderingError::MalformedInput => OrderError::Backend("malformed CSC pattern"),
        OrderingError::NonSymmetric => OrderError::Backend("skeleton is not symmetric"),
        OrderingError::DisconnectedGraph => OrderError::Backend("disconnected skeleton"),
        OrderingError::Internal(m) => OrderError::Backend(m),
    })
}

/// The connected components of a skeleton, found in O(n + m).
///
/// Components are numbered in order of their lowest node id, and each one's members are
/// listed in ascending node id — the same numbering `feral-metis` uses internally.
struct Components {
    /// `members[start[c]..start[c + 1]]` are the nodes of component `c`.
    start: Vec<usize>,
    members: Vec<u32>,
    /// `local[v]` is `v`'s position within its own component's member list.
    local: Vec<u32>,
}

impl Components {
    fn of(skeleton: &Csr) -> Self {
        let n = skeleton.n;
        // Depth-first labelling; `label` is reused below to hold the local ids.
        let mut label = vec![NONE; n];
        let mut count = 0u32;
        let mut stack: Vec<u32> = Vec::new();
        for s in 0..n {
            if label[s] != NONE {
                continue;
            }
            label[s] = count;
            stack.push(s as u32);
            while let Some(v) = stack.pop() {
                let v = v as usize;
                for &u in &skeleton.i[skeleton.p[v] as usize..skeleton.p[v + 1] as usize] {
                    if label[u as usize] == NONE {
                        label[u as usize] = count;
                        stack.push(u);
                    }
                }
            }
            count += 1;
        }

        // Counting sort by label. Scanning `v` upwards keeps each member list ascending.
        let count = count as usize;
        let mut start = vec![0usize; count + 1];
        for &c in &label {
            start[c as usize + 1] += 1;
        }
        for c in 0..count {
            start[c + 1] += start[c];
        }
        let mut next = start.clone();
        let mut members = vec![0u32; n];
        for (v, slot) in label.iter_mut().enumerate() {
            let c = *slot as usize;
            *slot = (next[c] - start[c]) as u32;
            members[next[c]] = v as u32;
            next[c] += 1;
        }
        Self {
            start,
            members,
            local: label,
        }
    }

    fn count(&self) -> usize {
        self.start.len() - 1
    }

    fn iter(&self) -> impl Iterator<Item = &[u32]> {
        self.start.windows(2).map(|w| &self.members[w[0]..w[1]])
    }
}

/// Copy `skeleton` into the `i32` CSC arrays `feral-metis` wants.
///
/// The skeleton is symmetric, so its CSR *is* its CSC and no transpose is needed. The copy
/// is not avoidable: [`Csr`] stores `u64` offsets and `u32` indices, the contract type
/// stores `i32` of both. Returns `None` if an `i32` would overflow; the largest offset is
/// `skeleton.i.len()`, so that is the [`MAX_SKELETON_ENTRIES`](super::MAX_SKELETON_ENTRIES)
/// ceiling, which [`MetisOrdering::order`] has already checked and [`check_size`] enforces
/// before any of this runs.
///
/// A converted-as-is copy: self-loops and duplicate entries, which the [`Csr`] contract
/// rules out, are *not* filtered here. They would be harmless: `feral-metis` drops the
/// diagonal and adjacent duplicates itself when it builds its graph
/// (`Graph::from_csc_pattern`; the rows are sorted, which `CscPattern::new` checks), and the
/// rest of Phase 1 ignores both, so an unclean skeleton produces the same order and the same
/// hierarchy (`an_unclean_skeleton_orders_like_a_clean_one`).
///
/// `nodes` picks the columns, in order, and `local` renames a row id into the pattern's
/// own numbering: the identity for the whole graph, or a component's local ids for one
/// component. `nodes` must be closed under adjacency — every neighbour of a listed node is
/// listed too — which a whole connected component is. The renaming must keep each row
/// sorted, which a rename by position in an ascending member list does.
///
/// [`check_size`]: super::check_size
fn csc_pattern(
    skeleton: &Csr,
    nodes: impl ExactSizeIterator<Item = u32>,
    local: impl Fn(u32) -> u32,
) -> Option<(Vec<i32>, Vec<i32>)> {
    let mut col_ptr = vec![0];
    let mut row_idx = Vec::new();
    append_columns(skeleton, nodes, local, &mut col_ptr, &mut row_idx)?;
    Some((col_ptr, row_idx))
}

/// Append one CSC column per node of `nodes` to `col_ptr`/`row_idx`, which must already
/// hold a valid prefix (at least the leading `0`). Appending is what lets several
/// components share one pattern.
fn append_columns(
    skeleton: &Csr,
    nodes: impl ExactSizeIterator<Item = u32>,
    local: impl Fn(u32) -> u32,
    col_ptr: &mut Vec<i32>,
    row_idx: &mut Vec<i32>,
) -> Option<()> {
    debug_assert_eq!(
        skeleton.p.len(),
        skeleton.n + 1,
        "Csr::p must hold n + 1 offsets"
    );
    debug_assert_eq!(col_ptr.last().map(|&c| c as usize), Some(row_idx.len()));

    col_ptr.reserve(nodes.len());
    for c in nodes {
        let lo = skeleton.p[c as usize] as usize;
        let hi = skeleton.p[c as usize + 1] as usize;
        for &r in &skeleton.i[lo..hi] {
            row_idx.push(i32::try_from(local(r)).ok()?);
        }
        col_ptr.push(i32::try_from(row_idx.len()).ok()?);
    }

    Some(())
}

/// The metric-independent half of the hierarchy: the elimination order and the chordal
/// supergraph it implies. Shared by `Arc` across customizations — nothing here depends on a
/// weight, so a new metric reuses it untouched.
///
/// # Layout
///
/// Two spaces, and keeping them straight is the whole game:
///
/// * **node space** — ids as the rest of the engine knows them, `0..n`.
/// * **rank space** — elimination order, `0..n`. `perm[rank] = node`, `iperm[node] = rank`.
///
/// Everything except `perm`/`iperm` themselves is in **rank space**.
///
/// The chordal supergraph is stored twice, as two CSRs over ranks:
///
/// * `cch_p`/`cch_i` — the *upward* graph. `cch_i[cch_p[r]..cch_p[r + 1]]` lists every rank
///   `> r` that `r` is adjacent to once fill-in is accounted for, sorted ascending. This is
///   the array Phase 2 writes weights parallel to and Phase 3 walks.
/// * `down_p`/`down_i` — its exact inverse. `lo` appears in `down_i[..]` of `hi` iff `hi`
///   appears in `cch_i[..]` of `lo`. Also sorted ascending. Only an incremental
///   recustomization needs it; the full sweep ignores it.
///
/// An **arc slot** is an index into `cch_i`: slot `k` with `cch_p[r] <= k < cch_p[r + 1]`
/// is the arc between rank `r` and rank `cch_i[k]`. Phase 2's four weight arrays are
/// indexed by exactly this slot, so `up_w[k]` is the `r -> cch_i[k]` weight and `dn_w[k]`
/// the `cch_i[k] -> r` weight. [`Topology::arc_slot`] resolves a rank pair to its slot.
pub(crate) struct Topology {
    n: usize,
    /// `perm[rank] = node id`.
    perm: Vec<u32>,
    /// `iperm[node id] = rank`.
    iperm: Vec<u32>,
    /// Upward adjacency offsets, `n + 1` of them.
    cch_p: Vec<u64>,
    /// Upward adjacency, rank space, ascending within each row.
    cch_i: Vec<u32>,
    /// Downward adjacency offsets, `n + 1` of them.
    down_p: Vec<u64>,
    /// Downward adjacency, rank space, ascending within each row.
    down_i: Vec<u32>,
}

impl Topology {
    /// Phase 1 with the default backend ([`MetisOrdering`]).
    ///
    /// # Errors
    /// [`CchError::GraphTooLarge`] when the graph exceeds the ordering backend's index
    /// width (`i32::MAX` nodes or skeleton arcs for `feral-metis` — the same limits
    /// [`check_size`](super::check_size) applies, so through `Cch::build` this is a
    /// backstop), and
    /// [`CchError::Internal`] if the backend fails for any other reason.
    pub(crate) fn build(skeleton: &Csr) -> Result<Self, CchError> {
        Self::build_with(skeleton, &MetisOrdering)
    }

    /// Phase 1 with an explicit backend — what the tests use to pin a known order, and the
    /// seam a C METIS binding would plug into.
    ///
    /// A non-size backend failure cannot come from user input (see
    /// [`OrderError::Backend`]), so it is a hard error rather than a silently degraded
    /// index, and it is logged as well as returned: it means a bug, and the error alone
    /// may only reach a client as a failed query.
    ///
    /// # Errors
    /// As [`Self::build`].
    pub(crate) fn build_with(
        skeleton: &Csr,
        ordering: &dyn Ordering,
    ) -> Result<Self, CchError> {
        let perm = match ordering.order(skeleton) {
            Ok(perm) => perm,
            Err(OrderError::TooLarge) => return Err(CchError::GraphTooLarge),
            Err(OrderError::Backend(why)) => {
                log_warning(&format!(
                    "CCH: nested-dissection ordering failed ({why}); the index was not built."
                ));
                return Err(CchError::Internal(why));
            }
        };
        Self::from_perm(skeleton, perm)
    }

    /// The elimination game itself: everything after the order is fixed.
    fn from_perm(
        skeleton: &Csr,
        perm: Vec<u32>,
    ) -> Result<Self, CchError> {
        let n = skeleton.n;
        debug_assert_eq!(skeleton.p.len(), n + 1, "Csr::p must hold n + 1 offsets");

        let Some(iperm) = invert(&perm, n) else {
            // The backend already self-checks for a bijection, so this is unreachable
            // twice over; refusing beats building a silently corrupt hierarchy.
            const WHY: &str = "ordering backend returned a non-permutation";
            log_warning(&format!("CCH: {WHY}; the index was not built."));
            return Err(CchError::Internal(WHY));
        };

        let parent = elimination_tree(skeleton, &perm, &iperm);
        let (cch_p, cch_i) = chordal_supergraph(skeleton, &perm, &iperm, &parent);
        let (down_p, down_i) = invert_adjacency(&cch_p, &cch_i);

        Ok(Self {
            n,
            perm,
            iperm,
            cch_p,
            cch_i,
            down_p,
            down_i,
        })
    }

    pub(crate) fn node_count(&self) -> usize {
        self.n
    }

    /// Total arcs in the chordal supergraph — also the length of each Phase 2 weight array.
    pub(crate) fn arc_count(&self) -> usize {
        self.cch_i.len()
    }

    /// `perm[rank]`.
    pub(crate) fn node_of(
        &self,
        rank: u32,
    ) -> u32 {
        self.perm[rank as usize]
    }

    /// `iperm[node]`.
    pub(crate) fn rank_of(
        &self,
        node: u32,
    ) -> u32 {
        self.iperm[node as usize]
    }

    /// Every rank above `rank` in the chordal supergraph, ascending.
    pub(crate) fn up(
        &self,
        rank: u32,
    ) -> &[u32] {
        let lo = self.cch_p[rank as usize] as usize;
        let hi = self.cch_p[rank as usize + 1] as usize;
        &self.cch_i[lo..hi]
    }

    /// The arc-slot range of `rank`'s upward arcs: `up(rank)[i]` is slot `up_range(rank).start + i`.
    pub(crate) fn up_range(
        &self,
        rank: u32,
    ) -> std::ops::Range<usize> {
        self.cch_p[rank as usize] as usize..self.cch_p[rank as usize + 1] as usize
    }

    /// Every rank below `rank` in the chordal supergraph, ascending.
    pub(crate) fn down(
        &self,
        rank: u32,
    ) -> &[u32] {
        let lo = self.down_p[rank as usize] as usize;
        let hi = self.down_p[rank as usize + 1] as usize;
        &self.down_i[lo..hi]
    }

    /// The arc slot of the arc `lo -> hi`, or `None` if the two are not adjacent in the
    /// chordal supergraph. `lo` must be the lower rank.
    ///
    /// Phase 2 calls this once per triangle, so it is a binary search over an ascending row
    /// rather than a hash lookup.
    pub(crate) fn arc_slot(
        &self,
        lo: u32,
        hi: u32,
    ) -> Option<usize> {
        debug_assert!(lo < hi, "arc_slot takes the lower rank first");
        let base = self.cch_p[lo as usize] as usize;
        self.up(lo).binary_search(&hi).ok().map(|i| base + i)
    }

    /// Heap bytes held by the six structural arrays.
    pub(crate) fn memory_usage(&self) -> usize {
        size_of::<Self>()
            + (self.perm.capacity() + self.iperm.capacity()) * size_of::<u32>()
            + (self.cch_p.capacity() + self.down_p.capacity()) * size_of::<u64>()
            + (self.cch_i.capacity() + self.down_i.capacity()) * size_of::<u32>()
    }
}

/// Warn through the host's log, falling back to stderr when this crate is not loaded into a
/// Redis (unit tests, benchmarks). Mirrors the panic hook in `src/module_init.rs`, the only
/// other place this crate reaches for `RedisModule_Log` directly.
///
/// A backend failure in [`Topology::build_with`] must not be silent in a release build: it
/// cannot be reached by user input, so seeing it at all means a bug.
fn log_warning(msg: &str) {
    // SAFETY: reading a `static mut` function pointer the module loader filled in, and
    // calling it with a NULL context -- exactly what the panic hook does. `Option<fn>` is
    // `Copy`, so no reference to the static is created.
    unsafe {
        if let Some(log) = RedisModule_Log
            && let Ok(c_msg) = CString::new(msg.replace('\0', " "))
        {
            log(
                std::ptr::null_mut(),
                c"warning".as_ptr(),
                c"%s".as_ptr(),
                c_msg.as_ptr(),
            );
            return;
        }
    }
    eprintln!("{msg}");
}

/// `iperm[perm[rank]] = rank`, or `None` if `perm` is not a permutation of `0..n`.
///
/// The backend's output is checked, not trusted: a non-bijection would corrupt every later
/// phase silently rather than loudly.
fn invert(
    perm: &[u32],
    n: usize,
) -> Option<Vec<u32>> {
    if perm.len() != n {
        return None;
    }
    let mut iperm = vec![NONE; n];
    for (rank, &node) in perm.iter().enumerate() {
        let slot = iperm.get_mut(node as usize)?;
        if *slot != NONE {
            return None; // node appears twice
        }
        *slot = rank as u32;
    }
    Some(iperm)
}

/// `parent[rank]` = the elimination tree parent of `rank`: its lowest-ranked neighbour of
/// *higher* rank **in the chordal supergraph**, or [`NONE`] if `rank` is a root (one root
/// per connected component).
///
/// The point of the union-find is that it gets that right without materializing a single
/// fill-in arc. Ranks are processed in increasing order; for each one, a `Find` is started
/// from every already-eliminated *original* neighbour. Path compression threads each early
/// rank's chain through every rank that previously tried and failed to claim it, so a later
/// walk lands where the chordal graph — not merely the original graph — says it should.
///
/// Ports `_build_elimination_tree` / `_find_and_union` from the C implementation. No
/// union-by-size is needed for the near-linear bound: ranks are processed in increasing
/// order, so every union already attaches a lower rank under a strictly higher one.
fn elimination_tree(
    skeleton: &Csr,
    perm: &[u32],
    iperm: &[u32],
) -> Vec<u32> {
    let n = skeleton.n;
    let mut parent = vec![NONE; n];
    // `ancestor[x]` is the highest rank known so far to be connected to `x` once fill-in is
    // accounted for — x's current representative — or NONE if x has never been reached.
    let mut ancestor = vec![NONE; n];

    for rank in 0..n as u32 {
        let u = perm[rank as usize] as usize;
        for &v in skeleton.row(u) {
            // Walk from this neighbour's rank while it stays below `rank`, compressing
            // every step onto the walk's endpoint. An unclaimed endpoint is claimed under
            // `rank`, which is by construction the lowest rank ever to try.
            let mut w = iperm[v as usize];
            while w < rank {
                let next = ancestor[w as usize];
                ancestor[w as usize] = rank;
                if next == NONE {
                    parent[w as usize] = rank;
                }
                w = next;
            }
        }
    }

    parent
}

/// The chordal supergraph as the upward CSR `(cch_p, cch_i)`.
///
/// Gilbert-Ng-Peyton child absorption:
///
/// ```text
/// up(r) = { w in skeleton : rank(w) > r }  u  U { up(c) \ {r} : c is an elimination-tree child of r }
/// ```
///
/// Eliminating a child turns its remaining (higher-ranked) neighbours `up(c)` into a
/// clique, and `r` is by construction the lowest-ranked member of that clique, so `r`
/// becomes adjacent to all the others — which are exactly `up(c)` minus `r`. No clique is
/// ever enumerated pairwise; each rank just inherits its children's sets.
///
/// Increasing rank order is automatically children-before-parents (`parent[c] > c`), which
/// is also why a single flat buffer works where C needed an array-of-arrays: every child's
/// row is already final and contiguous in `cch_i` when its parent reads it.
fn chordal_supergraph(
    skeleton: &Csr,
    perm: &[u32],
    iperm: &[u32],
    parent: &[u32],
) -> (Vec<u64>, Vec<u32>) {
    let n = skeleton.n;
    let (child_p, child_i) = children_of(parent);

    let mut cch_p: Vec<u64> = Vec::with_capacity(n + 1);
    cch_p.push(0);
    let mut cch_i: Vec<u32> = Vec::with_capacity(skeleton.i.len());
    // `mark[w] == stamp` means "w is already in the row being built". A per-rank stamp of
    // `rank + 1` (never 0) stands in for clearing `mark` on every rank.
    let mut mark = vec![0u32; n];
    let mut row: Vec<u32> = Vec::new();

    for rank in 0..n as u32 {
        row.clear();
        let stamp = rank + 1;

        // 1. rank's own upward neighbours in the skeleton
        let u = perm[rank as usize] as usize;
        for &v in skeleton.row(u) {
            let w = iperm[v as usize];
            if w > rank && mark[w as usize] != stamp {
                mark[w as usize] = stamp;
                row.push(w);
            }
        }

        // 2. absorb each child's upward set, minus `rank` itself
        let kids = child_p[rank as usize] as usize..child_p[rank as usize + 1] as usize;
        for &child in &child_i[kids] {
            let lo = cch_p[child as usize] as usize;
            let hi = cch_p[child as usize + 1] as usize;
            for &w in &cch_i[lo..hi] {
                if w != rank && mark[w as usize] != stamp {
                    mark[w as usize] = stamp;
                    row.push(w);
                }
            }
        }

        // ascending, so Phase 2 can binary-search an arc and Phase 3 walks in rank order
        row.sort_unstable();
        cch_i.extend_from_slice(&row);
        cch_p.push(cch_i.len() as u64);
    }

    (cch_p, cch_i)
}

/// Group the elimination tree's children by parent, as a CSR over ranks. A counting sort:
/// `child_i[child_p[r]..child_p[r + 1]]` are the ranks whose `parent` is `r`.
fn children_of(parent: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let n = parent.len();
    let mut child_p = vec![0u32; n + 1];
    for &p in parent {
        if p != NONE {
            child_p[p as usize + 1] += 1;
        }
    }
    for r in 0..n {
        child_p[r + 1] += child_p[r];
    }

    let mut child_i = vec![0u32; child_p[n] as usize];
    let mut fill = child_p.clone();
    for (c, &p) in parent.iter().enumerate() {
        if p != NONE {
            child_i[fill[p as usize] as usize] = c as u32;
            fill[p as usize] += 1;
        }
    }

    (child_p, child_i)
}

/// Transpose the upward CSR into the downward one. Filling in increasing `lo` order leaves
/// each downward row sorted ascending, mirroring the upward rows.
fn invert_adjacency(
    cch_p: &[u64],
    cch_i: &[u32],
) -> (Vec<u64>, Vec<u32>) {
    let n = cch_p.len() - 1;

    let mut down_p = vec![0u64; n + 1];
    for &hi in cch_i {
        down_p[hi as usize + 1] += 1;
    }
    for r in 0..n {
        down_p[r + 1] += down_p[r];
    }

    let mut down_i = vec![0u32; cch_i.len()];
    let mut fill = down_p.clone();
    for lo in 0..n {
        for &hi in &cch_i[cch_p[lo] as usize..cch_p[lo + 1] as usize] {
            down_i[fill[hi as usize] as usize] = lo as u32;
            fill[hi as usize] += 1;
        }
    }

    (down_p, down_i)
}

#[cfg(test)]
mod tests {
    use super::{
        CchError, Csr, IdentityOrdering, MetisOrdering, OrderError, Ordering, Topology,
        csc_pattern, invert, metis_perm,
    };
    use crate::index::cch::testkit::{
        EdgeList, disconnected_components, empty, grid, path, random_sparse, single_node, star,
        to_input, with_isolated_nodes, with_parallel_edges, with_self_loops,
    };

    /// `feral-metis` handed the whole skeleton in one call, as [`MetisOrdering`] did before
    /// it split components itself: the baseline the split is measured against.
    struct WholeGraphMetis;
    impl Ordering for WholeGraphMetis {
        fn order(
            &self,
            skeleton: &Csr,
        ) -> Result<Vec<u32>, OrderError> {
            let n = skeleton.n;
            let (col_ptr, row_idx) = csc_pattern(skeleton, 0..n as u32, |r| r).unwrap();
            Ok(metis_perm(n, &col_ptr, &row_idx)?
                .into_iter()
                .map(|p| p as u32)
                .collect())
        }
    }

    /// A grid, a scatter of tiny components (pairs, 3-paths, 4-cycles, 5-paths) and
    /// `isolated` trailing isolated nodes — the shape of a `node_cap`-sized road network.
    fn many_components(
        side: u32,
        tiny: usize,
        isolated: usize,
    ) -> EdgeList {
        let mut g = grid(side, side);
        for t in 0..tiny {
            let size = [2usize, 3, 4, 5][t % 4];
            let base = g.n as u32;
            g.n += size;
            for k in 1..size as u32 {
                g.push(base + k - 1, base + k, 1.0);
            }
            if size == 4 {
                g.push(base + 3, base, 1.0);
            }
        }
        with_isolated_nodes(g, isolated)
    }

    /// Every structural property Phase 2 and Phase 3 are entitled to assume. Any one of
    /// them failing means the elimination game is wrong, whatever the weights later say.
    fn check_invariants(
        topo: &Topology,
        skeleton: &Csr,
    ) {
        let n = skeleton.n;
        assert_eq!(topo.node_count(), n);
        assert_eq!(topo.perm.len(), n);
        assert_eq!(topo.iperm.len(), n);

        // perm and iperm are mutual inverses
        for node in 0..n as u32 {
            assert_eq!(
                topo.node_of(topo.rank_of(node)),
                node,
                "iperm/perm at {node}"
            );
        }

        // up rows: strictly ascending, strictly above their own rank
        for rank in 0..n as u32 {
            let up = topo.up(rank);
            for w in up.windows(2) {
                assert!(w[0] < w[1], "up({rank}) not strictly ascending");
            }
            for &w in up {
                assert!(w > rank, "up({rank}) holds {w}, not above it");
            }
            // down rows: strictly ascending, strictly below
            let down = topo.down(rank);
            for w in down.windows(2) {
                assert!(w[0] < w[1], "down({rank}) not strictly ascending");
            }
            for &w in down {
                assert!(w < rank, "down({rank}) holds {w}, not below it");
            }
        }

        // up and down are exact inverses, in both directions
        let mut up_arcs: Vec<(u32, u32)> = Vec::new();
        let mut down_arcs: Vec<(u32, u32)> = Vec::new();
        for rank in 0..n as u32 {
            up_arcs.extend(topo.up(rank).iter().map(|&hi| (rank, hi)));
            down_arcs.extend(topo.down(rank).iter().map(|&lo| (lo, rank)));
        }
        up_arcs.sort_unstable();
        down_arcs.sort_unstable();
        assert_eq!(up_arcs, down_arcs, "up and down disagree");
        assert_eq!(topo.arc_count(), up_arcs.len());

        // every skeleton edge survives as a chordal arc
        for u in 0..n {
            for &v in skeleton.row(u) {
                if v as usize == u {
                    continue; // the skeleton is self-loop-free, but do not depend on it
                }
                let (a, b) = (topo.rank_of(u as u32), topo.rank_of(v));
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                assert!(
                    topo.arc_slot(lo, hi).is_some(),
                    "skeleton edge {u}-{v} (ranks {lo},{hi}) missing from the supergraph"
                );
            }
        }

        // the chordal property itself: eliminating `x` makes its upper neighbours a
        // clique, so every pair in up(x) must already be adjacent. Phase 2's triangle
        // relaxation looks up exactly these arcs and treats a miss as a logic error.
        for x in 0..n as u32 {
            let up = topo.up(x);
            for (i, &y) in up.iter().enumerate() {
                for &z in &up[i + 1..] {
                    assert!(
                        topo.arc_slot(y, z).is_some(),
                        "chordal closure broken: {x} has upper neighbours {y} and {z}, \
                         but no arc joins them"
                    );
                }
            }
        }

        // arc_slot agrees with the flat layout the weight arrays are indexed by
        for lo in 0..n as u32 {
            let base = topo.up_range(lo).start;
            for (i, &hi) in topo.up(lo).iter().enumerate() {
                assert_eq!(topo.arc_slot(lo, hi), Some(base + i));
            }
        }
    }

    fn build_both(edges: &EdgeList) -> (Topology, Topology) {
        let input = to_input(edges);
        let metis = Topology::build(&input.skeleton).expect("metis build");
        let identity =
            Topology::build_with(&input.skeleton, &IdentityOrdering).expect("identity build");
        check_invariants(&metis, &input.skeleton);
        check_invariants(&identity, &input.skeleton);
        (metis, identity)
    }

    #[test]
    fn invariants_hold_on_a_grid() {
        build_both(&grid(6, 7));
    }

    #[test]
    fn invariants_hold_on_random_sparse_graphs() {
        for seed in 0..8u64 {
            let g = random_sparse(60, 180, 9, seed);
            build_both(&g);
        }
    }

    #[test]
    fn invariants_hold_on_a_disconnected_graph() {
        build_both(&disconnected_components(4, 12, 24, 5, 0xC0FF_EE));
    }

    /// Thousands of isolated nodes and tiny components around one large one: a valid,
    /// deterministic order, and exactly the one `feral-metis` produces on the whole graph —
    /// the crate also numbers components by lowest node id, and isolated or tiny components
    /// draw nothing from its RNG, so splitting them off here must change nothing.
    #[test]
    fn many_components_order_like_the_whole_graph() {
        for edges in [
            many_components(12, 300, 5000),
            with_isolated_nodes(grid(15, 15), 3000),
            many_components(3, 2000, 0),
            empty(4000),
        ] {
            let input = to_input(&edges);
            let split = Topology::build(&input.skeleton).expect("split build");
            check_invariants(&split, &input.skeleton);
            let again = Topology::build(&input.skeleton).expect("split build again");
            assert_eq!(split.perm, again.perm, "not deterministic");
            let whole =
                Topology::build_with(&input.skeleton, &WholeGraphMetis).expect("whole build");
            assert_eq!(split.perm, whole.perm, "n = {}", edges.n);
            assert_eq!(split.cch_i, whole.cch_i);
        }
    }

    /// Several components each big enough for real nested dissection, mixed with tiny and
    /// isolated ones. These are the components that draw from the backend's RNG, so this is
    /// what breaks if they stop sharing one backend call: ordered one call each, 27 of 40
    /// seeds of this shape came out worse, by 0.7% more arcs on average.
    #[test]
    fn several_large_components_order_like_the_whole_graph() {
        for seed in 0..8u64 {
            let mut edges = disconnected_components(4, 400, 1200, 9, seed);
            edges.n += 3; // isolated nodes between the large components and the tiny ones
            let base = edges.n as u32;
            edges.n += 6;
            for (u, v) in [(0, 1), (1, 2), (3, 4), (4, 5), (5, 3)] {
                edges.push(base + u, base + v, 1.0);
            }
            let edges = with_isolated_nodes(edges, 50);
            let input = to_input(&edges);
            let split = Topology::build(&input.skeleton).expect("split build");
            check_invariants(&split, &input.skeleton);
            let whole =
                Topology::build_with(&input.skeleton, &WholeGraphMetis).expect("whole build");
            assert_eq!(split.perm, whole.perm, "seed {seed}");
            assert_eq!(split.cch_i, whole.cch_i, "seed {seed}");
        }
    }

    /// Timing guard for the component split; run with
    /// `cargo test -p graph --release -- --ignored padded_skeleton --nocapture`.
    /// A 400x400 grid, then the same grid padded by 25% isolated rows as `node_cap` growth
    /// leaves it: padding must not cost more than the grid itself. The whole-graph backend
    /// call is timed alongside as the before.
    #[test]
    #[ignore = "timing; run in release"]
    fn padded_skeleton_orders_in_linear_time() {
        let time = |o: &dyn Ordering, s: &Csr| {
            let t = std::time::Instant::now();
            let perm = o.order(s).expect("order");
            (perm, t.elapsed().as_secs_f64())
        };
        let bare = to_input(&grid(400, 400));
        let padded = to_input(&with_isolated_nodes(grid(400, 400), 40_000));
        let (_, bare_s) = time(&MetisOrdering, &bare.skeleton);
        let (split, split_s) = time(&MetisOrdering, &padded.skeleton);
        let (whole, whole_s) = time(&WholeGraphMetis, &padded.skeleton);
        println!(
            "grid 400x400: {bare_s:.3}s; padded to {}: split {split_s:.3}s, whole graph {whole_s:.3}s",
            padded.skeleton.n
        );
        assert_eq!(split, whole);
        assert!(
            split_s < 2.0 * bare_s,
            "padding cost {split_s:.3}s against {bare_s:.3}s unpadded"
        );
    }

    #[test]
    fn invariants_hold_with_parallel_edges_and_self_loops() {
        let pairs: Vec<(u32, u32)> = (0..11u32).map(|u| (u, (u + 1) % 12)).collect();
        build_both(&with_parallel_edges(12, &pairs, 3, 7, 42));
        build_both(&with_self_loops(grid(3, 4), &[0, 5, 11], 2.0));
    }

    /// `csc_pattern` hands the skeleton to `feral-metis` as it is, without removing
    /// self-loops or duplicates, so this pins that the backend (and the rest of Phase 1) is
    /// indifferent to them: same permutation, same chordal supergraph, and the same again on
    /// a second run. If a `feral-metis` upgrade started to trust its input, this fails and
    /// `csc_pattern` has to filter again.
    #[test]
    fn an_unclean_skeleton_orders_like_a_clean_one() {
        for g in [
            grid(5, 6),
            random_sparse(60, 180, 9, 3),
            disconnected_components(4, 12, 24, 5, 0xC0FF_EE),
        ] {
            let clean = to_input(&g).skeleton;

            // Every row gains its own diagonal entry and a repeat of its first neighbour,
            // kept sorted as `CscPattern::new` demands.
            let mut dirty = Csr {
                n: clean.n,
                p: vec![0],
                i: Vec::new(),
            };
            for u in 0..clean.n {
                let mut row = clean.row(u).to_vec();
                row.push(u as u32);
                if let Some(&first) = clean.row(u).first() {
                    row.push(first);
                }
                row.sort_unstable();
                dirty.i.extend_from_slice(&row);
                dirty.p.push(dirty.i.len() as u64);
            }
            assert!(
                dirty.i.len() > clean.i.len(),
                "the skeleton was not dirtied"
            );

            let a = Topology::build(&clean).expect("clean build");
            let b = Topology::build(&dirty).expect("dirty build");
            let again = Topology::build(&dirty).expect("dirty build, again");
            assert_eq!(a.perm, b.perm, "dirty entries changed the order");
            assert_eq!(a.cch_p, b.cch_p);
            assert_eq!(a.cch_i, b.cch_i);
            assert_eq!(b.perm, again.perm, "the order is not deterministic");
            check_invariants(&b, &clean);
        }
    }

    #[test]
    fn invariants_hold_on_a_star_and_a_path() {
        build_both(&star(9, 1.0));
        build_both(&path(15));
    }

    #[test]
    fn empty_graph_builds_to_nothing() {
        let input = to_input(&empty(0));
        let topo = Topology::build(&input.skeleton).expect("empty build");
        assert_eq!(topo.node_count(), 0);
        assert_eq!(topo.arc_count(), 0);
        check_invariants(&topo, &input.skeleton);
    }

    #[test]
    fn single_node_builds_to_one_isolated_rank() {
        let input = to_input(&single_node());
        let topo = Topology::build(&input.skeleton).expect("single node build");
        assert_eq!(topo.node_count(), 1);
        assert_eq!(topo.arc_count(), 0);
        assert!(topo.up(0).is_empty());
        assert!(topo.down(0).is_empty());
        check_invariants(&topo, &input.skeleton);
    }

    #[test]
    fn nodes_without_edges_build_to_isolated_ranks() {
        let input = to_input(&empty(5));
        let topo = Topology::build(&input.skeleton).expect("edgeless build");
        assert_eq!(topo.arc_count(), 0);
        check_invariants(&topo, &input.skeleton);
    }

    /// The worked example in the C header (`cch.h`, "worked example"): a 4-node star
    /// eliminated hub-first forces the three leaves into a fill-in triangle.
    /// `IdentityOrdering` reproduces that order exactly (hub is node 0).
    #[test]
    fn hub_first_star_matches_the_c_worked_example() {
        let input = to_input(&star(3, 1.0));
        let topo = Topology::build_with(&input.skeleton, &IdentityOrdering).expect("build");

        assert_eq!(topo.up(0), &[1, 2, 3], "up(X) is its 3 original edges");
        assert_eq!(topo.up(1), &[2, 3], "up(A) absorbs up(X) minus A");
        assert_eq!(topo.up(2), &[3], "up(B) absorbs up(A) minus B");
        assert_eq!(topo.up(3), &[] as &[u32], "up(C) is the root");

        assert_eq!(topo.down(0), &[] as &[u32]);
        assert_eq!(topo.down(1), &[0]);
        assert_eq!(topo.down(2), &[0, 1]);
        assert_eq!(topo.down(3), &[0, 1, 2]);
    }

    /// Reversing that order (hub last) is the good one: no fill-in at all.
    #[test]
    fn hub_last_star_creates_no_fill_in() {
        /// `perm[rank] = rank + 1` for the leaves, hub (node 0) last.
        struct HubLast;
        impl Ordering for HubLast {
            fn order(
                &self,
                skeleton: &Csr,
            ) -> Result<Vec<u32>, OrderError> {
                let n = skeleton.n as u32;
                Ok((1..n).chain(std::iter::once(0)).collect())
            }
        }

        let input = to_input(&star(3, 1.0));
        let topo = Topology::build_with(&input.skeleton, &HubLast).expect("build");
        check_invariants(&topo, &input.skeleton);

        // 3 leaves each hold one arc to the hub, and the hub holds none
        assert_eq!(topo.arc_count(), 3);
        assert_eq!(topo.up(0), &[3]);
        assert_eq!(topo.up(1), &[3]);
        assert_eq!(topo.up(2), &[3]);
        assert_eq!(topo.up(3), &[] as &[u32]);
    }

    /// Determinism is what makes a rebuilt index equal to the original: same skeleton in,
    /// same six arrays out. `feral-metis`'s default seed is 1, so this covers the backend
    /// too, not just this module's bookkeeping.
    #[test]
    fn the_same_skeleton_always_produces_the_same_hierarchy() {
        for edges in [grid(5, 9), random_sparse(80, 240, 9, 7), star(12, 1.0)] {
            let input = to_input(&edges);
            let a = Topology::build(&input.skeleton).expect("build a");
            let b = Topology::build(&input.skeleton).expect("build b");
            assert_eq!(a.perm, b.perm);
            assert_eq!(a.iperm, b.iperm);
            assert_eq!(a.cch_p, b.cch_p);
            assert_eq!(a.cch_i, b.cch_i);
            assert_eq!(a.down_p, b.down_p);
            assert_eq!(a.down_i, b.down_i);
        }
    }

    /// Nested dissection exists to keep the supergraph small. On a 2D grid, eliminating in
    /// node-id order (row-major) is the pathological case; METIS should not lose to it.
    /// The number printed is the measurement, the assertion is the guard.
    #[test]
    fn nested_dissection_beats_node_id_order_on_a_grid() {
        let (metis, identity) = build_both(&grid(20, 20));
        println!(
            "grid 20x20: metis {} arcs, identity {} arcs",
            metis.arc_count(),
            identity.arc_count()
        );
        assert!(
            metis.arc_count() <= identity.arc_count(),
            "metis {} arcs vs identity {} arcs",
            metis.arc_count(),
            identity.arc_count()
        );
    }

    /// The ordering backend is `i32`-indexed, so its ceiling is below the contract's
    /// `u32::MAX`. Refuse, do not panic — and do it before touching the (here deliberately
    /// truncated) offset array, which is what makes this testable without 17 GB of RAM.
    #[test]
    fn a_graph_past_the_backend_ceiling_is_refused() {
        let too_big = Csr {
            n: i32::MAX as usize + 1,
            p: vec![0],
            i: Vec::new(),
        };
        assert_eq!(MetisOrdering.order(&too_big), Err(OrderError::TooLarge));
        assert_eq!(IdentityOrdering.order(&too_big), Err(OrderError::TooLarge));
        assert_eq!(
            Topology::build(&too_big).err(),
            Some(CchError::GraphTooLarge),
            "a graph too large for the backend is refused, not degraded"
        );
    }

    /// A backend failure is a hard error carrying the backend's reason — not a fallback to
    /// some other order, which would hide a bug behind a slower index.
    #[test]
    fn a_backend_failure_is_a_hard_error() {
        struct AlwaysFails;
        impl Ordering for AlwaysFails {
            fn order(
                &self,
                _skeleton: &Csr,
            ) -> Result<Vec<u32>, OrderError> {
                Err(OrderError::Backend("deliberate test failure"))
            }
        }

        let input = to_input(&grid(3, 3));
        assert_eq!(
            Topology::build_with(&input.skeleton, &AlwaysFails).err(),
            Some(CchError::Internal("deliberate test failure"))
        );
    }

    /// A backend that returns something other than a permutation is caught, not trusted.
    #[test]
    fn a_backend_returning_a_non_permutation_is_a_hard_error() {
        struct Duplicates;
        impl Ordering for Duplicates {
            fn order(
                &self,
                skeleton: &Csr,
            ) -> Result<Vec<u32>, OrderError> {
                Ok(vec![0; skeleton.n])
            }
        }

        let input = to_input(&grid(2, 2));
        assert_eq!(
            Topology::build_with(&input.skeleton, &Duplicates).err(),
            Some(CchError::Internal(
                "ordering backend returned a non-permutation"
            ))
        );
    }

    #[test]
    fn a_non_permutation_is_rejected() {
        assert_eq!(invert(&[0, 1, 2], 3), Some(vec![0, 1, 2]));
        assert_eq!(invert(&[2, 0, 1], 3), Some(vec![1, 2, 0]));
        assert_eq!(invert(&[0, 0, 1], 3), None, "duplicate node id");
        assert_eq!(invert(&[0, 1, 9], 3), None, "node id out of range");
        assert_eq!(invert(&[0, 1], 3), None, "wrong length");
    }
}
