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
//! (`src/algorithms/cch.c`). The C implementation orders with C METIS (`METIS_NodeND`);
//! this one with `feral-metis`, a pure-Rust multilevel nested dissection (see [`nd`]),
//! behind the [`Ordering`] trait. One deliberate difference: the chordal supergraph is one flat CSR
//! pair instead of C's array-of-arrays, which is possible because child absorption only
//! ever reads *lower* ranks — already finished and contiguous in the flat buffer by the
//! time their parent is processed.

// Phase 1 lands on its own, before anything outside this file calls it: `Cch` grows its
// topology field in the customization change (#3209), and the tests below are the only caller until then.
#![allow(dead_code)]

use std::ffi::CString;

use super::{CchError, Csr, nd};
use crate::index::redisearch::redis::RedisModule_Log;

/// Sentinel for "no such rank" in [`Topology`]'s scratch arrays. A graph has at most
/// [`MAX_NODES`](super::MAX_NODES) `< u64::MAX` nodes, so no real rank can collide with it.
pub(super) const NONE: u64 = u64::MAX;

/// Why an elimination order could not be produced.
///
/// Deliberately not [`CchError`]: the ordering backend is swappable, so its failure modes
/// stay abstract here and are mapped at the single call site in [`Topology::build_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderError {
    /// Beyond what this backend can index. [`NestedDissection`] indexes with `i64`, so for
    /// it that is more nodes than an `i64` counts, far past
    /// [`MAX_NODES`](super::MAX_NODES); [`check_size`](super::check_size) refuses graphs
    /// well below it, so through [`Cch::build`](super::Cch::build) this is a backstop.
    TooLarge,
    /// The skeleton is malformed, the backend rejected it, or the backend failed (an
    /// running out of memory included). Unreachable for a skeleton that meets
    /// [`CchInput`](super::CchInput)'s contract on a machine with the memory for it; it
    /// means a bug here or in the backend, never bad user input.
    Backend(&'static str),
}

/// A pluggable elimination-order backend.
///
/// Implementors take the undirected, self-loop-free skeleton and return `perm`, where
/// `perm[rank]` is the node id eliminated at that rank — "new-to-old", the same convention
/// METIS itself uses. The result must be a permutation of `0..skeleton.n`;
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
    ) -> Result<Vec<u64>, OrderError>;
}

/// Nested dissection by `feral-metis` with its default options: multilevel bisection into
/// node separators, approximate minimum degree on the small pieces.
///
/// The seed is fixed and each call owns its generator (see [`nd`]), so the same skeleton
/// always yields the same order, which is what makes a
/// rebuilt index byte-identical to the original.
///
/// Panics on a skeleton that is not symmetric and simple, which only a bug can produce
/// (see `nd::nd_graph`).
#[derive(Debug, Clone, Copy, Default)]
pub struct NestedDissection;

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
    ) -> Result<Vec<u64>, OrderError> {
        if skeleton.n > super::MAX_NODES {
            return Err(OrderError::TooLarge);
        }
        Ok((0..skeleton.n as u64).collect())
    }
}

impl Ordering for NestedDissection {
    fn order(
        &self,
        skeleton: &Csr,
    ) -> Result<Vec<u64>, OrderError> {
        // Past the graph's own ceiling no id can be numbered (and `NONE` stays free).
        if skeleton.n > super::MAX_NODES {
            return Err(OrderError::TooLarge);
        }
        // One call for the whole skeleton, all its components together: the backend orders
        // a disconnected graph itself. Only the isolated nodes are set aside (see
        // `nd::nd_graph`).
        nd::node_nd(nd::nd_graph(skeleton))
    }
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
/// The chordal supergraph is stored twice, as two [`Csr`]s over ranks:
///
/// * `up` (`cch_p`/`cch_i` in the docs: its `p` and `i`) — the *upward* graph. Row `r`
///   lists every rank `> r` that `r` is adjacent to once fill-in is accounted for, sorted
///   ascending. This is the array Phase 2 writes weights parallel to and Phase 3 walks.
/// * `down` — its exact inverse (the transpose). `lo` is in row `hi` of `down` iff `hi` is
///   in row `lo` of `up`. Also sorted ascending. Only an incremental recustomization needs
///   it; the full sweep ignores it.
///
/// An **arc slot** is an index into `cch_i`: slot `k` with `cch_p[r] <= k < cch_p[r + 1]`
/// is the arc between rank `r` and rank `cch_i[k]`. Phase 2's four weight arrays are
/// indexed by exactly this slot, so `up_w[k]` is the `r -> cch_i[k]` weight and `dn_w[k]`
/// the `cch_i[k] -> r` weight. [`Topology::arc_slot`] resolves a rank pair to its slot.
pub(crate) struct Topology {
    n: usize,
    /// `perm[rank] = node id`.
    perm: Box<[u64]>,
    /// `iperm[node id] = rank`.
    iperm: Box<[u64]>,
    /// Upward adjacency, rank space.
    up: Csr,
    /// Downward adjacency, rank space: the transpose of `up`.
    down: Csr,
}

impl Topology {
    /// Phase 1 with the default backend ([`NestedDissection`]).
    ///
    /// # Errors
    /// [`CchError::GraphTooLarge`] when the graph exceeds the ordering backend's index
    /// width (more nodes than the backend's `i64` indices count; [`check_size`](super::check_size)
    /// refuses far less, so through `Cch::build` this is a backstop), and
    /// [`CchError::Internal`] if the backend fails for any other reason.
    pub(crate) fn build(skeleton: &Csr) -> Result<Self, CchError> {
        Self::build_with(skeleton, &NestedDissection)
    }

    /// Phase 1 with an explicit backend — what the tests use to pin a known order.
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
        perm: Vec<u64>,
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
        let up = chordal_supergraph(skeleton, &perm, &iperm, &parent);
        let down = transpose(&up);

        Ok(Self {
            n,
            perm: perm.into(),
            iperm: iperm.into(),
            up,
            down,
        })
    }

    pub(crate) fn node_count(&self) -> usize {
        self.n
    }

    /// Total arcs in the chordal supergraph — also the length of each Phase 2 weight array.
    pub(crate) fn arc_count(&self) -> usize {
        self.up.i.len()
    }

    /// `perm[rank]`.
    pub(crate) fn node_of(
        &self,
        rank: u64,
    ) -> u64 {
        self.perm[rank as usize]
    }

    /// `iperm[node]`.
    pub(crate) fn rank_of(
        &self,
        node: u64,
    ) -> u64 {
        self.iperm[node as usize]
    }

    /// Every rank above `rank` in the chordal supergraph, ascending.
    pub(crate) fn up(
        &self,
        rank: u64,
    ) -> &[u64] {
        self.up.row(rank as usize)
    }

    /// The arc-slot range of `rank`'s upward arcs: `up(rank)[i]` is slot `up_range(rank).start + i`.
    pub(crate) fn up_range(
        &self,
        rank: u64,
    ) -> std::ops::Range<usize> {
        self.up.p[rank as usize] as usize..self.up.p[rank as usize + 1] as usize
    }

    /// Every rank below `rank` in the chordal supergraph, ascending.
    pub(crate) fn down(
        &self,
        rank: u64,
    ) -> &[u64] {
        self.down.row(rank as usize)
    }

    /// The arc slot of the arc `lo -> hi`, or `None` if the two are not adjacent in the
    /// chordal supergraph. `lo` must be the lower rank.
    ///
    /// Phase 2 calls this once per triangle, so it is a binary search over an ascending row
    /// rather than a hash lookup.
    pub(crate) fn arc_slot(
        &self,
        lo: u64,
        hi: u64,
    ) -> Option<usize> {
        debug_assert!(lo < hi, "arc_slot takes the lower rank first");
        let base = self.up.p[lo as usize] as usize;
        self.up(lo).binary_search(&hi).ok().map(|i| base + i)
    }

    /// Heap bytes held by the six structural arrays.
    pub(crate) fn memory_usage(&self) -> usize {
        let words = self.perm.len()
            + self.iperm.len()
            + self.up.p.len()
            + self.up.i.len()
            + self.down.p.len()
            + self.down.i.len();
        size_of::<Self>() + words * size_of::<u64>()
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
    perm: &[u64],
    n: usize,
) -> Option<Vec<u64>> {
    if perm.len() != n {
        return None;
    }
    let mut iperm = vec![NONE; n];
    for (rank, &node) in perm.iter().enumerate() {
        let slot = iperm.get_mut(node as usize)?;
        if *slot != NONE {
            return None; // node appears twice
        }
        *slot = rank as u64;
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
    perm: &[u64],
    iperm: &[u64],
) -> Vec<u64> {
    let n = skeleton.n;
    let mut parent = vec![NONE; n];
    // `ancestor[x]` is the highest rank known so far to be connected to `x` once fill-in is
    // accounted for — x's current representative — or NONE if x has never been reached.
    let mut ancestor = vec![NONE; n];

    for rank in 0..n as u64 {
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

/// The chordal supergraph as the upward CSR over ranks.
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
    perm: &[u64],
    iperm: &[u64],
    parent: &[u64],
) -> Csr {
    let n = skeleton.n;
    let children = children_of(parent);

    let mut cch_p: Vec<u64> = Vec::with_capacity(n + 1);
    cch_p.push(0);
    let mut cch_i: Vec<u64> = Vec::with_capacity(skeleton.i.len());
    // `mark[w] == stamp` means "w is already in the row being built". A per-rank stamp of
    // `rank + 1` (never 0) stands in for clearing `mark` on every rank.
    let mut mark: Vec<u64> = vec![0; n];
    let mut row: Vec<u64> = Vec::new();

    for rank in 0..n as u64 {
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
        for &child in children.row(rank as usize) {
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

    Csr {
        n,
        p: cch_p.into(),
        i: cch_i.into(),
    }
}

/// Group the elimination tree's children by parent, as a CSR over ranks: row `r` holds
/// the ranks whose `parent` is `r`, ascending. A counting sort.
fn children_of(parent: &[u64]) -> Csr {
    let n = parent.len();
    let mut child_p: Vec<u64> = vec![0; n + 1];
    for &p in parent {
        if p != NONE {
            child_p[p as usize + 1] += 1;
        }
    }
    for r in 0..n {
        child_p[r + 1] += child_p[r];
    }

    let mut child_i: Vec<u64> = vec![0; child_p[n] as usize];
    let mut fill = child_p.clone();
    for (c, &p) in parent.iter().enumerate() {
        if p != NONE {
            child_i[fill[p as usize] as usize] = c as u64;
            fill[p as usize] += 1;
        }
    }

    Csr {
        n,
        p: child_p.into(),
        i: child_i.into(),
    }
}

/// The transpose of `up`, the downward CSR. A counting sort: filling in increasing row
/// order leaves each transposed row sorted ascending.
fn transpose(up: &Csr) -> Csr {
    let n = up.n;

    let mut down_p = vec![0u64; n + 1];
    for &hi in &up.i {
        down_p[hi as usize + 1] += 1;
    }
    for r in 0..n {
        down_p[r + 1] += down_p[r];
    }

    let mut down_i = vec![0u64; up.i.len()];
    let mut fill = down_p.clone();
    for lo in 0..n {
        for &hi in up.row(lo) {
            down_i[fill[hi as usize] as usize] = lo as u64;
            fill[hi as usize] += 1;
        }
    }

    Csr {
        n,
        p: down_p.into(),
        i: down_i.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CchError, Csr, IdentityOrdering, NestedDissection, OrderError, Ordering, Topology, invert,
    };
    use crate::index::falkordb::cch::MAX_NODES;
    use crate::index::falkordb::cch::testkit::{
        EdgeList, disconnected_components, empty, grid, path, random_sparse, single_node, star,
        to_input, with_isolated_nodes, with_parallel_edges, with_self_loops,
    };

    /// A grid, a scatter of tiny components (pairs, 3-paths, 4-cycles, 5-paths) and
    /// `isolated` trailing isolated nodes — the shape of a `node_cap`-sized road network.
    fn many_components(
        side: u64,
        tiny: usize,
        isolated: usize,
    ) -> EdgeList {
        let mut g = grid(side, side);
        for t in 0..tiny {
            let size = [2usize, 3, 4, 5][t % 4];
            let base = g.n as u64;
            g.n += size;
            for k in 1..size as u64 {
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
        for node in 0..n as u64 {
            assert_eq!(
                topo.node_of(topo.rank_of(node)),
                node,
                "iperm/perm at {node}"
            );
        }

        // up rows: strictly ascending, strictly above their own rank
        for rank in 0..n as u64 {
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
        let mut up_arcs: Vec<(u64, u64)> = Vec::new();
        let mut down_arcs: Vec<(u64, u64)> = Vec::new();
        for rank in 0..n as u64 {
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
                let (a, b) = (topo.rank_of(u as u64), topo.rank_of(v));
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
        for x in 0..n as u64 {
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
        for lo in 0..n as u64 {
            let base = topo.up_range(lo).start;
            for (i, &hi) in topo.up(lo).iter().enumerate() {
                assert_eq!(topo.arc_slot(lo, hi), Some(base + i));
            }
        }
    }

    fn build_both(edges: &EdgeList) -> (Topology, Topology) {
        let input = to_input(edges);
        let nd = Topology::build(&input.skeleton).expect("nested dissection build");
        let identity =
            Topology::build_with(&input.skeleton, &IdentityOrdering).expect("identity build");
        check_invariants(&nd, &input.skeleton);
        check_invariants(&identity, &input.skeleton);
        (nd, identity)
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
        build_both(&disconnected_components(4, 12, 24, 5, 0x00C0_FFEE));
    }

    /// Thousands of isolated nodes and tiny components around large ones — the shape of a
    /// `node_cap`-sized road network — in one backend call: a valid order, and the same one
    /// on every run.
    #[test]
    fn many_components_order_validly_and_deterministically() {
        let mut several_large = disconnected_components(4, 400, 1200, 9, 3);
        several_large.n += 3; // isolated nodes between the large components and the tiny ones
        let base = several_large.n as u64;
        several_large.n += 6;
        for (u, v) in [(0, 1), (1, 2), (3, 4), (4, 5), (5, 3)] {
            several_large.push(base + u, base + v, 1.0);
        }
        for edges in [
            many_components(12, 300, 5000),
            with_isolated_nodes(grid(15, 15), 3000),
            many_components(3, 2000, 0),
            empty(4000),
            with_isolated_nodes(several_large, 50),
        ] {
            let input = to_input(&edges);
            let topo = Topology::build(&input.skeleton).expect("build");
            check_invariants(&topo, &input.skeleton);
            let again = Topology::build(&input.skeleton).expect("build again");
            assert_eq!(topo.perm, again.perm, "not deterministic, n = {}", edges.n);
            assert_eq!(topo.up.i, again.up.i);
        }
    }

    /// Isolated nodes take the lowest ranks, in id order, and the rest is ordered exactly
    /// as the backend orders the same graph without them.
    #[test]
    fn isolated_nodes_rank_first_and_do_not_change_the_rest() {
        let compact = grid(9, 7);
        // Interleave 40 isolated nodes: node `u` of the grid becomes `u + u / 2 + 1`.
        let lift = |u: u64| u + u / 2 + 1;
        let mut padded = EdgeList::new(compact.n + compact.n / 2 + 1 + 3);
        for &(u, v, w) in &compact.edges {
            padded.push(lift(u), lift(v), w);
        }
        let used: Vec<u64> = (0..compact.n as u64).map(lift).collect();
        let isolated: Vec<u64> = (0..padded.n as u64).filter(|v| !used.contains(v)).collect();

        let perm = NestedDissection
            .order(&to_input(&padded).skeleton)
            .expect("padded");
        let rest = NestedDissection
            .order(&to_input(&compact).skeleton)
            .expect("compact");
        assert_eq!(perm[..isolated.len()], isolated[..]);
        let lifted: Vec<u64> = rest.into_iter().map(lift).collect();
        assert_eq!(perm[isolated.len()..], lifted[..]);
    }

    /// Timing guard for padding; run with
    /// `cargo test -p graph --release -- --ignored padded_skeleton --nocapture`.
    /// A 400x400 grid, then the same grid padded by 25% isolated rows as `node_cap` growth
    /// leaves it: padding must not cost more than the grid itself.
    #[test]
    #[ignore = "timing; run in release"]
    fn padded_skeleton_orders_in_linear_time() {
        let time = |s: &Csr| {
            let t = std::time::Instant::now();
            let perm = NestedDissection.order(s).expect("order");
            (perm, t.elapsed().as_secs_f64())
        };
        let bare = to_input(&grid(400, 400));
        let padded = to_input(&with_isolated_nodes(grid(400, 400), 40_000));
        let (_, bare_s) = time(&bare.skeleton);
        let (_, padded_s) = time(&padded.skeleton);
        println!(
            "grid 400x400: {bare_s:.3}s; padded to {}: {padded_s:.3}s",
            padded.skeleton.n
        );
        assert!(
            padded_s < 2.0 * bare_s,
            "padding cost {padded_s:.3}s against {bare_s:.3}s unpadded"
        );
    }

    #[test]
    fn invariants_hold_with_parallel_edges_and_self_loops() {
        let pairs: Vec<(u64, u64)> = (0..11).map(|u| (u, (u + 1) % 12)).collect();
        build_both(&with_parallel_edges(12, &pairs, 3, 7, 42));
        build_both(&with_self_loops(grid(3, 4), &[0, 5, 11], 2.0));
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
        assert_eq!(topo.up(3), &[] as &[u64], "up(C) is the root");

        assert_eq!(topo.down(0), &[] as &[u64]);
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
            ) -> Result<Vec<u64>, OrderError> {
                let n = skeleton.n as u64;
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
        assert_eq!(topo.up(3), &[] as &[u64]);
    }

    /// Determinism is what makes a rebuilt index equal to the original: same skeleton in,
    /// same six arrays out. The backend's seed is fixed, so this covers the backend too,
    /// not just this module's bookkeeping.
    #[test]
    fn the_same_skeleton_always_produces_the_same_hierarchy() {
        for edges in [grid(5, 9), random_sparse(80, 240, 9, 7), star(12, 1.0)] {
            let input = to_input(&edges);
            let a = Topology::build(&input.skeleton).expect("build a");
            let b = Topology::build(&input.skeleton).expect("build b");
            assert_eq!(a.perm, b.perm);
            assert_eq!(a.iperm, b.iperm);
            assert_eq!(a.up.p, b.up.p);
            assert_eq!(a.up.i, b.up.i);
            assert_eq!(a.down.p, b.down.p);
            assert_eq!(a.down.i, b.down.i);
        }
    }

    /// Nested dissection exists to keep the supergraph small. On a 2D grid, eliminating in
    /// node-id order (row-major) is the pathological case; nested dissection should not lose to it.
    /// The number printed is the measurement, the assertion is the guard.
    #[test]
    fn nested_dissection_beats_node_id_order_on_a_grid() {
        let (nd, identity) = build_both(&grid(20, 20));
        println!(
            "grid 20x20: nested dissection {} arcs, identity {} arcs",
            nd.arc_count(),
            identity.arc_count()
        );
        assert!(
            nd.arc_count() <= identity.arc_count(),
            "nested dissection {} arcs vs identity {} arcs",
            nd.arc_count(),
            identity.arc_count()
        );
    }

    /// Past the graph's own ceiling, [`MAX_NODES`]. Refuse, do not panic — and do it before
    /// touching the (here deliberately truncated) offset array, which is what makes this
    /// testable without exabytes of RAM.
    #[test]
    fn a_graph_past_the_backend_ceiling_is_refused() {
        let too_big = Csr {
            n: MAX_NODES + 1,
            p: vec![0].into(),
            i: Vec::new().into(),
        };
        assert_eq!(NestedDissection.order(&too_big), Err(OrderError::TooLarge));
        assert_eq!(IdentityOrdering.order(&too_big), Err(OrderError::TooLarge));
        assert_eq!(
            Topology::build(&too_big).err(),
            Some(CchError::GraphTooLarge),
            "a graph too large for the backend is refused, not degraded"
        );
    }

    /// `NestedDissection.order(skeleton)` panics, with `message`.
    fn assert_panics_with(
        skeleton: &Csr,
        message: &str,
        case: &str,
    ) {
        let panic = std::panic::catch_unwind(|| NestedDissection.order(skeleton)).expect_err(case);
        let got = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
        assert_eq!(got, Some(message), "{case}");
    }

    /// A skeleton that would send the backend outside its arrays panics before the call,
    /// in the copy into the backend's arrays: `CchInput::build` cannot produce one, so it
    /// is a bug, never an error a user's graph could cause. Each case is a 3-node path
    /// `0 - 1 - 2` with one thing broken.
    #[test]
    fn a_malformed_skeleton_panics_and_is_not_ordered() {
        let path = |p: Vec<u64>, i: Vec<u64>| Csr {
            n: 3,
            p: p.into(),
            i: i.into(),
        };
        let valid = path(vec![0, 1, 3, 4], vec![1, 0, 2, 1]);
        assert!(NestedDissection.order(&valid).is_ok());

        for (case, skeleton) in [
            (
                "an offset past the end",
                path(vec![0, 1, 9, 4], vec![1, 0, 2, 1]),
            ),
            (
                "decreasing offsets",
                path(vec![0, 3, 1, 4], vec![1, 0, 2, 1]),
            ),
            ("too few offsets", path(vec![0, 1, 3], vec![1, 0, 2, 1])),
            ("an id past n", path(vec![0, 1, 3, 4], vec![1, 0, 7, 1])),
            (
                "an id of 2^32 + 1",
                path(vec![0, 1, 3, 4], vec![1, 0, (1 << 32) + 1, 1]),
            ),
        ] {
            assert_panics_with(&skeleton, "malformed CCH skeleton", case);
        }
    }

    /// The rest of the skeleton contract (sorted rows, no repeats, no self-loops, mirrored
    /// entries) holds by construction, and is checked in every build all the same (a
    /// panic, in release too): `Csr`'s fields are public, and the backend's result on
    /// anything else is undefined.
    #[test]
    fn a_skeleton_that_is_not_simple_undirected_panics() {
        let path = |p: Vec<u64>, i: Vec<u64>| Csr {
            n: 3,
            p: p.into(),
            i: i.into(),
        };
        for (case, skeleton) in [
            ("1 -> 2 not mirrored", path(vec![0, 1, 3, 3], vec![1, 0, 2])),
            (
                "2 -> 1 not mirrored, 2 not isolated",
                path(vec![0, 1, 2, 4], vec![1, 0, 0, 1]),
            ),
            ("row 1 unsorted", path(vec![0, 1, 3, 4], vec![1, 2, 0, 1])),
            (
                "row 1 repeats 0",
                path(vec![0, 1, 4, 5], vec![1, 0, 0, 2, 1]),
            ),
            (
                "a self-loop on 0",
                path(vec![0, 2, 4, 5], vec![0, 1, 0, 2, 1]),
            ),
        ] {
            assert!(skeleton.is_well_formed(), "{case}");
            assert_panics_with(
                &skeleton,
                "the CCH skeleton is not symmetric, sorted, without repeats or self-loops",
                case,
            );
        }
    }

    /// Node ids far above anything the small graphs reach: a grid on the top ids of a
    /// 3M-node id space, the rest isolated, comes back as a permutation of all of them with
    /// the grid's structure intact.
    #[test]
    fn a_large_sparse_id_space_orders_correctly() {
        let mut edges = grid(30, 30);
        let lift = 3_000_000 - edges.n as u64;
        for e in &mut edges.edges {
            e.0 += lift;
            e.1 += lift;
        }
        edges.n = 3_000_000;
        let input = to_input(&edges);
        let topo = Topology::build(&input.skeleton).expect("build");
        let identity =
            Topology::build_with(&input.skeleton, &IdentityOrdering).expect("identity build");
        assert_eq!(topo.node_count(), 3_000_000);
        check_invariants(&topo, &input.skeleton);
        assert!(topo.arc_count() <= identity.arc_count());
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
            ) -> Result<Vec<u64>, OrderError> {
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
            ) -> Result<Vec<u64>, OrderError> {
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
