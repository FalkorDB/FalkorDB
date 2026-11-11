//! Phase 1: nested-dissection ordering and triangulation (#3208).
//!
//! Phase 1 never looks at a weight. It reads the undirected skeleton of the graph
//! ([`CchInput::skeleton`](super::CchInput::skeleton)) and produces the *metric-independent*
//! half of the hierarchy, which later customizations reuse without rerunning any of this:
//!
//! 1. **Elimination order** ([`nd::nested_dissection_order`]) — a nested-dissection
//!    permutation. Nested dissection splits the graph with small node separators, recurses
//!    into each side, and assigns the deepest pieces the lowest ranks so every separator lands
//!    above everything it separates. That placement is what keeps shortcuts local instead of
//!    spreading.
//! 2. **Elimination game** ([`Topology::from_order`]) — eliminate nodes in rank order,
//!    joining each eliminated node's not-yet-eliminated neighbors into a clique. The result is
//!    the *chordal supergraph*: every arc, original or shortcut, that any Phase 2 could ever
//!    need to write a weight into.
//!
//! Ported from the C implementation's `CCH_EliminationOrder` + `CCH_ChordalTriangulation`
//! (`src/algorithms/cch.c`). The C implementation orders with C METIS (`METIS_NodeND`);
//! this one with `feral-metis`, a pure-Rust multilevel nested dissection (see [`nd`]). Two
//! deliberate differences from C: the chordal supergraph is one flat pair of arrays instead
//! of C's array-of-arrays, and there is no separate elimination-tree pass (C's
//! `_build_elimination_tree`): a rank's tree parent is the first entry of its finished upward
//! row, so each row is attached to its parent as soon as it is built
//! ([`chordal_supergraph`]).

use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Range;

use feral_metis::OrderingError;
use thiserror::Error;

use super::{CchError, Csr, id_index, nd};
use crate::graph::graph::NodeId;

/// A position in the elimination order: rank `0` is eliminated first. Its own type, so a rank
/// and a [`NodeId`] cannot be swapped by accident: [`Topology`] converts between the two
/// spaces with [`Topology::node_of`] and [`Topology::rank_of`], and nowhere else.
///
/// Stored as `index + 1` in a [`NonZeroUsize`]: every rank is below
/// [`MAX_NODES`](super::MAX_NODES) `< usize::MAX`, so the sum never overflows and never
/// is zero. That leaves zero free for `None`, so `Option<Rank>` is 8 bytes like `Rank`
/// and needs no sentinel. Adding one keeps the order, so comparing ranks compares indices.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub(crate) struct Rank(NonZeroUsize);

impl Rank {
    /// # Panics
    /// If `index` is `usize::MAX`, which no rank reaches.
    pub(crate) const fn new(index: usize) -> Self {
        match index.checked_add(1) {
            // `index + 1 >= 1`, so this is `Some`.
            Some(stored) => Self(NonZeroUsize::new(stored).unwrap()),
            None => panic!("rank index out of range"),
        }
    }

    /// This rank as an index into a rank-space array.
    pub(crate) const fn index(self) -> usize {
        self.0.get() - 1
    }
}

/// Prints the index, not the stored `index + 1`.
impl fmt::Debug for Rank {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        formatter.debug_tuple("Rank").field(&self.index()).finish()
    }
}

/// Why an elimination order could not be produced. Every case is a bug, never bad user
/// input: [`CchInput`](super::CchInput) cannot produce a skeleton the backend refuses or
/// a graph past [`MAX_NODES`](super::MAX_NODES), so a build that hits one fails with
/// [`CchError::Ordering`] rather than degrading to some other order.
///
/// Running out of memory is not one of them. The backend allocates through the module's
/// global allocator (Redis's `RedisModule_Alloc` in the server), and an allocation that fails
/// there aborts the process, as any allocation in Rust does.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum OrderError {
    /// More nodes than [`MAX_NODES`](super::MAX_NODES), the engine's own graph limit.
    #[error("the graph has more nodes than a CCH index is built for")]
    TooLarge,
    /// The skeleton breaks [`Csr`]'s shape or is not a simple undirected adjacency, so the
    /// backend would index outside its arrays. The message says which.
    #[error("{0}")]
    MalformedSkeleton(&'static str),
    /// The order is not a permutation of the skeleton's nodes. The message says how.
    #[error("{0}")]
    NotAPermutation(&'static str),
    /// The backend refused the skeleton or failed while ordering it.
    #[error("nested dissection failed: {0}")]
    Backend(#[from] OrderingError),
}

/// The metric-independent half of the hierarchy: the elimination order and the chordal
/// supergraph it implies. Shared by `Arc` across customizations — nothing here depends on a
/// weight, so a new metric reuses it untouched.
///
/// # Layout
///
/// Two spaces, kept apart by their types:
///
/// * **node space** — ids as the rest of the engine knows them, [`NodeId`]s `0..n`.
/// * **rank space** — elimination order, [`Rank`]s `0..n`. `rank_to_node[rank] = node`,
///   `node_to_rank[node] = rank`.
///
/// Everything except `rank_to_node`/`node_to_rank` themselves is in rank space.
///
/// The chordal supergraph is stored twice, as two [`RankRows`]:
///
/// * `upward` — row `r` lists every rank above `r` that `r` is adjacent to once fill-in is
///   accounted for, ascending. Phase 2 writes its weights parallel to `upward.neighbors`,
///   and Phase 3 walks it.
/// * `downward` — its transpose: `lower` is in row `higher` of `downward` iff `higher` is
///   in row `lower` of `upward`. Also ascending. Only an incremental recustomization needs
///   it; the full sweep ignores it.
///
/// An **arc slot** is an index into `upward.neighbors`: slot `k` in [`Topology::up_range`]`(r)`
/// is the arc between rank `r` and rank `upward.neighbors[k]`. Phase 2's weight arrays are
/// indexed by exactly this slot. [`Topology::arc_slot`] resolves a rank pair to its slot.
///
/// All six arrays are built once and never grow, so they are boxed slices: no spare
/// capacity is kept for the life of the index.
pub(crate) struct Topology {
    /// `rank_to_node[rank] = node`.
    rank_to_node: Box<[NodeId]>,
    /// `node_to_rank[node] = rank`.
    node_to_rank: Box<[Rank]>,
    /// Upward adjacency.
    upward: RankRows,
    /// Downward adjacency: the transpose of `upward`.
    #[allow(dead_code)] // built with `upward`; scoped recustomization (#3217) is its reader
    downward: RankRows,
}

/// Rows of ranks, one row per rank: a compressed sparse row over rank space.
struct RankRows {
    /// `rank count + 1` offsets into `neighbors`; row `r` is
    /// `neighbors[offsets[r]..offsets[r + 1]]`.
    offsets: Box<[usize]>,
    /// Each row's ranks, ascending.
    neighbors: Box<[Rank]>,
}

impl RankRows {
    /// The slots of `rank`'s row: `neighbors[slots(rank)]` is the row.
    fn slots(
        &self,
        rank: Rank,
    ) -> Range<usize> {
        row_slots(&self.offsets, rank)
    }

    fn row(
        &self,
        rank: Rank,
    ) -> &[Rank] {
        &self.neighbors[self.slots(rank)]
    }

    /// Bytes held by both arrays.
    #[cfg(test)]
    fn memory_usage(&self) -> usize {
        size_of_val(&*self.offsets) + size_of_val(&*self.neighbors)
    }
}

impl Topology {
    /// Phase 1: order `skeleton` by nested dissection ([`nd::nested_dissection_order`]),
    /// then play the elimination game in that order ([`Self::from_order`]).
    ///
    /// # Errors
    /// [`CchError::Ordering`] if the order cannot be produced; every case is a bug (see
    /// [`OrderError`]).
    pub(crate) fn build(skeleton: &Csr) -> Result<Self, CchError> {
        Self::from_order(skeleton, nd::nested_dissection_order(skeleton)?)
    }

    /// The elimination game for a given order, `rank_to_node[rank] = node`. Any permutation
    /// is a *correct* order: the order only decides how many fill-in arcs the game creates,
    /// that is, how big and how fast the hierarchy is. [`Self::build`] passes the
    /// nested-dissection one; the tests pin others.
    ///
    /// # Errors
    /// [`CchError::Ordering`]: [`OrderError::MalformedSkeleton`] if the skeleton is not well
    /// formed ([`Csr::is_well_formed`]), which the game would index outside of, and
    /// [`OrderError::NotAPermutation`] if `rank_to_node` is not a permutation of the
    /// skeleton's nodes. The order is checked, not trusted: a non-permutation would corrupt
    /// every later phase silently rather than loudly.
    pub(crate) fn from_order(
        skeleton: &Csr,
        rank_to_node: Vec<NodeId>,
    ) -> Result<Self, CchError> {
        if !skeleton.is_well_formed() {
            return Err(OrderError::MalformedSkeleton("malformed CCH skeleton").into());
        }
        let node_to_rank = invert(&rank_to_node, skeleton.node_count)?;
        let upward = chordal_supergraph(skeleton, &rank_to_node, &node_to_rank);
        let downward = transpose(&upward);

        Ok(Self {
            rank_to_node: rank_to_node.into_boxed_slice(),
            node_to_rank: node_to_rank.into_boxed_slice(),
            upward,
            downward,
        })
    }

    pub(crate) fn node_count(&self) -> usize {
        self.rank_to_node.len()
    }

    /// Every rank, lowest first.
    pub(crate) fn ranks(&self) -> impl DoubleEndedIterator<Item = Rank> + use<> {
        (0..self.node_count()).map(Rank::new)
    }

    /// Total arcs in the chordal supergraph — also the length of each Phase 2 weight array.
    pub(crate) fn arc_count(&self) -> usize {
        self.upward.neighbors.len()
    }

    /// The node eliminated at `rank`.
    pub(crate) fn node_of(
        &self,
        rank: Rank,
    ) -> NodeId {
        self.rank_to_node[rank.index()]
    }

    /// The rank `node` is eliminated at.
    pub(crate) fn rank_of(
        &self,
        node: NodeId,
    ) -> Rank {
        self.node_to_rank[node_index(node)]
    }

    /// The whole order as raw ids, `rank_to_node[rank] = node`: only the tests take it, for
    /// the kit's oracles, which work on `u64`s.
    #[cfg(test)]
    pub(crate) fn rank_to_node_ids(&self) -> Vec<u64> {
        self.rank_to_node
            .iter()
            .map(|&node| u64::from(node))
            .collect()
    }

    /// Every rank above `rank` in the chordal supergraph, ascending.
    pub(crate) fn up(
        &self,
        rank: Rank,
    ) -> &[Rank] {
        self.upward.row(rank)
    }

    /// The arc slots of `rank`'s upward arcs: `up(rank)[i]` is slot `up_range(rank).start + i`.
    pub(crate) fn up_range(
        &self,
        rank: Rank,
    ) -> Range<usize> {
        self.upward.slots(rank)
    }

    /// Every rank below `rank` in the chordal supergraph, ascending.
    #[cfg(test)] // scoped recustomization (#3217) is its first caller outside the tests
    pub(crate) fn down(
        &self,
        rank: Rank,
    ) -> &[Rank] {
        self.downward.row(rank)
    }

    /// The slot of the one arc joining `end` and `other_end`, in either order, or `None` if
    /// the two ranks are not adjacent in the chordal supergraph.
    ///
    /// The two ranks are the arc's ends, not the bounds of a span: no rank between them is
    /// read. Each arc is stored once, in the upward row of its lower end, so this
    /// binary-searches that row for the higher end. Phase 2 calls it once per original edge
    /// and the query once per arc it unpacks, so a search in one ascending row is cheap
    /// enough, and no hash map is kept.
    pub(crate) fn arc_slot(
        &self,
        end: Rank,
        other_end: Rank,
    ) -> Option<usize> {
        let (lower, higher) = if end < other_end {
            (end, other_end)
        } else {
            (other_end, end)
        };
        let slots = self.up_range(lower);
        self.upward.neighbors[slots.clone()]
            .binary_search(&higher)
            .ok()
            .map(|offset| slots.start + offset)
    }

    /// Bytes held by the topology: the struct itself (it lives on the heap, behind the
    /// `Arc` every version of the index shares) and its six arrays.
    #[cfg(test)]
    pub(crate) fn memory_usage(&self) -> usize {
        size_of::<Self>()
            + size_of_val(&*self.rank_to_node)
            + size_of_val(&*self.node_to_rank)
            + self.upward.memory_usage()
            + self.downward.memory_usage()
    }
}

/// `node` as an index into node-space arrays.
pub(super) fn node_index(node: NodeId) -> usize {
    id_index(u64::from(node))
}

/// The slots of `rank`'s row in a compressed row array with these `offsets`. Shared by
/// [`RankRows::slots`] and [`chordal_supergraph`], which reads its rows before they are a
/// [`RankRows`].
fn row_slots(
    offsets: &[usize],
    rank: Rank,
) -> Range<usize> {
    offsets[rank.index()]..offsets[rank.index() + 1]
}

/// `node_to_rank`, the inverse of `rank_to_node`, or an error if `rank_to_node` is not a
/// permutation of `0..node_count`.
fn invert(
    rank_to_node: &[NodeId],
    node_count: usize,
) -> Result<Vec<Rank>, OrderError> {
    if rank_to_node.len() != node_count {
        return Err(OrderError::NotAPermutation(
            "the order does not list every node exactly once",
        ));
    }
    let mut node_to_rank: Vec<Option<Rank>> = vec![None; node_count];
    for (index, &node) in rank_to_node.iter().enumerate() {
        match node_to_rank.get_mut(node_index(node)) {
            Some(slot @ None) => *slot = Some(Rank::new(index)),
            Some(_) => return Err(OrderError::NotAPermutation("the order lists a node twice")),
            None => {
                return Err(OrderError::NotAPermutation(
                    "the order lists a node the graph does not have",
                ));
            }
        }
    }
    // `node_count` nodes, each seen once: every slot is set. `Option<Rank>` and `Rank` have
    // the same layout, so this collects in place, without a second array.
    Ok(node_to_rank
        .into_iter()
        .map(|rank| rank.expect("a permutation sets every slot"))
        .collect())
}

/// The chordal supergraph as upward rows over ranks.
///
/// Gilbert-Ng-Peyton child absorption:
///
/// ```text
/// up(r) = { w in skeleton : rank(w) > r }  u  U { up(c) \ {r} : c is an elimination-tree child of r }
/// ```
///
/// Eliminating a child turns its remaining (higher-ranked) neighbors `up(c)` into a
/// clique, and `r` is by construction the lowest-ranked member of that clique, so `r`
/// becomes adjacent to all the others — which are exactly `up(c)` minus `r`. No clique is
/// ever enumerated pairwise; each rank just inherits its children's rows.
///
/// A rank's elimination-tree parent is its lowest higher neighbor in the chordal
/// supergraph: the first entry of its finished row. Ranks are built in increasing order and
/// a parent is always above its child, so every child's row is finished, and already
/// attached to its parent, before the parent's row is built. That is also why one flat
/// pair of arrays works where C needed an array-of-arrays: a child's row is final and
/// contiguous by the time its parent reads it.
fn chordal_supergraph(
    skeleton: &Csr,
    rank_to_node: &[NodeId],
    node_to_rank: &[Rank],
) -> RankRows {
    let node_count = rank_to_node.len();

    let mut offsets: Vec<usize> = Vec::with_capacity(node_count + 1);
    offsets.push(0);
    let mut neighbors: Vec<Rank> = Vec::with_capacity(skeleton.neighbors.len());
    // Each rank's elimination-tree children, as linked lists: a rank's first child, then
    // each child's next sibling, until `None`.
    let mut first_child: Vec<Option<Rank>> = vec![None; node_count];
    let mut next_sibling: Vec<Option<Rank>> = vec![None; node_count];
    let mut row: Vec<Rank> = Vec::new();

    for (index, &node) in rank_to_node.iter().enumerate() {
        let rank = Rank::new(index);
        row.clear();

        // 1. the rank's own higher neighbors in the skeleton
        row.extend(
            skeleton
                .row(node_index(node))
                .iter()
                .map(|&neighbor| node_to_rank[id_index(neighbor)])
                .filter(|&higher| higher > rank),
        );

        // 2. each child's row, minus the rank itself
        let mut next_child = first_child[index];
        while let Some(child) = next_child {
            row.extend(
                neighbors[row_slots(&offsets, child)]
                    .iter()
                    .filter(|&&higher| higher != rank),
            );
            next_child = next_sibling[child.index()];
        }

        // Ascending, so Phase 2 can binary-search an arc and Phase 3 walks in rank order;
        // a rank two children share is kept once.
        row.sort_unstable();
        row.dedup();

        if let Some(&parent) = row.first() {
            next_sibling[index] = first_child[parent.index()].replace(rank);
        }
        neighbors.extend_from_slice(&row);
        offsets.push(neighbors.len());
    }

    RankRows {
        offsets: offsets.into_boxed_slice(),
        neighbors: neighbors.into_boxed_slice(),
    }
}

/// The transpose of `upward`, the downward rows. A counting sort: filling in increasing row
/// order leaves each transposed row ascending.
fn transpose(upward: &RankRows) -> RankRows {
    let rank_count = upward.offsets.len() - 1;

    let mut offsets = vec![0; rank_count + 1];
    for &higher in &upward.neighbors {
        offsets[higher.index() + 1] += 1;
    }
    let mut running_total = 0;
    for offset in &mut offsets {
        running_total += *offset;
        *offset = running_total;
    }

    let mut neighbors = vec![Rank::new(0); upward.neighbors.len()];
    let mut next_free = offsets.clone();
    for lower in (0..rank_count).map(Rank::new) {
        for &higher in upward.row(lower) {
            neighbors[next_free[higher.index()]] = lower;
            next_free[higher.index()] += 1;
        }
    }

    RankRows {
        offsets: offsets.into_boxed_slice(),
        neighbors: neighbors.into_boxed_slice(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::{RngExt, SeedableRng};

    use super::{CchError, Csr, OrderError, Rank, Topology, invert};
    use crate::graph::graph::NodeId;
    use crate::graph::graphblas::matrix::Matrix;
    use crate::index::falkordb::cch::MAX_NODES;
    use crate::index::falkordb::cch::nd::nested_dissection_order;
    use crate::index::falkordb::cch::testkit::{
        disconnected_components, empty, entries, grid, node_count, path, random_sparse,
        single_node, star, to_input, weights, with_isolated_nodes,
    };

    /// `ids` as node ids.
    fn nodes(ids: impl IntoIterator<Item = u64>) -> Vec<NodeId> {
        ids.into_iter().map(NodeId::from).collect()
    }

    /// `indices` as ranks.
    fn ranks(indices: impl IntoIterator<Item = usize>) -> Vec<Rank> {
        indices.into_iter().map(Rank::new).collect()
    }

    /// Node-id order: `rank_to_node[rank] = rank`. It makes no attempt to reduce fill: a
    /// deterministic, hand-checkable order.
    fn identity_order(skeleton: &Csr) -> Vec<NodeId> {
        nodes(0..skeleton.node_count as u64)
    }

    /// A grid, a scatter of tiny components (pairs, 3-paths, 4-cycles, 5-paths) and
    /// `isolated` trailing isolated nodes — the shape of a `node_cap`-sized road network.
    fn many_components(
        side: u64,
        tiny: usize,
        isolated: usize,
    ) -> Matrix<f64> {
        let mut g = grid(side, side);
        for component in 0..tiny {
            let size = [2usize, 3, 4, 5][component % 4];
            let base = node_count(&g) as u64;
            g = with_isolated_nodes(g, size);
            for step in 1..size as u64 {
                g.set(base + step - 1, base + step, 1.0);
            }
            if size == 4 {
                g.set(base + 3, base, 1.0);
            }
        }
        with_isolated_nodes(g, isolated)
    }

    /// The simple undirected skeleton of `edges` over `node_count` nodes, built directly
    /// (no GraphBLAS), so thousands of small graphs stay cheap. Self-loops and repeats are
    /// dropped.
    fn skeleton_of(
        node_count: usize,
        edges: &[(usize, usize)],
    ) -> Csr {
        let mut rows: Vec<BTreeSet<u64>> = vec![BTreeSet::new(); node_count];
        for &(source, target) in edges {
            if source != target {
                rows[source].insert(target as u64);
                rows[target].insert(source as u64);
            }
        }
        let mut row_offsets = vec![0];
        let mut neighbors = Vec::new();
        for row in rows {
            neighbors.extend(row);
            row_offsets.push(neighbors.len() as u64);
        }
        Csr {
            node_count,
            row_offsets,
            neighbors,
        }
    }

    /// The elimination game played the slow, obvious way: eliminate the ranks in order and
    /// join every pair of each rank's higher neighbors with an arc. Returns every rank's
    /// higher neighbors, ascending — what [`Topology::up`] must return, rank for rank.
    fn naive_elimination_game(
        skeleton: &Csr,
        rank_to_node: &[NodeId],
    ) -> Vec<Vec<Rank>> {
        let node_count = skeleton.node_count;
        let mut rank_of_node = vec![0; node_count];
        for (rank, &node) in rank_to_node.iter().enumerate() {
            rank_of_node[usize::try_from(u64::from(node)).unwrap()] = rank;
        }
        let mut adjacent: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); node_count];
        for node in 0..node_count {
            for &neighbor in skeleton.row(node) {
                adjacent[rank_of_node[node]].insert(rank_of_node[neighbor as usize]);
            }
        }
        for rank in 0..node_count {
            let higher: Vec<usize> = adjacent[rank].range(rank + 1..).copied().collect();
            for (position, &first) in higher.iter().enumerate() {
                for &second in &higher[position + 1..] {
                    adjacent[first].insert(second);
                    adjacent[second].insert(first);
                }
            }
        }
        (0..node_count)
            .map(|rank| ranks(adjacent[rank].range(rank + 1..).copied()))
            .collect()
    }

    /// `topology` is exactly the supergraph the naive game builds for its order: every upward
    /// row, and every downward row as its transpose.
    fn assert_matches_naive_game(
        topology: &Topology,
        skeleton: &Csr,
        case: &str,
    ) {
        let expected_up = naive_elimination_game(skeleton, &topology.rank_to_node);
        let mut expected_down: Vec<Vec<Rank>> = vec![Vec::new(); expected_up.len()];
        for (lower, row) in expected_up.iter().enumerate() {
            for &higher in row {
                expected_down[higher.index()].push(Rank::new(lower));
            }
        }
        for rank in topology.ranks() {
            assert_eq!(
                topology.up(rank),
                &expected_up[rank.index()][..],
                "{case}: up({rank:?})"
            );
            assert_eq!(
                topology.down(rank),
                &expected_down[rank.index()][..],
                "{case}: down({rank:?})"
            );
        }
    }

    /// The structural properties Phase 2 and Phase 3 are entitled to assume, checked without
    /// a reference. [`assert_matches_naive_game`] pins the exact supergraph; this checks
    /// what holds of any correct one, on graphs too large for the naive game.
    fn check_invariants(
        topology: &Topology,
        skeleton: &Csr,
    ) {
        let node_count = skeleton.node_count;
        assert_eq!(topology.node_count(), node_count);
        assert_eq!(topology.node_to_rank.len(), node_count);

        // rank_to_node and node_to_rank are mutual inverses
        for node in nodes(0..node_count as u64) {
            assert_eq!(
                topology.node_of(topology.rank_of(node)),
                node,
                "node_to_rank/rank_to_node at {node:?}"
            );
        }

        for rank in topology.ranks() {
            // up rows: strictly ascending, strictly above their own rank
            let up = topology.up(rank);
            assert!(
                up.is_sorted_by(|a, b| a < b),
                "up({rank:?}) not strictly ascending"
            );
            assert!(
                up.iter().all(|&higher| higher > rank),
                "up({rank:?}) not above it"
            );
            // down rows: strictly ascending, strictly below
            let down = topology.down(rank);
            assert!(
                down.is_sorted_by(|a, b| a < b),
                "down({rank:?}) not strictly ascending"
            );
            assert!(
                down.iter().all(|&lower| lower < rank),
                "down({rank:?}) not below it"
            );
        }

        // up and down are exact inverses, in both directions
        let mut up_arcs: Vec<(Rank, Rank)> = Vec::new();
        let mut down_arcs: Vec<(Rank, Rank)> = Vec::new();
        for rank in topology.ranks() {
            up_arcs.extend(topology.up(rank).iter().map(|&higher| (rank, higher)));
            down_arcs.extend(topology.down(rank).iter().map(|&lower| (lower, rank)));
        }
        up_arcs.sort_unstable();
        down_arcs.sort_unstable();
        assert_eq!(up_arcs, down_arcs, "up and down disagree");
        assert_eq!(topology.arc_count(), up_arcs.len());

        // every skeleton edge survives as a chordal arc
        for source in 0..node_count {
            for &target in skeleton.row(source) {
                let (source_rank, target_rank) = (
                    topology.rank_of(NodeId::from(source as u64)),
                    topology.rank_of(NodeId::from(target)),
                );
                assert!(
                    topology.arc_slot(source_rank, target_rank).is_some(),
                    "skeleton edge {source}-{target} (ranks {source_rank:?}, {target_rank:?}) \
                     missing from the supergraph"
                );
            }
        }

        // the chordal property itself: eliminating `rank` makes its upper neighbors a
        // clique, so every pair in up(rank) must already be adjacent. Phase 2's triangle
        // relaxation looks up exactly these arcs and treats a miss as a logic error.
        for rank in topology.ranks() {
            let up = topology.up(rank);
            for (position, &middle) in up.iter().enumerate() {
                for &top in &up[position + 1..] {
                    assert!(
                        topology.arc_slot(middle, top).is_some(),
                        "chordal closure broken: {rank:?} has upper neighbors {middle:?} and \
                         {top:?}, but no arc joins them"
                    );
                }
            }
        }

        // arc_slot agrees with the flat layout the weight arrays are indexed by
        for lower in topology.ranks() {
            let base = topology.up_range(lower).start;
            for (offset, &higher) in topology.up(lower).iter().enumerate() {
                assert_eq!(topology.arc_slot(lower, higher), Some(base + offset));
                assert_eq!(
                    topology.arc_slot(higher, lower),
                    Some(base + offset),
                    "either order finds the same arc"
                );
            }
        }
    }

    fn build_both(edges: &Matrix<f64>) -> (Topology, Topology) {
        let input = to_input(edges);
        let dissected = Topology::build(&input.skeleton).expect("nested dissection build");
        let identity = Topology::from_order(&input.skeleton, identity_order(&input.skeleton))
            .expect("identity build");
        check_invariants(&dissected, &input.skeleton);
        check_invariants(&identity, &input.skeleton);
        (dissected, identity)
    }

    /// The kit's graph families, under nested dissection and the identity order: a grid,
    /// random sparse graphs, disconnected components, a star and a path. Each supergraph is
    /// exactly the naive game's.
    #[test]
    fn invariants_hold_on_kit_graphs() {
        let mut graphs = vec![
            grid(6, 7),
            disconnected_components(4, 12, 24, 5, 0x00C0_FFEE),
            star(9, 1.0),
            path(15),
        ];
        graphs.extend((0..8u64).map(|seed| random_sparse(60, 180, 9, seed)));
        for (case, edges) in graphs.iter().enumerate() {
            let skeleton = to_input(edges).skeleton;
            let (dissected, identity) = build_both(edges);
            assert_matches_naive_game(&dissected, &skeleton, &format!("kit {case}, dissected"));
            assert_matches_naive_game(&identity, &skeleton, &format!("kit {case}, identity"));
        }
    }

    /// The supergraph is exactly the naive elimination game's, on 2,000 small random graphs
    /// under random orders: up to 12 nodes, sparse to dense, isolated nodes included. Any
    /// missing arc, extra arc (up to a complete supergraph) or misplaced one fails.
    #[test]
    fn the_supergraph_is_the_naive_elimination_game() {
        let mut rng = SmallRng::seed_from_u64(0x0CC4);
        for case in 0..2_000 {
            let node_count = rng.random_range(0..=12usize);
            let pairs = node_count * node_count.saturating_sub(1) / 2;
            let edges: Vec<(usize, usize)> = (0..rng.random_range(0..=pairs))
                .map(|_| {
                    (
                        rng.random_range(0..node_count),
                        rng.random_range(0..node_count),
                    )
                })
                .collect();
            let skeleton = skeleton_of(node_count, &edges);
            let mut order = identity_order(&skeleton);
            order.shuffle(&mut rng);
            let topology = Topology::from_order(&skeleton, order).expect("a permutation");
            assert_matches_naive_game(&topology, &skeleton, &format!("random graph {case}"));
        }
    }

    /// Thousands of isolated nodes and tiny components around large ones — the shape of a
    /// `node_cap`-sized road network — in one backend call: a valid order, and the same one
    /// on every run. Determinism is what makes a rebuilt index equal to the original: same
    /// skeleton in, same six arrays out. The backend's seed is fixed, so this covers the
    /// backend too, not just this module's bookkeeping.
    #[test]
    fn many_components_order_validly_and_deterministically() {
        // 3 isolated nodes between the large components and the tiny ones
        let several_large = with_isolated_nodes(disconnected_components(4, 400, 1200, 9, 3), 3);
        let base = node_count(&several_large) as u64;
        let mut several_large = with_isolated_nodes(several_large, 6);
        for (source, target) in [(0, 1), (1, 2), (3, 4), (4, 5), (5, 3)] {
            several_large.set(base + source, base + target, 1.0);
        }
        for edges in [
            many_components(12, 300, 5000),
            with_isolated_nodes(grid(15, 15), 3000),
            many_components(3, 2000, 0),
            empty(4000),
            with_isolated_nodes(several_large, 50),
        ] {
            let input = to_input(&edges);
            let topology = Topology::build(&input.skeleton).expect("build");
            check_invariants(&topology, &input.skeleton);
            let again = Topology::build(&input.skeleton).expect("build again");
            let size = node_count(&edges);
            assert_eq!(
                topology.rank_to_node, again.rank_to_node,
                "not deterministic, n = {size}"
            );
            assert_eq!(topology.node_to_rank, again.node_to_rank);
            assert_eq!(topology.upward.offsets, again.upward.offsets);
            assert_eq!(topology.upward.neighbors, again.upward.neighbors);
            assert_eq!(topology.downward.offsets, again.downward.offsets);
            assert_eq!(topology.downward.neighbors, again.downward.neighbors);
        }
    }

    /// Isolated nodes take the lowest ranks, in id order, and the rest is ordered exactly
    /// as the backend orders the same graph without them.
    #[test]
    fn isolated_nodes_rank_first_and_do_not_change_the_rest() {
        let compact = grid(9, 7);
        // Interleave 40 isolated nodes: node `u` of the grid becomes `u + u / 2 + 1`.
        let lift = |node: u64| node + node / 2 + 1;
        let compact_count = node_count(&compact);
        let lifted: Vec<_> = entries(&compact)
            .into_iter()
            .map(|(source, target, weight)| (lift(source), lift(target), weight))
            .collect();
        let padded = weights(compact_count + compact_count / 2 + 1 + 3, &lifted);
        let used: Vec<u64> = (0..compact_count as u64).map(lift).collect();
        let isolated = nodes((0..node_count(&padded) as u64).filter(|node| !used.contains(node)));

        let order = nested_dissection_order(&to_input(&padded).skeleton).expect("padded");
        let rest = nested_dissection_order(&to_input(&compact).skeleton).expect("compact");
        assert_eq!(order[..isolated.len()], isolated[..]);
        let lifted = nodes(rest.into_iter().map(|node| lift(u64::from(node))));
        assert_eq!(order[isolated.len()..], lifted[..]);
    }

    /// Timing guard for padding; run with
    /// `cargo test -p graph --release -- --ignored padded_skeleton --nocapture`.
    /// A 400x400 grid, then the same grid padded by 25% isolated rows as `node_cap` growth
    /// leaves it: padding must not cost more than the grid itself.
    #[test]
    #[ignore = "timing; run in release"]
    fn padded_skeleton_orders_in_linear_time() {
        let time = |skeleton: &Csr| {
            let start = std::time::Instant::now();
            let order = nested_dissection_order(skeleton).expect("order");
            (order, start.elapsed().as_secs_f64())
        };
        let bare = to_input(&grid(400, 400));
        let padded = to_input(&with_isolated_nodes(grid(400, 400), 40_000));
        let (_, bare_seconds) = time(&bare.skeleton);
        let (_, padded_seconds) = time(&padded.skeleton);
        println!(
            "grid 400x400: {bare_seconds:.3}s; padded to {}: {padded_seconds:.3}s",
            padded.skeleton.node_count
        );
        assert!(
            padded_seconds < 2.0 * bare_seconds,
            "padding cost {padded_seconds:.3}s against {bare_seconds:.3}s unpadded"
        );
    }

    /// No nodes, one node, nodes without edges: every node is an isolated rank, no arcs.
    #[test]
    fn degenerate_skeletons_build_to_no_arcs() {
        for (case, edges) in [
            ("no nodes", empty(0)),
            ("a single node", single_node()),
            ("nodes without edges", empty(5)),
        ] {
            let input = to_input(&edges);
            let topology = Topology::build(&input.skeleton).expect(case);
            assert_eq!(topology.node_count(), node_count(&edges), "{case}");
            assert_eq!(topology.arc_count(), 0, "{case}");
            check_invariants(&topology, &input.skeleton);
        }
    }

    /// The worked example in C's `CCH_ChordalTriangulation` comment (`cch.c`): a 4-node star
    /// eliminated hub-first forces the three leaves into a fill-in triangle. Node-id order
    /// reproduces that order exactly (the hub is node 0). Reversing it (hub last) is the
    /// good order: no fill-in at all.
    #[test]
    fn hub_first_star_matches_the_c_worked_example() {
        let input = to_input(&star(3, 1.0));
        let topology = Topology::from_order(&input.skeleton, nodes(0..4)).expect("build");
        let rank = Rank::new;

        assert_eq!(
            topology.up(rank(0)),
            ranks([1, 2, 3]),
            "up(X) is its 3 original edges"
        );
        assert_eq!(
            topology.up(rank(1)),
            ranks([2, 3]),
            "up(A) absorbs up(X) minus A"
        );
        assert_eq!(
            topology.up(rank(2)),
            ranks([3]),
            "up(B) absorbs up(A) minus B"
        );
        assert_eq!(topology.up(rank(3)), ranks([]), "up(C) is the root");

        assert_eq!(topology.down(rank(0)), ranks([]));
        assert_eq!(topology.down(rank(1)), ranks([0]));
        assert_eq!(topology.down(rank(2)), ranks([0, 1]));
        assert_eq!(topology.down(rank(3)), ranks([0, 1, 2]));

        // the leaves first (`rank_to_node[rank] = rank + 1`), hub (node 0) last
        let topology = Topology::from_order(&input.skeleton, nodes([1, 2, 3, 0])).expect("build");
        check_invariants(&topology, &input.skeleton);

        // 3 leaves each hold one arc to the hub, and the hub holds none
        assert_eq!(topology.arc_count(), 3);
        assert_eq!(topology.up(rank(0)), ranks([3]));
        assert_eq!(topology.up(rank(1)), ranks([3]));
        assert_eq!(topology.up(rank(2)), ranks([3]));
        assert_eq!(topology.up(rank(3)), ranks([]));
    }

    /// Nested dissection exists to keep the supergraph small. On a 2D grid, eliminating in
    /// node-id order (row-major) is the pathological case; nested dissection must beat it,
    /// strictly, or an ordering that is node-id order would pass.
    /// The number printed is the measurement, the assertion is the guard.
    #[test]
    fn nested_dissection_beats_node_id_order_on_a_grid() {
        let (dissected, identity) = build_both(&grid(20, 20));
        println!(
            "grid 20x20: nested dissection {} arcs, identity {} arcs",
            dissected.arc_count(),
            identity.arc_count()
        );
        assert!(
            dissected.arc_count() < identity.arc_count(),
            "nested dissection {} arcs vs identity {} arcs",
            dissected.arc_count(),
            identity.arc_count()
        );
    }

    /// Past the graph's own ceiling, [`MAX_NODES`]. Refuse, do not panic — and do it before
    /// touching the (here deliberately truncated) offset array, which is what makes this
    /// testable without exabytes of RAM.
    #[test]
    fn a_graph_past_the_node_ceiling_is_refused() {
        let too_big = Csr {
            node_count: MAX_NODES + 1,
            row_offsets: vec![0],
            neighbors: Vec::new(),
        };
        assert_eq!(nested_dissection_order(&too_big), Err(OrderError::TooLarge));
        assert_eq!(
            Topology::build(&too_big).err(),
            Some(CchError::from(OrderError::TooLarge)),
            "a graph too large for the index is a bug, refused, not degraded"
        );
    }

    /// A skeleton that would send the backend outside its arrays is refused before the call,
    /// and never reaches the backend. `CchInput::build` cannot produce one, so it is a bug,
    /// never an error a user's graph could cause. Each case is a 3-node path `0 - 1 - 2`
    /// with one thing broken.
    #[test]
    fn a_malformed_skeleton_is_refused_and_not_ordered() {
        let path = |row_offsets: Vec<u64>, neighbors: Vec<u64>| Csr {
            node_count: 3,
            row_offsets,
            neighbors,
        };
        let valid = path(vec![0, 1, 3, 4], vec![1, 0, 2, 1]);
        assert!(nested_dissection_order(&valid).is_ok());

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
            assert_eq!(
                nested_dissection_order(&skeleton),
                Err(OrderError::MalformedSkeleton("malformed CCH skeleton")),
                "{case}"
            );
            assert_eq!(
                Topology::from_order(&skeleton, nodes(0..3)).err(),
                Some(CchError::from(OrderError::MalformedSkeleton(
                    "malformed CCH skeleton"
                ))),
                "{case}: the game refuses it too, rather than indexing past it"
            );
        }
    }

    /// The rest of the skeleton contract (sorted rows, no repeats, no self-loops, mirrored
    /// entries) holds by construction, and is checked in every build all the same (in
    /// release too): `Csr`'s fields are visible to the whole module, and the backend's result
    /// on anything else is undefined.
    #[test]
    fn a_skeleton_that_is_not_simple_undirected_is_refused() {
        let not_simple = OrderError::MalformedSkeleton(
            "the CCH skeleton is not symmetric, sorted, without repeats or self-loops",
        );
        let path = |row_offsets: Vec<u64>, neighbors: Vec<u64>| Csr {
            node_count: 3,
            row_offsets,
            neighbors,
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
            assert_eq!(
                nested_dissection_order(&skeleton),
                Err(not_simple.clone()),
                "{case}"
            );
            assert_eq!(
                Topology::build(&skeleton).err(),
                Some(CchError::from(not_simple.clone())),
                "{case}: a build fails rather than panics"
            );
        }
    }

    /// An order that is not a permutation is caught, not trusted.
    #[test]
    fn a_non_permutation_is_rejected() {
        let not_a_permutation = |reason| Err(OrderError::NotAPermutation(reason));
        assert_eq!(invert(&nodes([0, 1, 2]), 3), Ok(ranks([0, 1, 2])));
        assert_eq!(invert(&nodes([2, 0, 1]), 3), Ok(ranks([1, 2, 0])));
        assert_eq!(
            invert(&nodes([0, 0, 1]), 3),
            not_a_permutation("the order lists a node twice")
        );
        assert_eq!(
            invert(&nodes([0, 1, 9]), 3),
            not_a_permutation("the order lists a node the graph does not have")
        );
        assert_eq!(
            invert(&nodes([0, 1]), 3),
            not_a_permutation("the order does not list every node exactly once")
        );

        let input = to_input(&grid(2, 2));
        assert_eq!(
            Topology::from_order(&input.skeleton, nodes([0; 4])).err(),
            Some(CchError::from(OrderError::NotAPermutation(
                "the order lists a node twice"
            )))
        );
    }

    /// A rank orders and prints by its index, whatever it stores, and `Option<Rank>` costs
    /// nothing over a rank.
    #[test]
    fn ranks_order_by_index_and_none_is_free() {
        assert_eq!(size_of::<Option<Rank>>(), size_of::<usize>());
        let mut shuffled = ranks([5, 0, MAX_NODES - 1, 1, 3]);
        shuffled.sort_unstable();
        assert_eq!(shuffled, ranks([0, 1, 3, 5, MAX_NODES - 1]));
        assert!(Rank::new(0) < Rank::new(MAX_NODES - 1));
        assert_eq!(Rank::new(MAX_NODES - 1).index(), MAX_NODES - 1);
        assert_eq!(format!("{:?}", Rank::new(7)), "Rank(7)");
    }

    /// The memory count is the struct plus every array by length: on the hub-first star,
    /// 4 nodes each way, then 5 offsets and 6 arcs in each direction.
    #[test]
    fn memory_usage_counts_every_array_by_length() {
        let input = to_input(&star(3, 1.0));
        let topology = Topology::from_order(&input.skeleton, nodes(0..4)).expect("build");
        let word = size_of::<usize>();
        assert_eq!(
            topology.memory_usage(),
            size_of::<Topology>() + 2 * 4 * word + 2 * (5 + 6) * word
        );
    }
}
