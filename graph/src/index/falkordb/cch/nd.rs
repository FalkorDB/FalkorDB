//! The nested-dissection call behind Phase 1's elimination order
//! ([`nested_dissection_order`]).
//!
//! The backend is `feral-metis`, a pure-Rust multilevel nested dissection with AMD leaves,
//! from FalkorDB's fork of feral (<https://github.com/FalkorDB/feral>, branch
//! `falkordb/i64`) with 64-bit indices (its `FALKORDB.md` lists the upstream versions and
//! every change). CCH makes one call, `metis_order`, with its default options.
//! Two things about that call shape this file:
//!
//! * **The skeleton is copied once**, into `i64` arrays: the backend reads signed indices,
//!   where the skeleton's ids are `u64`. The copy also leaves out the isolated nodes and
//!   renumbers the rest ([`order_connected_nodes`]), and the skeleton is checked before it,
//!   since the backend indexes its arrays with these values.
//! * **Nothing is shared between calls.** The backend's generator is a SplitMix64 owned by
//!   the call and seeded with a fixed seed, and it touches no global state, no signal
//!   handler and no C library: the permutation depends on the skeleton alone, so calls run
//!   concurrently and the order is the same on every run and every platform.

use super::order::{OrderError, node_index};
use super::{Csr, MAX_NODES, id_index, index_id};
use crate::graph::graph::NodeId;
use feral_metis::{CscPattern, OrderingError};

/// Nested dissection with the backend's default options: `rank_to_node[rank] = node`, with
/// the isolated nodes first, in id order. Deterministic: see the module docs.
///
/// The skeleton must be the adjacency of a simple undirected graph
/// ([`Csr::is_symmetric_simple`]). `CchInput::build` produces it by construction: the
/// pattern union of the weights and their transpose in GraphBLAS, diagonal dropped, one
/// entry per pair, exported sorted. No graph data can make it otherwise, so a skeleton that
/// fails is a bug in the engine, not a property of the user's graph. The check runs in
/// release builds too, because the backend's result on anything else is undefined; it
/// allocates nothing and costs a binary search per entry, small next to the ordering.
///
/// Isolated nodes (empty rows) are not sent to the backend. They create no fill wherever
/// they are eliminated, so their ranks are free, and a skeleton sized by `node_cap` carries
/// every unused id slot as one: 294k of California's 1.54M nodes, which the backend would
/// otherwise copy into each of its own graph arrays.
///
/// # Errors
/// [`OrderError::TooLarge`] past [`MAX_NODES`], [`OrderError::MalformedSkeleton`] for a
/// skeleton that is not well formed ([`Csr::is_well_formed`]) or not symmetric and simple,
/// [`OrderError::Backend`] if the backend refuses the skeleton or fails, and
/// [`OrderError::NotAPermutation`] if it returns an order of the wrong length or an id out
/// of range (a repeated id is caught by [`Topology::from_order`](super::order::Topology::from_order)).
pub(super) fn nested_dissection_order(skeleton: &Csr) -> Result<Vec<NodeId>, OrderError> {
    // Past the graph's own ceiling no id can be numbered.
    if skeleton.node_count > MAX_NODES {
        return Err(OrderError::TooLarge);
    }
    if !skeleton.is_well_formed() {
        return Err(OrderError::MalformedSkeleton("malformed CCH skeleton"));
    }
    if !skeleton.is_symmetric_simple() {
        return Err(OrderError::MalformedSkeleton(
            "the CCH skeleton is not symmetric, sorted, without repeats or self-loops",
        ));
    }

    // One backend call for every connected node, all components together: the backend
    // orders a disconnected graph itself. The isolated nodes take the lowest ranks.
    let (mut rank_to_node, connected): (Vec<NodeId>, Vec<NodeId>) = (0..skeleton.node_count)
        .map(|node| NodeId::from(index_id(node)))
        .partition(|&node| skeleton.row(node_index(node)).is_empty());
    if connected.is_empty() {
        return Ok(rank_to_node);
    }
    let backend_order = order_connected_nodes(skeleton, &connected)?;
    if backend_order.len() != connected.len() {
        return Err(OrderError::NotAPermutation(
            "nested dissection returned a permutation of the wrong length",
        ));
    }
    rank_to_node.reserve(connected.len());
    for backend_id in backend_order {
        let node = usize::try_from(backend_id)
            .ok()
            .and_then(|index| connected.get(index))
            .ok_or(OrderError::NotAPermutation(
                "nested dissection returned a node id out of range",
            ))?;
        rank_to_node.push(*node);
    }
    Ok(rank_to_node)
}

/// Copy the skeleton's `connected` nodes into the backend's arrays, renumbered `0..m` in
/// the order given, and order them: the result lists backend ids, `0..m`. The copy (METIS's
/// `xadj` and `adjncy`) costs `8 (m + 1) + 8 nnz` bytes, plus `8 n` for the renumbering,
/// and is freed when this returns, before the caller builds the full order.
fn order_connected_nodes(
    skeleton: &Csr,
    connected: &[NodeId],
) -> Result<Vec<i64>, OrderError> {
    // `backend_id[node]` is `node`'s backend id; isolated nodes are never read.
    let mut backend_id = vec![0i64; skeleton.node_count];
    for (index, &node) in connected.iter().enumerate() {
        backend_id[node_index(node)] = backend_index(index);
    }

    let mut row_offsets = Vec::with_capacity(connected.len() + 1);
    let mut neighbors = Vec::with_capacity(skeleton.neighbors.len());
    row_offsets.push(0);
    for &node in connected {
        // A neighbor is connected too: its row holds at least `node`.
        neighbors.extend(
            skeleton
                .row(node_index(node))
                .iter()
                .map(|&neighbor| backend_id[id_index(neighbor)]),
        );
        row_offsets.push(backend_index(neighbors.len()));
    }

    // The backend checks offsets, ids and row order again. Linear in the skeleton.
    let pattern = CscPattern::new(connected.len(), &row_offsets, &neighbors)
        .ok_or(OrderingError::MalformedInput)?;
    Ok(feral_metis::metis_order(&pattern)?)
}

/// An index or count as the backend's `i64`.
///
/// # Panics
/// Never: every index here is below [`MAX_NODES`](super::MAX_NODES) `< 2^60`, and every
/// count of skeleton entries is below `isize::MAX`, the most a `Vec` holds.
fn backend_index(index: usize) -> i64 {
    i64::try_from(index).expect("an index fits the backend's i64")
}

#[cfg(test)]
mod tests {
    use super::nested_dissection_order;
    use crate::graph::graph::NodeId;
    use crate::index::falkordb::cch::testkit::{grid, random_sparse, to_input};

    /// Calls share nothing: eight threads ordering the same skeletons at once each get the
    /// order a lone call gives.
    #[test]
    fn concurrent_calls_order_alike() {
        for skeleton in [
            to_input(&grid(40, 40)).skeleton,
            to_input(&random_sparse(2_000, 6_000, 9, 11)).skeleton,
        ] {
            let alone = nested_dissection_order(&skeleton).expect("order");
            let together: Vec<Vec<NodeId>> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..8)
                    .map(|_| scope.spawn(|| nested_dissection_order(&skeleton).expect("order")))
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().expect("thread"))
                    .collect()
            });
            for (thread, order) in together.iter().enumerate() {
                assert_eq!(*order, alone, "thread {thread} ordered differently");
            }
        }
    }
}
