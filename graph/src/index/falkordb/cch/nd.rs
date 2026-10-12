//! The nested-dissection call behind [`NestedDissection`](super::order::NestedDissection).
//!
//! The backend is `feral-metis`, a pure-Rust multilevel nested dissection with AMD leaves,
//! vendored under `vendor/feral` with 64-bit indices (its README lists the upstream
//! versions and every change). CCH makes one call, `metis_order`, with its default options.
//! Two things about that call shape this file:
//!
//! * **The skeleton is copied once**, into `i64` arrays: the backend reads signed indices,
//!   where the skeleton's ids are `u64`. The copy also leaves out the isolated nodes and
//!   renumbers the rest ([`nd_graph`]), and it is where every value the backend will index
//!   with is bounds-checked.
//! * **Nothing is shared between calls.** The backend's generator is a SplitMix64 owned by
//!   the call and seeded with a fixed seed, and it touches no global state, no signal
//!   handler and no C library: the permutation depends on the skeleton alone, so calls run
//!   concurrently and the order is the same on every run and every platform.

use super::Csr;
use super::order::{NONE, OrderError};
use feral_metis::{CscPattern, OrderingError};

/// The skeleton as the backend reads it, `xadj` and `adjncy`, without its isolated nodes.
/// Transient: built just before the call, freed just after.
pub(super) struct NdGraph {
    xadj: Vec<i64>,
    adjncy: Vec<i64>,
    /// `nodes[k]` is the skeleton node the backend knows as `k`, ascending.
    nodes: Vec<u64>,
    /// The skeleton nodes with no neighbour, ascending. They take the lowest ranks.
    isolated: Vec<u64>,
}

/// Copy `skeleton` into the backend's arrays.
///
/// The skeleton must be the adjacency of a simple undirected graph
/// ([`Csr::is_symmetric_simple`]): what `CchInput::build` produces by construction, and
/// checked here in every build, since `Csr`'s fields are public and the backend's result is
/// undefined on anything else. The check allocates nothing and costs a binary search per
/// entry, small next to the ordering.
///
/// Isolated nodes (empty rows) are left out and the rest renumbered in id order. They
/// create no fill wherever they are eliminated, so their ranks are free, and a skeleton
/// sized by `node_cap` carries every unused id slot as one: 294k of California's 1.54M
/// nodes, which the backend would otherwise copy into each of its own graph arrays.
///
/// The copy costs `8 (m + 1) + 8 nnz` bytes for the `m` nodes the backend sees, plus `8 n`
/// for the renumbering, while the backend runs; it is freed before the permutation is
/// returned.
pub(super) fn nd_graph(skeleton: &Csr) -> Result<NdGraph, OrderError> {
    if !skeleton.is_well_formed() {
        return Err(OrderError::Backend("malformed skeleton"));
    }
    if !skeleton.is_symmetric_simple() {
        return Err(OrderError::Backend(
            "the skeleton is not symmetric, sorted, without repeats or self-loops",
        ));
    }
    let n = skeleton.n;

    // `renumber[u]` is `u`'s backend id, or `NONE` for an isolated node.
    let mut renumber = vec![NONE; n];
    let mut nodes = Vec::new();
    let mut isolated = Vec::new();
    for (u, id) in renumber.iter_mut().enumerate() {
        if skeleton.row(u).is_empty() {
            isolated.push(u as u64);
        } else {
            *id = nodes.len() as u64;
            nodes.push(u as u64);
        }
    }

    // CSR row pointers: `m + 1` of them, starting at 0.
    let mut xadj = Vec::with_capacity(nodes.len() + 1);
    let mut adjncy = Vec::with_capacity(skeleton.i.len());
    xadj.push(0);
    for &u in &nodes {
        for &v in skeleton.row(u as usize) {
            // Never `NONE`: a neighbour's row holds at least `u` (the skeleton is
            // symmetric). Below `n <= MAX_NODES < 2^60`, so it fits an `i64`.
            adjncy.push(renumber[v as usize] as i64);
        }
        xadj.push(adjncy.len() as i64);
    }
    Ok(NdGraph {
        xadj,
        adjncy,
        nodes,
        isolated,
    })
}

/// Nested dissection with the backend's default options: `perm[rank] = node`, the same
/// "new-to-old" convention as [`Ordering`](super::order::Ordering), with the isolated
/// nodes first. Deterministic: see the module docs.
pub(super) fn node_nd(graph: NdGraph) -> Result<Vec<u64>, OrderError> {
    let NdGraph {
        xadj,
        adjncy,
        nodes,
        isolated: mut perm_out,
    } = graph;
    let n = nodes.len();
    if n == 0 {
        return Ok(perm_out);
    }
    let perm = {
        // Offsets, ids and sorted rows again (`nd_graph` checked them). Linear in the
        // skeleton.
        let pattern = CscPattern::new(n, &xadj, &adjncy)
            .ok_or(OrderError::Backend("the backend rejected the skeleton"))?;
        feral_metis::metis_order(&pattern).map_err(|e| match e {
            OrderingError::IndexOverflow => OrderError::TooLarge,
            _ => OrderError::Backend("nested dissection failed"),
        })?
    };
    drop((xadj, adjncy));
    if perm.len() != n {
        return Err(OrderError::Backend(
            "nested dissection returned a permutation of the wrong length",
        ));
    }
    perm_out.reserve(n);
    for v in perm {
        let node =
            usize::try_from(v)
                .ok()
                .and_then(|v| nodes.get(v))
                .ok_or(OrderError::Backend(
                    "nested dissection returned a node id out of range",
                ))?;
        perm_out.push(*node);
    }
    Ok(perm_out)
}

#[cfg(test)]
mod tests {
    use crate::index::falkordb::cch::order::{NestedDissection, Ordering};
    use crate::index::falkordb::cch::testkit::{grid, random_sparse, to_input};

    /// Calls share nothing: eight threads ordering the same skeletons at once each get the
    /// order a lone call gives.
    #[test]
    fn concurrent_calls_order_alike() {
        for skeleton in [
            to_input(&grid(40, 40)).skeleton,
            to_input(&random_sparse(2_000, 6_000, 9, 11)).skeleton,
        ] {
            let alone = NestedDissection.order(&skeleton).expect("order");
            let together: Vec<Vec<u64>> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..8)
                    .map(|_| s.spawn(|| NestedDissection.order(&skeleton).expect("order")))
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("thread"))
                    .collect()
            });
            for (k, order) in together.iter().enumerate() {
                assert_eq!(*order, alone, "thread {k} ordered differently");
            }
        }
    }
}
