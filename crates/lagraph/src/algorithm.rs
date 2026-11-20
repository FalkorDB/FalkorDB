//! The algorithms, as free functions: inputs in the C API's order, and the
//! objects LAGraph allocates for its outputs returned.
//!
//! These are LAGraph's "advanced" entry points. They read the cached
//! properties a graph already has and fail with
//! [`NotCached`](crate::Status::NotCached) for one it lacks; none computes
//! a property on its own.

use std::os::raw::c_int;

use graphblas::{ErrorKind, GrB_Index, GrbType, Vector};

use crate::{Error, Graph, Result, error::call, output::Out, sys};

/// PageRank (`LAGr_PageRank`): the rank of every node, and the number of
/// iterations taken. Sinks (nodes with no out-edge) spread their rank over
/// all nodes, so the ranks sum to 1.
///
/// Needs the out-degrees, and on a directed graph the transpose:
/// [`Graph::cache_out_degree`], [`Graph::cache_transpose`].
///
/// # Errors
///
/// [`NotCached`](crate::Status::NotCached);
/// [`ConvergenceFailure`](crate::Status::ConvergenceFailure) when
/// `max_iterations` is reached before the change drops below `tolerance`;
/// out of memory.
pub fn page_rank<T: GrbType>(
    graph: &Graph<T>,
    damping: f32,
    tolerance: f32,
    max_iterations: i32,
) -> Result<(Vector<f32>, i32)> {
    let mut centrality = Out::new();
    let mut iterations = 0;
    // SAFETY: valid out-pointers and a live graph, which the function only
    // reads.
    call(|msg| unsafe {
        sys::LAGr_PageRank(
            centrality.pointer(),
            &raw mut iterations,
            graph.as_raw(),
            damping,
            tolerance,
            max_iterations,
            msg,
        )
    })?;
    Ok((centrality.adopt()?, iterations))
}

/// Betweenness centrality (`LAGr_Betweenness`), computed from the shortest
/// paths that start at `sources`: exact when `sources` is every node, an
/// estimate from a sample otherwise.
///
/// Needs the transpose on a directed graph: [`Graph::cache_transpose`].
///
/// # Errors
///
/// [`NotCached`](crate::Status::NotCached); `GrB_INVALID_INDEX` for a
/// source that is not a node; `GrB_INVALID_VALUE` for no sources on a graph
/// that has nodes (LAGraph would return 1.0 for every node, which is not a
/// centrality) or more than `i32::MAX` of them; out of memory.
pub fn betweenness<T: GrbType>(
    graph: &Graph<T>,
    sources: &[GrB_Index],
) -> Result<Vector<f64>> {
    // A graph with no nodes has no sources to give, and an empty result.
    if sources.is_empty() && graph.adjacency().nrows() > 0 {
        return Err(Error::graphblas(
            ErrorKind::InvalidValue,
            "betweenness needs at least one source node",
        ));
    }
    let count = i32::try_from(sources.len())
        .map_err(|_| Error::graphblas(ErrorKind::InvalidValue, "more than i32::MAX sources"))?;
    let mut centrality = Out::new();
    // SAFETY: a valid out-pointer, a live graph the function only reads,
    // and `count` indices that outlive the call.
    call(|msg| unsafe {
        sys::LAGr_Betweenness(
            centrality.pointer(),
            graph.as_raw(),
            sources.as_ptr(),
            count,
            msg,
        )
    })?;
    centrality.adopt()
}

/// Connected components (`LAGr_ConnectedComponents`): for every node, the
/// id of a representative node of its component.
///
/// The graph must be undirected, or directed and known to be symmetric
/// ([`Graph::cache_symmetric_structure`]).
///
/// Takes the graph by `&mut` because LAGraph does: on a large graph it
/// moves the adjacency matrix's arrays out while it samples them, which
/// leaves a 0-by-0 matrix, and moves them back before returning. It also
/// pins the matrix to the sparse format for that; this function puts the
/// format setting back.
///
/// # Errors
///
/// [`SymmetricStructureRequired`](crate::Status::SymmetricStructureRequired);
/// out of memory. A failure while the arrays are moved out leaves the
/// adjacency matrix 0-by-0, and LAGraph leaks the arrays.
pub fn connected_components<T: GrbType>(graph: &mut Graph<T>) -> Result<Vector<u64>> {
    let adjacency = graph.adjacency().as_raw();
    let field = sys::GxB_Option_Field::GxB_SPARSITY_CONTROL as c_int;
    let mut sparsity_control = 0;
    // SAFETY: a live matrix, a valid out-pointer and an integer field.
    let info = unsafe { sys::GrB_Matrix_get_INT32(adjacency, &raw mut sparsity_control, field) };
    if info != sys::GrB_Info::GrB_SUCCESS {
        return Err(Error::graphblas(
            ErrorKind::from_code(info as i32),
            "reading the sparsity control",
        ));
    }

    let mut component = Out::new();
    // SAFETY: a valid out-pointer and a live graph, held exclusively for
    // the call.
    let outcome = call(|msg| unsafe {
        sys::LAGr_ConnectedComponents(component.pointer(), graph.as_raw(), msg)
    });
    // SAFETY: the same matrix and field. After a failed call the matrix may
    // be the 0-by-0 one, where this has nothing to restore, so its status is
    // only reported when the call itself succeeded.
    let restored = unsafe { sys::GrB_Matrix_set_INT32(adjacency, sparsity_control, field) };
    outcome?;
    if restored != sys::GrB_Info::GrB_SUCCESS {
        return Err(Error::graphblas(
            ErrorKind::from_code(restored as i32),
            "restoring the sparsity control",
        ));
    }
    component.adopt_node_ids()
}
