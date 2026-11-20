//! The algorithms from LAGraph's `experimental` folder (`LAGraphX.h`) that
//! the engine runs. LAGraph makes no stability promise for them; the
//! contract written on each function here is the one its C source has at
//! the pinned revision.
//!
//! Where the C function takes a null out-pointer to mean "do not compute
//! this output", the function here takes a selector and returns `None` in
//! that slot.

use graphblas::{BuiltinType, ErrorKind, GrB_Index, GrbType, Matrix, Vector};

use crate::{Error, Graph, Result, error::call, init::ensure_initialized, output::Out, sys};

/// The outputs of [`breadth_first_search_extended`] to compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Traversal {
    /// The level of every reached node: its distance from the source.
    Levels,
    /// The parent of every reached node in the search tree.
    Parents,
    Both,
}

impl Traversal {
    const fn levels(self) -> bool {
        matches!(self, Self::Levels | Self::Both)
    }

    const fn parents(self) -> bool {
        matches!(self, Self::Parents | Self::Both)
    }
}

/// What [`breadth_first_search_extended`] found. A node the search did not
/// reach has no entry in either vector.
#[derive(Debug)]
pub struct BreadthFirstSearch {
    /// `level(source) = 0`; `None` unless asked for.
    pub level: Option<Vector<u64>>,
    /// `parent(source) = source`; `None` unless asked for.
    pub parent: Option<Vector<u64>>,
}

/// An index or a count the C API takes as an `int64_t` with a negative
/// value meaning "none".
fn optional_i64(
    value: Option<u64>,
    what: &str,
) -> Result<i64> {
    value.map_or(Ok(-1), |value| {
        i64::try_from(value).map_err(|_| {
            Error::graphblas(ErrorKind::InvalidValue, &format!("{what} over i64::MAX"))
        })
    })
}

/// Breadth-first search from `source` (`LAGr_BreadthFirstSearch_Extended`),
/// stopping after `max_level` levels and as soon as `destination` is
/// reached, when either is given.
///
/// `many_expected` says the search will reach a fair share of the graph, so
/// the outputs are held as bitmap or full vectors; pass `false` for a
/// search expected to stay small.
///
/// Uses the transpose and the out-degrees for its push/pull heuristic when
/// they are cached, and is push-only when they are not; it never fails for
/// a missing one.
///
/// # Errors
///
/// `GrB_INVALID_INDEX` when `source` is not a node; out of memory.
pub fn breadth_first_search_extended<T: GrbType>(
    outputs: Traversal,
    graph: &Graph<T>,
    source: GrB_Index,
    max_level: Option<u64>,
    destination: Option<GrB_Index>,
    many_expected: bool,
) -> Result<BreadthFirstSearch> {
    let max_level = optional_i64(max_level, "max_level")?;
    let destination = optional_i64(destination, "destination")?;
    let mut level = Out::new();
    let mut parent = Out::new();
    // SAFETY: each out-pointer is valid or null, null being the C API's
    // "not wanted"; a live graph, which the function only reads.
    call(|msg| unsafe {
        sys::LAGr_BreadthFirstSearch_Extended(
            level.pointer_if(outputs.levels()),
            parent.pointer_if(outputs.parents()),
            graph.as_raw(),
            source,
            max_level,
            destination,
            many_expected,
            msg,
        )
    })?;
    Ok(BreadthFirstSearch {
        level: level.adopt_node_ids_if(outputs.levels())?,
        parent: parent.adopt_node_ids_if(outputs.parents())?,
    })
}

mod sealed {
    pub trait Sealed {}
}

/// The element types [`harmonic_centrality`] accepts for node weights:
/// `bool` (every listed node counts once) and `i64`. Sealed.
pub trait NodeWeight: BuiltinType + sealed::Sealed {}
impl sealed::Sealed for bool {}
impl sealed::Sealed for i64 {}
impl NodeWeight for bool {}
impl NodeWeight for i64 {}

/// What [`harmonic_centrality`] found, with an entry for every node that
/// took part.
#[derive(Debug)]
pub struct HarmonicCentrality {
    pub scores: Vector<f64>,
    /// The estimated number of nodes each one reaches; `None` unless asked
    /// for.
    pub reachable: Option<Vector<u64>>,
}

/// An estimate of the harmonic centrality of every node listed in
/// `node_weights` (`LAGr_HarmonicCentrality`).
///
/// The score of a node `u` is the sum over the other listed nodes `v` of
/// `weight(v) / distance(u, v)`, with reachability counted by HyperLogLog
/// sketches, so the scores are approximate.
///
/// `node_weights` has one position per node. The nodes that take part are
/// the ones with an entry; the search runs on the sub-graph they induce.
/// With `reachable = true` the estimated number of nodes each one reaches
/// is returned as well.
///
/// Reads only the adjacency matrix; no cached property is used. Built-in
/// element types only: LAGraph casts the adjacency to `bool`, which a
/// user-defined type cannot be.
///
/// # Errors
///
/// `GrB_INVALID_VALUE` for a negative weight; `GrB_NOT_IMPLEMENTED` for a
/// weight over 1,000,000; `GrB_DIMENSION_MISMATCH` when `node_weights` is
/// not one position per node; out of memory.
pub fn harmonic_centrality<T: BuiltinType, W: NodeWeight>(
    reachable: bool,
    graph: &Graph<T>,
    node_weights: &Vector<W>,
) -> Result<HarmonicCentrality> {
    // LAGraph sizes the scores to `node_weights` and never compares it with
    // the graph, so a vector of another size would be accepted.
    let nodes = graph.adjacency().nrows();
    if node_weights.size() != nodes {
        return Err(Error::graphblas(
            ErrorKind::DimensionMismatch,
            "node_weights needs one position per node",
        ));
    }
    let mut scores = Out::new();
    let mut reachable_nodes = Out::new();
    // SAFETY: a valid out-pointer and one that is valid or null; a live
    // graph and a live vector of a type the function accepts, both only
    // read.
    call(|msg| unsafe {
        sys::LAGr_HarmonicCentrality(
            scores.pointer(),
            reachable_nodes.pointer_if(reachable),
            graph.as_raw(),
            node_weights.as_raw(),
            msg,
        )
    })?;
    // With no node taking part LAGraph returns before it allocates the
    // second output: nobody reaches anything.
    let reachable = if reachable && reachable_nodes.is_null() {
        Some(Vector::new(nodes)?)
    } else {
        reachable_nodes.adopt_if(reachable)?
    };
    Ok(HarmonicCentrality {
        scores: scores.adopt()?,
        reachable,
    })
}

/// Community detection by label propagation (`LAGraph_cdlp`).
///
/// Every node starts with its own id as its label and repeatedly takes the
/// most frequent label among its neighbours, the smallest on a tie, for at
/// most `max_iterations` rounds. The result is each node's label.
///
/// Edge values are ignored. On a directed graph a neighbour counts once
/// per direction.
///
/// # Errors
///
/// Out of memory. At the LAGraph pin, running out of memory in the first
/// allocations frees an uninitialised handle in LAGraph's own cleanup, so
/// that failure is undefined behaviour, not an error; to be fixed in
/// LAGraph.
pub fn community_detection_label_propagation<T: GrbType>(
    graph: &Graph<T>,
    max_iterations: i32,
) -> Result<Vector<u64>> {
    let mut labels = Out::new();
    // SAFETY: a valid out-pointer and a live graph. The C parameter is not
    // `const`, but the function only copies `G->A` and reads `G->kind`.
    call(|msg| unsafe {
        sys::LAGraph_cdlp(labels.pointer(), graph.as_raw(), max_iterations, msg)
    })?;
    labels.adopt()
}

/// The element types [`minimum_spanning_forest`] accepts as edge weights.
///
/// Each comes with the type LAGraph computes in and returns the forest in:
/// `i64` for `bool` and every integer type, `f64` for `f32` and `f64`.
/// Sealed.
pub trait EdgeWeight: BuiltinType + sealed::Sealed {
    /// The element type of the forest matrix.
    type Forest: BuiltinType;
}

macro_rules! edge_weight {
    ($forest:ty: $($weight:ty),+) => {$(
        impl EdgeWeight for $weight {
            type Forest = $forest;
        }
    )+};
}
impl sealed::Sealed for i8 {}
impl sealed::Sealed for i16 {}
impl sealed::Sealed for i32 {}
impl sealed::Sealed for u8 {}
impl sealed::Sealed for u16 {}
impl sealed::Sealed for u32 {}
impl sealed::Sealed for u64 {}
impl sealed::Sealed for f32 {}
impl sealed::Sealed for f64 {}
edge_weight!(i64: bool, i8, i16, i32, i64, u8, u16, u32, u64);
edge_weight!(f64: f32, f64);

/// What [`minimum_spanning_forest`] found.
#[derive(Debug)]
pub struct SpanningForest<F: BuiltinType> {
    /// The forest's edges, each stored once, with its weight.
    pub edges: Matrix<F>,
    /// The representative node of every node's tree; `None` unless asked
    /// for.
    pub component: Option<Vector<u64>>,
}

/// A minimum spanning forest of the undirected graph whose weights are
/// `a` (`LAGraph_msf`), and with `component_ids = true` the tree every node
/// is in.
///
/// Takes the matrix, not a [`Graph`]: LAGraph's function does. `a` must be
/// symmetric. `sanitize = true` makes it so first, as `min(A, Aᵀ)` over the
/// union of the two patterns; pass `false` only for a matrix already known
/// to be symmetric, where it saves that pass. LAGraph does not check: on a
/// matrix that is not symmetric, with `sanitize = false`, the call can loop
/// forever (three nodes with edges 0→1, 1→2, 2→0 do it).
///
/// A `u64` weight over `i64::MAX` wraps when LAGraph converts it.
///
/// # Errors
///
/// `GrB_DIMENSION_MISMATCH` when `a` is not square; out of memory.
pub fn minimum_spanning_forest<W: EdgeWeight>(
    component_ids: bool,
    a: &Matrix<W>,
    sanitize: bool,
) -> Result<SpanningForest<W::Forest>> {
    // The one algorithm that takes no `Graph`, so the one that has to ask.
    ensure_initialized()?;
    let mut forest = Out::new();
    let mut components = Out::new();
    // SAFETY: a valid out-pointer and one that is valid or null; a live
    // matrix of a built-in type, which the function copies and does not
    // modify.
    call(|msg| unsafe {
        sys::LAGraph_msf(
            forest.pointer(),
            components.pointer_if(component_ids),
            a.as_raw(),
            sanitize,
            msg,
        )
    })?;
    Ok(SpanningForest {
        edges: forest.adopt()?,
        component: components.adopt_if(component_ids)?,
    })
}

/// The outputs of [`max_flow`] to compute besides the flow's value.
///
/// Ask for one or the other. At the pinned LAGraph revision asking for both
/// fails with `GrB_NULL_POINTER`: building the flow matrix frees the
/// operator the residual is built with (`LAGr_MaxFlow.c`,
/// `LG_FREE_WORK_EXCEPT_R`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlowOutputs {
    /// The flow on every edge.
    pub flow: bool,
    /// The residual network.
    pub residual: bool,
}

/// What [`max_flow`] found.
#[derive(Debug)]
pub struct MaxFlow {
    /// The largest flow from the source to the sink.
    pub value: f64,
    /// The flow on every edge; `None` unless asked for.
    pub flow: Option<Matrix<f64>>,
    /// The residual network; `None` unless asked for.
    pub residual: Option<Matrix<f64>>,
}

/// The maximum flow from `source` to `sink`, where the adjacency matrix
/// holds edge capacities (`LAGr_MaxFlow`, push-relabel).
///
/// Needs the smallest weight cached, and on a directed graph the
/// transpose: [`Graph::cache_min_weight`], [`Graph::cache_transpose`].
/// Capacities must be positive. LAGraph does not verify that (it tests the
/// cached scalar's handle, not its value), so neither does this function.
/// Built-in element types only: LAGraph reads the capacities as `f64`,
/// which a user-defined type cannot be cast to.
///
/// # Errors
///
/// `GrB_UNINITIALIZED_OBJECT` when the smallest weight is not cached;
/// [`NotCached`](crate::Status::NotCached) for the transpose;
/// `GrB_INVALID_VALUE` when `source` or `sink` is not a node;
/// [`ConvergenceFailure`](crate::Status::ConvergenceFailure) past LAGraph's
/// iteration limit; out of memory.
/// [`InvalidGraph`](crate::Status::InvalidGraph) on a one-node graph:
/// LAGraph searches a matrix of its own there, which GraphBLAS holds by
/// column, and this crate cannot pin it.
pub fn max_flow<T: BuiltinType>(
    outputs: FlowOutputs,
    graph: &Graph<T>,
    source: GrB_Index,
    sink: GrB_Index,
) -> Result<MaxFlow> {
    let mut value = 0.0;
    let mut flow = Out::new();
    let mut residual = Out::new();
    // SAFETY: a valid out-pointer for the value and two that are valid or
    // null; a live graph. The C parameter is not `const`, but the function
    // only reads the graph.
    call(|msg| unsafe {
        sys::LAGr_MaxFlow(
            &raw mut value,
            flow.pointer_if(outputs.flow),
            residual.pointer_if(outputs.residual),
            graph.as_raw(),
            source,
            sink,
            msg,
        )
    })?;
    Ok(MaxFlow {
        value,
        flow: flow.adopt_if(outputs.flow)?,
        residual: residual.adopt_if(outputs.residual)?,
    })
}
