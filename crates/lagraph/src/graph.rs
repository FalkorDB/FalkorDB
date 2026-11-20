//! [`Graph<T>`]: an adjacency matrix together with what LAGraph caches
//! about it.

use std::{
    fmt,
    mem::ManuallyDrop,
    ptr::{self, null_mut},
};

use graphblas::{GrbType, Matrix, Orientation};

use crate::{Result, error::call, init::ensure_initialized, sys};

/// How LAGraph reads the adjacency matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `A(i, j)` is the edge between `i` and `j`, and both `A(i, j)` and
    /// `A(j, i)` are stored. LAGraph does not check this; on an unsymmetric
    /// matrix its results are undefined.
    Undirected,
    /// `A(i, j)` is the edge from `i` to `j`.
    Directed,
}

impl Kind {
    const fn raw(self) -> sys::LAGraph_Kind {
        match self {
            Self::Undirected => sys::LAGraph_Kind::LAGraph_ADJACENCY_UNDIRECTED,
            Self::Directed => sys::LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED,
        }
    }
}

/// An owned `LAGraph_Graph` over a `Matrix<T>`.
///
/// The matrix is moved in and stays this graph's until
/// [`into_matrix`](Self::into_matrix). There is no way to lend LAGraph a
/// matrix that something else can still read: an algorithm is free to
/// modify `G->A` while it runs and restore it at the end, and
/// [`connected_components`](crate::connected_components) does. There is no
/// way to change the matrix in place either, because that would silently
/// invalidate the cached properties; take it out, change it, and build a
/// new graph.
///
/// Cached properties are computed on request by the `cache_*` methods and
/// freed with the graph.
pub struct Graph<T> {
    raw: sys::LAGraph_Graph,
    /// The matrix `raw->A` points at. Owned here, not by LAGraph: `raw->A`
    /// is detached before `LAGraph_Delete`. No wrapped function replaces
    /// the handle in `raw->A`, so the two never diverge.
    adjacency: Matrix<T>,
    kind: Kind,
}

// SAFETY: a graph is its matrix (`Send`) and a heap object that holds
// GraphBLAS handles, none tied to a thread. Not `Sync`, like the matrix.
unsafe impl<T: Send> Send for Graph<T> {}

impl<T: GrbType> Graph<T> {
    /// A graph over `adjacency`, which must be square.
    ///
    /// The matrix is pinned to row-major storage first, and the new graph
    /// is run through `LAGraph_CheckGraph`, so a matrix LAGraph cannot use
    /// is refused here and not by the first algorithm.
    ///
    /// # Errors
    ///
    /// [`InvalidGraph`](crate::Status::InvalidGraph) when the matrix is not
    /// square; out of memory; LAGraph not initialised. The matrix is
    /// dropped.
    pub fn new(
        mut adjacency: Matrix<T>,
        kind: Kind,
    ) -> Result<Self> {
        ensure_initialized()?;
        adjacency.set_orientation(Orientation::ByRow)?;

        let mut raw = null_mut();
        // `LAGraph_New` moves the handle into `G->A` and nulls the variable
        // it was given, so it gets a copy of the handle to null.
        let mut moved = adjacency.as_raw();
        // SAFETY: valid out-pointers and a live matrix.
        call(|msg| unsafe { sys::LAGraph_New(&raw mut raw, &raw mut moved, kind.raw(), msg) })?;
        let graph = Self {
            raw,
            adjacency,
            kind,
        };
        graph.check()?;
        Ok(graph)
    }

    /// `LAGraph_CheckGraph`: the matrix is square and held by row, and
    /// every cached property has the shape and type LAGraph expects.
    ///
    /// # Errors
    ///
    /// [`InvalidGraph`](crate::Status::InvalidGraph), with the reason in the
    /// message.
    pub fn check(&self) -> Result<()> {
        // SAFETY: a live graph.
        call(|msg| unsafe { sys::LAGraph_CheckGraph(self.raw, msg) }).map(drop)
    }

    /// The adjacency matrix, to read.
    #[must_use]
    pub const fn adjacency(&self) -> &Matrix<T> {
        &self.adjacency
    }

    #[must_use]
    pub const fn kind(&self) -> Kind {
        self.kind
    }

    /// Delete the graph and its cached properties, and give the adjacency
    /// matrix back.
    #[must_use]
    pub fn into_matrix(self) -> Matrix<T> {
        let mut graph = ManuallyDrop::new(self);
        graph.delete();
        // SAFETY: `graph` is never dropped, so this is the only read of the
        // field, and `delete` detached the matrix from the LAGraph object.
        unsafe { ptr::read(&raw const graph.adjacency) }
    }

    /// Compute and keep `Aᵀ` (`LAGraph_Cached_AT`), pinned to row-major
    /// like the adjacency matrix. Nothing to do for an undirected graph.
    ///
    /// # Errors
    ///
    /// Out of memory.
    pub fn cache_transpose(&mut self) -> Result<()> {
        // SAFETY: a live graph.
        call(|msg| unsafe { sys::LAGraph_Cached_AT(self.raw, msg) })?;
        self.pin_transpose()
    }

    /// Compute and keep the out-degree of every node
    /// (`LAGraph_Cached_OutDegree`).
    ///
    /// # Errors
    ///
    /// Out of memory.
    pub fn cache_out_degree(&mut self) -> Result<()> {
        // SAFETY: a live graph.
        call(|msg| unsafe { sys::LAGraph_Cached_OutDegree(self.raw, msg) }).map(drop)
    }

    /// Compute and keep the in-degree of every node
    /// (`LAGraph_Cached_InDegree`). Nothing to do for an undirected graph,
    /// whose degrees are the out-degrees.
    ///
    /// # Errors
    ///
    /// Out of memory.
    pub fn cache_in_degree(&mut self) -> Result<()> {
        // SAFETY: a live graph.
        call(|msg| unsafe { sys::LAGraph_Cached_InDegree(self.raw, msg) }).map(drop)
    }

    /// Find out whether the matrix has a symmetric structure and keep the
    /// answer (`LAGraph_Cached_IsSymmetricStructure`). On a directed graph
    /// this also computes and keeps the transpose.
    ///
    /// # Errors
    ///
    /// Out of memory.
    pub fn cache_symmetric_structure(&mut self) -> Result<bool> {
        // SAFETY: a live graph.
        call(|msg| unsafe { sys::LAGraph_Cached_IsSymmetricStructure(self.raw, msg) })?;
        self.pin_transpose()?;
        Ok(self.is_symmetric_structure() == Some(true))
    }

    /// Compute and keep the smallest edge weight (`LAGraph_Cached_EMin`).
    ///
    /// # Errors
    ///
    /// Out of memory; an element type LAGraph has no minimum for.
    pub fn cache_min_weight(&mut self) -> Result<()> {
        // SAFETY: a live graph.
        call(|msg| unsafe { sys::LAGraph_Cached_EMin(self.raw, msg) }).map(drop)
    }

    /// Whether the structure is symmetric, if that is known: always for an
    /// undirected graph, and for a directed one after
    /// [`cache_symmetric_structure`](Self::cache_symmetric_structure).
    #[must_use]
    pub fn is_symmetric_structure(&self) -> Option<bool> {
        // SAFETY: a live graph; LAGraph only ever stores one of the three
        // enumerators in this field.
        match unsafe { (*self.raw).is_symmetric_structure } {
            sys::LAGraph_Boolean::LAGraph_TRUE => Some(true),
            sys::LAGraph_Boolean::LAGraph_FALSE => Some(false),
            sys::LAGraph_Boolean::LAGraph_BOOLEAN_UNKNOWN => None,
        }
    }

    /// Whether the transpose is cached.
    #[must_use]
    pub fn has_transpose(&self) -> bool {
        // SAFETY: a live graph.
        !unsafe { (*self.raw).AT }.is_null()
    }

    /// The raw graph, for the algorithms in this crate.
    pub(crate) const fn as_raw(&self) -> sys::LAGraph_Graph {
        self.raw
    }

    /// Hold the cached transpose by row. LAGraph allocates it with
    /// `GrB_Matrix_new`, which holds an `n x 1` matrix by column, and then
    /// rejects it in `LAGraph_CheckGraph`.
    fn pin_transpose(&mut self) -> Result<()> {
        // SAFETY: a live graph.
        let transpose = unsafe { (*self.raw).AT };
        if transpose.is_null() {
            return Ok(());
        }
        // SAFETY: a live matrix that LAGraph created with the adjacency
        // matrix's type. It is adopted only for this one call and handed
        // straight back, so the graph stays its owner.
        let mut transpose = unsafe { Matrix::<T>::from_raw(transpose) };
        let pinned = transpose.set_orientation(Orientation::ByRow);
        let _ = transpose.into_raw();
        Ok(pinned?)
    }
}

impl<T> Graph<T> {
    /// Free the LAGraph object and its cached properties, leaving the
    /// adjacency matrix to its Rust owner.
    fn delete(&mut self) {
        // SAFETY: a live graph. Detaching `A` first keeps `LAGraph_Delete`
        // from freeing the matrix `self.adjacency` owns.
        unsafe { (*self.raw).A = null_mut() };
        // SAFETY: a live graph; `LAGraph_Delete` nulls `self.raw`. Its only
        // failures are frees GraphBLAS refuses, which leave nothing to do.
        let _ = call(|msg| unsafe { sys::LAGraph_Delete(&raw mut self.raw, msg) });
    }
}

impl<T> Drop for Graph<T> {
    fn drop(&mut self) {
        self.delete();
    }
}

impl<T: GrbType> fmt::Debug for Graph<T> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("Graph")
            .field("kind", &self.kind)
            .field("adjacency", &self.adjacency)
            .finish_non_exhaustive()
    }
}
