//! Safe Rust abstraction and RAII lifecycle management for SuiteSparse LAGraph.
//!
//! LAGraph implements high-performance graph algorithms on top of SuiteSparse:GraphBLAS.
//! This module encapsulates `LAGraph_Graph` pointers in the [`LAGraph`] RAII struct,
//! guaranteeing deterministic destruction without memory leaks, handling borrowed adjacency
//! matrices, and providing typed algorithm interfaces returning RAII [`Vector<T>`] instances.

use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    os::raw::c_char,
    ptr::{addr_of_mut, null_mut},
};

use super::{
    GrB_Index, GrB_Info, GrB_Matrix, GrB_Matrix_dup, GrB_Matrix_extractTuples_FP64,
    GrB_Matrix_free, GrB_Matrix_nvals, GrB_Matrix_set_INT32, GrB_Orientation, GrB_Vector,
    GrB_Vector_free, GxB_Option_Field,
    lagraph_bindings::{self, LAGraph_Boolean, LAGraph_Graph, LAGraph_Graph_struct, LAGraph_Kind},
    lagraphx_bindings,
    matrix::Matrix,
    vector::Vector,
};

/// 256-byte diagnostic message buffer for LAGraph operations.
///
/// Element type is `c_char` (not `i8`) to account for architecture-dependent
/// signedness (signed on x86_64, unsigned on aarch64 Linux).
pub type LagMsg = [c_char; 256];

/// Creates a new, zero-initialized LAGraph message buffer.
#[must_use]
pub const fn new_msg() -> LagMsg {
    [0; 256]
}

/// Decodes an LAGraph diagnostic message buffer into a Rust [`String`].
///
/// Scans within the bounded 256-byte message array for a NUL terminator,
/// avoiding reading out-of-bounds if no terminator is present.
#[must_use]
pub fn msg_to_string(msg: &LagMsg) -> String {
    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(msg.as_ptr().cast::<u8>(), msg.len()) };
    let nul_idx = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..nul_idx]).into_owned()
}

/// Formats an LAGraph error message from return code and diagnostic buffer.
#[must_use]
pub fn format_lagraph_error(
    info: i32,
    msg: &LagMsg,
    op: &str,
) -> String {
    let err_str = msg_to_string(msg);
    if err_str.trim().is_empty() {
        format!("{op} failed: {info}")
    } else {
        format!("{op} failed: {info}; {err_str}")
    }
}

/// Checks an LAGraph return code and returns `Ok(())` or a descriptive `Err`.
///
/// If `info != 0`, extracts the error message from `msg` if non-empty.
pub fn check_lagraph(
    info: i32,
    msg: &LagMsg,
    op: &str,
) -> Result<(), String> {
    if info == 0 {
        Ok(())
    } else {
        Err(format_lagraph_error(info, msg, op))
    }
}

/// RAII guard for temporary raw `GrB_Matrix` handles produced by algorithms.
///
/// Automatically frees the matrix handle on drop unless explicitly disarmed.
#[derive(Debug)]
pub struct MatrixGuard(GrB_Matrix);

unsafe impl Send for MatrixGuard {}

impl Drop for MatrixGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = GrB_Matrix_free(addr_of_mut!(self.0));
            }
        }
    }
}

impl MatrixGuard {
    /// Adopts exclusive ownership of a raw `GrB_Matrix` handle into an RAII guard.
    ///
    /// # Safety
    /// The caller must ensure `m` is a valid GraphBLAS matrix handle or null,
    /// and transfers exclusive ownership of `m` to the returned [`MatrixGuard`].
    #[must_use]
    pub const unsafe fn from_raw(m: GrB_Matrix) -> Self {
        Self(m)
    }

    /// Returns the raw `GrB_Matrix` pointer.
    #[must_use]
    pub const fn inner(&self) -> GrB_Matrix {
        self.0
    }

    /// Consumes the guard, returning the raw `GrB_Matrix` without dropping it.
    #[must_use]
    pub const fn into_raw(mut self) -> GrB_Matrix {
        let raw = self.0;
        self.0 = null_mut();
        std::mem::forget(self);
        raw
    }

    /// Extracts `(rows, cols, vals)` from an FP64 matrix.
    #[must_use]
    pub fn extract_tuples_fp64(&self) -> (Vec<u64>, Vec<u64>, Vec<f64>) {
        if self.0.is_null() {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        unsafe {
            let mut nvals: GrB_Index = 0;
            let info = GrB_Matrix_nvals(addr_of_mut!(nvals), self.0);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let mut rows = vec![0u64; nvals as usize];
            let mut cols = vec![0u64; nvals as usize];
            let mut vals = vec![0.0f64; nvals as usize];
            let mut nvals_out = nvals;
            let info = GrB_Matrix_extractTuples_FP64(
                rows.as_mut_ptr(),
                cols.as_mut_ptr(),
                vals.as_mut_ptr(),
                addr_of_mut!(nvals_out),
                self.0,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            rows.truncate(nvals_out as usize);
            cols.truncate(nvals_out as usize);
            vals.truncate(nvals_out as usize);
            (rows, cols, vals)
        }
    }
}

/// Safe RAII wrapper around raw [`LAGraph_Graph`].
///
/// Automatically deletes the underlying graph resources on drop.
/// When `borrowed` is true, the adjacency matrix `A` is owned by an external container
/// (e.g., [`Matrix<bool>`]), so `A` is detached before calling `LAGraph_Delete` to
/// prevent freeing the caller's matrix.
///
/// The lifetime parameter `'a` ties the wrapper to the source matrix when borrowed
/// through [`LAGraph::from_borrowed_matrix`].
pub struct LAGraph<'a> {
    graph: LAGraph_Graph,
    borrowed: bool,
    _marker: PhantomData<&'a ()>,
}

unsafe impl Send for LAGraph<'_> {}

impl Drop for LAGraph<'_> {
    fn drop(&mut self) {
        if !self.graph.is_null() {
            unsafe {
                if self.borrowed
                    && let Some(g_ref) = self.graph.as_mut()
                {
                    g_ref.A = null_mut();
                }
                let mut msg = new_msg();
                let _ =
                    lagraph_bindings::LAGraph_Delete(addr_of_mut!(self.graph), msg.as_mut_ptr());
            }
        }
    }
}

impl<'a> LAGraph<'a> {
    /// Creates a new [`LAGraph`] wrapper from a GraphBLAS adjacency matrix.
    ///
    /// # Safety
    /// - If `borrowed` is false, the caller transfers exclusive ownership of `adj`
    ///   to the returned graph. On failure or destruction, `adj` is freed.
    /// - If `borrowed` is true, the caller must ensure that `adj` remains valid
    ///   for the lifetime `'a`.
    pub unsafe fn new(
        adj: GrB_Matrix,
        kind: LAGraph_Kind,
        borrowed: bool,
    ) -> Result<LAGraph<'a>, String> {
        let mut g: LAGraph_Graph = null_mut();
        let mut msg = new_msg();
        let mut adj_mut = adj;
        let info = unsafe {
            lagraph_bindings::LAGraph_New(
                addr_of_mut!(g),
                addr_of_mut!(adj_mut),
                kind,
                msg.as_mut_ptr(),
            )
        };
        if info != 0 {
            if !borrowed {
                unsafe {
                    let _ = GrB_Matrix_free(addr_of_mut!(adj_mut));
                }
            }
            return Err(format_lagraph_error(info, &msg, "LAGraph_New"));
        }
        if g.is_null() {
            return Err(String::from("LAGraph_New returned null graph"));
        }
        Ok(Self {
            graph: g,
            borrowed,
            _marker: PhantomData,
        })
    }

    /// Creates an [`LAGraph`] instance from a borrowed [`Matrix<bool>`].
    ///
    /// To ensure that cached LAGraph properties (e.g. `AT`, out-degree) cannot be
    /// corrupted or reused invalidly after another shared owner of the `Matrix`
    /// mutates the adjacency, a private adjacency snapshot is created via `GrB_Matrix_dup`.
    /// The returned graph takes exclusive ownership of this snapshot (`borrowed = false`),
    /// preserving cache reuse across operations on this [`LAGraph`] instance while isolating
    /// it from any subsequent mutations on the source matrix.
    pub fn from_borrowed_matrix(
        matrix: &'a Matrix<bool>,
        kind: LAGraph_Kind,
    ) -> Result<LAGraph<'a>, String> {
        let mut snapshot = MaybeUninit::<GrB_Matrix>::uninit();
        let info = unsafe { GrB_Matrix_dup(snapshot.as_mut_ptr(), matrix.inner()) };
        if info != GrB_Info::GrB_SUCCESS {
            return Err(format!("GrB_Matrix_dup failed: {info:?}"));
        }
        let snapshot_m = unsafe { snapshot.assume_init() };
        unsafe { Self::new(snapshot_m, kind, false) }
    }

    /// Creates an [`LAGraph`] instance by lending a locally owned, unshared [`Matrix<bool>`]
    /// directly without duplicating its underlying storage.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `matrix` is uniquely owned and not shared or mutated by other handles during traversal.
    /// - `matrix` outlives the returned [`LAGraph`] instance (which is also constrained by `'a`).
    pub unsafe fn from_lent_matrix(
        matrix: &'a Matrix<bool>,
        kind: LAGraph_Kind,
    ) -> Result<LAGraph<'a>, String> {
        unsafe { Self::new(matrix.inner(), kind, true) }
    }

    /// Consumes the wrapper and returns the raw `LAGraph_Graph` pointer without dropping.
    #[must_use]
    pub const fn into_raw(mut self) -> LAGraph_Graph {
        let raw = self.graph;
        self.graph = null_mut();
        std::mem::forget(self);
        raw
    }

    /// Frees a raw `LAGraph_Graph` pointer, detaching `A` if `borrowed` is true.
    ///
    /// # Safety
    /// The caller must ensure `*g` is a valid pointer or null.
    pub unsafe fn free_raw(
        g: &mut LAGraph_Graph,
        borrowed: bool,
    ) {
        if !g.is_null() {
            let raw = *g;
            *g = null_mut();
            drop(LAGraph {
                graph: raw,
                borrowed,
                _marker: PhantomData,
            });
        }
    }

    /// Returns the raw `LAGraph_Graph` pointer.
    #[must_use]
    pub const fn inner(&self) -> LAGraph_Graph {
        self.graph
    }

    /// Returns a mutable reference to the underlying [`LAGraph_Graph_struct`], if not null.
    ///
    /// # Safety
    /// The caller must not replace or invalidate `A`, `AT`, or any cached resource handles.
    #[must_use]
    const unsafe fn inner_mut(&mut self) -> Option<&mut LAGraph_Graph_struct> {
        if self.graph.is_null() {
            None
        } else {
            unsafe { self.graph.as_mut() }
        }
    }

    /// Whether the underlying adjacency matrix is borrowed from an external owner.
    #[must_use]
    pub const fn is_borrowed(&self) -> bool {
        self.borrowed
    }

    /// Precomputes and caches the transpose matrix `AT` on the graph,
    /// pinning it to row-major layout to prevent LAGraph failure on 1x1 matrices (#3349).
    pub fn cache_transpose(&mut self) -> Result<(), String> {
        let mut msg = new_msg();
        let info = unsafe { lagraph_bindings::LAGraph_Cached_AT(self.graph, msg.as_mut_ptr()) };
        if info != 0 && info != lagraph_bindings::LAGRAPH_CACHE_NOT_NEEDED as i32 {
            return Err(format_lagraph_error(info, &msg, "LAGraph_Cached_AT"));
        }
        if let Some(graph) = unsafe { self.inner_mut() }
            && !graph.AT.is_null()
        {
            unsafe {
                GrB_Matrix_set_INT32(
                    graph.AT,
                    GrB_Orientation::GrB_ROWMAJOR as i32,
                    GxB_Option_Field::GrB_STORAGE_ORIENTATION_HINT as _,
                );
            }
        }
        Ok(())
    }

    /// Precomputes and caches the out-degree vector on the graph.
    pub fn cache_out_degree(&mut self) -> Result<(), String> {
        let mut msg = new_msg();
        let info =
            unsafe { lagraph_bindings::LAGraph_Cached_OutDegree(self.graph, msg.as_mut_ptr()) };
        check_lagraph(info, &msg, "LAGraph_Cached_OutDegree")
    }

    /// Precomputes and caches the minimum edge weight on the graph.
    pub fn cache_emin(&mut self) -> Result<(), String> {
        let mut msg = new_msg();
        let info = unsafe { lagraph_bindings::LAGraph_Cached_EMin(self.graph, msg.as_mut_ptr()) };
        check_lagraph(info, &msg, "LAGraph_Cached_EMin")
    }

    /// Sets whether the graph has a known symmetric structure.
    pub fn set_symmetric_structure(
        &mut self,
        is_symmetric: bool,
    ) -> Result<(), String> {
        let g_ref = unsafe { self.inner_mut() }
            .ok_or_else(|| String::from("LAGraph graph pointer is null"))?;
        g_ref.is_symmetric_structure = if is_symmetric {
            LAGraph_Boolean::LAGraph_TRUE
        } else {
            LAGraph_Boolean::LAGraph_FALSE
        };
        Ok(())
    }

    /// Runs PageRank on the graph.
    ///
    /// Precomputes and caches `AT` and `out_degree`, then invokes `LAGr_PageRank`.
    pub fn page_rank(
        &mut self,
        damping: f64,
        tol: f64,
        max_iters: usize,
    ) -> Result<(Vector<f64>, i32), String> {
        self.cache_transpose()?;
        self.cache_out_degree()?;

        let mut centrality: GrB_Vector = null_mut();
        let mut iters: i32 = 0;
        let mut msg = new_msg();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "FFI parameter types expect f32 and c_int"
        )]
        let info = unsafe {
            lagraph_bindings::LAGr_PageRank(
                addr_of_mut!(centrality),
                addr_of_mut!(iters),
                self.graph,
                damping as f32,
                tol as f32,
                max_iters as i32,
                msg.as_mut_ptr(),
            )
        };
        if info != 0 {
            if !centrality.is_null() {
                unsafe {
                    let _ = GrB_Vector_free(addr_of_mut!(centrality));
                }
            }
            return Err(format_lagraph_error(info, &msg, "LAGr_PageRank"));
        }
        Ok((unsafe { Vector::from_raw(centrality) }, iters))
    }

    /// Runs Connected Components on the undirected graph.
    ///
    /// Sets `is_symmetric_structure = LAGraph_TRUE`, then invokes `LAGr_ConnectedComponents`.
    pub fn connected_components(&mut self) -> Result<Vector<i64>, String> {
        self.set_symmetric_structure(true)?;

        let mut component: GrB_Vector = null_mut();
        let mut msg = new_msg();
        let info = unsafe {
            lagraph_bindings::LAGr_ConnectedComponents(
                addr_of_mut!(component),
                self.graph,
                msg.as_mut_ptr(),
            )
        };
        if info != 0 {
            if !component.is_null() {
                unsafe {
                    let _ = GrB_Vector_free(addr_of_mut!(component));
                }
            }
            return Err(format_lagraph_error(info, &msg, "LAGr_ConnectedComponents"));
        }
        Ok(unsafe { Vector::from_raw(component) })
    }

    /// Runs Betweenness Centrality on the graph from a set of source nodes.
    ///
    /// Precomputes and caches `AT` and `out_degree`, then invokes `LAGr_Betweenness`.
    pub fn betweenness(
        &mut self,
        sources: &[u64],
    ) -> Result<Vector<f64>, String> {
        self.cache_transpose()?;
        self.cache_out_degree()?;

        let mut centrality: GrB_Vector = null_mut();
        let mut msg = new_msg();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "FFI parameter expects c_int count"
        )]
        let info = unsafe {
            lagraph_bindings::LAGr_Betweenness(
                addr_of_mut!(centrality),
                self.graph,
                sources.as_ptr(),
                sources.len() as i32,
                msg.as_mut_ptr(),
            )
        };
        if info != 0 {
            if !centrality.is_null() {
                unsafe {
                    let _ = GrB_Vector_free(addr_of_mut!(centrality));
                }
            }
            return Err(format_lagraph_error(info, &msg, "LAGr_Betweenness"));
        }
        Ok(unsafe { Vector::from_raw(centrality) })
    }

    /// Runs extended Breadth-First Search from a source node up to `max_level`.
    ///
    /// If `want_parent` is true, returns `(Some(level), Some(parent))`.
    /// If `want_parent` is false, returns `(Some(level), None)`.
    pub fn breadth_first_search(
        &mut self,
        source: u64,
        max_level: i64,
        want_parent: bool,
    ) -> Result<(Option<Vector<i64>>, Option<Vector<i64>>), String> {
        let mut level: GrB_Vector = null_mut();
        let mut parent: GrB_Vector = null_mut();
        let mut msg = new_msg();
        let info = unsafe {
            lagraphx_bindings::LAGr_BreadthFirstSearch_Extended(
                addr_of_mut!(level),
                if want_parent {
                    addr_of_mut!(parent)
                } else {
                    null_mut()
                },
                self.graph,
                source,
                max_level,
                -1,
                false,
                msg.as_mut_ptr(),
            )
        };
        if info != 0 {
            if !level.is_null() {
                unsafe {
                    let _ = GrB_Vector_free(addr_of_mut!(level));
                }
            }
            if !parent.is_null() {
                unsafe {
                    let _ = GrB_Vector_free(addr_of_mut!(parent));
                }
            }
            return Err(format_lagraph_error(
                info,
                &msg,
                "LAGr_BreadthFirstSearch_Extended",
            ));
        }
        let level_vec = if level.is_null() {
            None
        } else {
            Some(unsafe { Vector::from_raw(level) })
        };
        let parent_vec = if parent.is_null() {
            None
        } else {
            Some(unsafe { Vector::from_raw(parent) })
        };
        Ok((level_vec, parent_vec))
    }

    /// Runs Community Detection via Label Propagation (CDLP) on the undirected graph.
    ///
    /// Sets `is_symmetric_structure = LAGraph_TRUE`, then invokes `LAGraph_cdlp`.
    pub fn cdlp(
        &mut self,
        max_iters: usize,
    ) -> Result<Vector<i64>, String> {
        self.set_symmetric_structure(true)?;

        let mut cdlp: GrB_Vector = null_mut();
        let mut msg = new_msg();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "FFI parameter expects c_int count"
        )]
        let info = unsafe {
            lagraphx_bindings::LAGraph_cdlp(
                addr_of_mut!(cdlp),
                self.graph,
                max_iters as i32,
                msg.as_mut_ptr(),
            )
        };
        if info != 0 {
            if !cdlp.is_null() {
                unsafe {
                    let _ = GrB_Vector_free(addr_of_mut!(cdlp));
                }
            }
            return Err(format_lagraph_error(info, &msg, "LAGraph_cdlp"));
        }
        Ok(unsafe { Vector::from_raw(cdlp) })
    }

    /// Runs Harmonic Centrality on the graph.
    pub fn harmonic_centrality(
        &mut self,
        nodes: &Vector<bool>,
    ) -> Result<(Vector<f64>, Vector<i64>), String> {
        let mut scores: GrB_Vector = null_mut();
        let mut reachable_nodes: GrB_Vector = null_mut();
        let mut msg = new_msg();
        let info = unsafe {
            lagraphx_bindings::LAGr_HarmonicCentrality(
                addr_of_mut!(scores),
                addr_of_mut!(reachable_nodes),
                self.graph,
                nodes.ptr(),
                msg.as_mut_ptr(),
            )
        };
        if info != 0 {
            if !scores.is_null() {
                unsafe {
                    let _ = GrB_Vector_free(addr_of_mut!(scores));
                }
            }
            if !reachable_nodes.is_null() {
                unsafe {
                    let _ = GrB_Vector_free(addr_of_mut!(reachable_nodes));
                }
            }
            return Err(format_lagraph_error(info, &msg, "LAGr_HarmonicCentrality"));
        }
        Ok((unsafe { Vector::from_raw(scores) }, unsafe {
            Vector::from_raw(reachable_nodes)
        }))
    }

    /// Runs Max Flow on the graph, returning the maximum flow and an RAII [`MatrixGuard`]
    /// wrapping the flow matrix.
    pub fn max_flow(
        &mut self,
        src: u64,
        sink: u64,
    ) -> Result<(f64, MatrixGuard), String> {
        self.cache_transpose()?;
        self.cache_emin()?;
        let mut lag_max_flow: f64 = 0.0;
        let mut flow_mtx: GrB_Matrix = null_mut();
        let mut msg = new_msg();
        let info = unsafe {
            lagraphx_bindings::LAGr_MaxFlow(
                addr_of_mut!(lag_max_flow),
                addr_of_mut!(flow_mtx),
                null_mut(),
                self.graph,
                src,
                sink,
                msg.as_mut_ptr(),
            )
        };
        if info != 0 {
            if !flow_mtx.is_null() {
                unsafe {
                    let _ = GrB_Matrix_free(addr_of_mut!(flow_mtx));
                }
            }
            return Err(format_lagraph_error(info, &msg, "LAGr_MaxFlow"));
        }
        if flow_mtx.is_null() {
            return Err(String::from("LAGr_MaxFlow returned null flow matrix"));
        }
        Ok((lag_max_flow, unsafe { MatrixGuard::from_raw(flow_mtx) }))
    }
}

impl LAGraph<'static> {
    /// Creates an [`LAGraph`] instance taking ownership of an unmanaged `GrB_Matrix`.
    ///
    /// # Safety
    /// The caller must ensure that `adj` is a valid GraphBLAS matrix handle and yields
    /// exclusive ownership to the returned wrapper.
    pub unsafe fn from_owned_raw(
        adj: GrB_Matrix,
        kind: LAGraph_Kind,
    ) -> Result<Self, String> {
        unsafe { Self::new(adj, kind, false) }
    }

    /// Constructs an [`LAGraph`] wrapper from an existing raw pointer.
    ///
    /// # Safety
    /// The caller must ensure `graph` is a valid `LAGraph_Graph` pointer.
    #[must_use]
    pub const unsafe fn from_raw_unmanaged(
        graph: LAGraph_Graph,
        borrowed: bool,
    ) -> Self {
        Self {
            graph,
            borrowed,
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lagraph_lifecycle() {
        crate::graph::graphblas::test_init::ensure_init();
        let matrix = Matrix::<bool>::new(5, 5);
        let mut clone = matrix.clone();
        let mut lag =
            LAGraph::from_borrowed_matrix(&matrix, LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED)
                .expect("LAGraph creation should succeed");
        // Private adjacency snapshot means LAGraph owns its matrix snapshot
        assert!(!lag.is_borrowed());
        assert!(!lag.inner().is_null());

        // Cache properties
        lag.cache_transpose()
            .expect("cache transpose should succeed");
        lag.cache_out_degree()
            .expect("cache out degree should succeed");

        // Mutating source matrix via another shared owner does not alter the private snapshot
        clone.set(0, 1, true);

        // Snapshot remains intact and BFS operates reliably
        let (level, _) = lag
            .breadth_first_search(0, 1, false)
            .expect("BFS should succeed");
        let level_tuples = level
            .expect("level vector should be present")
            .extract_tuples();
        assert_eq!(level_tuples.len(), 1); // Only node 0 (isolated in snapshot)

        drop(lag);

        // Lending an unshared matrix directly avoids duplication and detaches on drop
        let unshared = Matrix::<bool>::new(5, 5);
        let lent_lag = unsafe {
            LAGraph::from_lent_matrix(&unshared, LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED)
        }
        .expect("LAGraph lending should succeed");
        assert!(lent_lag.is_borrowed());
        assert!(!lent_lag.inner().is_null());
        drop(lent_lag);
    }

    #[test]
    fn test_lagraph_msg_decoding() {
        let msg = new_msg();
        let err_str = msg_to_string(&msg);
        assert_eq!(err_str, "");

        let res = check_lagraph(0, &msg, "test_op");
        assert!(res.is_ok());

        let err = check_lagraph(-1, &msg, "test_op");
        assert_eq!(err.unwrap_err(), "test_op failed: -1");

        let err_formatted = format_lagraph_error(-1, &msg, "test_op");
        assert_eq!(err_formatted, "test_op failed: -1");

        // Bounded decode when no NUL terminator is present in the 256-byte buffer
        #[allow(clippy::cast_possible_wrap)]
        let full_msg = [b'x' as c_char; 256];
        let full_str = msg_to_string(&full_msg);
        assert_eq!(full_str.len(), 256);
        assert_eq!(full_str, "x".repeat(256));
    }

    #[test]
    fn test_matrix_guard_null_safety() {
        let guard = unsafe { MatrixGuard::from_raw(null_mut()) };
        assert!(guard.inner().is_null());
        let (rows, cols, vals) = guard.extract_tuples_fp64();
        assert_eq!(rows.len(), 0);
        assert_eq!(cols.len(), 0);
        assert_eq!(vals.len(), 0);
        drop(guard);
    }
}
