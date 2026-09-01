//! Graph algorithm procedures backed by LAGraph / GraphBLAS.
//!
//! Each procedure is exposed as a Cypher `CALL` statement and returns
//! a result set of `Map` rows.  The general execution flow is:
//!
//! ```text
//!  Cypher query                  Rust runtime
//!  CALL algo.pageRank(...)  -->  algo_pagerank()
//!                                   |
//!                                   v
//!                            +-----------------+
//!                            | Build adjacency |  graph.build_adjacency_matrix()
//!                            | matrix (GrB)    |  or build_symmetric_adjacency_matrix()
//!                            +-----------------+
//!                                   |
//!                                   v
//!                            +-----------------+
//!                            | Compact & re-   |  build_compact_adj()
//!                            | index to 0..n-1 |  maps active node IDs to dense indices
//!                            +-----------------+
//!                                   |
//!                                   v
//!                            +-----------------+
//!                            | LAGraph / ext   |  FFI call into C library
//!                            | algorithm       |  (PageRank, WCC, BFS, ...)
//!                            +-----------------+
//!                                   |
//!                                   v
//!                            +-----------------+
//!                            | Map compact IDs |  compact_to_id[] lookup
//!                            | back to originals|
//!                            +-----------------+
//!                                   |
//!                                   v
//!                            Return List<Map>
//! ```
//!
//! ## Available algorithms
//!
//! ```text
//!  Cypher procedure          Algorithm           Yields
//! ───────────────────────────────────────────────────────────────────
//!  algo.pageRank(l, t)       LAGr_PageRank       {node, score}
//!  algo.WCC(config?)         LAGr_ConnectedComp  {node, componentId}
//!  algo.betweenness(config?) LAGr_Betweenness    {node, score}
//!  algo.BFS(src, depth, t)   LAGr_BFS_Extended   {nodes, edges}
//!  algo.labelPropagation(..) LAGraph_cdlp        {node, communityId}
//!  algo.MSF(config?)         LAGraph_msf         {nodes, edges}
//!  algo.SPpaths(config)      Dijkstra (Rust)     {path, pathWeight, pathCost}
//!  algo.SSpaths(config)      Dijkstra (Rust)     {path, pathWeight, pathCost}
//! ```
//!
//! ## Compact adjacency matrix
//!
//! LAGraph operates on dense 0..n-1 indexed matrices, but the graph
//! may have gaps in its node ID space (deleted nodes).  `build_compact_adj`
//! creates a compacted boolean `GrB_Matrix` and two-way mappings
//! (`id_to_compact` / `compact_to_id`) so results can be translated
//! back to the original node IDs.
//!
//! ## Configuration
//!
//! Most algorithms accept an optional config `Map` with keys like
//! `nodeLabels`, `relationshipTypes`, `samplingSize`, `maxIterations`,
//! etc.  Invalid keys are rejected via `validate_config_map`.

#![allow(clippy::unnecessary_wraps)]
#![allow(unsafe_op_in_unsafe_fn)]

use super::{FnType, Functions, Type, empty_procedure_batch};
use crate::graph::graphblas::tensor::compound_key_inverse;
use crate::{
    graph::{
        attribute_store::AttributeStore,
        graph::{EdgeDirection, Graph, NodeId, RelationshipId},
        graphblas::lagraph_bindings::{self, LAGraph_Boolean, LAGraph_Graph, LAGraph_Kind},
    },
    runtime::{
        batch::{Batch, Column, classify_stored_column},
        ordermap::OrderMap,
        runtime::Runtime,
        value::Value,
    },
};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::mem::swap;
use std::ptr::null_mut;
use std::sync::Arc;
use thin_vec::{ThinVec, thin_vec};

// ── Helpers ─────────────────────────────────────────────────────────────

/// LAGraph message buffer type. Element type is `c_char` (not `i8`) because
/// `char` signedness is platform-dependent — signed on amd64, **unsigned on
/// arm64 Linux**. The LAGraph FFI expects `*mut c_char`, so the buffer must
/// match or the call site fails to compile on arm64.
type LagMsg = [std::os::raw::c_char; 256];

const fn new_msg() -> LagMsg {
    [0; 256]
}

fn msg_to_string(msg: &LagMsg) -> String {
    unsafe { std::ffi::CStr::from_ptr(msg.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Context for the user-defined GraphBLAS weight operator used by `algo.MSF`.
///
/// Holds a borrowed pointer to just the relationship [`AttributeStore`] — the
/// only graph state the operator reads — rather than the whole graph. The
/// pointer is valid for the whole `GrB_Matrix_apply` call: the store is
/// read-locked and never mutated during it, and the operator is freed before the
/// lock releases. Also carries the resolved attribute index and objective sign.
///
/// `#[repr(C)]` plain-old-data so GraphBLAS can memcpy it as the thunk operand
/// and hand a pointer to it to every invocation of [`msf_scored_edge_index_op`].
#[repr(C)]
#[derive(Clone, Copy)]
struct MsfWeightCtx {
    attrs: *const AttributeStore,
    attr_idx: u16,
    maximize: bool,
    /// Unweighted MSF: every edge scores `1.0`, skipping the attribute lookup so
    /// the unweighted path shares the weighted path's parallel apply.
    unit: bool,
}

/// Score one edge id for MSF: the weight attribute (negated when maximizing),
/// `1.0` for unweighted, `±inf` for missing / non-numeric weights (never
/// selected by the min-reduce).
///
/// # Panic safety
/// Called from GraphBLAS OpenMP worker threads (the global panic hook aborts
/// the process), so it must not panic: only a read-only attribute-store lookup
/// that takes a per-shard read lock and clones a `Value` (never panics). The
/// attribute-store pointer stays valid because the caller holds the store
/// read-locked across the whole synchronous apply and frees the operators
/// before releasing it.
#[inline]
unsafe fn msf_score(
    ctx: &MsfWeightCtx,
    edge_id: u64,
) -> f64 {
    if ctx.unit {
        return 1.0;
    }
    let miss = if ctx.maximize {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    // No null check on `attrs`: the caller always sets it from a live
    // `&AttributeStore` (held read-locked for the whole apply), so it can't be
    // null, and a null check couldn't catch the only real risk — a dangling
    // pointer — anyway.
    let raw = match (*ctx.attrs).get_attr_by_idx(edge_id, ctx.attr_idx) {
        Some(Value::Float(f)) => f,
        Some(Value::Int(k)) => k as f64,
        _ => miss,
    };
    if ctx.maximize { -raw } else { raw }
}

/// User-defined GraphBLAS **index-unary** operator for the multi-edge
/// matrix `me[compound_key(src,dst)][edge_id]`, where the column index `j` *is*
/// the edge id (the `bool` value carries no information). Emits the full
/// `{score, edge}` pair so the subsequent monoid row-reduction resolves each
/// pair to its minimum-score edge entirely inside GraphBLAS.
///
/// # Panic safety
/// Runs on GraphBLAS worker threads; see [`msf_score`].
unsafe extern "C" fn msf_scored_edge_index_op(
    z: *mut std::os::raw::c_void,
    _x: *const std::os::raw::c_void,
    _i: crate::graph::graphblas::GrB_Index,
    j: crate::graph::graphblas::GrB_Index,
    y: *const std::os::raw::c_void,
) {
    if z.is_null() || y.is_null() {
        return;
    }
    let ctx = &*y.cast::<MsfWeightCtx>();
    *z.cast::<ScoredEdge>() = ScoredEdge {
        score: msf_score(ctx, j),
        edge: j,
    };
}

/// User-defined GraphBLAS **index-unary** operator for the inline UINT64
/// forward matrix `m`, where the matrix *value* `x` is the edge id (unlike
/// `me`, where the column index is). Writes the full `{score, edge}` pair so
/// one apply produces everything the min-by-score build needs.
///
/// # Panic safety
/// Runs on GraphBLAS worker threads; see [`msf_score`].
unsafe extern "C" fn msf_scored_edge_value_op(
    z: *mut std::os::raw::c_void,
    x: *const std::os::raw::c_void,
    _i: crate::graph::graphblas::GrB_Index,
    _j: crate::graph::graphblas::GrB_Index,
    y: *const std::os::raw::c_void,
) {
    if z.is_null() || x.is_null() || y.is_null() {
        return;
    }
    let ctx = &*y.cast::<MsfWeightCtx>();
    let edge = *x.cast::<u64>();
    *z.cast::<ScoredEdge>() = ScoredEdge {
        score: msf_score(ctx, edge),
        edge,
    };
}

/// Plain-old-data `{score, edge}` pair, the value type of the `rel_adj` matrix.
///
/// Building `rel_adj` with [`msf_keep_min_score`] as the duplicate-combiner makes
/// each node pair resolve to the relationship id of its minimum-score edge — the
/// exact edge the weighted forest selects — so a forest edge's relationship can
/// be recovered with an O(1) `extractElement` instead of re-scanning the tensor.
#[repr(C)]
#[derive(Clone, Copy)]
struct ScoredEdge {
    score: f64,
    edge: u64,
}

/// User-defined GraphBLAS **binary** operator combining two [`ScoredEdge`] by
/// keeping the smaller score (ties keep the first operand). Used as the `dup`
/// when building and symmetrizing `rel_adj`.
///
/// # Panic safety
/// Runs on GraphBLAS worker threads (the panic hook aborts the process), so it
/// does only null-guarded plain-old-data reads and never unwinds.
unsafe extern "C" fn msf_keep_min_score(
    z: *mut std::os::raw::c_void,
    x: *const std::os::raw::c_void,
    y: *const std::os::raw::c_void,
) {
    if z.is_null() || x.is_null() || y.is_null() {
        return;
    }
    let x = &*x.cast::<ScoredEdge>();
    let y = &*y.cast::<ScoredEdge>();
    // Tie-break on edge id: keeps the op commutative, which the parallel
    // monoid row-reduction of `me` requires (operands arrive in arbitrary
    // order across worker threads), and makes results deterministic.
    *z.cast::<ScoredEdge>() = if y.score < x.score || (y.score == x.score && y.edge < x.edge) {
        *y
    } else {
        *x
    };
}

/// User-defined GraphBLAS **unary** operator projecting an [`ScoredEdge`] to its
/// `score` (FP64). Applied to the finished `rel_adj` to derive the plain-weight
/// `weighted_adj` Boruvka needs, so the score is stored once (in `rel_adj`) rather
/// than built into a second matrix from a parallel array.
///
/// # Panic safety
/// Runs on GraphBLAS worker threads (the panic hook aborts the process); does
/// only null-guarded plain-old-data reads and never unwinds.
unsafe extern "C" fn msf_score_of(
    z: *mut std::os::raw::c_void,
    x: *const std::os::raw::c_void,
) {
    if z.is_null() || x.is_null() {
        return;
    }
    *z.cast::<f64>() = (*x.cast::<ScoredEdge>()).score;
}

/// Extract an optional string-or-null from a Value, returning an error if
/// the value is of any other type.
fn opt_string(
    v: &Value,
    name: &str,
) -> Result<Option<Arc<String>>, String> {
    match v {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s.clone())),
        _ => Err(format!("Type mismatch: expected String or Null for {name}")),
    }
}

/// Validate a map-style config for the given allowed keys.
fn validate_config_map(
    map: &OrderMap<Arc<String>, Value>,
    allowed: &[&str],
) -> Result<(), String> {
    for key in map.keys() {
        if !allowed.iter().any(|a| *a == key.as_str()) {
            return Err(format!("Unknown parameter: {key}"));
        }
    }
    Ok(())
}

/// Extract `nodeLabels` from config map. Must be a list of strings if present.
fn extract_node_labels(map: &OrderMap<Arc<String>, Value>) -> Result<Vec<Arc<String>>, String> {
    match map.get(&Arc::new(String::from("nodeLabels"))) {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::List(list)) => {
            let mut labels = Vec::with_capacity(list.len());
            for v in list.iter() {
                match v {
                    Value::String(s) => labels.push(s.clone()),
                    _ => return Err(String::from("nodeLabels must be an array of strings")),
                }
            }
            Ok(labels)
        }
        _ => Err(String::from("nodeLabels must be an array of strings")),
    }
}

/// Extract `relationshipTypes` from config map. Must be a list of strings if present.
fn extract_rel_types(map: &OrderMap<Arc<String>, Value>) -> Result<Vec<Arc<String>>, String> {
    match map.get(&Arc::new(String::from("relationshipTypes"))) {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::List(list)) => {
            let mut types = Vec::with_capacity(list.len());
            for v in list.iter() {
                match v {
                    Value::String(s) => types.push(s.clone()),
                    _ => {
                        return Err(String::from(
                            "relationshipTypes must be an array of strings",
                        ));
                    }
                }
            }
            Ok(types)
        }
        _ => Err(String::from(
            "relationshipTypes must be an array of strings",
        )),
    }
}

/// Parse config from a single argument: either Null (no config) or a Map.
fn parse_config(args: &[Value]) -> Result<OrderMap<Arc<String>, Value>, String> {
    if args.is_empty() {
        return Ok(OrderMap::default());
    }
    match &args[0] {
        Value::Null => Ok(OrderMap::default()),
        Value::Map(m) => Ok((**m).clone()),
        _ => Err(String::from(
            "Invalid argument type: expected a map or null",
        )),
    }
}

/// Create an LAGraph_Graph from a raw GrB_Matrix.
/// The graph takes ownership of the matrix pointer—caller must NOT free it
/// after this call.
unsafe fn create_lagraph_graph(
    adj: crate::graph::graphblas::GrB_Matrix,
    kind: LAGraph_Kind,
) -> Result<LAGraph_Graph, String> {
    let mut g: LAGraph_Graph = null_mut();
    let mut msg = new_msg();
    let mut adj_mut = adj;
    let info = lagraph_bindings::LAGraph_New(&raw mut g, &raw mut adj_mut, kind, msg.as_mut_ptr());
    if info != 0 {
        // LAGraph_New did not take ownership; free the matrix to avoid a leak.
        crate::graph::graphblas::GrB_Matrix_free(&raw mut adj_mut);
        return Err(format!("LAGraph_New failed: {info}"));
    }
    if g.is_null() {
        return Err(String::from("LAGraph_New returned null graph"));
    }
    Ok(g)
}

/// Free an LAGraph_Graph.
unsafe fn delete_lagraph_graph(g: &mut LAGraph_Graph) {
    let mut msg = new_msg();
    lagraph_bindings::LAGraph_Delete(g, msg.as_mut_ptr());
}

/// Create an LAGraph_Graph that borrows `adj`: the caller keeps ownership of
/// the matrix and must free the graph with [`delete_lagraph_graph_borrowed`].
/// `LAGraph_New` only fails before taking the matrix, so the error path never
/// leaves the handle inside a graph.
unsafe fn create_lagraph_graph_borrowed(
    adj: crate::graph::graphblas::GrB_Matrix,
    kind: LAGraph_Kind,
) -> Result<LAGraph_Graph, String> {
    let mut g: LAGraph_Graph = null_mut();
    let mut msg = new_msg();
    let mut adj_mut = adj;
    let info = lagraph_bindings::LAGraph_New(&raw mut g, &raw mut adj_mut, kind, msg.as_mut_ptr());
    if info != 0 {
        return Err(format!("LAGraph_New failed: {info}"));
    }
    if g.is_null() {
        return Err(String::from("LAGraph_New returned null graph"));
    }
    Ok(g)
}

/// Free an LAGraph_Graph whose adjacency matrix is borrowed: detach `G->A`
/// first so `LAGraph_Delete` does not free the caller-owned handle.
unsafe fn delete_lagraph_graph_borrowed(g: &mut LAGraph_Graph) {
    if let Some(graph) = g.as_mut() {
        graph.A = null_mut();
    }
    delete_lagraph_graph(g);
}

/// Free an `LAGraph_Graph` whose adjacency matrix may be borrowed.
///
/// The procedures below take one of two paths: the unfiltered path lends
/// LAGraph the matrix it just built and keeps the `Matrix` wrapper that frees
/// it, while the compact path hands over a raw matrix nothing else owns. This
/// picks the matching teardown so the handle is freed exactly once.
unsafe fn delete_lagraph_graph_maybe_borrowed(
    g: &mut LAGraph_Graph,
    borrowed: bool,
) {
    if borrowed {
        delete_lagraph_graph_borrowed(g);
    } else {
        delete_lagraph_graph(g);
    }
}

/// Extract GrB_Vector entries as (index, f64) pairs.
unsafe fn extract_vector_f64(v: crate::graph::graphblas::GrB_Vector) -> Vec<(u64, f64)> {
    use crate::graph::graphblas::{GrB_Index, GrB_Vector_extractTuples_FP64, GrB_Vector_nvals};
    let mut nvals: GrB_Index = 0;
    GrB_Vector_nvals(&raw mut nvals, v);
    let mut indices = vec![0u64; nvals as usize];
    let mut values = vec![0.0f64; nvals as usize];
    let mut nvals_out = nvals;
    GrB_Vector_extractTuples_FP64(
        indices.as_mut_ptr(),
        values.as_mut_ptr(),
        &raw mut nvals_out,
        v,
    );
    indices.into_iter().zip(values).collect()
}

/// Extract GrB_Vector entries as (index, i64) pairs.
unsafe fn extract_vector_i64(v: crate::graph::graphblas::GrB_Vector) -> Vec<(u64, i64)> {
    use crate::graph::graphblas::{GrB_Index, GrB_Vector_extractTuples_INT64, GrB_Vector_nvals};
    let mut nvals: GrB_Index = 0;
    GrB_Vector_nvals(&raw mut nvals, v);
    let mut indices = vec![0u64; nvals as usize];
    let mut values = vec![0i64; nvals as usize];
    let mut nvals_out = nvals;
    GrB_Vector_extractTuples_INT64(
        indices.as_mut_ptr(),
        values.as_mut_ptr(),
        &raw mut nvals_out,
        v,
    );
    indices.into_iter().zip(values).collect()
}

/// Build a node Value with a given NodeId.
const fn node_value(id: NodeId) -> Value {
    Value::Node(id)
}

/// Collect all active node IDs from the graph matching ANY of the given labels.
/// If labels is empty, returns all active node IDs.
fn collect_node_ids(
    g: &Graph,
    labels: &[Arc<String>],
) -> Vec<u64> {
    use crate::runtime::orderset::OrderSet;
    if labels.is_empty() {
        let empty: OrderSet<Arc<String>> = OrderSet::default();
        return g.get_nodes(&empty, 0).map(u64::from).collect();
    }
    // Union: collect nodes matching any of the labels
    let mut ids = FxHashSet::default();
    for label in labels {
        let mut label_set = OrderSet::default();
        label_set.insert(label.clone());
        for n in g.get_nodes(&label_set, 0) {
            ids.insert(u64::from(n));
        }
    }
    let mut result: Vec<u64> = ids.into_iter().collect();
    result.sort_unstable();
    result
}

/// Build a HashSet of all active node IDs (regardless of label).
fn active_node_set(g: &Graph) -> FxHashSet<u64> {
    use crate::runtime::orderset::OrderSet;
    let empty: OrderSet<Arc<String>> = OrderSet::default();
    g.get_nodes(&empty, 0).map(u64::from).collect()
}

/// Bulk-fill an empty BOOL matrix with an all-`true` coordinate pattern, as an
/// **iso** matrix — one value shared by the whole pattern.
///
/// `GxB_Matrix_build_Scalar` rather than `GrB_Matrix_build_BOOL` over a
/// `vec![true; n]`, because since GraphBLAS 10.5.0 the two are no longer
/// equivalent: `GrB_Matrix_build_*` "always build a non-iso matrix"
/// (`Source/builder/GB_build.c`), the post-iso check that used to notice an
/// all-identical value array having been dropped, while `GxB_Matrix_build_Scalar`
/// is documented in that same file as always iso. Non-iso costs a byte per
/// entry — measured at 37,108 against 33,020 bytes for a 4,096-entry pattern —
/// and takes LAGraph off the iso fast paths its algorithms select on. The
/// scalar form also needs no values array at all, which was the only reason to
/// materialize one.
///
/// Duplicate coordinates still collapse to a single entry, exactly as they did
/// under the `GxB_ANY_BOOL` dup operator this replaces: an iso build discards
/// duplicates. The symmetric caller relies on that for self-loops, which it
/// pushes from both directions.
///
/// # Safety
/// `m` must be a valid, empty, BOOL-typed matrix whose dimensions cover every
/// coordinate in `rows` and `cols`, which must be the same length.
unsafe fn build_bool_pattern(
    m: crate::graph::graphblas::GrB_Matrix,
    rows: &[crate::graph::graphblas::GrB_Index],
    cols: &[crate::graph::graphblas::GrB_Index],
) {
    use crate::graph::graphblas::{
        GrB_BOOL, GrB_Index, GrB_Scalar, GrB_Scalar_free, GrB_Scalar_new,
        GrB_Scalar_setElement_BOOL, GxB_Matrix_build_Scalar,
    };

    debug_assert_eq!(rows.len(), cols.len());
    if rows.is_empty() {
        return;
    }
    let mut scalar: GrB_Scalar = null_mut();
    GrB_Scalar_new(&raw mut scalar, GrB_BOOL);
    GrB_Scalar_setElement_BOOL(scalar, true);
    GxB_Matrix_build_Scalar(
        m,
        rows.as_ptr(),
        cols.as_ptr(),
        scalar,
        rows.len() as GrB_Index,
    );
    GrB_Scalar_free(&raw mut scalar);
}

/// Build compact adjacency directly from relationship tensors.
/// Avoids materializing the full node_cap × node_cap matrix.
/// Returns (compact_matrix_handle, id_to_compact, compact_to_id, n).
/// The caller owns the returned GrB_Matrix and must free it.
unsafe fn build_compact_adj_from_tensors(
    g: &crate::graph::graph::Graph,
    rel_types: &[Arc<String>],
    active: &FxHashSet<u64>,
) -> (
    crate::graph::graphblas::GrB_Matrix,
    FxHashMap<u64, u64>,
    Vec<u64>,
    u64,
) {
    use crate::graph::graphblas::{GrB_BOOL, GrB_Index, GrB_Matrix, GrB_Matrix_new};

    // Build mapping: original_id -> compact_id
    let mut sorted_ids: Vec<u64> = active.iter().copied().collect();
    sorted_ids.sort_unstable();
    let n = sorted_ids.len() as u64;

    let mut id_to_compact: FxHashMap<u64, u64> =
        FxHashMap::with_capacity_and_hasher(sorted_ids.len(), FxBuildHasher);
    for (compact, &orig) in sorted_ids.iter().enumerate() {
        id_to_compact.insert(orig, compact as u64);
    }

    // Iterate directly over tensors, collect edges in compact form
    let mut ri: Vec<GrB_Index> = Vec::new();
    let mut ci: Vec<GrB_Index> = Vec::new();

    if rel_types.is_empty() {
        // No relationship type filter: include all relationship tensors.
        for tensor in g.relationship_tensors() {
            for (src_id, dst_id, _edge_id) in tensor.iter(0, u64::MAX, false) {
                if let (Some(&cr), Some(&cc)) =
                    (id_to_compact.get(&src_id), id_to_compact.get(&dst_id))
                {
                    ri.push(cr);
                    ci.push(cc);
                }
            }
        }
    } else {
        for rel_type in rel_types {
            if let Some(type_id) = g.get_type_id(rel_type) {
                let tensor = &g.relationship_tensors()[usize::from(type_id)];
                for (src_id, dst_id, _edge_id) in tensor.iter(0, u64::MAX, false) {
                    if let (Some(&cr), Some(&cc)) =
                        (id_to_compact.get(&src_id), id_to_compact.get(&dst_id))
                    {
                        ri.push(cr);
                        ci.push(cc);
                    }
                }
            }
        }
    }

    // Build compact matrix in one bulk call
    let mut compact: GrB_Matrix = null_mut();
    GrB_Matrix_new(&raw mut compact, GrB_BOOL, n, n);
    build_bool_pattern(compact, &ri, &ci);

    (compact, id_to_compact, sorted_ids, n)
}

/// Build compact symmetric adjacency directly from relationship tensors (avoids materialization).
/// Returns (compact_matrix_handle, id_to_compact, compact_to_id, n).
/// The caller owns the returned GrB_Matrix and must free it.
unsafe fn build_compact_adj_symmetric_from_tensors(
    g: &crate::graph::graph::Graph,
    rel_types: &[Arc<String>],
    active: &FxHashSet<u64>,
) -> (
    crate::graph::graphblas::GrB_Matrix,
    FxHashMap<u64, u64>,
    Vec<u64>,
    u64,
) {
    use crate::graph::graphblas::{GrB_BOOL, GrB_Index, GrB_Matrix, GrB_Matrix_new};

    // Build mapping: original_id -> compact_id
    let mut sorted_ids: Vec<u64> = active.iter().copied().collect();
    sorted_ids.sort_unstable();
    let n = sorted_ids.len() as u64;

    let mut id_to_compact: FxHashMap<u64, u64> =
        FxHashMap::with_capacity_and_hasher(sorted_ids.len(), FxBuildHasher);
    for (compact, &orig) in sorted_ids.iter().enumerate() {
        id_to_compact.insert(orig, compact as u64);
    }

    // Iterate directly over tensors, collect edges in both directions for symmetry
    let mut ri: Vec<GrB_Index> = Vec::new();
    let mut ci: Vec<GrB_Index> = Vec::new();

    if rel_types.is_empty() {
        // No relationship type filter: include all relationship tensors.
        for tensor in g.relationship_tensors() {
            for (src_id, dst_id, _edge_id) in tensor.iter(0, u64::MAX, false) {
                if let (Some(&cr), Some(&cc)) =
                    (id_to_compact.get(&src_id), id_to_compact.get(&dst_id))
                {
                    ri.push(cr);
                    ci.push(cc);
                    ri.push(cc);
                    ci.push(cr);
                }
            }
        }
    } else {
        for rel_type in rel_types {
            if let Some(type_id) = g.get_type_id(rel_type) {
                let tensor = &g.relationship_tensors()[usize::from(type_id)];
                for (src_id, dst_id, _edge_id) in tensor.iter(0, u64::MAX, false) {
                    if let (Some(&cr), Some(&cc)) =
                        (id_to_compact.get(&src_id), id_to_compact.get(&dst_id))
                    {
                        ri.push(cr);
                        ci.push(cc);
                        ri.push(cc);
                        ci.push(cr);
                    }
                }
            }
        }
    }

    // Build compact matrix in one bulk call
    let mut compact: GrB_Matrix = null_mut();
    GrB_Matrix_new(&raw mut compact, GrB_BOOL, n, n);
    build_bool_pattern(compact, &ri, &ci);

    (compact, id_to_compact, sorted_ids, n)
}

// ── Registration ────────────────────────────────────────────────────────

pub fn register(funcs: &mut Functions) {
    register_pagerank(funcs);
    register_wcc(funcs);
    register_betweenness(funcs);
    register_bfs(funcs);
    register_cdlp(funcs);
    register_msf(funcs);
    register_sp_paths(funcs);
    register_ss_paths(funcs);
    register_harmonic_centrality(funcs);
    register_maxflow(funcs);
}

// ── algo.pageRank ───────────────────────────────────────────────────────

fn register_pagerank(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.pageRank",
        args: [Type::Any, Type::Any],
        ret: Type::Any,
        procedure: ["node", "score"],
        fn algo_pagerank(runtime, args) {
            let label = opt_string(&args[0], "label")?;
            let rel_type = opt_string(&args[1], "relationshipType")?;

            let g = runtime.g.borrow();
            if g.node_count() == 0 {
                return Ok(empty_procedure_batch());
            }

            let rel_types: Vec<Arc<String>> = rel_type.into_iter().collect();

            unsafe {
                use crate::graph::graphblas::{
                    lagraph_bindings, GrB_Matrix, GrB_Matrix_resize, GrB_Vector,
                    GrB_Vector_free,
                };

                // Match C implementation fast path for unfiltered run.
                // If a label is provided but it covers all active nodes, the filtered
                // and unfiltered graphs are equivalent, so skip compact rebuild.
                let use_unfiltered = label
                    .as_ref()
                    .is_none_or(|lbl| g.label_node_count(lbl.as_str()) == g.node_count());

                // Holds the adjacency matrix on the paths that lend it to LAGraph;
                // `None` on the compact path, whose matrix LAGraph owns.
                let mut owned_adj: Option<crate::graph::graphblas::matrix::Matrix<bool>> = None;
                let (lag_adj, compact_to_id): (GrB_Matrix, Option<Vec<u64>>) = if use_unfiltered {
                    // `adj` is freshly built and uniquely owned, so lend its handle to
                    // LAGraph rather than duplicating it; the borrowed delete detaches
                    // `G->A` so only the `Matrix` wrapper frees it.
                    let adj = g.build_adjacency_matrix(&rel_types);
                    let raw_adj: GrB_Matrix = adj.inner();
                    owned_adj = Some(adj);
                    let n = g.node_count() + g.deleted_nodes_count();
                    GrB_Matrix_resize(raw_adj, n, n);
                    (raw_adj, None)
                } else {
                    let active = collect_node_ids(&g, std::slice::from_ref(label.as_ref().unwrap()))
                        .into_iter()
                        .collect();
                    let (lag_adj, _id_to_compact, compact_to_id, _n) =
                        build_compact_adj_from_tensors(&g, &rel_types, &active);
                    (lag_adj, Some(compact_to_id))
                };

                let borrowed = owned_adj.is_some();
                let mut lag_g = if borrowed {
                    create_lagraph_graph_borrowed(lag_adj, LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED)?
                } else {
                    create_lagraph_graph(lag_adj, LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED)?
                };

                // Cache AT and OutDegree (required for PageRank)
                let mut msg = new_msg();
                lagraph_bindings::LAGraph_Cached_AT(lag_g, msg.as_mut_ptr());
                lagraph_bindings::LAGraph_Cached_OutDegree(lag_g, msg.as_mut_ptr());

                // Run PageRank
                let mut centrality: GrB_Vector = null_mut();
                let mut iters: i32 = 0;
                let info = lagraph_bindings::LAGr_PageRank(
                    &raw mut centrality,
                    &raw mut iters,
                    lag_g,
                    0.85,  // damping
                    1e-4,  // tolerance
                    100,   // max iterations
                    msg.as_mut_ptr(),
                );

                if info != 0 {
                    delete_lagraph_graph_maybe_borrowed(&mut lag_g, borrowed);
                    return Err(format!("LAGr_PageRank failed: {info}"));
                }

                // Extract results (compact indices)
                let entries = extract_vector_f64(centrality);

                // Free LAGraph resources
                GrB_Vector_free(&raw mut centrality);
                delete_lagraph_graph_maybe_borrowed(&mut lag_g, borrowed);

                let has_deleted = g.deleted_nodes_count() != 0;
                let mut node_ids = Vec::with_capacity(entries.len());
                let mut scores = Vec::with_capacity(entries.len());
                for (compact_idx, score) in entries {
                    let orig_id = compact_to_id
                        .as_ref()
                        .map_or(compact_idx, |m| m[compact_idx as usize]);
                    if has_deleted && g.is_node_deleted(NodeId::from(orig_id)) {
                        continue;
                    }
                    node_ids.push(NodeId::from(orig_id));
                    scores.push(score);
                }

                Ok(Batch::from_columns([
                    Column::NodeIds(node_ids),
                    Column::Floats(scores),
                ]))
            }
        }
    );
}

// ── algo.WCC ────────────────────────────────────────────────────────────

fn register_wcc(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.WCC",
        args: [Type::Optional(Box::new(Type::Any))],
        ret: Type::Any,
        procedure: ["node", "componentId"],
        fn algo_wcc(runtime, args) {
            let config = parse_config(args)?;
            if !config.is_empty() {
                validate_config_map(&config, &["nodeLabels", "relationshipTypes"])?;
            }
            let node_labels = extract_node_labels(&config)?;
            let rel_types = extract_rel_types(&config)?;

            let g = runtime.g.borrow();
            if g.node_count() == 0 {
                return Ok(empty_procedure_batch());
            }

            unsafe {
                use crate::graph::graphblas::{
                    lagraph_bindings, GrB_Matrix, GrB_Matrix_resize, GrB_Vector,
                    GrB_Vector_free,
                };

                // Match C implementation fast path for unfiltered run.
                // Holds the adjacency matrix on the paths that lend it to LAGraph;
                // `None` on the compact path, whose matrix LAGraph owns.
                let mut owned_adj: Option<crate::graph::graphblas::matrix::Matrix<bool>> = None;
                let (lag_adj, compact_to_id): (GrB_Matrix, Option<Vec<u64>>) = if node_labels.is_empty() {
                    // `adj` is freshly built and uniquely owned, so lend its handle to
                    // LAGraph rather than duplicating it; the borrowed delete detaches
                    // `G->A` so only the `Matrix` wrapper frees it.
                    let adj = g.build_symmetric_adjacency_matrix(&rel_types);
                    let raw_adj: GrB_Matrix = adj.inner();
                    owned_adj = Some(adj);
                    let n = g.node_count() + g.deleted_nodes_count();
                    GrB_Matrix_resize(raw_adj, n, n);
                    (raw_adj, None)
                } else {
                    let a: FxHashSet<u64> =
                        collect_node_ids(&g, &node_labels).into_iter().collect();
                    if a.is_empty() {
                        return Ok(empty_procedure_batch());
                    }
                    let (lag_adj, _id_to_compact, compact_to_id, _n) =
                        build_compact_adj_symmetric_from_tensors(&g, &rel_types, &a);
                    (lag_adj, Some(compact_to_id))
                };

                let borrowed = owned_adj.is_some();
                let mut lag_g = if borrowed {
                    create_lagraph_graph_borrowed(lag_adj, LAGraph_Kind::LAGraph_ADJACENCY_UNDIRECTED)?
                } else {
                    create_lagraph_graph(lag_adj, LAGraph_Kind::LAGraph_ADJACENCY_UNDIRECTED)?
                };

                // Cache symmetric structure
                let mut msg = new_msg();
                let lag_g_ref = lag_g.as_mut().ok_or_else(|| String::from("LAGraph graph pointer is null"))?;
                lag_g_ref.is_symmetric_structure = LAGraph_Boolean::LAGraph_TRUE;

                // Run connected components
                let mut component: GrB_Vector = null_mut();
                let info = lagraph_bindings::LAGr_ConnectedComponents(
                    &raw mut component,
                    lag_g,
                    msg.as_mut_ptr(),
                );

                if info != 0 {
                    delete_lagraph_graph_maybe_borrowed(&mut lag_g, borrowed);
                    return Err(format!("LAGr_ConnectedComponents failed: {info}"));
                }

                let entries = extract_vector_i64(component);

                let has_deleted = g.deleted_nodes_count() != 0;
                let mut node_ids = Vec::with_capacity(entries.len());
                let mut component_ids = Vec::with_capacity(entries.len());
                for (compact_idx, comp_id) in entries {
                    let orig_id = compact_to_id
                        .as_ref()
                        .map_or(compact_idx, |m| m[compact_idx as usize]);
                    if has_deleted && g.is_node_deleted(NodeId::from(orig_id)) {
                        continue;
                    }
                    node_ids.push(NodeId::from(orig_id));
                    component_ids.push(comp_id);
                }

                GrB_Vector_free(&raw mut component);
                delete_lagraph_graph_maybe_borrowed(&mut lag_g, borrowed);

                Ok(Batch::from_columns([
                    Column::NodeIds(node_ids),
                    Column::Ints(component_ids),
                ]))
            }
        }
    );
}

// ── algo.betweenness ────────────────────────────────────────────────────

fn register_betweenness(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.betweenness",
        args: [Type::Optional(Box::new(Type::Any))],
        ret: Type::Any,
        procedure: ["node", "score"],
        fn algo_betweenness(runtime, args) {
            let config = parse_config(args)?;
            if !config.is_empty() {
                validate_config_map(&config, &["nodeLabels", "relationshipTypes", "samplingSize", "samplingSeed"])?;
            }
            let node_labels = extract_node_labels(&config)?;
            let rel_types = extract_rel_types(&config)?;

            // Parse samplingSize
            let sampling_size: usize = match config.get(&Arc::new(String::from("samplingSize"))) {
                None | Some(Value::Null) => 16,
                Some(Value::Int(n)) => {
                    if *n <= 0 {
                        return Err(String::from("samplingSize must be a positive integer"));
                    }
                    *n as i32
                }
                _ => return Err(String::from("samplingSize must be a positive integer")),
            } as usize;

            // Parse samplingSeed
            let sampling_seed: u64 = match config.get(&Arc::new(String::from("samplingSeed"))) {
                None | Some(Value::Null) => 0,
                Some(Value::Int(n)) => *n as u64,
                _ => return Err(String::from("samplingSeed must be an integer")),
            };

            let g = runtime.g.borrow();
            if g.node_count() == 0 {
                return Ok(empty_procedure_batch());
            }

            unsafe {
                use crate::graph::graphblas::{
                    lagraph_bindings, GrB_Matrix, GrB_Matrix_resize, GrB_Vector,
                    GrB_Vector_free,
                };

                // Match C implementation fast path for unfiltered run.
                // Holds the adjacency matrix on the paths that lend it to LAGraph;
                // `None` on the compact path, whose matrix LAGraph owns.
                let mut owned_adj: Option<crate::graph::graphblas::matrix::Matrix<bool>> = None;
                let (compact_adj, compact_to_id): (GrB_Matrix, Option<Vec<u64>>) = if node_labels.is_empty() {
                    // `adj` is freshly built and uniquely owned, so lend its handle to
                    // LAGraph rather than duplicating it; the borrowed delete detaches
                    // `G->A` so only the `Matrix` wrapper frees it.
                    let adj = g.build_adjacency_matrix(&rel_types);
                    let raw_adj: GrB_Matrix = adj.inner();
                    owned_adj = Some(adj);
                    let n = g.node_count() + g.deleted_nodes_count();
                    GrB_Matrix_resize(raw_adj, n, n);
                    (raw_adj, None)
                } else {
                    let node_set: FxHashSet<u64> =
                        collect_node_ids(&g, &node_labels).into_iter().collect();
                    let (compact_adj, _id_to_compact, compact_to_id, _n) =
                        build_compact_adj_from_tensors(&g, &rel_types, &node_set);
                    (compact_adj, Some(compact_to_id))
                };

                let borrowed = owned_adj.is_some();
                let mut lag_g = if borrowed {
                    create_lagraph_graph_borrowed(compact_adj, LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED)?
                } else {
                    create_lagraph_graph(compact_adj, LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED)?
                };

                let mut msg = new_msg();
                lagraph_bindings::LAGraph_Cached_AT(lag_g, msg.as_mut_ptr());
                lagraph_bindings::LAGraph_Cached_OutDegree(lag_g, msg.as_mut_ptr());

                // Select source nodes for sampling (all from compact matrix)
                let n_nodes = compact_to_id.as_ref().map_or_else(
                    || (g.node_count() + g.deleted_nodes_count()) as usize,
                    Vec::len,
                );
                let sources: Vec<u64> = if n_nodes == 0 {
                    vec![]
                } else if sampling_size >= n_nodes {
                    (0..n_nodes as u64).collect()
                } else {
                    let actual_samples = sampling_size;
                    let mut srcs = Vec::with_capacity(actual_samples);
                    let mut rng = sampling_seed;
                    let mut used = FxHashSet::default();
                    for i in 0..actual_samples {
                        let idx = if sampling_seed == 0 {
                            i % n_nodes
                        } else {
                            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                            (rng as usize >> 33) % n_nodes
                        };
                        if used.insert(idx) {
                            srcs.push(idx as u64);
                        }
                    }
                    srcs
                };

                let mut centrality: GrB_Vector = null_mut();
                let info = lagraph_bindings::LAGr_Betweenness(
                    &raw mut centrality,
                    lag_g,
                    sources.as_ptr(),
                    sources.len() as i32,
                    msg.as_mut_ptr(),
                );

                if info != 0 {
                    delete_lagraph_graph_maybe_borrowed(&mut lag_g, borrowed);
                    return Err(format!("LAGr_Betweenness failed: {info}"));
                }

                let entries = extract_vector_f64(centrality);

                GrB_Vector_free(&raw mut centrality);
                delete_lagraph_graph_maybe_borrowed(&mut lag_g, borrowed);

                // All compact indices map to valid nodes (already label-filtered)
                let mut node_ids = Vec::with_capacity(entries.len());
                let mut scores = Vec::with_capacity(entries.len());
                for (compact_idx, score) in entries {
                    let orig_id = compact_to_id
                        .as_ref()
                        .map_or(compact_idx, |m| m[compact_idx as usize]);
                    if g.is_node_deleted(NodeId::from(orig_id)) {
                        continue;
                    }
                    node_ids.push(NodeId::from(orig_id));
                    scores.push(score);
                }

                Ok(Batch::from_columns([
                    Column::NodeIds(node_ids),
                    Column::Floats(scores),
                ]))
            }
        }
    );
}

// ── algo.BFS ────────────────────────────────────────────────────────────

fn register_bfs(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.BFS",
        args: [Type::Any, Type::Any, Type::Any],
        ret: Type::Any,
        procedure: ["nodes", "edges"],
        fn algo_bfs(runtime, args, yields) {
            // arg 0: source node (Node or Null)
            let source_id = match &args[0] {
                Value::Node(id) => *id,
                Value::Null => return Ok(empty_procedure_batch()),
                _ => return Err(String::from("Source must be a node or null")),
            };

            // arg 1: max depth (Int, -1 for unlimited)
            let max_depth = match &args[1] {
                Value::Int(n) => *n,
                _ => return Err(String::from("maxDepth must be an integer")),
            };

            // arg 2: relationship type (String or Null)
            let rel_type = opt_string(&args[2], "relationshipType")?;

            let g = runtime.g.borrow();
            if g.node_count() == 0 {
                return Ok(empty_procedure_batch());
            }
            if g.is_node_deleted(source_id) {
                return Err(String::from("Source node not found in graph"));
            }

            let rel_types: Vec<Arc<String>> = rel_type.into_iter().collect();
            let adj = g.build_adjacency_matrix(&rel_types);

            // Parent tracking is only needed to reconstruct edges; skip it
            // (like the C implementation passes pPI = NULL) when the edges
            // column was not yielded.
            let want_edges = yields & 0b10 != 0;

            unsafe {
                use crate::graph::graphblas::{GrB_Vector, GrB_Vector_free, lagraphx_bindings};

                // Run directly on full adjacency; no compaction needed for BFS.
                // `adj` is freshly built and uniquely owned, so lend its handle
                // to LAGraph instead of duplicating it; the borrowed delete
                // detaches `G->A` so only the `Matrix` wrapper frees it.
                let compact_source = u64::from(source_id);
                let mut lag_g = create_lagraph_graph_borrowed(
                    adj.inner(),
                    LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED,
                )?;

                let mut msg = new_msg();

                let mut level: GrB_Vector = null_mut();
                let mut parent: GrB_Vector = null_mut();

                // Use the extended BFS with max_level
                let max_level = if max_depth < 0 { -1i64 } else { max_depth };
                let info = lagraphx_bindings::LAGr_BreadthFirstSearch_Extended(
                    &raw mut level,
                    if want_edges { &raw mut parent } else { null_mut() },
                    lag_g,
                    compact_source,
                    max_level,
                    -1, // no specific destination
                    false, // many_expected
                    msg.as_mut_ptr(),
                );

                if info != 0 {
                    delete_lagraph_graph_borrowed(&mut lag_g);
                    return Err(format!("LAGr_BreadthFirstSearch_Extended failed: {info}"));
                }

                let has_deleted = g.deleted_nodes_count() != 0;
                let mut nodes: ThinVec<Value>;
                let mut edges: ThinVec<Value> = ThinVec::new();

                if want_edges {
                    // Extract parent vector to reconstruct nodes and edges
                    let parent_entries = extract_vector_i64(parent);
                    nodes = ThinVec::with_capacity(parent_entries.len());
                    edges.reserve(parent_entries.len());

                    for (compact_idx, compact_par) in &parent_entries {
                        if *compact_idx == compact_source {
                            continue; // skip source itself
                        }
                        let orig_child = *compact_idx;
                        let orig_parent = *compact_par as u64;
                        if has_deleted
                            && (g.is_node_deleted(NodeId::from(orig_child))
                                || g.is_node_deleted(NodeId::from(orig_parent)))
                        {
                            continue;
                        }
                        let child = NodeId::from(orig_child);
                        let parent_node = NodeId::from(orig_parent);
                        nodes.push(node_value(child));

                        // Find the relationship from parent to child without
                        // materializing an intermediate Vec per row.
                        if let Some(rel_id) =
                            g.get_src_dest_relationships(parent_node, child, &rel_types).next()
                        {
                            edges.push(Value::Relationship(rel_id));
                        }
                    }
                } else {
                    // No parent vector: reached nodes are the level entries
                    // (level(src) = 0, so skip the source).
                    let level_entries = extract_vector_i64(level);
                    nodes = ThinVec::with_capacity(level_entries.len());

                    for (compact_idx, _lvl) in &level_entries {
                        if *compact_idx == compact_source {
                            continue; // skip source itself
                        }
                        if has_deleted && g.is_node_deleted(NodeId::from(*compact_idx)) {
                            continue;
                        }
                        nodes.push(node_value(NodeId::from(*compact_idx)));
                    }
                }

                if !level.is_null() {
                    GrB_Vector_free(&raw mut level);
                }
                if !parent.is_null() {
                    GrB_Vector_free(&raw mut parent);
                }
                delete_lagraph_graph_borrowed(&mut lag_g);

                // If no nodes were reached, return empty list (no result row)
                if nodes.is_empty() {
                    return Ok(empty_procedure_batch());
                }

                Ok(Batch::from_columns([
                    Column::Values(vec![Value::List(Arc::new(nodes))]),
                    Column::Values(vec![Value::List(Arc::new(edges))]),
                ]))
            }
        }
    );
}

// ── algo.labelPropagation ───────────────────────────────────────────────

fn register_cdlp(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.labelPropagation",
        args: [Type::Optional(Box::new(Type::Any))],
        ret: Type::Any,
        procedure: ["node", "communityId"],
        fn algo_cdlp(runtime, args) {
            let config = parse_config(args)?;
            if !config.is_empty() {
                validate_config_map(&config, &["nodeLabels", "relationshipTypes", "maxIterations"])?;
            }
            let node_labels = extract_node_labels(&config)?;
            let rel_types = extract_rel_types(&config)?;

            // Parse maxIterations
            let max_iterations: i32 = match config.get(&Arc::new(String::from("maxIterations"))) {
                None | Some(Value::Null) => 10,
                Some(Value::Int(n)) => {
                    if *n <= 0 {
                        return Err(String::from("maxIterations must be a positive integer"));
                    }
                    *n as i32
                }
                _ => return Err(String::from("maxIterations must be a positive integer")),
            };

            let g = runtime.g.borrow();
            if g.node_count() == 0 {
                return Ok(empty_procedure_batch());
            }

            unsafe {
                use crate::graph::graphblas::{
                    GrB_Matrix, GrB_Matrix_resize, GrB_Vector, lagraphx_bindings,
                    GrB_Vector_free,
                };

                // Match C implementation fast path for unfiltered run.
                // Holds the adjacency matrix on the paths that lend it to LAGraph;
                // `None` on the compact path, whose matrix LAGraph owns.
                let mut owned_adj: Option<crate::graph::graphblas::matrix::Matrix<bool>> = None;
                let (lag_adj, compact_to_id): (GrB_Matrix, Option<Vec<u64>>) = if node_labels.is_empty() {
                    // `adj` is freshly built and uniquely owned, so lend its handle to
                    // LAGraph rather than duplicating it; the borrowed delete detaches
                    // `G->A` so only the `Matrix` wrapper frees it.
                    let adj = g.build_symmetric_adjacency_matrix(&rel_types);
                    let raw_adj: GrB_Matrix = adj.inner();
                    owned_adj = Some(adj);
                    let n = g.node_count() + g.deleted_nodes_count();
                    GrB_Matrix_resize(raw_adj, n, n);
                    (raw_adj, None)
                } else {
                    let a: FxHashSet<u64> =
                        collect_node_ids(&g, &node_labels).into_iter().collect();
                    if a.is_empty() {
                        return Ok(empty_procedure_batch());
                    }
                    let (lag_adj, _id_to_compact, compact_to_id, _n) =
                        build_compact_adj_symmetric_from_tensors(&g, &rel_types, &a);
                    (lag_adj, Some(compact_to_id))
                };

                let borrowed = owned_adj.is_some();
                let mut lag_g = if borrowed {
                    create_lagraph_graph_borrowed(lag_adj, LAGraph_Kind::LAGraph_ADJACENCY_UNDIRECTED)?
                } else {
                    create_lagraph_graph(lag_adj, LAGraph_Kind::LAGraph_ADJACENCY_UNDIRECTED)?
                };

                let mut msg = new_msg();
                let lag_g_ref = lag_g.as_mut().ok_or_else(|| String::from("LAGraph graph pointer is null"))?;
                lag_g_ref.is_symmetric_structure = LAGraph_Boolean::LAGraph_TRUE;

                let mut cdlp: GrB_Vector = null_mut();
                let info = lagraphx_bindings::LAGraph_cdlp(
                    &raw mut cdlp,
                    lag_g,
                    max_iterations,
                    msg.as_mut_ptr(),
                );

                if info != 0 {
                    delete_lagraph_graph_maybe_borrowed(&mut lag_g, borrowed);
                    return Err(format!("LAGraph_cdlp failed: {info}"));
                }

                let entries = extract_vector_i64(cdlp);

                let has_deleted = g.deleted_nodes_count() != 0;
                let mut node_ids = Vec::with_capacity(entries.len());
                let mut community_ids = Vec::with_capacity(entries.len());
                for (compact_idx, community_id) in entries {
                    let orig_id = compact_to_id
                        .as_ref()
                        .map_or(compact_idx, |m| m[compact_idx as usize]);
                    if has_deleted && g.is_node_deleted(NodeId::from(orig_id)) {
                        continue;
                    }
                    node_ids.push(NodeId::from(orig_id));
                    community_ids.push(community_id);
                }

                GrB_Vector_free(&raw mut cdlp);
                delete_lagraph_graph_maybe_borrowed(&mut lag_g, borrowed);

                Ok(Batch::from_columns([
                    Column::NodeIds(node_ids),
                    Column::Ints(community_ids),
                ]))
            }
        }
    );
}

// ── algo.MSF ────────────────────────────────────────────────────────────

fn register_msf(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.MSF",
        args: [Type::Optional(Box::new(Type::Any))],
        ret: Type::Any,
        procedure: ["nodes", "edges"],
        fn algo_msf(runtime, args) {
            let config = parse_config(args)?;
            if !config.is_empty() {
                validate_config_map(&config, &["nodeLabels", "relationshipTypes", "weightAttribute", "objective"])?;
            }
            let node_labels = extract_node_labels(&config)?;
            let rel_types = extract_rel_types(&config)?;

            // Validate that rel_types exist in the graph (if specified)
            {
                let g = runtime.g.borrow();
                for rt in &rel_types {
                    if g.get_type_id(rt).is_none() {
                        return Err(format!("Relationship type '{rt}' does not exist"));
                    }
                }
            }

            // Parse weightAttribute
            let weight_attr: Option<Arc<String>> = match config.get(&Arc::new(String::from("weightAttribute"))) {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                _ => return Err(String::from("weightAttribute must be a string")),
            };

            // Parse objective (minimize or maximize)
            let maximize = match config.get(&Arc::new(String::from("objective"))) {
                None | Some(Value::Null) => false, // default: minimize
                Some(Value::String(s)) => match s.as_str() {
                    "minimize" => false,
                    "maximize" => true,
                    other => return Err(format!("Invalid objective: '{other}'. Expected 'minimize' or 'maximize'")),
                },
                _ => return Err(String::from("objective must be a string")),
            };

            let g = runtime.g.borrow();

            // Validate weight attribute exists as a relationship attribute
            if let Some(ref attr) = weight_attr
                && g.get_relationship_attribute_id(attr).is_none() {
                    return Err(format!("Weight attribute '{attr}' does not exist"));
                }

            if g.node_count() == 0 {
                return Ok(empty_procedure_batch());
            }

            // Get the set of nodes we care about
            let active_nodes = collect_node_ids(&g, &node_labels);
            if active_nodes.is_empty() {
                return Ok(empty_procedure_batch());
            }

            // Build a weighted FP64 adjacency matrix
            // For unweighted: use 1.0 for all entries
            // For weighted: use the attribute value
            unsafe {
                use crate::graph::graphblas::{GrB_BOOL, GrB_BinaryOp, GrB_BinaryOp_free, GrB_BinaryOp_new, GrB_DESC_SC, GrB_DESC_T1, GrB_Descriptor, GrB_FP64, GrB_Index, GrB_IndexUnaryOp, GrB_IndexUnaryOp_free, GrB_IndexUnaryOp_new, GrB_Info, GrB_Matrix, GrB_Matrix_apply, GrB_Matrix_apply_IndexOp_UDT, GrB_Matrix_build_UDT, GrB_Matrix_eWiseAdd_BinaryOp, GrB_Matrix_extract, GrB_Matrix_extractElement_UDT, GrB_Matrix_extractTuples_FP64, GrB_Matrix_free, GrB_Matrix_ncols, GrB_Matrix_new, GrB_Matrix_nrows, GrB_Matrix_nvals, GrB_Matrix_reduce_Monoid, GrB_Matrix_select_UINT64, GrB_Matrix_wait, GrB_Monoid, GrB_Monoid_free, GrB_Monoid_new_UDT, GrB_IDENTITY_UINT64, GrB_SECOND_UINT64, GrB_Type, GrB_Type_free, GrB_Type_new, GrB_UINT64, GrB_UnaryOp, GrB_UnaryOp_free, GrB_UnaryOp_new, GrB_VALUENE_UINT64, GrB_Vector, GrB_Vector_extractTuples_UDT, GrB_Vector_free, GrB_Vector_new, GrB_Vector_nvals, GrB_WaitMode, lagraphx_bindings};

                let active_set: FxHashSet<u64> = active_nodes.iter().copied().collect();

                // Build compact mapping (sorted original IDs -> 0..n-1)
                let mut sorted_ids: Vec<u64> = active_nodes;
                sorted_ids.sort_unstable();
                let n = sorted_ids.len() as u64;

                // Use Vec-based mapping for O(1) compact ID lookup
                // Map: original_id -> compact_id (for fast lookups)
                let max_id = sorted_ids.last().copied().unwrap_or(0);
                let mut id_to_compact_vec = vec![u64::MAX; (max_id + 1) as usize];
                for (compact, &orig) in sorted_ids.iter().enumerate() {
                    id_to_compact_vec[orig as usize] = compact as u64;
                }

                // Iterate relationships and fill weighted compact matrix.
                // The chosen relationship per node pair is tracked in
                // `pair_to_rel`; no separate edge-id matrix is needed.
                let mut weighted_adj: GrB_Matrix = null_mut();
                GrB_Matrix_new(&raw mut weighted_adj, GrB_FP64, n, n);

                // Chosen relationship per undirected compact pair (min,max) ->
                // relationship id; filled by the O(1) forest-edge recovery below.
                let mut pair_to_rel: FxHashMap<(u64, u64), RelationshipId> =
                    FxHashMap::default();

                // Tensors in play (all, or just the requested relationship types).
                let sel_tensors: Vec<&crate::graph::graphblas::tensor::Tensor> =
                    if rel_types.is_empty() {
                        g.relationship_tensors().iter().collect()
                    } else {
                        rel_types
                            .iter()
                            .filter_map(|rt| g.get_type_id(rt))
                            .map(|tid| &g.relationship_tensors()[tid.0])
                            .collect()
                    };

                // C-parity algebraic build (see FalkorDB C's
                // `get_sub_weight_matrix`): each tensor's effective edge-id
                // matrix is compacted with a GrB extract and scored with one
                // parallel apply straight into `rel_adj` via a min-by-score
                // eWiseAdd — no host-side COO triplets for the main pass.
                // Multi-edge pairs (all their ids stored under compound-key
                // rows in `me`) are monoid-row-reduced to one
                // min-score entry per pair inside GraphBLAS; only the
                // compound-key → (src,dst) index split happens on the host,
                // into these small per-pair COO arrays.
                let mut ov_rows: Vec<GrB_Index> = Vec::new();
                let mut ov_cols: Vec<GrB_Index> = Vec::new();
                let mut ov_vals: Vec<ScoredEdge> = Vec::new();

                // The same parallel index-unary apply serves both paths: weighted
                // reads the attribute; unweighted (`unit`) returns 1.0 — so neither
                // pays a single-threaded host walk of `me`.
                let (attr_idx, unit) = if let Some(attr) = weight_attr.as_ref() {
                    let idx = g
                        .get_relationship_attribute_id(attr)
                        .ok_or_else(|| String::from("Weight attribute does not exist"))?
                        as u16;
                    (idx, false)
                } else {
                    (0u16, true)
                };
                let ctx = MsfWeightCtx {
                    attrs: &raw const *g.relationship_attrs(),
                    attr_idx,
                    maximize,
                    unit,
                };
                let mut ctx_type: GrB_Type = null_mut();
                GrB_Type_new(&raw mut ctx_type, std::mem::size_of::<MsfWeightCtx>());
                // `{score, edge}` UDT, created up front: the inline pass scores
                // straight into it, and it stays alive through the forest-edge
                // recovery at the end.
                let mut scored_edge_type: GrB_Type = null_mut();
                GrB_Type_new(&raw mut scored_edge_type, std::mem::size_of::<ScoredEdge>());
                let mut inline_op: GrB_IndexUnaryOp = null_mut();
                GrB_IndexUnaryOp_new(
                    &raw mut inline_op,
                    Some(msf_scored_edge_value_op),
                    scored_edge_type,
                    GrB_UINT64,
                    ctx_type,
                );
                // Overflow op: scores `me` entries, where the *column index* is
                // the edge id (the bool value is a placeholder).
                let mut ov_op: GrB_IndexUnaryOp = null_mut();
                GrB_IndexUnaryOp_new(
                    &raw mut ov_op,
                    Some(msf_scored_edge_index_op),
                    scored_edge_type,
                    GrB_BOOL,
                    ctx_type,
                );
                // `rel_adj` accumulates every tensor's scored, compacted edges
                // with a min-by-score merge, so each pair resolves to the
                // relationship id of its minimum-score edge across all types.
                let mut min_by_score: GrB_BinaryOp = null_mut();
                GrB_BinaryOp_new(
                    &raw mut min_by_score,
                    Some(msf_keep_min_score),
                    scored_edge_type,
                    scored_edge_type,
                    scored_edge_type,
                );
                // Monoid form of the same op, for the parallel row-reduction of
                // `me`. The identity never wins a comparison: any real edge has
                // `edge < u64::MAX`, so even an `+inf`-scored edge beats it.
                let identity = ScoredEdge {
                    score: f64::INFINITY,
                    edge: u64::MAX,
                };
                let mut min_monoid: GrB_Monoid = null_mut();
                GrB_Monoid_new_UDT(
                    &raw mut min_monoid,
                    min_by_score,
                    std::ptr::from_ref(&identity)
                        .cast_mut()
                        .cast::<std::os::raw::c_void>(),
                );
                let mut rel_adj: GrB_Matrix = null_mut();
                GrB_Matrix_new(&raw mut rel_adj, scored_edge_type, n, n);
                for tensor in &sel_tensors {
                    if tensor.edge_count() == 0 {
                        continue;
                    }

                    // Inline pass: every single-edge pair's edge id is the
                    // UINT64 *value* of the forward matrix `m` (multi-edge
                    // pairs hold the MULTI_EDGE sentinel and are handled by
                    // the multi pass below). Materialize the effective matrix
                    // — (m ∖ dm) ∪ dp, where dp wins on shadowed pairs — then
                    // score every id with one parallel apply directly into
                    // `{score, edge}` pairs and bulk-extract them.
                    tensor.wait();
                    let (fm, fdp, fdm) = (tensor.fwd_m(), tensor.fwd_dp(), tensor.fwd_dm());
                    let mut eff: GrB_Matrix = null_mut();
                    GrB_Matrix_new(&raw mut eff, GrB_UINT64, fm.nrows(), fm.ncols());
                    // eff<¬dm> = m
                    GrB_Matrix_apply(
                        eff,
                        fdm.inner(),
                        null_mut(),
                        GrB_IDENTITY_UINT64,
                        fm.inner(),
                        GrB_DESC_SC,
                    );
                    if fdp.nvals() != 0 {
                        // eff = eff ⊕ dp; SECOND picks dp's value for pairs
                        // where dp shadows m (in-place update).
                        GrB_Matrix_eWiseAdd_BinaryOp(
                            eff,
                            null_mut(),
                            null_mut(),
                            GrB_SECOND_UINT64,
                            eff,
                            fdp.inner(),
                            null_mut(),
                        );
                    }
                    if tensor.has_multi_edge() {
                        // Drop MULTI_EDGE sentinels: those pairs' real ids are
                        // scored by the multi pass below.
                        GrB_Matrix_select_UINT64(
                            eff,
                            null_mut(),
                            null_mut(),
                            GrB_VALUENE_UINT64,
                            eff,
                            u64::MAX,
                            null_mut(),
                        );
                    }
                    let mut eff_nvals: GrB_Index = 0;
                    GrB_Matrix_nvals(&raw mut eff_nvals, eff);
                    if eff_nvals != 0 {
                        // Compact A(I,I) extract: entries incident to inactive
                        // nodes drop out and indices are renumbered to 0..n-1
                        // inside GraphBLAS, replacing the old extractTuples →
                        // host filter/remap → COO build round-trip.
                        let mut eff_c: GrB_Matrix = null_mut();
                        GrB_Matrix_new(&raw mut eff_c, GrB_UINT64, n, n);
                        GrB_Matrix_extract(
                            eff_c,
                            null_mut(),
                            null_mut(),
                            eff,
                            sorted_ids.as_ptr(),
                            n,
                            sorted_ids.as_ptr(),
                            n,
                            null_mut(),
                        );
                        let mut scored_c: GrB_Matrix = null_mut();
                        GrB_Matrix_new(&raw mut scored_c, scored_edge_type, n, n);
                        GrB_Matrix_apply_IndexOp_UDT(
                            scored_c,
                            null_mut(),
                            null_mut(),
                            inline_op,
                            eff_c,
                            (&raw const ctx).cast::<std::os::raw::c_void>(),
                            null_mut(),
                        );
                        GrB_Matrix_eWiseAdd_BinaryOp(
                            rel_adj,
                            null_mut(),
                            null_mut(),
                            min_by_score,
                            rel_adj,
                            scored_c,
                            null_mut(),
                        );
                        GrB_Matrix_free(&raw mut eff_c);
                        GrB_Matrix_free(&raw mut scored_c);
                    }
                    GrB_Matrix_free(&raw mut eff);

                    // Multi pass: every edge id of a multi-edge pair lives in
                    // its block's `me[row][edge_id]`; skip it entirely
                    // on single-edge tensors.
                    if !tensor.has_multi_edge() {
                        continue;
                    }
                    // One `me` matrix per block, and a pair's identifiers live
                    // in exactly one of them, so the winners simply accumulate
                    // across blocks — there is nothing to merge between them.
                    // Every graph within 2^30 nodes per axis has one block.
                    for (block, me) in tensor.edge_versioned_all() {
                        if me.nvals() == 0 {
                            continue;
                        }
                        me.wait();
                        // Two read-only edge sources per tensor: live base edges `m`
                        // (masked by ¬dm to drop deletions) and pending additions `dp`
                        // (disjoint from dm).
                        let sources: [(GrB_Matrix, GrB_Matrix, GrB_Descriptor); 2] = [
                            (me.m().inner(), me.dm().inner(), GrB_DESC_SC),
                            (me.dp().inner(), null_mut(), null_mut()),
                        ];
                        // v[compound_key(src,dst)] = the pair's minimum-score
                        // edge, via a parallel monoid row-reduction instead of a host
                        // iterator walk; the accum merges the two sources.
                        let mut me_rows: GrB_Index = 0;
                        GrB_Matrix_nrows(&raw mut me_rows, me.m().inner());
                        let mut v: GrB_Vector = null_mut();
                        GrB_Vector_new(&raw mut v, scored_edge_type, me_rows);
                        for (src_mat, mask, desc) in sources {
                            let mut src_nvals: GrB_Index = 0;
                            GrB_Matrix_nvals(&raw mut src_nvals, src_mat);
                            if src_nvals == 0 {
                                continue;
                            }
                            let mut n_rows: GrB_Index = 0;
                            let mut n_cols: GrB_Index = 0;
                            GrB_Matrix_nrows(&raw mut n_rows, src_mat);
                            GrB_Matrix_ncols(&raw mut n_cols, src_mat);

                            let mut scored: GrB_Matrix = null_mut();
                            GrB_Matrix_new(&raw mut scored, scored_edge_type, n_rows, n_cols);
                            GrB_Matrix_apply_IndexOp_UDT(
                                scored,
                                mask,
                                null_mut(),
                                ov_op,
                                src_mat,
                                (&raw const ctx).cast::<std::os::raw::c_void>(),
                                desc,
                            );
                            GrB_Matrix_reduce_Monoid(
                                v,
                                null_mut(),
                                min_by_score,
                                min_monoid,
                                scored,
                                null_mut(),
                            );
                            GrB_Matrix_free(&raw mut scored);
                        }

                        // Host step: only the compound-key → (src,dst) index split —
                        // arithmetic GraphBLAS cannot express — one entry per
                        // multi-edge pair, not per edge.
                        let mut v_nvals: GrB_Index = 0;
                        GrB_Vector_nvals(&raw mut v_nvals, v);
                        if v_nvals != 0 {
                            let mut keys: Vec<GrB_Index> = vec![0; v_nvals as usize];
                            let mut vals: Vec<ScoredEdge> =
                                vec![ScoredEdge { score: 0.0, edge: 0 }; v_nvals as usize];
                            let mut nv = v_nvals;
                            GrB_Vector_extractTuples_UDT(
                                keys.as_mut_ptr(),
                                vals.as_mut_ptr().cast::<std::os::raw::c_void>(),
                                &raw mut nv,
                                v,
                            );
                            for (key, se) in keys.iter().zip(&vals) {
                                let (src, dst) = compound_key_inverse(block, *key);
                                let src_original = src as usize;
                                let dst_original = dst as usize;
                                if src_original < id_to_compact_vec.len()
                                    && id_to_compact_vec[src_original] != u64::MAX
                                    && dst_original < id_to_compact_vec.len()
                                    && id_to_compact_vec[dst_original] != u64::MAX
                                {
                                    ov_rows.push(id_to_compact_vec[src_original]);
                                    ov_cols.push(id_to_compact_vec[dst_original]);
                                    ov_vals.push(*se);
                                }
                            }
                        }
                        GrB_Vector_free(&raw mut v);
                    }
                }
                GrB_IndexUnaryOp_free(&raw mut ov_op);
                GrB_Monoid_free(&raw mut min_monoid);
                GrB_IndexUnaryOp_free(&raw mut inline_op);
                GrB_Type_free(&raw mut ctx_type);

                // Merge the multi-edge winners (if any) into `rel_adj`, then
                // symmetrize to the undirected minimum. Boruvka's plain-FP64
                // `weighted_adj` is the score component projected out of
                // `rel_adj` (`msf_score_of`), so the score lives once — in
                // `rel_adj`, which outlives this block for the forest-edge
                // recovery below.
                if !ov_rows.is_empty() {
                    let mut ov_mat: GrB_Matrix = null_mut();
                    GrB_Matrix_new(&raw mut ov_mat, scored_edge_type, n, n);
                    GrB_Matrix_build_UDT(
                        ov_mat,
                        ov_rows.as_ptr(),
                        ov_cols.as_ptr(),
                        ov_vals.as_ptr().cast::<std::os::raw::c_void>(),
                        ov_rows.len() as GrB_Index,
                        min_by_score,
                    );
                    GrB_Matrix_eWiseAdd_BinaryOp(
                        rel_adj,
                        null_mut(),
                        null_mut(),
                        min_by_score,
                        rel_adj,
                        ov_mat,
                        null_mut(),
                    );
                    GrB_Matrix_free(&raw mut ov_mat);
                }
                let mut adj_nvals: GrB_Index = 0;
                GrB_Matrix_nvals(&raw mut adj_nvals, rel_adj);
                if adj_nvals != 0 {
                    GrB_Matrix_eWiseAdd_BinaryOp(
                        rel_adj,
                        null_mut(),
                        null_mut(),
                        min_by_score,
                        rel_adj,
                        rel_adj,
                        GrB_DESC_T1,
                    );
                    let mut score_op: GrB_UnaryOp = null_mut();
                    GrB_UnaryOp_new(&raw mut score_op, Some(msf_score_of), GrB_FP64, scored_edge_type);
                    GrB_Matrix_apply(
                        weighted_adj,
                        null_mut(),
                        null_mut(),
                        score_op,
                        rel_adj,
                        null_mut(),
                    );
                    GrB_UnaryOp_free(&raw mut score_op);
                }
                GrB_BinaryOp_free(&raw mut min_by_score);

                GrB_Matrix_wait(weighted_adj, GrB_WaitMode::GrB_COMPLETE as i32);

                // Run Boruvka MSF
                let mut forest_edges: GrB_Matrix = null_mut();
                let mut component_id: GrB_Vector = null_mut();
                let mut msg = new_msg();

                let info = lagraphx_bindings::LAGraph_msf(
                    &raw mut forest_edges,
                    &raw mut component_id,
                    weighted_adj,
                    false, // input is already compacted and de-duplicated
                    msg.as_mut_ptr(),
                );

                GrB_Matrix_free(&raw mut weighted_adj);

                if info != 0 {
                    return Err(format!("LAGraph_msf failed: {info}"));
                }

                // Extract component assignments (compact indices)
                let comp_entries = extract_vector_i64(component_id);

                // Extract forest edges (compact indices)
                let mut forest_nvals: GrB_Index = 0;
                GrB_Matrix_nvals(&raw mut forest_nvals, forest_edges);
                let mut f_rows = vec![0u64; forest_nvals as usize];
                let mut f_cols = vec![0u64; forest_nvals as usize];
                let mut f_vals = vec![0.0f64; forest_nvals as usize];
                let mut nvals_out = forest_nvals;
                GrB_Matrix_extractTuples_FP64(
                    f_rows.as_mut_ptr(),
                    f_cols.as_mut_ptr(),
                    f_vals.as_mut_ptr(),
                    &raw mut nvals_out,
                    forest_edges,
                );

                GrB_Matrix_free(&raw mut forest_edges);
                GrB_Vector_free(&raw mut component_id);

                // Recover each forest edge's relationship id from `rel_adj` with a
                // single O(1) extract per pair (both orientations resolve, since
                // rel_adj is symmetrized). The min-by-score build already picked the
                // edge the weighted forest selects, so weighted and unweighted share
                // this path — no per-pair tensor scan, no weight re-read. Touching
                // only the ~n forest pairs is leaner than masking the full, ~m-entry
                // UDT `rel_adj` down to the forest and bulk-extracting it (measured
                // neutral-to-slower: the UDT mask sweeps every cell to keep ~n).
                for i in 0..nvals_out as usize {
                    let cs = f_rows[i];
                    let cd = f_cols[i];
                    let (a, b) = if cs <= cd { (cs, cd) } else { (cd, cs) };
                    let mut sr = ScoredEdge { score: 0.0, edge: 0 };
                    if GrB_Matrix_extractElement_UDT(
                        (&raw mut sr).cast::<std::os::raw::c_void>(),
                        rel_adj,
                        cs,
                        cd,
                    ) == GrB_Info::GrB_SUCCESS
                    {
                        pair_to_rel.insert((a, b), RelationshipId::from(sr.edge));
                    }
                }
                GrB_Matrix_free(&raw mut rel_adj);
                GrB_Type_free(&raw mut scored_edge_type);

                // Use Vec for component mapping instead of HashMap (compact indices are 0..n-1)
                let mut compact_to_component_vec = vec![i64::MIN; sorted_ids.len()];
                for (compact_idx, comp) in &comp_entries {
                    compact_to_component_vec[*compact_idx as usize] = *comp;
                }

                // Group nodes by component
                let mut components: std::collections::BTreeMap<i64, Vec<u64>> = std::collections::BTreeMap::new();
                for (compact_idx, comp) in &comp_entries {
                    let orig_id = sorted_ids[*compact_idx as usize];
                    if !active_set.contains(&orig_id) {
                        continue;
                    }
                    components.entry(*comp).or_default().push(orig_id);
                }

                // Add isolated nodes (in active set but not in comp_entries)
                let in_comp: FxHashSet<u64> = comp_entries.iter().map(|(ci, _)| sorted_ids[*ci as usize]).collect();
                for &nid in &sorted_ids {
                    if !in_comp.contains(&nid) && !g.is_node_deleted(NodeId::from(nid)) {
                        components.entry(nid as i64).or_default().push(nid);
                    }
                }

                let mut component_edges: std::collections::BTreeMap<i64, ThinVec<Value>> =
                    std::collections::BTreeMap::new();
                let mut seen_pairs: FxHashSet<(u64, u64)> =
                    FxHashSet::default();

                // Single pass over forest edges; avoid per-component scans.
                for i in 0..nvals_out as usize {
                    let compact_src = f_rows[i];
                    let compact_dst = f_cols[i];
                    let (a, b) = if compact_src <= compact_dst {
                        (compact_src, compact_dst)
                    } else {
                        (compact_dst, compact_src)
                    };
                    if !seen_pairs.insert((a, b)) {
                        continue;
                    }

                    // Use Vec-based lookup: O(1) instead of HashMap O(log n)
                    if compact_src as usize >= compact_to_component_vec.len() || compact_to_component_vec[compact_src as usize] == i64::MIN {
                        continue;
                    }
                    if compact_dst as usize >= compact_to_component_vec.len() || compact_to_component_vec[compact_dst as usize] == i64::MIN {
                        continue;
                    }

                    let component_id = compact_to_component_vec[compact_src as usize];
                    if compact_to_component_vec[compact_dst as usize] != component_id {
                        continue;
                    }

                    let Some(rel_id) = pair_to_rel.get(&(a, b)).copied() else {
                        continue;
                    };
                    component_edges
                        .entry(component_id)
                        .or_default()
                        .push(Value::Relationship(rel_id));
                }

                // Build forest rows from component nodes and collected edges.
                let mut nodes_col = Vec::with_capacity(components.len());
                let mut edges_col = Vec::with_capacity(components.len());
                for (component_id, node_ids) in &components {
                    let tree_nodes: ThinVec<Value> = node_ids
                        .iter()
                        .map(|&nid| node_value(NodeId::from(nid)))
                        .collect();
                    let tree_edges = component_edges.remove(component_id).unwrap_or_default();

                    nodes_col.push(Value::List(Arc::new(tree_nodes)));
                    edges_col.push(Value::List(Arc::new(tree_edges)));
                }

                Ok(Batch::from_columns([
                    Column::Values(nodes_col),
                    Column::Values(edges_col),
                ]))
            }
        }
    );
}

// ── algo.SPpaths / algo.SSpaths ─────────────────────────────────────────

/// Configuration for SPpaths and SSpaths algorithms.
struct PathAlgoConfig {
    source: NodeId,
    target: Option<NodeId>,
    rel_types: Vec<Arc<String>>,
    rel_direction: EdgeDirection,
    max_len: u32,
    weight_prop: Option<Arc<String>>,
    cost_prop: Option<Arc<String>>,
    max_cost: Option<f64>,
    path_count: i64,
}

fn parse_common_path_config(
    config: &OrderMap<Arc<String>, Value>,
    source: NodeId,
    target: Option<NodeId>,
) -> Result<PathAlgoConfig, String> {
    let rel_types = match config.get(&Arc::new(String::from("relTypes"))) {
        None | Some(Value::Null) => vec![],
        Some(Value::List(list)) => {
            let mut types = Vec::with_capacity(list.len());
            for v in list.iter() {
                match v {
                    Value::String(s) => types.push(s.clone()),
                    _ => return Err(String::from("relTypes must be array of strings")),
                }
            }
            types
        }
        _ => return Err(String::from("relTypes must be array of strings")),
    };

    let rel_direction = match config.get(&Arc::new(String::from("relDirection"))) {
        None | Some(Value::Null) => EdgeDirection::Outgoing,
        Some(Value::String(s)) => s.parse().map_err(|()| {
            String::from("relDirection values must be 'incoming', 'outgoing' or 'both'")
        })?,
        _ => {
            return Err(String::from(
                "relDirection values must be 'incoming', 'outgoing' or 'both'",
            ));
        }
    };

    let max_len = match config.get(&Arc::new(String::from("maxLen"))) {
        None | Some(Value::Null) => u32::MAX,
        Some(Value::Int(n)) => {
            if *n < 0 {
                return Err(String::from("maxLen must be non-negative integer"));
            }
            *n as u32
        }
        _ => return Err(String::from("maxLen must be integer")),
    };

    let weight_prop = match config.get(&Arc::new(String::from("weightProp"))) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        _ => return Err(String::from("weightProp must be string")),
    };

    let cost_prop = match config.get(&Arc::new(String::from("costProp"))) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        _ => return Err(String::from("costProp must be string")),
    };

    let max_cost = match config.get(&Arc::new(String::from("maxCost"))) {
        None | Some(Value::Null) => None,
        Some(Value::Int(n)) => Some(*n as f64),
        Some(Value::Float(f)) => Some(*f),
        _ => return Err(String::from("maxCost must be numeric")),
    };

    let path_count = match config.get(&Arc::new(String::from("pathCount"))) {
        None | Some(Value::Null) => 1,
        Some(Value::Int(n)) => {
            if *n < 0 {
                return Err(String::from("pathCount must be a non-negative integer"));
            }
            *n
        }
        _ => return Err(String::from("pathCount must be integer")),
    };

    Ok(PathAlgoConfig {
        source,
        target,
        rel_types,
        rel_direction,
        max_len,
        weight_prop,
        cost_prop,
        max_cost,
        path_count,
    })
}

fn parse_sp_config(args: &[Value]) -> Result<PathAlgoConfig, String> {
    let config = parse_config(args)?;

    let source_key = Arc::new(String::from("sourceNode"));
    let target_key = Arc::new(String::from("targetNode"));

    let source_val = config.get(&source_key);
    let target_val = config.get(&target_key);

    let source_present = matches!(source_val, Some(v) if !matches!(v, Value::Null));
    let target_present = matches!(target_val, Some(v) if !matches!(v, Value::Null));

    if !source_present || !target_present {
        return Err(String::from("sourceNode and targetNode are required"));
    }

    let source = match source_val.unwrap() {
        Value::Node(id) => *id,
        _ => {
            return Err(String::from(
                "sourceNode and targetNode must be of type Node",
            ));
        }
    };
    let target = match target_val.unwrap() {
        Value::Node(id) => *id,
        _ => {
            return Err(String::from(
                "sourceNode and targetNode must be of type Node",
            ));
        }
    };

    parse_common_path_config(&config, source, Some(target))
}

fn parse_ss_config(args: &[Value]) -> Result<PathAlgoConfig, String> {
    let config = parse_config(args)?;

    let source_key = Arc::new(String::from("sourceNode"));
    let source_val = config.get(&source_key);

    let source_present = matches!(source_val, Some(v) if !matches!(v, Value::Null));
    if !source_present {
        return Err(String::from("sourceNode is required"));
    }

    let source = match source_val.unwrap() {
        Value::Node(id) => *id,
        _ => return Err(String::from("sourceNode must be of type Node")),
    };

    parse_common_path_config(&config, source, None)
}

fn to_numeric_value(v: f64) -> Value {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < (i64::MAX as f64) {
        Value::Int(v as i64)
    } else {
        Value::Float(v)
    }
}

/// A path discovered by one of the searches below: its edges in traversal
/// order (each kept in its original direction as
/// `(edge_id, edge_src, edge_dst)`), followed by the path's weight and cost.
type FoundPath = (Vec<(RelationshipId, NodeId, NodeId)>, f64, f64);

/// How often the path searches poll the query deadline and the memory cap.
/// Polling every iteration would mean an `Instant::now()` per edge examined;
/// 1024 keeps that overhead invisible while still aborting a runaway search
/// long before it can exhaust memory.
const PATH_ALGO_CHECK_INTERVAL: u64 = 1024;

/// Order candidate paths the way `algo.SPpaths` / `algo.SSpaths` report them:
/// ascending weight, then ascending cost, then ascending hop count. Mirrors
/// the C engine's `path_cmp`.
fn cmp_found_path(
    a: &FoundPath,
    b: &FoundPath,
) -> Ordering {
    a.1.partial_cmp(&b.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
        .then_with(|| a.0.len().cmp(&b.0.len()))
}

/// Read a numeric edge attribute, falling back to `default` when the property
/// was not configured, is absent from the edge, or is not numeric.
fn edge_numeric_attr(
    g: &Graph,
    edge_id: RelationshipId,
    prop: Option<&Arc<String>>,
    default: f64,
) -> f64 {
    prop.map_or(default, |p| {
        match g.get_relationship_attribute(edge_id, p) {
            Some(Value::Float(f)) => f,
            Some(Value::Int(i)) => i as f64,
            _ => default,
        }
    })
}

/// The endpoint an edge leads to when leaving `from`, or `None` when the edge
/// does not leave `from` in the requested direction.
fn far_endpoint(
    from: NodeId,
    edge_src: NodeId,
    edge_dst: NodeId,
    dir: EdgeDirection,
) -> Option<NodeId> {
    match dir {
        EdgeDirection::Outgoing => (edge_src == from).then_some(edge_dst),
        EdgeDirection::Incoming => (edge_dst == from).then_some(edge_src),
        EdgeDirection::Both => {
            if edge_src == from {
                Some(edge_dst)
            } else if edge_dst == from {
                Some(edge_src)
            } else {
                None
            }
        }
    }
}

/// Every edge leaving `node`, as `(edge_id, edge_src, edge_dst, far_endpoint)`.
///
/// Eager on purpose. Returning the underlying lazy iterator and keeping one
/// suspended per depth reads like the cheaper option — O(depth) iterator
/// positions instead of O(depth x degree) materialised tuples — but measured on
/// pokec-small (max out-degree 426, max both-direction degree 797) it was a
/// wash on memory and ~1.6x slower overall: DFS depth here is small, so the
/// tuples cost tens of KB, while each `next()` re-enters a `flat_map`/`chain`
/// stack over freshly attached GraphBLAS iterators. Collecting once per node
/// visit pays that setup a single time and turns the inner step into a
/// `Vec::pop`.
fn node_successors(
    g: &Graph,
    config: &PathAlgoConfig,
    node: NodeId,
) -> Vec<(RelationshipId, NodeId, NodeId, NodeId)> {
    g.get_node_relationships_by_type(node, &config.rel_types, config.rel_direction)
        .filter_map(|(edge_src, edge_dst, edge_id)| {
            far_endpoint(node, edge_src, edge_dst, config.rel_direction)
                .map(|far| (edge_id, edge_src, edge_dst, far))
        })
        .collect()
}

/// The cheapest known way to reach one node. One of these per *node* reached,
/// which is what bounds Dijkstra's memory to O(V) — as opposed to keeping
/// state per path explored, which is exponential in the hop count.
struct DijkstraLabel {
    parent: NodeId,
    /// Edge reaching this node, in its original direction.
    edge: (RelationshipId, NodeId, NodeId),
    weight: f64,
}

/// A `(weight, node)` candidate waiting to be finalised. Small enough to store
/// by value, so pushing never allocates; superseded entries are left in the
/// heap and skipped at pop time (lazy deletion).
#[derive(PartialEq)]
struct DijkstraItem {
    weight: f64,
    node: NodeId,
}

impl Eq for DijkstraItem {}

impl Ord for DijkstraItem {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        // `BinaryHeap` is a max-heap; reverse the weight comparison so the
        // lightest candidate pops first.
        other
            .weight
            .partial_cmp(&self.weight)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for DijkstraItem {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Textbook Dijkstra from `config.source` to `target`: O((V+E) log V) time and
/// one label per reached node.
///
/// Only sound for the single-cheapest-path shape with no `maxCost` and no
/// `maxLen` — see the dispatch guard in [`run_path_algo`]. Assumes a
/// non-negative `weightProp` on every edge, exactly as the C engine's
/// `Dijkstra_ShortestPath` does: a negative weight breaks the invariant that a
/// node popped from the heap has its final distance.
fn dijkstra_single_path(
    runtime: &Runtime,
    g: &Graph,
    config: &PathAlgoConfig,
    target: NodeId,
) -> Result<Option<FoundPath>, String> {
    let mut labels: FxHashMap<NodeId, DijkstraLabel> = FxHashMap::default();
    let mut settled: FxHashSet<NodeId> = FxHashSet::default();
    let mut heap: BinaryHeap<DijkstraItem> = BinaryHeap::new();

    heap.push(DijkstraItem {
        weight: 0.0,
        node: config.source,
    });

    let mut iters: u64 = 0;
    let mut reached = false;

    while let Some(DijkstraItem { weight, node }) = heap.pop() {
        iters += 1;
        if iters.is_multiple_of(PATH_ALGO_CHECK_INTERVAL) {
            runtime.check_timeout()?;
            runtime.check_mem_capacity()?;
        }

        // a lighter entry for this node was already finalised
        if !settled.insert(node) {
            continue;
        }
        if node == target {
            reached = true;
            break;
        }

        for (edge_src, edge_dst, edge_id) in
            g.get_node_relationships_by_type(node, &config.rel_types, config.rel_direction)
        {
            let Some(far) = far_endpoint(node, edge_src, edge_dst, config.rel_direction) else {
                continue;
            };
            if settled.contains(&far) {
                continue;
            }

            let far_weight =
                weight + edge_numeric_attr(g, edge_id, config.weight_prop.as_ref(), 1.0);
            // A NaN or infinite weight has no meaningful ordering: relaxing on
            // it would settle a label no later comparison can displace, and the
            // procedure would report `nan` as a path weight. Skip the edge.
            if !far_weight.is_finite() {
                continue;
            }
            if labels
                .get(&far)
                .is_none_or(|label| far_weight < label.weight)
            {
                labels.insert(
                    far,
                    DijkstraLabel {
                        parent: node,
                        edge: (edge_id, edge_src, edge_dst),
                        weight: far_weight,
                    },
                );
                heap.push(DijkstraItem {
                    weight: far_weight,
                    node: far,
                });
            }
        }
    }

    if !reached {
        return Ok(None);
    }

    // walk the parent chain back to the source, then flip to traversal order
    let mut edges = Vec::new();
    let mut cur = target;
    while cur != config.source {
        let label = &labels[&cur];
        edges.push(label.edge);
        cur = label.parent;
    }
    edges.reverse();

    // `costProp` is not what the search optimises for, just an attribute the
    // caller asked to have reported: sum it over the winning path only.
    let cost = edges.iter().fold(0.0, |acc, &(edge_id, _, _)| {
        acc + edge_numeric_attr(g, edge_id, config.cost_prop.as_ref(), 0.0)
    });

    Ok(Some((edges, labels[&target].weight, cost)))
}

/// Level-synchronised BFS from `config.source` looking for `target`, honouring
/// `relTypes` / `relDirection` / `maxLen`, keeping one parent record per node.
///
/// Returns the weight and cost of the path it found, or `None` when no path
/// exists within `maxLen`. `None` lets the caller skip enumeration entirely:
/// if no path exists structurally then none can satisfy `maxCost` either, and
/// without this pre-pass an unreachable pair drives the enumeration through
/// every simple path in the reachable component.
fn bfs_find_bound(
    runtime: &Runtime,
    g: &Graph,
    config: &PathAlgoConfig,
    target: NodeId,
) -> Result<Option<(f64, f64)>, String> {
    if config.max_len == 0 {
        return Ok(None);
    }

    let mut parents: FxHashMap<NodeId, (NodeId, RelationshipId)> = FxHashMap::default();
    let mut seen: FxHashSet<NodeId> = FxHashSet::default();
    seen.insert(config.source);

    let mut frontier = vec![config.source];
    let mut next_frontier: Vec<NodeId> = Vec::new();
    let mut iters: u64 = 0;
    let mut found = false;

    'levels: for _hop in 1..=config.max_len {
        if frontier.is_empty() {
            break;
        }
        for &node in &frontier {
            iters += 1;
            if iters.is_multiple_of(PATH_ALGO_CHECK_INTERVAL) {
                runtime.check_timeout()?;
                runtime.check_mem_capacity()?;
            }

            for (edge_src, edge_dst, edge_id) in
                g.get_node_relationships_by_type(node, &config.rel_types, config.rel_direction)
            {
                let Some(far) = far_endpoint(node, edge_src, edge_dst, config.rel_direction) else {
                    continue;
                };
                if !seen.insert(far) {
                    continue;
                }
                parents.insert(far, (node, edge_id));
                if far == target {
                    found = true;
                    break 'levels;
                }
                next_frontier.push(far);
            }
        }
        swap(&mut frontier, &mut next_frontier);
        next_frontier.clear();
    }

    if !found {
        return Ok(None);
    }

    // Collect the parent chain, then fold source->target — the same order
    // `enumerate_paths` accumulates in. Folding along the reverse walk instead
    // would be a different summation order, and float addition is not
    // associative: the seed could land an ULP below the forward sum of the very
    // same edges, and the `next_weight > bound` test would then prune the path
    // this bound came from. With 0.1/0.2/0.3 that is exactly what happens
    // (forward 0.6000000000000001 vs reverse 0.6), losing the only result.
    let mut chain = Vec::new();
    let mut cur = target;
    while cur != config.source {
        let (parent, edge_id) = parents[&cur];
        chain.push(edge_id);
        cur = parent;
    }
    chain.reverse();

    let mut weight = 0.0;
    let mut cost = 0.0;
    for edge_id in chain {
        weight += edge_numeric_attr(g, edge_id, config.weight_prop.as_ref(), 1.0);
        cost += edge_numeric_attr(g, edge_id, config.cost_prop.as_ref(), 0.0);
    }

    Ok(Some((weight, cost)))
}

/// Fold a newly found path into the result set and tighten the weight bound
/// the search prunes with.
fn record_found_path(
    results: &mut Vec<FoundPath>,
    found: FoundPath,
    path_count: i64,
    bound: &mut f64,
) {
    match path_count {
        // the single cheapest path
        1 => {
            if results
                .first()
                .is_none_or(|best| cmp_found_path(&found, best) == Ordering::Less)
            {
                *bound = found.1;
                results.clear();
                results.push(found);
            }
        }
        // every path tied at the minimum weight
        0 => match results.first() {
            Some(best) if found.1 > best.1 => {}
            Some(best) if found.1 < best.1 => {
                *bound = found.1;
                results.clear();
                results.push(found);
            }
            Some(_) => results.push(found),
            None => {
                *bound = found.1;
                results.push(found);
            }
        },
        // the k cheapest paths, which may legitimately span a range of weights.
        // `results` is kept sorted so the worst kept path is always last: once
        // full, a candidate that cannot beat it is rejected in O(1). Sorting the
        // whole vector per found path instead is O(P*k log k) and measurably
        // quadratic in `pathCount` — C keeps a size-k heap for the same reason.
        k => {
            let k = k as usize;
            if results.len() == k {
                if cmp_found_path(&found, &results[k - 1]) != Ordering::Less {
                    return;
                }
                results.pop();
            }
            let pos =
                results.partition_point(|kept| cmp_found_path(kept, &found) == Ordering::Less);
            results.insert(pos, found);
            if results.len() == k {
                *bound = results[k - 1].1;
            }
        }
    }
}

/// Enumerate simple paths from `config.source`, depth-first with backtracking.
///
/// Only ever holds one live path plus the untried successors at each of its
/// depths, so memory is O(maxLen x degree) regardless of how many paths get
/// explored. (A best-first search over paths, by contrast, must hold every
/// frontier path at once — exponential in the hop count.)
///
/// `bound` is an upper limit on path weight, tightened as better results are
/// found; `maxCost` prunes independently since cost only ever increases.
fn enumerate_paths(
    runtime: &Runtime,
    g: &Graph,
    config: &PathAlgoConfig,
    initial_bound: f64,
) -> Result<Vec<FoundPath>, String> {
    // untried successors at each depth; index `d` belongs to the node reached
    // after `d` edges. `None` is a depth we deliberately do not expand from
    // (maxLen reached, or an SPpaths target), which reads the same as exhausted.
    let mut levels = Vec::new();
    // the single live path: nodes visited, and the edges between them with
    // their weight and cost so backtracking is a subtraction rather than
    // another attribute lookup
    let mut nodes: Vec<NodeId> = vec![config.source];
    // Each entry carries the *prefix* weight and cost of the path up to and
    // including that edge, not the edge's own values. Restoring the running
    // totals on backtrack is then a pop rather than a subtraction: `w + x - x`
    // does not round-trip in f64, so subtracting let a dead-end sibling branch
    // leave the accumulator an ULP high and prune the real path on the way back
    // down. Folding each prefix from its parent's makes a given edge sequence
    // always sum to the same bits, whatever order the DFS reached it in.
    let mut edges: Vec<(RelationshipId, NodeId, NodeId, f64, f64)> = Vec::new();
    let mut on_path: FxHashSet<NodeId> = FxHashSet::default();
    on_path.insert(config.source);

    levels.push((config.max_len > 0).then(|| node_successors(g, config, config.source)));

    let mut results: Vec<FoundPath> = Vec::new();
    let mut bound = initial_bound;
    let mut iters: u64 = 0;

    loop {
        iters += 1;
        if iters.is_multiple_of(PATH_ALGO_CHECK_INTERVAL) {
            runtime.check_timeout()?;
            runtime.check_mem_capacity()?;
        }

        let depth = edges.len();
        let advance = levels[depth].as_mut().and_then(Vec::pop);
        if let Some((edge_id, edge_src, edge_dst, next)) = advance {
            // simple paths only: a node may appear at most once
            if on_path.contains(&next) {
                continue;
            }

            let (weight, cost) = edges.last().map_or((0.0, 0.0), |&(_, _, _, w, c)| (w, c));
            let next_weight =
                weight + edge_numeric_attr(g, edge_id, config.weight_prop.as_ref(), 1.0);
            let next_cost = cost + edge_numeric_attr(g, edge_id, config.cost_prop.as_ref(), 0.0);

            // C gates on `weight + w <= max_weight` (max_weight defaulting to
            // DBL_MAX), which rejects NaN and infinity. The negated `> bound`
            // form admits both, and a non-finite prefix then propagates into
            // every path explored below this one — reporting finite paths as
            // `nan`. Test finiteness explicitly so the two agree.
            if !next_weight.is_finite() || next_weight > bound {
                continue;
            }
            if !next_cost.is_finite() || config.max_cost.is_some_and(|max| next_cost > max) {
                continue;
            }

            edges.push((edge_id, edge_src, edge_dst, next_weight, next_cost));
            nodes.push(next);
            on_path.insert(next);

            // SSpaths (no target) reports every non-empty path; SPpaths only
            // those ending at the target
            let at_target = config.target.is_none_or(|t| next == t);
            if at_target {
                let found = (
                    edges
                        .iter()
                        .map(|&(id, src, dst, _, _)| (id, src, dst))
                        .collect(),
                    next_weight,
                    next_cost,
                );
                record_found_path(&mut results, found, config.path_count, &mut bound);
            }

            // No point extending past an SPpaths target: it is on the path
            // now, so no longer path can end there.
            let expand = !(at_target && config.target.is_some())
                && u32::try_from(edges.len()).is_ok_and(|len| len < config.max_len);
            levels.push(expand.then(|| node_successors(g, config, next)));
        } else {
            // this depth is exhausted; step back up
            if depth == 0 {
                break;
            }
            levels.pop();
            edges.pop().expect("depth > 0 implies a live edge");
            let left = nodes.pop().expect("depth > 0 implies a live node");
            on_path.remove(&left);
        }
    }

    Ok(results)
}

/// Build the three-column procedure result: `path`, `pathWeight`, `pathCost`.
fn build_path_batch(
    config: &PathAlgoConfig,
    results: Vec<FoundPath>,
) -> super::ProcedureBatch {
    let mut paths = Vec::with_capacity(results.len());
    let mut path_weights = Vec::with_capacity(results.len());
    let mut path_costs = Vec::with_capacity(results.len());

    for (edges, weight, cost) in results {
        let mut path_elems = ThinVec::with_capacity(edges.len() * 2 + 1);
        path_elems.push(Value::Node(config.source));
        let mut prev = config.source;
        for (edge_id, edge_src, edge_dst) in edges {
            path_elems.push(Value::Relationship(edge_id));
            let next = if prev == edge_src { edge_dst } else { edge_src };
            path_elems.push(Value::Node(next));
            prev = next;
        }

        paths.push(Value::Path(Arc::new(path_elems)));
        path_weights.push(to_numeric_value(weight));
        path_costs.push(to_numeric_value(cost));
    }

    Batch::from_columns([
        Column::Values(paths),
        classify_stored_column(path_weights),
        classify_stored_column(path_costs),
    ])
}

fn run_path_algo(
    runtime: &Runtime,
    config: &PathAlgoConfig,
) -> Result<super::ProcedureBatch, String> {
    let g = runtime.g.borrow();

    // Fast path: one cheapest path, no cost cap, no hop cap — exactly what
    // Dijkstra solves, in O((V+E) log V) time and O(V) memory. The general
    // enumeration below has to consider every simple path, which is
    // exponential in the hop count even when it prunes well.
    //
    // `source == target` is excluded on purpose: Dijkstra would "reach" the
    // source at distance zero having traversed no edges, but these procedures
    // only report paths of at least one edge. Let the enumeration handle that
    // degenerate case, where it correctly finds nothing.
    if let Some(target) = config.target
        && config.path_count == 1
        && config.max_cost.is_none()
        && config.max_len == u32::MAX
        && config.source != target
    {
        let found = dijkstra_single_path(runtime, &g, config, target)?;
        return Ok(build_path_batch(config, found.into_iter().collect()));
    }

    // Otherwise, first ask whether any path honouring relTypes/relDirection/
    // maxLen exists at all. An unreachable pair then costs one linear sweep
    // instead of walking every simple path in the component, and the weight
    // this finds seeds the enumeration's pruning bound.
    let mut bound = f64::INFINITY;
    if let Some(target) = config.target {
        let Some((bound_weight, bound_cost)) = bfs_find_bound(runtime, &g, config, target)? else {
            return Ok(build_path_batch(config, Vec::new()));
        };

        // k-minimal has to fill k candidates before a weight bound means
        // anything, and those can legitimately be heavier than this one.
        if (config.path_count == 0 || config.path_count == 1)
            && config
                .max_cost
                .is_none_or(|max_cost| bound_cost <= max_cost)
        {
            bound = bound_weight;
        }
    }

    let mut results = enumerate_paths(runtime, &g, config, bound)?;
    results.sort_by(cmp_found_path);
    Ok(build_path_batch(config, results))
}

fn register_sp_paths(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.SPpaths",
        args: [Type::Any],
        ret: Type::Any,
        procedure: ["path", "pathWeight", "pathCost"],
        fn algo_sp_paths(runtime, args) {
            let config = parse_sp_config(args)?;
            run_path_algo(runtime, &config)
        }
    );
}

fn register_ss_paths(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.SSpaths",
        args: [Type::Any],
        ret: Type::Any,
        procedure: ["path", "pathWeight", "pathCost"],
        fn algo_ss_paths(runtime, args) {
            let config = parse_ss_config(args)?;
            run_path_algo(runtime, &config)
        }
    );
}

// ── algo.HarmonicCentrality ─────────────────────────────────────────────

fn register_harmonic_centrality(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.HarmonicCentrality",
        args: [Type::Optional(Box::new(Type::Any))],
        ret: Type::Any,
        procedure: ["node", "score", "reachable"],
        fn algo_harmonic_centrality(runtime, args) {
            use crate::graph::graphblas::{
                lagraphx_bindings,
                GrB_ALL,
                GrB_BOOL,
                GrB_Vector,
                GrB_Vector_assign_BOOL,
                GrB_Vector_free,
                GrB_Vector_new,
            };

            let config = parse_config(args)?;
            if !config.is_empty() {
                validate_config_map(&config, &["nodeLabels", "relationshipTypes"])?;
            }
            let node_labels = extract_node_labels(&config)?;
            let rel_types = extract_rel_types(&config)?;

            let g = runtime.g.borrow();
            for rt in &rel_types {
                if g.get_type_id(rt).is_none() {
                    return Err(format!("Relationship type '{rt}' does not exist"));
                }
            }

            if g.node_count() == 0 {
                return Ok(empty_procedure_batch());
            }

            let node_set: FxHashSet<u64> = if node_labels.is_empty() {
                active_node_set(&g)
            } else {
                collect_node_ids(&g, &node_labels).into_iter().collect()
            };

            if node_set.is_empty() {
                return Ok(empty_procedure_batch());
            }

            unsafe {
                use crate::graph::graphblas::{
                    GrB_Matrix, GrB_Matrix_eWiseMult_BinaryOp, GrB_Matrix_ncols,
                    GrB_Matrix_new, GrB_Matrix_nrows, GrB_ONEB_BOOL,
                };

                // Match C implementation fast path for unfiltered run.
                let (compact_adj, compact_to_id): (GrB_Matrix, Option<Vec<u64>>) = if node_labels.is_empty() {
                    let adj = g.build_adjacency_matrix(&rel_types);
                    // Hand LAGraph an ISO boolean matrix (a single shared `true`
                    // value), exactly as the C module's Delta_Matrix_export does via
                    // GrB_ONEB_BOOL. The generic HyperBall mxv inside
                    // LAGr_HarmonicCentrality only takes GraphBLAS's fast dot4 path
                    // when the adjacency is iso; a plain non-iso dup makes it punt to
                    // the ~3x slower generic dot2. eWiseMult(adj, adj, ONEB) rebuilds
                    // the same pattern with every value collapsed to iso `true`.
                    let mut nrows: u64 = 0;
                    let mut ncols: u64 = 0;
                    GrB_Matrix_nrows(&raw mut nrows, adj.inner());
                    GrB_Matrix_ncols(&raw mut ncols, adj.inner());
                    let mut raw_adj: GrB_Matrix = std::ptr::null_mut();
                    GrB_Matrix_new(&raw mut raw_adj, GrB_BOOL, nrows, ncols);
                    GrB_Matrix_eWiseMult_BinaryOp(
                        raw_adj,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        GrB_ONEB_BOOL,
                        adj.inner(),
                        adj.inner(),
                        std::ptr::null_mut(),
                    );
                    let n = g.node_count() + g.deleted_nodes_count();
                    crate::graph::graphblas::GrB_Matrix_resize(raw_adj, n, n);
                    (raw_adj, None)
                } else {
                    let (compact_adj, _id_to_compact, compact_to_id, _n) =
                        build_compact_adj_from_tensors(&g, &rel_types, &node_set);
                    (compact_adj, Some(compact_to_id))
                };

                let mut lag_g = create_lagraph_graph(
                    compact_adj,
                    LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED,
                )?;

                let mut msg = new_msg();
                // LAGr_HarmonicCentrality only reads G->A (it builds its own compact
                // submatrix internally); it never uses G->AT or G->out_degree, so we
                // skip caching them here, matching the C module which also does not.

                runtime.check_timeout()?;
                let mut nodes: GrB_Vector = null_mut();
                let node_vec_len = compact_to_id
                    .as_ref()
                    .map_or_else(|| g.node_count(), |m| m.len() as u64);
                GrB_Vector_new(&raw mut nodes, GrB_BOOL, node_vec_len);
                GrB_Vector_assign_BOOL(
                    nodes,
                    null_mut(),
                    null_mut(),
                    true,
                    GrB_ALL,
                    node_vec_len,
                    null_mut(),
                );

                let mut scores: GrB_Vector = null_mut();
                let mut reachable_nodes: GrB_Vector = null_mut();
                let info = lagraphx_bindings::LAGr_HarmonicCentrality(
                    &raw mut scores,
                    &raw mut reachable_nodes,
                    lag_g,
                    nodes,
                    msg.as_mut_ptr(),
                );

                GrB_Vector_free(&raw mut nodes);
                if info != 0 {
                    delete_lagraph_graph(&mut lag_g);
                    return Err(format!("LAGr_HarmonicCentrality failed: {info}"));
                }

                let score_entries = extract_vector_f64(scores);
                let mut reachable_by_idx = vec![0_i64; node_vec_len as usize];
                for (compact_idx, reachable) in extract_vector_i64(reachable_nodes) {
                    reachable_by_idx[compact_idx as usize] = reachable;
                }

                let mut node_ids = Vec::with_capacity(score_entries.len());
                let mut scores_col = Vec::with_capacity(score_entries.len());
                let mut reachable_col = Vec::with_capacity(score_entries.len());
                for (compact_idx, score) in score_entries {
                    let orig_id = compact_to_id
                        .as_ref()
                        .map_or(compact_idx, |m| m[compact_idx as usize]);
                    if g.is_node_deleted(NodeId::from(orig_id)) {
                        continue;
                    }
                    node_ids.push(NodeId::from(orig_id));
                    scores_col.push(score);
                    reachable_col.push(reachable_by_idx[compact_idx as usize]);
                }

                GrB_Vector_free(&raw mut scores);
                GrB_Vector_free(&raw mut reachable_nodes);
                delete_lagraph_graph(&mut lag_g);

                Ok(Batch::from_columns([
                    Column::NodeIds(node_ids),
                    Column::Floats(scores_col),
                    Column::Ints(reachable_col),
                ]))
            }
        }
    );
}

// ── algo.maxFlow ────────────────────────────────────────────────────────

fn register_maxflow(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.maxFlow",
        args: [Type::Any],
        ret: Type::Any,
        procedure: ["nodes", "edges", "edgeFlows", "maxFlow"],
        fn algo_maxflow(runtime, args) {

            let config = match &args[0] {
                Value::Map(m) => (**m).clone(),
                _ => return Err(String::from(
                    "algo.maxFlow expects a single configuration map",
                )),
            };

            validate_config_map(&config, &[
                "sourceNodes", "targetNodes", "relationshipTypes",
                "capacityProperty", "nodeLabels", "defaultCapacity",
            ])?;

            let node_labels = extract_node_labels(&config)?;
            let rel_types = extract_rel_types(&config)?;

            if rel_types.len() != 1 {
                return Err(String::from(
                    "algo.maxFlow: 'relationshipTypes' is required and must contain exactly one type",
                ));
            }

            let parse_nodes = |key: &str| -> Result<Vec<NodeId>, String> {
                match config.get(&Arc::new(String::from(key))) {
                    Some(Value::List(list)) => {
                        let mut nodes = Vec::with_capacity(list.len());
                        for v in list.iter() {
                            match v {
                                Value::Node(id) => nodes.push(*id),
                                _ => return Err(format!(
                                    "algo.maxFlow: '{key}' must be an array of nodes",
                                )),
                            }
                        }
                        Ok(nodes)
                    }
                    _ => Err(format!(
                        "algo.maxFlow: '{key}' must be an array of nodes",
                    )),
                }
            };

            let srcs = parse_nodes("sourceNodes")?;
            let sinks = parse_nodes("targetNodes")?;

            if srcs.is_empty() || sinks.is_empty() {
                return Err(String::from(
                    "algo.maxFlow: expects at least one source and one sink",
                ));
            }

            let src_set: FxHashSet<u64> =
                srcs.iter().map(|n| u64::from(*n)).collect();
            if sinks.iter().any(|s| src_set.contains(&u64::from(*s))) {
                return Err(String::from(
                    "algo.maxFlow: source and sink sets must be disjoint",
                ));
            }

            let capacity_attr: Arc<String> = match config
                .get(&Arc::new(String::from("capacityProperty")))
            {
                Some(Value::String(s)) => s.clone(),
                _ => return Err(String::from(
                    "algo.maxFlow: 'capacityProperty' is required and must be a string",
                )),
            };

            let default_cap: Option<f64> = match config
                .get(&Arc::new(String::from("defaultCapacity")))
            {
                None | Some(Value::Null) => None,
                Some(Value::Float(f)) if *f >= 0.0 => Some(*f),
                #[allow(clippy::cast_precision_loss)]
                Some(Value::Int(i)) if *i >= 0 => Some(*i as f64),
                _ => return Err(String::from(
                    "algo.maxFlow: 'defaultCapacity' must be a non-negative number",
                )),
            };

            let g = runtime.g.borrow();

            let Some(type_id) = g.get_type_id(&rel_types[0]) else {
                return Err(format!(
                    "algo.maxFlow: relationship type '{}' does not exist",
                    rel_types[0],
                ));
            };
            if g.relationship_tensors()[type_id.0].has_multi_edge() {
                return Err(format!(
                    "algo.maxFlow: relationship type '{}' contains multi-edges; maxFlow requires a simple adjacency matrix",
                    rel_types[0],
                ));
            }
            let capacity_attr_idx = g
                .get_relationship_attribute_id(&capacity_attr)
                .map(|idx| idx as u16);
            if capacity_attr_idx.is_none() && default_cap.is_none() {
                return Err(String::from(
                    "algo.maxFlow: invalid or missing attribute and no default attribute specified",
                ));
            }

            let label_filter: Option<FxHashSet<u64>> =
                if node_labels.is_empty() {
                    None
                } else {
                    Some(collect_node_ids(&g, &node_labels).into_iter().collect())
                };

            let tensor = &g.relationship_tensors()[type_id.0];
            // Stream every edge via `iter_edges`, which walks the inline UINT64
            // forward matrix directly (the multi-edge matrix `me` is empty —
            // maxFlow rejects multi-edge tensors above), rather than iterating
            // pair-by-pair or bulk-collecting into throwaway vectors.
            let edge_upper = tensor.edge_count() as usize;
            let mut rel_pairs: Vec<(u64, u64)> = Vec::with_capacity(edge_upper);
            let mut rel_ids: Vec<RelationshipId> = Vec::with_capacity(edge_upper);
            for (src, dst, edge_id) in tensor.iter_edges() {
                if let Some(ref filter) = label_filter
                    && (!filter.contains(&src) || !filter.contains(&dst))
                {
                    continue;
                }
                rel_pairs.push((src, dst));
                rel_ids.push(RelationshipId::from(edge_id));
            }

            let mut attr_values = Vec::with_capacity(rel_ids.len());
            if let Some(attr_idx) = capacity_attr_idx {
                g.get_relationship_attributes_by_idx(&rel_ids, attr_idx, &Value::Null, &mut attr_values);
            } else {
                attr_values.resize(rel_ids.len(), Value::Null);
            }

            let mut edges_with_cap: Vec<(u64, u64, RelationshipId, f64)> =
                Vec::with_capacity(rel_ids.len());
            for ((src_dst, rel_id), value) in rel_pairs.into_iter().zip(rel_ids).zip(attr_values) {
                let (s, d) = src_dst;
                let parsed = match value {
                    Value::Float(f) if f >= 0.0 => Some(f),
                    #[allow(clippy::cast_precision_loss)]
                    Value::Int(i) if i >= 0 => Some(i as f64),
                    _ => None,
                };
                let Some(cap_value) = parsed.or(default_cap) else {
                    return Err(String::from(
                        "algo.maxFlow: invalid or missing attribute and no default attribute specified",
                    ));
                };
                edges_with_cap.push((s, d, rel_id, cap_value));
            }

            if edges_with_cap.is_empty() {
                return Ok(Batch::from_columns([
                    Column::Values(vec![Value::List(Arc::new(thin_vec![]))]),
                    Column::Values(vec![Value::List(Arc::new(thin_vec![]))]),
                    Column::Values(vec![Value::List(Arc::new(thin_vec![]))]),
                    Column::Floats(vec![0.0]),
                ]));
            }

            let multi_srcs = srcs.len() > 1;
            let multi_sinks = sinks.len() > 1;

            // Fast path: when node IDs are dense and there is no label filter,
            // avoid per-call sort/dedup/hash remapping and use original IDs.
            let use_identity_compaction = label_filter.is_none() && g.deleted_nodes_count() == 0;

            let (compact_edges, edge_meta, original_edge_count, min_cap, max_cap, src_id, sink_id, total_nodes) =
                if use_identity_compaction {
                    let base_dim = (g.max_node_id() + 1) as usize;
                    let super_src = base_dim;
                    let super_sink = base_dim + usize::from(multi_srcs);
                    let total_nodes = base_dim + usize::from(multi_srcs) + usize::from(multi_sinks);

                    let mut compact_edges: Vec<(usize, usize, f64)> = Vec::with_capacity(
                        edges_with_cap.len() + srcs.len() + sinks.len(),
                    );
                    let mut edge_meta: Vec<(u64, u64, RelationshipId)> =
                        Vec::with_capacity(edges_with_cap.len());
                    let mut original_edge_count = 0usize;

                    let mut min_cap = f64::INFINITY;
                    let mut max_cap = 0.0_f64;
                    for (s, d, rel_id, c) in &edges_with_cap {
                        if *c <= 0.0 {
                            continue;
                        }
                        compact_edges.push((*s as usize, *d as usize, *c));
                        edge_meta.push((*s, *d, *rel_id));
                        original_edge_count += 1;
                        if *c < min_cap {
                            min_cap = *c;
                        }
                        if *c > max_cap {
                            max_cap = *c;
                        }
                    }

                    let mut src_id = u64::from(srcs[0]) as usize;
                    let mut sink_id = u64::from(sinks[0]) as usize;
                    #[allow(clippy::cast_lossless)]
                    let big_cap: f64 = i32::MAX as f64;
                    if multi_srcs {
                        src_id = super_src;
                        for s in &srcs {
                            compact_edges.push((src_id, u64::from(*s) as usize, big_cap));
                        }
                    }
                    if multi_sinks {
                        sink_id = super_sink;
                        for s in &sinks {
                            compact_edges.push((u64::from(*s) as usize, sink_id, big_cap));
                        }
                    }

                    (
                        compact_edges,
                        edge_meta,
                        original_edge_count,
                        min_cap,
                        max_cap,
                        src_id,
                        sink_id,
                        total_nodes,
                    )
                } else {
                    // Re-index onto a dense node ID space to avoid huge sparse
                    // dimensions from tombstoned node IDs.
                    let mut compact_to_id: Vec<u64> = srcs.iter().map(|n| u64::from(*n)).collect();
                    compact_to_id.extend(sinks.iter().map(|n| u64::from(*n)));
                    compact_to_id.extend(edges_with_cap.iter().flat_map(|(s, d, _, _)| [*s, *d]));
                    compact_to_id.sort_unstable();
                    compact_to_id.dedup();

                    let mut id_to_compact: FxHashMap<u64, usize> =
                        FxHashMap::with_capacity_and_hasher(compact_to_id.len(), FxBuildHasher);
                    for (compact_id, orig_id) in compact_to_id.iter().copied().enumerate() {
                        id_to_compact.insert(orig_id, compact_id);
                    }

                    let base_dim = compact_to_id.len();
                    let super_src = base_dim;
                    let super_sink = base_dim + usize::from(multi_srcs);
                    let total_nodes = base_dim + usize::from(multi_srcs) + usize::from(multi_sinks);

                    let mut compact_edges: Vec<(usize, usize, f64)> = Vec::with_capacity(
                        edges_with_cap.len() + srcs.len() + sinks.len(),
                    );
                    let mut edge_meta: Vec<(u64, u64, RelationshipId)> =
                        Vec::with_capacity(edges_with_cap.len());
                    let mut original_edge_count = 0usize;

                    let mut min_cap = f64::INFINITY;
                    let mut max_cap = 0.0_f64;
                    for (s, d, rel_id, c) in &edges_with_cap {
                        if *c <= 0.0 {
                            continue;
                        }
                        if let (Some(&cs), Some(&cd)) = (id_to_compact.get(s), id_to_compact.get(d)) {
                            compact_edges.push((cs, cd, *c));
                            edge_meta.push((*s, *d, *rel_id));
                            original_edge_count += 1;
                            if *c < min_cap {
                                min_cap = *c;
                            }
                            if *c > max_cap {
                                max_cap = *c;
                            }
                        }
                    }

                    let mut src_id = *id_to_compact
                        .get(&u64::from(srcs[0]))
                        .ok_or_else(|| String::from("algo.maxFlow: source node is not in graph"))?;
                    let mut sink_id = *id_to_compact
                        .get(&u64::from(sinks[0]))
                        .ok_or_else(|| String::from("algo.maxFlow: sink node is not in graph"))?;
                    #[allow(clippy::cast_lossless)]
                    let big_cap: f64 = i32::MAX as f64;
                    if multi_srcs {
                        src_id = super_src;
                        for s in &srcs {
                            if let Some(&cs) = id_to_compact.get(&u64::from(*s)) {
                                compact_edges.push((src_id, cs, big_cap));
                            }
                        }
                    }
                    if multi_sinks {
                        sink_id = super_sink;
                        for s in &sinks {
                            if let Some(&cs) = id_to_compact.get(&u64::from(*s)) {
                                compact_edges.push((cs, sink_id, big_cap));
                            }
                        }
                    }

                    (
                        compact_edges,
                        edge_meta,
                        original_edge_count,
                        min_cap,
                        max_cap,
                        src_id,
                        sink_id,
                        total_nodes,
                    )
                };

            #[allow(clippy::cast_lossless)]
            let overflow_factor = u32::MAX as f64;
            if min_cap.is_finite() && max_cap >= min_cap * overflow_factor {
                return Err(String::from(
                    "algo.maxFlow: capacity range too wide (max >= min * 2^32); narrow the capacity values to avoid internal overflow",
                ));
            }

            let (max_flow, flows) = unsafe {
                use crate::graph::graphblas::{
                    GrB_FP64, GrB_Index, GrB_MAX_FP64, GrB_Matrix, GrB_Matrix_build_FP64,
                    GrB_Matrix_extractTuples_FP64, GrB_Matrix_free, GrB_Matrix_new,
                    GrB_Matrix_nvals, GrB_Matrix_wait, GrB_WaitMode, lagraph_bindings,
                    lagraphx_bindings,
                };

                let mut cap_mtx: GrB_Matrix = null_mut();
                GrB_Matrix_new(&raw mut cap_mtx, GrB_FP64, total_nodes as u64, total_nodes as u64);
                // Bulk-build the capacity matrix in a single GrB_Matrix_build
                // instead of a per-edge GrB_Matrix_setElement loop (which pays
                // FFI + pending-tuple bookkeeping and periodic reallocation per
                // entry). Every (u, v) here is unique — maxFlow rejects
                // multi-edge tensors above and the super-source/sink nodes get
                // fresh ids — so the GrB_MAX_FP64 dup operator never merges
                // distinct capacities; it is only a required argument and
                // collapses the identical-capacity super-edges safely.
                let mut b_rows: Vec<GrB_Index> = Vec::with_capacity(compact_edges.len());
                let mut b_cols: Vec<GrB_Index> = Vec::with_capacity(compact_edges.len());
                let mut b_vals: Vec<f64> = Vec::with_capacity(compact_edges.len());
                for (u, v, c) in &compact_edges {
                    if *c > 0.0 {
                        b_rows.push(*u as GrB_Index);
                        b_cols.push(*v as GrB_Index);
                        b_vals.push(*c);
                    }
                }
                if !b_rows.is_empty() {
                    GrB_Matrix_build_FP64(
                        cap_mtx,
                        b_rows.as_ptr(),
                        b_cols.as_ptr(),
                        b_vals.as_ptr(),
                        b_rows.len() as GrB_Index,
                        GrB_MAX_FP64,
                    );
                }
                GrB_Matrix_wait(cap_mtx, GrB_WaitMode::GrB_COMPLETE as i32);

                let mut lag_g = create_lagraph_graph(
                    cap_mtx,
                    LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED,
                )?;
                let mut msg = new_msg();
                lagraph_bindings::LAGraph_Cached_AT(lag_g, msg.as_mut_ptr());
                lagraph_bindings::LAGraph_Cached_EMin(lag_g, msg.as_mut_ptr());

                let mut lag_max_flow: f64 = 0.0;
                let mut flow_mtx: GrB_Matrix = null_mut();
                let info = lagraphx_bindings::LAGr_MaxFlow(
                    &raw mut lag_max_flow,
                    &raw mut flow_mtx,
                    null_mut(),
                    lag_g,
                    src_id as u64,
                    sink_id as u64,
                    msg.as_mut_ptr(),
                );

                delete_lagraph_graph(&mut lag_g);

                if info != 0 || flow_mtx.is_null() {
                    if !flow_mtx.is_null() {
                        GrB_Matrix_free(&raw mut flow_mtx);
                    }
                    let detail = msg_to_string(&msg);
                    return Err(format!("LAGr_MaxFlow failed: {info}; {detail}"));
                }

                let mut nvals: GrB_Index = 0;
                GrB_Matrix_nvals(&raw mut nvals, flow_mtx);
                let mut rows = vec![0u64; nvals as usize];
                let mut cols = vec![0u64; nvals as usize];
                let mut vals = vec![0.0f64; nvals as usize];
                let mut nvals_out = nvals;
                GrB_Matrix_extractTuples_FP64(
                    rows.as_mut_ptr(),
                    cols.as_mut_ptr(),
                    vals.as_mut_ptr(),
                    &raw mut nvals_out,
                    flow_mtx,
                );
                GrB_Matrix_free(&raw mut flow_mtx);

                let mut flow_by_edge: FxHashMap<(usize, usize), f64> =
                    FxHashMap::with_capacity_and_hasher(nvals_out as usize, FxBuildHasher);
                for i in 0..nvals_out as usize {
                    if vals[i] > 0.0 {
                        flow_by_edge.insert((rows[i] as usize, cols[i] as usize), vals[i]);
                    }
                }

                let mut flows = vec![0.0_f64; original_edge_count];
                for (i, (u, v, _)) in compact_edges
                    .iter()
                    .take(original_edge_count)
                    .enumerate()
                {
                    if let Some(f) = flow_by_edge.get(&(*u, *v)) {
                        flows[i] = *f;
                    }
                }

                (lag_max_flow, flows)
            };

            let mut used_nodes: std::collections::BTreeSet<u64> =
                std::collections::BTreeSet::new();
            let mut edges_out: ThinVec<Value> = thin_vec![];
            let mut flows_out: ThinVec<Value> = thin_vec![];

            for (i, flow) in flows.into_iter().take(original_edge_count).enumerate() {
                if flow == 0.0 {
                    continue;
                }
                let (src_orig, dst_orig, rel_id) = edge_meta[i];
                used_nodes.insert(src_orig);
                used_nodes.insert(dst_orig);
                edges_out.push(Value::Relationship(rel_id));
                flows_out.push(Value::Float(flow));
            }

            let nodes_out: ThinVec<Value> = used_nodes
                .iter()
                .map(|&n| node_value(NodeId::from(n)))
                .collect();

            Ok(Batch::from_columns([
                Column::Values(vec![Value::List(Arc::new(nodes_out))]),
                Column::Values(vec![Value::List(Arc::new(edges_out))]),
                Column::Values(vec![Value::List(Arc::new(flows_out))]),
                Column::Floats(vec![max_flow]),
            ]))
        }
    );
}
