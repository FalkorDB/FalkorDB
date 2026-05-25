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

use super::{FnType, Functions, Type};
use crate::{
    graph::{
        graph::{Graph, NodeId, RelationshipId},
        graphblas::lagraph_bindings::{self, LAGraph_Boolean, LAGraph_Graph, LAGraph_Kind},
    },
    runtime::{ordermap::OrderMap, runtime::Runtime, value::Value},
};
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

/// Build a result row as Value::Map.
fn make_row(entries: Vec<(&str, Value)>) -> Value {
    let mut map = OrderMap::default();
    for (k, v) in entries {
        map.insert(Arc::new(String::from(k)), v);
    }
    Value::Map(Arc::new(map))
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
    let mut ids = std::collections::HashSet::new();
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
fn active_node_set(g: &Graph) -> std::collections::HashSet<u64> {
    use crate::runtime::orderset::OrderSet;
    let empty: OrderSet<Arc<String>> = OrderSet::default();
    g.get_nodes(&empty, 0).map(u64::from).collect()
}

/// Compact adjacency mapping: re-indexes active nodes to 0..n-1.
/// Returns (compact_matrix_handle, id_to_compact, compact_to_id, n).
/// The caller owns the returned GrB_Matrix and must free it.
unsafe fn build_compact_adj(
    adj: &crate::graph::graphblas::matrix::Matrix,
    active: &std::collections::HashSet<u64>,
) -> (
    crate::graph::graphblas::GrB_Matrix,
    std::collections::HashMap<u64, u64>,
    Vec<u64>,
    u64,
) {
    use crate::graph::graphblas::{
        GrB_BOOL, GrB_Index, GrB_Matrix, GrB_Matrix_extractTuples_BOOL, GrB_Matrix_new,
        GrB_Matrix_nvals, GrB_Matrix_setElement_BOOL, GrB_Matrix_wait, GrB_WaitMode,
    };

    let raw_adj = adj.inner();

    // Build mapping: original_id -> compact_id
    let mut sorted_ids: Vec<u64> = active.iter().copied().collect();
    sorted_ids.sort_unstable();
    let n = sorted_ids.len() as u64;

    let mut id_to_compact: std::collections::HashMap<u64, u64> =
        std::collections::HashMap::with_capacity(sorted_ids.len());
    for (compact, &orig) in sorted_ids.iter().enumerate() {
        id_to_compact.insert(orig, compact as u64);
    }

    // Extract tuples from the original matrix
    let mut nvals: GrB_Index = 0;
    GrB_Matrix_nvals(&raw mut nvals, raw_adj);
    let mut rows = vec![0u64; nvals as usize];
    let mut cols = vec![0u64; nvals as usize];
    let mut vals = vec![false; nvals as usize];
    let mut nvals_out = nvals;
    GrB_Matrix_extractTuples_BOOL(
        rows.as_mut_ptr(),
        cols.as_mut_ptr(),
        vals.as_mut_ptr(),
        &raw mut nvals_out,
        raw_adj,
    );

    // Build compact matrix
    let mut compact: GrB_Matrix = null_mut();
    GrB_Matrix_new(&raw mut compact, GrB_BOOL, n, n);

    for i in 0..nvals_out as usize {
        if let (Some(&cr), Some(&cc)) = (id_to_compact.get(&rows[i]), id_to_compact.get(&cols[i])) {
            GrB_Matrix_setElement_BOOL(compact, true, cr, cc);
        }
    }

    // Wait for pending operations
    GrB_Matrix_wait(compact, GrB_WaitMode::GrB_COMPLETE as i32);

    (compact, id_to_compact, sorted_ids, n)
}

/// Same as build_compact_adj but for a symmetric matrix -- also sets
/// the reverse direction.
unsafe fn build_compact_adj_symmetric(
    adj: &crate::graph::graphblas::matrix::Matrix,
    active: &std::collections::HashSet<u64>,
) -> (
    crate::graph::graphblas::GrB_Matrix,
    std::collections::HashMap<u64, u64>,
    Vec<u64>,
    u64,
) {
    use crate::graph::graphblas::{
        GrB_BOOL, GrB_Index, GrB_Matrix, GrB_Matrix_extractTuples_BOOL, GrB_Matrix_new,
        GrB_Matrix_nvals, GrB_Matrix_setElement_BOOL, GrB_Matrix_wait, GrB_WaitMode,
    };

    let raw_adj = adj.inner();

    let mut sorted_ids: Vec<u64> = active.iter().copied().collect();
    sorted_ids.sort_unstable();
    let n = sorted_ids.len() as u64;

    let mut id_to_compact: std::collections::HashMap<u64, u64> =
        std::collections::HashMap::with_capacity(sorted_ids.len());
    for (compact, &orig) in sorted_ids.iter().enumerate() {
        id_to_compact.insert(orig, compact as u64);
    }

    let mut nvals: GrB_Index = 0;
    GrB_Matrix_nvals(&raw mut nvals, raw_adj);
    let mut rows = vec![0u64; nvals as usize];
    let mut cols = vec![0u64; nvals as usize];
    let mut vals = vec![false; nvals as usize];
    let mut nvals_out = nvals;
    GrB_Matrix_extractTuples_BOOL(
        rows.as_mut_ptr(),
        cols.as_mut_ptr(),
        vals.as_mut_ptr(),
        &raw mut nvals_out,
        raw_adj,
    );

    let mut compact: GrB_Matrix = null_mut();
    GrB_Matrix_new(&raw mut compact, GrB_BOOL, n, n);

    for i in 0..nvals_out as usize {
        if let (Some(&cr), Some(&cc)) = (id_to_compact.get(&rows[i]), id_to_compact.get(&cols[i])) {
            GrB_Matrix_setElement_BOOL(compact, true, cr, cc);
            GrB_Matrix_setElement_BOOL(compact, true, cc, cr);
        }
    }

    GrB_Matrix_wait(compact, GrB_WaitMode::GrB_COMPLETE as i32);

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
                return Ok(Value::List(Arc::new(thin_vec![])));
            }

            let rel_types: Vec<Arc<String>> = rel_type.into_iter().collect();
            let adj = g.build_adjacency_matrix(&rel_types);

            unsafe {
                use crate::graph::graphblas::{lagraph_bindings, GrB_Vector, GrB_Vector_free};

                let active = active_node_set(&g);
                let (compact_adj, _id_to_compact, compact_to_id, _n) =
                    build_compact_adj(&adj, &active);

                let mut lag_g = create_lagraph_graph(compact_adj, LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED)?;

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
                    delete_lagraph_graph(&mut lag_g);
                    return Err(format!("LAGr_PageRank failed: {info}"));
                }

                // Extract results (compact indices)
                let entries = extract_vector_f64(centrality);

                // Free LAGraph resources
                GrB_Vector_free(&raw mut centrality);
                delete_lagraph_graph(&mut lag_g);

                // Map back to original IDs and filter by label
                let label_filter: Option<std::collections::HashSet<u64>> = label.as_ref().map(|l| {
                    collect_node_ids(&g, std::slice::from_ref(l)).into_iter().collect()
                });

                let mut results: ThinVec<Value> = thin_vec![];
                for (compact_idx, score) in entries {
                    let orig_id = compact_to_id[compact_idx as usize];
                    if let Some(ref filter) = label_filter
                        && !filter.contains(&orig_id) {
                            continue;
                        }
                    results.push(make_row(vec![
                        ("node", node_value(NodeId::from(orig_id))),
                        ("score", Value::Float(score)),
                    ]));
                }

                Ok(Value::List(Arc::new(results)))
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
            let config = parse_config(&args)?;
            if !config.is_empty() {
                validate_config_map(&config, &["nodeLabels", "relationshipTypes"])?;
            }
            let node_labels = extract_node_labels(&config)?;
            let rel_types = extract_rel_types(&config)?;

            let g = runtime.g.borrow();
            if g.node_count() == 0 {
                return Ok(Value::List(Arc::new(thin_vec![])));
            }

            // Build symmetric adjacency matrix (WCC needs undirected)
            let adj = g.build_symmetric_adjacency_matrix(&rel_types);

            unsafe {
                use crate::graph::graphblas::{lagraph_bindings, GrB_Vector, GrB_Vector_free};

                let active = active_node_set(&g);
                let (compact_adj, _id_to_compact, compact_to_id, _n) =
                    build_compact_adj_symmetric(&adj, &active);

                let mut lag_g = create_lagraph_graph(compact_adj, LAGraph_Kind::LAGraph_ADJACENCY_UNDIRECTED)?;

                // Cache symmetric structure
                let mut msg = new_msg();
                let lag_g_ref = lag_g.as_mut().ok_or_else(|| String::from("LAGraph graph pointer is null"))?;
                lag_g_ref.is_symmetric_structure = LAGraph_Boolean::LAGraph_TRUE;
                lagraph_bindings::LAGraph_Cached_OutDegree(lag_g, msg.as_mut_ptr());
                lagraph_bindings::LAGraph_Cached_NSelfEdges(lag_g, msg.as_mut_ptr());
                lagraph_bindings::LAGraph_DeleteSelfEdges(lag_g, msg.as_mut_ptr());

                // Run connected components
                let mut component: GrB_Vector = null_mut();
                let info = lagraph_bindings::LAGr_ConnectedComponents(
                    &raw mut component,
                    lag_g,
                    msg.as_mut_ptr(),
                );

                if info != 0 {
                    delete_lagraph_graph(&mut lag_g);
                    return Err(format!("LAGr_ConnectedComponents failed: {info}"));
                }

                let entries = extract_vector_i64(component);

                GrB_Vector_free(&raw mut component);
                delete_lagraph_graph(&mut lag_g);

                // Map compact indices back to original IDs and filter by labels
                let label_filter: Option<std::collections::HashSet<u64>> = if node_labels.is_empty() {
                    None
                } else {
                    Some(collect_node_ids(&g, &node_labels).into_iter().collect())
                };

                let mut results: ThinVec<Value> = thin_vec![];
                for (compact_idx, comp_id) in entries {
                    let orig_id = compact_to_id[compact_idx as usize];
                    if let Some(ref filter) = label_filter
                        && !filter.contains(&orig_id) {
                            continue;
                        }
                    results.push(make_row(vec![
                        ("node", node_value(NodeId::from(orig_id))),
                        ("componentId", Value::Int(comp_id)),
                    ]));
                }

                Ok(Value::List(Arc::new(results)))
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
            let config = parse_config(&args)?;
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
                return Ok(Value::List(Arc::new(thin_vec![])));
            }

            let adj = g.build_adjacency_matrix(&rel_types);

            unsafe {
                use crate::graph::graphblas::{lagraph_bindings, GrB_Vector, GrB_Vector_free};

                // When nodeLabels is specified, build compact matrix from only
                // those nodes (subgraph-induced betweenness). Otherwise use all.
                let node_set: std::collections::HashSet<u64> = if node_labels.is_empty() {
                    active_node_set(&g)
                } else {
                    collect_node_ids(&g, &node_labels).into_iter().collect()
                };

                let (compact_adj, _id_to_compact, compact_to_id, _n) =
                    build_compact_adj(&adj, &node_set);

                let mut lag_g = create_lagraph_graph(compact_adj, LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED)?;

                let mut msg = new_msg();
                lagraph_bindings::LAGraph_Cached_AT(lag_g, msg.as_mut_ptr());
                lagraph_bindings::LAGraph_Cached_OutDegree(lag_g, msg.as_mut_ptr());

                // Select source nodes for sampling (all from compact matrix)
                let n_nodes = compact_to_id.len();
                let sources: Vec<u64> = if n_nodes == 0 {
                    vec![]
                } else if sampling_size >= n_nodes {
                    (0..n_nodes as u64).collect()
                } else {
                    let actual_samples = sampling_size;
                    let mut srcs = Vec::with_capacity(actual_samples);
                    let mut rng = sampling_seed;
                    let mut used = std::collections::HashSet::new();
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
                    delete_lagraph_graph(&mut lag_g);
                    return Err(format!("LAGr_Betweenness failed: {info}"));
                }

                let entries = extract_vector_f64(centrality);

                GrB_Vector_free(&raw mut centrality);
                delete_lagraph_graph(&mut lag_g);

                // All compact indices map to valid nodes (already label-filtered)
                let mut results: ThinVec<Value> = thin_vec![];
                for (compact_idx, score) in entries {
                    let orig_id = compact_to_id[compact_idx as usize];
                    results.push(make_row(vec![
                        ("node", node_value(NodeId::from(orig_id))),
                        ("score", Value::Float(score)),
                    ]));
                }

                Ok(Value::List(Arc::new(results)))
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
        fn algo_bfs(runtime, args) {
            // arg 0: source node (Node or Null)
            let source_id = match &args[0] {
                Value::Node(id) => *id,
                Value::Null => return Ok(Value::List(Arc::new(thin_vec![]))),
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
                return Ok(Value::List(Arc::new(thin_vec![])));
            }

            let rel_types: Vec<Arc<String>> = rel_type.into_iter().collect();
            let adj = g.build_adjacency_matrix(&rel_types);

            unsafe {
                use crate::graph::graphblas::{lagraph_bindings, GrB_Vector, lagraphx_bindings, GrB_Vector_free, GrB_Matrix_free};

                let active = active_node_set(&g);
                let (mut compact_adj, id_to_compact, compact_to_id, _n) =
                    build_compact_adj(&adj, &active);

                // Map source_id to compact index
                let Some(&compact_source) = id_to_compact.get(&u64::from(source_id)) else {
                    GrB_Matrix_free(&raw mut compact_adj);
                    return Err(String::from("Source node not found in graph"));
                };

                let mut lag_g = create_lagraph_graph(compact_adj, LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED)?;

                let mut msg = new_msg();
                lagraph_bindings::LAGraph_Cached_AT(lag_g, msg.as_mut_ptr());
                lagraph_bindings::LAGraph_Cached_OutDegree(lag_g, msg.as_mut_ptr());

                let mut level: GrB_Vector = null_mut();
                let mut parent: GrB_Vector = null_mut();

                // Use the extended BFS with max_level
                let max_level = if max_depth < 0 { -1i64 } else { max_depth };
                let info = lagraphx_bindings::LAGr_BreadthFirstSearch_Extended(
                    &raw mut level,
                    &raw mut parent,
                    lag_g,
                    compact_source,
                    max_level,
                    -1, // no specific destination
                    false, // many_expected
                    msg.as_mut_ptr(),
                );

                if info != 0 {
                    delete_lagraph_graph(&mut lag_g);
                    return Err(format!("LAGr_BreadthFirstSearch_Extended failed: {info}"));
                }

                // Extract parent vector to reconstruct nodes and edges (compact indices)
                let parent_entries = extract_vector_i64(parent);

                // Build nodes list (excluding the source itself) and edges list
                let mut nodes: ThinVec<Value> = thin_vec![];
                let mut edges: ThinVec<Value> = thin_vec![];

                for (compact_idx, compact_par) in &parent_entries {
                    if *compact_idx == compact_source {
                        continue; // skip source itself
                    }
                    let orig_child = compact_to_id[*compact_idx as usize];
                    let orig_parent = compact_to_id[*compact_par as u64 as usize];
                    let child = NodeId::from(orig_child);
                    let parent_node = NodeId::from(orig_parent);
                    nodes.push(node_value(child));

                    // Find the relationship from parent to child
                    let rels: Vec<RelationshipId> = g.get_src_dest_relationships(parent_node, child, &rel_types).collect();
                    if let Some(rel_id) = rels.first() {
                        edges.push(Value::Relationship(Box::new((*rel_id, parent_node, child))));
                    }
                }

                if !level.is_null() {
                    GrB_Vector_free(&raw mut level);
                }
                if !parent.is_null() {
                    GrB_Vector_free(&raw mut parent);
                }
                delete_lagraph_graph(&mut lag_g);

                // If no nodes were reached, return empty list (no result row)
                if nodes.is_empty() {
                    return Ok(Value::List(Arc::new(thin_vec![])));
                }

                let result = make_row(vec![
                    ("nodes", Value::List(Arc::new(nodes))),
                    ("edges", Value::List(Arc::new(edges))),
                ]);

                Ok(Value::List(Arc::new(thin_vec![result])))
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
            let config = parse_config(&args)?;
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
                return Ok(Value::List(Arc::new(thin_vec![])));
            }

            // CDLP needs symmetric adjacency
            let adj = g.build_symmetric_adjacency_matrix(&rel_types);

            unsafe {
                use crate::graph::graphblas::{GrB_Vector, lagraphx_bindings, GrB_Vector_free};

                let active = active_node_set(&g);
                let (compact_adj, _id_to_compact, compact_to_id, _n) =
                    build_compact_adj_symmetric(&adj, &active);

                let mut lag_g = create_lagraph_graph(compact_adj, LAGraph_Kind::LAGraph_ADJACENCY_UNDIRECTED)?;

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
                    delete_lagraph_graph(&mut lag_g);
                    return Err(format!("LAGraph_cdlp failed: {info}"));
                }

                let entries = extract_vector_i64(cdlp);

                GrB_Vector_free(&raw mut cdlp);
                delete_lagraph_graph(&mut lag_g);

                let label_filter: Option<std::collections::HashSet<u64>> = if node_labels.is_empty() {
                    None
                } else {
                    Some(collect_node_ids(&g, &node_labels).into_iter().collect())
                };

                let mut results: ThinVec<Value> = thin_vec![];
                for (compact_idx, community_id) in entries {
                    let orig_id = compact_to_id[compact_idx as usize];
                    if let Some(ref filter) = label_filter
                        && !filter.contains(&orig_id) {
                            continue;
                        }
                    results.push(make_row(vec![
                        ("node", node_value(NodeId::from(orig_id))),
                        ("communityId", Value::Int(community_id)),
                    ]));
                }

                Ok(Value::List(Arc::new(results)))
            }
        }
    );
}

// ── algo.MSF ────────────────────────────────────────────────────────────

/// Find the best relationship from src to dst matching the given types and weight criteria.
/// Returns (RelationshipId, src_node, dst_node) or None if no relationship exists.
fn get_rel_weight(
    g: &Graph,
    rel_id: RelationshipId,
    weight_attr: Option<&Arc<String>>,
) -> f64 {
    weight_attr.map_or(1.0, |attr| {
        match g.get_relationship_attribute(rel_id, attr) {
            Some(Value::Float(f)) => f,
            Some(Value::Int(i)) => i as f64,
            _ => f64::NAN,
        }
    })
}

fn find_best_relationship(
    g: &Graph,
    src: NodeId,
    dst: NodeId,
    types: &[Arc<String>],
    weight_attr: Option<&Arc<String>>,
    maximize: bool,
) -> Option<(RelationshipId, NodeId, NodeId)> {
    let mut best: Option<(RelationshipId, f64)> = None;
    let missing_sentinel = if maximize {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    for rel_id in g.get_src_dest_relationships(src, dst, types) {
        if let Some(attr) = weight_attr {
            let w = match g.get_relationship_attribute(rel_id, attr) {
                Some(Value::Float(f)) => f,
                Some(Value::Int(i)) => i as f64,
                _ => missing_sentinel,
            };
            best = Some(match best {
                None => (rel_id, w),
                Some((_, prev_w)) if (maximize && w > prev_w) || (!maximize && w < prev_w) => {
                    (rel_id, w)
                }
                Some(prev) => prev,
            });
        } else {
            return Some((rel_id, src, dst));
        }
    }
    best.map(|(rel_id, _)| (rel_id, src, dst))
}

fn register_msf(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.MSF",
        args: [Type::Optional(Box::new(Type::Any))],
        ret: Type::Any,
        procedure: ["nodes", "edges"],
        fn algo_msf(runtime, args) {
            let config = parse_config(&args)?;
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
                return Ok(Value::List(Arc::new(thin_vec![])));
            }

            // Get the set of nodes we care about
            let active_nodes = collect_node_ids(&g, &node_labels);
            if active_nodes.is_empty() {
                return Ok(Value::List(Arc::new(thin_vec![])));
            }

            // Build a weighted FP64 adjacency matrix
            // For unweighted: use 1.0 for all entries
            // For weighted: use the attribute value
            unsafe {
                use crate::graph::graphblas::{GrB_Matrix, GrB_Matrix_new, GrB_FP64, GrB_Matrix_extractElement_FP64, GrB_Info, GrB_Matrix_setElement_FP64, GrB_Matrix_wait, GrB_WaitMode, GrB_Vector, lagraphx_bindings, GrB_Matrix_free, GrB_Index, GrB_Matrix_nvals, GrB_Matrix_extractTuples_FP64, GrB_Vector_free};
                use crate::runtime::orderset::OrderSet;

                let active_set: std::collections::HashSet<u64> = active_nodes.iter().copied().collect();

                // Build compact mapping (sorted original IDs -> 0..n-1)
                let mut sorted_ids: Vec<u64> = active_nodes;
                sorted_ids.sort_unstable();
                let n = sorted_ids.len() as u64;

                let mut id_to_compact: std::collections::HashMap<u64, u64> = std::collections::HashMap::with_capacity(sorted_ids.len());
                for (compact, &orig) in sorted_ids.iter().enumerate() {
                    id_to_compact.insert(orig, compact as u64);
                }

                // Iterate relationships and fill weighted compact matrix
                let empty_types: Vec<Arc<String>> = vec![];
                let types_to_use = if rel_types.is_empty() { &empty_types } else { &rel_types };

                let empty_label_set: OrderSet<Arc<String>> = OrderSet::default();

                // Create compact FP64 matrix
                let mut weighted_adj: GrB_Matrix = null_mut();
                GrB_Matrix_new(&raw mut weighted_adj, GrB_FP64, n, n);

                // For each edge, pick the best weight and write to compact matrix
                // Don't filter by labels here — the active_set/id_to_compact check below
                // already ensures both endpoints match the requested labels.
                for (src, dst) in g.get_relationships(types_to_use, &empty_label_set, &empty_label_set, None, None) {
                    let src_u = u64::from(src);
                    let dst_u = u64::from(dst);

                    // Both endpoints must be in active set
                    let (Some(&cs), Some(&cd)) = (id_to_compact.get(&src_u), id_to_compact.get(&dst_u)) else {
                        continue;
                    };

                    let weight = weight_attr.as_ref().map_or(1.0, |attr| {
                        // For missing/non-numeric weight attributes:
                        // minimize: use +INF (never chosen as minimum)
                        // maximize: use -INF, which negated = +INF (never chosen as minimum)
                        let missing_sentinel = if maximize { f64::NEG_INFINITY } else { f64::INFINITY };
                        let mut best_w: Option<f64> = None;
                        for rel_id in g.get_src_dest_relationships(src, dst, types_to_use) {
                            let w = match g.get_relationship_attribute(rel_id, attr) {
                                Some(Value::Float(f)) => f,
                                Some(Value::Int(i)) => i as f64,
                                _ => missing_sentinel,
                            };
                            best_w = Some(match best_w {
                                None => w,
                                Some(prev) if maximize => prev.max(w),
                                Some(prev) => prev.min(w),
                            });
                        }
                        let w = best_w.unwrap_or(missing_sentinel);
                        if maximize { -w } else { w }
                    });

                    // Only write if this is a better weight than existing
                    // (handles case where both (src,dst) and (dst,src) are iterated)
                    let mut existing = 0.0f64;
                    let has_existing = GrB_Matrix_extractElement_FP64(&raw mut existing, weighted_adj, cs, cd);
                    if has_existing != GrB_Info::GrB_SUCCESS || weight < existing {
                        GrB_Matrix_setElement_FP64(weighted_adj, weight, cs, cd);
                        GrB_Matrix_setElement_FP64(weighted_adj, weight, cd, cs);
                    }
                }

                GrB_Matrix_wait(weighted_adj, GrB_WaitMode::GrB_COMPLETE as i32);

                // Run Boruvka MSF
                let mut forest_edges: GrB_Matrix = null_mut();
                let mut component_id: GrB_Vector = null_mut();
                let mut msg = new_msg();

                let info = lagraphx_bindings::LAGraph_msf(
                    &raw mut forest_edges,
                    &raw mut component_id,
                    weighted_adj,
                    true, // sanitize
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

                // Group nodes by component (map compact indices back)
                let mut components: std::collections::BTreeMap<i64, Vec<u64>> = std::collections::BTreeMap::new();
                for (compact_idx, comp) in &comp_entries {
                    let orig_id = sorted_ids[*compact_idx as usize];
                    if !active_set.contains(&orig_id) {
                        continue;
                    }
                    components.entry(*comp).or_default().push(orig_id);
                }

                // Add isolated nodes (in active set but not in comp_entries)
                let in_comp: std::collections::HashSet<u64> = comp_entries.iter().map(|(ci, _)| sorted_ids[*ci as usize]).collect();
                for &nid in &sorted_ids {
                    if !in_comp.contains(&nid) && !g.is_node_deleted(NodeId::from(nid)) {
                        components.entry(nid as i64).or_default().push(nid);
                    }
                }

                // Build forest edge list per component
                let mut results: ThinVec<Value> = thin_vec![];
                for node_ids in components.values() {
                    let node_set: std::collections::HashSet<u64> = node_ids.iter().copied().collect();

                    let tree_nodes: ThinVec<Value> = node_ids
                        .iter()
                        .map(|&nid| node_value(NodeId::from(nid)))
                        .collect();

                    let mut tree_edges: ThinVec<Value> = thin_vec![];

                    // Find forest edges where both endpoints are in this component
                    for i in 0..nvals_out as usize {
                        let compact_src = f_rows[i];
                        let compact_dst = f_cols[i];
                        let orig_src = sorted_ids[compact_src as usize];
                        let orig_dst = sorted_ids[compact_dst as usize];
                        if node_set.contains(&orig_src) && node_set.contains(&orig_dst) {
                            let src_node = NodeId::from(orig_src);
                            let dst_node = NodeId::from(orig_dst);

                            // Check both directions and pick the overall best relationship
                            let fwd_best = find_best_relationship(&g, src_node, dst_node, types_to_use, weight_attr.as_ref(), maximize);
                            let rev_best = find_best_relationship(&g, dst_node, src_node, types_to_use, weight_attr.as_ref(), maximize);

                            let best = match (fwd_best, rev_best) {
                                (Some(f), Some(r)) => {
                                    // Compare weights to pick overall best
                                    let f_w = get_rel_weight(&g, f.0, weight_attr.as_ref());
                                    let r_w = get_rel_weight(&g, r.0, weight_attr.as_ref());
                                    if maximize {
                                        if r_w > f_w { Some(r) } else { Some(f) }
                                    } else if r_w < f_w { Some(r) } else { Some(f) }
                                }
                                (Some(f), None) => Some(f),
                                (None, Some(r)) => Some(r),
                                (None, None) => None,
                            };

                            if let Some((rel_id, s, d)) = best {
                                tree_edges.push(Value::Relationship(Box::new((rel_id, s, d))));
                            }
                        }
                    }

                    results.push(make_row(vec![
                        ("nodes", Value::List(Arc::new(tree_nodes))),
                        ("edges", Value::List(Arc::new(tree_edges))),
                    ]));
                }

                Ok(Value::List(Arc::new(results)))
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
    rel_direction: String,
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
        None | Some(Value::Null) => String::from("outgoing"),
        Some(Value::String(s)) => match s.as_str() {
            "incoming" | "outgoing" | "both" => s.to_string(),
            _ => {
                return Err(String::from(
                    "relDirection values must be 'incoming', 'outgoing' or 'both'",
                ));
            }
        },
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

fn run_path_algo(
    runtime: &Runtime,
    config: &PathAlgoConfig,
) -> Result<Value, String> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    struct State {
        weight: f64,
        cost: f64,
        path_len: u32,
        current: NodeId,
        visited: std::collections::HashSet<u64>,
        // Edges stored in original direction (edge_id, edge_src, edge_dst)
        edges: Vec<(RelationshipId, NodeId, NodeId)>,
    }

    impl Eq for State {}
    impl PartialEq for State {
        fn eq(
            &self,
            other: &Self,
        ) -> bool {
            self.weight == other.weight
                && self.cost == other.cost
                && self.path_len == other.path_len
        }
    }
    impl PartialOrd for State {
        fn partial_cmp(
            &self,
            other: &Self,
        ) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for State {
        fn cmp(
            &self,
            other: &Self,
        ) -> Ordering {
            // Min-heap: reverse comparison (smallest weight first)
            other
                .weight
                .partial_cmp(&self.weight)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    other
                        .cost
                        .partial_cmp(&self.cost)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| other.path_len.cmp(&self.path_len))
        }
    }

    let g = runtime.g.borrow();

    let mut heap = BinaryHeap::new();
    let mut initial_visited = std::collections::HashSet::new();
    initial_visited.insert(u64::from(config.source));

    heap.push(State {
        weight: 0.0,
        cost: 0.0,
        path_len: 0,
        current: config.source,
        visited: initial_visited,
        edges: Vec::new(),
    });

    let mut results: Vec<(Vec<(RelationshipId, NodeId, NodeId)>, f64, f64)> = Vec::new();
    let mut min_weight: Option<f64> = None;

    while let Some(state) = heap.pop() {
        // Stop if we have enough results
        if config.path_count > 0 && results.len() >= config.path_count as usize {
            break;
        }
        // pathCount=0: collect all paths with minimum weight
        if config.path_count == 0
            && let Some(mw) = min_weight
            && state.weight > mw + f64::EPSILON
        {
            break;
        }

        // Check if at target (must have at least one edge)
        let at_target = state.path_len > 0
            && match config.target {
                Some(t) => state.current == t,
                None => true, // SSpaths: any non-empty path is a result
            };

        if at_target {
            if config.path_count == 0 && min_weight.is_none() {
                min_weight = Some(state.weight);
            }
            results.push((state.edges.clone(), state.weight, state.cost));
            // For SPpaths at target, don't explore further from target
            if config.target.is_some() {
                continue;
            }
        }

        // Don't explore further if at maxLen
        if state.path_len >= config.max_len {
            continue;
        }

        for (edge_src, edge_dst, edge_id) in
            g.get_node_relationships_by_type(state.current, &config.rel_types)
        {
            let neighbor = match config.rel_direction.as_str() {
                "outgoing" => {
                    if edge_src == state.current {
                        Some(edge_dst)
                    } else {
                        None
                    }
                }
                "incoming" => {
                    if edge_dst == state.current {
                        Some(edge_src)
                    } else {
                        None
                    }
                }
                _ => {
                    // "both"
                    if edge_src == state.current {
                        Some(edge_dst)
                    } else if edge_dst == state.current {
                        Some(edge_src)
                    } else {
                        None
                    }
                }
            };

            let Some(next) = neighbor else { continue };
            if state.visited.contains(&u64::from(next)) {
                continue;
            }

            let edge_weight = config.weight_prop.as_ref().map_or(1.0, |prop| {
                match g.get_relationship_attribute(edge_id, prop) {
                    Some(Value::Float(f)) => f,
                    Some(Value::Int(i)) => i as f64,
                    _ => 1.0,
                }
            });

            let edge_cost = config.cost_prop.as_ref().map_or(0.0, |prop| {
                match g.get_relationship_attribute(edge_id, prop) {
                    Some(Value::Float(f)) => f,
                    Some(Value::Int(i)) => i as f64,
                    _ => 0.0,
                }
            });

            let new_weight = state.weight + edge_weight;
            let new_cost = state.cost + edge_cost;

            // Prune by maxCost (cost only increases with positive edge costs)
            if let Some(mc) = config.max_cost
                && new_cost > mc
            {
                continue;
            }

            let mut new_visited = state.visited.clone();
            new_visited.insert(u64::from(next));

            let mut new_edges = state.edges.clone();
            // Always store edges in original direction (edge_src, edge_dst)
            new_edges.push((edge_id, edge_src, edge_dst));

            heap.push(State {
                weight: new_weight,
                cost: new_cost,
                path_len: state.path_len + 1,
                current: next,
                visited: new_visited,
                edges: new_edges,
            });
        }
    }

    let mut result_list: ThinVec<Value> = ThinVec::with_capacity(results.len());
    for (edges, weight, cost) in results {
        let mut path_elems = ThinVec::new();
        path_elems.push(Value::Node(config.source));
        for (eid, esrc, edst) in &edges {
            let prev_id = path_elems.iter().rev().find_map(|v| {
                if let Value::Node(id) = v {
                    Some(*id)
                } else {
                    None
                }
            });
            path_elems.push(Value::Relationship(Box::new((*eid, *esrc, *edst))));
            let next = if prev_id == Some(*esrc) { *edst } else { *esrc };
            path_elems.push(Value::Node(next));
        }

        result_list.push(make_row(vec![
            ("path", Value::Path(Arc::new(path_elems))),
            ("pathWeight", to_numeric_value(weight)),
            ("pathCost", to_numeric_value(cost)),
        ]));
    }

    Ok(Value::List(Arc::new(result_list)))
}

fn register_sp_paths(funcs: &mut Functions) {
    cypher_fn!(funcs, "algo.SPpaths",
        args: [Type::Any],
        ret: Type::Any,
        procedure: ["path", "pathWeight", "pathCost"],
        fn algo_sp_paths(runtime, args) {
            let config = parse_sp_config(&args)?;
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
            let config = parse_ss_config(&args)?;
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
            use crate::runtime::orderset::OrderSet;

            let config = parse_config(&args)?;
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
                return Ok(Value::List(Arc::new(thin_vec![])));
            }

            let nodes_in_set: Vec<u64> = if node_labels.is_empty() {
                let mut v: Vec<u64> = active_node_set(&g).into_iter().collect();
                v.sort_unstable();
                v
            } else {
                collect_node_ids(&g, &node_labels)
            };

            if nodes_in_set.is_empty() {
                return Ok(Value::List(Arc::new(thin_vec![])));
            }

            let node_set: std::collections::HashSet<u64> =
                nodes_in_set.iter().copied().collect();

            let empty_label_set: OrderSet<Arc<String>> = OrderSet::default();
            let mut adj: std::collections::HashMap<u64, Vec<u64>> =
                std::collections::HashMap::with_capacity(nodes_in_set.len());
            for (src, dst) in g.get_relationships(
                &rel_types, &empty_label_set, &empty_label_set, None, None,
            ) {
                let s = u64::from(src);
                let d = u64::from(dst);
                if node_set.contains(&s) && node_set.contains(&d) && s != d {
                    adj.entry(s).or_default().push(d);
                }
            }

            let mut results: ThinVec<Value> = thin_vec![];
            let mut dist: std::collections::HashMap<u64, u64> =
                std::collections::HashMap::new();
            let mut queue: std::collections::VecDeque<u64> =
                std::collections::VecDeque::new();
            for &start in &nodes_in_set {
                dist.clear();
                queue.clear();
                dist.insert(start, 0);
                queue.push_back(start);
                let mut score: f64 = 0.0;
                let mut reachable: i64 = 0;
                while let Some(u) = queue.pop_front() {
                    let du = dist[&u];
                    if let Some(neighbors) = adj.get(&u) {
                        for &v in neighbors {
                            if let std::collections::hash_map::Entry::Vacant(e) =
                                dist.entry(v)
                            {
                                let dv = du + 1;
                                e.insert(dv);
                                queue.push_back(v);
                                #[allow(clippy::cast_precision_loss)]
                                {
                                    score += 1.0 / (dv as f64);
                                }
                                reachable += 1;
                            }
                        }
                    }
                }
                results.push(make_row(vec![
                    ("node", node_value(NodeId::from(start))),
                    ("score", Value::Float(score)),
                    ("reachable", Value::Int(reachable)),
                ]));
            }

            Ok(Value::List(Arc::new(results)))
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
            use crate::runtime::orderset::OrderSet;

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

            let src_set: std::collections::HashSet<u64> =
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

            let type_id = match g.get_type_id(&rel_types[0]) {
                Some(t) => t,
                None => return Err(format!(
                    "algo.maxFlow: relationship type '{}' does not exist",
                    rel_types[0],
                )),
            };
            if g.relationship_tensors()[type_id.0].has_multi_edge() {
                return Err(format!(
                    "algo.maxFlow: relationship type '{}' contains multi-edges; maxFlow requires a simple adjacency matrix",
                    rel_types[0],
                ));
            }
            if g.get_relationship_attribute_id(&capacity_attr).is_none()
                && default_cap.is_none()
            {
                return Err(String::from(
                    "algo.maxFlow: invalid or missing attribute and no default attribute specified",
                ));
            }

            let label_filter: Option<std::collections::HashSet<u64>> =
                if node_labels.is_empty() {
                    None
                } else {
                    Some(collect_node_ids(&g, &node_labels).into_iter().collect())
                };

            let empty_label_set: OrderSet<Arc<String>> = OrderSet::default();

            let mut edges_with_cap: Vec<(u64, u64, f64)> = Vec::new();
            for (src, dst) in g.get_relationships(
                &rel_types, &empty_label_set, &empty_label_set, None, None,
            ) {
                let s = u64::from(src);
                let d = u64::from(dst);
                if let Some(ref filter) = label_filter
                    && (!filter.contains(&s) || !filter.contains(&d))
                {
                    continue;
                }
                let mut cap: Option<f64> = None;
                for rel_id in g.get_src_dest_relationships(src, dst, &rel_types) {
                    let parsed = match g.get_relationship_attribute(rel_id, &capacity_attr) {
                        Some(Value::Float(f)) if f >= 0.0 => Some(f),
                        #[allow(clippy::cast_precision_loss)]
                        Some(Value::Int(i)) if i >= 0 => Some(i as f64),
                        _ => None,
                    };
                    if parsed.is_some() {
                        cap = parsed;
                        break;
                    }
                }
                let cap_value = match cap.or(default_cap) {
                    Some(c) => c,
                    None => return Err(String::from(
                        "algo.maxFlow: invalid or missing attribute and no default attribute specified",
                    )),
                };
                edges_with_cap.push((s, d, cap_value));
            }

            if edges_with_cap.is_empty() {
                let result = make_row(vec![
                    ("nodes", Value::List(Arc::new(thin_vec![]))),
                    ("edges", Value::List(Arc::new(thin_vec![]))),
                    ("edgeFlows", Value::List(Arc::new(thin_vec![]))),
                    ("maxFlow", Value::Float(0.0)),
                ]);
                return Ok(Value::List(Arc::new(thin_vec![result])));
            }

            unsafe {
                use crate::graph::graphblas::{
                    GrB_FP64, GrB_Index, GrB_Matrix, GrB_Matrix_extractTuples_FP64,
                    GrB_Matrix_free, GrB_Matrix_new, GrB_Matrix_nvals,
                    GrB_Matrix_setElement_FP64, GrB_Matrix_wait, GrB_WaitMode,
                    lagraph_bindings, lagraphx_bindings,
                };

                let multi_srcs = srcs.len() > 1;
                let multi_sinks = sinks.len() > 1;
                let base_dim = g.node_cap();
                let dim = base_dim + u64::from(multi_srcs) + u64::from(multi_sinks);

                let mut cap_mtx: GrB_Matrix = null_mut();
                GrB_Matrix_new(&raw mut cap_mtx, GrB_FP64, dim, dim);

                let mut min_cap = f64::INFINITY;
                let mut max_cap = 0.0_f64;
                for (s, d, c) in &edges_with_cap {
                    if *c > 0.0 {
                        GrB_Matrix_setElement_FP64(cap_mtx, *c, *s, *d);
                        if *c < min_cap { min_cap = *c; }
                        if *c > max_cap { max_cap = *c; }
                    }
                }

                let mut src_id = u64::from(srcs[0]);
                let mut sink_id = u64::from(sinks[0]);
                #[allow(clippy::cast_lossless)]
                let big_cap: f64 = i32::MAX as f64;
                if multi_srcs {
                    src_id = base_dim;
                    for s in &srcs {
                        GrB_Matrix_setElement_FP64(cap_mtx, big_cap, src_id, u64::from(*s));
                    }
                }
                if multi_sinks {
                    sink_id = base_dim + u64::from(multi_srcs);
                    for s in &sinks {
                        GrB_Matrix_setElement_FP64(cap_mtx, big_cap, u64::from(*s), sink_id);
                    }
                }

                GrB_Matrix_wait(cap_mtx, GrB_WaitMode::GrB_COMPLETE as i32);

                #[allow(clippy::cast_lossless)]
                let overflow_factor = u32::MAX as f64;
                if min_cap.is_finite() && max_cap >= min_cap * overflow_factor {
                    GrB_Matrix_free(&raw mut cap_mtx);
                    return Err(String::from(
                        "algo.maxFlow: capacity range too wide (max >= min * 2^32); narrow the capacity values to avoid internal overflow",
                    ));
                }

                let mut lag_g = create_lagraph_graph(
                    cap_mtx, LAGraph_Kind::LAGraph_ADJACENCY_DIRECTED,
                )?;

                let mut msg = new_msg();
                lagraph_bindings::LAGraph_Cached_AT(lag_g, msg.as_mut_ptr());
                lagraph_bindings::LAGraph_Cached_EMin(lag_g, msg.as_mut_ptr());

                let mut max_flow: f64 = 0.0;
                let mut flow_mtx: GrB_Matrix = null_mut();
                let info = lagraphx_bindings::LAGr_MaxFlow(
                    &raw mut max_flow,
                    &raw mut flow_mtx,
                    lag_g,
                    src_id,
                    sink_id,
                    msg.as_mut_ptr(),
                );

                delete_lagraph_graph(&mut lag_g);

                if info != 0 {
                    if !flow_mtx.is_null() {
                        GrB_Matrix_free(&raw mut flow_mtx);
                    }
                    return Err(format!("LAGr_MaxFlow failed: {info}"));
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

                let mut used_nodes: std::collections::BTreeSet<u64> =
                    std::collections::BTreeSet::new();
                let mut edges_out: ThinVec<Value> = thin_vec![];
                let mut flows_out: ThinVec<Value> = thin_vec![];

                for i in 0..nvals_out as usize {
                    let r = rows[i];
                    let c = cols[i];
                    let v = vals[i];
                    if r >= base_dim || c >= base_dim || v == 0.0 {
                        continue;
                    }
                    let src_n = NodeId::from(r);
                    let dst_n = NodeId::from(c);
                    used_nodes.insert(r);
                    used_nodes.insert(c);
                    if let Some(rel_id) = g
                        .get_src_dest_relationships(src_n, dst_n, &rel_types)
                        .next()
                    {
                        edges_out.push(Value::Relationship(
                            Box::new((rel_id, src_n, dst_n)),
                        ));
                        flows_out.push(Value::Float(v));
                    }
                }

                let nodes_out: ThinVec<Value> = used_nodes
                    .iter()
                    .map(|&n| node_value(NodeId::from(n)))
                    .collect();

                let result = make_row(vec![
                    ("nodes", Value::List(Arc::new(nodes_out))),
                    ("edges", Value::List(Arc::new(edges_out))),
                    ("edgeFlows", Value::List(Arc::new(flows_out))),
                    ("maxFlow", Value::Float(max_flow)),
                ]);

                Ok(Value::List(Arc::new(thin_vec![result])))
            }
        }
    );
}
