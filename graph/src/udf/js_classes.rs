//! # JS Class Bindings for Graph Types
//!
//! This module creates JavaScript object representations of FalkorDB graph
//! entities (Node, Edge, Path) so that UDF functions can inspect and traverse
//! the graph from within JavaScript code.
//!
//! ## JS Object Shapes
//!
//! Each graph entity is represented as a plain JS object with hidden
//! `__falkor_type` / `__falkor_*_id` markers used for round-trip conversion
//! back to Rust values (see [`type_convert`](super::type_convert)).
//!
//! ```text
//! Node {
//!   id:         u64,
//!   labels:     [String, ...],
//!   attributes: { key: value, ... },
//!   getNeighbors(config?):  -> [Node] | [Edge]
//!   // hidden: __falkor_type = "node", __falkor_node_id
//! }
//!
//! Edge {
//!   id:         u64,
//!   type:       String,            // relationship type name
//!   source:     Node,
//!   target:     Node,
//!   attributes: { key: value, ... },
//!   // hidden: __falkor_type = "edge", __falkor_edge_id/src/dst
//! }
//!
//! Path {
//!   nodes:         [Node, ...],
//!   relationships: [Edge, ...],
//!   length:        usize,
//!   // hidden: __falkor_type = "path"
//! }
//! ```
//!
//! ## Graph Context
//!
//! JS closures cannot directly capture Rust references with non-static
//! lifetimes. To work around this, the current [`Graph`](crate::graph::graph::Graph)
//! is stored in a thread-local (`CURRENT_GRAPH`) and set/cleared around each
//! UDF invocation by [`set_current_graph`] / [`clear_current_graph`].
//!
//! ## Traversal
//!
//! - `node.getNeighbors(config?)` -- single-hop neighbor lookup with optional
//!   direction, type/label filters, and return type (nodes or edges).
//! - `graph.traverse(nodes, config?)` -- multi-source BFS traversal with
//!   configurable `maxDepth`, direction, type/label filters, and timeout
//!   enforcement via the UDF deadline.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use atomic_refcell::AtomicRefCell;
use rquickjs::{Array, Ctx, Function, Object, Value as JsValue};

use crate::graph::graph::{Graph, NodeId, RelationshipId};
use crate::runtime::runtime::Runtime;
use crate::runtime::value::Value;
use crate::udf::type_convert;

thread_local! {
    static CURRENT_GRAPH: RefCell<Option<Arc<AtomicRefCell<Graph>>>> = const { RefCell::new(None) };
}

pub fn set_current_graph(g: Arc<AtomicRefCell<Graph>>) {
    CURRENT_GRAPH.with(|cell| {
        *cell.borrow_mut() = Some(g);
    });
}

pub fn clear_current_graph() {
    CURRENT_GRAPH.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

fn with_current_graph<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&Arc<AtomicRefCell<Graph>>) -> Result<R, String>,
{
    CURRENT_GRAPH.with(|cell| {
        let guard = cell.borrow();
        let g = guard.as_ref().ok_or("No graph context available")?;
        f(g)
    })
}

/// Collect edges from a node in a given direction.
/// Returns vec of (rel_id, src_id, dst_id) as u64.
fn collect_edges(
    g: &Graph,
    node_id: u64,
    direction: &str,
) -> Vec<(u64, u64, u64)> {
    let nid = NodeId::from(node_id);
    let mut edges = Vec::new();
    for (src, dst, rel_id) in g.get_node_relationships(nid) {
        let src_u64: u64 = src.into();
        let dst_u64: u64 = dst.into();
        let rel_u64: u64 = rel_id.into();
        match direction {
            "outgoing" => {
                if src_u64 == node_id {
                    edges.push((rel_u64, src_u64, dst_u64));
                }
            }
            "incoming" => {
                if dst_u64 == node_id {
                    edges.push((rel_u64, src_u64, dst_u64));
                }
            }
            _ => {
                // "both"
                edges.push((rel_u64, src_u64, dst_u64));
            }
        }
    }
    edges
}

/// Create a JS Node object with id, labels, attributes properties and getNeighbors() method.
pub fn create_js_node<'js>(
    ctx: &Ctx<'js>,
    node_id: u64,
    graph: &Arc<AtomicRefCell<Graph>>,
) -> Result<JsValue<'js>, String> {
    let obj = Object::new(ctx.clone()).map_err(|e| format!("JS object error: {e}"))?;

    let g = graph.borrow();
    let nid = NodeId::from(node_id);

    // Hidden type markers
    obj.set("__falkor_type", "node")
        .map_err(|e| format!("JS set error: {e}"))?;
    obj.set("__falkor_node_id", node_id)
        .map_err(|e| format!("JS set error: {e}"))?;

    // .id - internal node ID
    obj.set("id", node_id)
        .map_err(|e| format!("JS set error: {e}"))?;

    // .labels - array of label strings
    let labels_arr = Array::new(ctx.clone()).map_err(|e| format!("JS array error: {e}"))?;
    for (i, label_name) in g.get_node_labels(nid).enumerate() {
        labels_arr
            .set(i, label_name.as_str())
            .map_err(|e| format!("JS set error: {e}"))?;
    }
    obj.set("labels", labels_arr)
        .map_err(|e| format!("JS set error: {e}"))?;

    // .attributes - object with all node properties
    let attrs_obj = Object::new(ctx.clone()).map_err(|e| format!("JS object error: {e}"))?;
    for (attr_name, value) in g.get_node_all_attrs(nid) {
        let js_val = type_convert::value_to_js(ctx, &value, graph, None)?;
        attrs_obj
            .set(attr_name.as_str(), js_val)
            .map_err(|e| format!("JS set error: {e}"))?;
    }
    obj.set("attributes", attrs_obj)
        .map_err(|e| format!("JS set error: {e}"))?;

    drop(g);

    // .getNeighbors(config?) method
    // We embed the node_id in a JS closure to avoid Rust lifetime issues with rquickjs closures
    let node_id_js = node_id;

    // Register the global helper if not already done
    let globals = ctx.globals();
    let has_helper: bool = globals
        .get::<_, JsValue>("__falkor_getNeighbors")
        .is_ok_and(|v| v.is_function());
    if !has_helper {
        let helper = Function::new(ctx.clone(), js_get_neighbors_entry)
            .map_err(|e| format!("JS function error: {e}"))?;
        globals
            .set("__falkor_getNeighbors", helper)
            .map_err(|e| format!("JS set error: {e}"))?;
    }

    // Expose the object via temp global so JS eval can attach the method
    globals
        .set("__tmp_obj", obj.as_value().clone())
        .map_err(|e| format!("JS set error: {e}"))?;

    // Create the method via JS eval, closing over this node's numeric ID
    ctx.eval::<(), _>(format!(
        "Object.defineProperty(globalThis.__tmp_obj, 'getNeighbors', {{\
            value: function(config) {{ return __falkor_getNeighbors({node_id_js}, config); }},\
            enumerable: false\
        }})"
    ))
    .map_err(|e| format!("Failed to set getNeighbors: {e}"))?;

    // Clean up temp ref
    ctx.eval::<(), _>("delete globalThis.__tmp_obj;")
        .map_err(|e| format!("JS cleanup error: {e}"))?;

    Ok(obj.into_value())
}

/// Create a JS Edge object with id, type, source, target, attributes.
pub fn create_js_edge<'js>(
    ctx: &Ctx<'js>,
    rel_id: u64,
    src_id: u64,
    dst_id: u64,
    graph: &Arc<AtomicRefCell<Graph>>,
    runtime: Option<&Runtime<'_>>,
) -> Result<JsValue<'js>, String> {
    let obj = Object::new(ctx.clone()).map_err(|e| format!("JS object error: {e}"))?;

    let rid = RelationshipId::from(rel_id);

    // Hidden type markers
    obj.set("__falkor_type", "edge")
        .map_err(|e| format!("JS set error: {e}"))?;
    obj.set("__falkor_edge_id", rel_id)
        .map_err(|e| format!("JS set error: {e}"))?;
    obj.set("__falkor_edge_src", src_id)
        .map_err(|e| format!("JS set error: {e}"))?;
    obj.set("__falkor_edge_dst", dst_id)
        .map_err(|e| format!("JS set error: {e}"))?;

    // .id - internal edge ID
    obj.set("id", rel_id)
        .map_err(|e| format!("JS set error: {e}"))?;

    // .type - relationship type string. Resolve through the runtime when
    // available so a relationship deleted/created earlier in the same query is
    // handled; otherwise fall back to the committed graph.
    let type_name = match runtime {
        Some(rt) => rt
            .get_relationship_type(rid)
            .unwrap_or_else(|| Arc::new(String::new())),
        None => {
            let g = graph.borrow();
            g.get_type(g.get_relationship_type_id(rid))
                .unwrap_or_else(|| Arc::new(String::new()))
        }
    };
    obj.set("type", type_name.as_str())
        .map_err(|e| format!("JS set error: {e}"))?;

    // .source and .target - node objects
    let source_node = create_js_node(ctx, src_id, graph)?;
    let target_node = create_js_node(ctx, dst_id, graph)?;
    obj.set("source", source_node)
        .map_err(|e| format!("JS set error: {e}"))?;
    obj.set("target", target_node)
        .map_err(|e| format!("JS set error: {e}"))?;

    // .attributes - properties (runtime-aware, same fallback as the type).
    let attrs_obj = Object::new(ctx.clone()).map_err(|e| format!("JS object error: {e}"))?;
    match runtime {
        Some(rt) => {
            for (attr_name, value) in rt.get_relationship_attrs(rid) {
                let js_val = type_convert::value_to_js(ctx, &value, graph, runtime)?;
                attrs_obj
                    .set(attr_name.as_str(), js_val)
                    .map_err(|e| format!("JS set error: {e}"))?;
            }
        }
        None => {
            let g = graph.borrow();
            for (attr_name, value) in g.get_relationship_all_attrs(rid) {
                let js_val = type_convert::value_to_js(ctx, &value, graph, runtime)?;
                attrs_obj
                    .set(attr_name.as_str(), js_val)
                    .map_err(|e| format!("JS set error: {e}"))?;
            }
        }
    }
    obj.set("attributes", attrs_obj)
        .map_err(|e| format!("JS set error: {e}"))?;

    Ok(obj.into_value())
}

/// Create a JS Path object with nodes, relationships, length.
pub fn create_js_path<'js>(
    ctx: &Ctx<'js>,
    path_values: &[Value],
    graph: &Arc<AtomicRefCell<Graph>>,
    runtime: Option<&Runtime<'_>>,
) -> Result<JsValue<'js>, String> {
    let obj = Object::new(ctx.clone()).map_err(|e| format!("JS object error: {e}"))?;

    obj.set("__falkor_type", "path")
        .map_err(|e| format!("JS set error: {e}"))?;

    let nodes_arr = Array::new(ctx.clone()).map_err(|e| format!("JS array error: {e}"))?;
    let rels_arr = Array::new(ctx.clone()).map_err(|e| format!("JS array error: {e}"))?;

    let mut node_idx = 0;
    let mut rel_idx = 0;
    for val in path_values {
        match val {
            Value::Node(nid) => {
                let js_node = create_js_node(ctx, (*nid).into(), graph)?;
                nodes_arr
                    .set(node_idx, js_node)
                    .map_err(|e| format!("JS set error: {e}"))?;
                node_idx += 1;
            }
            Value::Relationship(rel_id) => {
                let (src, dst) = match runtime {
                    Some(rt) => rt.get_relationship_endpoints(*rel_id),
                    None => graph.borrow().get_relationship_endpoints(*rel_id),
                };
                let js_edge = create_js_edge(
                    ctx,
                    (*rel_id).into(),
                    u64::from(src),
                    u64::from(dst),
                    graph,
                    runtime,
                )?;
                rels_arr
                    .set(rel_idx, js_edge)
                    .map_err(|e| format!("JS set error: {e}"))?;
                rel_idx += 1;
            }
            _ => {}
        }
    }

    obj.set("nodes", nodes_arr)
        .map_err(|e| format!("JS set error: {e}"))?;
    obj.set("relationships", rels_arr)
        .map_err(|e| format!("JS set error: {e}"))?;
    obj.set("length", rel_idx)
        .map_err(|e| format!("JS set error: {e}"))?;

    Ok(obj.into_value())
}

/// Entry point for getNeighbors from JS. Takes (node_id, config?) as arguments.
/// This is a standalone function (no closure captures) so rquickjs lifetime inference works.
#[allow(clippy::needless_pass_by_value)]
fn js_get_neighbors_entry<'js>(
    ctx: Ctx<'js>,
    node_id: u64,
    config: rquickjs::function::Opt<Object<'js>>,
) -> Result<JsValue<'js>, rquickjs::Error> {
    match js_get_neighbors(&ctx, node_id, config.0.as_ref()) {
        Ok(val) => Ok(val),
        Err(e) => {
            let msg = match rquickjs::String::from_str(ctx.clone(), &e) {
                Ok(s) => s,
                Err(_) => rquickjs::String::from_str(ctx.clone(), "internal error")
                    .map_err(|_| rquickjs::Error::Exception)?,
            };
            Err(ctx.throw(msg.into_value()))
        }
    }
}

/// Implementation of node.getNeighbors(config?)
fn js_get_neighbors<'js>(
    ctx: &Ctx<'js>,
    node_id: u64,
    config: Option<&Object<'js>>,
) -> Result<JsValue<'js>, String> {
    let mut direction = "outgoing".to_string();
    let mut type_filters: Vec<String> = Vec::new();
    let mut label_filters: Vec<String> = Vec::new();
    let mut return_type = "nodes".to_string();

    if let Some(cfg) = config {
        if let Ok(d) = cfg.get::<_, String>("direction") {
            match d.to_lowercase().as_str() {
                "outgoing" | "incoming" | "both" => direction = d.to_lowercase(),
                _ => return Err(format!("Invalid direction: '{d}'")),
            }
        }
        // Validate types: must be an array if present
        if let Ok(val) = cfg.get::<_, JsValue>("types")
            && !val.is_undefined()
        {
            if !val.is_array() {
                return Err("'types' must be an array of strings".into());
            }
            let types: Array = val.into_array().unwrap();
            for i in 0..types.len() {
                if let Ok(t) = types.get::<String>(i) {
                    type_filters.push(t);
                }
            }
        }
        // Validate labels: must be an array if present
        if let Ok(val) = cfg.get::<_, JsValue>("labels")
            && !val.is_undefined()
        {
            if !val.is_array() {
                return Err("'labels' must be an array of strings".into());
            }
            let labels: Array = val.into_array().unwrap();
            for i in 0..labels.len() {
                if let Ok(l) = labels.get::<String>(i) {
                    label_filters.push(l);
                }
            }
        }
        if let Ok(rt) = cfg.get::<_, String>("returnType") {
            match rt.to_lowercase().as_str() {
                "nodes" | "edges" => return_type = rt.to_lowercase(),
                _ => return Err(format!("Invalid returnType: '{rt}'")),
            }
        }
        // Validate distance if present
        if let Ok(val) = cfg.get::<_, JsValue>("distance")
            && !val.is_undefined()
        {
            let d = if let Some(i) = val.as_int() {
                i as i64
            } else if let Some(f) = val.as_float() {
                if f < 0.0 || f != f.floor() {
                    return Err("'distance' must be a non-negative integer".into());
                }
                f as i64
            } else {
                return Err("'distance' must be a non-negative integer".into());
            };
            if d < 0 {
                return Err("'distance' must be a non-negative integer".into());
            }
            if d != 1 {
                return Err(
                    "getNeighbors only supports distance=1; use graph.traverse() for multi-hop"
                        .into(),
                );
            }
        }
    }

    with_current_graph(|graph| {
        let g = graph.borrow();
        let results = Array::new(ctx.clone()).map_err(|e| format!("JS array error: {e}"))?;
        let mut result_idx = 0;

        let mut neighbor_edges = collect_edges(&g, node_id, &direction);

        // Filter by relationship types
        if !type_filters.is_empty() {
            neighbor_edges.retain(|(rel_id, _, _)| {
                let rid = RelationshipId::from(*rel_id);
                let type_id = g.get_relationship_type_id(rid);
                g.get_type(type_id)
                    .is_some_and(|tn| type_filters.iter().any(|f| f == tn.as_str()))
            });
        }

        // Filter by neighbor labels
        if !label_filters.is_empty() {
            neighbor_edges.retain(|(_rel_id, src_id, dst_id)| {
                let neighbor_id = if *src_id == node_id { *dst_id } else { *src_id };
                let nid = NodeId::from(neighbor_id);
                g.get_node_labels(nid)
                    .any(|label_name| label_filters.iter().any(|f| f == label_name.as_str()))
            });
        }

        drop(g);

        let mut seen_neighbors: HashSet<u64> = HashSet::new();
        for (rel_id, src_id, dst_id) in &neighbor_edges {
            if return_type == "edges" {
                let js_edge = create_js_edge(ctx, *rel_id, *src_id, *dst_id, graph, None)?;
                results
                    .set(result_idx, js_edge)
                    .map_err(|e| format!("JS set error: {e}"))?;
                result_idx += 1;
            } else {
                let neighbor_id = if *src_id == node_id { *dst_id } else { *src_id };
                if seen_neighbors.insert(neighbor_id) {
                    let js_node = create_js_node(ctx, neighbor_id, graph)?;
                    results
                        .set(result_idx, js_node)
                        .map_err(|e| format!("JS set error: {e}"))?;
                    result_idx += 1;
                }
            }
        }

        Ok(results.into_value())
    })
}

/// graph.traverse(nodes, config) - multi-source BFS traversal.
#[allow(clippy::needless_pass_by_value)]
pub fn js_traverse<'js>(
    ctx: Ctx<'js>,
    nodes: Array<'js>,
    config: rquickjs::function::Opt<Object<'js>>,
) -> Result<JsValue<'js>, rquickjs::Error> {
    match js_traverse_impl(&ctx, &nodes, config.0.as_ref()) {
        Ok(val) => Ok(val),
        Err(e) => {
            let msg = match rquickjs::String::from_str(ctx.clone(), &e) {
                Ok(s) => s,
                Err(_) => rquickjs::String::from_str(ctx.clone(), "internal error")
                    .map_err(|_| rquickjs::Error::Exception)?,
            };
            Err(ctx.throw(msg.into_value()))
        }
    }
}

fn js_traverse_impl<'js>(
    ctx: &Ctx<'js>,
    nodes: &Array<'js>,
    config: Option<&Object<'js>>,
) -> Result<JsValue<'js>, String> {
    let mut direction = "outgoing".to_string();
    let mut type_filters: Vec<String> = Vec::new();
    let mut label_filters: Vec<String> = Vec::new();
    let mut return_type = "nodes".to_string();
    let mut max_depth: u32 = 1;

    if let Some(cfg) = config {
        if let Ok(d) = cfg.get::<_, String>("direction") {
            match d.to_lowercase().as_str() {
                "outgoing" | "incoming" | "both" => direction = d.to_lowercase(),
                _ => return Err(format!("Invalid direction: '{d}'")),
            }
        }
        if let Ok(types) = cfg.get::<_, Array>("types") {
            for i in 0..types.len() {
                if let Ok(t) = types.get::<String>(i) {
                    type_filters.push(t);
                }
            }
        }
        if let Ok(labels) = cfg.get::<_, Array>("labels") {
            for i in 0..labels.len() {
                if let Ok(l) = labels.get::<String>(i) {
                    label_filters.push(l);
                }
            }
        }
        if let Ok(rt) = cfg.get::<_, String>("returnType") {
            match rt.to_lowercase().as_str() {
                "nodes" | "edges" => return_type = rt.to_lowercase(),
                _ => return Err(format!("Invalid returnType: '{rt}'")),
            }
        }
        if let Ok(d) = cfg.get::<_, u32>("maxDepth") {
            max_depth = d;
        }
    }

    // Collect source node IDs
    let mut source_ids: Vec<u64> = Vec::new();
    for i in 0..nodes.len() {
        let node: Object = nodes.get(i).map_err(|e| format!("Array get error: {e}"))?;
        let nid: u64 = node
            .get("__falkor_node_id")
            .map_err(|e| format!("Node ID error: {e}"))?;
        source_ids.push(nid);
    }

    with_current_graph(|graph| {
        // Per-source traversal: each source node gets its own independent BFS
        let outer_results = Array::new(ctx.clone()).map_err(|e| format!("JS array error: {e}"))?;

        // Build a deadline from the UDF timeout so the BFS cannot run forever.
        // Use the same capped value as the QuickJS interrupt handler so the
        // traversal cannot bypass JS_TIMEOUT_ABSOLUTE_CAP_MS.
        let effective_ms = crate::udf::js_context::compute_effective_js_timeout_ms();
        let deadline = Some(Instant::now() + Duration::from_millis(effective_ms));

        for (src_idx, &start_id) in source_ids.iter().enumerate() {
            let inner_results =
                Array::new(ctx.clone()).map_err(|e| format!("JS array error: {e}"))?;
            let mut result_idx = 0;
            let mut visited: HashSet<u64> = HashSet::new();
            visited.insert(start_id);

            // Track seen relationship IDs per source node to avoid emitting
            // the same edge twice when direction == "both" and returnType == "edges".
            // An edge A->B discovered from A at depth N would otherwise be
            // rediscovered from B at depth N+1.
            let dedup_edges = return_type == "edges" && direction == "both";
            let mut seen_edges: HashMap<u64, HashSet<u64>> = HashMap::new();

            let mut frontier = vec![start_id];

            for _ in 0..max_depth {
                // Check deadline before processing each depth level
                if let Some(dl) = deadline
                    && Instant::now() > dl
                {
                    return Err("UDF Exception: Query timed out".to_string());
                }

                let mut next_frontier = Vec::new();
                let mut edges_to_create: Vec<(u64, u64, u64)> = Vec::new();

                {
                    let g = graph.borrow();

                    for &nid in &frontier {
                        let mut neighbor_edges = collect_edges(&g, nid, &direction);

                        if !type_filters.is_empty() {
                            neighbor_edges.retain(|(rel_id, _, _)| {
                                let rid = RelationshipId::from(*rel_id);
                                let type_id = g.get_relationship_type_id(rid);
                                g.get_type(type_id)
                                    .is_some_and(|tn| type_filters.iter().any(|f| f == tn.as_str()))
                            });
                        }

                        for (rel_id, src_id, dst_id) in neighbor_edges {
                            let neighbor_id = if src_id == nid { dst_id } else { src_id };

                            if !label_filters.is_empty() {
                                let nb_node_id = NodeId::from(neighbor_id);
                                let matches = g.get_node_labels(nb_node_id).any(|label_name| {
                                    label_filters.iter().any(|f| f == label_name.as_str())
                                });
                                if !matches {
                                    continue;
                                }
                            }

                            if return_type == "edges" {
                                // When direction is "both", de-dup edges so that
                                // an edge A->B found from A isn't re-emitted when
                                // discovered from B at the next depth.
                                if dedup_edges {
                                    let source_seen = seen_edges.entry(nid).or_default();
                                    if source_seen.insert(rel_id) {
                                        // Also mark in the neighbor's set so it
                                        // won't be emitted again from the other side.
                                        seen_edges.entry(neighbor_id).or_default().insert(rel_id);
                                        edges_to_create.push((rel_id, src_id, dst_id));
                                    } else {
                                        // Already emitted from this source — but
                                        // it was from the other endpoint; skip.
                                    }
                                } else {
                                    edges_to_create.push((rel_id, src_id, dst_id));
                                }
                            }

                            if visited.insert(neighbor_id) {
                                next_frontier.push(neighbor_id);
                            }
                        }
                    }
                }

                // Check deadline after building the frontier
                if let Some(dl) = deadline
                    && Instant::now() > dl
                {
                    return Err("UDF Exception: Query timed out".to_string());
                }

                if return_type == "nodes" {
                    for &nid in &next_frontier {
                        let js_node = create_js_node(ctx, nid, graph)?;
                        inner_results
                            .set(result_idx, js_node)
                            .map_err(|e| format!("JS set error: {e}"))?;
                        result_idx += 1;
                    }
                } else {
                    for (rel_id, src_id, dst_id) in &edges_to_create {
                        let js_edge = create_js_edge(ctx, *rel_id, *src_id, *dst_id, graph, None)?;
                        inner_results
                            .set(result_idx, js_edge)
                            .map_err(|e| format!("JS set error: {e}"))?;
                        result_idx += 1;
                    }
                }

                frontier = next_frontier;
                if frontier.is_empty() {
                    break;
                }
            }

            outer_results
                .set(src_idx, inner_results)
                .map_err(|e| format!("JS set error: {e}"))?;
        }

        Ok(outer_results.into_value())
    })
}
