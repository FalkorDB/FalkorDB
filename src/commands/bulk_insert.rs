use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{ThreadedGraph, ffi},
    redis_type::GRAPH_TYPE,
};
use graph::{
    graph::graph::{Graph, NodeId, RelationshipId},
    graph::graphblas::matrix::{Matrix, New, Set},
    graph::graphblas::tensor::GrB_INDEX_MAX,
    runtime::{ordermap::OrderMap, pending::PendingRelationship, value::Value},
};
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisResult, RedisString, RedisValue};
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Number of records to process between yields to Redis.
const YIELD_INTERVAL: usize = 10_000;

// Binary property type markers (matching Python bulk loader's TYPE enum)
const BI_NULL: u8 = 0;
const BI_BOOL: u8 = 1;
const BI_DOUBLE: u8 = 2;
const BI_STRING: u8 = 3;
const BI_LONG: u8 = 4;
const BI_ARRAY: u8 = 5;

fn read_cstring<'a>(
    data: &'a [u8],
    idx: &mut usize,
) -> Result<&'a str, String> {
    let start = *idx;
    let end = data[start..]
        .iter()
        .position(|&b| b == 0)
        .ok_or("unterminated string in bulk data")?
        + start;
    let s = std::str::from_utf8(&data[start..end])
        .map_err(|e| format!("invalid UTF-8 in bulk data: {e}"))?;
    *idx = end + 1; // skip null terminator
    Ok(s)
}

fn read_u32_ne(
    data: &[u8],
    idx: &mut usize,
) -> Result<u32, String> {
    if *idx + 4 > data.len() {
        return Err("unexpected end of bulk data reading u32".to_string());
    }
    let val = u32::from_ne_bytes(data[*idx..*idx + 4].try_into().unwrap());
    *idx += 4;
    Ok(val)
}

fn read_u64_ne(
    data: &[u8],
    idx: &mut usize,
) -> Result<u64, String> {
    if *idx + 8 > data.len() {
        return Err("unexpected end of bulk data reading u64".to_string());
    }
    let val = u64::from_ne_bytes(data[*idx..*idx + 8].try_into().unwrap());
    *idx += 8;
    Ok(val)
}

fn read_i64_ne(
    data: &[u8],
    idx: &mut usize,
) -> Result<i64, String> {
    if *idx + 8 > data.len() {
        return Err("unexpected end of bulk data reading i64".to_string());
    }
    let val = i64::from_ne_bytes(data[*idx..*idx + 8].try_into().unwrap());
    *idx += 8;
    Ok(val)
}

fn read_f64_ne(
    data: &[u8],
    idx: &mut usize,
) -> Result<f64, String> {
    if *idx + 8 > data.len() {
        return Err("unexpected end of bulk data reading f64".to_string());
    }
    let val = f64::from_ne_bytes(data[*idx..*idx + 8].try_into().unwrap());
    *idx += 8;
    Ok(val)
}

fn read_property(
    data: &[u8],
    idx: &mut usize,
) -> Result<Value, String> {
    if *idx >= data.len() {
        return Err("unexpected end of bulk data reading property type".to_string());
    }
    let type_byte = data[*idx];
    *idx += 1;

    match type_byte {
        BI_NULL => Ok(Value::Null),
        BI_BOOL => {
            if *idx >= data.len() {
                return Err("unexpected end of bulk data reading bool".to_string());
            }
            let val = data[*idx] != 0;
            *idx += 1;
            Ok(Value::Bool(val))
        }
        BI_DOUBLE => {
            let val = read_f64_ne(data, idx)?;
            Ok(Value::Float(val))
        }
        BI_LONG => {
            let val = read_i64_ne(data, idx)?;
            Ok(Value::Int(val))
        }
        BI_STRING => {
            let s = read_cstring(data, idx)?;
            Ok(Value::String(Arc::new(s.to_string())))
        }
        BI_ARRAY => {
            let len = read_i64_ne(data, idx)?;
            if len < 0 {
                return Err(format!("negative array length in bulk data: {len}"));
            }
            let len = len as usize;
            let mut arr = thin_vec::ThinVec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_property(data, idx)?);
            }
            Ok(Value::List(Arc::new(arr)))
        }
        _ => Err(format!("unknown bulk property type: {type_byte}")),
    }
}

/// Parse header: label names (colon-separated) + property names
fn parse_header(
    data: &[u8],
    idx: &mut usize,
) -> Result<(Vec<String>, Vec<Arc<String>>), String> {
    // Read colon-delimited label/type names
    let labels_str = read_cstring(data, idx)?;
    let labels: Vec<String> = labels_str
        .split(':')
        .map(std::string::ToString::to_string)
        .collect();

    // Read property count (4 bytes)
    let prop_count = read_u32_ne(data, idx)? as usize;

    // Read property names
    let mut prop_names = Vec::with_capacity(prop_count);
    for _ in 0..prop_count {
        let name = read_cstring(data, idx)?;
        prop_names.push(Arc::new(name.to_string()));
    }

    Ok((labels, prop_names))
}

fn flush_node_batch(
    g: &mut Graph,
    nodes_bitmap: &mut RoaringTreemap,
    label_matrix: &mut Matrix,
    attrs: &mut FxHashMap<u64, OrderMap<Arc<String>, Value>>,
) -> Result<(), String> {
    if nodes_bitmap.is_empty() {
        return Ok(());
    }

    g.create_nodes(nodes_bitmap);

    let mut index_add_docs: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
    g.set_nodes_labels(label_matrix, &mut index_add_docs);

    if !attrs.is_empty() {
        g.set_nodes_attributes(attrs, &mut index_add_docs)?;
    }

    nodes_bitmap.clear();
    *label_matrix = Matrix::new(GrB_INDEX_MAX, GrB_INDEX_MAX);
    attrs.clear();

    Ok(())
}

fn process_node_token(
    ctx: &Context,
    g: &mut Graph,
    data: &[u8],
    node_ids: &[NodeId],
    node_id_cursor: &mut usize,
) -> Result<(), String> {
    let mut idx = 0;
    let (labels, prop_names) = parse_header(data, &mut idx)?;

    // Get or create label IDs
    let label_ids: Vec<_> = labels.iter().map(|l| g.get_label_id_mut(l)).collect();

    // Process node records in batches, flushing to graph and yielding periodically
    let mut nodes_bitmap = RoaringTreemap::new();
    let mut label_matrix = Matrix::new(GrB_INDEX_MAX, GrB_INDEX_MAX);
    let mut attrs: FxHashMap<u64, OrderMap<Arc<String>, Value>> = FxHashMap::default();

    let mut records_in_batch = 0usize;
    while idx < data.len() {
        if *node_id_cursor >= node_ids.len() {
            return Err("bulk data contains more node records than advertised count".to_string());
        }
        let node_id = node_ids[*node_id_cursor];
        *node_id_cursor += 1;
        let raw_id: u64 = node_id.into();
        nodes_bitmap.insert(raw_id);

        // Set labels
        for &lid in &label_ids {
            label_matrix.set(raw_id, lid.0 as u64, true);
        }

        // Read properties
        let mut node_attrs = OrderMap::default();
        for prop_name in &prop_names {
            let val = read_property(data, &mut idx)?;
            if !matches!(val, Value::Null) {
                node_attrs.insert(prop_name.clone(), val);
            }
        }

        if !node_attrs.is_empty() {
            attrs.insert(raw_id, node_attrs);
        }

        records_in_batch += 1;
        if records_in_batch >= YIELD_INTERVAL {
            flush_node_batch(g, &mut nodes_bitmap, &mut label_matrix, &mut attrs)?;
            records_in_batch = 0;
            unsafe { ffi::yield_ctx(ctx.ctx, ffi::YIELD_FLAG_CLIENTS) };
        }
    }

    // Flush remaining
    flush_node_batch(g, &mut nodes_bitmap, &mut label_matrix, &mut attrs)?;

    Ok(())
}

fn flush_edge_batch(
    g: &mut Graph,
    rels: &mut FxHashMap<RelationshipId, PendingRelationship>,
    rel_attrs: &mut FxHashMap<u64, OrderMap<Arc<String>, Value>>,
    index_add_edge_docs: &mut FxHashMap<u64, RoaringTreemap>,
) -> Result<(), String> {
    if rels.is_empty() {
        return Ok(());
    }

    g.create_relationships(rels);

    if !rel_attrs.is_empty() {
        g.set_relationships_attributes(rel_attrs, index_add_edge_docs)?;
    }

    rels.clear();
    rel_attrs.clear();

    Ok(())
}

fn process_edge_token(
    ctx: &Context,
    g: &mut Graph,
    data: &[u8],
    rel_ids: &[RelationshipId],
    rel_id_cursor: &mut usize,
) -> Result<(), String> {
    let mut idx = 0;
    let (type_names, prop_names) = parse_header(data, &mut idx)?;

    // Edges must have exactly one type
    if type_names.len() != 1 {
        return Err(format!(
            "edges must have exactly one type, got {}",
            type_names.len()
        ));
    }
    let type_name = Arc::new(type_names[0].clone());
    // Ensure the relationship type exists
    g.get_type_id_mut(&type_name);

    // Process edge records in batches, flushing to graph and yielding periodically
    let mut rels: FxHashMap<RelationshipId, PendingRelationship> = FxHashMap::default();
    let mut rel_attrs: FxHashMap<u64, OrderMap<Arc<String>, Value>> = FxHashMap::default();
    let mut index_add_edge_docs: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();

    let mut records_in_batch = 0usize;
    while idx < data.len() {
        if *rel_id_cursor >= rel_ids.len() {
            return Err("bulk data contains more edge records than advertised count".to_string());
        }
        let src_id = read_u64_ne(data, &mut idx)?;
        let dst_id = read_u64_ne(data, &mut idx)?;

        let rel_id = rel_ids[*rel_id_cursor];
        *rel_id_cursor += 1;

        let pending = PendingRelationship::new(
            NodeId::from(src_id),
            NodeId::from(dst_id),
            type_name.clone(),
        );
        rels.insert(rel_id, pending);

        // Read properties
        let mut edge_attrs = OrderMap::default();
        for prop_name in &prop_names {
            let val = read_property(data, &mut idx)?;
            if !matches!(val, Value::Null) {
                edge_attrs.insert(prop_name.clone(), val);
            }
        }

        if !edge_attrs.is_empty() {
            rel_attrs.insert(rel_id.into(), edge_attrs);
        }

        records_in_batch += 1;
        if records_in_batch >= YIELD_INTERVAL {
            flush_edge_batch(g, &mut rels, &mut rel_attrs, &mut index_add_edge_docs)?;
            records_in_batch = 0;
            unsafe { ffi::yield_ctx(ctx.ctx, ffi::YIELD_FLAG_CLIENTS) };
        }
    }

    // Flush remaining
    flush_edge_batch(g, &mut rels, &mut rel_attrs, &mut index_add_edge_docs)?;

    Ok(())
}

pub fn graph_bulk_insert(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    if args.len() < 3 {
        return Err(redis_module::RedisError::WrongArity);
    }

    let mut args = args.into_iter().skip(1);
    let key_str = args.next_arg()?;

    // Check for BEGIN token
    let next = args.next_str()?;
    let (begin, node_count_str) = if next == "BEGIN" {
        (true, args.next_str()?)
    } else {
        (false, next)
    };

    // Get or create graph - must happen BEFORE parsing counts (matches C behavior)
    let key = ctx.open_key_writable(&key_str);
    let graph = if begin {
        // BEGIN: verify key doesn't exist, then create
        if key
            .get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)?
            .is_some()
        {
            return Err(redis_module::RedisError::String(format!(
                "Graph with name '{key_str}' cannot be created, as key '{key_str}' already exists."
            )));
        }
        let g = Arc::new(RwLock::new(ThreadedGraph::new(
            *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
            &key_str.to_string(),
        )));
        key.set_value(&GRAPH_TYPE, g.clone())?;
        g
    } else {
        // No BEGIN: graph must already exist
        if let Some(g) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
            g.clone()
        } else {
            return Err(redis_module::RedisError::Str(
                "ERR Invalid graph operation on empty key",
            ));
        }
    };

    // Parse counts
    let node_count: usize = node_count_str
        .parse()
        .map_err(|_| redis_module::RedisError::Str("Error parsing node count."))?;
    let edge_count: usize = args
        .next_str()?
        .parse()
        .map_err(|_| redis_module::RedisError::Str("Error parsing relation count."))?;
    let node_token_count: usize = args
        .next_str()?
        .parse()
        .map_err(|_| redis_module::RedisError::Str("Error parsing node token count."))?;
    let rel_token_count: usize = args
        .next_str()?
        .parse()
        .map_err(|_| redis_module::RedisError::Str("Error parsing relation token count."))?;

    // Collect remaining binary token args
    let tokens: Vec<RedisString> = args.collect();
    if tokens.len() != node_token_count + rel_token_count {
        return Err(redis_module::RedisError::Str(
            "Bulk insert format error, token count mismatch.",
        ));
    }

    // Acquire MVCC write lock
    let mut tg = graph.write();
    let Some(g_arc) = tg.graph.write() else {
        return Err(redis_module::RedisError::String(
            "ERR write lock unavailable".to_string(),
        ));
    };

    let result: Result<(), String> = (|| {
        let mut g = g_arc.borrow_mut();

        // Reserve node and relationship IDs upfront
        let node_ids = g.reserve_nodes(node_count);
        let rel_ids = g.reserve_relationships(edge_count);

        let mut node_id_cursor = 0usize;
        let mut rel_id_cursor = 0usize;

        // Process node tokens, yielding to Redis periodically
        // so the server stays responsive (e.g. PING)
        for i in 0..node_token_count {
            let data = tokens[i].as_slice();
            process_node_token(ctx, &mut g, data, &node_ids, &mut node_id_cursor)?;
        }

        // Process edge tokens
        for i in 0..rel_token_count {
            let data = tokens[node_token_count + i].as_slice();
            process_edge_token(ctx, &mut g, data, &rel_ids, &mut rel_id_cursor)?;
        }

        // Commit attribute stores
        g.commit_attrs()?;

        Ok(())
    })();

    match result {
        Ok(()) => {
            tg.graph.commit(g_arc);
            let value = tg.graph.read().borrow().maybe_flush_caches();
            if let Err(e) = value {
                ctx.log_warning(&format!("FalkorDB: cache flush failed: {e}"));
            }
            ctx.replicate_verbatim();

            let reply = format!("{node_count} nodes created, {edge_count} relations created");
            Ok(RedisValue::SimpleString(reply))
        }
        Err(e) => {
            tg.graph.rollback();
            Err(redis_module::RedisError::String(format!(
                "ERR bulk insert failed: {e}"
            )))
        }
    }
}
