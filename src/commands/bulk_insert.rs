use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{BlockedClient, ThreadedGraph, ffi},
    redis_type::GRAPH_TYPE,
};
use graph::{
    graph::graph::{Graph, NodeId, RelationshipId},
    runtime::value::Value,
    threadpool::spawn,
};
use parking_lot::RwLock;
use redis_module::{Context, ContextFlags, NextArg, RedisResult, RedisString, RedisValue, raw};
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
use std::sync::Arc;

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
    // Parse a possibly-nested value without recursion: `BI_ARRAY` elements are
    // expanded onto an explicit work stack instead of recursive calls, so
    // arbitrarily deep attacker-supplied `GRAPH.BULKINSERT` data cannot overflow
    // the call stack (an uncatchable abort) and arrays of any depth are
    // supported. Each frame holds a partially-built array and how many elements
    // it still expects.
    let mut stack: Vec<(thin_vec::ThinVec<Value>, usize)> = Vec::new();

    loop {
        // Decode the next scalar, or open a new array frame and keep reading.
        let mut value = {
            if *idx >= data.len() {
                return Err("unexpected end of bulk data reading property type".to_string());
            }
            let type_byte = data[*idx];
            *idx += 1;

            match type_byte {
                BI_NULL => Value::Null,
                BI_BOOL => {
                    if *idx >= data.len() {
                        return Err("unexpected end of bulk data reading bool".to_string());
                    }
                    let val = data[*idx] != 0;
                    *idx += 1;
                    Value::Bool(val)
                }
                BI_DOUBLE => Value::Float(read_f64_ne(data, idx)?),
                BI_LONG => Value::Int(read_i64_ne(data, idx)?),
                BI_STRING => Value::String(Arc::new(read_cstring(data, idx)?.to_string())),
                BI_ARRAY => {
                    let len = read_i64_ne(data, idx)?;
                    if len < 0 {
                        return Err(format!("negative array length in bulk data: {len}"));
                    }
                    let len = len as usize;
                    // Cap the pre-allocation to the bytes that remain: every
                    // element consumes at least one byte from `data`, so a `len`
                    // larger than the remaining input is malformed and must not
                    // drive a huge up-front allocation (memory-exhaustion DoS).
                    let cap = len.min(data.len().saturating_sub(*idx));
                    let arr = thin_vec::ThinVec::with_capacity(cap);
                    if len == 0 {
                        Value::List(Arc::new(arr))
                    } else {
                        stack.push((arr, len));
                        continue;
                    }
                }
                _ => return Err(format!("unknown bulk property type: {type_byte}")),
            }
        };

        // Attach the finished value to the innermost open array, completing and
        // bubbling up any array whose element count is now satisfied.
        loop {
            match stack.last_mut() {
                None => return Ok(value),
                Some((arr, remaining)) => {
                    arr.push(value);
                    *remaining -= 1;
                    if *remaining == 0 {
                        let (arr, _) = stack.pop().unwrap();
                        value = Value::List(Arc::new(arr));
                    } else {
                        break;
                    }
                }
            }
        }
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

/// Yield to Redis if running on the main thread (non-null context).
/// No-op when called from a background thread (null context).
#[inline]
unsafe fn maybe_yield(raw_ctx: *mut raw::RedisModuleCtx) {
    if !raw_ctx.is_null() {
        unsafe { ffi::yield_ctx(raw_ctx, ffi::YIELD_FLAG_CLIENTS) };
    }
}

fn process_node_token(
    g: &mut Graph,
    data: &[u8],
    node_ids: &[NodeId],
    node_id_cursor: &mut usize,
    raw_ctx: *mut raw::RedisModuleCtx,
) -> Result<(), String> {
    let mut idx = 0;
    let (labels, prop_names) = parse_header(data, &mut idx)?;

    // Get or create label IDs
    let label_ids: Vec<_> = labels.iter().map(|l| g.get_label_id_mut(l)).collect();

    // Pre-resolve property name → attribute index once
    let attr_ids: Vec<u16> = prop_names
        .iter()
        .map(|name| g.get_or_create_node_attr_id(name))
        .collect();

    // Collect all node data first, then insert at the end
    let mut nodes_bitmap = RoaringTreemap::new();
    let mut label_rows: Vec<u64> = Vec::new();
    let mut label_cols: Vec<u64> = Vec::new();
    let mut resolved_attrs: Vec<(u64, Vec<(u16, Value)>)> = Vec::new();

    while idx < data.len() {
        if *node_id_cursor >= node_ids.len() {
            return Err("bulk data contains more node records than advertised count".to_string());
        }
        let node_id = node_ids[*node_id_cursor];
        *node_id_cursor += 1;
        let raw_id: u64 = node_id.into();
        nodes_bitmap.insert(raw_id);

        for &lid in &label_ids {
            label_rows.push(raw_id);
            label_cols.push(lid.0 as u64);
        }

        if !attr_ids.is_empty() {
            let mut entries: Vec<(u16, Value)> = Vec::with_capacity(attr_ids.len());
            for &attr_id in &attr_ids {
                let val = read_property(data, &mut idx)?;
                if !matches!(val, Value::Null) {
                    entries.push((attr_id, val));
                }
            }
            if !entries.is_empty() {
                resolved_attrs.push((raw_id, entries));
            }
        }
    }

    if nodes_bitmap.is_empty() {
        return Ok(());
    }

    g.create_nodes(&nodes_bitmap);
    unsafe { maybe_yield(raw_ctx) };

    let mut index_add_docs: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
    g.set_nodes_labels_bulk(&label_rows, &label_cols, &mut index_add_docs);
    unsafe { maybe_yield(raw_ctx) };

    if !resolved_attrs.is_empty() {
        g.import_node_attrs_resolved(&mut resolved_attrs);
        unsafe { maybe_yield(raw_ctx) };
    }

    Ok(())
}

fn process_edge_token(
    g: &mut Graph,
    data: &[u8],
    rel_ids: &[RelationshipId],
    rel_id_cursor: &mut usize,
    raw_ctx: *mut raw::RedisModuleCtx,
) -> Result<(), String> {
    let mut idx = 0;
    let (type_names, prop_names) = parse_header(data, &mut idx)?;

    if type_names.len() != 1 {
        return Err(format!(
            "edges must have exactly one type, got {}",
            type_names.len()
        ));
    }
    let type_name = Arc::new(type_names[0].clone());
    g.get_type_id_mut(&type_name);

    let attr_ids: Vec<u16> = prop_names
        .iter()
        .map(|name| g.get_or_create_rel_attr_id(name))
        .collect();

    // Collect all edge data first, then bulk-insert at the end
    let mut srcs: Vec<u64> = Vec::new();
    let mut dsts: Vec<u64> = Vec::new();
    let mut edge_ids: Vec<u64> = Vec::new();
    let mut resolved_rel_attrs: Vec<(u64, Vec<(u16, Value)>)> = Vec::new();

    while idx < data.len() {
        if *rel_id_cursor >= rel_ids.len() {
            return Err("bulk data contains more edge records than advertised count".to_string());
        }
        let src_id = read_u64_ne(data, &mut idx)?;
        let dst_id = read_u64_ne(data, &mut idx)?;

        let rel_id = rel_ids[*rel_id_cursor];
        *rel_id_cursor += 1;

        srcs.push(src_id);
        dsts.push(dst_id);
        edge_ids.push(rel_id.into());

        if !attr_ids.is_empty() {
            let mut entries: Vec<(u16, Value)> = Vec::with_capacity(attr_ids.len());
            for &attr_id in &attr_ids {
                let val = read_property(data, &mut idx)?;
                if !matches!(val, Value::Null) {
                    entries.push((attr_id, val));
                }
            }
            if !entries.is_empty() {
                resolved_rel_attrs.push((rel_id.into(), entries));
            }
        }
    }

    if srcs.is_empty() {
        return Ok(());
    }

    g.create_relationships_bulk(&type_name, &srcs, &dsts, &edge_ids);
    unsafe { maybe_yield(raw_ctx) };

    if !resolved_rel_attrs.is_empty() {
        g.import_relationship_attrs_resolved(&mut resolved_rel_attrs);
        unsafe { maybe_yield(raw_ctx) };
    }

    Ok(())
}

/// Process bulk insert on a background thread (no yield needed — main thread handles PING).
fn bulk_insert_sync(
    g: &mut Graph,
    tokens: &[&[u8]],
    node_count: usize,
    edge_count: usize,
    node_token_count: usize,
    rel_token_count: usize,
) -> Result<(), String> {
    let node_ids = g.reserve_nodes(node_count);
    let rel_ids = g.reserve_relationships(edge_count);
    let mut node_id_cursor = 0usize;
    let mut rel_id_cursor = 0usize;

    let null_ctx = std::ptr::null_mut();
    for token in tokens.iter().take(node_token_count) {
        process_node_token(g, token, &node_ids, &mut node_id_cursor, null_ctx)?;
    }

    for token in tokens.iter().skip(node_token_count).take(rel_token_count) {
        process_edge_token(g, token, &rel_ids, &mut rel_id_cursor, null_ctx)?;
    }

    g.commit_attrs()?;
    // Flush delta-plus into base to prevent large dp from slowing subsequent commands
    g.flush_for_bulk();
    Ok(())
}

/// Process bulk insert synchronously with periodic yield for PING handling.
fn bulk_insert_sync_yield(
    g: &mut Graph,
    tokens: &[&[u8]],
    node_count: usize,
    edge_count: usize,
    node_token_count: usize,
    rel_token_count: usize,
    raw_ctx: *mut raw::RedisModuleCtx,
) -> Result<(), String> {
    let node_ids = g.reserve_nodes(node_count);
    let rel_ids = g.reserve_relationships(edge_count);
    let mut node_id_cursor = 0usize;
    let mut rel_id_cursor = 0usize;

    for token in tokens.iter().take(node_token_count) {
        process_node_token(g, token, &node_ids, &mut node_id_cursor, raw_ctx)?;
        // Yield to let Redis process PING from other clients
        unsafe { maybe_yield(raw_ctx) };
    }

    for token in tokens.iter().skip(node_token_count).take(rel_token_count) {
        process_edge_token(g, token, &rel_ids, &mut rel_id_cursor, raw_ctx)?;
        unsafe { maybe_yield(raw_ctx) };
    }

    g.commit_attrs()?;
    // Flush delta-plus into base to prevent O(N²) dp accumulation across commands
    g.flush_for_bulk();
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

    // Get or create graph
    let key = ctx.open_key_writable(&key_str);
    let graph = if begin {
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
        crate::graph_core::register_graph(key_str.to_string(), g.clone());
        g
    } else if let Some(g) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        g.clone()
    } else {
        return Err(redis_module::RedisError::Str(
            "ERR Invalid graph operation on empty key",
        ));
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
    let token_strings: Vec<RedisString> = args.collect();
    if token_strings.len() != node_token_count + rel_token_count {
        return Err(redis_module::RedisError::Str(
            "Bulk insert format error, token count mismatch.",
        ));
    }

    // Inside MULTI/EXEC: blocking commands are not allowed, run synchronously
    // with RM_Yield to let Redis process PING between operations.
    if ctx.get_flags().contains(ContextFlags::MULTI) {
        let tokens: Vec<&[u8]> = token_strings
            .iter()
            .map(redis_module::RedisString::as_slice)
            .collect();
        let mut tg = graph.write();
        let Some(g_arc) = tg.graph.write() else {
            return Err(redis_module::RedisError::String(
                "ERR write lock unavailable".to_string(),
            ));
        };
        let result = {
            let mut g = g_arc.borrow_mut();
            bulk_insert_sync_yield(
                &mut g,
                &tokens,
                node_count,
                edge_count,
                node_token_count,
                rel_token_count,
                ctx.ctx,
            )
        };
        return match result {
            Ok(()) => {
                tg.graph.commit(g_arc);
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
        };
    }

    // Block the client and process on a background thread so the main
    // Redis thread stays free to handle PING and other commands.
    let bc = unsafe { BlockedClient::new(ctx.ctx) };
    let token_data: Vec<Vec<u8>> = token_strings
        .iter()
        .map(|rs| rs.as_slice().to_vec())
        .collect();
    spawn(
        move || {
            let mut tg = graph.write();
            let Some(g_arc) = tg.graph.write() else {
                let ts_ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
                unsafe { ffi::lock_thread_safe_ctx(ts_ctx) };
                let cerr = ffi::sanitise_error("ERR write lock unavailable");
                unsafe { ffi::reply_error(ts_ctx, cerr.as_ptr()) };
                unsafe { ffi::unlock_thread_safe_ctx(ts_ctx) };
                drop(bc);
                unsafe { ffi::free_thread_safe_context(ts_ctx) };
                return;
            };
            let result = {
                let mut g = g_arc.borrow_mut();
                let tokens: Vec<&[u8]> = token_data.iter().map(std::vec::Vec::as_slice).collect();
                bulk_insert_sync(
                    &mut g,
                    &tokens,
                    node_count,
                    edge_count,
                    node_token_count,
                    rel_token_count,
                )
            };
            let ts_ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
            match result {
                Ok(()) => {
                    tg.graph.commit(g_arc);
                    unsafe { ffi::lock_thread_safe_ctx(ts_ctx) };
                    raw::replicate_verbatim(ts_ctx);
                    let reply =
                        format!("{node_count} nodes created, {edge_count} relations created");
                    let c_reply = std::ffi::CString::new(reply).expect("reply has no NUL bytes");
                    raw::reply_with_simple_string(ts_ctx, c_reply.as_ptr());
                    unsafe { ffi::unlock_thread_safe_ctx(ts_ctx) };
                }
                Err(e) => {
                    tg.graph.rollback();
                    unsafe { ffi::lock_thread_safe_ctx(ts_ctx) };
                    let cerr = ffi::sanitise_error(format!("ERR bulk insert failed: {e}"));
                    unsafe { ffi::reply_error(ts_ctx, cerr.as_ptr()) };
                    unsafe { ffi::unlock_thread_safe_ctx(ts_ctx) };
                }
            }
            drop(bc);
            unsafe { ffi::free_thread_safe_context(ts_ctx) };
        },
        None,
    );
    Ok(RedisValue::NoReply)
}
