use crate::query_session::{QuerySession, hold_gil};
use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{BlockedClient, ThreadedGraph, ffi},
    redis_type::GRAPH_TYPE,
    telemetry,
};
use graph::{
    graph::graph::{Graph, NodeId, RelationshipId},
    identifier_limits::validate_identifier_len,
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
///
/// `entity` names the kind of identifier the leading names carry — a label
/// for a node token, a relationship type for an edge token — so an overlong
/// name is reported the way the query paths report it.
fn parse_header(
    data: &[u8],
    idx: &mut usize,
    entity: &str,
) -> Result<(Vec<String>, Vec<Arc<String>>), String> {
    // Read colon-delimited label/type names
    let labels_str = read_cstring(data, idx)?;
    let labels: Vec<String> = labels_str
        .split(':')
        .map(std::string::ToString::to_string)
        .collect();
    for label in &labels {
        validate_identifier_len(label, entity)?;
    }

    // Read property count (4 bytes)
    let prop_count = read_u32_ne(data, idx)? as usize;

    // Read property names
    let mut prop_names = Vec::with_capacity(prop_count);
    for _ in 0..prop_count {
        let name = read_cstring(data, idx)?;
        validate_identifier_len(name, "Property name")?;
        prop_names.push(Arc::new(name.to_string()));
    }

    Ok((labels, prop_names))
}

/// Drop the graph key that a failed `BEGIN` batch had just created.
///
/// Mirrors C's bulk-insert cleanup (`cmd_bulk_insert.c`): a first batch that
/// fails must leave no key behind, otherwise re-running the loader against a
/// corrected input trips the "already exists" guard instead of retrying.
///
/// Requires the GIL — the caller either runs on the main Redis thread or holds
/// it explicitly.
fn discard_created_graph(
    ctx: &Context,
    key_str: &RedisString,
) {
    telemetry::delete_stream(ctx, &key_str.to_string());
    let key = ctx.open_key_writable(key_str);
    let _ = key.delete();
}

/// Yield to Redis if running on the main thread (non-null context).
/// No-op when called from a background thread (null context).
#[inline]
unsafe fn maybe_yield(raw_ctx: *mut raw::RedisModuleCtx) {
    if !raw_ctx.is_null() {
        unsafe { ffi::yield_ctx(raw_ctx, ffi::YIELD_FLAG_CLIENTS) };
    }
}

/// Index documents collected while processing bulk tokens.
///
/// RediSearch is not versioned, so anything published into it cannot be undone by the
/// MVCC rollback that a later failing token triggers. Accumulate across the whole insert
/// and publish once, on the success path only — otherwise a batch that fails midway
/// leaves documents behind for entities that were never committed (and whose ids may be
/// handed to different entities later).
#[derive(Default)]
struct BulkIndexDocs {
    nodes: FxHashMap<u64, RoaringTreemap>,
    edges: FxHashMap<u64, RoaringTreemap>,
}

impl BulkIndexDocs {
    /// Publish into the indexes. Call only once every token has succeeded, and while
    /// `g` is still the un-published fork — a committed version may be borrowed by
    /// concurrent readers.
    fn publish(
        &mut self,
        g: &mut Graph,
    ) {
        if !self.nodes.is_empty() {
            g.commit_index(&mut self.nodes, &mut FxHashMap::default());
        }
        if !self.edges.is_empty() {
            g.commit_edge_index(&mut self.edges, &mut FxHashMap::default());
        }
    }
}

fn process_node_token(
    g: &mut Graph,
    data: &[u8],
    node_ids: &[NodeId],
    node_id_cursor: &mut usize,
    raw_ctx: *mut raw::RedisModuleCtx,
    docs: &mut BulkIndexDocs,
) -> Result<(), String> {
    let mut idx = 0;
    let (labels, prop_names) = parse_header(data, &mut idx, "Label name")?;

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

    g.set_nodes_labels_bulk(&label_rows, &label_cols, &mut docs.nodes);
    unsafe { maybe_yield(raw_ctx) };

    // `import_node_attrs_resolved` marks these nodes for indexing, collecting into the
    // insert-wide accumulator that `BulkIndexDocs::publish` flushes once every token has
    // succeeded. Nothing else on this path would: GRAPH.BULK does not run the post-load
    // `populate_indexes_sync` rebuild (that is the RDB / replica path).
    //
    // `set_nodes_labels_bulk` above also collects, but only for nodes that already have
    // attributes — none do at this point — so it contributes nothing here and the two do
    // not double up.
    if !resolved_attrs.is_empty() {
        g.import_node_attrs_resolved(&mut resolved_attrs, &label_ids, &mut docs.nodes);
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
    docs: &mut BulkIndexDocs,
) -> Result<(), String> {
    let mut idx = 0;
    let (type_names, prop_names) = parse_header(data, &mut idx, "Relationship type")?;

    if type_names.len() != 1 {
        return Err(format!(
            "edges must have exactly one type, got {}",
            type_names.len()
        ));
    }
    let type_name = Arc::new(type_names[0].clone());
    let type_id = g.get_type_id_mut(&type_name);

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
        // Same as the node path: collect now, publish once the whole insert has succeeded.
        g.import_relationship_attrs_resolved(&mut resolved_rel_attrs, type_id, &mut docs.edges);
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
    docs: &mut BulkIndexDocs,
) -> Result<(), String> {
    let node_ids = g.reserve_nodes(node_count);
    let rel_ids = g.reserve_relationships(edge_count);
    let mut node_id_cursor = 0usize;
    let mut rel_id_cursor = 0usize;

    let null_ctx = std::ptr::null_mut();
    for token in tokens.iter().take(node_token_count) {
        process_node_token(g, token, &node_ids, &mut node_id_cursor, null_ctx, docs)?;
    }

    for token in tokens.iter().skip(node_token_count).take(rel_token_count) {
        process_edge_token(g, token, &rel_ids, &mut rel_id_cursor, null_ctx, docs)?;
    }

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
    docs: &mut BulkIndexDocs,
) -> Result<(), String> {
    let node_ids = g.reserve_nodes(node_count);
    let rel_ids = g.reserve_relationships(edge_count);
    let mut node_id_cursor = 0usize;
    let mut rel_id_cursor = 0usize;

    for token in tokens.iter().take(node_token_count) {
        process_node_token(g, token, &node_ids, &mut node_id_cursor, raw_ctx, docs)?;
        // Yield to let Redis process PING from other clients
        unsafe { maybe_yield(raw_ctx) };
    }

    for token in tokens.iter().skip(node_token_count).take(rel_token_count) {
        process_edge_token(g, token, &rel_ids, &mut rel_id_cursor, raw_ctx, docs)?;
        unsafe { maybe_yield(raw_ctx) };
    }

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

    // Get or create graph. The key handle is scoped to this block so a later
    // failure can re-open the key to delete it (see `discard_created_graph`)
    // without a second live handle to the same key.
    let graph = {
        let key = ctx.open_key_writable(&key_str);
        if begin {
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
                "ERR another write is in progress, retry the query".to_string(),
            ));
        };
        let mut docs = BulkIndexDocs::default();
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
                &mut docs,
            )
        };
        return match result {
            Ok(()) => {
                // Every token succeeded, so the index documents are safe to publish. Do it
                // while `g_arc` is still the un-published fork: after the swap it may be
                // borrowed by concurrent readers, and on the error arm below it is thrown
                // away — which is precisely what must happen to the documents too.
                docs.publish(&mut g_arc.borrow_mut());
                tg.graph.commit(g_arc);
                ctx.replicate_verbatim();
                let reply = format!("{node_count} nodes created, {edge_count} relations created");
                Ok(RedisValue::SimpleString(reply))
            }
            Err(e) => {
                tg.graph.rollback();
                drop(tg);
                if begin {
                    discard_created_graph(ctx, &key_str);
                }
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
    // `RedisString` is tied to the calling context, so carry the name as a
    // plain `String` for the cleanup path to rebuild on the worker thread.
    let graph_name = key_str.to_string();
    spawn(
        move || {
            let ts_ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
            // Build, commit and replicate under one session, which releases its locks
            // when this block ends — before the client is unblocked below.
            let result: Result<(), String> = 'phase: {
                let session = QuerySession::begin(&graph);
                // Phase 1: build the new version as a reader, GIL-free. What we take
                // here is the MVCC *write slot* — the single-writer marker on
                // `MvccGraph`, not a lock — so if another writer holds it, fail with a
                // retryable error rather than blocking.
                let Some(g_arc) = session.with_graph(|tg| tg.graph.write()) else {
                    break 'phase Err(
                        "ERR another write is in progress, retry the query".to_string()
                    );
                };
                let mut docs = BulkIndexDocs::default();
                let inserted = {
                    let mut g = g_arc.borrow_mut();
                    let tokens: Vec<&[u8]> =
                        token_data.iter().map(std::vec::Vec::as_slice).collect();
                    bulk_insert_sync(
                        &mut g,
                        &tokens,
                        node_count,
                        edge_count,
                        node_token_count,
                        rel_token_count,
                        &mut docs,
                    )
                };
                if let Err(e) = inserted {
                    session.with_graph(|tg| tg.graph.rollback());
                    break 'phase Err(format!("ERR bulk insert failed: {e}"));
                }
                // Phase 2: commit + replicate as a writer, so the commit Arc-swap
                // happens under the GIL and stays fork-safe (#452). Escalation takes
                // the global lock before the write lock, never the reverse (#726).
                if let Err(e) = session.upgrade_to_write() {
                    // Release the MVCC slot from phase 1 — only commit()/rollback()
                    // clear it, so skipping this leaves the graph permanently
                    // unwritable.
                    session.with_graph(|tg| tg.graph.rollback());
                    break 'phase Err(e);
                }
                // Publish the index documents inside the writer scope, matching
                // `CommitOp`: the index is not MVCC, so readers must be excluded while
                // it is mutated, and a read query holds L1-read for its whole session.
                // Published in phase 1 instead, a concurrent reader could see documents
                // for entities that exist only in this uncommitted fork and get ids it
                // cannot resolve against its own snapshot.
                //
                // Every token has succeeded by now, so nothing reaches the index for an
                // insert that later rolls back. The cost is a GIL hold proportional to
                // the number of indexed rows — paid only when the graph actually has an
                // index, since `docs` is otherwise empty.
                docs.publish(&mut g_arc.borrow_mut());
                session
                    .with_graph_mut(|tg| tg.graph.commit(g_arc))
                    .expect("writer mode after upgrade_to_write");
                raw::replicate_verbatim(ts_ctx);
                Ok(())
            };
            match result {
                Ok(()) => {
                    let reply =
                        format!("{node_count} nodes created, {edge_count} relations created");
                    let c_reply = std::ffi::CString::new(reply).expect("reply has no NUL bytes");
                    raw::reply_with_simple_string(ts_ctx, c_reply.as_ptr());
                }
                Err(msg) => {
                    if begin {
                        // The session — and with it the GIL — is gone by now, so
                        // retake the GIL for this keyspace write.
                        let _gil = hold_gil();
                        let cleanup_ctx = Context::new(ts_ctx);
                        let key_name = cleanup_ctx.create_string(graph_name.as_str());
                        discard_created_graph(&cleanup_ctx, &key_name);
                    }
                    let cerr = ffi::sanitise_error(msg);
                    unsafe { ffi::reply_error(ts_ctx, cerr.as_ptr()) };
                }
            }
            drop(bc);
            unsafe { ffi::free_thread_safe_context(ts_ctx) };
        },
        None,
    );
    Ok(RedisValue::NoReply)
}
