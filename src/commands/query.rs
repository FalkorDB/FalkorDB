//! `GRAPH.QUERY` command handler.
//!
//! Executes read/write Cypher queries, auto-creates graphs when missing,
//! and supports compact output and memory-tracking flags.
//!
//! ## Execution flow
//! ```text
//! GRAPH.QUERY key query [--compact] [--track-memory]
//!        |
//!        +--> parse flags
//!        +--> open writable key
//!        +--> create graph if missing
//!        +--> delegate to graph_core::query_mut(..., write=true)
//!        +--> return NoReply (client unblocked asynchronously later)
//! ```
//!
//! The handler is intentionally thin: it validates command arguments and
//! chooses the target graph, while runtime execution is centralized in
//! `graph_core`.

use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{ThreadedGraph, query_mut},
    redis_type::GRAPH_TYPE,
};
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisResult, RedisString, raw};
use std::ffi::CString;
use std::sync::Arc;
#[cfg(feature = "fuzz")]
use std::sync::atomic::{AtomicI32, Ordering};
#[cfg(feature = "fuzz")]
use std::{fs::File, io::Write};

#[cfg(feature = "fuzz")]
static FILE_ID: AtomicI32 = AtomicI32::new(0);

#[allow(unused_imports)]
pub fn graph_query(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let key_str = args.next_arg()?;
    let query = args.next_str()?;

    #[cfg(feature = "fuzz")]
    {
        let id = FILE_ID.fetch_add(1, Ordering::Relaxed);
        let mut file = File::create(format!("fuzz/corpus/fuzz_target_runtime/output{id}.txt"))?;
        file.write_all(query.as_bytes())?;
    }

    let mut compact = false;
    let mut track_memory = false;
    let mut version_check: Option<u64> = None;
    while let Ok(arg) = args.next_str() {
        if arg == "--compact" {
            compact = true;
        } else if arg == "--track-memory" {
            track_memory = true;
        } else if arg == "version" {
            let ver_str = args.next_str()?;
            version_check = Some(ver_str.parse::<u64>()?);
        }
    }

    // Try read-only key access first to avoid triggering WATCH on existing graphs.
    let read_key = ctx.open_key(&key_str);
    let key_name = Arc::new(key_str.to_string());

    if let Some(graph) = read_key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        let graph = graph.clone();

        // Check version if provided
        if let Some(provided_version) = version_check {
            let current_schema_version = graph.read().graph.read().borrow().schema_version;
            if provided_version != current_schema_version {
                drop(read_key);
                drop(graph);
                // Return array with [error, version]
                raw::reply_with_array(ctx.ctx, 2);
                let err_msg = CString::new("ERR invalid graph version").unwrap();
                raw::reply_with_error(ctx.ctx, err_msg.as_ptr());
                raw::reply_with_long_long(ctx.ctx, current_schema_version as i64);
                return Ok(redis_module::RedisValue::NoReply);
            }
        }

        drop(read_key);
        return query_mut(ctx, &graph, query, compact, true, track_memory, key_name);
    }

    // Graph doesn't exist - open writable key to create it.
    drop(read_key);
    let key = ctx.open_key_writable(&key_str);
    // Re-check: another client may have created it between our read and write open.
    if let Some(graph) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        let graph = graph.clone();

        // Check version if provided
        if let Some(provided_version) = version_check {
            let current_schema_version = graph.read().graph.read().borrow().schema_version;
            if provided_version != current_schema_version {
                // Return array with [error, version]
                raw::reply_with_array(ctx.ctx, 2);
                let err_msg = CString::new("ERR invalid graph version").unwrap();
                raw::reply_with_error(ctx.ctx, err_msg.as_ptr());
                raw::reply_with_long_long(ctx.ctx, current_schema_version as i64);
                return Ok(redis_module::RedisValue::NoReply);
            }
        }

        return query_mut(ctx, &graph, query, compact, true, track_memory, key_name);
    }

    let graph = Arc::new(RwLock::new(ThreadedGraph::new(
        *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
        &key_str.to_string(),
    )));

    // For a newly-created graph, the initial schema_version is 0
    if let Some(provided_version) = version_check
        && provided_version != 0
    {
        // Return array with [error, version]
        raw::reply_with_array(ctx.ctx, 2);
        let err_msg = CString::new("ERR invalid graph version").unwrap();
        raw::reply_with_error(ctx.ctx, err_msg.as_ptr());
        raw::reply_with_long_long(ctx.ctx, 0i64);
        return Ok(redis_module::RedisValue::NoReply);
    }

    let result = query_mut(ctx, &graph, query, compact, true, track_memory, key_name);
    key.set_value(&GRAPH_TYPE, graph)?;
    result
}
