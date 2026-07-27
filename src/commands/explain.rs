//! `GRAPH.EXPLAIN` command handler.
//!
//! Parses a query and returns the execution plan tree without running the query.
//!
//! ## Output shape
//! The response is a linearized DFS traversal of the plan tree, where each
//! operator string is left-indented by depth to preserve hierarchy.
//!
//! ```text
//! Results
//!  ├─ Project
//!  │   └─ Filter
//!  │       └─ NodeByLabelScan
//! ```
//!
//! Like `GRAPH.QUERY`, planning happens on the thread pool with the client blocked;
//! the main thread only resolves the graph key. Building the plan inline would
//! deadlock: the handler holds the GIL while waiting for the graph read lock, and a
//! committing write holds the write lock while waiting for the GIL (issue #726).

use crate::{
    commands::EMPTY_KEY_ERR,
    graph_core::{BlockedClient, ThreadedGraph, ffi},
    redis_type::GRAPH_TYPE,
};
use graph::{graph::graph::Plan, threadpool::spawn};
use orx_tree::{Dfs, NodeRef};
use parking_lot::RwLock;
use redis_module::{
    Context, ContextFlags, NextArg, RedisError, RedisResult, RedisString, RedisValue, raw,
};
use std::{os::raw::c_char, sync::Arc};

/// Acquire the graph read lock, build the plan, reply the linearized tree. Runs on a
/// worker thread, or synchronously for MULTI/REPLICATED.
fn explain(
    ctx: &Context,
    graph: &Arc<RwLock<ThreadedGraph>>,
    query: &str,
) -> RedisResult {
    // L1-read for exactly the plan build, through a session: the optimizer consults
    // the index to pick scans, and a session is what publishes the lock mode that the
    // GIL lock-order assertion reads (#726).
    let Plan { plan, .. } = {
        let session = crate::query_session::QuerySession::begin(graph);
        session.with_graph(|tg| tg.graph.read().borrow().get_plan(query))
    }
    .map_err(RedisError::String)?;
    let ops = plan.root().indices::<Dfs>().collect::<Vec<_>>();
    raw::reply_with_array(ctx.ctx, ops.len() as _);
    for idx in ops {
        let node = plan.node(idx);
        let depth = node.depth();
        let str = format!("{}{}", " ".repeat(depth * 4), plan.node(idx).data());
        raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    }
    RedisResult::Ok(RedisValue::NoReply)
}

pub fn graph_explain(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let key = args.next_arg()?;
    let query = args.next_str()?;

    let key = ctx.open_key(&key);
    let Some(graph) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? else {
        return EMPTY_KEY_ERR;
    };
    let graph = graph.clone();

    // Blocking clients are not allowed inside MULTI/EXEC, and replicated
    // commands must complete before the handler returns (same rules as
    // GRAPH.QUERY) — run synchronously in those cases.
    if ctx.get_flags().contains(ContextFlags::MULTI)
        || ctx.get_flags().contains(ContextFlags::REPLICATED)
    {
        return explain(ctx, &graph, query);
    }

    // Run on the thread pool like GRAPH.QUERY — see the module docs for why inline
    // on the main thread deadlocks (#726).
    let bc = unsafe { BlockedClient::new(ctx.ctx) };
    let query: Arc<str> = Arc::from(query);
    spawn(
        move || {
            let ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
            let ctx = Context::new(ctx);
            if let Err(err) = explain(&ctx, &graph, &query) {
                let cerr = ffi::sanitise_error(err.to_string());
                unsafe { ffi::reply_error(ctx.ctx, cerr.as_ptr()) };
            }
            drop(bc);
            unsafe { ffi::free_thread_safe_context(ctx.ctx) };
        },
        None,
    );

    RedisResult::Ok(RedisValue::NoReply)
}
