//! `GRAPH.RECORD` command handler.
//!
//! Runs a query in recording mode and returns operator-level execution trace
//! data for debugging and testing.
//!
//! ## What is returned
//! ```text
//! [
//!   recorded operator outputs,
//!   plan structure + variable names
//! ]
//! ```
//!
//! Recording mode executes the normal planning/runtime path but captures
//! intermediate environments per operator index to help diagnose planning or
//! runtime mismatches.
//!
//! Like `GRAPH.QUERY`, execution happens on the thread pool with the client
//! blocked; the main thread only resolves (or creates) the graph key.

use crate::query_session::QuerySession;
use crate::{
    config::{CONFIGURATION_CACHE_SIZE, CONFIGURATION_IMPORT_FOLDER},
    graph_core::{BlockedClient, ThreadedGraph, ffi},
    redis_type::GRAPH_TYPE,
    reply::reply_verbose_value,
};
use graph::{
    graph::graph::Plan,
    planner::IR,
    runtime::{
        eval::evaluate_param,
        runtime::{GetVariables, Runtime},
    },
    threadpool::spawn,
};
use orx_tree::{Bfs, Collection, NodeRef};
use parking_lot::RwLock;
use redis_module::{
    Context, ContextFlags, NextArg, RedisError, RedisResult, RedisString, RedisValue, raw,
};
use std::{collections::HashMap, os::raw::c_char, sync::Arc};

#[inline]
fn record_mut(
    ctx: &Context,
    graph: &Arc<RwLock<ThreadedGraph>>,
    query: &str,
) -> RedisResult {
    // A recorded query may be a write, so run it under a real session: it takes the
    // per-graph lock, escalates at the first `Commit` like any write, and mutates a
    // *private* MVCC version. The version is then rolled back rather than committed —
    // RECORD is an introspection command, so it reports the operator trace of a write
    // without letting it land, and never replicates.
    let session = QuerySession::begin(graph);
    let Plan {
        plan, parameters, ..
    } = session
        .with_graph(|tg| tg.graph.read().borrow().get_plan(query))
        .map_err(RedisError::String)?;
    let parameters = parameters
        .into_iter()
        .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
        .collect::<Result<HashMap<_, _>, String>>()
        .map_err(RedisError::String)?;
    let is_write = plan.iter().any(|n| {
        matches!(
            n,
            IR::Commit | IR::CreateIndex { .. } | IR::DropIndex { .. }
        )
    });
    // Writes need the MVCC write slot so they build a private version instead of
    // mutating the published one in place; reads run on the committed snapshot.
    let g = if is_write {
        let Some(g) = session.with_graph(|tg| tg.graph.write()) else {
            return Err(RedisError::String(
                "ERR another write is in progress, retry the query".to_string(),
            ));
        };
        g
    } else {
        session.with_graph(|tg| tg.graph.read())
    };
    let runtime = Runtime::new(
        g,
        parameters,
        is_write,
        plan.clone(),
        true,
        (*CONFIGURATION_IMPORT_FOLDER.lock(ctx)).clone(),
        -1,
        false,
        None,
        0,
        None,
        &session,
    );
    let _ = runtime.query();
    if is_write {
        // Discard the private version and release the slot; index documents published
        // by any inner `Commit` are undone the same way a failed write undoes them.
        let committed = session.with_graph(|tg| tg.graph.read());
        runtime.resync_published_indexes(&committed);
        session.with_graph(|tg| tg.graph.rollback());
    }
    let ids = plan.root().indices::<Bfs>().collect::<Vec<_>>();
    raw::reply_with_array(ctx.ctx, 2);
    raw::reply_with_array(ctx.ctx, runtime.record.borrow().len() as _);
    for (idx, res) in runtime.record.borrow().iter() {
        raw::reply_with_array(ctx.ctx, 3);
        raw::reply_with_long_long(ctx.ctx, ids.iter().position(|id| *id == *idx).unwrap() as _);
        match res {
            Err(err) => {
                raw::reply_with_long_long(ctx.ctx, 0);
                raw::reply_with_string_buffer(ctx.ctx, err.as_ptr().cast::<c_char>(), err.len());
            }
            Ok(row) => {
                raw::reply_with_long_long(ctx.ctx, 1);
                let vars = plan.node(*idx).get_variables();
                raw::reply_with_array(ctx.ctx, vars.len() as _);
                for name in &vars {
                    if row.is_bound_by_id(name.id) {
                        match row.get_by_id(name.id) {
                            None => {
                                raw::reply_with_null(ctx.ctx);
                            }
                            Some(value) => {
                                reply_verbose_value(ctx, &runtime, value);
                            }
                        }
                    } else {
                        raw::reply_with_null(ctx.ctx);
                    }
                }
            }
        }
    }
    drop(runtime);

    raw::reply_with_array(ctx.ctx, ids.len() as _);
    for idx in plan.root().indices::<Bfs>() {
        raw::reply_with_array(ctx.ctx, 4);
        raw::reply_with_long_long(ctx.ctx, ids.iter().position(|id| *id == idx).unwrap() as _);
        match plan.node(idx).parent() {
            Some(parent_idx) => {
                raw::reply_with_long_long(
                    ctx.ctx,
                    ids.iter().position(|id| *id == parent_idx.idx()).unwrap() as _,
                );
            }
            None => {
                raw::reply_with_null(ctx.ctx);
            }
        }
        let node = plan.node(idx).data().to_string();
        raw::reply_with_string_buffer(ctx.ctx, node.as_ptr().cast::<c_char>(), node.len());
        let vars = plan.node(idx).get_variables();
        raw::reply_with_array(ctx.ctx, vars.len() as _);
        for var in vars {
            raw::reply_with_string_buffer(
                ctx.ctx,
                var.as_str().as_ptr().cast::<c_char>(),
                var.as_str().len(),
            );
        }
    }
    Ok(RedisValue::NoReply)
}

pub fn graph_record(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let key_str = args.next_arg()?;
    let query = args.next_str()?;

    let key = ctx.open_key_writable(&key_str);

    let graph = if let Some(graph) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        graph.clone()
    } else {
        let graph = Arc::new(RwLock::new(ThreadedGraph::new(
            *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
            &key_str.to_string(),
        )));
        key.set_value(&GRAPH_TYPE, graph.clone())?;
        crate::graph_core::register_graph(key_str.to_string(), graph.clone());
        graph
    };

    // Blocking clients are not allowed inside MULTI/EXEC, and replicated
    // commands must complete before the handler returns (same rules as
    // GRAPH.QUERY) — run synchronously in those cases.
    if ctx.get_flags().contains(ContextFlags::MULTI)
        || ctx.get_flags().contains(ContextFlags::REPLICATED)
    {
        return record_mut(ctx, &graph, query);
    }

    // Run on the thread pool like GRAPH.QUERY. Executing on the main thread
    // deadlocks the server: the handler holds the GIL while waiting for the
    // ThreadedGraph read lock, while a committing write query holds the
    // write lock and waits for the GIL.
    let bc = unsafe { BlockedClient::new(ctx.ctx) };
    let query: Arc<str> = Arc::from(query);
    spawn(
        move || {
            let ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
            let ctx = Context::new(ctx);
            if let Err(err) = record_mut(&ctx, &graph, &query) {
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
