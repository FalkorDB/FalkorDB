//! `GRAPH.PROFILE` command handler.
//!
//! Executes a Cypher query and returns the execution plan annotated with
//! per-operator statistics (records produced, execution time).

use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{
        ThreadedGraph, c_graph_key, c_graph_name, profile_mut, register_graph, up_to_nul,
    },
    redis_type::GRAPH_TYPE,
};
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisResult, RedisString};
use std::sync::Arc;

pub fn graph_profile(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let key_str = args.next_arg()?;
    // C ends the query at its first NUL byte; see `up_to_nul`.
    let query = up_to_nul(args.next_str()?);
    let mut timeout: Option<i64> = None;
    while let Ok(arg) = args.next_str() {
        // Matched case-insensitively, as the C dispatcher does with strcasecmp:
        // `TIMEOUT` is the documented spelling.
        if arg.eq_ignore_ascii_case("timeout")
            && let Ok(t_str) = args.next_str()
        {
            timeout = t_str.parse::<i64>().ok();
        }
    }

    let key_name: Arc<str> = Arc::from(c_graph_name(&key_str));

    // Try read-only key access first.
    let read_key = ctx.open_key(&key_str);
    if let Some(graph) = read_key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        let graph = graph.clone();
        drop(read_key);
        return profile_mut(ctx, &graph, query, &key_name, timeout);
    }

    // Graph doesn't exist - open writable key to create it.
    drop(read_key);
    let key = ctx.open_key_writable(&key_str);
    if let Some(graph) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        let graph = graph.clone();
        return profile_mut(ctx, &graph, query, &key_name, timeout);
    }

    let name = key_name.to_string();
    let graph = Arc::new(RwLock::new(ThreadedGraph::new(
        *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
        &name,
    )));
    let result = profile_mut(ctx, &graph, query, &key_name, timeout);
    // Stored under C's key, not the addressed one — see `c_graph_key`.
    let create_key = ctx.open_key_writable(&c_graph_key(ctx, &key_str));
    drop(key);
    create_key.set_value(&GRAPH_TYPE, graph.clone())?;
    register_graph(name, graph);
    result
}
