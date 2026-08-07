//! `GRAPH.PROFILE` command handler.
//!
//! Executes a Cypher query and returns the execution plan annotated with
//! per-operator statistics (records produced, execution time).

use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{ThreadedGraph, profile_mut},
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
    let query = args.next_str()?;
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

    let key_name: Arc<str> = Arc::from(key_str.to_string());

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

    let graph = Arc::new(RwLock::new(ThreadedGraph::new(
        *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
        &key_str.to_string(),
    )));
    let result = profile_mut(ctx, &graph, query, &key_name, timeout);
    key.set_value(&GRAPH_TYPE, graph.clone())?;
    crate::graph_core::register_graph(key_str.to_string(), graph);
    result
}
