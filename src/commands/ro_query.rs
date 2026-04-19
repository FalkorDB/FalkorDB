//! `GRAPH.RO_QUERY` command handler.
//!
//! Executes read-only Cypher queries against an existing graph key and rejects
//! missing graph keys with a command-level error.
//!
//! ## Execution flow
//! ```text
//! GRAPH.RO_QUERY key query [flags]
//!        |
//!        +--> key must already exist
//!        +--> delegate to graph_core::query_mut(..., write=false)
//!        +--> runtime rejects write plans detected in query IR
//! ```
//!
//! This command preserves a strict read contract for clients that require
//! non-mutating behavior.

use crate::{
    commands::EMPTY_KEY_ERR,
    graph_core::{ThreadedGraph, query_mut},
    redis_type::GRAPH_TYPE,
};
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisResult, RedisString};
use std::sync::Arc;

pub fn graph_ro_query(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let key = args.next_arg()?;
    let query = args.next_str()?;
    let mut compact = false;
    let mut track_memory = false;
    while let Ok(arg) = args.next_str() {
        if arg == "--compact" {
            compact = true;
        } else if arg == "--track-memory" {
            track_memory = true;
        }
    }

    let key_name: Arc<str> = Arc::from(key.to_string());
    let key = ctx.open_key(&key);

    (key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)?).map_or(EMPTY_KEY_ERR, |graph| {
        query_mut(ctx, graph, query, compact, false, track_memory, key_name)
    })
}
