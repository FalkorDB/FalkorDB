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
    graph_core::{ThreadedGraph, query_mut, up_to_nul},
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
    // C ends the query at its first NUL byte; see `up_to_nul`.
    let query = up_to_nul(args.next_str()?);
    let mut compact = false;
    let mut track_memory = false;
    let mut version_check: Option<u64> = None;
    let mut timeout: Option<i64> = None;
    while let Ok(arg) = args.next_str() {
        // Matched case-insensitively, as the C dispatcher does with strcasecmp:
        // `TIMEOUT` is the documented spelling.
        if arg.eq_ignore_ascii_case("--compact") {
            compact = true;
        } else if arg.eq_ignore_ascii_case("--track-memory") {
            track_memory = true;
        } else if arg.eq_ignore_ascii_case("version") {
            let ver_str = args.next_str()?;
            version_check = Some(ver_str.parse::<u64>()?);
        } else if arg.eq_ignore_ascii_case("timeout")
            && let Ok(t_str) = args.next_str()
        {
            timeout = t_str.parse::<i64>().ok();
        }
    }

    // The key the graph lives at, not C's name for it — see `graph_query`. A read-only
    // query never creates one, so it is always the key the command named.
    let key_name: Arc<str> = Arc::from(key.to_string());
    let key = ctx.open_key(&key);

    (key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)?).map_or(EMPTY_KEY_ERR, |graph| {
        query_mut(
            ctx,
            graph,
            query,
            compact,
            false,
            track_memory,
            key_name,
            timeout,
            version_check,
        )
    })
}
