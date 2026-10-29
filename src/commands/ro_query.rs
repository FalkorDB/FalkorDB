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
    commands::{
        EMPTY_KEY_ERR,
        query_args::{QueryFlags, parse_query_flags},
    },
    graph_core::{ThreadedGraph, query_mut, up_to_nul},
    redis_type::{GRAPH_TYPE, wrong_type_compat},
};
use parking_lot::RwLock;
use std::sync::Arc;
use valkey_module::{Context, NextArg, ValkeyResult, ValkeyString};

pub fn graph_ro_query(
    ctx: &Context,
    args: Vec<ValkeyString>,
) -> ValkeyResult {
    let QueryFlags {
        compact,
        track_memory,
        timeout,
        version: version_check,
    } = parse_query_flags(&args)?;
    let mut args = args.into_iter().skip(1);
    let key = args.next_arg()?;
    // C ends the query at its first NUL byte; see `up_to_nul`.
    let query = up_to_nul(args.next_str()?);

    // The key the graph lives at, not C's name for it — see `graph_query`. A read-only
    // query never creates one, so it is always the key the command named.
    let key_name: Arc<str> = Arc::from(key.to_string());
    let key = ctx.open_key(&key);

    (key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)
        .map_err(wrong_type_compat)?)
    .map_or(EMPTY_KEY_ERR, |graph| {
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
