//! `GRAPH.DELETE` command handler.
//!
//! Deletes an existing graph key from Redis, removing all associated graph
//! data (adjacency matrices, attribute stores, indices, etc.).
//!
//! ## Syntax
//! ```text
//! GRAPH.DELETE <key>
//! ```
//!
//! ## Execution flow
//! ```text
//! GRAPH.DELETE key
//!        |
//!        +--> validate arity (exactly 2 args)
//!        +--> open key as writable
//!        +--> verify key holds a graph native type
//!        |       |
//!        |       +--> yes: delegate to Redis key.delete()
//!        |       +--> no:  return EMPTY_KEY_ERR
//! ```
//!
//! The actual graph teardown (dropping the `ThreadedGraph`, stopping its
//! write-serialization thread, and freeing GraphBLAS matrices) happens
//! when Redis removes the key and the custom type's `free` callback fires.

use crate::{
    commands::EMPTY_KEY_ERR,
    graph_core::{ThreadedGraph, c_graph_name},
    redis_type::GRAPH_TYPE,
    telemetry,
};
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisError, RedisResult, RedisString};
use std::sync::Arc;

pub fn graph_delete(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }

    let mut args = args.into_iter().skip(1);
    let key = args.next_arg()?;
    // The stream is named after the graph's C name (`telemetry{%s}` of
    // `gc->graph_name`), which is the key truncated at its first NUL.
    let key_name = c_graph_name(&key);
    let key = ctx.open_key_writable(&key);
    if key
        .get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)?
        .is_some()
    {
        // Delete the telemetry stream before removing the graph key.
        telemetry::delete_stream(ctx, &key_name);
        let result = key.delete();
        ctx.replicate_verbatim();
        result
    } else {
        EMPTY_KEY_ERR
    }
}
