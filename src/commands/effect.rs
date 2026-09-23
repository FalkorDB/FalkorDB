//! `GRAPH.EFFECT` command handler.
//!
//! Applies serialized effects (mutations) received from the primary to
//! maintain replica consistency. The payload is built on the primary by
//! `CommitOp`, once per commit, into the query's one
//! [`graph::effects::EffectsBuffer`], and carries the exact mutations that
//! commit made.
//!
//! ## Command syntax
//! ```text
//! GRAPH.EFFECT <key> <effects_buffer>
//! ```

use crate::divergence_guard;
use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{ThreadedGraph, c_graph_key, c_graph_name, register_graph},
    redis_type::GRAPH_TYPE,
};
use graph::effects::EffectsPayload;
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisResult, RedisString, RedisValue};
use std::sync::Arc;

pub fn graph_effect(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let key_str = args.next_arg()?;
    let effects_buf = args.next_arg()?;

    let buf = effects_buf.as_slice();
    if buf.is_empty() {
        return Ok(RedisValue::SimpleStringStatic("OK"));
    }

    // Open existing graph or create a new one. Looked up by the full key bytes
    // and created under the truncated name, which is C's own asymmetry — see
    // `c_graph_key`.
    let key = ctx.open_key_writable(&key_str);
    let graph = if let Some(g) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        g.clone()
    } else {
        let name = c_graph_name(&key_str);
        let g = Arc::new(RwLock::new(ThreadedGraph::new(
            *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
            &name,
        )));
        // The same pair `GRAPH.QUERY`, `GRAPH.CONSTRAINT`, `GRAPH.PROFILE` and
        // `GRAPH.BULK` create through. Using the raw key here instead put a
        // graph whose key holds a NUL at a different key, under a different
        // name, from the one every other command would have made — and this is
        // the replication path, so the divergence would be between a replica
        // and the master it is meant to be copying.
        let create_key = ctx.open_key_writable(&c_graph_key(ctx, &key_str));
        create_key.set_value(&GRAPH_TYPE, g.clone())?;
        register_graph(name, g.clone());
        g
    };

    let mut tg = graph.write();
    let Some(g_arc) = tg.graph.write() else {
        return Err(redis_module::RedisError::String(
            "ERR another write is in progress, retry the query".to_string(),
        ));
    };

    let result = {
        let mut g = g_arc.borrow_mut();
        // Dispatched on the version the buffer declares, not on the one this
        // build writes — during a rolling upgrade the primary is routinely on a
        // different build. A version with no impl at all is divergence rather
        // than a compatibility case; `apply` rejects it and the guard below
        // forces a resync.
        EffectsPayload::apply(&mut g, buf).map_err(|e| e.to_string())
    };

    match result {
        Ok(()) => {
            tg.graph.commit(g_arc);
            ctx.replicate_verbatim();
            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        Err(e) => {
            // Rolled back, so this graph is exactly as it was — but the write
            // the master already made is now missing here, and every later
            // effect lands on a graph that has drifted. Redis will not break
            // the link over an error reply, so without this the replica serves
            // wrong data until someone notices.
            tg.graph.rollback();
            // `on_failure` decides for itself whether this was replayed; a
            // client-sent payload returns the error below and nothing more.
            divergence_guard::on_failure(ctx, &key_str.to_string(), "GRAPH.EFFECT", &e, Some(buf));
            Err(redis_module::RedisError::String(format!(
                "ERR effect apply failed: {e}"
            )))
        }
    }
}
