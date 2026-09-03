//! `GRAPH.EFFECT` command handler.
//!
//! Applies serialized effects (mutations) received from the primary to
//! maintain replica consistency.  The binary effects buffer is produced
//! by `Pending::build_effects_buffer()` on the primary and contains the
//! exact mutations that occurred during query execution.
//!
//! ## Command syntax
//! ```text
//! GRAPH.EFFECT <key> <effects_buffer>
//! ```

use crate::{config::CONFIGURATION_CACHE_SIZE, graph_core::ThreadedGraph, redis_type::GRAPH_TYPE};
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

    // Open existing graph or create a new one
    let key = ctx.open_key_writable(&key_str);
    let graph = if let Some(g) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        g.clone()
    } else {
        let g = Arc::new(RwLock::new(ThreadedGraph::new(
            *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
            &key_str.to_string(),
        )));
        key.set_value(&GRAPH_TYPE, g.clone())?;
        crate::graph_core::register_graph(key_str.to_string(), g.clone());
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
        // v3 is the only format. A buffer announcing anything else came from a
        // peer speaking a language this build does not, which is divergence —
        // `apply_effects` rejects it and the guard below forces a resync.
        graph::effects::v3::apply::apply_effects(&mut g, buf).map_err(|e| e.to_string())
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
            crate::divergence_guard::on_failure(ctx, &key_str.to_string(), "GRAPH.EFFECT", &e);
            Err(redis_module::RedisError::String(format!(
                "ERR effect apply failed: {e}"
            )))
        }
    }
}
