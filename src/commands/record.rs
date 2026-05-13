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

use crate::{
    config::{CONFIGURATION_CACHE_SIZE, CONFIGURATION_IMPORT_FOLDER},
    graph_core::ThreadedGraph,
    redis_type::GRAPH_TYPE,
    reply::reply_verbose_value,
};
use graph::{
    graph::graph::Plan,
    runtime::{
        eval::evaluate_param,
        pool::Pool,
        runtime::{GetVariables, Runtime},
    },
};
use orx_tree::{Bfs, Collection, NodeRef};
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisError, RedisResult, RedisString, RedisValue, raw};
use std::{collections::HashMap, os::raw::c_char, sync::Arc};

#[inline]
fn record_mut(
    ctx: &Context,
    graph: &Arc<RwLock<ThreadedGraph>>,
    query: &str,
) -> RedisResult {
    let Plan {
        plan, parameters, ..
    } = graph
        .read()
        .graph
        .read()
        .borrow()
        .get_plan(query)
        .map_err(RedisError::String)?;
    let parameters = parameters
        .into_iter()
        .map(|(k, v)| Ok((k, evaluate_param(&v.root())?)))
        .collect::<Result<HashMap<_, _>, String>>()
        .map_err(RedisError::String)?;
    let env_pool = Pool::new();
    let runtime = Runtime::new(
        graph.read().graph.read(),
        parameters,
        true,
        plan.clone(),
        true,
        (*CONFIGURATION_IMPORT_FOLDER.lock(ctx)).clone(),
        &env_pool,
        -1,
        false,
        None,
        0,
        None,
    );
    let _ = runtime.query();
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
            Ok((values, bound)) => {
                raw::reply_with_long_long(ctx.ctx, 1);
                let vars = plan.node(*idx).get_variables();
                raw::reply_with_array(ctx.ctx, vars.len() as _);
                for name in &vars {
                    if bound.test(name.id as usize) {
                        match values.get(name.id as usize) {
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

    if let Some(graph) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        record_mut(ctx, graph, query)?;
    } else {
        let graph = Arc::new(RwLock::new(ThreadedGraph::new(
            *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
            &key_str.to_string(),
        )));
        record_mut(ctx, &graph, query)?;
        key.set_value(&GRAPH_TYPE, graph.clone())?;
        crate::graph_core::register_graph(key_str.to_string(), graph);
    }

    RedisResult::Ok(RedisValue::NoReply)
}
