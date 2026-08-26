use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{ThreadedGraph, register_graph, up_to_nul},
    redis_type::GRAPH_TYPE,
    serializers,
};
use graph::graph::mvcc_graph::MvccGraph;
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisError, RedisResult, RedisString, RedisValue};
use std::sync::Arc;

pub fn graph_restore(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    if args.len() != 3 {
        return Err(RedisError::WrongArity);
    }

    let mut args = args.into_iter().skip(1);
    let dest_key_name = args.next_arg()?;
    let data_arg = args.next_arg()?;

    let dest_name = std::str::from_utf8(dest_key_name.as_slice())
        .map_err(|_| RedisError::Str("ERR destination key is not valid UTF-8"))?;

    // The destination *key* is the full bytes the command named, both for the existence
    // check and for the write — C opens `argv[1]` twice and never rebuilds the key from
    // the name. Only the graph's own *name* is truncated: C takes it from the payload
    // header, which `GRAPH.COPY` wrote as a C string.
    let graph_name = up_to_nul(dest_name);

    // Verify dest key does not already exist.
    let dest_key = ctx.open_key_writable(&dest_key_name);
    if dest_key
        .get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)?
        .is_some()
    {
        return Err(RedisError::Str("restore graph failed, key already exists"));
    }
    if dest_key.key_type() != redis_module::KeyType::Empty {
        return Err(RedisError::Str("restore graph failed, key already exists"));
    }

    let cache_size = *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize;

    let data = data_arg.as_slice();
    let new_graph = serializers::decoder::vec_load_graph(data, cache_size, graph_name)
        .map_err(RedisError::String)?;

    // Wrap the decoded graph and set on dest key.
    let mvcc = MvccGraph::from_graph(new_graph);
    let graph_arc = mvcc.read();
    graph_arc.borrow().set_indexer_graph(graph_arc.clone());
    let tg = ThreadedGraph::from_mvcc(mvcc);
    let boxed = Arc::new(RwLock::new(tg));

    dest_key.set_value(&GRAPH_TYPE, boxed.clone())?;
    // Registered under the key it was written to — see `graph_core::register_graph`.
    register_graph(dest_name.to_string(), boxed);

    // Replicate verbatim so sub-replicas also receive GRAPH.RESTORE.
    ctx.replicate_verbatim();

    Ok(RedisValue::SimpleStringStatic("OK"))
}
