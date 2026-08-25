use crate::{
    config::CONFIGURATION_CACHE_SIZE,
    graph_core::{ThreadedGraph, c_graph_key, c_graph_name},
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
    let new_graph = serializers::decoder::vec_load_graph(data, cache_size, dest_name)
        .map_err(RedisError::String)?;

    // Wrap the decoded graph and set on dest key.
    let mvcc = MvccGraph::from_graph(new_graph);
    let graph_arc = mvcc.read();
    graph_arc.borrow().set_indexer_graph(graph_arc.clone());
    let tg = ThreadedGraph::from_mvcc(mvcc);
    let boxed = Arc::new(RwLock::new(tg));

    // Attached under C's name, at the key rebuilt from it — see `c_graph_key`.
    let create_key = ctx.open_key_writable(&c_graph_key(ctx, &dest_key_name));
    drop(dest_key);
    create_key.set_value(&GRAPH_TYPE, boxed.clone())?;
    crate::graph_core::register_graph(c_graph_name(&dest_key_name), boxed);

    // Replicate verbatim so sub-replicas also receive GRAPH.RESTORE.
    ctx.replicate_verbatim();

    Ok(RedisValue::SimpleStringStatic("OK"))
}
