//! `GRAPH.SLOWLOG` command handler.
//!
//! ## Syntax
//! ```text
//! GRAPH.SLOWLOG <key>          -- list slowlog entries
//! GRAPH.SLOWLOG <key> RESET    -- clear slowlog
//! ```

use crate::{graph_core::ThreadedGraph, redis_type::GRAPH_TYPE};
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisError, RedisResult, RedisString, RedisValue};
use std::sync::Arc;

use super::EMPTY_KEY_ERR;

pub fn graph_slowlog(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let argc = args.len();

    if !(1..=2).contains(&argc) {
        return Err(RedisError::WrongArity);
    }

    let key_name = args.next_arg()?;
    let key = ctx.open_key(&key_name);

    let graph = match key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        Some(g) => g.clone(),
        None => return EMPTY_KEY_ERR,
    };

    // Handle optional subcommand.
    if let Ok(sub) = args.next_str() {
        if sub.eq_ignore_ascii_case("RESET") {
            graph.read().slow_log.reset();
            return Ok(RedisValue::SimpleString("OK".into()));
        }
        return Err(RedisError::Str("ERR Unknown subcommand"));
    }

    // Default: return slowlog entries.
    unsafe { graph.read().slow_log.reply(ctx.ctx) };
    Ok(RedisValue::NoReply)
}
