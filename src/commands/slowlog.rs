//! `GRAPH.SLOWLOG` command handler.
//!
//! ## Syntax
//! ```text
//! GRAPH.SLOWLOG <key>          -- list slowlog entries
//! GRAPH.SLOWLOG <key> RESET    -- clear slowlog
//! ```

use crate::{
    graph_core::ThreadedGraph,
    redis_type::{GRAPH_TYPE, wrong_type_compat},
};
use parking_lot::RwLock;
use std::sync::Arc;
use valkey_module::{Context, NextArg, ValkeyError, ValkeyResult, ValkeyString, ValkeyValue};

use super::EMPTY_KEY_ERR;

pub fn graph_slowlog(
    ctx: &Context,
    args: Vec<ValkeyString>,
) -> ValkeyResult {
    let mut args = args.into_iter().skip(1);
    let argc = args.len();

    if !(1..=2).contains(&argc) {
        return Err(ValkeyError::WrongArity);
    }

    let key_name = args.next_arg()?;
    let key = ctx.open_key(&key_name);

    let graph = match key
        .get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)
        .map_err(wrong_type_compat)?
    {
        Some(g) => g.clone(),
        None => return EMPTY_KEY_ERR,
    };

    // Handle optional subcommand.
    if let Ok(sub) = args.next_str() {
        if sub.eq_ignore_ascii_case("RESET") {
            graph.read().slow_log.reset();
            return Ok(ValkeyValue::SimpleString("OK".into()));
        }
        return Err(ValkeyError::Str("ERR Unknown subcommand"));
    }

    // Default: return slowlog entries.
    unsafe { graph.read().slow_log.reply(ctx.ctx) };
    Ok(ValkeyValue::NoReply)
}
