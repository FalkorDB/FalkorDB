//! `GRAPH.LIST` command handler.
//!
//! Lists all graph keys currently stored in the Redis instance by scanning
//! for keys of native type `graphdata`.
//!
//! ## Syntax
//! ```text
//! GRAPH.LIST
//! ```
//!
//! ## How it works
//! The handler iterates over `SCAN` cursors with a `TYPE graphdata` filter
//! until the cursor returns to `"0"`, accumulating matching key names into
//! a single array response.
//!
//! ```text
//! SCAN 0 TYPE graphdata
//!   |
//!   +--> cursor=17, [key1, key2]  --+
//!   |                                |  accumulate
//! SCAN 17 TYPE graphdata             |
//!   |                                |
//!   +--> cursor=0,  [key3]       --+
//!   |                                |
//!   +--> done, return [key1, key2, key3]
//! ```
//!
//! No arguments are accepted beyond the command name itself; extra arguments
//! result in a `WrongArity` error.

use redis_module::{Context, RedisError, RedisResult, RedisString, RedisValue};

/// The name `GRAPH.LIST` replies for one key `SCAN` returned.
///
/// Two things have to hold at once. C replies `gc->graph_name` per graph, which is the
/// key truncated at its first NUL — never the raw key bytes — and a keyspace scan is
/// how this handler finds graphs, so the same truncation is applied to what the scan
/// returned. And C replies each name with `RM_ReplyWithStringBuffer`, which is
/// length-framed. Replying a name as a RESP *status* instead
/// (`RedisValue::SimpleString`, i.e. `RM_ReplyWithSimpleString`) escapes nothing, so a
/// key holding CR/LF let a client inject arbitrary RESP into the *next* reply on the
/// connection. `StringBuffer` is C's reply, and carries the bytes it is handed.
///
/// `SCAN` returns a name as a `SimpleString` when its bytes are valid UTF-8 and as a
/// `StringBuffer` when they are not; both arms need the same treatment, or the bytes
/// past a NUL still leak for a non-UTF-8 key.
fn c_reply_name(name: RedisValue) -> RedisValue {
    let mut bytes = match name {
        RedisValue::SimpleString(s) | RedisValue::BulkString(s) => s.into_bytes(),
        RedisValue::StringBuffer(b) => b,
        other => return other,
    };
    if let Some(end) = bytes.iter().position(|b| *b == 0) {
        bytes.truncate(end);
    }
    RedisValue::StringBuffer(bytes)
}

#[allow(clippy::needless_pass_by_value)]
pub fn graph_list(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    if args.len() != 1 {
        return Err(RedisError::WrongArity);
    }

    let mut a = [
        ctx.create_string("0"),
        ctx.create_string("TYPE"),
        ctx.create_string("graphdata"),
    ];
    let mut res = Vec::new();
    loop {
        let call_res = ctx.call("SCAN", a.iter().collect::<Vec<_>>().as_slice())?;
        match call_res {
            RedisValue::Array(mut arr) => {
                if let RedisValue::Array(arr) = arr.remove(1) {
                    res.extend(arr.into_iter().map(c_reply_name));
                }
                if let RedisValue::SimpleString(i) = arr.remove(0) {
                    if i == "0" {
                        return Ok(RedisValue::Array(res));
                    }
                    a[0] = ctx.create_string(i);
                }
            }
            _ => return Err(RedisError::Str("ERR Failed to list graphs")),
        }
    }
}
