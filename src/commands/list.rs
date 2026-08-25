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

/// Re-type a key name from `SCAN` as a bulk string so it is replied byte for byte.
///
/// `SCAN` hands back key names as `RedisValue::SimpleString` whenever they are valid
/// UTF-8, and replying with that variant goes through `CString::new(..).unwrap()` onto
/// `RM_ReplyWithSimpleString` — a NUL-terminated, single-line RESP status. A graph key
/// is arbitrary bytes, so that variant both aborted the process on a key holding an
/// interior NUL (#2490) and mangled a key holding CR or LF. The C engine replies to
/// `GRAPH.LIST` with `RM_ReplyWithStringBuffer` per name; a bulk string is the same
/// thing, and carries the bytes it was given.
fn binary_safe(name: RedisValue) -> RedisValue {
    match name {
        RedisValue::SimpleString(s) | RedisValue::BulkString(s) => {
            RedisValue::StringBuffer(s.into_bytes())
        }
        other => other,
    }
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
                    res.extend(arr.into_iter().map(binary_safe));
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
