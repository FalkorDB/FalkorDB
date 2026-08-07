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
                    res.extend(arr);
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
