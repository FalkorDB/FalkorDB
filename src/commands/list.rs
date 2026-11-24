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

use valkey_module::{
    CallOptionsBuilder, CallReply, CallResult, Context, ValkeyError, ValkeyResult, ValkeyString,
    ValkeyValue,
};

const LIST_FAILED: ValkeyError = ValkeyError::Str("ERR Failed to list graphs");

/// The name `GRAPH.LIST` replies for one key `SCAN` returned.
///
/// Two things have to hold at once. C replies `gc->graph_name` per graph, which is the
/// key truncated at its first NUL — never the raw key bytes — and a keyspace scan is
/// how this handler finds graphs, so the same truncation is applied to what the scan
/// returned. And C replies each name with `RM_ReplyWithStringBuffer`, which is
/// length-framed. Replying a name as a RESP *status* instead
/// (`ValkeyValue::SimpleString`, i.e. `RM_ReplyWithSimpleString`) escapes nothing, so a
/// key holding CR/LF let a client inject arbitrary RESP into the *next* reply on the
/// connection. `StringBuffer` is C's reply, and carries the bytes it is handed.
fn c_reply_name(key: &[u8]) -> ValkeyValue {
    let end = key.iter().position(|b| *b == 0).unwrap_or(key.len());
    ValkeyValue::StringBuffer(key[..end].to_vec())
}

/// The bytes of a string element of a `SCAN` reply.
///
/// A key is binary, so these bytes need not be UTF-8. `valkey-module` converts a
/// `Context::call` reply into a `ValkeyValue` with `to_string().unwrap()`, which panics
/// on a key such as `b"\xc3\x28g"`, and a panic here aborts the server. The reply is
/// read as raw bytes instead.
fn reply_bytes<'a>(reply: &'a CallResult<'_>) -> Option<&'a [u8]> {
    match reply {
        Ok(CallReply::String(s)) => Some(s.as_bytes()),
        _ => None,
    }
}

#[allow(clippy::needless_pass_by_value)]
pub fn graph_list(
    ctx: &Context,
    args: Vec<ValkeyString>,
) -> ValkeyResult {
    if args.len() != 1 {
        return Err(ValkeyError::WrongArity);
    }

    // `v`, the format `Context::call` uses, so the server sees the same call.
    let options = CallOptionsBuilder::new().build();
    let mut a = [
        ctx.create_string("0"),
        ctx.create_string("TYPE"),
        ctx.create_string("graphdata"),
    ];
    let mut res = Vec::new();
    loop {
        let reply: CallResult =
            ctx.call_ext("SCAN", &options, a.iter().collect::<Vec<_>>().as_slice());
        let CallReply::Array(page) = reply.map_err(ValkeyError::from)? else {
            return Err(LIST_FAILED);
        };
        // `[cursor, [key, ...]]`
        let (Some(cursor), Some(Ok(CallReply::Array(keys)))) = (page.get(0), page.get(1)) else {
            return Err(LIST_FAILED);
        };
        for key in keys.iter() {
            res.push(c_reply_name(reply_bytes(&key).ok_or(LIST_FAILED)?));
        }
        let cursor = reply_bytes(&cursor).ok_or(LIST_FAILED)?;
        if cursor == b"0" {
            return Ok(ValkeyValue::Array(res));
        }
        // From the bytes: `create_string` goes through `CString::new(..).unwrap()`.
        a[0] = ValkeyString::create_from_slice(ctx.get_raw(), cursor);
    }
}
