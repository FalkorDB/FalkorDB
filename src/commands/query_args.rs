//! Argument parsing shared by `GRAPH.QUERY`, `GRAPH.RO_QUERY`, `GRAPH.PROFILE` and
//! `GRAPH.EXPLAIN`.
//!
//! ```text
//! GRAPH.<CMD> key query [--compact] [TIMEOUT ms] [version v] [--track-memory]
//! ```
//!
//! Mirrors C's `_validate_command_arity` + `_read_flags` (`cmd_dispatcher.c`), which
//! all four commands go through:
//! - 3 to 8 arguments (command name included), otherwise a wrong-arity error;
//! - flags are matched ASCII case-insensitively on the bytes up to the first NUL
//!   (C's `strcasecmp` on the C string), so a non-UTF-8 argument is just an
//!   unknown flag, and unknown flags are skipped;
//! - `TIMEOUT` and `version` values are read with Redis's `string2ll`: canonical
//!   decimal only (no `+`, no leading zeros, no spaces). A missing, non-numeric or
//!   negative timeout, or a version outside `0..=u32::MAX`, is an error.
//!
//! `--track-memory` is Rust-only.

use crate::config::TIMEOUT_MAX;
use redis_module::{RedisError, RedisString};
use std::sync::atomic::Ordering;

/// C's `_validate_command_arity` upper bound for the query commands.
const MAX_ARGS: usize = 8;

const TIMEOUT_ERR: &str = "Failed to parse query timeout value";
const VERSION_ERR: &str = "Failed to parse graph version value";
const TIMEOUT_MAX_ERR: &str =
    "The query TIMEOUT parameter value cannot exceed the TIMEOUT_MAX configuration parameter value";

/// Flags that follow the key and the query.
#[derive(Default)]
pub struct QueryFlags {
    pub compact: bool,
    pub track_memory: bool,
    pub timeout: Option<i64>,
    pub version: Option<u64>,
}

/// Checks the arity of a query command and parses its flags (`args[3..]`).
///
/// `args` is the full argument vector, command name included.
pub fn parse_query_flags(args: &[RedisString]) -> Result<QueryFlags, RedisError> {
    if args.len() < 3 || args.len() > MAX_ARGS {
        return Err(RedisError::WrongArity);
    }
    let mut flags = QueryFlags::default();
    let mut rest = args[3..].iter();
    while let Some(arg) = rest.next() {
        let arg = up_to_nul(arg.as_slice());
        if arg.eq_ignore_ascii_case(b"--compact") {
            flags.compact = true;
        } else if arg.eq_ignore_ascii_case(b"--track-memory") {
            flags.track_memory = true;
        } else if arg.eq_ignore_ascii_case(b"timeout") {
            let t = rest
                .next()
                .and_then(|v| v.parse_integer().ok())
                .ok_or(RedisError::Str(TIMEOUT_ERR))?;
            let max = TIMEOUT_MAX.load(Ordering::Relaxed);
            if max > 0 && t > max {
                return Err(RedisError::Str(TIMEOUT_MAX_ERR));
            }
            if t < 0 {
                return Err(RedisError::Str(TIMEOUT_ERR));
            }
            flags.timeout = Some(t);
        } else if arg.eq_ignore_ascii_case(b"version") {
            let v = rest
                .next()
                .and_then(|v| v.parse_integer().ok())
                .and_then(|v| u32::try_from(v).ok())
                .ok_or(RedisError::Str(VERSION_ERR))?;
            flags.version = Some(u64::from(v));
        }
    }
    Ok(flags)
}

/// The bytes a C string built from `s` would hold.
fn up_to_nul(s: &[u8]) -> &[u8] {
    s.iter().position(|&b| b == 0).map_or(s, |end| &s[..end])
}
