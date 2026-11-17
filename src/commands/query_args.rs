//! Argument parsing shared by `GRAPH.QUERY`, `GRAPH.RO_QUERY`, `GRAPH.PROFILE` and
//! `GRAPH.EXPLAIN`.
//!
//! ```text
//! GRAPH.<CMD> key query [--compact] [TIMEOUT ms] [version v] [SCHEMA_HASH h] [--track-memory]
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
//! `--track-memory` and `SCHEMA_HASH` are Rust-only. `SCHEMA_HASH h` (#3757) is parsed for all
//! four commands, and only `GRAPH.QUERY` and `GRAPH.RO_QUERY` act on it. Its value is a
//! 64-bit schema hash, in either form a client may hold: the signed integer the reply
//! sends it as, or the unsigned one with the same bits. Older servers skip it as an
//! unknown flag, which is why it fits within [`MAX_ARGS`] with the flags a client sends
//! alongside it: a client that sent more would get an arity error from them.

use crate::config::TIMEOUT_MAX;
use std::sync::atomic::Ordering;
use valkey_module::{ValkeyError, ValkeyString};

/// C's `_validate_command_arity` upper bound for the query commands.
const MAX_ARGS: usize = 8;

const TIMEOUT_ERR: &str = "Failed to parse query timeout value";
const VERSION_ERR: &str = "Failed to parse graph version value";
const SCHEMA_ERR: &str = "Failed to parse schema hash value";
const TIMEOUT_MAX_ERR: &str =
    "The query TIMEOUT parameter value cannot exceed the TIMEOUT_MAX configuration parameter value";

/// Flags that follow the key and the query.
#[derive(Default)]
pub struct QueryFlags {
    pub compact: bool,
    pub track_memory: bool,
    pub timeout: Option<i64>,
    pub version: Option<u64>,
    /// `SCHEMA_HASH h`: the client's cached schema hash.
    pub schema: Option<u64>,
}

/// Checks the arity of a query command and parses its flags (`args[3..]`).
///
/// `args` is the full argument vector, command name included.
pub fn parse_query_flags(args: &[ValkeyString]) -> Result<QueryFlags, ValkeyError> {
    if args.len() < 3 || args.len() > MAX_ARGS {
        return Err(ValkeyError::WrongArity);
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
                .ok_or(ValkeyError::Str(TIMEOUT_ERR))?;
            let max = TIMEOUT_MAX.load(Ordering::Relaxed);
            if max > 0 && t > max {
                return Err(ValkeyError::Str(TIMEOUT_MAX_ERR));
            }
            if t < 0 {
                return Err(ValkeyError::Str(TIMEOUT_ERR));
            }
            flags.timeout = Some(t);
        } else if arg.eq_ignore_ascii_case(b"version") {
            let v = rest
                .next()
                .and_then(|v| v.parse_integer().ok())
                .and_then(|v| u32::try_from(v).ok())
                .ok_or(ValkeyError::Str(VERSION_ERR))?;
            flags.version = Some(u64::from(v));
        } else if arg.eq_ignore_ascii_case(b"schema_hash") {
            let h = rest
                .next()
                .and_then(parse_schema_hash)
                .ok_or(ValkeyError::Str(SCHEMA_ERR))?;
            flags.schema = Some(h);
        }
    }
    Ok(flags)
}

/// A schema hash, as the signed integer a reply carries it as (`string2ll`, like the
/// other values) or as the unsigned one with the same bits: canonical decimal, so
/// above `i64::MAX` a leading digit 1-9 and digits only, and at most `u64::MAX`.
fn parse_schema_hash(v: &ValkeyString) -> Option<u64> {
    if let Ok(signed) = v.parse_integer() {
        return Some(signed.cast_unsigned());
    }
    let bytes = v.as_slice();
    if !bytes.first().is_some_and(|b| (b'1'..=b'9').contains(b))
        || !bytes.iter().all(u8::is_ascii_digit)
    {
        return None;
    }
    std::str::from_utf8(bytes).ok()?.parse().ok()
}

/// The bytes a C string built from `s` would hold.
fn up_to_nul(s: &[u8]) -> &[u8] {
    s.iter().position(|&b| b == 0).map_or(s, |end| &s[..end])
}
