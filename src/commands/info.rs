//! `GRAPH.INFO` command handler.
//!
//! ## Syntax
//! ```text
//! GRAPH.INFO [RunningQueries] [WaitingQueries]
//! ```
//!
//! Returns information about currently running and/or waiting queries
//! across all graphs.

use crate::telemetry;
use redis_module::{Context, RedisError, RedisResult, RedisString, RedisValue};

#[allow(clippy::needless_pass_by_value)]
pub fn graph_info(
    _ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let args: Vec<String> = args
        .into_iter()
        .skip(1) // skip command name
        .map(|a| a.to_string_lossy())
        .collect();

    if args.is_empty() {
        return Err(RedisError::WrongArity);
    }

    let mut want_running = false;
    let mut want_waiting = false;

    for arg in &args {
        match arg.to_ascii_lowercase().as_str() {
            "runningqueries" => want_running = true,
            "waitingqueries" => want_waiting = true,
            _ => {
                return Err(RedisError::String(format!("ERR Unknown section: {arg}")));
            }
        }
    }

    let mut result = Vec::new();

    if want_running {
        result.push(RedisValue::BulkString("# Running queries".into()));
        result.push(RedisValue::Array(telemetry::running_queries_reply()));
    }

    if want_waiting {
        result.push(RedisValue::BulkString("# Waiting queries".into()));
        result.push(RedisValue::Array(telemetry::waiting_queries_reply()));
    }

    Ok(RedisValue::Array(result))
}
