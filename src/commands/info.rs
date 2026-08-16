//! `GRAPH.INFO` command handler.
//!
//! ## Syntax
//! ```text
//! GRAPH.INFO [RunningQueries] [WaitingQueries] [ObjectPool]
//! ```
//!
//! Returns information about currently running and/or waiting queries
//! across all graphs, plus string-pool stats. With no section given, every
//! section is reported, same as the C engine.

use crate::telemetry;
use graph::runtime::string_pool;
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

    // no section asked for means all of them: clients such as the node
    // client's `db.info()` send a bare GRAPH.INFO
    let all = args.is_empty();

    let mut want_running = all;
    let mut want_waiting = all;
    let mut want_object_pool = all;

    for arg in &args {
        match arg.to_ascii_lowercase().as_str() {
            "runningqueries" => want_running = true,
            "waitingqueries" => want_waiting = true,
            "objectpool" => want_object_pool = true,
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

    if want_object_pool {
        let (count, avg) = string_pool::global().stats();
        let avg_str = format_avg(avg);
        result.push(RedisValue::BulkString("Object Pool".into()));
        result.push(RedisValue::Array(vec![
            RedisValue::Array(vec![
                RedisValue::BulkString("Unique Objects in Pool".into()),
                RedisValue::Integer(count as i64),
            ]),
            RedisValue::Array(vec![
                RedisValue::BulkString("Average References per Object".into()),
                RedisValue::BulkString(avg_str),
            ]),
        ]));
    }

    Ok(RedisValue::Array(result))
}

fn format_avg(avg: f64) -> String {
    if avg.fract() == 0.0 {
        format!("{}", avg as i64)
    } else {
        format!("{avg}")
    }
}
