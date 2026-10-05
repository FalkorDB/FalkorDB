//! `GRAPH.CONFIG` command handler.
//!
//! Implements graph-specific runtime configuration via GET and SET subcommands.
//!
//! ## Syntax
//! ```text
//! GRAPH.CONFIG GET <name>       -- retrieve a single config value
//! GRAPH.CONFIG GET *            -- retrieve all config values
//! GRAPH.CONFIG SET <name> <val> [<name> <val> ...]  -- set one or more values
//! ```
//!
//! ## Configuration categories
//!
//! Runtime-settable (via SET):
//!   TIMEOUT, TIMEOUT_DEFAULT, TIMEOUT_MAX, RESULTSET_SIZE,
//!   MAX_QUEUED_QUERIES, QUERY_MEM_CAPACITY, DELTA_MAX_PENDING_CHANGES,
//!   VKEY_MAX_ENTITY_COUNT, JS_HEAP_SIZE, JS_STACK_SIZE,
//!   CMD_INFO, MAX_INFO_QUERIES, ASYNC_DELETE, DELAY_INDEXING
//!
//! Read-only (SET returns an error):
//!   THREAD_COUNT, INDEX_WORKER_THREADS, OMP_THREAD_COUNT, CACHE_SIZE,
//!   NODE_CREATION_BUFFER, BOLT_PORT, IMPORT_FOLDER, TEMP_FOLDER
//!
//! ASYNC_DELETE and DELAY_INDEXING are settable, as they are in C, but nothing in
//! this engine reads either one yet: graph teardown is always off-thread (see
//! `graph_core::graph_free`, which cannot free inline — `Index::drop` takes the GIL,
//! which the main thread already holds while the free callback runs), and the
//! decoders always build indexes eagerly. So `SET` on them records a value and
//! changes no behaviour; C reads both.
//!
//! ## Multi-SET semantics
//! When multiple name-value pairs are provided in a single SET, all pairs are
//! validated first. If any validation fails, no values are applied (atomic
//! all-or-nothing). If JS_HEAP_SIZE or JS_STACK_SIZE are changed, the UDF
//! repository version is bumped once after all values are applied so
//! concurrent queries do not see a partial configuration update.

use crate::config::{
    ASYNC_DELETE, BOLT_PORT, CONFIG_NAMES, CONFIGURATION_CACHE_SIZE, CONFIGURATION_CMD_INFO,
    CONFIGURATION_DELAY_INDEXING, CONFIGURATION_IMPORT_FOLDER, CONFIGURATION_INDEX_WORKER_THREADS,
    CONFIGURATION_JS_HEAP_SIZE, CONFIGURATION_JS_STACK_SIZE, CONFIGURATION_NODE_CREATION_BUFFER,
    CONFIGURATION_TEMP_FOLDER, CONFIGURATION_VKEY_MAX_ENTITY_COUNT, DELTA_MAX_PENDING_CHANGES,
    EFFECTS_COMPRESSION, EFFECTS_THRESHOLD_DEPRECATED, MAX_INFO_QUERIES, MAX_INFO_QUERIES_CAP,
    MAX_QUEUED_QUERIES, OMP_THREAD_COUNT, QUERY_MEM_CAPACITY, RESULTSET_SIZE, TIMEOUT,
    TIMEOUT_DEFAULT, TIMEOUT_MAX, get_thread_count, normalize_node_creation_buffer,
};
use redis_module::{Context, NextArg, RedisResult, RedisString, RedisValue};
use std::sync::atomic::Ordering;

/// Get a single config value by name.
fn config_get_one(
    ctx: &Context,
    name: &str,
) -> Result<RedisValue, String> {
    let val: RedisValue = match name {
        "TIMEOUT" => RedisValue::Integer(TIMEOUT.load(Ordering::Relaxed)),
        "TIMEOUT_DEFAULT" => RedisValue::Integer(TIMEOUT_DEFAULT.load(Ordering::Relaxed)),
        "TIMEOUT_MAX" => RedisValue::Integer(TIMEOUT_MAX.load(Ordering::Relaxed)),
        "CACHE_SIZE" => RedisValue::Integer(*CONFIGURATION_CACHE_SIZE.lock(ctx)),
        "ASYNC_DELETE" => RedisValue::Integer(ASYNC_DELETE.load(Ordering::Relaxed)),
        "OMP_THREAD_COUNT" => {
            let v = OMP_THREAD_COUNT.load(Ordering::Relaxed);
            let v = if v > 0 { v } else { get_thread_count(ctx) };
            RedisValue::Integer(v)
        }
        "THREAD_COUNT" => RedisValue::Integer(get_thread_count(ctx)),
        "INDEX_WORKER_THREADS" => {
            RedisValue::Integer(*CONFIGURATION_INDEX_WORKER_THREADS.lock(ctx))
        }
        "RESULTSET_SIZE" => RedisValue::Integer(RESULTSET_SIZE.load(Ordering::Relaxed)),
        "VKEY_MAX_ENTITY_COUNT" => {
            RedisValue::Integer(*CONFIGURATION_VKEY_MAX_ENTITY_COUNT.lock(ctx))
        }
        "MAX_QUEUED_QUERIES" => {
            RedisValue::Integer(MAX_QUEUED_QUERIES.load(Ordering::Relaxed) as i64)
        }
        "QUERY_MEM_CAPACITY" => RedisValue::Integer(QUERY_MEM_CAPACITY.load(Ordering::Relaxed)),
        "DELTA_MAX_PENDING_CHANGES" => {
            RedisValue::Integer(DELTA_MAX_PENDING_CHANGES.load(Ordering::Relaxed))
        }
        "NODE_CREATION_BUFFER" => RedisValue::Integer(normalize_node_creation_buffer(
            *CONFIGURATION_NODE_CREATION_BUFFER.lock(ctx),
        )),
        "CMD_INFO" => {
            RedisValue::Integer(i64::from(CONFIGURATION_CMD_INFO.load(Ordering::Relaxed)))
        }
        "MAX_INFO_QUERIES" => RedisValue::Integer(MAX_INFO_QUERIES.load(Ordering::Relaxed)),
        "EFFECTS_COMPRESSION" => RedisValue::Integer(EFFECTS_COMPRESSION.load(Ordering::Relaxed)),
        // Deprecated and read by nothing; reported so a GET round-trips a SET.
        "EFFECTS_THRESHOLD" => {
            RedisValue::Integer(EFFECTS_THRESHOLD_DEPRECATED.load(Ordering::Relaxed))
        }
        "BOLT_PORT" => RedisValue::Integer(BOLT_PORT.load(Ordering::Relaxed)),
        "DELAY_INDEXING" => RedisValue::Integer(i64::from(*CONFIGURATION_DELAY_INDEXING.lock(ctx))),
        "IMPORT_FOLDER" => RedisValue::BulkString((*CONFIGURATION_IMPORT_FOLDER.lock(ctx)).clone()),
        "TEMP_FOLDER" => RedisValue::BulkString((*CONFIGURATION_TEMP_FOLDER.lock(ctx)).clone()),
        "JS_HEAP_SIZE" => RedisValue::Integer(*CONFIGURATION_JS_HEAP_SIZE.lock(ctx)),
        "JS_STACK_SIZE" => RedisValue::Integer(*CONFIGURATION_JS_STACK_SIZE.lock(ctx)),
        _ => return Err(format!("Unknown configuration field '{name}'")),
    };
    Ok(RedisValue::Array(vec![
        RedisValue::BulkString(name.to_string()),
        val,
    ]))
}

/// Validate and parse a config SET request. Returns the parsed value or error.
fn validate_config_set(
    name: &str,
    value: &str,
) -> Result<ConfigValue, String> {
    match name {
        // Runtime-settable integer configs
        "TIMEOUT"
        | "TIMEOUT_DEFAULT"
        | "TIMEOUT_MAX"
        | "QUERY_MEM_CAPACITY"
        | "DELTA_MAX_PENDING_CHANGES"
        | "EFFECTS_COMPRESSION"
        | "EFFECTS_THRESHOLD" => {
            let v: i64 = value
                .parse()
                .map_err(|_| format!("Failed to set config value {name} to {value}"))?;
            if v < 0 {
                return Err(format!("Failed to set config value {name} to {value}"));
            }
            Ok(ConfigValue::Int(v))
        }

        // Runtime-settable boolean configs
        "ASYNC_DELETE" | "CMD_INFO" | "DELAY_INDEXING" => {
            let v = match value.to_lowercase().as_str() {
                "yes" | "1" | "true" => 1i64,
                "no" | "0" | "false" => 0i64,
                _ => return Err(format!("Failed to set config value {name} to {value}")),
            };
            Ok(ConfigValue::Int(v))
        }
        "RESULTSET_SIZE" => {
            let v: i64 = value
                .parse()
                .map_err(|_| format!("Failed to set config value {name} to {value}"))?;
            // Any negative value means unlimited, stored as -1
            Ok(ConfigValue::Int(if v < 0 { -1 } else { v }))
        }
        "MAX_QUEUED_QUERIES" => {
            let v: i64 = value
                .parse()
                .map_err(|_| format!("Failed to set config value {name} to {value}"))?;
            if v <= 0 {
                return Err(format!("Failed to set config value {name} to {value}"));
            }
            Ok(ConfigValue::Uint(v as u64))
        }
        "VKEY_MAX_ENTITY_COUNT" => {
            let v: i64 = value
                .parse()
                .map_err(|_| format!("Failed to set config value {name} to {value}"))?;
            Ok(ConfigValue::Int(v))
        }
        "MAX_INFO_QUERIES" => {
            let v: i64 = value
                .parse()
                .map_err(|_| format!("Failed to set config value {name} to {value}"))?;
            if v < 0 {
                return Err(format!("Failed to set config value {name} to {value}"));
            }
            // C clamps to the cap instead of failing, so a too-large value is
            // accepted and reported back as the cap
            Ok(ConfigValue::Int(v.min(MAX_INFO_QUERIES_CAP)))
        }
        "JS_HEAP_SIZE" | "JS_STACK_SIZE" => {
            let v: i64 = value
                .parse()
                .map_err(|_| format!("Failed to set config value {name} to {value}"))?;
            if v < 0 {
                return Err(format!(
                    "Failed to set config value {name} to {value} - value must be non-negative"
                ));
            }
            Ok(ConfigValue::Int(v))
        }
        // Read-only configs
        "THREAD_COUNT"
        | "INDEX_WORKER_THREADS"
        | "OMP_THREAD_COUNT"
        | "CACHE_SIZE"
        | "NODE_CREATION_BUFFER"
        | "BOLT_PORT"
        | "IMPORT_FOLDER"
        | "TEMP_FOLDER" => {
            Err("This configuration parameter cannot be set at run-time".to_string())
        }
        _ => Err(format!("Unknown configuration field '{name}'")),
    }
}

/// Cross-validate timeout configuration constraints.
///
/// Rules:
/// - Cannot set deprecated TIMEOUT when TIMEOUT_DEFAULT or TIMEOUT_MAX are active
/// - TIMEOUT_DEFAULT cannot exceed TIMEOUT_MAX (when TIMEOUT_MAX > 0)
/// - TIMEOUT_MAX cannot be lower than TIMEOUT_DEFAULT (when TIMEOUT_DEFAULT > 0)
fn validate_timeout_cross_constraints(validated: &[(&str, ConfigValue)]) -> Result<(), String> {
    // Compute what the new values will be after applying.
    let mut new_timeout_default = TIMEOUT_DEFAULT.load(Ordering::Relaxed);
    let mut new_timeout_max = TIMEOUT_MAX.load(Ordering::Relaxed);

    let mut setting_timeout = false;
    for (name, val) in validated {
        match *name {
            "TIMEOUT" => {
                setting_timeout = true;
            }
            "TIMEOUT_DEFAULT" => new_timeout_default = val.as_i64(),
            "TIMEOUT_MAX" => new_timeout_max = val.as_i64(),
            _ => {}
        }
    }

    // Cannot set deprecated TIMEOUT when new configs are active
    if setting_timeout && (new_timeout_default > 0 || new_timeout_max > 0) {
        return Err("The TIMEOUT configuration parameter is deprecated. Please set TIMEOUT_MAX and TIMEOUT_DEFAULT instead".to_string());
    }

    // TIMEOUT_DEFAULT cannot exceed TIMEOUT_MAX (when TIMEOUT_MAX > 0)
    if new_timeout_default > 0 && new_timeout_max > 0 && new_timeout_default > new_timeout_max {
        // Determine which one is being set to produce the right error message
        let setting_default = validated.iter().any(|(n, _)| *n == "TIMEOUT_DEFAULT");
        if setting_default {
            return Err("TIMEOUT_DEFAULT configuration parameter cannot be set to a value higher than TIMEOUT_MAX".to_string());
        }
        return Err("TIMEOUT_MAX configuration parameter cannot be set to a value lower than TIMEOUT_DEFAULT".to_string());
    }

    Ok(())
}

enum ConfigValue {
    Int(i64),
    Uint(u64),
}

/// Apply a validated config value.
fn apply_config_set(
    ctx: &Context,
    name: &str,
    val: &ConfigValue,
) {
    match name {
        "TIMEOUT" => TIMEOUT.store(val.as_i64(), Ordering::Relaxed),
        "TIMEOUT_DEFAULT" => TIMEOUT_DEFAULT.store(val.as_i64(), Ordering::Relaxed),
        "TIMEOUT_MAX" => TIMEOUT_MAX.store(val.as_i64(), Ordering::Relaxed),
        "RESULTSET_SIZE" => RESULTSET_SIZE.store(val.as_i64(), Ordering::Relaxed),
        "MAX_QUEUED_QUERIES" => MAX_QUEUED_QUERIES.store(val.as_u64(), Ordering::Relaxed),
        "QUERY_MEM_CAPACITY" => QUERY_MEM_CAPACITY.store(val.as_i64(), Ordering::Relaxed),
        "DELTA_MAX_PENDING_CHANGES" => {
            DELTA_MAX_PENDING_CHANGES.store(val.as_i64(), Ordering::Relaxed);
        }
        "EFFECTS_COMPRESSION" => EFFECTS_COMPRESSION.store(val.as_i64(), Ordering::Relaxed),
        "EFFECTS_THRESHOLD" => {
            EFFECTS_THRESHOLD_DEPRECATED.store(val.as_i64(), Ordering::Relaxed);
        }
        "ASYNC_DELETE" => ASYNC_DELETE.store(val.as_i64(), Ordering::Relaxed),
        "CMD_INFO" => CONFIGURATION_CMD_INFO.store(val.as_i64() != 0, Ordering::Relaxed),
        "MAX_INFO_QUERIES" => MAX_INFO_QUERIES.store(val.as_i64(), Ordering::Relaxed),
        "DELAY_INDEXING" => {
            *CONFIGURATION_DELAY_INDEXING.lock(ctx) = val.as_i64() != 0;
        }
        "VKEY_MAX_ENTITY_COUNT" => {
            *CONFIGURATION_VKEY_MAX_ENTITY_COUNT.lock(ctx) = val.as_i64();
        }
        "JS_HEAP_SIZE" => {
            *CONFIGURATION_JS_HEAP_SIZE.lock(ctx) = val.as_i64();
            graph::udf::js_context::JS_HEAP_SIZE
                .store(val.as_i64(), std::sync::atomic::Ordering::Relaxed);
        }
        "JS_STACK_SIZE" => {
            *CONFIGURATION_JS_STACK_SIZE.lock(ctx) = val.as_i64();
            graph::udf::js_context::JS_STACK_SIZE
                .store(val.as_i64(), std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }
}

impl ConfigValue {
    const fn as_i64(&self) -> i64 {
        match self {
            Self::Int(v) => *v,
            Self::Uint(v) => *v as i64,
        }
    }
    const fn as_u64(&self) -> u64 {
        match self {
            Self::Int(v) => *v as u64,
            Self::Uint(v) => *v,
        }
    }
}

pub fn graph_config(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let sub_command = args.next_str()?;

    match sub_command.to_uppercase().as_str() {
        "GET" => {
            let name = args.next_str()?;
            if name == "*" {
                // Return all configs in order.
                let mut result = Vec::with_capacity(CONFIG_NAMES.len());
                for &cfg_name in CONFIG_NAMES {
                    result.push(
                        config_get_one(ctx, cfg_name).map_err(redis_module::RedisError::String)?,
                    );
                }
                Ok(RedisValue::Array(result))
            } else {
                let upper = name.to_uppercase();
                config_get_one(ctx, &upper).map_err(redis_module::RedisError::String)
            }
        }
        "SET" => {
            // Collect all name-value pairs.
            let mut pairs = Vec::new();
            while let Ok(n) = args.next_str() {
                let name = n.to_uppercase();
                let value = args.next_str().map_err(|_| {
                    redis_module::RedisError::Str("Missing value for configuration parameter")
                })?;
                pairs.push((name, value.to_string()));
            }

            if pairs.is_empty() {
                return Err(redis_module::RedisError::Str(
                    "Missing configuration parameter name",
                ));
            }

            // Validate all pairs first (for atomic multi-set).
            let mut validated = Vec::with_capacity(pairs.len());
            for (name, value) in &pairs {
                let v =
                    validate_config_set(name, value).map_err(redis_module::RedisError::String)?;
                validated.push((name.as_str(), v));
            }

            // Cross-validate timeout configuration constraints.
            validate_timeout_cross_constraints(&validated)
                .map_err(redis_module::RedisError::String)?;

            // Apply all validated values.
            let mut js_config_changed = false;
            for (name, val) in validated {
                if name == "JS_HEAP_SIZE" || name == "JS_STACK_SIZE" {
                    js_config_changed = true;
                }
                apply_config_set(ctx, name, &val);
            }

            // Bump UDF repo version once after all JS config fields are applied
            // so concurrent queries cannot rebuild between fields.
            if js_config_changed {
                graph::udf::get_udf_repo().bump_version();
            }

            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        _ => Err(redis_module::RedisError::String(
            "Unknown subcommand for GRAPH.CONFIG".to_string(),
        )),
    }
}
