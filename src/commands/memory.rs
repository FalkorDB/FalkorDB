//! `GRAPH.MEMORY USAGE` command handler.
//!
//! Reports detailed per-component memory usage for a graph key, returned in
//! megabytes (integer division, matching the FalkorDB C implementation).
//!
//! ## Syntax
//! ```text
//! GRAPH.MEMORY USAGE <key> [SAMPLES <count>]
//! ```
//!
//! `SAMPLES` controls how many attribute entries are sampled to estimate
//! storage size (default: 100). Higher values give more accurate estimates
//! at the cost of longer computation.
//!
//! ## Response structure
//! A flat array of 9 key-value pairs (18 elements total):
//!
//! ```text
//! [ "total_graph_sz_mb",                          <int>,
//!   "label_matrices_sz_mb",                       <int>,
//!   "relation_matrices_sz_mb",                    <int>,
//!   "amortized_node_block_sz_mb",                 <int>,
//!   "amortized_node_attributes_by_label_sz_mb",   [label, mb, ...],
//!   "amortized_unlabeled_nodes_attributes_sz_mb", <int>,
//!   "amortized_edge_block_sz_mb",                 <int>,
//!   "amortized_edge_attributes_by_type_sz_mb",    [type, mb, ...],
//!   "indices_sz_mb",                              <int> ]
//! ```
//!
//! The `total_graph_sz_mb` value is the sum of all other MB-rounded
//! components, which avoids truncation discrepancies when clients verify
//! the total against the individual parts.
//!
//! Sampling can be non-trivial, so like `GRAPH.QUERY`/`GRAPH.RECORD` the report
//! is built on the thread pool with the client blocked; the main thread only
//! resolves the graph key. Running inline on the main thread would also
//! deadlock the server against a committing write (issue #726): the handler
//! holds the GIL while waiting for the read lock, while the write holds the
//! write lock and waits for the GIL.

use crate::{
    graph_core::{BlockedClient, ThreadedGraph, ffi},
    redis_type::GRAPH_TYPE,
};
use graph::threadpool::spawn;
use parking_lot::RwLock;
use redis_module::{
    Context, ContextFlags, NextArg, RedisError, RedisResult, RedisString, RedisValue,
};
use std::sync::Arc;

const MB: usize = 1 << 20;

/// The critical section: take the graph read lock, sample the attribute stores,
/// and build the flat key-value reply array. Runs on a worker thread (or
/// synchronously for MULTI/REPLICATED), never inline on the main thread under
/// the GIL.
#[allow(clippy::too_many_lines)]
fn memory_report(
    graph: &Arc<RwLock<ThreadedGraph>>,
    samples: usize,
) -> RedisResult {
    let report = graph
        .read()
        .graph
        .read()
        .borrow()
        .memory_usage_report(samples);

    // Convert each component to MB using integer division, then compute total
    // as the sum of MB-rounded values (matches the C implementation, avoiding
    // truncation discrepancies in the test assertion).
    let label_matrices_mb = (report.label_matrices_sz / MB) as i64;
    let relation_matrices_mb = (report.relation_matrices_sz / MB) as i64;
    let node_block_mb = (report.node_block_storage_sz / MB) as i64;
    let unlabeled_node_attr_mb = (report.unlabeled_node_attr_sz / MB) as i64;
    let edge_block_mb = (report.edge_block_storage_sz / MB) as i64;
    let indices_mb = (report.indices_sz / MB) as i64;

    let mut node_attr_by_label_mb: Vec<(String, i64)> = Vec::new();
    let mut node_attr_sum_mb: i64 = 0;
    for (name, sz) in &report.node_attr_by_label {
        let mb = (*sz / MB) as i64;
        node_attr_sum_mb += mb;
        node_attr_by_label_mb.push((name.as_str().to_owned(), mb));
    }

    let mut edge_attr_by_type_mb: Vec<(String, i64)> = Vec::new();
    let mut edge_attr_sum_mb: i64 = 0;
    for (name, sz) in &report.edge_attr_by_type {
        let mb = (*sz / MB) as i64;
        edge_attr_sum_mb += mb;
        edge_attr_by_type_mb.push((name.as_str().to_owned(), mb));
    }

    let total_mb = indices_mb
        + node_block_mb
        + unlabeled_node_attr_mb
        + edge_block_mb
        + label_matrices_mb
        + node_attr_sum_mb
        + edge_attr_sum_mb
        + relation_matrices_mb;

    // Build a flat array: [key, value, key, value, ...] (9 pairs = 18 elements)
    let mut out = Vec::with_capacity(18);

    // 1. total_graph_sz_mb
    out.push(RedisValue::SimpleString("total_graph_sz_mb".into()));
    out.push(RedisValue::Integer(total_mb));

    // 2. label_matrices_sz_mb
    out.push(RedisValue::SimpleString("label_matrices_sz_mb".into()));
    out.push(RedisValue::Integer(label_matrices_mb));

    // 3. relation_matrices_sz_mb
    out.push(RedisValue::SimpleString("relation_matrices_sz_mb".into()));
    out.push(RedisValue::Integer(relation_matrices_mb));

    // 4. amortized_node_block_sz_mb
    out.push(RedisValue::SimpleString(
        "amortized_node_block_sz_mb".into(),
    ));
    out.push(RedisValue::Integer(node_block_mb));

    // 5. amortized_node_attributes_by_label_sz_mb (nested flat map)
    out.push(RedisValue::SimpleString(
        "amortized_node_attributes_by_label_sz_mb".into(),
    ));
    let mut label_attrs = Vec::new();
    for (name, mb) in &node_attr_by_label_mb {
        label_attrs.push(RedisValue::SimpleString(name.clone()));
        label_attrs.push(RedisValue::Integer(*mb));
    }
    out.push(RedisValue::Array(label_attrs));

    // 6. amortized_unlabeled_nodes_attributes_sz_mb
    out.push(RedisValue::SimpleString(
        "amortized_unlabeled_nodes_attributes_sz_mb".into(),
    ));
    out.push(RedisValue::Integer(unlabeled_node_attr_mb));

    // 7. amortized_edge_block_sz_mb
    out.push(RedisValue::SimpleString(
        "amortized_edge_block_sz_mb".into(),
    ));
    out.push(RedisValue::Integer(edge_block_mb));

    // 8. amortized_edge_attributes_by_type_sz_mb (nested flat map)
    out.push(RedisValue::SimpleString(
        "amortized_edge_attributes_by_type_sz_mb".into(),
    ));
    let mut type_attrs = Vec::new();
    for (name, mb) in &edge_attr_by_type_mb {
        type_attrs.push(RedisValue::SimpleString(name.clone()));
        type_attrs.push(RedisValue::Integer(*mb));
    }
    out.push(RedisValue::Array(type_attrs));

    // 9. indices_sz_mb
    out.push(RedisValue::SimpleString("indices_sz_mb".into()));
    out.push(RedisValue::Integer(indices_mb));

    Ok(RedisValue::Array(out))
}

pub fn graph_memory(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    // GRAPH.MEMORY USAGE <key> [SAMPLES <count>]
    // args[0] = "GRAPH.MEMORY"
    let mut args = args.into_iter().skip(1);
    let arg_count = args.len();

    // Must have 2 or 4 remaining args: USAGE <key> [SAMPLES <n>]
    if arg_count != 2 && arg_count != 4 {
        return Err(RedisError::WrongArity);
    }

    // First arg must be "USAGE"
    let subcmd = args.next_arg()?;
    if !subcmd.to_string_lossy().eq_ignore_ascii_case("USAGE") {
        return Err(RedisError::Str(
            "ERR unknown subcommand. Try GRAPH.MEMORY USAGE <key> [SAMPLES <count>]",
        ));
    }

    let key_name = args.next_arg()?;

    // Parse optional SAMPLES <count>
    let samples: usize = if arg_count == 4 {
        let samples_kw = args.next_arg()?;
        if !samples_kw.to_string_lossy().eq_ignore_ascii_case("SAMPLES") {
            return Err(RedisError::Str("ERR expected SAMPLES keyword"));
        }
        let count_str = args.next_arg()?;
        let count_s = count_str.to_string_lossy();
        // Reject negative values (starts with '-')
        if count_s.starts_with('-') {
            return Err(RedisError::Str(
                "ERR SAMPLES count must be a positive integer",
            ));
        }
        let count = count_s
            .parse::<usize>()
            .map_err(|_| RedisError::Str("ERR SAMPLES count must be a positive integer"))?;
        if count == 0 {
            return Err(RedisError::Str(
                "ERR SAMPLES count must be a positive integer",
            ));
        }
        count
    } else {
        100
    };

    let key = ctx.open_key(&key_name);
    let graph = key
        .get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)?
        .ok_or(RedisError::Str("Graph does not exist"))?
        .clone();

    // Blocking clients are not allowed inside MULTI/EXEC, and replicated
    // commands must complete before the handler returns (same rules as
    // GRAPH.QUERY) — run synchronously in those cases.
    if ctx.get_flags().contains(ContextFlags::MULTI)
        || ctx.get_flags().contains(ContextFlags::REPLICATED)
    {
        return memory_report(&graph, samples);
    }

    // Run on the thread pool (issue #726): executing on the main thread holds
    // the GIL while waiting for the read lock, which deadlocks against a
    // committing write holding the write lock and waiting for the GIL.
    let bc = unsafe { BlockedClient::new(ctx.ctx) };
    spawn(
        move || {
            let ctx = unsafe { ffi::get_thread_safe_context(bc.inner) };
            let ctx = Context::new(ctx);
            let _ = ctx.reply(memory_report(&graph, samples));
            drop(bc);
            unsafe { ffi::free_thread_safe_context(ctx.ctx) };
        },
        None,
    );

    Ok(RedisValue::NoReply)
}
