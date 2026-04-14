//! Query result serialization helpers.
//!
//! Converts runtime values and summaries into Redis protocol replies in
//! compact and verbose formats, including execution statistics.
//!
//! ## Response envelope
//! Both compact and verbose outputs follow this top-level shape:
//! ```text
//! [
//!   header columns,
//!   result rows,
//!   execution statistics
//! ]
//! ```
//!
//! ## Compact vs verbose
//! - Compact: each value is tagged with a numeric type code for efficient
//!   client parsing.
//! - Verbose: values are emitted in human-readable forms (labels/type names,
//!   formatted temporal values).
//!
//! This separation keeps wire compatibility with clients that expect either
//! machine-oriented or human-oriented output.

use graph::runtime::{
    runtime::{QueryStatistics, ResultSummary, Runtime},
    value::Value,
};
use redis_module::{Context, raw};
use std::os::raw::c_char;

#[allow(clippy::too_many_lines)]
pub fn reply_compact_value(
    ctx: &Context,
    runtime: &Runtime<'_>,
    r: &Value,
) {
    match r {
        Value::Null => {
            raw::reply_with_long_long(ctx.ctx, 1);
            raw::reply_with_null(ctx.ctx);
        }
        Value::Bool(x) => {
            raw::reply_with_long_long(ctx.ctx, 4);
            let str = if *x { "true" } else { "false" };
            raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
        }
        Value::Int(x) => {
            raw::reply_with_long_long(ctx.ctx, 3);
            raw::reply_with_long_long(ctx.ctx, *x as _);
        }
        Value::Float(x) => {
            raw::reply_with_long_long(ctx.ctx, 5);
            let str = format!("{x:.14e}");
            raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
        }
        Value::String(x) => {
            raw::reply_with_long_long(ctx.ctx, 2);
            raw::reply_with_string_buffer(ctx.ctx, x.as_str().as_ptr().cast::<c_char>(), x.len());
        }
        Value::Datetime(ts) => {
            raw::reply_with_long_long(ctx.ctx, 13);
            raw::reply_with_long_long(ctx.ctx, *ts as _);
        }
        Value::Date(ts) => {
            raw::reply_with_long_long(ctx.ctx, 14);
            raw::reply_with_long_long(ctx.ctx, *ts as _);
        }
        Value::Time(ts) => {
            raw::reply_with_long_long(ctx.ctx, 15);
            raw::reply_with_long_long(ctx.ctx, *ts as _);
        }
        Value::Duration(dur) => {
            raw::reply_with_long_long(ctx.ctx, 16);
            raw::reply_with_long_long(ctx.ctx, *dur as _);
        }
        Value::List(values) => {
            raw::reply_with_long_long(ctx.ctx, 6);
            raw::reply_with_array(ctx.ctx, values.len() as _);
            for v in values.iter() {
                raw::reply_with_array(ctx.ctx, 2);
                reply_compact_value(ctx, runtime, v);
            }
        }
        Value::Map(map) => {
            raw::reply_with_long_long(ctx.ctx, 10);
            raw::reply_with_array(ctx.ctx, (map.len() * 2) as _);

            for (key, value) in map.iter() {
                raw::reply_with_string_buffer(
                    ctx.ctx,
                    key.as_str().as_ptr().cast::<c_char>(),
                    key.len(),
                );
                raw::reply_with_array(ctx.ctx, 2);
                reply_compact_value(ctx, runtime, value);
            }
        }
        Value::Node(id) => {
            raw::reply_with_long_long(ctx.ctx, 8);
            raw::reply_with_array(ctx.ctx, 3);
            raw::reply_with_long_long(ctx.ctx, u64::from(*id) as _);
            let dn = runtime.deleted_nodes.borrow();
            if let Some(x) = dn.get(id) {
                raw::reply_with_array(ctx.ctx, x.labels.len() as _);
                for label in &x.labels {
                    raw::reply_with_long_long(ctx.ctx, usize::from(*label) as _);
                }
                raw::reply_with_array(ctx.ctx, x.attrs.len() as _);
                for (key, value) in x.attrs.iter() {
                    raw::reply_with_array(ctx.ctx, 3);
                    let key = runtime.g.borrow().get_node_attribute_id(key).unwrap();
                    raw::reply_with_long_long(ctx.ctx, key as _);
                    reply_compact_value(ctx, runtime, value);
                }
            } else {
                let bg = runtime.g.borrow();
                raw::reply_with_array(ctx.ctx, i64::from(raw::REDISMODULE_POSTPONED_LEN));
                let labels_len = bg
                    .get_node_label_ids(*id)
                    .inspect(|label| {
                        raw::reply_with_long_long(ctx.ctx, usize::from(*label) as _);
                    })
                    .count();
                unsafe {
                    raw::RedisModule_ReplySetArrayLength.unwrap()(ctx.ctx, labels_len as _);
                }

                raw::reply_with_array(ctx.ctx, i64::from(raw::REDISMODULE_POSTPONED_LEN));
                let attrs: Vec<_> = bg.get_node_all_attrs_by_id(*id).collect();
                for (key, value) in &attrs {
                    raw::reply_with_array(ctx.ctx, 3);
                    raw::reply_with_long_long(ctx.ctx, (*key).into());
                    reply_compact_value(ctx, runtime, value);
                }
                let attrs_len = attrs.len();
                drop(bg);
                unsafe {
                    raw::RedisModule_ReplySetArrayLength.unwrap()(ctx.ctx, attrs_len as _);
                }
            }
        }
        Value::Relationship(rel) => {
            let (rel_id, rel_src, rel_dst) = (&rel.0, &rel.1, &rel.2);
            raw::reply_with_long_long(ctx.ctx, 7);
            raw::reply_with_array(ctx.ctx, 5);
            raw::reply_with_long_long(ctx.ctx, u64::from(*rel_id) as _);
            let dr = runtime.deleted_relationships.borrow();
            if let Some(x) = dr.get(rel_id) {
                let bg = runtime.g.borrow();
                let type_id = bg.get_type_id(&x.type_name).unwrap();
                raw::reply_with_long_long(ctx.ctx, usize::from(type_id) as _);
                raw::reply_with_long_long(ctx.ctx, u64::from(*rel_src) as _);
                raw::reply_with_long_long(ctx.ctx, u64::from(*rel_dst) as _);
                let node_attr_offset = bg.node_attribute_count();
                raw::reply_with_array(ctx.ctx, x.attrs.len() as _);
                for (key, value) in x.attrs.iter() {
                    raw::reply_with_array(ctx.ctx, 3);
                    let key = bg.get_relationship_attribute_id(key).unwrap();
                    raw::reply_with_long_long(ctx.ctx, (key + node_attr_offset) as _);
                    reply_compact_value(ctx, runtime, value);
                }
                drop(bg);
            } else {
                let bg = runtime.g.borrow();
                raw::reply_with_long_long(
                    ctx.ctx,
                    usize::from(bg.get_relationship_type_id(*rel_id)) as _,
                );
                raw::reply_with_long_long(ctx.ctx, u64::from(*rel_src) as _);
                raw::reply_with_long_long(ctx.ctx, u64::from(*rel_dst) as _);
                let node_attr_offset = bg.node_attribute_count() as i64;
                raw::reply_with_array(ctx.ctx, i64::from(raw::REDISMODULE_POSTPONED_LEN));
                let attrs: Vec<_> = bg.get_relationship_all_attrs_by_id(*rel_id).collect();
                for (key, value) in &attrs {
                    raw::reply_with_array(ctx.ctx, 3);
                    raw::reply_with_long_long(ctx.ctx, i64::from(*key) + node_attr_offset);
                    reply_compact_value(ctx, runtime, value);
                }
                let attrs_len = attrs.len();
                drop(bg);
                unsafe {
                    raw::RedisModule_ReplySetArrayLength.unwrap()(ctx.ctx, attrs_len as _);
                }
            }
        }
        Value::Path(path) => {
            raw::reply_with_long_long(ctx.ctx, 9);
            raw::reply_with_array(ctx.ctx, 2);

            let mut nodes = 0;
            let mut rels = 0;
            for node in path.iter() {
                match node {
                    Value::Node(_) => nodes += 1,
                    Value::Relationship(_) => rels += 1,
                    _ => unreachable!("Path should only contain nodes and relationships"),
                }
            }

            raw::reply_with_array(ctx.ctx, 2);
            raw::reply_with_long_long(ctx.ctx, 6);
            raw::reply_with_array(ctx.ctx, nodes);
            for node in path.iter() {
                match node {
                    Value::Node(_) => {
                        raw::reply_with_array(ctx.ctx, 2);
                        reply_compact_value(ctx, runtime, node);
                    }
                    Value::Relationship(_) => {}
                    _ => unreachable!("Path should only contain nodes and relationships"),
                }
            }

            raw::reply_with_array(ctx.ctx, 2);
            raw::reply_with_long_long(ctx.ctx, 6);
            raw::reply_with_array(ctx.ctx, rels);
            for node in path.iter() {
                match node {
                    Value::Node(_) => {}
                    Value::Relationship(_) => {
                        raw::reply_with_array(ctx.ctx, 2);
                        reply_compact_value(ctx, runtime, node);
                    }
                    _ => unreachable!("Path should only contain nodes and relationships"),
                }
            }
        }
        Value::VecF32(vec) => {
            raw::reply_with_long_long(ctx.ctx, 12);
            raw::reply_with_array(ctx.ctx, vec.len() as _);
            for f in vec.iter() {
                raw::reply_with_double(ctx.ctx, f64::from(*f));
            }
        }
        Value::Point(point) => {
            raw::reply_with_long_long(ctx.ctx, 11);
            raw::reply_with_array(ctx.ctx, 2);

            let lat_str = format!("{:.15}", point.latitude);
            let lat_str = lat_str.trim_end_matches('0').trim_end_matches('.');
            raw::reply_with_string_buffer(
                ctx.ctx,
                lat_str.as_ptr().cast::<c_char>(),
                lat_str.len(),
            );

            let lon_str = format!("{:.15}", point.longitude);
            let lon_str = lon_str.trim_end_matches('0').trim_end_matches('.');
            raw::reply_with_string_buffer(
                ctx.ctx,
                lon_str.as_ptr().cast::<c_char>(),
                lon_str.len(),
            );
        }
    }
}

#[allow(clippy::too_many_lines)]
pub fn reply_verbose_value(
    ctx: &Context,
    runtime: &Runtime<'_>,
    r: &Value,
) {
    match r {
        Value::Null => {
            raw::reply_with_null(ctx.ctx);
        }
        Value::Bool(x) => {
            let str = if *x { "true" } else { "false" };
            raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
        }
        Value::Int(x) => {
            raw::reply_with_long_long(ctx.ctx, *x as _);
        }
        Value::Float(x) => {
            let str = format!("{x:.14e}");
            raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
        }
        Value::String(x) => {
            raw::reply_with_string_buffer(ctx.ctx, x.as_str().as_ptr().cast::<c_char>(), x.len());
        }
        Value::Datetime(ts) => {
            let formatted = Value::format_datetime(*ts);
            raw::reply_with_string_buffer(
                ctx.ctx,
                formatted.as_ptr().cast::<c_char>(),
                formatted.len(),
            );
        }
        Value::Date(ts) => {
            let formatted = Value::format_date(*ts);
            raw::reply_with_string_buffer(
                ctx.ctx,
                formatted.as_ptr().cast::<c_char>(),
                formatted.len(),
            );
        }
        Value::Time(ts) => {
            let formatted = Value::format_time(*ts);
            raw::reply_with_string_buffer(
                ctx.ctx,
                formatted.as_ptr().cast::<c_char>(),
                formatted.len(),
            );
        }
        Value::Duration(dur) => {
            let formatted = Value::format_duration(*dur);
            raw::reply_with_string_buffer(
                ctx.ctx,
                formatted.as_ptr().cast::<c_char>(),
                formatted.len(),
            );
        }
        Value::List(values) => {
            raw::reply_with_array(ctx.ctx, values.len() as _);
            for v in values.iter() {
                reply_verbose_value(ctx, runtime, v);
            }
        }
        Value::Map(map) => {
            raw::reply_with_array(ctx.ctx, (map.len() * 2) as _);

            for (key, value) in map.iter() {
                raw::reply_with_string_buffer(
                    ctx.ctx,
                    key.as_str().as_ptr().cast::<c_char>(),
                    key.len(),
                );
                reply_verbose_value(ctx, runtime, value);
            }
        }
        Value::Node(id) => {
            raw::reply_with_array(ctx.ctx, 3);
            raw::reply_with_long_long(ctx.ctx, u64::from(*id) as _);
            let bg = runtime.g.borrow();
            let dn = runtime.deleted_nodes.borrow();
            if let Some(x) = dn.get(id) {
                raw::reply_with_array(ctx.ctx, x.labels.len() as _);
                for label in &x.labels {
                    let label = bg.get_label_by_id(*label);
                    raw::reply_with_string_buffer(
                        ctx.ctx,
                        label.as_ptr().cast::<c_char>(),
                        label.len(),
                    );
                }
                raw::reply_with_array(ctx.ctx, x.attrs.len() as _);
                for (key, value) in x.attrs.iter() {
                    raw::reply_with_array(ctx.ctx, 2);
                    raw::reply_with_string_buffer(
                        ctx.ctx,
                        key.as_ptr().cast::<c_char>(),
                        key.len(),
                    );
                    reply_verbose_value(ctx, runtime, value);
                }
            } else {
                raw::reply_with_array(ctx.ctx, i64::from(raw::REDISMODULE_POSTPONED_LEN));
                let labels_len = bg
                    .get_node_labels(*id)
                    .inspect(|label| {
                        raw::reply_with_string_buffer(
                            ctx.ctx,
                            label.as_ptr().cast::<c_char>(),
                            label.len(),
                        );
                    })
                    .count();
                unsafe {
                    raw::RedisModule_ReplySetArrayLength.unwrap()(ctx.ctx, labels_len as _);
                }

                raw::reply_with_array(ctx.ctx, i64::from(raw::REDISMODULE_POSTPONED_LEN));
                let attrs_len = bg
                    .get_node_all_attrs(*id)
                    .inspect(|(key, value)| {
                        raw::reply_with_array(ctx.ctx, 2);
                        raw::reply_with_string_buffer(
                            ctx.ctx,
                            key.as_ptr().cast::<c_char>(),
                            key.len(),
                        );
                        reply_verbose_value(ctx, runtime, value);
                    })
                    .count();
                drop(bg);
                unsafe {
                    raw::RedisModule_ReplySetArrayLength.unwrap()(ctx.ctx, attrs_len as _);
                }
            }
        }
        Value::Relationship(rel) => {
            let (rel_id, rel_src, rel_dst) = (&rel.0, &rel.1, &rel.2);
            raw::reply_with_array(ctx.ctx, 5);
            raw::reply_with_long_long(ctx.ctx, u64::from(*rel_id) as _);
            let dr = runtime.deleted_relationships.borrow();
            if let Some(x) = dr.get(rel_id) {
                let bg = runtime.g.borrow();
                let type_id = bg.get_type_id(&x.type_name).unwrap();
                raw::reply_with_long_long(ctx.ctx, usize::from(type_id) as _);
                raw::reply_with_long_long(ctx.ctx, u64::from(*rel_src) as _);
                raw::reply_with_long_long(ctx.ctx, u64::from(*rel_dst) as _);
                raw::reply_with_array(ctx.ctx, x.attrs.len() as _);
                for (key, value) in x.attrs.iter() {
                    raw::reply_with_array(ctx.ctx, 2);
                    raw::reply_with_string_buffer(
                        ctx.ctx,
                        key.as_ptr().cast::<c_char>(),
                        key.len(),
                    );
                    reply_verbose_value(ctx, runtime, value);
                }
            } else {
                let bg = runtime.g.borrow();
                let rel_type = bg.get_relationship_type_id(*rel_id);
                raw::reply_with_long_long(ctx.ctx, usize::from(rel_type) as _);
                raw::reply_with_long_long(ctx.ctx, u64::from(*rel_src) as _);
                raw::reply_with_long_long(ctx.ctx, u64::from(*rel_dst) as _);
                raw::reply_with_array(ctx.ctx, i64::from(raw::REDISMODULE_POSTPONED_LEN));
                let attrs_len = bg
                    .get_relationship_all_attrs(*rel_id)
                    .inspect(|(key, value)| {
                        raw::reply_with_array(ctx.ctx, 2);
                        raw::reply_with_string_buffer(
                            ctx.ctx,
                            key.as_ptr().cast::<c_char>(),
                            key.len(),
                        );
                        reply_verbose_value(ctx, runtime, value);
                    })
                    .count();
                drop(bg);
                unsafe {
                    raw::RedisModule_ReplySetArrayLength.unwrap()(ctx.ctx, attrs_len as _);
                }
            }
        }
        Value::Path(path) => {
            raw::reply_with_array(ctx.ctx, path.len() as _);

            for node in path.iter() {
                match node {
                    Value::Relationship(_) | Value::Node(_) => {
                        reply_verbose_value(ctx, runtime, node);
                    }
                    _ => unreachable!("Path should only contain nodes and relationships"),
                }
            }
        }
        Value::VecF32(vec) => {
            raw::reply_with_array(ctx.ctx, vec.len() as _);
            for f in vec.iter() {
                raw::reply_with_double(ctx.ctx, f64::from(*f));
            }
        }
        Value::Point(point) => {
            let str = format!(
                "point({{latitude:{}, longitude:{}}})",
                point.latitude, point.longitude
            );
            raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
        }
    }
}

pub fn reply_stats(
    ctx: &Context,
    stats: &QueryStatistics,
    version: u64,
) {
    let mut stats_len = 3;
    if stats.labels_added > 0 {
        stats_len += 1;
    }
    if stats.labels_removed > 0 {
        stats_len += 1;
    }
    if stats.nodes_created > 0 {
        stats_len += 1;
    }
    if stats.nodes_deleted > 0 {
        stats_len += 1;
    }
    if stats.properties_set > 0 {
        stats_len += 1;
    }
    if stats.properties_removed > 0 {
        stats_len += 1;
    }
    if stats.relationships_created > 0 {
        stats_len += 1;
    }
    if stats.relationships_deleted > 0 {
        stats_len += 1;
    }
    if stats.indexes_created > 0 {
        stats_len += 1;
    }
    if stats.indexes_dropped > 0 {
        stats_len += 1;
    }

    raw::reply_with_array(ctx.ctx, stats_len.into());
    if stats.labels_added > 0 {
        let str = format!("Labels added: {}", stats.labels_added);
        raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    }
    if stats.labels_removed > 0 {
        let str = format!("Labels removed: {}", stats.labels_removed);
        raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    }
    if stats.nodes_created > 0 {
        let str = format!("Nodes created: {}", stats.nodes_created);
        raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    }
    if stats.nodes_deleted > 0 {
        let str = format!("Nodes deleted: {}", stats.nodes_deleted);
        raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    }
    if stats.properties_set > 0 {
        let str = format!("Properties set: {}", stats.properties_set);
        raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    }
    if stats.properties_removed > 0 {
        let str = format!("Properties removed: {}", stats.properties_removed);
        raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    }
    if stats.relationships_created > 0 {
        let str = format!("Relationships created: {}", stats.relationships_created);
        raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    }
    if stats.relationships_deleted > 0 {
        let str = format!("Relationships deleted: {}", stats.relationships_deleted);
        raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    }
    if stats.indexes_created > 0 {
        let str = format!("Indices created: {}", stats.indexes_created);
        raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    }
    if stats.indexes_dropped > 0 {
        let str = format!("Indices deleted: {}", stats.indexes_dropped);
        raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    }
    let str = format!("Cached execution: {}", i32::from(stats.cached));
    raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    let mut buffer = ryu::Buffer::new();
    let str = buffer.format(stats.execution_time);
    let str = format!("Query internal execution time: {str} milliseconds");
    raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
    let str = format!("Graph version: {version}");
    raw::reply_with_string_buffer(ctx.ctx, str.as_ptr().cast::<c_char>(), str.len());
}

fn reply_result<const COMPACT: bool>(
    ctx: &Context,
    runtime: &Runtime<'_>,
    result: &ResultSummary<'_>,
) {
    raw::reply_with_array(ctx.ctx, 3);
    raw::reply_with_array(ctx.ctx, runtime.return_names.len() as _);
    for name in &runtime.return_names {
        raw::reply_with_array(ctx.ctx, 2);
        raw::reply_with_long_long(ctx.ctx, 1);
        raw::reply_with_string_buffer(
            ctx.ctx,
            name.as_str().as_ptr().cast::<c_char>(),
            name.as_str().len(),
        );
    }
    let total: usize = result
        .result
        .iter()
        .map(graph::runtime::batch::Batch::active_len)
        .sum();
    raw::reply_with_array(ctx.ctx, total as _);
    for batch in &result.result {
        for row in batch.active_env_iter() {
            raw::reply_with_array(ctx.ctx, runtime.return_names.len() as _);
            for name in &runtime.return_names {
                if COMPACT {
                    raw::reply_with_array(ctx.ctx, 2);
                    reply_compact_value(ctx, runtime, row.get(name).unwrap());
                } else {
                    reply_verbose_value(ctx, runtime, row.get(name).unwrap());
                }
            }
        }
    }
    reply_stats(ctx, &result.stats, runtime.g.borrow().version);
}

pub fn reply_verbose(
    ctx: &Context,
    runtime: &Runtime<'_>,
    result: &ResultSummary<'_>,
) {
    reply_result::<false>(ctx, runtime, result);
}

pub fn reply_compact(
    ctx: &Context,
    runtime: &Runtime<'_>,
    result: &ResultSummary<'_>,
) {
    reply_result::<true>(ctx, runtime, result);
}
