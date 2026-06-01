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
use std::fmt::Write;
use std::os::raw::c_char;
use std::sync::Arc;

/// Build a Cypher-style string representation of a value, matching the C
/// FalkorDB `SIValue_ToString` output used for verbose list/map/path/vector.
fn format_value_to_string(
    runtime: &Runtime<'_>,
    v: &Value,
    out: &mut String,
) {
    match v {
        Value::Null => out.push_str("NULL"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Value::Float(f) => {
            let _ = write!(out, "{f:.6}");
        }
        Value::String(s) => out.push_str(s),
        Value::Datetime(ts) => out.push_str(&Value::format_datetime(*ts)),
        Value::Date(ts) => out.push_str(&Value::format_date(*ts)),
        Value::Time(ts) => out.push_str(&Value::format_time(*ts)),
        Value::Duration(d) => out.push_str(&Value::format_duration(*d)),
        Value::List(values) | Value::Path(values) => {
            out.push('[');
            for (i, item) in values.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                format_value_to_string(runtime, item, out);
            }
            out.push(']');
        }
        Value::Map(map) => {
            out.push('{');
            for (i, (k, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(k);
                out.push_str(": ");
                format_value_to_string(runtime, val, out);
            }
            out.push('}');
        }
        Value::Node(id) => {
            let _ = write!(out, "({})", u64::from(*id));
        }
        Value::Relationship(rel) => {
            let _ = write!(out, "[{}]", u64::from(rel.0));
        }
        Value::VecF32(vec) => {
            out.push('<');
            for (i, f) in vec.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{:.6}", f64::from(*f));
            }
            out.push('>');
        }
        Value::Point(point) => {
            let _ = write!(
                out,
                "point({{latitude: {:.6}, longitude: {:.6}}})",
                point.latitude, point.longitude
            );
        }
    }
}

fn reply_with_str(
    ctx: &Context,
    s: &str,
) {
    raw::reply_with_string_buffer(ctx.ctx, s.as_ptr().cast::<c_char>(), s.len());
}

/// Format a double using C's `%.*g` (`precision` significant digits, shortest
/// of %e/%f with trailing zeros stripped). Calls libc snprintf for exact parity
/// with FalkorDB C output.
pub fn format_g(
    d: f64,
    precision: i32,
) -> String {
    let fmt = c"%.*g";
    let mut buf = [0u8; 64];
    let n = unsafe {
        libc::snprintf(
            buf.as_mut_ptr().cast::<c_char>(),
            buf.len(),
            fmt.as_ptr(),
            precision,
            d,
        )
    };
    let n = n.max(0) as usize;
    std::str::from_utf8(&buf[..n.min(buf.len() - 1)])
        .unwrap_or("")
        .to_owned()
}

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
            let str = format_g(*x, 15);
            reply_with_str(ctx, &str);
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

                let attrs = bg.get_node_all_attrs_by_id(*id);
                raw::reply_with_array(ctx.ctx, attrs.len() as _);
                for (key, value) in attrs.iter() {
                    raw::reply_with_array(ctx.ctx, 3);
                    raw::reply_with_long_long(ctx.ctx, *key as _);
                    reply_compact_value(ctx, runtime, value);
                }
                drop(bg);
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
                raw::reply_with_array(ctx.ctx, x.attrs.len() as _);
                for (key, value) in x.attrs.iter() {
                    raw::reply_with_array(ctx.ctx, 3);
                    let key = bg.get_global_attribute_id(key).unwrap();
                    raw::reply_with_long_long(ctx.ctx, key as _);
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
                let attrs = bg.get_relationship_all_attrs_by_id(*rel_id);
                raw::reply_with_array(ctx.ctx, attrs.len() as _);
                for (key, value) in attrs.iter() {
                    raw::reply_with_array(ctx.ctx, 3);
                    raw::reply_with_long_long(
                        ctx.ctx,
                        bg.rel_attr_id_to_global(*key).unwrap_or(0) as _,
                    );
                    reply_compact_value(ctx, runtime, value);
                }
                drop(bg);
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

            let lat_str = format_g(f64::from(point.latitude), 15);
            reply_with_str(ctx, &lat_str);
            let lon_str = format_g(f64::from(point.longitude), 15);
            reply_with_str(ctx, &lon_str);
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
            let str = format_g(*x, 15);
            reply_with_str(ctx, &str);
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
        Value::List(_) | Value::Map(_) | Value::Path(_) | Value::VecF32(_) => {
            let mut s = String::new();
            format_value_to_string(runtime, r, &mut s);
            reply_with_str(ctx, &s);
        }
        Value::Node(id) => {
            // [ ["id", id], ["labels", [...]], ["properties", [[name,value]…]] ]
            raw::reply_with_array(ctx.ctx, 3);

            raw::reply_with_array(ctx.ctx, 2);
            reply_with_str(ctx, "id");
            raw::reply_with_long_long(ctx.ctx, u64::from(*id) as _);

            raw::reply_with_array(ctx.ctx, 2);
            reply_with_str(ctx, "labels");

            let bg = runtime.g.borrow();
            let dn = runtime.deleted_nodes.borrow();
            if let Some(x) = dn.get(id) {
                raw::reply_with_array(ctx.ctx, x.labels.len() as _);
                for label in &x.labels {
                    let label = bg.get_label_by_id(*label);
                    reply_with_str(ctx, &label);
                }

                raw::reply_with_array(ctx.ctx, 2);
                reply_with_str(ctx, "properties");
                raw::reply_with_array(ctx.ctx, x.attrs.len() as _);
                for (key, value) in x.attrs.iter() {
                    raw::reply_with_array(ctx.ctx, 2);
                    reply_with_str(ctx, key);
                    reply_verbose_value(ctx, runtime, value);
                }
            } else {
                raw::reply_with_array(ctx.ctx, i64::from(raw::REDISMODULE_POSTPONED_LEN));
                let labels_len = bg
                    .get_node_labels(*id)
                    .inspect(|label| reply_with_str(ctx, label))
                    .count();
                unsafe {
                    raw::RedisModule_ReplySetArrayLength.unwrap()(ctx.ctx, labels_len as _);
                }

                raw::reply_with_array(ctx.ctx, 2);
                reply_with_str(ctx, "properties");
                let attrs = bg.get_node_all_attrs(*id);
                raw::reply_with_array(ctx.ctx, attrs.len() as _);
                for (key, value) in &attrs {
                    raw::reply_with_array(ctx.ctx, 2);
                    reply_with_str(ctx, key);
                    reply_verbose_value(ctx, runtime, value);
                }
                drop(bg);
            }
        }
        Value::Relationship(rel) => {
            // [ ["id",id], ["type",name], ["src_node",s], ["dest_node",d],
            //   ["properties", [[name,value]…]] ]
            let (rel_id, rel_src, rel_dst) = (&rel.0, &rel.1, &rel.2);
            raw::reply_with_array(ctx.ctx, 5);

            raw::reply_with_array(ctx.ctx, 2);
            reply_with_str(ctx, "id");
            raw::reply_with_long_long(ctx.ctx, u64::from(*rel_id) as _);

            let bg = runtime.g.borrow();
            let dr = runtime.deleted_relationships.borrow();
            let (type_name, attrs_iter): (Arc<String>, Vec<(Arc<String>, Value)>) =
                dr.get(rel_id).map_or_else(
                    || {
                        let type_id = bg.get_relationship_type_id(*rel_id);
                        let name = bg
                            .get_type(type_id)
                            .unwrap_or_else(|| Arc::new(String::new()));
                        let attrs = bg.get_relationship_all_attrs(*rel_id);
                        (name, attrs)
                    },
                    |x| {
                        (
                            x.type_name.clone(),
                            x.attrs
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect(),
                        )
                    },
                );

            raw::reply_with_array(ctx.ctx, 2);
            reply_with_str(ctx, "type");
            reply_with_str(ctx, &type_name);

            raw::reply_with_array(ctx.ctx, 2);
            reply_with_str(ctx, "src_node");
            raw::reply_with_long_long(ctx.ctx, u64::from(*rel_src) as _);

            raw::reply_with_array(ctx.ctx, 2);
            reply_with_str(ctx, "dest_node");
            raw::reply_with_long_long(ctx.ctx, u64::from(*rel_dst) as _);

            raw::reply_with_array(ctx.ctx, 2);
            reply_with_str(ctx, "properties");
            raw::reply_with_array(ctx.ctx, attrs_iter.len() as _);
            for (key, value) in &attrs_iter {
                raw::reply_with_array(ctx.ctx, 2);
                reply_with_str(ctx, key);
                reply_verbose_value(ctx, runtime, value);
            }
            drop(bg);
        }
        Value::Point(point) => {
            let str = format!(
                "point({{latitude: {:.6}, longitude: {:.6}}})",
                point.latitude, point.longitude
            );
            reply_with_str(ctx, &str);
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
        reply_with_str(ctx, &str);
    }
    if stats.labels_removed > 0 {
        let str = format!("Labels removed: {}", stats.labels_removed);
        reply_with_str(ctx, &str);
    }
    if stats.nodes_created > 0 {
        let str = format!("Nodes created: {}", stats.nodes_created);
        reply_with_str(ctx, &str);
    }
    if stats.properties_set > 0 {
        let str = format!("Properties set: {}", stats.properties_set);
        reply_with_str(ctx, &str);
    }
    if stats.properties_removed > 0 {
        let str = format!("Properties removed: {}", stats.properties_removed);
        reply_with_str(ctx, &str);
    }
    if stats.relationships_created > 0 {
        let str = format!("Relationships created: {}", stats.relationships_created);
        reply_with_str(ctx, &str);
    }
    if stats.nodes_deleted > 0 {
        let str = format!("Nodes deleted: {}", stats.nodes_deleted);
        reply_with_str(ctx, &str);
    }
    if stats.relationships_deleted > 0 {
        let str = format!("Relationships deleted: {}", stats.relationships_deleted);
        reply_with_str(ctx, &str);
    }
    if stats.indexes_created > 0 {
        let str = format!("Indices created: {}", stats.indexes_created);
        reply_with_str(ctx, &str);
    }
    if stats.indexes_dropped > 0 {
        let str = format!("Indices deleted: {}", stats.indexes_dropped);
        reply_with_str(ctx, &str);
    }
    let str = format!("Cached execution: {}", i32::from(stats.cached));
    reply_with_str(ctx, &str);
    let str = format!(
        "Query internal execution time: {:.6} milliseconds",
        stats.execution_time
    );
    reply_with_str(ctx, &str);
    let str = format!("Graph version: {version}");
    reply_with_str(ctx, &str);
}

fn reply_result<const COMPACT: bool>(
    ctx: &Context,
    runtime: &Runtime<'_>,
    result: &ResultSummary<'_>,
) {
    if runtime.return_names.is_empty() {
        // No RETURN clause — send only stats (matches C FalkorDB behavior).
        raw::reply_with_array(ctx.ctx, 1);
        reply_stats(ctx, &result.stats, runtime.g.borrow().version);
        return;
    }
    raw::reply_with_array(ctx.ctx, 3);
    raw::reply_with_array(ctx.ctx, runtime.return_names.len() as _);
    for name in &runtime.return_names {
        if COMPACT {
            raw::reply_with_array(ctx.ctx, 2);
            raw::reply_with_long_long(ctx.ctx, 1);
            raw::reply_with_string_buffer(
                ctx.ctx,
                name.as_str().as_ptr().cast::<c_char>(),
                name.as_str().len(),
            );
        } else {
            reply_with_str(ctx, name.as_str());
        }
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
