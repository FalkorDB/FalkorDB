//! Database introspection and management procedures.
//!
//! These are invoked via Cypher `CALL` statements and return result
//! sets (lists of maps).  Each procedure is registered with
//! `FnType::Procedure(yields)` so the binder knows which columns the
//! `YIELD` clause can reference.
//!
//! ```text
//!  Cypher procedure                         Yields                          Notes
//! ──────────────────────────────────────────────────────────────────────────────────
//!  db.labels()                              {label}                         all node labels
//!  db.relationshiptypes()                   {relationshipType}              all rel types
//!  db.propertykeys()                        {propertyKey}                   all property keys
//!  db.indexes()                             {label, properties, types, ..}  index catalog
//!  db.meta.stats()                          {labels, relTypes, nodeCount,.. } graph statistics
//!  db.idx.fulltext.createNodeIndex(map)     (none)                          write procedure
//!  db.idx.fulltext.queryNodes(label, query) {node, score}                   not yet supported
//! ```
//!
//! Read-only procedures are registered with `write = false`; the
//! full-text index creation procedure uses the `write procedure:`
//! macro arm so it can be used inside write queries.

#![allow(clippy::unnecessary_wraps)]

use super::{FnArguments, FnType, Functions, Type, get_functions};
use crate::{
    index::indexer::{IndexInfo, IndexType},
    runtime::{ordermap::OrderMap, runtime::Runtime, value::Value},
};
use std::sync::Arc;
use thin_vec::{ThinVec, thin_vec};

pub fn register(funcs: &mut Functions) {
    // ── db.labels ──────────────────────────────────────────────────────
    cypher_fn!(funcs, "db.labels",
        args: [],
        ret: Type::Any,
        procedure: ["label"],
        fn db_labels(runtime, _args) {
            Ok(Value::List(Arc::new(
                runtime
                    .g
                    .borrow()
                    .get_labels()
                    .iter()
                    .map(|l| {
                        let mut map = OrderMap::default();
                        map.insert(Arc::new(String::from("label")), Value::String(l.clone()));
                        Value::Map(Arc::new(map))
                    })
                    .collect(),
            )))
        }
    );

    // ── db.relationshipTypes ───────────────────────────────────────────
    cypher_fn!(funcs, "db.relationshipTypes",
        args: [],
        ret: Type::Any,
        procedure: ["relationshipType"],
        fn db_types(runtime, _args) {
            Ok(Value::List(Arc::new(
                runtime
                    .g
                    .borrow()
                    .get_types()
                    .iter()
                    .map(|t| {
                        let mut map = OrderMap::default();
                        map.insert(
                            Arc::new(String::from("relationshipType")),
                            Value::String(t.clone()),
                        );
                        Value::Map(Arc::new(map))
                    })
                    .collect(),
            )))
        }
    );

    // ── db.propertyKeys ────────────────────────────────────────────────
    cypher_fn!(funcs, "db.propertyKeys",
        args: [],
        ret: Type::Any,
        procedure: ["propertyKey"],
        fn db_properties(runtime, _args) {
            Ok(Value::List(Arc::new(
                runtime
                    .g
                    .borrow()
                    .get_attrs()
                    .map(|p| {
                        let mut map = OrderMap::default();
                        map.insert(
                            Arc::new(String::from("propertyKey")),
                            Value::String(p.clone()),
                        );
                        Value::Map(Arc::new(map))
                    })
                    .collect(),
            )))
        }
    );

    // ── db.indexes ─────────────────────────────────────────────────────
    cypher_fn!(funcs, "db.indexes",
        args: [],
        ret: Type::Any,
        procedure: ["label", "properties", "types", "options", "language", "stopwords", "entitytype", "status", "info"],
        fn db_indexes(runtime, _args) {
            Ok(Value::List(Arc::new(
                runtime
                    .g
                    .borrow()
                    .index_info()
                    .into_iter()
                    .map(
                        |IndexInfo {
                             label,
                             pending,
                             progress,
                             total,
                             fields,
                             language,
                             stopwords,
                             entity_type,
                         }| {
                            let mut map = OrderMap::default();
                            map.insert(Arc::new(String::from("label")), Value::String(label));
                            let mut sorted_keys: Vec<_> = fields.keys().cloned().collect();
                            sorted_keys.sort();
                            map.insert(
                                Arc::new(String::from("properties")),
                                Value::List(Arc::new(sorted_keys.iter().map(|f| Value::String(f.clone())).collect())),
                            );
                            let mut types_map = OrderMap::default();
                            // Per-attribute index types. A single attribute
                            // can hold several fields of the same index type
                            // (e.g. a range index spans `range:a`,
                            // `range:a:numeric:arr`, `range:a:string:arr`),
                            // so dedupe — but do not impose an order; callers
                            // that care should compare as sets.
                            for attr in &sorted_keys {
                                let mut seen = [false; 3];
                                let mut types = thin_vec![];
                                for field in &fields[attr] {
                                    let (slot, name) = match field.ty {
                                        IndexType::Range => (0, "RANGE"),
                                        IndexType::Vector => (1, "VECTOR"),
                                        IndexType::Fulltext => (2, "FULLTEXT"),
                                    };
                                    if !seen[slot] {
                                        seen[slot] = true;
                                        types.push(Value::String(Arc::new(name.to_string())));
                                    }
                                }
                                types_map.insert(attr.clone(), Value::List(Arc::new(types)));
                            }
                            map.insert(Arc::new(String::from("types")), Value::Map(Arc::new(types_map)));
                            map.insert(Arc::new(String::from("options")), Value::Null);
                            map.insert(
                                Arc::new(String::from("language")),
                                language.map_or_else(|| Value::Null, Value::String),
                            );
                            map.insert(
                                Arc::new(String::from("stopwords")),
                                stopwords.map_or_else(
                                    || Value::Null,
                                    |sw| Value::List(Arc::new(sw.into_iter().map(Value::String).collect())),
                                ),
                            );
                            map.insert(
                                Arc::new(String::from("entitytype")),
                                Value::String(Arc::new(entity_type)),
                            );
                            map.insert(
                                Arc::new(String::from("status")),
                                if pending > 0 {
                                    Value::String(Arc::new(format!(
                                        "[Indexing] {progress}/{total}: UNDER CONSTRUCTION"
                                    )))
                                } else {
                                    Value::String(Arc::new(String::from("OPERATIONAL")))
                                },
                            );

                            // Build the `info.fields` list: one entry per
                            // underlying RediSearch field name, not per
                            // user-facing attribute. A single range attr
                            // expands into up to three RediSearch fields —
                            // the scalar (e.g. `range:a`) plus the numeric-
                            // and string-array companions used by list
                            // lookups (`a:numeric:arr`, `a:string:arr`) —
                            // and we surface all of them here for parity
                            // with the original FalkorDB implementation.
                            // Walk attributes in `sorted_keys` order
                            // (not the underlying HashMap's random
                            // iteration order) so the output is
                            // deterministic across runs — tests and
                            // clients parsing this list can rely on
                            // stable ordering.
                            let mut rs_field_names = thin_vec![];
                            for attr_key in &sorted_keys {
                                let Some(field_list) = fields.get(attr_key) else {
                                    continue;
                                };
                                for field in field_list {
                                    // Primary RediSearch field name.
                                    let name = field.name.to_str().unwrap_or("").to_string();
                                    if !name.is_empty() {
                                        rs_field_names.push(name);
                                    }
                                    // Companion array fields, present only
                                    // for range indexes.
                                    if let Some(arr_name) = field.numeric_arr_name()
                                        && let Ok(s) = arr_name.to_str() {
                                            rs_field_names.push(s.to_string());
                                        }
                                    if let Some(arr_name) = field.string_arr_name()
                                        && let Ok(s) = arr_name.to_str() {
                                            rs_field_names.push(s.to_string());
                                        }
                                }
                            }
                            // Sentinel field used by the engine to track
                            // non-indexable properties.
                            rs_field_names.push(String::from("NONE_INDEXABLE_FIELDS"));

                            // Wrap each field name as `{name: "..."}` — the
                            // map shape the Python test helpers expect.
                            let fields_list: ThinVec<Value> = rs_field_names
                                .into_iter()
                                .map(|name| {
                                    let mut field_map = OrderMap::default();
                                    field_map.insert(
                                        Arc::new(String::from("name")),
                                        Value::String(Arc::new(name)),
                                    );
                                    Value::Map(Arc::new(field_map))
                                })
                                .collect();
                            let mut info_map = OrderMap::default();
                            info_map.insert(
                                Arc::new(String::from("fields")),
                                Value::List(Arc::new(fields_list)),
                            );
                            map.insert(Arc::new(String::from("info")), Value::Map(Arc::new(info_map)));

                            Value::Map(Arc::new(map))
                        },
                    )
                    .collect(),
            )))
        }
    );

    // ── db.meta.stats ─────────────────────────────────────────────────
    cypher_fn!(funcs, "db.meta.stats",
        args: [],
        ret: Type::Any,
        procedure: ["labels", "relTypes", "relCount", "nodeCount", "labelCount", "relTypeCount", "propertyKeyCount"],
        fn db_meta_stats(runtime, _args) {
            let g = runtime.g.borrow();

            // Build labels map: label_name -> node count for that label
            let mut labels_map = OrderMap::default();
            for (idx, name) in g.get_labels().iter().enumerate() {
                labels_map.insert(
                    name.clone(),
                    Value::Int(g.label_node_count_by_idx(idx) as i64),
                );
            }

            // Build relTypes map: type_name -> edge count for that type
            let mut rel_types_map = OrderMap::default();
            for (idx, name) in g.get_types().iter().enumerate() {
                rel_types_map.insert(
                    name.clone(),
                    Value::Int(g.type_edge_count(idx) as i64),
                );
            }

            let mut row = OrderMap::default();
            row.insert(Arc::new(String::from("labels")), Value::Map(Arc::new(labels_map)));
            row.insert(Arc::new(String::from("relTypes")), Value::Map(Arc::new(rel_types_map)));
            row.insert(Arc::new(String::from("relCount")), Value::Int(g.relationship_count() as i64));
            row.insert(Arc::new(String::from("nodeCount")), Value::Int(g.node_count() as i64));
            row.insert(Arc::new(String::from("labelCount")), Value::Int(g.get_labels().len() as i64));
            row.insert(Arc::new(String::from("relTypeCount")), Value::Int(g.get_types().len() as i64));
            row.insert(Arc::new(String::from("propertyKeyCount")), Value::Int(g.property_key_count() as i64));

            Ok(Value::List(Arc::new(thin_vec![Value::Map(Arc::new(row))])))
        }
    );

    // ── db.idx.fulltext.createNodeIndex ────────────────────────────────
    cypher_fn!(funcs, "db.idx.fulltext.createNodeIndex",
        args: [Type::Map],
        ret: Type::Any,
        write procedure: [],
        fn db_fulltext_create_node_index(_runtime, _args) {
            Ok(Value::List(Arc::new(thin_vec![])))
        }
    );

    // ── db.idx.fulltext.queryNodes ─────────────────────────────────────
    cypher_fn!(funcs, "db.idx.fulltext.queryNodes",
        args: [Type::String, Type::String],
        ret: Type::Any,
        procedure: ["node", "score"],
        fn db_fulltext_query_nodes(_, _) {
            Err(String::from("db.idx.fulltext.queryNodes() is not supported in this version"))
        }
    );

    // ── db.idx.fulltext.queryRelationships ─────────────────────────────
    // Registered so the binder validates the `relationship` / `score`
    // yields. Execution is rewritten to `IR::EdgeByFulltextScan` in the
    // planner (see `planner::mod::plan` — the `QueryIR::Call` arm), so
    // this body is never actually reached at runtime.
    cypher_fn!(funcs, "db.idx.fulltext.queryRelationships",
        args: [Type::String, Type::String],
        ret: Type::Any,
        procedure: ["relationship", "score"],
        fn db_fulltext_query_relationships(_, _) {
            Err(String::from("db.idx.fulltext.queryRelationships() is not supported in this version"))
        }
    );

    // ── db.idx.fulltext.drop ──────────────────────────────────────────
    cypher_fn!(funcs, "db.idx.fulltext.drop",
        args: [Type::String],
        ret: Type::Any,
        write procedure: [],
        fn db_fulltext_drop(_runtime, _args) {
            Ok(Value::List(Arc::new(thin_vec![])))
        }
    );

    // ── db.idx.vector.queryNodes ──────────────────────────────────────
    cypher_fn!(funcs, "db.idx.vector.queryNodes",
        args: [Type::String, Type::String, Type::Any, Type::Any],
        ret: Type::Any,
        procedure: ["node", "score"],
        fn db_vector_query_nodes(_, _) {
            Err(String::from("db.idx.vector.queryNodes() is not yet supported"))
        }
    );

    // ── db.idx.vector.queryRelationships ──────────────────────────────
    cypher_fn!(funcs, "db.idx.vector.queryRelationships",
        args: [Type::String, Type::String, Type::Any, Type::Any],
        ret: Type::Any,
        procedure: ["relationship", "score"],
        fn db_vector_query_relationships(_, _) {
            Err(String::from("db.idx.vector.queryRelationships() is not yet supported"))
        }
    );

    // ── db.constraints ────────────────────────────────────────────────
    cypher_fn!(funcs, "db.constraints",
        args: [],
        ret: Type::Any,
        procedure: ["type", "label", "properties", "entitytype", "status"],
        fn db_constraints(_runtime, _args) {
            // No constraints support yet — return empty result set.
            Ok(Value::List(Arc::new(thin_vec![])))
        }
    );

    // ── dbms.procedures ───────────────────────────────────────────────
    cypher_fn!(funcs, "dbms.procedures",
        args: [],
        ret: Type::Any,
        procedure: ["name", "mode"],
        fn dbms_procedures(_, _args) {
            let funcs = get_functions();
            let mut rows: Vec<Value> = funcs
                .iter()
                .filter(|f| matches!(f.fn_type, FnType::Procedure(_)))
                .map(|f| {
                    let mut map = OrderMap::default();
                    map.insert(
                        Arc::new(String::from("name")),
                        Value::String(Arc::new(f.name.clone())),
                    );
                    map.insert(
                        Arc::new(String::from("mode")),
                        Value::String(Arc::new(String::from(
                            if f.write { "WRITE" } else { "READ" },
                        ))),
                    );
                    Value::Map(Arc::new(map))
                })
                .collect();
            rows.sort_by(|a, b| {
                let a_name = if let Value::Map(m) = a {
                    m.get(&Arc::new(String::from("name")))
                        .and_then(|v| if let Value::String(s) = v { Some(s.as_str()) } else { None })
                        .unwrap_or("")
                } else {
                    ""
                };
                let b_name = if let Value::Map(m) = b {
                    m.get(&Arc::new(String::from("name")))
                        .and_then(|v| if let Value::String(s) = v { Some(s.as_str()) } else { None })
                        .unwrap_or("")
                } else {
                    ""
                };
                a_name.cmp(b_name)
            });
            Ok(Value::List(Arc::new(rows.into())))
        }
    );

    // ── dbms.functions ────────────────────────────────────────────────
    cypher_fn!(funcs, "dbms.functions",
        args: [],
        ret: Type::Any,
        procedure: ["name", "return_type", "arguments", "internal", "reducible", "aggregation", "variable_len", "udf"],
        fn dbms_functions(_, _args) {
            let funcs = get_functions();
            let mut rows: Vec<Value> = funcs
                .iter()
                .filter(|f| !matches!(f.fn_type, FnType::Procedure(_)))
                .map(|f| {
                    let mut map = OrderMap::default();
                    map.insert(
                        Arc::new(String::from("name")),
                        Value::String(Arc::new(f.name.clone())),
                    );
                    map.insert(
                        Arc::new(String::from("return_type")),
                        Value::String(Arc::new(type_to_dbms_string(&f.ret_type))),
                    );
                    let args_list: thin_vec::ThinVec<Value> = match &f.args_type {
                        FnArguments::Fixed(types) => types
                            .iter()
                            .map(|t| Value::String(Arc::new(type_to_dbms_string(t))))
                            .collect(),
                        FnArguments::VarLength(t) => {
                            thin_vec::thin_vec![Value::String(Arc::new(type_to_dbms_string(t)))]
                        }
                    };
                    map.insert(
                        Arc::new(String::from("arguments")),
                        Value::List(Arc::new(args_list)),
                    );
                    map.insert(
                        Arc::new(String::from("internal")),
                        Value::Bool(matches!(f.fn_type, FnType::Internal)),
                    );
                    let reducible = !f.non_deterministic
                        && !matches!(f.fn_type, FnType::Aggregation { .. } | FnType::Procedure(_));
                    map.insert(
                        Arc::new(String::from("reducible")),
                        Value::Bool(reducible),
                    );
                    map.insert(
                        Arc::new(String::from("aggregation")),
                        Value::Bool(matches!(f.fn_type, FnType::Aggregation { .. })),
                    );
                    map.insert(
                        Arc::new(String::from("variable_len")),
                        Value::Bool(matches!(f.args_type, FnArguments::VarLength(_))),
                    );
                    map.insert(
                        Arc::new(String::from("udf")),
                        Value::Bool(matches!(f.fn_type, FnType::Udf)),
                    );
                    Value::Map(Arc::new(map))
                })
                .collect();
            rows.sort_by(|a, b| {
                let a_name = if let Value::Map(m) = a {
                    m.get(&Arc::new(String::from("name")))
                        .and_then(|v| if let Value::String(s) = v { Some(s.as_str()) } else { None })
                        .unwrap_or("")
                } else {
                    ""
                };
                let b_name = if let Value::Map(m) = b {
                    m.get(&Arc::new(String::from("name")))
                        .and_then(|v| if let Value::String(s) = v { Some(s.as_str()) } else { None })
                        .unwrap_or("")
                } else {
                    ""
                };
                a_name.cmp(b_name)
            });
            Ok(Value::List(Arc::new(rows.into())))
        }
    );
}

/// Convert a `Type` to a display string matching the original FalkorDB format.
fn type_to_dbms_string(t: &Type) -> String {
    match t {
        Type::Any => format_union(&[
            "Map",
            "Node",
            "Edge",
            "List",
            "Path",
            "Datetime",
            "Date",
            "Time",
            "Duration",
            "String",
            "Boolean",
            "Integer",
            "Float",
            "Null",
            "Pointer",
            "Point",
            "Vectorf32",
        ]),
        Type::Null => String::from("Null"),
        Type::Bool => String::from("Boolean"),
        Type::Int => String::from("Integer"),
        Type::Float => String::from("Float"),
        Type::String => String::from("String"),
        Type::List(_) => String::from("List"),
        Type::Map => String::from("Map"),
        Type::Node => String::from("Node"),
        Type::Relationship => String::from("Edge"),
        Type::Path => String::from("Path"),
        Type::VecF32 => String::from("Vectorf32"),
        Type::Point => String::from("Point"),
        Type::Datetime => String::from("Datetime"),
        Type::Date => String::from("Date"),
        Type::Time => String::from("Time"),
        Type::Duration => String::from("Duration"),
        Type::Union(types) => {
            let strs: Vec<String> = types.iter().map(type_to_dbms_string).collect();
            format_union_strings(&strs)
        }
        Type::Optional(inner) => type_to_dbms_string(inner),
    }
}

fn format_union(types: &[&str]) -> String {
    match types.len() {
        0 => String::new(),
        1 => String::from(types[0]),
        2 => format!("{} or {}", types[0], types[1]),
        _ => {
            let (last, rest) = types.split_last().unwrap();
            format!("{}, or {last}", rest.join(", "))
        }
    }
}

fn format_union_strings(types: &[String]) -> String {
    match types.len() {
        0 => String::new(),
        1 => types[0].clone(),
        2 => format!("{} or {}", types[0], types[1]),
        _ => {
            let (last, rest) = types.split_last().unwrap();
            format!(
                "{}, or {last}",
                rest.iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}
