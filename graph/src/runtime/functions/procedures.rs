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

use super::{
    FnArguments, FnType, Functions, Type, empty_procedure_batch, get_functions, get_udf_functions,
    udf_version,
};
use crate::{
    index::indexer::{IndexInfo, IndexType},
    runtime::{
        batch::{Batch, Column},
        ordermap::OrderMap,
        runtime::Runtime,
        value::Value,
    },
};
use parking_lot::RwLock;
use std::sync::Arc;
use thin_vec::{ThinVec, thin_vec};

/// Cached dbms.functions() result columns, invalidated when the UDF registry
/// changes. Rebuilding the table formats a type-union string per argument of
/// every function — far more expensive than the Arc clones a cache hit costs.
static DBMS_FUNCTIONS_CACHE: RwLock<Option<(u64, [Vec<Value>; 8])>> = RwLock::new(None);

pub fn register(funcs: &mut Functions) {
    // ── db.labels ──────────────────────────────────────────────────────
    cypher_fn!(funcs, "db.labels",
        args: [],
        ret: Type::Any,
        procedure: ["label"],
        fn db_labels(runtime, _args) {
            let col_label: Vec<Value> = runtime
                .g
                .borrow()
                .get_labels()
                .iter()
                .map(|l| Value::String(l.clone()))
                .collect();
            Ok(Batch::from_columns([Column::Values(col_label)]))
        }
    );

    // ── db.relationshipTypes ───────────────────────────────────────────
    cypher_fn!(funcs, "db.relationshipTypes",
        args: [],
        ret: Type::Any,
        procedure: ["relationshipType"],
        fn db_types(runtime, _args) {
            let col_type: Vec<Value> = runtime
                .g
                .borrow()
                .get_types()
                .iter()
                .map(|t| Value::String(t.clone()))
                .collect();
            Ok(Batch::from_columns([Column::Values(col_type)]))
        }
    );

    // ── db.propertyKeys ────────────────────────────────────────────────
    cypher_fn!(funcs, "db.propertyKeys",
        args: [],
        ret: Type::Any,
        procedure: ["propertyKey"],
        fn db_properties(runtime, _args) {
            let col_key: Vec<Value> = runtime
                .g
                .borrow()
                .get_attrs()
                .map(|p| Value::String(p.clone()))
                .collect();
            Ok(Batch::from_columns([Column::Values(col_key)]))
        }
    );

    // ── db.indexes ─────────────────────────────────────────────────────
    cypher_fn!(funcs, "db.indexes",
        args: [],
        ret: Type::Any,
        procedure: ["label", "properties", "types", "options", "language", "stopwords", "entitytype", "status", "info"],
        fn db_indexes(runtime, _args) {
            let infos = runtime.g.borrow().index_info();
            let mut col_label = Vec::with_capacity(infos.len());
            let mut col_properties = Vec::with_capacity(infos.len());
            let mut col_types = Vec::with_capacity(infos.len());
            let mut col_options = Vec::with_capacity(infos.len());
            let mut col_language = Vec::with_capacity(infos.len());
            let mut col_stopwords = Vec::with_capacity(infos.len());
            let mut col_entitytype = Vec::with_capacity(infos.len());
            let mut col_status = Vec::with_capacity(infos.len());
            let mut col_info = Vec::with_capacity(infos.len());

            for IndexInfo {
                label,
                pending,
                progress,
                total,
                fields,
                field_order,
                language,
                stopwords,
                entity_type,
            } in infos {
                            // Build all 9 columns directly (columnar approach)
                            col_label.push(Value::String(label));
                            col_properties.push(Value::List(Arc::new(field_order.iter().map(|f| Value::String(f.clone())).collect())));

                            let mut types_map = OrderMap::default();
                            for attr in &field_order {
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
                            col_types.push(Value::Map(Arc::new(types_map)));
                            col_options.push(Value::Null);
                            col_language.push(language.map_or_else(|| Value::Null, Value::String));

                            let is_fulltext = field_order.iter().any(|attr| {
                                fields.get(attr).is_some_and(|fs| fs.iter().any(|f| f.ty == IndexType::Fulltext))
                            });
                            col_stopwords.push(stopwords.map_or_else(
                                || if is_fulltext { Value::List(Arc::new(thin_vec![])) } else { Value::Null },
                                |sw| Value::List(Arc::new(sw.into_iter().map(Value::String).collect())),
                            ));
                            col_entitytype.push(Value::String(Arc::new(entity_type)));
                            col_status.push(if pending > 0 {
                                Value::String(Arc::new(format!(
                                    "[Indexing] {progress}/{total}: UNDER CONSTRUCTION"
                                )))
                            } else {
                                Value::String(Arc::new(String::from("OPERATIONAL")))
                            });

                            // Build the `info.fields` list: one entry per
                            // underlying RediSearch field name, not per
                            // user-facing attribute. A single range attr
                            // expands into up to three RediSearch fields —
                            // the scalar (e.g. `range:a`) plus the numeric-
                            // and string-array companions used by list
                            // lookups (`a:numeric:arr`, `a:string:arr`) —
                            // and we surface all of them here for parity
                            // with the original FalkorDB implementation.
                            // Walk attributes in `field_order` (declaration
                            // order) so the output is deterministic across
                            // runs — tests and clients parsing this list
                            // can rely on stable ordering.
                            let mut rs_field_names = thin_vec![];
                            for attr_key in &field_order {
                                let Some(field_list) = fields.get(attr_key) else {
                                    continue;
                                };
                                for field in field_list {
                                    let name = field.name.to_str().unwrap_or("").to_string();
                                    if !name.is_empty() {
                                        rs_field_names.push(name);
                                    }
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
                            rs_field_names.push(String::from("NONE_INDEXABLE_FIELDS"));

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
                            col_info.push(Value::Map(Arc::new(info_map)));
            }

            Ok(Batch::from_columns([
                Column::Values(col_label),
                Column::Values(col_properties),
                Column::Values(col_types),
                Column::Values(col_options),
                Column::Values(col_language),
                Column::Values(col_stopwords),
                Column::Values(col_entitytype),
                Column::Values(col_status),
                Column::Values(col_info),
            ]))
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

            // Build 7 columns directly (columnar approach)
            // Maps stay as Column::Values; Ints are promoted to Column::Ints for efficiency
            Ok(Batch::from_columns([
                Column::Values(vec![Value::Map(Arc::new(labels_map))]),
                Column::Values(vec![Value::Map(Arc::new(rel_types_map))]),
                Column::Ints(vec![g.relationship_count() as i64]),
                Column::Ints(vec![g.node_count() as i64]),
                Column::Ints(vec![g.get_labels().len() as i64]),
                Column::Ints(vec![g.get_types().len() as i64]),
                Column::Ints(vec![g.property_key_count() as i64]),
            ]))
        }
    );

    // ── db.idx.fulltext.createNodeIndex ────────────────────────────────
    cypher_fn!(funcs, "db.idx.fulltext.createNodeIndex",
        args: [Type::Map],
        ret: Type::Any,
        write procedure: [],
        fn db_fulltext_create_node_index(_runtime, _args) {
            Ok(empty_procedure_batch())
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
            Err(String::from("db.idx.fulltext.drop() is not supported in this version"))
        }
    );

    // ── db.idx.vector.queryNodes ───────────────────────────────────────
    // Registered for binder-side arg-type checking and YIELD validation.
    // Execution is rewritten to `IR::NodeByVectorScan` in the planner,
    // so this body is never reached at runtime.
    //
    // Args: (label, attribute, k, vector). The binder rejects mismatches
    // with `"Invalid arguments for procedure ..."`, which the
    // `test_vecsim::test06_validate_arguments` negative cases pin to.
    cypher_fn!(funcs, "db.idx.vector.queryNodes",
        args: [Type::String, Type::String, Type::Int, Type::VecF32],
        ret: Type::Any,
        procedure: ["node", "score"],
        fn db_vector_query_nodes(_, _) {
            Err(String::from("db.idx.vector.queryNodes() is rewritten by the planner"))
        }
    );

    // ── db.idx.vector.queryRelationships ───────────────────────────────
    cypher_fn!(funcs, "db.idx.vector.queryRelationships",
        args: [Type::String, Type::String, Type::Int, Type::VecF32],
        ret: Type::Any,
        procedure: ["relationship", "score"],
        fn db_vector_query_relationships(_, _) {
            Err(String::from("db.idx.vector.queryRelationships() is rewritten by the planner"))
        }
    );

    // ── db.constraints ────────────────────────────────────────────────
    cypher_fn!(funcs, "db.constraints",
        args: [],
        ret: Type::Any,
        procedure: ["type", "label", "properties", "entitytype", "status"],
        fn db_constraints(runtime, _args) {
            let g = runtime.g.borrow();
            let constraints = g.constraints();
            let mut col_type = Vec::with_capacity(constraints.len());
            let mut col_label = Vec::with_capacity(constraints.len());
            let mut col_properties = Vec::with_capacity(constraints.len());
            let mut col_entitytype = Vec::with_capacity(constraints.len());
            let mut col_status = Vec::with_capacity(constraints.len());
            for c in constraints {
                // positional order: type, label, properties, entitytype, status
                col_type.push(Value::String(Arc::new(c.ct.to_string())));
                col_label.push(Value::String(c.label.clone()));
                col_properties.push(Value::List(Arc::new(
                    c.properties.iter().map(|p| Value::String(p.clone())).collect(),
                )));
                col_entitytype.push(Value::String(Arc::new(c.entity_type.to_string())));
                col_status.push(Value::String(Arc::new(c.status.to_string())));
            }

            Ok(Batch::from_columns([
                Column::Values(col_type),
                Column::Values(col_label),
                Column::Values(col_properties),
                Column::Values(col_entitytype),
                Column::Values(col_status),
            ]))
        }
    );

    // ── dbms.procedures ───────────────────────────────────────────────
    cypher_fn!(funcs, "dbms.procedures",
        args: [],
        ret: Type::Any,
        procedure: ["name", "mode"],
        fn dbms_procedures(_, _args) {
            let funcs = get_functions();
            // Collect references and sort by name up front so we never pay
            // for map lookups (and the Arc<String> allocations they used to
            // require) inside the sort comparator.
            let mut procs: Vec<&Arc<super::GraphFn>> = funcs
                .iter()
                .filter(|f| matches!(f.fn_type, FnType::Procedure(_)))
                .collect();
            procs.sort_by(|a, b| a.name.cmp(&b.name));

            let mut col_name = Vec::with_capacity(procs.len());
            let mut col_mode = Vec::with_capacity(procs.len());
            for f in procs {
                col_name.push(Value::String(Arc::new(f.name.clone())));
                col_mode.push(Value::String(Arc::new(String::from(
                    if f.write { "WRITE" } else { "READ" },
                ))));
            }

            Ok(Batch::from_columns([
                Column::Values(col_name),
                Column::Values(col_mode),
            ]))
        }
    );

    // ── dbms.functions ────────────────────────────────────────────────
    cypher_fn!(funcs, "dbms.functions",
        args: [],
        ret: Type::Any,
        procedure: ["name", "return_type", "arguments", "internal", "reducible", "aggregation", "variable_len", "udf"],
        fn dbms_functions(_, _args) {
            let version = udf_version();
            if let Some((v, cols)) = DBMS_FUNCTIONS_CACHE.read().as_ref()
                && *v == version
            {
                return Ok(Batch::from_columns(cols.clone().map(Column::Values)));
            }
            let funcs = get_functions();
            let mut seen_names = std::collections::HashSet::new();
            // Gather all function references (built-ins + UDFs), de-duplicated
            // by lowercase name, then sort by name once. Sorting the function
            // refs is far cheaper than building maps and sorting them while
            // re-allocating Arc<String> keys for every comparison.
            let mut entries: Vec<Arc<super::GraphFn>> = Vec::new();

            // Built-in functions first (non-procedure entries).
            for f in funcs.iter().filter(|f| !matches!(f.fn_type, FnType::Procedure(_))) {
                if seen_names.insert(f.name.to_lowercase()) {
                    entries.push(Arc::clone(f));
                }
            }

            // UDF entries from the dynamic registry.
            for f in get_udf_functions() {
                if seen_names.insert(f.name.to_lowercase()) {
                    entries.push(f);
                }
            }

            entries.sort_by(|a, b| a.name.cmp(&b.name));

            let mut col_name = Vec::with_capacity(entries.len());
            let mut col_return_type = Vec::with_capacity(entries.len());
            let mut col_arguments = Vec::with_capacity(entries.len());
            let mut col_internal = Vec::with_capacity(entries.len());
            let mut col_reducible = Vec::with_capacity(entries.len());
            let mut col_aggregation = Vec::with_capacity(entries.len());
            let mut col_variable_len = Vec::with_capacity(entries.len());
            let mut col_udf = Vec::with_capacity(entries.len());

            for f in &entries {
                col_name.push(Value::String(Arc::new(f.name.clone())));
                col_return_type.push(Value::String(Arc::new(type_to_dbms_string(&f.ret_type))));

                let args_list: thin_vec::ThinVec<Value> = match &f.args_type {
                    FnArguments::Fixed(types) => types
                        .iter()
                        .map(|t| Value::String(Arc::new(type_to_dbms_string(t))))
                        .collect(),
                    FnArguments::VarLength(t) => {
                        thin_vec::thin_vec![Value::String(Arc::new(type_to_dbms_string(t)))]
                    }
                };
                col_arguments.push(Value::List(Arc::new(args_list)));

                col_internal.push(Value::Bool(matches!(f.fn_type, FnType::Internal)));

                let reducible = !f.non_deterministic
                    && !matches!(f.fn_type, FnType::Aggregation { .. } | FnType::Procedure(_));
                col_reducible.push(Value::Bool(reducible));

                col_aggregation.push(Value::Bool(matches!(f.fn_type, FnType::Aggregation { .. })));
                col_variable_len.push(Value::Bool(matches!(f.args_type, FnArguments::VarLength(_))));
                col_udf.push(Value::Bool(matches!(f.fn_type, FnType::Udf)));
            }

            let cols = [
                col_name,
                col_return_type,
                col_arguments,
                col_internal,
                col_reducible,
                col_aggregation,
                col_variable_len,
                col_udf,
            ];
            *DBMS_FUNCTIONS_CACHE.write() = Some((version, cols.clone()));
            Ok(Batch::from_columns(cols.map(Column::Values)))
        }
    );
}

/// Convert a `Type` to a display string matching the original `FalkorDB` format.
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
                    .map(std::string::String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}
