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
//!  db.idx.cch.query(map)                    {pathWeight, path}              CCH shortest path
//! ```
//!
//! Read-only procedures are registered with `write = false`; the
//! full-text index creation procedure uses the `write procedure:`
//! macro arm so it can be used inside write queries.

#![allow(clippy::unnecessary_wraps)]

use super::{
    FnArguments, FnType, Functions, Type, empty_procedure_batch, get_functions, get_udf_functions,
};
use crate::{
    graph::graph::{Graph, NodeId, RelationshipId},
    index::{
        cch::{CchIndexDef, input::cheapest_edge},
        indexer::{IndexInfo, IndexType},
    },
    runtime::{
        batch::{Batch, Column},
        ordermap::OrderMap,
        runtime::Runtime,
        value::Value,
    },
};
use std::sync::{Arc, LazyLock};
use thin_vec::{ThinVec, thin_vec};

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
                                for field in &fields[attr] {
                                    let slot = match field.ty {
                                        IndexType::Range => 0,
                                        IndexType::Vector => 1,
                                        IndexType::Fulltext => 2,
                                    };
                                    seen[slot] = true;
                                }
                                let mut types = thin_vec![];
                                for (slot, name) in
                                    [(0, "RANGE"), (1, "VECTOR"), (2, "FULLTEXT")]
                                {
                                    if seen[slot] {
                                        types.push(Value::String(Arc::new(name.to_string())));
                                    }
                                }
                                types_map.insert(attr.clone(), Value::List(Arc::new(types)));
                            }
                            col_types.push(Value::Map(Arc::new(types_map)));

                            // Per-attribute options map: vector attributes
                            // surface their HNSW parameters, everything else
                            // gets an empty map (parity with FalkorDB C).
                            let mut options_map = OrderMap::default();
                            for attr in &field_order {
                                let mut attr_opts = OrderMap::default();
                                if let Some(vopts) = fields
                                    .get(attr)
                                    .and_then(|fs| fs.iter().find_map(|f| f.vector_options()))
                                {
                                    attr_opts.insert(
                                        Arc::new(String::from("dimension")),
                                        // `try_from`, not `as`. This was
                                        // `i64::from` on a `u32` until
                                        // `dimension` widened to the wire's
                                        // `u64`, and a cast reintroduced the
                                        // silent narrowing this work removes
                                        // elsewhere. A dimension always fits:
                                        // the parser takes it from a
                                        // non-negative `Value::Int`. One that
                                        // does not came from a corrupt source,
                                        // and saturating reports it as
                                        // implausible rather than negative.
                                        Value::Int(
                                            i64::try_from(vopts.dimension)
                                                .unwrap_or(i64::MAX),
                                        ),
                                    );
                                    attr_opts.insert(
                                        Arc::new(String::from("similarityFunction")),
                                        Value::String(Arc::new(
                                            vopts
                                                .similarity_function
                                                .clone()
                                                .unwrap_or_else(|| String::from("euclidean")),
                                        )),
                                    );
                                    attr_opts.insert(
                                        Arc::new(String::from("M")),
                                        Value::Int(vopts.m.unwrap_or(16) as i64),
                                    );
                                    attr_opts.insert(
                                        Arc::new(String::from("efConstruction")),
                                        Value::Int(vopts.ef_construction.unwrap_or(200) as i64),
                                    );
                                    attr_opts.insert(
                                        Arc::new(String::from("efRuntime")),
                                        Value::Int(vopts.ef_runtime.unwrap_or(10) as i64),
                                    );
                                }
                                options_map.insert(attr.clone(), Value::Map(Arc::new(attr_opts)));
                            }
                            col_options.push(Value::Map(Arc::new(options_map)));

                            // Language defaults to english and stopwords to an
                            // empty list (parity with FalkorDB C).
                            col_language.push(Value::String(
                                language.unwrap_or_else(|| Arc::new(String::from("english"))),
                            ));
                            col_stopwords.push(Value::List(Arc::new(stopwords.map_or_else(
                                ThinVec::new,
                                |sw| sw.into_iter().map(Value::String).collect(),
                            ))));
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

            // Graph-level CCH indexes come after the RediSearch-backed ones, like C.
            for row in cch_index_rows(&runtime.g.borrow()) {
                col_label.push(row.label);
                col_properties.push(row.properties);
                col_types.push(row.types);
                col_options.push(Value::Map(Arc::new(OrderMap::default())));
                col_language.push(Value::String(Arc::new(String::new())));
                col_stopwords.push(Value::List(Arc::new(thin_vec![])));
                col_entitytype.push(Value::String(Arc::new(String::from("RELATIONSHIP"))));
                col_status.push(Value::String(Arc::new(String::from("OPERATIONAL"))));
                col_info.push(Value::Map(Arc::new(OrderMap::default())));
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

    // ── db.idx.cch.query ────────────────────────────────────────────────
    // CALL db.idx.cch.query({sourceNode, targetNode, relTypes, weightProp})
    //      YIELD pathWeight, path
    //
    // Read-only (usable in GRAPH.RO_QUERY): the hierarchy is a side index, not
    // a graph mutation. Argument validation mirrors C's `_read_config`
    // (proc_cch_idx_query.c) error-for-error; then the index for exactly that
    // definition answers the query (`cch_query_rows`).
    cypher_fn!(funcs, "db.idx.cch.query",
        args: [Type::Map],
        ret: Type::Any,
        procedure: ["pathWeight", "path"],
        fn db_idx_cch_query(runtime, args) {
            let config = parse_cch_query_config(&runtime.g.borrow(), args)?;
            // Inside a write query, the index must reflect that query's earlier writes.
            let def = cch_query_def(&runtime.g.borrow(), &config);
            if let Some(def) = def {
                runtime.flush_cch_before_reading(&def)?;
            }
            let g = runtime.g.borrow();
            let rows = cch_query_rows(&g, &config)?;
            let (weights, paths): (Vec<Value>, Vec<Value>) = rows.into_iter().unzip();
            Ok(Batch::from_columns([
                Column::Values(weights),
                Column::Values(paths),
            ]))
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
            // Built-in rows are precomputed and only ever cloned. UDF rows are
            // built fresh, so the per-call cost scales with the number of UDFs
            // (typically a handful) rather than the ~110 built-ins. Reading the
            // registry on every call also means a newly registered UDF is
            // visible immediately — no cache, no version, no staleness window.
            let mut udfs: Vec<Arc<super::GraphFn>> = get_udf_functions()
                .into_iter()
                .filter(|f| {
                    // Built-ins take precedence on a lowercase-name collision.
                    BUILTIN_LOWER_NAMES
                        .binary_search(&f.name.to_lowercase())
                        .is_err()
                })
                .collect();

            if udfs.is_empty() {
                return Ok(Batch::from_columns(BUILTIN_COLUMNS.clone().map(Column::Values)));
            }

            udfs.sort_by(|a, b| a.name.cmp(&b.name));
            let udf_cols = build_columns(&udfs);

            // Both sides are sorted by name; merge them so the output keeps the
            // global name order without re-sorting or rebuilding built-in rows.
            let (n, m) = (BUILTIN_ENTRIES.len(), udfs.len());
            let mut picks: Vec<(bool, usize)> = Vec::with_capacity(n + m);
            let (mut i, mut j) = (0usize, 0usize);
            while i < n || j < m {
                let take_udf = i >= n
                    || (j < m && udfs[j].name < BUILTIN_ENTRIES[i].name);
                if take_udf {
                    picks.push((true, j));
                    j += 1;
                } else {
                    picks.push((false, i));
                    i += 1;
                }
            }

            let cols: [Vec<Value>; 8] = std::array::from_fn(|c| {
                picks
                    .iter()
                    .map(|&(is_udf, idx)| {
                        if is_udf { udf_cols[c][idx].clone() } else { BUILTIN_COLUMNS[c][idx].clone() }
                    })
                    .collect()
            });
            Ok(Batch::from_columns(cols.map(Column::Values)))
        }
    );
}

// ── db.idx.cch.query helpers ────────────────────────────────────────────
//
// Split out from `register` so the argument-parsing/validation can be unit
// tested directly against a hand-built `Graph`, without needing a `Runtime`.

/// Validated `db.idx.cch.query({sourceNode, targetNode, relTypes, weightProp})`
/// config. `rel_types` and `weight_prop` are checked to exist in the graph's
/// schema/attribute tables; `cch_query_rows` resolves them to a `CchIndexDef`.
#[derive(Debug)]
struct CchQueryConfig {
    source: NodeId,
    target: NodeId,
    rel_types: Vec<Arc<String>>,
    weight_prop: Arc<String>,
}

/// Parses and validates the single map argument to `db.idx.cch.query`,
/// matching C's `_read_config` (`proc_cch_idx_query.c`) error for error:
/// sourceNode/targetNode presence and type, relTypes shape/non-emptiness and
/// existence, weightProp type and existence.
fn parse_cch_query_config(
    g: &Graph,
    args: &[Value],
) -> Result<CchQueryConfig, String> {
    let Some(Value::Map(map)) = args.first() else {
        return Err(String::from(
            "db.idx.cch.query expects a single map argument",
        ));
    };

    let source_val = map.get_str("sourceNode");
    let target_val = map.get_str("targetNode");
    let source_present = matches!(source_val, Some(v) if !matches!(v, Value::Null));
    let target_present = matches!(target_val, Some(v) if !matches!(v, Value::Null));
    if !source_present || !target_present {
        return Err(String::from(
            "db.idx.cch.query requires sourceNode and targetNode",
        ));
    }
    let (source, target) = match (source_val.unwrap(), target_val.unwrap()) {
        (Value::Node(s), Value::Node(t)) => (*s, *t),
        _ => {
            return Err(String::from(
                "db.idx.cch.query, sourceNode/targetNode must be nodes",
            ));
        }
    };

    let rel_types = match map.get_str("relTypes") {
        Some(Value::List(list)) => {
            let mut types = Vec::with_capacity(list.len());
            for v in list.iter() {
                match v {
                    Value::String(s) => types.push(s.clone()),
                    _ => {
                        return Err(String::from(
                            "db.idx.cch.query, relTypes must be an array of strings",
                        ));
                    }
                }
            }
            types
        }
        _ => {
            return Err(String::from(
                "db.idx.cch.query, relTypes must be an array of strings",
            ));
        }
    };
    if rel_types.is_empty() {
        return Err(String::from("db.idx.cch.query, relTypes must not be empty"));
    }
    for t in &rel_types {
        if g.get_relationship_matrix(t).is_none() {
            return Err(format!("db.idx.cch.query, unknown relationship type: {t}"));
        }
    }

    let weight_prop = match map.get_str("weightProp") {
        Some(Value::String(s)) => s.clone(),
        _ => return Err(String::from("db.idx.cch.query requires string weightProp")),
    };
    if g.get_relationship_attribute_id(&weight_prop).is_none() {
        return Err(format!(
            "db.idx.cch.query, unknown attribute: {weight_prop}"
        ));
    }

    Ok(CchQueryConfig {
        source,
        target,
        rel_types,
        weight_prop,
    })
}

/// Resolves a CCH hierarchy path -- already-unpacked repo `NodeId`s, in
/// order -- into a `Value::Path`, choosing the cheapest edge among
/// `rel_types` at each hop by `weight_attr` (a graph-wide property-key id,
/// as returned by `Graph::get_relationship_attribute_id`). A
/// missing/non-numeric attribute defaults to weight 1.0, the same rule
/// `CchInput::weights` uses. Mirrors C's `_best_road_edge`
/// (`proc_cch_idx_query.c`).
///
/// Each hop is a direct `(src, dst)` lookup per indexed type
/// ([`cheapest_edge`]), not a scan of `from`'s outgoing edges: the cost is
/// the pair's own edges. Ties go to the earlier type in `rel_types`, then to
/// the lowest edge id.
///
/// Unpacking rank space back into repo `NodeId`s is `Cch::query` plus the
/// index's node permutation, not this helper's job. This only
/// turns an already-resolved node sequence into the `path` yield.
///
/// # Errors
/// If a consecutive pair in `node_ids` has no realizing edge among
/// `rel_types`: the hierarchy and the live graph have gone out of sync.
pub(crate) fn cch_path_from_node_ids(
    g: &Graph,
    node_ids: &[NodeId],
    rel_types: &[Arc<String>],
    weight_attr: u16,
) -> Result<Value, String> {
    let Some(&first) = node_ids.first() else {
        return Err(String::from("db.idx.cch.query: empty hierarchy path"));
    };

    let mut elems: ThinVec<Value> = ThinVec::with_capacity(node_ids.len() * 2);
    elems.push(Value::Node(first));

    // Resolved once; each hop is then one per-pair lookup per type.
    let tensors: Vec<_> = rel_types
        .iter()
        .filter_map(|t| g.get_relationship_matrix(t))
        .collect();
    let attrs = g.relationship_attrs();
    for pair in node_ids.windows(2) {
        let (from, to) = (pair[0], pair[1]);
        let best = cheapest_edge(
            tensors.iter().copied(),
            attrs,
            weight_attr,
            u64::from(from),
            u64::from(to),
        );
        let Some((eid, _)) = best else {
            return Err(format!(
                "db.idx.cch.query: hierarchy inconsistent with the graph, no edge among the \
                 indexed types from node {} to node {}",
                u64::from(from),
                u64::from(to)
            ));
        };
        elems.push(Value::Relationship(RelationshipId::from(eid)));
        elems.push(Value::Node(to));
    }

    Ok(Value::Path(Arc::new(elems)))
}

/// Non-procedure built-in functions, sorted by name.
///
/// `Functions` is keyed by lowercase name, so built-ins are already unique —
/// the de-duplication in `dbms.functions()` exists only to drop a UDF that
/// collides with a built-in. Built-ins are registered once during
/// `init_functions` and never change, so both this list and the columns
/// derived from it are computed on first use and then only cloned. That is
/// why the procedure needs no result cache: the immutable part is immutable
/// by construction, with no version to compare and no staleness window.
static BUILTIN_ENTRIES: LazyLock<Vec<Arc<super::GraphFn>>> = LazyLock::new(|| {
    let mut entries: Vec<Arc<super::GraphFn>> = get_functions()
        .iter()
        .filter(|f| !matches!(f.fn_type, FnType::Procedure(_)))
        .map(Arc::clone)
        .collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
});

/// Lowercase built-in names, sorted, for the UDF shadow check. Lets each UDF
/// cost one binary search instead of re-lowercasing every built-in per call.
static BUILTIN_LOWER_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    let mut names: Vec<String> = BUILTIN_ENTRIES
        .iter()
        .map(|f| f.name.to_lowercase())
        .collect();
    names.sort_unstable();
    names
});

/// The eight `dbms.functions()` columns for [`BUILTIN_ENTRIES`].
static BUILTIN_COLUMNS: LazyLock<[Vec<Value>; 8]> =
    LazyLock::new(|| build_columns(&BUILTIN_ENTRIES));

/// Builds the eight `dbms.functions()` columns for `entries`.
fn build_columns(entries: &[Arc<super::GraphFn>]) -> [Vec<Value>; 8] {
    let n = entries.len();
    let mut col_name = Vec::with_capacity(n);
    let mut col_return_type = Vec::with_capacity(n);
    let mut col_arguments = Vec::with_capacity(n);
    let mut col_internal = Vec::with_capacity(n);
    let mut col_reducible = Vec::with_capacity(n);
    let mut col_aggregation = Vec::with_capacity(n);
    let mut col_variable_len = Vec::with_capacity(n);
    let mut col_udf = Vec::with_capacity(n);

    for f in entries {
        col_name.push(Value::String(Arc::new(f.name.clone())));
        col_return_type.push(type_to_dbms_value(&f.ret_type));

        let args_list: ThinVec<Value> = match &f.args_type {
            FnArguments::Fixed(types) => types.iter().map(type_to_dbms_value).collect(),
            FnArguments::VarLength(t) => thin_vec![type_to_dbms_value(t)],
        };
        col_arguments.push(Value::List(Arc::new(args_list)));

        col_internal.push(Value::Bool(matches!(f.fn_type, FnType::Internal)));
        col_reducible.push(Value::Bool(
            !f.non_deterministic
                && !matches!(f.fn_type, FnType::Aggregation { .. } | FnType::Procedure(_)),
        ));
        col_aggregation.push(Value::Bool(matches!(f.fn_type, FnType::Aggregation { .. })));
        col_variable_len.push(Value::Bool(matches!(
            f.args_type,
            FnArguments::VarLength(_)
        )));
        col_udf.push(Value::Bool(matches!(f.fn_type, FnType::Udf)));
    }

    [
        col_name,
        col_return_type,
        col_arguments,
        col_internal,
        col_reducible,
        col_aggregation,
        col_variable_len,
        col_udf,
    ]
}

/// Convert a `Type` to a display string matching the original `FalkorDB` format.
/// The `Type::Any` spelling: a 17-name union, ~130 bytes. Built once — most
/// functions accept and return `Any`, so `dbms.functions()` would otherwise
/// rebuild this string a few hundred times per call.
static ANY_UNION: LazyLock<Arc<String>> =
    LazyLock::new(|| Arc::new(type_to_dbms_string(&Type::Any)));

/// A type's `dbms.functions()` display string, as a ready-to-clone [`Value`].
///
/// Every spelling except the composite `Union` case is fixed at compile time,
/// so it is interned and handed out as an `Arc` refcount bump instead of a
/// fresh allocation. This is what makes rebuilding the procedure table cheap
/// enough not to need a result cache.
fn type_to_dbms_value(t: &Type) -> Value {
    /// Interns one fixed spelling in its own `static`.
    macro_rules! interned {
        ($name:literal) => {{
            static S: LazyLock<Arc<String>> = LazyLock::new(|| Arc::new(String::from($name)));
            Value::String(Arc::clone(&S))
        }};
    }
    match t {
        Type::Any => Value::String(Arc::clone(&ANY_UNION)),
        Type::Null => interned!("Null"),
        Type::Bool => interned!("Boolean"),
        Type::Int => interned!("Integer"),
        Type::Float => interned!("Float"),
        Type::String => interned!("String"),
        Type::List(_) => interned!("List"),
        Type::Map => interned!("Map"),
        Type::Node => interned!("Node"),
        Type::Relationship => interned!("Edge"),
        Type::Path => interned!("Path"),
        Type::VecF32 => interned!("Vectorf32"),
        Type::Point => interned!("Point"),
        Type::Datetime => interned!("Datetime"),
        Type::Date => interned!("Date"),
        Type::Time => interned!("Time"),
        Type::Duration => interned!("Duration"),
        // Composite: the spelling depends on the members, so it is built.
        // Rare — no registered function uses it in a hot column.
        Type::Union(_) => Value::String(Arc::new(type_to_dbms_string(t))),
        Type::Optional(inner) => type_to_dbms_value(inner),
    }
}

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

/// One `db.indexes()` row for a CCH index, shaped like C's `_EmitCCHIndex`: the relationship types
/// joined by `,` as the label, the weight attribute as the only property, `{weight: ['CCH']}` as
/// the types. The RediSearch-specific columns are filled by the caller.
struct CchIndexRow {
    label: Value,
    properties: Value,
    types: Value,
}

fn cch_index_rows(g: &Graph) -> Vec<CchIndexRow> {
    let type_names = g.get_types();
    let attr_names = g.get_relationship_attribute_names();
    g.cch_indexes()
        .filter_map(|def| {
            let rels: Option<Vec<&str>> = def
                .rel_types()
                .iter()
                .map(|t| type_names.get(t.0).map(|n| n.as_str()))
                .collect();
            let weight = attr_names.get(usize::from(def.weight_attr))?.clone();
            let mut types = OrderMap::default();
            types.insert(
                weight.clone(),
                Value::List(Arc::new(thin_vec![Value::String(Arc::new(String::from(
                    "CCH"
                )))])),
            );
            Some(CchIndexRow {
                label: Value::String(Arc::new(rels?.join(","))),
                properties: Value::List(Arc::new(thin_vec![Value::String(weight)])),
                types: Value::Map(Arc::new(types)),
            })
        })
        .collect()
}

/// Looks up the index named by `config` and answers the query: no row when the nodes are
/// unconnected, one `(pathWeight, path)` row otherwise. Mirrors C's `Proc_CCHIdxQueryInvoke`.
///
/// The hierarchy is the one this graph version holds. Inside a write query the procedure
/// flushes that query's changes to it first (`Runtime::flush_cch_before_reading`).
fn cch_query_rows(
    g: &Graph,
    config: &CchQueryConfig,
) -> Result<Vec<(Value, Value)>, String> {
    let index = cch_query_def(g, config).and_then(|def| g.cch_index(&def));
    let Some(index) = index else {
        return Err(String::from(
            "db.idx.cch.query: no CCH index over these relationship types and weight \
             attribute; create one with CREATE CCH INDEX FOR ()-[e:<types>]->() ON (e.<weight>)",
        ));
    };

    // Node ids beyond u32 cannot be in the hierarchy: no path, like C's out-of-range check.
    let (Ok(src), Ok(dst)) = (
        u32::try_from(u64::from(config.source)),
        u32::try_from(u64::from(config.target)),
    ) else {
        return Ok(vec![]);
    };
    let Some((weight, nodes)) = index.cch.query(src, dst) else {
        return Ok(vec![]);
    };
    let nodes: Vec<NodeId> = nodes
        .into_iter()
        .map(|n| NodeId::from(u64::from(n)))
        .collect();
    let path = cch_path_from_node_ids(g, &nodes, &config.rel_types, attr_index(g, config)?)?;
    Ok(vec![(Value::Float(weight), path)])
}

/// The index definition `config` names, if its types and weight attribute all exist.
fn cch_query_def(
    g: &Graph,
    config: &CchQueryConfig,
) -> Option<CchIndexDef> {
    let types: Option<Vec<_>> = config.rel_types.iter().map(|t| g.get_type_id(t)).collect();
    let attr = g
        .get_relationship_attribute_id(&config.weight_prop)
        .and_then(|a| u16::try_from(a).ok());
    types
        .zip(attr)
        .map(|(types, attr)| CchIndexDef::new(types, attr))
}

fn attr_index(
    g: &Graph,
    config: &CchQueryConfig,
) -> Result<u16, String> {
    g.get_relationship_attribute_id(&config.weight_prop)
        .and_then(|a| u16::try_from(a).ok())
        .ok_or_else(|| {
            format!(
                "db.idx.cch.query, unknown attribute: {}",
                config.weight_prop
            )
        })
}

#[cfg(test)]
mod cch_query_tests {
    use super::{
        CchIndexDef, CchQueryConfig, NodeId, RelationshipId, cch_index_rows,
        cch_path_from_node_ids, cch_query_rows, parse_cch_query_config,
    };
    use crate::graph::{graph::Graph, graphblas::test_init};
    use crate::index::index_docs::IndexDocs;
    use crate::runtime::ordermap::OrderMap;
    use crate::runtime::value::Value;
    use roaring::RoaringTreemap;
    use rustc_hash::FxHashMap;
    use std::sync::Arc;
    use thin_vec::thin_vec;

    fn s(v: &str) -> Arc<String> {
        Arc::new(String::from(v))
    }

    /// Reserve `n` node ids the way one query's `CreateOp` does: open the
    /// graph's batch, then ask the space for ids this query does not hold yet.
    fn reserve_nodes(
        g: &mut Graph,
        n: usize,
    ) -> Vec<u64> {
        g.roll_id_batches().expect("a consistent space");
        g.node_id_space()
            .reserve(n, &RoaringTreemap::new())
            .expect("reserved")
    }

    fn reserve_relationships(
        g: &Graph,
        n: usize,
    ) -> Vec<u64> {
        g.relationship_id_space()
            .reserve(n, &RoaringTreemap::new())
            .expect("reserved")
    }

    fn map_arg(pairs: Vec<(&str, Value)>) -> Vec<Value> {
        let mut m = OrderMap::default();
        for (k, v) in pairs {
            m.insert(s(k), v);
        }
        vec![Value::Map(Arc::new(m))]
    }

    /// A graph with a `ROAD` schema (and `t` attribute) but no `w` attribute,
    /// matching test_cch.py's test03_query_validation fixture.
    fn fixture_graph() -> Graph {
        test_init::ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, "cch_query_validation_test");
        let node_ids = reserve_nodes(&mut g, 2);
        g.create_nodes(&node_ids.iter().copied().collect()).unwrap();

        g.get_type_id_mut("ROAD");
        let rel_ids = reserve_relationships(&g, 1);
        g.create_relationships_bulk(&s("ROAD"), &[node_ids[0]], &[node_ids[1]], &[rel_ids[0]])
            .unwrap();
        let t_attr = g.get_or_create_rel_attr_id(&s("t"));
        let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        attrs.insert(rel_ids[0], vec![(t_attr, Value::Int(2))]);
        let mut index_add_edge_docs = IndexDocs::default();
        g.set_relationships_attributes(&attrs, &mut index_add_edge_docs)
            .unwrap();
        g
    }

    #[test]
    fn requires_source_and_target() {
        let g = fixture_graph();
        let args = map_arg(vec![("sourceNode", Value::Node(0u64.into()))]);
        let err = parse_cch_query_config(&g, &args).unwrap_err();
        assert!(err.contains("sourceNode and targetNode"), "{err}");
    }

    #[test]
    fn source_and_target_must_be_nodes() {
        let g = fixture_graph();
        let args = map_arg(vec![
            ("sourceNode", Value::Int(1)),
            ("targetNode", Value::Node(1u64.into())),
            (
                "relTypes",
                Value::List(Arc::new(thin_vec![Value::String(s("ROAD"))])),
            ),
            ("weightProp", Value::String(s("w"))),
        ]);
        let err = parse_cch_query_config(&g, &args).unwrap_err();
        assert!(err.contains("must be nodes"), "{err}");
    }

    #[test]
    fn rejects_unknown_relationship_type() {
        let g = fixture_graph();
        let args = map_arg(vec![
            ("sourceNode", Value::Node(0u64.into())),
            ("targetNode", Value::Node(1u64.into())),
            (
                "relTypes",
                Value::List(Arc::new(thin_vec![Value::String(s("NOPE"))])),
            ),
            ("weightProp", Value::String(s("w"))),
        ]);
        let err = parse_cch_query_config(&g, &args).unwrap_err();
        assert!(err.contains("unknown relationship type"), "{err}");
    }

    #[test]
    fn rejects_empty_rel_types() {
        let g = fixture_graph();
        let args = map_arg(vec![
            ("sourceNode", Value::Node(0u64.into())),
            ("targetNode", Value::Node(1u64.into())),
            ("relTypes", Value::List(Arc::new(thin_vec![]))),
            ("weightProp", Value::String(s("w"))),
        ]);
        let err = parse_cch_query_config(&g, &args).unwrap_err();
        assert!(err.contains("must not be empty"), "{err}");
    }

    #[test]
    fn rejects_unknown_attribute() {
        let g = fixture_graph();
        let args = map_arg(vec![
            ("sourceNode", Value::Node(0u64.into())),
            ("targetNode", Value::Node(1u64.into())),
            (
                "relTypes",
                Value::List(Arc::new(thin_vec![Value::String(s("ROAD"))])),
            ),
            ("weightProp", Value::String(s("nope"))),
        ]);
        let err = parse_cch_query_config(&g, &args).unwrap_err();
        assert!(err.contains("unknown attribute"), "{err}");
    }

    #[test]
    fn accepts_a_real_attribute_with_no_cch_index() {
        // 't' is a real attribute but no CCH index is built over it -- that
        // is the index lookup's error, so this must get past argument
        // validation cleanly.
        let g = fixture_graph();
        let args = map_arg(vec![
            ("sourceNode", Value::Node(0u64.into())),
            ("targetNode", Value::Node(1u64.into())),
            (
                "relTypes",
                Value::List(Arc::new(thin_vec![Value::String(s("ROAD"))])),
            ),
            ("weightProp", Value::String(s("t"))),
        ]);
        let CchQueryConfig {
            source,
            target,
            rel_types,
            weight_prop,
        } = parse_cch_query_config(&g, &args).unwrap();
        assert_eq!(source, 0u64.into());
        assert_eq!(target, 1u64.into());
        assert_eq!(rel_types, vec![s("ROAD")]);
        assert_eq!(weight_prop, s("t"));
    }

    #[test]
    fn rejects_non_map_argument() {
        let g = fixture_graph();
        let err = parse_cch_query_config(&g, &[Value::Int(1)]).unwrap_err();
        assert!(err.contains("single map argument"), "{err}");
    }

    // ---- cch_path_from_node_ids --------------------------------------------

    /// Builds n0 -[ROAD w=10]-> n1, n0 -[FERRY w=2]-> n1 (parallel, cheaper),
    /// n1 -[ROAD]-> n2 (no `w` set, defaults to 1.0).
    fn path_fixture() -> (Graph, Vec<NodeId>, u16, Vec<RelationshipId>) {
        test_init::ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, "cch_path_helper_test");
        let node_ids = reserve_nodes(&mut g, 3);
        g.create_nodes(&node_ids.iter().copied().collect()).unwrap();

        g.get_type_id_mut("ROAD");
        g.get_type_id_mut("FERRY");

        let rel_ids = reserve_relationships(&g, 3);
        g.create_relationships_bulk(&s("ROAD"), &[node_ids[0]], &[node_ids[1]], &[rel_ids[0]])
            .unwrap();
        g.create_relationships_bulk(&s("FERRY"), &[node_ids[0]], &[node_ids[1]], &[rel_ids[1]])
            .unwrap();
        g.create_relationships_bulk(&s("ROAD"), &[node_ids[1]], &[node_ids[2]], &[rel_ids[2]])
            .unwrap();

        let w_attr = g.get_or_create_rel_attr_id(&s("w"));
        let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        attrs.insert(rel_ids[0], vec![(w_attr, Value::Float(10.0))]);
        attrs.insert(rel_ids[1], vec![(w_attr, Value::Float(2.0))]);
        // rel_ids[2] gets no `w` attribute at all -> defaults to 1.0.
        let mut index_add_edge_docs = IndexDocs::default();
        g.set_relationships_attributes(&attrs, &mut index_add_edge_docs)
            .unwrap();

        let node_ids = node_ids.into_iter().map(NodeId::from).collect();
        let rel_ids = rel_ids.into_iter().map(RelationshipId::from).collect();
        (g, node_ids, w_attr, rel_ids)
    }

    #[test]
    fn picks_the_cheapest_parallel_edge_and_defaults_missing_weight() {
        let (g, node_ids, w_attr, rel_ids) = path_fixture();
        let rel_types = vec![s("ROAD"), s("FERRY")];

        let path = cch_path_from_node_ids(&g, &node_ids, &rel_types, w_attr).unwrap();
        let Value::Path(elems) = path else {
            panic!("expected a Path")
        };

        assert_eq!(elems.len(), 5, "Node, Rel, Node, Rel, Node");
        assert_eq!(elems[0], Value::Node(node_ids[0]));
        // the cheaper FERRY edge wins over the pricier parallel ROAD edge
        assert_eq!(elems[1], Value::Relationship(rel_ids[1]));
        assert_eq!(elems[2], Value::Node(node_ids[1]));
        // only option for this hop; no `w` attribute set -> still realized
        assert_eq!(elems[3], Value::Relationship(rel_ids[2]));
        assert_eq!(elems[4], Value::Node(node_ids[2]));
    }

    /// A tie goes to the earlier type in `rel_types`: the first hop's two
    /// parallel edges both weigh 2.0 once ROAD's is lowered, so the order of
    /// the types picks the edge, as the outgoing-edge scan did.
    #[test]
    fn a_tie_goes_to_the_earlier_type() {
        let (mut g, node_ids, w_attr, rel_ids) = path_fixture();
        let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        attrs.insert(u64::from(rel_ids[0]), vec![(w_attr, Value::Int(2))]);
        let mut docs = IndexDocs::default();
        g.set_relationships_attributes(&attrs, &mut docs).unwrap();

        for (types, want) in [
            (vec![s("ROAD"), s("FERRY")], rel_ids[0]),
            (vec![s("FERRY"), s("ROAD")], rel_ids[1]),
        ] {
            let Value::Path(elems) =
                cch_path_from_node_ids(&g, &node_ids[..2], &types, w_attr).unwrap()
            else {
                panic!("expected a Path")
            };
            assert_eq!(elems[1], Value::Relationship(want), "{types:?}");
        }
    }

    #[test]
    fn single_node_path_is_just_that_node() {
        let (g, node_ids, w_attr, _rel_ids) = path_fixture();
        let rel_types = vec![s("ROAD"), s("FERRY")];

        let path = cch_path_from_node_ids(&g, &node_ids[..1], &rel_types, w_attr).unwrap();
        let Value::Path(elems) = path else {
            panic!("expected a Path")
        };
        assert_eq!(elems.len(), 1);
        assert_eq!(elems[0], Value::Node(node_ids[0]));
    }

    #[test]
    fn errors_when_no_realizing_edge_exists() {
        let (g, node_ids, w_attr, _rel_ids) = path_fixture();
        // no edge at all from n2 back to n0
        let rel_types = vec![s("ROAD"), s("FERRY")];
        let err = cch_path_from_node_ids(&g, &[node_ids[2], node_ids[0]], &rel_types, w_attr)
            .unwrap_err();
        assert!(err.contains("hierarchy inconsistent"), "{err}");
    }

    // ---- the real index: create, list, query -----------------------------------

    fn indexed_fixture() -> (Graph, Vec<NodeId>, Vec<RelationshipId>) {
        let (mut g, nodes, w_attr, rels) = path_fixture();
        let types = vec![
            g.get_type_id("ROAD").unwrap(),
            g.get_type_id("FERRY").unwrap(),
        ];
        g.create_cch_index(CchIndexDef::new(types, w_attr))
            .expect("build");
        (g, nodes, rels)
    }

    fn config(
        g: &Graph,
        src: NodeId,
        dst: NodeId,
        types: &[&str],
        w: &str,
    ) -> Result<CchQueryConfig, String> {
        let args = map_arg(vec![
            ("sourceNode", Value::Node(src)),
            ("targetNode", Value::Node(dst)),
            (
                "relTypes",
                Value::List(Arc::new(
                    types.iter().map(|t| Value::String(s(t))).collect(),
                )),
            ),
            ("weightProp", Value::String(s(w))),
        ]);
        parse_cch_query_config(g, &args)
    }

    #[test]
    fn query_returns_the_cheapest_path_through_the_real_index() {
        let (g, n, rels) = indexed_fixture();
        let cfg = config(&g, n[0], n[2], &["ROAD", "FERRY"], "w").unwrap();
        let rows = cch_query_rows(&g, &cfg).unwrap();
        assert_eq!(rows.len(), 1);
        // FERRY (2) beats the parallel ROAD (10); the second hop defaults to 1
        assert_eq!(rows[0].0, Value::Float(3.0));
        let Value::Path(p) = &rows[0].1 else {
            panic!("path")
        };
        assert_eq!(p[1], Value::Relationship(rels[1]));
        assert_eq!(p[3], Value::Relationship(rels[2]));
    }

    #[test]
    fn query_same_node_is_weight_zero_and_unreachable_is_no_row() {
        let (g, n, _) = indexed_fixture();
        let same = cch_query_rows(
            &g,
            &config(&g, n[1], n[1], &["ROAD", "FERRY"], "w").unwrap(),
        )
        .unwrap();
        assert_eq!(same[0].0, Value::Float(0.0));
        let back = cch_query_rows(
            &g,
            &config(&g, n[2], n[0], &["ROAD", "FERRY"], "w").unwrap(),
        )
        .unwrap();
        assert!(back.is_empty(), "no path n2 -> n0");
    }

    #[test]
    fn query_without_a_matching_index_says_so() {
        let (g, n, _) = indexed_fixture();
        // a different relationship set is a different index
        let cfg = config(&g, n[0], n[1], &["ROAD"], "w").unwrap();
        let err = cch_query_rows(&g, &cfg).unwrap_err();
        assert!(err.contains("no CCH index"), "{err}");
    }

    #[test]
    fn db_indexes_lists_the_cch_index_like_c() {
        let (g, _, _) = indexed_fixture();
        let rows = cch_index_rows(&g);
        assert_eq!(rows.len(), 1);
        // rel types sorted by id (ROAD was registered first), joined by ','
        assert_eq!(rows[0].label, Value::String(s("ROAD,FERRY")));
        assert_eq!(
            rows[0].properties,
            Value::List(Arc::new(thin_vec![Value::String(s("w"))]))
        );
        let Value::Map(m) = &rows[0].types else {
            panic!("types")
        };
        assert_eq!(
            m.get_str("w"),
            Some(&Value::List(Arc::new(thin_vec![Value::String(s("CCH"))])))
        );
    }

    #[test]
    fn error_messages_use_cs_wording() {
        use crate::{index::cch::CchError, runtime::runtime::cch_error_message};
        assert!(cch_error_message(&CchError::AlreadyExists).contains("already exists"));
        assert!(cch_error_message(&CchError::NotFound).contains("no CCH index"));
        assert_eq!(
            cch_error_message(&CchError::Internal("boom")),
            "CCH index error: boom"
        );
    }
}
