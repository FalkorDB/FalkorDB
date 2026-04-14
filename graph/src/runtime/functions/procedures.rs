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

use super::{FnType, Functions, Type};
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

    // ── db.relationshiptypes ───────────────────────────────────────────
    cypher_fn!(funcs, "db.relationshiptypes",
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

    // ── db.propertykeys ────────────────────────────────────────────────
    cypher_fn!(funcs, "db.propertykeys",
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
                            for attr in &sorted_keys {
                                let field_list = &fields[attr];
                                let mut types = thin_vec![];
                                for field in field_list {
                                    match field.ty {
                                        IndexType::Range => {
                                            types.push(Value::String(Arc::new(String::from("RANGE"))));
                                        }
                                        IndexType::Fulltext => {
                                            types.push(Value::String(Arc::new(String::from("FULLTEXT"))));
                                        }
                                        IndexType::Vector => {
                                            types.push(Value::String(Arc::new(String::from("VECTOR"))));
                                        }
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
                                Value::String(Arc::new(String::from("NODE"))),
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
                            map.insert(Arc::new(String::from("info")), Value::Null);

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
}
