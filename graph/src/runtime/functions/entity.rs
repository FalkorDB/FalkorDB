//! Graph-entity inspection functions.
//!
//! Functions that inspect nodes, relationships, and paths -- the structural
//! elements of the property graph.
//!
//! ```text
//!  Cypher              Function        Returns
//! ──────────────────────────────────────────────────
//!  labels(n)           labels()        [String]
//!  typeOf(x)           type_of()       String
//!  hasLabels(n, [...]) has_labels()    Bool
//!  id(n)               id()            Int
//!  properties(n)       properties()    Map
//!  startNode(r)        start_node()    Node
//!  endNode(r)          end_node()      Node
//!  length(p)           length()        Int (edges in path)
//!  type(r)             relationship_type() String
//!  keys(n)             keys()          [String]
//!  exists(x)           exists()        Bool
//! ```
//!
//! Every function receives a `ThinVec<Value>` of already-validated
//! arguments and returns `Result<Value, String>`.  A `&Runtime` is
//! provided for graph access (e.g. resolving node labels from the
//! label matrices).

#![allow(clippy::unnecessary_wraps)]

use super::{FnType, Functions, Type};
use crate::graph::graph::NodeId;
use crate::runtime::value::ValueGetType;
use crate::runtime::{runtime::Runtime, value::Value};
use std::sync::Arc;
use thin_vec::ThinVec;

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "labels",
        args: [Type::Union(vec![Type::Node, Type::Null])],
        ret: Type::Union(vec![Type::List(Box::new(Type::String)), Type::Null]),
        fn labels(runtime, args) {
            match args.into_iter().next() {
                Some(Value::Node(id)) => {
                    let labels = runtime.get_node_labels(id);
                    Ok(Value::List(Arc::new(labels.into_iter().map(Value::String).collect())))
                }
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "typeOf",
        args: [Type::Any],
        ret: Type::String,
        fn type_of(_runtime, args) {
            let type_name = match args.into_iter().next() {
                Some(Value::Null) => "Null",
                Some(Value::Bool(_)) => "Boolean",
                Some(Value::Int(_)) => "Integer",
                Some(Value::Float(_)) => "Float",
                Some(Value::String(_)) => "String",
                Some(Value::List(_)) => "List",
                Some(Value::Map(_)) => "Map",
                Some(Value::Node(_)) => "Node",
                Some(Value::Relationship(_)) => "Edge",
                Some(Value::Path(_)) => "Path",
                Some(Value::VecF32(_)) => "Vectorf32",
                Some(Value::Point(_)) => "Point",
                Some(Value::Datetime(_)) => "Datetime",
                Some(Value::Date(_)) => "Date",
                Some(Value::Time(_)) => "Time",
                Some(Value::Duration(_)) => "Duration",
                None => unreachable!(),
            };
            Ok(Value::String(Arc::new(String::from(type_name))))
        }
    );

    cypher_fn!(funcs, "hasLabels",
        args: [
            Type::Union(vec![Type::Node, Type::Null]),
            Type::List(Box::new(Type::Any)),
        ],
        ret: Type::Union(vec![Type::Bool, Type::Null]),
        fn has_labels(runtime, args) {
            let mut iter = args.into_iter();
            match (iter.next(), iter.next()) {
                (Some(Value::Node(id)), Some(Value::List(required_labels))) => {
                    // Validate that all items in the list are strings
                    for label_value in required_labels.iter() {
                        match label_value {
                            Value::String(_) => {}
                            Value::Int(_) => {
                                return Err("Type mismatch: expected String but was Integer".to_string());
                            }
                            Value::Float(_) => {
                                return Err("Type mismatch: expected String but was Float".to_string());
                            }
                            Value::Bool(_) => {
                                return Err("Type mismatch: expected String but was Boolean".to_string());
                            }
                            _ => return Err("Type mismatch: expected String".to_string()),
                        }
                    }

                    // Get the actual labels of the node
                    let node_labels = runtime.get_node_labels(id);
                    // Check if all required labels are present
                    let has_all = required_labels.iter().all(|req_label| {
                        if let Value::String(req_str) = req_label {
                            node_labels.iter().any(|node_label| node_label == req_str)
                        } else {
                            false
                        }
                    });

                    Ok(Value::Bool(has_all))
                }
                (Some(Value::Null), _) | (_, Some(Value::Null)) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "id",
        args: [Type::Union(vec![
            Type::Node,
            Type::Relationship,
            Type::Null,
        ])],
        ret: Type::Union(vec![Type::Int, Type::Null]),
        fn id(_runtime, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Node(id)) => Ok(Value::Int(u64::from(id) as i64)),
                Some(Value::Relationship(rel)) => Ok(Value::Int(u64::from(rel.0) as i64)),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "properties",
        args: [Type::Union(vec![
            Type::Map,
            Type::Node,
            Type::Relationship,
            Type::Null,
        ])],
        ret: Type::Union(vec![Type::Map, Type::Null]),
        fn properties(runtime, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Map(map)) => Ok(Value::Map(map)),
                Some(Value::Node(id)) => Ok(Value::Map(Arc::new(runtime.get_node_attrs(id)))),
                Some(Value::Relationship(rel)) => {
                    Ok(Value::Map(Arc::new(runtime.get_relationship_attrs(rel.0))))
                }
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "startnode",
        args: [Type::Relationship],
        ret: Type::Union(vec![Type::Node, Type::Null]),
        fn start_node(_runtime, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Relationship(rel)) => Ok(Value::Node(rel.1)),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "endnode",
        args: [Type::Relationship],
        ret: Type::Union(vec![Type::Node, Type::Null]),
        fn end_node(_runtime, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Relationship(rel)) => Ok(Value::Node(rel.2)),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "length",
        args: [Type::Union(vec![Type::Path, Type::Null])],
        ret: Type::Union(vec![Type::Int, Type::Null]),
        fn length(_runtime, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Path(path)) => Ok(Value::Int(path.len() as i64 / 2)),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "keys",
        args: [Type::Union(vec![
            Type::Map,
            Type::Node,
            Type::Relationship,
            Type::Null,
        ])],
        ret: Type::Union(vec![Type::List(Box::new(Type::String)), Type::Null]),
        fn keys(runtime, args) {
            match args.into_iter().next() {
                Some(Value::Map(map)) => Ok(Value::List(Arc::new(
                    map.keys().cloned().map(Value::String).collect(),
                ))),
                Some(Value::Node(id)) => Ok(Value::List(Arc::new(
                    runtime
                        .get_node_attrs(id)
                        .into_iter()
                        .map(|(k, _)| Value::String(k))
                        .collect::<ThinVec<_>>(),
                ))),
                Some(Value::Relationship(rel)) => Ok(Value::List(Arc::new(
                    runtime
                        .get_relationship_attrs(rel.0)
                        .into_iter()
                        .map(|(k, _)| Value::String(k))
                        .collect::<ThinVec<_>>(),
                ))),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "type",
        args: [Type::Union(vec![Type::Relationship, Type::Null])],
        ret: Type::Union(vec![Type::String, Type::Null]),
        fn relationship_type(runtime, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Relationship(rel)) => runtime
                    .get_relationship_type(rel.0)
                    .map_or_else(|| Ok(Value::Null), |type_name| Ok(Value::String(type_name))),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "exists",
        args: [Type::Any],
        ret: Type::Union(vec![Type::Bool, Type::Null]),
        fn exists(_, args) {
            let mut iter = args.into_iter();
            match iter.next() {
                Some(Value::Null) => Ok(Value::Bool(false)),
                _ => Ok(Value::Bool(true)),
            }
        }
    );

    cypher_fn!(funcs, "indegree",
        var_arg: Type::Any,
        ret: Type::Union(vec![Type::Int, Type::Null]),
        fn indegree(runtime, args) {
            let (id, types) = parse_degree_args("indegree", args)?;
            id.map_or_else(|| Ok(Value::Null), |id| {
                    let count = if types.is_empty() {
                        runtime.get_node_indegree(id)
                    } else {
                        runtime.get_node_indegree_by_type(id, &types)
                    };
                    Ok(Value::Int(count as i64))
                })
        }
    );

    cypher_fn!(funcs, "outdegree",
        var_arg: Type::Any,
        ret: Type::Union(vec![Type::Int, Type::Null]),
        fn outdegree(runtime, args) {
            let (id, types) = parse_degree_args("outdegree", args)?;
            id.map_or_else(|| Ok(Value::Null), |id| {
                    let count = if types.is_empty() {
                        runtime.get_node_outdegree(id)
                    } else {
                        runtime.get_node_outdegree_by_type(id, &types)
                    };
                    Ok(Value::Int(count as i64))
                })
        }
    );
}

/// Parse arguments for indegree/outdegree functions.
///
/// Accepted call signatures:
///   degree(node)                          → all types
///   degree(node, `type1`, `type2`, ...)   → filter by string varargs
///   degree(node, [`type1`, `type2`])      → filter by list of strings
fn parse_degree_args(
    fn_name: &str,
    args: ThinVec<Value>,
) -> Result<(Option<NodeId>, Vec<Arc<String>>), String> {
    if args.is_empty() {
        return Err(format!(
            "Received 0 arguments to function '{fn_name}', expected at least 1"
        ));
    }

    let mut iter = args.into_iter();
    let node = iter.next().unwrap();

    let id = match node {
        Value::Node(id) => Some(id),
        Value::Null => None,
        other => {
            return Err(format!(
                "Type mismatch: expected Node but was {:?}",
                other.get_type()
            ));
        }
    };

    let rest: ThinVec<Value> = iter.collect();
    if rest.is_empty() {
        return Ok((id, Vec::new()));
    }

    // Single list argument: degree(node, ['type1', 'type2'])
    if rest.len() == 1 {
        if let Value::List(ref list) = rest[0] {
            let mut types = Vec::with_capacity(list.len());
            for v in list.iter() {
                match v {
                    Value::String(s) => {
                        if !types.contains(s) {
                            types.push(s.clone());
                        }
                    }
                    other => {
                        return Err(format!(
                            "Type mismatch: expected String but was {:?}",
                            other.get_type()
                        ));
                    }
                }
            }
            return Ok((id, types));
        }
    } else if matches!(rest[0], Value::List(_)) {
        // List followed by extra args: wrong argument count
        return Err(format!(
            "Received {} arguments to function '{fn_name}', expected at most 2",
            rest.len() + 1
        ));
    }

    // Varargs of strings: degree(node, 'type1', 'type2', ...)
    let mut types = Vec::with_capacity(rest.len());
    for v in rest {
        match v {
            Value::String(s) => {
                if !types.contains(&s) {
                    types.push(s);
                }
            }
            other => {
                return Err(format!(
                    "Type mismatch: expected String but was {:?}",
                    other.get_type()
                ));
            }
        }
    }

    Ok((id, types))
}
