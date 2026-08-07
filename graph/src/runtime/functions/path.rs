//! Path decomposition functions.
//!
//! A Cypher `Path` is stored as a flat `ThinVec<Value>` alternating
//! between nodes and relationships:
//!
//! ```text
//!  Path = [Node, Rel, Node, Rel, Node, ...]
//!          ^^^^       ^^^^       ^^^^
//!        nodes()     nodes()   nodes()
//!               ^^^        ^^^
//!          relationships()  relationships()
//! ```
//!
//! | Cypher              | Function          | Returns               |
//! |----------------------|-------------------|-----------------------|
//! | `nodes(path)`       | `nodes()`         | `[Node]`              |
//! | `relationships(path)`| `relationships()` | `[Relationship]`     |

#![allow(clippy::unnecessary_wraps)]

use super::{FnType, Functions, Type};
use crate::runtime::{runtime::Runtime, value::Value};
use std::sync::Arc;

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "nodes",
        args: [Type::union([Type::Path, Type::Null])],
        ret: Type::union([Type::List(Box::new(Type::Node)), Type::Null]),
        fn nodes(_, args) {
            match args.first() {
                Some(Value::Path(values)) => Ok(Value::List(Arc::new(
                    values
                        .iter()
                        .filter_map(|v| {
                            if let Value::Node(_) = v {
                                Some(v.clone())
                            } else {
                                None
                            }
                        })
                        .collect(),
                ))),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "relationships",
        args: [Type::union([Type::Path, Type::Null])],
        ret: Type::union([Type::List(Box::new(Type::Relationship)), Type::Null]),
        fn relationships(_, args) {
            match args.first() {
                Some(Value::Path(values)) => Ok(Value::List(Arc::new(
                    values
                        .iter()
                        .filter_map(|v| {
                            if let Value::Relationship(_) = v {
                                Some(v.clone())
                            } else {
                                None
                            }
                        })
                        .collect(),
                ))),
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );
}
