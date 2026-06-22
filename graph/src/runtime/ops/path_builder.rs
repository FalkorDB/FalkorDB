//! Batch-mode path builder operator — assembles named path values.
//!
//! Implements Cypher named paths like `p = (a)-[r]->(b)`. For each path
//! definition, reads the component variable columns via `read_columns`,
//! maps each row into a `Value::Path` in alternating format
//! `[Node, Rel, Node, Rel, ..., Node]`, and writes the result column
//! back via `write_column`.
//!
//! ```text
//!  Path definition: p = (a)-[r]->(b)
//!  Variables: [a, r, b]
//!
//!  Row: a=Node(5), r=Rel(1,5,7), b=Node(7)
//!                    │
//!                    ▼
//!  p = Path([Node(5), Rel(1,5,7), Node(7)])
//! ```
//!
//! Handles variable-length relationships (stored as `Value::Path` or
//! `Value::List` of edges) by inlining path elements and deduplicating
//! shared endpoint nodes. Supports reversed VLT paths by detecting
//! direction mismatches and walking edges in reverse order.

use std::sync::Arc;

use crate::parser::ast::{QueryPath, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx};
use thin_vec::ThinVec;

pub struct PathBuilderOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    paths: &'a [Arc<QueryPath<Variable>>],
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> PathBuilderOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        paths: &'a [Arc<QueryPath<Variable>>],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            paths,
            idx,
        }
    }
}

impl<'a> Iterator for PathBuilderOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = match self.child.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        for path in self.paths {
            let path_values: Result<Vec<Value>, String> = batch
                .active_indices()
                .map(|row_idx| {
                    let mut elems = ThinVec::new();
                    let mut skip_next = false;
                    for var in &path.vars {
                        if skip_next {
                            skip_next = false;
                            continue;
                        }
                        if !batch.is_bound_at(var.id, row_idx) {
                            return Err(format!("Variable {} not found", var.as_str()));
                        }
                        let val = batch.value_at(var.id, row_idx).unwrap_or(Value::Null);
                        // Variable-length relationship: the VLT operator stores
                        // the result as a Path in alternating [Node, Rel, Node, ...]
                        // format. Incorporate it directly, skipping the leading
                        // node (which duplicates the preceding node already in elems)
                        // and the following endpoint variable.
                        if let Value::Path(path_elems) = &val {
                            if path_elems.len() > 1 {
                                // Check if VLT path direction matches the pattern
                                // direction. For incoming patterns (m)<-[*]-(n), the
                                // VLT traverses from n to m but the pattern starts
                                // at m. Detect this by checking if the VLT path's
                                // first node matches the preceding node in elems.
                                let prev_id = elems.iter().rev().find_map(|v| {
                                    if let Value::Node(id) = v {
                                        Some(*id)
                                    } else {
                                        None
                                    }
                                });
                                let vlt_first_matches = match path_elems.first() {
                                    Some(Value::Node(id)) => prev_id == Some(*id),
                                    _ => true,
                                };
                                if vlt_first_matches {
                                    // Normal case: skip first node, append rest
                                    for elem in path_elems.iter().skip(1) {
                                        elems.push(elem.clone());
                                    }
                                } else {
                                    // Reversed case: VLT path goes n->...->m but
                                    // pattern needs m->...->n. Walk the relationships
                                    // in reverse, using "other endpoint" to resolve nodes.
                                    for elem in path_elems.iter().rev().skip(1) {
                                        match elem {
                                            Value::Relationship(rel) => {
                                                elems.push(elem.clone());
                                                let cur =
                                                    elems.iter().rev().skip(1).find_map(|v| {
                                                        if let Value::Node(id) = v {
                                                            Some(*id)
                                                        } else {
                                                            None
                                                        }
                                                    });
                                                let (src, dst) =
                                                    self.runtime.get_relationship_endpoints(*rel);
                                                let next = if cur == Some(src) { dst } else { src };
                                                elems.push(Value::Node(next));
                                            }
                                            Value::Node(_) => {
                                                // Skip intermediate nodes — we compute them from edges
                                            }
                                            other => elems.push(other.clone()),
                                        }
                                    }
                                }
                            }
                            // Skip the following endpoint node variable (it
                            // duplicates the last node in the path).
                            skip_next = true;
                        } else if let Value::List(edges) = &val {
                            if !edges.is_empty() {
                                for edge in edges.iter() {
                                    // Determine the next node: whichever endpoint
                                    // differs from the preceding node in the path.
                                    // This handles incoming/bidirectional edges where
                                    // the stored edge direction may oppose the
                                    // traversal direction.
                                    let prev_id = elems.iter().rev().find_map(|v| {
                                        if let Value::Node(id) = v {
                                            Some(*id)
                                        } else {
                                            None
                                        }
                                    });
                                    elems.push(edge.clone());
                                    if let Value::Relationship(rel) = edge {
                                        let (src, dst) =
                                            self.runtime.get_relationship_endpoints(*rel);
                                        let next = if prev_id == Some(src) { dst } else { src };
                                        elems.push(Value::Node(next));
                                    }
                                }
                            }
                            // 0-hop: skip the following endpoint node since it
                            // duplicates the preceding node already in elems.
                            // The last edge's destination is the same as the
                            // following endpoint node, so skip it.
                            skip_next = true;
                        } else {
                            elems.push(val);
                        }
                    }
                    Ok(Value::Path(Arc::new(elems)))
                })
                .collect();

            let path_values = match path_values {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };

            batch.write_column(path.var.id, path_values);
        }

        Some(Ok(batch))
    }
}
