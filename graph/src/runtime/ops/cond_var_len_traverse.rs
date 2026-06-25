//! Batch-mode variable-length traverse operator — multi-hop relationship expansion.
//!
//! Implements Cypher patterns like `(a)-[*2..5]->(b)`. For each active row
//! in each input batch, enumerates all simple paths (no repeated edges within
//! a single path) from the source node up to `max_hops` away, yielding result
//! rows for destinations reached at or beyond `min_hops`. Output rows are
//! accumulated into batches of up to `BATCH_SIZE`.
//!
//! ```text
//!  DFS traversal from source node (min_hops=1, max_hops=3):
//!
//!       A ──e1──► B ──e2──► C ──e3──► D
//!       │                   │
//!       └──e4──► E ──e5────►┘
//!
//!  Stack frames:  (A, [A], {})
//!                  ├── (B, [A,e1,B], {e1})        emit at hop 1
//!                  │    ├── (C, [A,e1,B,e2,C], {e1,e2})  emit at hop 2
//!                  │    │    └── (D, [...,e3,D], {e1,e2,e3})  emit at hop 3
//!                  │    └── ...
//!                  └── (E, [A,e4,E], {e4})        emit at hop 1
//!                       └── (C, [A,e4,E,e5,C], {e4,e5})  emit at hop 2
//! ```
//!
//! Path elements use alternating format: `[Node, Rel, Node, Rel, ..., Node]`.
//! Edge uniqueness within each path is tracked with a `RoaringTreemap` of
//! used edge IDs. Adjacency lists are lazily cached per node to avoid
//! creating GraphBLAS iterators at every DFS step.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::graph::graph::{NodeId, RelationshipId};
use crate::parser::ast::{QueryExpr, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    eval::ExprEval,
    row::{Row, RowView},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use roaring::RoaringTreemap;
use thin_vec::ThinVec;

pub struct CondVarLenTraverseOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    pending: VecDeque<Row>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    /// Optional per-hop edge filter expression (absorbed from WHERE clause by the optimizer).
    edge_filter: Option<&'a QueryExpr<Variable>>,
    /// When false, the path/relationship-list binding is never read downstream,
    /// so `expand_row` skips building the per-row `Value::Path` (see the
    /// `reduce_var_len_path` optimizer pass).
    emit_path: bool,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> CondVarLenTraverseOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        edge_filter: Option<&'a QueryExpr<Variable>>,
        emit_path: bool,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            pending: VecDeque::new(),
            relationship_pattern,
            edge_filter,
            emit_path,
            idx,
        }
    }

    fn expand_row(
        &self,
        vars: &Row,
        out: &mut Vec<Row>,
    ) -> Result<(), String> {
        let relationship_pattern = self.relationship_pattern;

        // Evaluate edge attribute filter (e.g. {connects: 'BC'})
        let filter_attrs = ExprEval::from_runtime(self.runtime).eval(
            &relationship_pattern.attrs,
            relationship_pattern.attrs.root().idx(),
            Some(vars),
            None,
        )?;
        let has_edge_filter = matches!(&filter_attrs, Value::Map(m) if !m.is_empty());

        let from_id = vars
            .get_by_id(relationship_pattern.from.alias.id)
            .and_then(|v| match v {
                Value::Node(id) => Some(*id),
                _ => None,
            });
        if from_id.is_none() && vars.is_bound_by_id(relationship_pattern.from.alias.id) {
            return Ok(());
        }
        let to_id = vars
            .get_by_id(relationship_pattern.to.alias.id)
            .and_then(|v| match v {
                Value::Node(id) => Some(*id),
                _ => None,
            });
        if to_id.is_none() && vars.is_bound_by_id(relationship_pattern.to.alias.id) {
            return Ok(());
        }

        let min_hops = relationship_pattern.min_hops.unwrap_or(1);
        let max_hops = relationship_pattern.max_hops.unwrap_or(u32::MAX);
        let bidirectional = relationship_pattern.bidirectional;

        // When `to` is bound but `from` is unbound (e.g. `(:L1)<-[:R1*]-()`)
        // we reverse the traversal: start from the bound `to` node and follow
        // edges in the opposite direction, emitting destinations as `from`.
        let reversed = from_id.is_none() && to_id.is_some() && !bidirectional;

        // Get starting nodes
        let start_nodes: Vec<NodeId> = if reversed {
            vec![to_id.unwrap()]
        } else {
            from_id.map_or_else(
                || {
                    self.runtime
                        .g
                        .borrow()
                        .get_nodes(&relationship_pattern.from.labels, 0)
                        .collect()
                },
                |id| vec![id],
            )
        };

        let dest_labels = if reversed {
            &relationship_pattern.from.labels
        } else {
            &relationship_pattern.to.labels
        };
        let dest_id = if reversed { from_id } else { to_id };

        let g = self.runtime.g.borrow();

        for start_node in start_nodes {
            // Handle 0-hop case: start node itself is a valid result.
            if min_hops == 0
                && (dest_id.is_none() || dest_id == Some(start_node))
                && (dest_labels.is_empty()
                    || dest_labels
                        .iter()
                        .all(|l| g.get_node_labels(start_node).any(|nl| nl == *l)))
            {
                let mut env = vars.clone();
                env.insert(&relationship_pattern.from.alias, Value::Node(start_node));
                env.insert(&relationship_pattern.to.alias, Value::Node(start_node));
                if self.emit_path {
                    let mut path_elems = ThinVec::new();
                    path_elems.push(Value::Node(start_node));
                    env.insert(
                        &relationship_pattern.alias,
                        Value::Path(Arc::new(path_elems)),
                    );
                }
                out.push(env);
            }

            // Pre-collect adjacency list for this start node's DFS to avoid
            // creating GraphBLAS iterators at every DFS step.
            // adj[node_id] = Vec<(edge_src, edge_dst, edge_id)>
            // We build it lazily as we discover new nodes.
            let mut adj_cache: std::collections::HashMap<
                u64,
                Vec<(NodeId, NodeId, RelationshipId)>,
            > = std::collections::HashMap::new();

            // DFS to enumerate paths with no repeated edges.
            // Each stack frame: (node, path_elems, used_edges, nodes_in_path).
            // path_elems uses alternating Path format: [Node, Rel, Node, Rel, ...]
            // `nodes_in_path` mirrors the Node entries of `path_elems` as a set
            // so cycle detection is O(log n) instead of O(path_len) per step
            // (PERF-1).
            let mut stack: Vec<(NodeId, ThinVec<Value>, RoaringTreemap, RoaringTreemap)> =
                Vec::new();
            {
                // The path is only materialized when consumed downstream;
                // otherwise it stays empty (hop-counting uses `nodes_in_path`).
                let mut initial_path = ThinVec::new();
                if self.emit_path {
                    initial_path.push(Value::Node(start_node));
                }
                let mut initial_nodes = RoaringTreemap::new();
                initial_nodes.insert(u64::from(start_node));
                stack.push((
                    start_node,
                    initial_path,
                    RoaringTreemap::new(),
                    initial_nodes,
                ));
            }

            while let Some((current, mut path, mut used_edges, mut nodes_in_path)) = stack.pop() {
                // Hops so far = distinct nodes visited - 1. Derived from
                // `nodes_in_path` (always maintained for cycle detection) rather
                // than `path.len()`, so path materialization can be skipped
                // entirely when `emit_path` is false.
                let hops_so_far = (nodes_in_path.len() as u32).saturating_sub(1);
                let hop = hops_so_far + 1;
                if hop > max_hops {
                    continue;
                }

                // Lazily cache adjacency list to avoid creating GraphBLAS iterators at every DFS step.
                let edges = adj_cache.entry(u64::from(current)).or_insert_with(|| {
                    g.get_node_relationships_by_type(current, &relationship_pattern.types)
                        .collect()
                });
                let mut valid_neighbors: Vec<(NodeId, NodeId, RelationshipId, NodeId)> = Vec::new();

                for &(edge_src, edge_dst, edge_id) in edges.iter() {
                    // Skip already-used edges (relationship uniqueness)
                    if used_edges.contains(u64::from(edge_id)) {
                        continue;
                    }

                    let neighbor = if reversed {
                        if edge_dst == current {
                            Some(edge_src)
                        } else {
                            None
                        }
                    } else if edge_src == current {
                        Some(edge_dst)
                    } else if bidirectional && edge_dst == current {
                        Some(edge_src)
                    } else {
                        None
                    };
                    if let Some(dest) = neighbor {
                        // Check edge attribute filter (inline {key: value})
                        if has_edge_filter && let Value::Map(filter_map) = &filter_attrs {
                            let mut matches = true;
                            for (attr, avalue) in filter_map.iter() {
                                match g.get_relationship_attribute(edge_id, attr) {
                                    Some(pvalue) if pvalue == *avalue => {}
                                    _ => {
                                        matches = false;
                                        break;
                                    }
                                }
                            }
                            if !matches {
                                continue;
                            }
                        }

                        // Check WHERE-clause edge filter (absorbed by optimizer)
                        if let Some(edge_filter) = self.edge_filter {
                            let mut filter_env = vars.clone();
                            filter_env
                                .insert(&relationship_pattern.alias, Value::Relationship(edge_id));
                            let result = ExprEval::from_runtime(self.runtime).eval(
                                edge_filter,
                                edge_filter.root().idx(),
                                Some(&filter_env),
                                None,
                            )?;
                            match result {
                                Value::Bool(true) => {}
                                _ => continue,
                            }
                        }

                        valid_neighbors.push((edge_src, edge_dst, edge_id, dest));
                    }
                }

                // Process valid neighbors with clone optimization:
                // The last neighbor can take ownership of `path`, `used_edges`,
                // and `nodes_in_path` instead of cloning.
                let n_valid = valid_neighbors.len();
                for (ni, &(_, _, edge_id, dest)) in valid_neighbors.iter().enumerate() {
                    let is_last = ni + 1 == n_valid;

                    let will_emit = hop >= min_hops
                        && (dest_id.is_none() || dest_id == Some(dest))
                        && (dest_labels.is_empty()
                            || dest_labels
                                .iter()
                                .all(|l| g.get_node_labels(dest).any(|nl| nl == *l)));

                    let node_already_in_path = nodes_in_path.contains(u64::from(dest));
                    let will_continue = hop < max_hops && !node_already_in_path;

                    if !will_emit && !will_continue {
                        continue;
                    }

                    // Build the new path: reuse `path` for the last neighbor,
                    // clone otherwise. When `emit_path` is false the path is
                    // never read, so `path` stays empty and these clones and
                    // pushes are no-ops.
                    let mut new_path = if is_last {
                        std::mem::replace(&mut path, ThinVec::new())
                    } else {
                        path.clone()
                    };
                    if self.emit_path {
                        new_path.push(Value::Relationship(edge_id));
                        new_path.push(Value::Node(dest));
                    }

                    if will_emit && will_continue {
                        let mut env = vars.clone();
                        if reversed {
                            env.insert(&relationship_pattern.from.alias, Value::Node(dest));
                            env.insert(&relationship_pattern.to.alias, Value::Node(start_node));
                        } else {
                            env.insert(&relationship_pattern.from.alias, Value::Node(start_node));
                            env.insert(&relationship_pattern.to.alias, Value::Node(dest));
                        }
                        // The path feeds both the emitted row and the stack
                        // continuation. Clone once for the output `Value::Path`
                        // and move the original onto the stack below. When the
                        // path isn't emitted, the (empty) `new_path` moves
                        // straight onto the stack with no clone.
                        if self.emit_path {
                            env.insert(
                                &relationship_pattern.alias,
                                Value::Path(Arc::new(new_path.clone())),
                            );
                        }
                        let owned = new_path;
                        out.push(env);
                        let mut next_used = if is_last {
                            std::mem::replace(&mut used_edges, RoaringTreemap::new())
                        } else {
                            used_edges.clone()
                        };
                        next_used.insert(u64::from(edge_id));
                        let mut next_nodes = if is_last {
                            std::mem::replace(&mut nodes_in_path, RoaringTreemap::new())
                        } else {
                            nodes_in_path.clone()
                        };
                        next_nodes.insert(u64::from(dest));
                        stack.push((dest, owned, next_used, next_nodes));
                    } else if will_emit {
                        // Emit only — move path directly into Arc
                        let mut env = vars.clone();
                        if reversed {
                            env.insert(&relationship_pattern.from.alias, Value::Node(dest));
                            env.insert(&relationship_pattern.to.alias, Value::Node(start_node));
                        } else {
                            env.insert(&relationship_pattern.from.alias, Value::Node(start_node));
                            env.insert(&relationship_pattern.to.alias, Value::Node(dest));
                        }
                        if self.emit_path {
                            env.insert(
                                &relationship_pattern.alias,
                                Value::Path(Arc::new(new_path)),
                            );
                        }
                        out.push(env);
                    } else if will_continue {
                        // Continue only — move path to stack
                        let mut next_used = if is_last {
                            std::mem::replace(&mut used_edges, RoaringTreemap::new())
                        } else {
                            used_edges.clone()
                        };
                        next_used.insert(u64::from(edge_id));
                        let mut next_nodes = if is_last {
                            std::mem::replace(&mut nodes_in_path, RoaringTreemap::new())
                        } else {
                            nodes_in_path.clone()
                        };
                        next_nodes.insert(u64::from(dest));
                        stack.push((dest, new_path, next_used, next_nodes));
                    }
                }
            }
        }

        Ok(())
    }
}

impl<'a> Iterator for CondVarLenTraverseOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut builder = BatchBuilder::new();

        // Drain leftover rows from previous call.
        super::drain_pending(&mut self.pending, &mut builder);

        while builder.len() < BATCH_SIZE {
            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            };

            for row in batch.active_indices() {
                let vars = BatchRow::new(&batch, row).to_owned_row();
                let mut expanded = Vec::new();
                if let Err(e) = self.expand_row(&vars, &mut expanded) {
                    return Some(Err(e));
                }
                self.pending.extend(expanded);

                super::drain_pending(&mut self.pending, &mut builder);

                if builder.len() >= BATCH_SIZE {
                    break;
                }
            }
        }

        if builder.is_empty() {
            None
        } else {
            Some(Ok(builder.finish()))
        }
    }
}
