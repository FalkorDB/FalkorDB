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

use std::sync::Arc;

use crate::graph::graph::{NodeId, RelationshipId};
use crate::parser::ast::{QueryExpr, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    eval::ExprEval,
    row::RowView,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use roaring::RoaringTreemap;
use smallvec::SmallVec;
use thin_vec::ThinVec;

use super::batched_result_emitter::{BatchedResultEmitter, RowIter, VarLenEndpoints};

/// A single expanded var-length result: the resolved `from`/`to` endpoint node
/// ids plus an optional materialized path value (`Some` only when the path is
/// read downstream, i.e. `emit_path`).
type VarLenResult = (NodeId, NodeId, Option<Value>);

/// Per-input-row result buffer. Inline capacity 4 keeps a low-fan-out row (a
/// handful of reachable destinations) off the heap, so the emitter can drain it
/// via a concrete `smallvec::IntoIter` (no `Box`/vtable) while still spilling
/// for high-fan-out rows.
type ResultBuf = SmallVec<[VarLenResult; 4]>;

pub struct CondVarLenTraverseOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Holds the parent batch being expanded and performs the shared
    /// pack-and-gather emit: each active row's DFS results are packed into
    /// columnar batches via `gather`, replicating the parent columns once per
    /// result instead of cloning the parent env per emitted row. Crucially the
    /// emitter resumes a partially-drained input batch across `next()` calls, so
    /// a batch that produces more than `BATCH_SIZE` results never drops rows.
    pub(crate) emitter: BatchedResultEmitter<'a, VarLenResult>,
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
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        edge_filter: Option<&'a QueryExpr<Variable>>,
        emit_path: bool,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let emitter = BatchedResultEmitter::with_binding(VarLenEndpoints {
            from: relationship_pattern.from.alias.id,
            to: relationship_pattern.to.alias.id,
            // A shared endpoint alias (`(a)-[*]->(a)`) binds one column; keep the
            // `to` value (last-insert-wins) to match the row builder.
            distinct: relationship_pattern.from.alias.id != relationship_pattern.to.alias.id,
            path: emit_path.then_some(relationship_pattern.alias.id),
        });
        Self {
            runtime,
            child,
            emitter,
            relationship_pattern,
            edge_filter,
            emit_path,
            idx,
        }
    }

    /// Enumerate all var-length results reachable from `batch[row_idx]`'s source
    /// binding, pushing each as a `(from, to, opt_path)` tuple into `out`. The
    /// endpoint node ids are already mapped onto the pattern's `from`/`to`
    /// aliases (honoring reversal), so the emitter binds them without any
    /// transpose. Path values are built only when `emit_path` is set. Callers
    /// pass the op's fields explicitly so this runs inside the emitter closure
    /// without borrowing the emitter through `&self`.
    fn expand_row(
        runtime: &Runtime,
        relationship_pattern: &QueryRelationship<Arc<String>, Arc<String>, Variable>,
        edge_filter: Option<&QueryExpr<Variable>>,
        emit_path: bool,
        batch: &Batch,
        row_idx: usize,
        out: &mut ResultBuf,
    ) -> Result<(), String> {
        let vars = BatchRow::new(batch, row_idx);

        // Evaluate edge attribute filter (e.g. {connects: 'BC'})
        let filter_attrs = ExprEval::from_runtime(runtime).eval(
            &relationship_pattern.attrs,
            relationship_pattern.attrs.root().idx(),
            Some(&vars),
            None,
        )?;
        let has_edge_filter = matches!(&filter_attrs, Value::Map(m) if !m.is_empty());

        let from_id = vars
            .value_at(relationship_pattern.from.alias.id)
            .and_then(|v| match v {
                Value::Node(id) => Some(id),
                _ => None,
            });
        if from_id.is_none() && batch.is_bound_at(relationship_pattern.from.alias.id, row_idx) {
            return Ok(());
        }
        let to_id = vars
            .value_at(relationship_pattern.to.alias.id)
            .and_then(|v| match v {
                Value::Node(id) => Some(id),
                _ => None,
            });
        if to_id.is_none() && batch.is_bound_at(relationship_pattern.to.alias.id, row_idx) {
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
                    runtime
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

        let g = runtime.g.borrow();

        for start_node in start_nodes {
            // Handle 0-hop case: start node itself is a valid result.
            if min_hops == 0
                && (dest_id.is_none() || dest_id == Some(start_node))
                && (dest_labels.is_empty()
                    || dest_labels
                        .iter()
                        .all(|l| g.get_node_labels(start_node).any(|nl| nl == *l)))
            {
                // `from` and `to` both bind the start node for a 0-hop result.
                let path = emit_path.then(|| {
                    let mut path_elems = ThinVec::new();
                    path_elems.push(Value::Node(start_node));
                    Value::Path(Arc::new(path_elems))
                });
                out.push((start_node, start_node, path));
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
                if emit_path {
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
                        if let Some(edge_filter) = edge_filter {
                            let mut filter_env = vars.to_owned_row();
                            filter_env
                                .insert(&relationship_pattern.alias, Value::Relationship(edge_id));
                            let result = ExprEval::from_runtime(runtime).eval(
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
                    if emit_path {
                        new_path.push(Value::Relationship(edge_id));
                        new_path.push(Value::Node(dest));
                    }

                    // Map the traversal endpoints onto the pattern's from/to
                    // aliases (swapped for a reversed traversal).
                    let (from_node, to_node) = if reversed {
                        (dest, start_node)
                    } else {
                        (start_node, dest)
                    };

                    if will_emit && will_continue {
                        // The path feeds both the emitted result and the stack
                        // continuation. Clone once for the output `Value::Path`
                        // and move the original onto the stack below. When the
                        // path isn't emitted, the (empty) `new_path` moves
                        // straight onto the stack with no clone.
                        let emit_path_val =
                            emit_path.then(|| Value::Path(Arc::new(new_path.clone())));
                        let owned = new_path;
                        out.push((from_node, to_node, emit_path_val));
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
                        let emit_path_val = emit_path.then(|| Value::Path(Arc::new(new_path)));
                        out.push((from_node, to_node, emit_path_val));
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
        let runtime = self.runtime;
        let rp = self.relationship_pattern;
        let edge_filter = self.edge_filter;
        let emit_path = self.emit_path;
        loop {
            // Expand each active parent row's DFS results on demand and pack them
            // across rows into one gathered batch. `expand_row` enumerates a
            // row's results eagerly into a `SmallVec` (the DFS is stack-based and
            // borrows the graph, so it can't stream lazily), which the emitter
            // then drains — replacing the per-result env clone + row-builder
            // transpose with a single columnar `gather`. Because the emitter
            // resumes a partially-drained batch across `next()` calls, an input
            // batch that yields more than `BATCH_SIZE` results never drops rows.
            // When the seeded batch is exhausted (`Ok(None)`), pull and seed the
            // next child batch.
            match self.emitter.emit_lazy(|batch, row| {
                let mut expanded = ResultBuf::new();
                Self::expand_row(
                    runtime,
                    rp,
                    edge_filter,
                    emit_path,
                    batch,
                    row,
                    &mut expanded,
                )?;
                if expanded.is_empty() {
                    Ok(None)
                } else {
                    // `spread` drains through a concrete `smallvec::IntoIter` —
                    // no `Box`/vtable, and no heap for a low-fan-out row.
                    Ok(Some(RowIter::spread(expanded.into_iter())))
                }
            }) {
                Ok(Some(out)) => return Some(Ok(out)),
                Ok(None) => match self.child.next() {
                    Some(Ok(batch)) => self.emitter.seed(batch),
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                },
                Err(e) => return Some(Err(e)),
            }
        }
    }
}
