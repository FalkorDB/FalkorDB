//! Batch-mode all-shortest-paths operator — finds all shortest paths between
//! two bound nodes.
//!
//! Implements Cypher `MATCH p = allShortestPaths((a)-[*]->(b))`.
//! Requires both endpoints to be already bound in the environment.
//!
//! ```text
//!  Phase 1: BFS (find shortest distance + collect predecessors)
//!
//!       src ──► A ──► B ──► dst        distances:  src=0, A=1, B=2, dst=3
//!        \          /                   predecessors[dst] = [(B, edge5)]
//!         \──► C ──/                    predecessors[B]   = [(A, edge2), (C, edge4)]
//!                                       predecessors[A]   = [(src, edge1)]
//!                                       predecessors[C]   = [(src, edge3)]
//!
//!  Phase 2: DFS (backtrack from dst to src using predecessors)
//!
//!       dst ──► B ──► A ──► src         path 1: src->A->B->dst
//!       dst ──► B ──► C ──► src         path 2: src->C->B->dst
//! ```
//!
//! For each input row, runs a BFS from `src` to find the shortest distance to
//! `dst` and records all predecessor edges at each level. Then backtracks via
//! DFS from `dst` to `src` to enumerate every shortest path. Supports edge
//! attribute filters, hop-length constraints, and bidirectional patterns.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::graph::graph::{EdgeDirection, NodeId, RelationshipId};
use crate::parser::ast::{AllShortestPaths, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    eval::ExprEval,
    row::RowView,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use thin_vec::ThinVec;

use super::batched_result_emitter::{BatchedResultEmitter, RowIter};

pub struct AllShortestPathsOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Holds the parent batch being expanded and performs the shared
    /// pack-and-gather emit, binding each enumerated shortest path (a
    /// `Value::List` of edges) to the path alias. The emitter resumes a
    /// partially-drained batch across `next()` calls, so an input batch that
    /// produces more than `BATCH_SIZE` paths never drops rows.
    pub(crate) emitter: BatchedResultEmitter<'a, Value>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> AllShortestPathsOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            emitter: BatchedResultEmitter::with_binding(relationship_pattern.alias.id),
            relationship_pattern,
            idx,
        }
    }

    /// Enumerate every shortest path for `batch[row_idx]`, returning them as a
    /// [`RowIter`] of `Value::List` edge lists, or `None` when the row yields no
    /// paths. The BFS runs eagerly (it borrows the graph), but the DFS backtrack
    /// reads only the owned `predecessors` snapshot — so it is returned as a
    /// lazy iterator that walks the backtrack stack one path at a time while the
    /// emitter packs, never materializing the full path set. Callers pass the
    /// op's fields explicitly so this runs inside the emitter closure without
    /// borrowing the emitter through `&self`.
    fn expand_row(
        runtime: &Runtime,
        rp: &QueryRelationship<Arc<String>, Arc<String>, Variable>,
        batch: &Batch,
        row_idx: usize,
    ) -> Result<Option<RowIter<'a, Value>>, String> {
        let vars = BatchRow::new(batch, row_idx);

        // Evaluate edge attribute filter
        let filter_attrs = ExprEval::from_runtime(runtime).eval(
            &rp.attrs,
            rp.attrs.root().idx(),
            Some(&vars),
            None,
        )?;
        let has_edge_filter = matches!(&filter_attrs, Value::Map(m) if !m.is_empty());

        // Get source node
        let src_val = vars.value_at(rp.from.alias.id);
        let src_id = match src_val {
            Some(Value::Node(id)) => id,
            Some(Value::Null) | None => return Ok(None), // NULL endpoint → no results
            Some(_) => {
                return Err(String::from(
                    "encountered unexpected type in Record; expected Node",
                ));
            }
        };

        // Get destination node
        let dst_val = vars.value_at(rp.to.alias.id);
        let dst_id = match dst_val {
            Some(Value::Node(id)) => id,
            Some(Value::Null) | None => return Ok(None),
            Some(_) => {
                return Err(String::from(
                    "encountered unexpected type in Record; expected Node",
                ));
            }
        };

        let max_hops = rp.max_hops.unwrap_or(u32::MAX);
        let min_hops = rp.min_hops.unwrap_or(1);
        let bidirectional = rp.bidirectional;
        let g = runtime.g.borrow();

        // BFS phase: find shortest distance and collect predecessors
        // predecessor map: node -> list of (prev_node, edge_id, edge_src, edge_dst)
        let mut predecessors: HashMap<u64, Vec<(u64, RelationshipId)>> = HashMap::new();
        let mut distances: HashMap<u64, u32> = HashMap::new();
        let mut queue: VecDeque<u64> = VecDeque::new();

        let src = u64::from(src_id);
        let dst = u64::from(dst_id);
        let is_cycle = src == dst;

        distances.insert(src, 0);
        queue.push_back(src);

        let mut shortest_dist: Option<u32> = None;

        while let Some(current) = queue.pop_front() {
            let current_dist = distances[&current];

            // If we found the target and current level exceeds shortest, stop
            if let Some(sd) = shortest_dist
                && current_dist >= sd
            {
                continue;
            }

            if current_dist >= max_hops {
                continue;
            }

            let current_node = NodeId::from(current);
            let direction = if bidirectional {
                EdgeDirection::Both
            } else {
                EdgeDirection::Outgoing
            };
            for (edge_src, edge_dst, edge_id) in
                g.get_node_relationships_by_type(current_node, &rp.types, direction)
            {
                let neighbor = if bidirectional {
                    if edge_src == current_node {
                        Some(u64::from(edge_dst))
                    } else if edge_dst == current_node && edge_src != current_node {
                        Some(u64::from(edge_src))
                    } else {
                        None
                    }
                } else {
                    // Both forward and reversed directed follow outgoing edges.
                    // In the reversed case the pattern is written right-to-left
                    // (e.g. `(v4)<-[*]-(v1)`) but `from` is still the arrow
                    // source (v1), so BFS from v1 along outgoing edges is correct.
                    if edge_src == current_node {
                        Some(u64::from(edge_dst))
                    } else {
                        None
                    }
                };

                let Some(next) = neighbor else {
                    continue;
                };

                // Check edge attribute filter
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

                let next_dist = current_dist + 1;

                // Special case: cycle detection (src == dst)
                // When we find an edge back to src, record it as a predecessor
                // even though src is already in dist at distance 0.
                if is_cycle && next == src {
                    if next_dist < min_hops {
                        // Below min_hops: don't record as a valid cycle
                        continue;
                    }
                    if let Some(sd) = shortest_dist {
                        if next_dist == sd {
                            // Same-distance cycle: add predecessor
                            predecessors
                                .entry(next)
                                .or_default()
                                .push((current, edge_id));
                        }
                        // If next_dist > sd, skip (longer cycle)
                    } else {
                        // First cycle found
                        shortest_dist = Some(next_dist);
                        predecessors
                            .entry(next)
                            .or_default()
                            .push((current, edge_id));
                    }
                    continue;
                }

                if let Some(&existing_dist) = distances.get(&next) {
                    if next_dist == existing_dist {
                        // Same-distance path: add predecessor
                        predecessors
                            .entry(next)
                            .or_default()
                            .push((current, edge_id));
                    }
                    // If next_dist > existing_dist, skip (already found shorter path)
                } else {
                    // First time reaching this node
                    distances.insert(next, next_dist);
                    predecessors
                        .entry(next)
                        .or_default()
                        .push((current, edge_id));
                    if next == dst && next_dist >= min_hops {
                        shortest_dist = Some(next_dist);
                    }
                    // Only enqueue if we haven't exceeded max_hops
                    if next_dist < max_hops {
                        queue.push_back(next);
                    }
                }
            }
        }

        // If destination not reached, no paths
        if !predecessors.contains_key(&dst) {
            return Ok(None);
        }

        // DFS backtrack from dst to src, streamed lazily: the emitter pulls one
        // path at a time off the backtrack stack, so the full path set is never
        // materialized (a diamond-shaped graph can hold exponentially many
        // shortest paths). The iterator owns the `predecessors` snapshot and
        // reads nothing else — no graph borrow escapes.
        let reverse = rp.all_shortest_paths == AllShortestPaths::Reversed;
        let mut stack: Vec<(u64, ThinVec<Value>)> = vec![(dst, ThinVec::new())];
        Ok(Some(RowIter::many(Box::new(std::iter::from_fn(
            move || {
                while let Some((node, edges)) = stack.pop() {
                    if node == src && !edges.is_empty() {
                        let mut path = edges;
                        if !is_cycle {
                            // Built dst→src; reverse in place to src→dst. (Cycles
                            // keep the DFS predecessor-chain order.)
                            path.reverse();
                        }
                        if reverse {
                            path.reverse();
                        }
                        return Some(Value::List(Arc::new(path)));
                    }
                    if let Some(preds) = predecessors.get(&node) {
                        for &(prev, edge_id) in preds {
                            let mut new_edges = edges.clone();
                            new_edges.push(Value::Relationship(edge_id));
                            stack.push((prev, new_edges));
                        }
                    }
                }
                None
            },
        )))))
    }
}

impl<'a> Iterator for AllShortestPathsOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let runtime = self.runtime;
        let rp = self.relationship_pattern;
        loop {
            // Enumerate each active parent row's shortest paths (the BFS +
            // DFS-backtrack borrows the graph, so it runs eagerly) and let the
            // emitter pack them across rows into one gathered batch. The
            // emitter resumes a partially-drained batch across `next()` calls,
            // so a row producing more than `BATCH_SIZE` paths never drops
            // sibling rows. When exhausted (`Ok(None)`), pull the next batch.
            match self
                .emitter
                .emit_lazy(|batch, row| Self::expand_row(runtime, rp, batch, row))
            {
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
