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

use crate::graph::graph::NodeId;
use crate::parser::ast::{AllShortestPaths, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    eval::ExprEval,
    row::{Row, RowView},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use thin_vec::ThinVec;

pub struct AllShortestPathsOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    pending: VecDeque<Row>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> AllShortestPathsOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            pending: VecDeque::new(),
            relationship_pattern,
            idx,
        }
    }

    fn expand_row(
        &self,
        vars: &Row,
        out: &mut Vec<Row>,
    ) -> Result<(), String> {
        let rp = self.relationship_pattern;

        // Evaluate edge attribute filter
        let filter_attrs = ExprEval::from_runtime(self.runtime).eval(
            &rp.attrs,
            rp.attrs.root().idx(),
            Some(vars),
            None,
        )?;
        let has_edge_filter = matches!(&filter_attrs, Value::Map(m) if !m.is_empty());

        // Get source node
        let src_val = vars.get_by_id(rp.from.alias.id);
        let src_id = match src_val {
            Some(Value::Node(id)) => *id,
            Some(Value::Null) | None => return Ok(()), // NULL endpoint → no results
            Some(_) => {
                return Err(String::from(
                    "encountered unexpected type in Record; expected Node",
                ));
            }
        };

        // Get destination node
        let dst_val = vars.get_by_id(rp.to.alias.id);
        let dst_id = match dst_val {
            Some(Value::Node(id)) => *id,
            Some(Value::Null) | None => return Ok(()),
            Some(_) => {
                return Err(String::from(
                    "encountered unexpected type in Record; expected Node",
                ));
            }
        };

        let max_hops = rp.max_hops.unwrap_or(u32::MAX);
        let min_hops = rp.min_hops.unwrap_or(1);
        let bidirectional = rp.bidirectional;
        let g = self.runtime.g.borrow();

        // BFS phase: find shortest distance and collect predecessors
        // predecessor map: node -> list of (prev_node, edge_id, edge_src, edge_dst)
        let mut predecessors: HashMap<u64, Vec<(u64, u64, u64, u64)>> = HashMap::new();
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
            for (edge_src, edge_dst, edge_id) in
                g.get_node_relationships_by_type(current_node, &rp.types)
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
                            predecessors.entry(next).or_default().push((
                                current,
                                u64::from(edge_id),
                                u64::from(edge_src),
                                u64::from(edge_dst),
                            ));
                        }
                        // If next_dist > sd, skip (longer cycle)
                    } else {
                        // First cycle found
                        shortest_dist = Some(next_dist);
                        predecessors.entry(next).or_default().push((
                            current,
                            u64::from(edge_id),
                            u64::from(edge_src),
                            u64::from(edge_dst),
                        ));
                    }
                    continue;
                }

                if let Some(&existing_dist) = distances.get(&next) {
                    if next_dist == existing_dist {
                        // Same-distance path: add predecessor
                        predecessors.entry(next).or_default().push((
                            current,
                            u64::from(edge_id),
                            u64::from(edge_src),
                            u64::from(edge_dst),
                        ));
                    }
                    // If next_dist > existing_dist, skip (already found shorter path)
                } else {
                    // First time reaching this node
                    distances.insert(next, next_dist);
                    predecessors.entry(next).or_default().push((
                        current,
                        u64::from(edge_id),
                        u64::from(edge_src),
                        u64::from(edge_dst),
                    ));
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
            return Ok(());
        }

        // DFS to enumerate all shortest paths from dst back to src
        let mut paths: Vec<ThinVec<Value>> = Vec::new();
        let mut stack: Vec<(u64, ThinVec<Value>)> = Vec::new();
        stack.push((dst, ThinVec::new()));

        while let Some((node, edges)) = stack.pop() {
            if node == src && !edges.is_empty() {
                if is_cycle {
                    // For cycles, keep the DFS order (predecessor chain order)
                    paths.push(edges);
                } else {
                    // Reverse the path (we built it dst→src, reverse to src→dst)
                    let mut path = ThinVec::with_capacity(edges.len());
                    for edge in edges.iter().rev() {
                        path.push(edge.clone());
                    }
                    paths.push(path);
                }
                continue;
            }

            if let Some(preds) = predecessors.get(&node) {
                for &(prev, edge_id_raw, esrc, edst) in preds {
                    let mut new_edges = edges.clone();
                    new_edges.push(Value::Relationship(Box::new((
                        crate::graph::graph::RelationshipId::from(edge_id_raw),
                        NodeId::from(esrc),
                        NodeId::from(edst),
                    ))));
                    stack.push((prev, new_edges));
                }
            }
        }

        // Emit results
        for mut path in paths {
            if rp.all_shortest_paths == AllShortestPaths::Reversed {
                path.reverse();
            }
            let mut env = vars.clone();
            // Store the edge list for the path builder
            env.insert(&rp.alias, Value::List(Arc::new(path)));
            out.push(env);
        }

        Ok(())
    }
}

impl<'a> Iterator for AllShortestPathsOp<'a> {
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
