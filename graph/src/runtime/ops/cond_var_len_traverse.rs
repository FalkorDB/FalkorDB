//! Batch-mode variable-length traverse operator — multi-hop relationship expansion.
//!
//! Implements Cypher patterns like `(a)-[*2..5]->(b)`. For each active row
//! in each input batch, enumerates all trails (no repeated edges within a
//! single path; nodes may repeat) from the source node up to `max_hops` away, yielding result
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
//! Edge uniqueness within each path is tracked with a small inline vec of
//! used edge IDs checked by linear scan (paths are short, so this beats a
//! set — same approach as the C engine's `Path_ContainsEdge`). Adjacency
//! lists are lazily cached per node to avoid creating GraphBLAS iterators
//! at every DFS step.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use ahash::RandomState;

use crate::graph::graph::{LabelId, NodeId, RelationshipId};
use crate::parser::ast::{QueryExpr, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    eval::ExprEval,
    row::{Row, RowView},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};
use smallvec::SmallVec;
use thin_vec::ThinVec;

use super::batched_result_emitter::{BatchedResultEmitter, RowIter, VarLenEndpoints};

/// A single expanded var-length result: the resolved `from`/`to` endpoint node
/// ids plus an optional materialized path value (`Some` only when the path is
/// read downstream, i.e. `emit_path`).
type VarLenResult = (NodeId, NodeId, Option<Value>);

/// Emissions produced by one DFS frame, yielded before the next frame runs.
/// Inline capacity 4 keeps a low-fan-out frame off the heap.
type FrameBuf = SmallVec<[VarLenResult; 4]>;

/// Edge IDs already used on the current path, checked by linear scan.
/// Inline capacity covers typical hop bounds; deeper paths spill to the heap.
type UsedEdges = SmallVec<[u64; 12]>;

/// Streaming var-length DFS over one input row: yields `(from, to, opt_path)`
/// results one at a time as the emitter packs, never materializing the row's
/// full result set.
///
/// All traversal state is owned; the graph is re-borrowed for the duration of a
/// single DFS frame and released before yielding, so **no graph borrow is held
/// while the iterator sits suspended in the emitter**. That makes suspension
/// safe alongside writes: mutations go to `pending` and are applied by a
/// `Commit` op, which only runs once its entire subtree (including this
/// iterator) is exhausted — so the graph provably cannot change between two
/// frames of a live traversal, and the commit's exclusive borrow never races a
/// held read borrow.
struct VarLenIter<'a> {
    runtime: &'a Runtime<'a>,
    rp: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    /// WHERE-clause per-hop edge filter plus the owned input row used as its
    /// evaluation environment (only cloned when a filter is present).
    edge_filter: Option<(&'a QueryExpr<Variable>, Row)>,
    emit_path: bool,
    /// Evaluated inline edge-attribute filter (`{k: v}`), shared by all hops.
    filter_attrs: Value,
    has_edge_filter: bool,
    reversed: bool,
    bidirectional: bool,
    min_hops: u32,
    max_hops: u32,
    dest_id: Option<NodeId>,
    /// Destination label filter resolved to ids once per row, so the per-hop
    /// check is a matrix probe (`node_has_label_id`) instead of a
    /// GraphBLAS-iterator walk plus string compares per candidate.
    dest_label_ids: SmallVec<[LabelId; 2]>,
    /// A destination label doesn't exist in the graph: no node can match, so
    /// every emission is suppressed without probing.
    dest_label_missing: bool,
    /// Remaining DFS start nodes (one for a bound endpoint, every label node
    /// otherwise).
    start_nodes: std::vec::IntoIter<NodeId>,
    /// Start node of the DFS currently running (the fixed endpoint of every
    /// emission).
    current_start: Option<NodeId>,
    /// Lazily-built adjacency cache, shared across the row's whole DFS.
    adj_cache: HashMap<u64, Vec<(NodeId, NodeId, RelationshipId)>, RandomState>,
    /// DFS frames: (node, path_elems, used_edges, depth). Depth counts edges
    /// traversed so far; uniqueness is edge-based (Cypher trail semantics —
    /// nodes may repeat, relationships may not), matching the C engine's
    /// `Path_ContainsEdge` check.
    stack: Vec<(NodeId, ThinVec<Value>, UsedEdges, u32)>,
    /// Reusable per-frame buffer of filter-passing `(edge, dest)` neighbors,
    /// so no Vec is allocated per DFS frame.
    scratch: Vec<(RelationshipId, NodeId)>,
    /// Current frame's emissions, stored reversed so `pop()` yields them in
    /// adjacency order. Bounded by the frame's fan-out.
    buf: FrameBuf,
    /// Mid-stream WHERE-filter evaluation error, parked for the op to surface
    /// after the emit that hit it (the item stream itself has no error channel).
    error: Rc<RefCell<Option<String>>>,
}

impl VarLenIter<'_> {
    /// Open the DFS for one start node: queue the 0-hop emission when
    /// applicable and push the initial frame. Borrows the graph only for the
    /// duration of this step.
    fn begin_start_node(
        &mut self,
        start_node: NodeId,
    ) {
        self.current_start = Some(start_node);
        let rt = self.runtime;
        let g = rt.g.borrow();
        // 0-hop case: the start node itself is a valid result.
        if self.min_hops == 0
            && (self.dest_id.is_none() || self.dest_id == Some(start_node))
            && !self.dest_label_missing
            && self
                .dest_label_ids
                .iter()
                .all(|l| g.node_has_label_id(start_node, *l))
        {
            let path = self.emit_path.then(|| {
                let mut path_elems = ThinVec::new();
                path_elems.push(Value::Node(start_node));
                Value::Path(Arc::new(path_elems))
            });
            self.buf.push((start_node, start_node, path));
        }
        // The path is only materialized when consumed downstream; otherwise it
        // stays empty (hop-counting uses the frame's depth counter).
        let mut initial_path = ThinVec::new();
        if self.emit_path {
            initial_path.push(Value::Node(start_node));
        }
        self.stack
            .push((start_node, initial_path, UsedEdges::new(), 0));
    }

    /// Process one DFS frame under a single graph borrow: collect the frame's
    /// valid neighbors, queue its emissions into [`buf`](Self::buf) (reversed,
    /// Advance the DFS under a **single graph borrow** until the current frame
    /// produces emissions (queued into [`buf`](Self::buf), reversed so `pop()`
    /// yields adjacency order) or the stack is exhausted. Frames that emit
    /// nothing (dead ends, over-max-hops prunes) are processed in the same
    /// tight loop, so the borrow/field overhead is paid per *emitting* frame,
    /// not per frame. The borrow is released on return — always before a
    /// yield. A WHERE-filter evaluation error is parked in
    /// [`error`](Self::error) and halts the traversal.
    #[allow(clippy::too_many_lines)]
    fn advance(&mut self) {
        let rt = self.runtime;
        let g = rt.g.borrow();
        let rp = self.rp;
        let start_node = self.current_start.expect("frame requires a start node");
        let reversed = self.reversed;
        let bidirectional = self.bidirectional;
        let (min_hops, max_hops) = (self.min_hops, self.max_hops);
        let dest_id = self.dest_id;
        let emit_path = self.emit_path;
        let has_edge_filter = self.has_edge_filter;
        let dest_label_missing = self.dest_label_missing;
        let dest_label_ids = self.dest_label_ids.clone();
        let evaluator = ExprEval::from_runtime(rt);

        while let Some((current, mut path, mut used_edges, depth)) = self.stack.pop() {
            let hop = depth + 1;
            if hop > max_hops {
                continue;
            }

            // Lazily cache the adjacency list to avoid creating GraphBLAS iterators
            // at every DFS step.
            let edges = self.adj_cache.entry(u64::from(current)).or_insert_with(|| {
                g.get_node_relationships_by_type(current, &rp.types)
                    .collect()
            });
            self.scratch.clear();

            for &(edge_src, edge_dst, edge_id) in edges.iter() {
                // Skip already-used edges (relationship uniqueness)
                if used_edges.contains(&u64::from(edge_id)) {
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
                    if has_edge_filter && let Value::Map(filter_map) = &self.filter_attrs {
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

                    // Check WHERE-clause edge filter (absorbed by optimizer).
                    // The env row is reused across edges: `insert` overwrites
                    // the alias slot in place, so no per-edge row clone.
                    if let Some((filter_expr, filter_env)) = &mut self.edge_filter {
                        filter_env.insert(&rp.alias, Value::Relationship(edge_id));
                        match evaluator.eval(
                            filter_expr,
                            filter_expr.root().idx(),
                            Some(&*filter_env),
                            None,
                        ) {
                            Ok(Value::Bool(true)) => {}
                            Ok(_) => continue,
                            Err(e) => {
                                *self.error.borrow_mut() = Some(e);
                                return;
                            }
                        }
                    }

                    self.scratch.push((edge_id, dest));
                }
            }

            // Process valid neighbors with clone optimization:
            // The last neighbor can take ownership of `path` and `used_edges`
            // instead of cloning.
            let n_valid = self.scratch.len();
            for ni in 0..n_valid {
                let (edge_id, dest) = self.scratch[ni];
                let is_last = ni + 1 == n_valid;

                let will_emit = hop >= min_hops
                    && (dest_id.is_none() || dest_id == Some(dest))
                    && !dest_label_missing
                    && dest_label_ids.iter().all(|l| g.node_has_label_id(dest, *l));

                let will_continue = hop < max_hops;

                if !will_emit && !will_continue {
                    continue;
                }

                // Build the new path: reuse `path` for the last neighbor, clone
                // otherwise. When `emit_path` is false the path is never read, so
                // `path` stays empty and these clones and pushes are no-ops.
                let mut new_path = if is_last {
                    std::mem::replace(&mut path, ThinVec::new())
                } else {
                    path.clone()
                };
                if emit_path {
                    new_path.push(Value::Relationship(edge_id));
                    new_path.push(Value::Node(dest));
                }

                // Map the traversal endpoints onto the pattern's from/to aliases
                // (swapped for a reversed traversal).
                let (from_node, to_node) = if reversed {
                    (dest, start_node)
                } else {
                    (start_node, dest)
                };

                if will_emit && will_continue {
                    // The path feeds both the emitted result and the stack
                    // continuation. Clone once for the output `Value::Path` and
                    // move the original onto the stack below. When the path isn't
                    // emitted, the (empty) `new_path` moves straight onto the stack
                    // with no clone.
                    let emit_path_val = emit_path.then(|| Value::Path(Arc::new(new_path.clone())));
                    let owned = new_path;
                    self.buf.push((from_node, to_node, emit_path_val));
                    let mut next_used = if is_last {
                        std::mem::take(&mut used_edges)
                    } else {
                        used_edges.clone()
                    };
                    next_used.push(u64::from(edge_id));
                    self.stack.push((dest, owned, next_used, hop));
                } else if will_emit {
                    // Emit only — move path directly into Arc
                    let emit_path_val = emit_path.then(|| Value::Path(Arc::new(new_path)));
                    self.buf.push((from_node, to_node, emit_path_val));
                } else if will_continue {
                    // Continue only — move path to stack
                    let mut next_used = if is_last {
                        std::mem::take(&mut used_edges)
                    } else {
                        used_edges.clone()
                    };
                    next_used.push(u64::from(edge_id));
                    self.stack.push((dest, new_path, next_used, hop));
                }
            }

            // Frame produced emissions: reverse so `pop()` in `next()` yields
            // them front-first, and release the borrow before yielding.
            if !self.buf.is_empty() {
                self.buf.reverse();
                return;
            }
        }
    }
}

impl Iterator for VarLenIter<'_> {
    type Item = VarLenResult;

    fn next(&mut self) -> Option<VarLenResult> {
        loop {
            if let Some(item) = self.buf.pop() {
                return Some(item);
            }
            // Checked only on refill (not per item): an error can only be
            // parked while a frame runs, and the op discards the packed batch
            // on error anyway.
            if self.error.borrow().is_some() {
                return None;
            }
            if self.stack.is_empty() {
                let start = self.start_nodes.next()?;
                self.begin_start_node(start);
            } else {
                self.advance();
            }
        }
    }
}

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
    /// Shared mid-stream error slot. Traversals stream lazily through the
    /// emitter, so a WHERE-filter evaluation error can surface while draining
    /// (after the row's iterator was built); the failing iterator parks it here
    /// and `next()` aborts the query on the emit that hit it.
    error: Rc<RefCell<Option<String>>>,
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
            error: Rc::new(RefCell::new(None)),
            idx,
        }
    }

    /// Resolve the row's endpoints and filters, then return a streaming
    /// [`VarLenIter`] over its var-length results (or `None` when an endpoint is
    /// bound to a non-node). Each yielded `(from, to, opt_path)` is already
    /// mapped onto the pattern's aliases (honoring reversal), so the emitter
    /// binds it without any transpose. Callers pass the op's fields explicitly
    /// so this runs inside the emitter closure without borrowing the emitter
    /// through `&self`.
    fn expand_row(
        runtime: &'a Runtime<'a>,
        rp: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        edge_filter: Option<&'a QueryExpr<Variable>>,
        emit_path: bool,
        error: &Rc<RefCell<Option<String>>>,
        batch: &Batch,
        row_idx: usize,
    ) -> Result<Option<RowIter<'a, VarLenResult>>, String> {
        let vars = BatchRow::new(batch, row_idx);

        // Evaluate edge attribute filter (e.g. {connects: 'BC'})
        let filter_attrs = ExprEval::from_runtime(runtime).eval(
            &rp.attrs,
            rp.attrs.root().idx(),
            Some(&vars),
            None,
        )?;
        let has_edge_filter = matches!(&filter_attrs, Value::Map(m) if !m.is_empty());

        let from_id = vars.value_at(rp.from.alias.id).and_then(|v| match v {
            Value::Node(id) => Some(id),
            _ => None,
        });
        if from_id.is_none() && batch.is_bound_at(rp.from.alias.id, row_idx) {
            return Ok(None);
        }
        let to_id = vars.value_at(rp.to.alias.id).and_then(|v| match v {
            Value::Node(id) => Some(id),
            _ => None,
        });
        if to_id.is_none() && batch.is_bound_at(rp.to.alias.id, row_idx) {
            return Ok(None);
        }

        let min_hops = rp.min_hops.unwrap_or(1);
        let max_hops = rp.max_hops.unwrap_or(u32::MAX);
        let bidirectional = rp.bidirectional;

        // When `to` is bound but `from` is unbound (e.g. `(:L1)<-[:R1*]-()`)
        // we reverse the traversal: start from the bound `to` node and follow
        // edges in the opposite direction, emitting destinations as `from`.
        let reversed = from_id.is_none() && to_id.is_some() && !bidirectional;

        // Get starting nodes
        let start_nodes: Vec<NodeId> = if reversed {
            vec![to_id.unwrap()]
        } else {
            from_id.map_or_else(
                || runtime.g.borrow().get_nodes(&rp.from.labels, 0).collect(),
                |id| vec![id],
            )
        };
        let dest_id = if reversed { from_id } else { to_id };

        // Resolve the destination label filter to ids once per row; the DFS
        // then checks labels with a matrix probe instead of iterating each
        // candidate's labels and comparing strings.
        let dest_labels = if reversed {
            &rp.from.labels
        } else {
            &rp.to.labels
        };
        let mut dest_label_ids: SmallVec<[LabelId; 2]> = SmallVec::new();
        let mut dest_label_missing = false;
        {
            let g = runtime.g.borrow();
            for l in dest_labels.iter() {
                if let Some(id) = g.get_label_id(l) {
                    dest_label_ids.push(id);
                } else {
                    dest_label_missing = true;
                    break;
                }
            }
        }

        // The WHERE filter environment is the input row extended per edge; the
        // row is only cloned out of the batch when a filter is present.
        let edge_filter = edge_filter.map(|f| (f, vars.to_owned_row()));

        Ok(Some(RowIter::many(Box::new(VarLenIter {
            runtime,
            rp,
            edge_filter,
            emit_path,
            filter_attrs,
            has_edge_filter,
            reversed,
            bidirectional,
            min_hops,
            max_hops,
            dest_id,
            dest_label_ids,
            dest_label_missing,
            start_nodes: start_nodes.into_iter(),
            current_start: None,
            adj_cache: HashMap::default(),
            stack: Vec::new(),
            scratch: Vec::new(),
            buf: FrameBuf::new(),
            error: Rc::clone(error),
        }))))
    }
}

impl<'a> Iterator for CondVarLenTraverseOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let runtime = self.runtime;
        let rp = self.relationship_pattern;
        let edge_filter = self.edge_filter;
        let emit_path = self.emit_path;
        let error = Rc::clone(&self.error);
        loop {
            // Each active parent row expands through a streaming `VarLenIter`:
            // the emitter pulls `(from, to, opt_path)` results one at a time
            // while packing them across rows into one gathered batch, so a
            // high-fan-out row never materializes its full result set. The
            // graph is re-borrowed per DFS frame and released before every
            // yield, so no borrow is held while the iterator sits suspended in
            // the emitter — a `Commit`'s exclusive borrow only runs once this
            // subtree is exhausted, so the graph cannot change between frames
            // of a live traversal. When the seeded batch is exhausted
            // (`Ok(None)`), pull and seed the next child batch.
            match self.emitter.emit_lazy(|batch, row| {
                // A previous traversal parked a filter-eval error: abort
                // instead of expanding further rows.
                if let Some(e) = error.borrow().as_ref() {
                    return Err(e.clone());
                }
                Self::expand_row(runtime, rp, edge_filter, emit_path, &error, batch, row)
            }) {
                Ok(Some(out)) => {
                    // A traversal may have failed mid-drain; surface the error
                    // instead of the partially-packed batch.
                    if let Some(e) = self.error.borrow_mut().take() {
                        return Some(Err(e));
                    }
                    return Some(Ok(out));
                }
                Ok(None) => {
                    if let Some(e) = self.error.borrow_mut().take() {
                        return Some(Err(e));
                    }
                    match self.child.next() {
                        Some(Ok(batch)) => self.emitter.seed(batch),
                        Some(Err(e)) => return Some(Err(e)),
                        None => return None,
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}
