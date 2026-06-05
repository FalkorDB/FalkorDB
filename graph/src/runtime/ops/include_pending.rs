//! Composable operator that augments a child scan with pending mutations.
//!
//! Wraps any scan operator (NodeByLabelScan, NodeByIndexScan, etc.) and:
//! 1. Filters out pending-deleted nodes and nodes with pending label removals
//! 2. After child exhaustion, emits pending-created nodes and nodes with
//!    pending label additions
//!
//! Used inside MERGE match sub-plans so they see in-flight mutations from
//! prior clauses in the same query.

use std::sync::Arc;

use crate::graph::graph::NodeId;
use crate::parser::ast::{QueryNode, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    row::{Row, RowView},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx};
use roaring::RoaringTreemap;

pub struct IncludePendingOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    /// Extra pending nodes to emit after child exhaustion.
    pending_extra: Option<Box<dyn Iterator<Item = NodeId> + 'a>>,
    /// Nodes to filter out of child results.
    deleted_nodes: Option<RoaringTreemap>,
    label_removed_nodes: Option<RoaringTreemap>,
    /// Whether we've finished pulling from the child.
    child_exhausted: bool,
    /// Correlated argument rows captured from the enclosing sub-plan's
    /// `Argument`. Pending-created nodes are paired with each of these so the
    /// downstream filter can resolve correlated variables (e.g. `row.x`) even
    /// when the wrapped scan returned no matching rows.
    argument_rows: Vec<Row>,
    /// Pending-emission cursor: the pending node currently being paired with
    /// argument rows, and the next argument-row index to pair it with.
    cur_pending_node: Option<NodeId>,
    arg_idx: usize,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> IncludePendingOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        node_pattern: &'a QueryNode<Arc<String>, Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            node_pattern,
            pending_extra: None,
            deleted_nodes: None,
            label_removed_nodes: None,
            child_exhausted: false,
            argument_rows: Vec::new(),
            cur_pending_node: None,
            arg_idx: 0,
            idx,
        }
    }

    /// Snapshot the correlated argument rows so pending-created nodes can be
    /// paired with them even when the wrapped scan yields no rows.
    pub(crate) fn capture_argument(
        &mut self,
        batch: &Batch,
    ) {
        self.argument_rows = batch
            .active_indices()
            .map(|r| BatchRow::new(batch, r).to_owned_row())
            .collect();
    }

    /// Initialize pending state (deleted/removed sets and extra nodes to emit).
    fn init_pending_state(&mut self) {
        let g = self.runtime.g.borrow();
        let label_ids: Vec<_> = self
            .node_pattern
            .labels
            .iter()
            .filter_map(|l| g.get_label_id(l))
            .collect();
        drop(g);

        let pending = self.runtime.pending.borrow();

        self.deleted_nodes = Some(pending.deleted_nodes());
        self.label_removed_nodes = Some(pending.nodes_with_pending_label_removes(&label_ids));

        let pending_nodes = if label_ids.len() == self.node_pattern.labels.len() {
            pending.get_pending_nodes_with_labels(&label_ids)
        } else {
            RoaringTreemap::new()
        };

        drop(pending);

        self.pending_extra = Some(Box::new(pending_nodes.into_iter().map(NodeId::from)));
    }

    /// Emit a batch from pending_extra nodes.
    fn emit_pending_batch(&mut self) -> Option<Result<Batch<'a>, String>> {
        let alias = &self.node_pattern.alias;

        let mut builder = BatchBuilder::new();
        // Number of correlated argument rows to pair each pending node with.
        // When uncorrelated (no argument), emit a single row per node with an
        // empty base env.
        let arg_len = self.argument_rows.len().max(1);

        while builder.len() < BATCH_SIZE {
            // Advance to the next pending node when the current one has been
            // paired with every argument row.
            if self.cur_pending_node.is_none() {
                let iter = self.pending_extra.as_mut()?;
                match iter.next() {
                    Some(id) => {
                        self.cur_pending_node = Some(id);
                        self.arg_idx = 0;
                    }
                    None => break,
                }
            }
            let id = self.cur_pending_node.unwrap();

            while self.arg_idx < arg_len && builder.len() < BATCH_SIZE {
                let mut row = self
                    .argument_rows
                    .get(self.arg_idx)
                    .cloned()
                    .unwrap_or_else(Row::new);
                row.insert(alias, Value::Node(id));
                builder.push_row(&row);
                self.arg_idx += 1;
            }

            if self.arg_idx >= arg_len {
                self.cur_pending_node = None;
            }
        }

        if builder.is_empty() {
            None
        } else {
            Some(Ok(builder.finish()))
        }
    }
}

impl<'a> Iterator for IncludePendingOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Lazily initialize pending state on first call.
        if self.deleted_nodes.is_none() {
            self.init_pending_state();
        }

        // Phase 1: Pull from child, filtering out deleted/removed nodes.
        if !self.child_exhausted {
            loop {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        let deleted = self.deleted_nodes.as_ref().unwrap();
                        let removed = self.label_removed_nodes.as_ref().unwrap();
                        let alias = &self.node_pattern.alias;

                        // Filter the batch: keep only rows whose node is not deleted/removed
                        let mut builder = BatchBuilder::new();
                        for row in batch.active_indices() {
                            let env = BatchRow::new(&batch, row).to_owned_row();
                            if let Some(Value::Node(nid)) = env.get_by_id(alias.id) {
                                let raw: u64 = (*nid).into();
                                if deleted.contains(raw) || removed.contains(raw) {
                                    continue;
                                }
                            } else {
                                debug_assert!(
                                    false,
                                    "IncludePendingOp: missing node binding for alias"
                                );
                                continue;
                            }
                            builder.push_row(&env);
                        }

                        if !builder.is_empty() {
                            return Some(Ok(builder.finish()));
                        }
                        // All rows filtered, pull next batch from child
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => {
                        self.child_exhausted = true;
                        break;
                    }
                }
            }
        }

        // Phase 2: Emit pending-created and label-added nodes.
        self.emit_pending_batch()
    }
}
