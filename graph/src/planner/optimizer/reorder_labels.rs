//! Label reordering optimization pass.
//!
//! Sorts the labels in each `NodeByLabelScan` by their schema label_id
//! (insertion order in the graph). Labels not yet registered in the schema
//! are placed at the end. This produces deterministic display output that
//! reflects schema order rather than user-input or alphabetical order.

use std::sync::Arc;

use orx_tree::{Bfs, DynTree, NodeIdx, NodeRef};

use crate::{graph::graph::Graph, parser::ast::QueryNode, runtime::orderset::OrderSet};

use super::super::IR;

pub fn reorder_labels(
    plan: &mut DynTree<IR>,
    graph: &Graph,
) {
    let indices: Vec<NodeIdx<_>> = plan.root().indices::<Bfs>().collect();
    for idx in indices {
        if let IR::NodeByLabelScan { node } = plan.node(idx).data() {
            let mut labeled: Vec<(Arc<String>, usize)> = node
                .labels
                .iter()
                .map(|l| {
                    let id = graph.get_label_id(l).map_or(usize::MAX, |lid| lid.0);
                    (l.clone(), id)
                })
                .collect();
            // Stable sort by id; ties keep input order.
            labeled.sort_by_key(|(_, id)| *id);
            let sorted: Vec<Arc<String>> = labeled.into_iter().map(|(l, _)| l).collect();
            let same = sorted.iter().zip(node.labels.iter()).all(|(a, b)| a == b)
                && sorted.len() == node.labels.len();
            if same {
                continue;
            }
            let new_node = Arc::new(QueryNode::new(
                node.alias.clone(),
                OrderSet::from_vec(sorted),
                node.attrs.clone(),
            ));
            *plan.node_mut(idx).data_mut() = IR::NodeByLabelScan { node: new_node };
        }
    }
}
