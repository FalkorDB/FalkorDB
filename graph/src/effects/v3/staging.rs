//! Staging a `Pending` for the effects tests, through the runtime's own API.
//!
//! The real write path goes through the runtime, which needs a whole query to
//! drive it — a graph, a plan, an executed pipeline — so a test that wants one
//! node of one shape would otherwise have to build all of it.
//!
//! These are an extension trait rather than inherent methods on `Pending`
//! because they are test scaffolding and do not belong in the production type's
//! surface. Everything here that *can* go through `Pending`'s public API does,
//! so the tests exercise the same entry points the runtime ops do — including
//! `validate_node_property` and the sortedness asserts, which the previous
//! hand-rolled staging bypassed.
//!
//! Two fields have no public setter on purpose: `commit` fills them from the
//! graph as it deletes, because the label matrices and edge endpoints are gone
//! by the time effects are encoded. Those are set directly here.

use crate::{
    graph::graph::{DeletedEdge, LabelId, NodeId, RelationshipId},
    runtime::{orderset::OrderSet, pending::Pending, value::Value},
};

fn label_set(labels: &[u64]) -> OrderSet<LabelId> {
    labels.iter().map(|&l| LabelId(l as usize)).collect()
}

pub(crate) trait StagePending {
    /// Stage a created node, with its labels and attributes.
    fn stage_created_node(
        &mut self,
        id: u64,
        labels: &[u64],
        attrs: &[(u16, Value)],
    );

    /// Stage a property update on a node that already existed.
    fn stage_updated_node(
        &mut self,
        id: u64,
        attrs: &[(u16, Value)],
    );

    /// Stage a property change on an edge this query did not create.
    fn stage_updated_edge(
        &mut self,
        id: u64,
        attrs: &[(u16, Value)],
    );

    /// Stage a label change on a node this query did not create.
    fn stage_label_change(
        &mut self,
        id: u64,
        labels: &[u64],
    );

    /// Stage deleted edges, endpoints and type included.
    fn stage_deleted_edges(
        &mut self,
        edges: Vec<DeletedEdge>,
    );

    /// Stage a deleted node together with the labels it carried.
    fn stage_deleted_node(
        &mut self,
        id: u64,
        labels: &[u64],
    );
}

impl StagePending for Pending {
    fn stage_created_node(
        &mut self,
        id: u64,
        labels: &[u64],
        attrs: &[(u16, Value)],
    ) {
        // Order matters: `set_node_attributes` routes to the new-nodes map only
        // once the id is known to be created by this query.
        self.created_nodes(&[NodeId::from(id)]);
        if !labels.is_empty() {
            self.set_node_labels(NodeId::from(id), &label_set(labels));
        }
        self.set_node_attributes(NodeId::from(id), attrs.to_vec())
            .expect("test attrs are valid node properties");
    }

    fn stage_updated_node(
        &mut self,
        id: u64,
        attrs: &[(u16, Value)],
    ) {
        self.set_node_attributes(NodeId::from(id), attrs.to_vec())
            .expect("test attrs are valid node properties");
    }

    fn stage_updated_edge(
        &mut self,
        id: u64,
        attrs: &[(u16, Value)],
    ) {
        self.set_relationship_attributes(RelationshipId::from(id), attrs.to_vec())
            .expect("test attrs are valid relationship properties");
    }

    fn stage_label_change(
        &mut self,
        id: u64,
        labels: &[u64],
    ) {
        self.set_node_labels(NodeId::from(id), &label_set(labels));
    }

    fn stage_deleted_edges(
        &mut self,
        edges: Vec<DeletedEdge>,
    ) {
        self.deleted_endpoints = edges;
    }

    fn stage_deleted_node(
        &mut self,
        id: u64,
        labels: &[u64],
    ) {
        self.deleted_node(NodeId::from(id));
        for &label in labels {
            self.deleted_node_labels.push((id, label));
        }
        // `Graph::delete_nodes` yields these in ascending node order and the
        // emitter walks the two in step, so a test staging out of order must
        // not produce a shape the real path cannot.
        self.deleted_node_labels.sort_unstable();
    }
}
