/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

use std::{ffi::c_char, ptr::null_mut};

use crate::binding::graph::*;

enum UndoOp {
    CreateNodes(Vec<Node>),
    CreateEdges(Vec<Edge>),
    DeleteNodes(Vec<(NodeID, AttributeSet, Vec<LabelID>)>),
    DeleteEdges(Vec<(EntityID, NodeID, NodeID, RelationID, AttributeSet)>),
    UpdateNodes(Vec<(Node, AttributeSet)>),
    UpdateEdges(Vec<(Edge, AttributeSet)>),
    AddLabels(Vec<(Node, Vec<LabelID>)>),
    RemoveLabels(Vec<(Node, Vec<LabelID>)>),
    AddSchema(SchemaID, SchemaType),
    AddAttribute(AttributeID),
    CreateIndex(SchemaType, *const c_char, *const c_char, IndexFieldType),
}

pub struct UndoLog {
    ops: Vec<UndoOp>,
}

impl Drop for UndoLog {
    fn drop(&mut self) {
        for op in self.ops.iter_mut() {
            match op {
                UndoOp::UpdateNodes(vec) => {
                    for (_, set) in vec {
                        unsafe { AttributeSet_Free(set) };
                    }
                }
                UndoOp::UpdateEdges(vec) => {
                    for (_, set) in vec {
                        unsafe { AttributeSet_Free(set) };
                    }
                }
                UndoOp::DeleteNodes(vec) => {
                    for (_, set, _) in vec {
                        unsafe { AttributeSet_Free(set) };
                    }
                }
                UndoOp::DeleteEdges(vec) => {
                    for (_, _, _, _, set) in vec {
                        unsafe { AttributeSet_Free(set) };
                    }
                }
                _ => {}
            }
        }
    }
}

impl UndoLog {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn create_node(
        &mut self,
        node: Node,
    ) {
        if self.ops.is_empty() {
            self.ops.push(UndoOp::CreateNodes(vec![node]));
            return;
        }
        let last_op = self.ops.last_mut().unwrap();
        if let UndoOp::CreateNodes(nodes) = last_op {
            nodes.push(node);
        } else {
            self.ops.push(UndoOp::CreateNodes(vec![node]));
        }
    }

    pub fn create_edge(
        &mut self,
        edge: Edge,
    ) {
        if self.ops.is_empty() {
            self.ops.push(UndoOp::CreateEdges(vec![edge]));
            return;
        }
        let last_op = self.ops.last_mut().unwrap();
        if let UndoOp::CreateEdges(edges) = last_op {
            edges.push(edge);
        } else {
            self.ops.push(UndoOp::CreateEdges(vec![edge]));
        }
    }

    pub fn delete_node(
        &mut self,
        node_id: NodeID,
        set: AttributeSet,
        labels: Vec<LabelID>,
    ) {
        if self.ops.is_empty() {
            self.ops
                .push(UndoOp::DeleteNodes(vec![(node_id, set, labels)]));
            return;
        }
        let last_op = self.ops.last_mut().unwrap();
        if let UndoOp::DeleteNodes(vec) = last_op {
            vec.push((node_id, set, labels));
        } else {
            self.ops
                .push(UndoOp::DeleteNodes(vec![(node_id, set, labels)]));
        }
    }

    pub fn delete_edge(
        &mut self,
        edge_id: EntityID,
        src_id: NodeID,
        dest_id: NodeID,
        relation_id: RelationID,
        set: AttributeSet,
    ) {
        if self.ops.is_empty() {
            self.ops.push(UndoOp::DeleteEdges(vec![(
                edge_id,
                src_id,
                dest_id,
                relation_id,
                set,
            )]));
            return;
        }
        let last_op = self.ops.last_mut().unwrap();
        if let UndoOp::DeleteEdges(vec) = last_op {
            vec.push((edge_id, src_id, dest_id, relation_id, set));
        } else {
            self.ops.push(UndoOp::DeleteEdges(vec![(
                edge_id,
                src_id,
                dest_id,
                relation_id,
                set,
            )]));
        }
    }

    pub fn update_node(
        &mut self,
        node: Node,
        old_set: AttributeSet,
    ) {
        if self.ops.is_empty() {
            self.ops.push(UndoOp::UpdateNodes(vec![(node, old_set)]));
            return;
        }
        let last_op = self.ops.last_mut().unwrap();
        if let UndoOp::UpdateNodes(vec) = last_op {
            vec.push((node, old_set));
        } else {
            self.ops.push(UndoOp::UpdateNodes(vec![(node, old_set)]));
        }
    }

    pub fn update_edge(
        &mut self,
        edge: Edge,
        old_set: AttributeSet,
    ) {
        if self.ops.is_empty() {
            self.ops.push(UndoOp::UpdateEdges(vec![(edge, old_set)]));
            return;
        }
        let last_op = self.ops.last_mut().unwrap();
        if let UndoOp::UpdateEdges(vec) = last_op {
            vec.push((edge, old_set));
        } else {
            self.ops.push(UndoOp::UpdateEdges(vec![(edge, old_set)]));
        }
    }

    pub fn add_labels(
        &mut self,
        node: Node,
        labels: Vec<LabelID>,
    ) {
        if self.ops.is_empty() {
            self.ops.push(UndoOp::AddLabels(vec![(node, labels)]));
            return;
        }
        let last_op = self.ops.last_mut().unwrap();
        if let UndoOp::AddLabels(vec) = last_op {
            vec.push((node, labels));
        } else {
            self.ops.push(UndoOp::AddLabels(vec![(node, labels)]));
        }
    }

    pub fn remove_labels(
        &mut self,
        node: Node,
        labels: Vec<LabelID>,
    ) {
        if self.ops.is_empty() {
            self.ops.push(UndoOp::RemoveLabels(vec![(node, labels)]));
            return;
        }
        let last_op = self.ops.last_mut().unwrap();
        if let UndoOp::RemoveLabels(vec) = last_op {
            vec.push((node, labels));
        } else {
            self.ops.push(UndoOp::RemoveLabels(vec![(node, labels)]));
        }
    }

    pub fn add_schema(
        &mut self,
        schema_id: SchemaID,
        schema_type: SchemaType,
    ) {
        self.ops.push(UndoOp::AddSchema(schema_id, schema_type));
    }

    pub fn add_attribute(
        &mut self,
        attribute_id: AttributeID,
    ) {
        self.ops.push(UndoOp::AddAttribute(attribute_id));
    }

    pub fn create_index(
        &mut self,
        schema_type: SchemaType,
        label: *const c_char,
        field: *const c_char,
        index_field_type: IndexFieldType,
    ) {
        self.ops.push(UndoOp::CreateIndex(
            schema_type,
            label,
            field,
            index_field_type,
        ));
    }

    pub unsafe fn rollback(
        &mut self,
        gc: &mut GraphContextAPI,
    ) {
        let mut g = gc.get_graph();
        for op in self.ops.drain(..).rev() {
            match op {
                UndoOp::CreateNodes(mut nodes) => {
                    for node in nodes.iter_mut().rev() {
                        gc.delete_node_from_indices(node, null_mut(), 0);
                    }
                    g.delete_nodes(nodes.as_mut_ptr(), nodes.len() as u64);
                }
                UndoOp::CreateEdges(mut edges) => {
                    for edge in edges.iter_mut().rev() {
                        gc.delete_edge_from_indices(edge);
                    }
                    g.delete_edges(edges.as_mut_ptr(), edges.len() as u64);
                }
                UndoOp::DeleteNodes(mut vec) => {
                    for (node_id, set, labels) in vec.iter_mut().rev() {
                        let mut node = Node {
                            attributes: null_mut(),
                            id: -1,
                        };
                        g.create_node(&mut node, labels.as_mut_ptr(), labels.len() as u32);
                        debug_assert!(*node_id >= node.id);
                        node.attributes.write(*set);
                        gc.add_node_to_indices(&mut node);
                    }
                }
                UndoOp::DeleteEdges(mut vec) => {
                    for (edge_id, src_id, dest_id, relation_id, set) in vec.iter_mut().rev() {
                        let mut edge = Edge {
                            attributes: null_mut(),
                            id: -1,
                            relationship: null_mut(),
                            relation_id: *relation_id,
                            src_id: *src_id,
                            dest_id: *dest_id,
                        };
                        g.create_edge(edge.src_id, edge.dest_id, edge.relation_id, &mut edge);
                        debug_assert!(*edge_id >= edge.id);
                        edge.attributes.write(*set);
                        gc.add_edge_to_indices(&mut edge);
                    }
                }
                UndoOp::UpdateNodes(mut vec) => {
                    for (node, old_set) in vec.iter_mut().rev() {
                        node.set_attributes(old_set);
                        gc.add_node_to_indices(node);
                    }
                }
                UndoOp::UpdateEdges(mut vec) => {
                    for (edge, old_set) in vec.iter_mut().rev() {
                        edge.set_attributes(old_set);
                        gc.add_edge_to_indices(edge);
                    }
                }
                UndoOp::AddLabels(mut vec) => {
                    for (node, labels) in vec.iter_mut().rev() {
                        gc.delete_node_from_indices(node, labels.as_mut_ptr(), labels.len() as u32);
                        g.remove_node_labels(node.id, labels.as_mut_ptr(), labels.len() as u32);
                    }
                }
                UndoOp::RemoveLabels(mut vec) => {
                    for (node, labels) in vec.iter_mut().rev() {
                        g.label_node(node.id, labels.as_mut_ptr(), labels.len() as u32);
                        gc.add_node_to_indices(node);
                    }
                }
                UndoOp::AddSchema(schema_id, schema_type) => {
                    gc.remove_schema(schema_id, schema_type);
                    if schema_type == SchemaType::Node {
                        g.remove_label(schema_id);
                    } else {
                        g.remove_relation(schema_id);
                    }
                }
                UndoOp::AddAttribute(attribute_id) => {
                    gc.remove_attribute(attribute_id);
                }
                UndoOp::CreateIndex(schema_type, label, field, index_field_type) => {
                    gc.delete_index(schema_type, label, field, index_field_type);
                }
            }
        }
    }
}
