//! Batch-mode remove operator — removes properties and labels from nodes/relationships.
//!
//! For each active row in each input batch, evaluates the remove items and
//! records property nullifications or label removals in the pending batch.
//!
//! Supports two REMOVE forms:
//! - `REMOVE n.prop` — set the property to NULL (effectively deleting it)
//! - `REMOVE n:Label` — remove a label from a node

use crate::parser::ast::{ExprIR, QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    orderset::OrderSet,
    row::RowView,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct RemoveOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    items: &'a Vec<QueryExpr<Variable>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> RemoveOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        items: &'a Vec<QueryExpr<Variable>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            items,
            idx,
        }
    }
}

impl<'a> Iterator for RemoveOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.child.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        if let Err(e) = self.runtime.remove_batch(self.items, &batch) {
            return Some(Err(e));
        }

        Some(Ok(batch))
    }
}
impl Runtime<'_> {
    pub fn remove_batch(
        &self,
        items: &Vec<QueryExpr<Variable>>,
        batch: &Batch<'_>,
    ) -> Result<(), String> {
        for row in batch.active_indices() {
            let env = BatchRow::new(batch, row);
            self.remove(items, &env)?;
        }
        Ok(())
    }

    pub fn remove<R: RowView + ?Sized>(
        &self,
        items: &Vec<QueryExpr<Variable>>,
        vars: &R,
    ) -> Result<(), String> {
        for item in items {
            let (entity, property, labels) = match item.root().data() {
                ExprIR::Property(property) => (
                    ExprEval::from_runtime(self).eval(
                        item,
                        item.root().child(0).idx(),
                        Some(vars),
                        None,
                    )?,
                    Some(property),
                    None,
                ),
                ExprIR::FuncInvocation(func) if func.name == "hasLabels" => {
                    let labels = item
                        .root()
                        .child(1)
                        .children()
                        .filter_map(|c| match c.data() {
                            ExprIR::Constant(Value::String(label)) => Some(label.clone()),
                            _ => None,
                        })
                        .collect::<OrderSet<_>>();

                    (
                        ExprEval::from_runtime(self).eval(
                            item,
                            item.root().child(0).idx(),
                            Some(vars),
                            None,
                        )?,
                        None,
                        Some(labels),
                    )
                }
                _ => {
                    unreachable!("remove target must be Property or hasLabels");
                }
            };
            match entity {
                Value::Node(node) => {
                    if (self.g.borrow().is_node_deleted(node)
                        && !self.pending.borrow().is_node_created(node))
                        || self.pending.borrow().is_node_deleted(node)
                    {
                        continue;
                    }
                    if let Some(property) = property {
                        self.set_pending_node_attr(node, property, Value::Null)?;
                    }
                    if let Some(labels) = labels {
                        let mut current_labels = self
                            .g
                            .borrow()
                            .get_node_label_ids(node)
                            .collect::<OrderSet<_>>();
                        self.pending
                            .borrow()
                            .update_node_labels(node, &mut current_labels);
                        let labels = labels
                            .iter()
                            .filter_map(|l| self.g.borrow_mut().get_label_id(l.as_str()))
                            .filter(|l| current_labels.contains(l))
                            .collect::<Vec<_>>();
                        self.pending.borrow_mut().remove_node_labels(node, &labels);
                    }
                }
                Value::Relationship(rel) => {
                    if let Some(property) = property {
                        self.set_pending_relationship_attr(rel, property, Value::Null)?;
                    }
                    if labels.is_some() {
                        return Err(String::from(
                            "Type mismatch: expected Node but was Relationship",
                        ));
                    }
                }
                Value::Null => {}
                _ => {
                    return Err(format!(
                        "Type mismatch: expected Node or Relationship but was {}",
                        entity.name()
                    ));
                }
            }
        }
        Ok(())
    }
}
