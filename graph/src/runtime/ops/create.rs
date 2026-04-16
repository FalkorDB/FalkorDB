//! Batch-mode create operator — creates nodes and relationships.
//!
//! For each active row in each input batch, resolves the create pattern
//! (lazily on first row) and calls `Runtime::create` to reserve IDs and
//! record mutations in the pending batch.
//!
//! ```text
//!  Input batch ──► resolve pattern (once)
//!                       │
//!           ┌───────────┴───────────┐
//!           │  for each node:       │
//!           │    reserve IDs        │
//!           │    record labels      │
//!           │    eval + set attrs   │
//!           │    write ID column    │
//!           ├───────────────────────┤
//!           │  for each rel:        │
//!           │    validate endpoints │
//!           │    reserve IDs        │
//!           │    record type + attrs│
//!           │    write ID column    │
//!           └───────────┬───────────┘
//!                       │
//!              output batch (with new IDs bound)
//! ```

use std::cell::OnceCell;
use std::sync::Arc;

use crate::graph::graph::LabelId;
use crate::parser::ast::{QueryGraph, QueryNode, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::ordermap::OrderMap;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct CreateOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    pattern: QueryGraph<Arc<String>, Arc<String>, Variable>,
    resolved_pattern: OnceCell<QueryGraph<Arc<String>, LabelId, Variable>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> CreateOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        pattern: &QueryGraph<Arc<String>, Arc<String>, Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            pattern: pattern.clone(),
            resolved_pattern: OnceCell::new(),
            idx,
        }
    }
}

impl<'a> Iterator for CreateOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = match self.child.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        let resolved_pattern = self.resolved_pattern.get_or_init(|| {
            let resolved = self.runtime.resolve_pattern(&self.pattern);
            self.runtime.pending.borrow_mut().resize(
                self.runtime.g.borrow().node_cap(),
                self.runtime.g.borrow().labels_count(),
            );
            resolved
        });
        if let Err(e) = self.runtime.create_batch(resolved_pattern, &mut batch) {
            return Some(Err(e));
        }

        Some(Ok(batch))
    }
}

impl Runtime<'_> {
    pub fn create_batch(
        &self,
        pattern: &QueryGraph<Arc<String>, LabelId, Variable>,
        batch: &mut Batch<'_>,
    ) -> Result<(), String> {
        // Track which aliases are created in this pattern for validation skip
        let created_aliases: std::collections::HashSet<u32> =
            pattern.nodes().iter().map(|n| n.alias.id).collect();

        // Process nodes: reserve IDs, evaluate attrs, write IDs back via write_column
        for node in pattern.nodes() {
            let active_len = batch.active_len();

            // Reserve all node IDs at once
            let node_ids = self.g.borrow_mut().reserve_nodes(active_len);

            // Record creations and set labels in batch
            {
                let mut pending = self.pending.borrow_mut();
                pending.created_nodes(&node_ids);
                pending.set_nodes_labels(&node_ids, &node.labels);
            }

            // Evaluate attributes per row, then batch-insert into pending.
            // NOTE: eval() may borrow pending internally (e.g. property reads),
            // so we cannot hold pending.borrow_mut() across eval calls.
            let mut all_attrs: Vec<OrderMap<Arc<String>, Value>> = Vec::with_capacity(active_len);
            for (_i, row) in batch.active_indices().enumerate() {
                let env = batch.env_ref(row);
                let attrs = ExprEval::from_runtime(self).eval(
                    &node.attrs,
                    node.attrs.root().idx(),
                    Some(env),
                    None,
                )?;
                match attrs {
                    Value::Map(attrs) => all_attrs.push(Arc::unwrap_or_clone(attrs)),
                    other => {
                        return Err(format!(
                            "Expected map for node attributes, got {}",
                            other.name()
                        ));
                    }
                }
            }
            // Single borrow to insert all evaluated attrs
            {
                let mut pending = self.pending.borrow_mut();
                for (i, attrs) in all_attrs.into_iter().enumerate() {
                    pending.set_node_attributes(node_ids[i], attrs)?;
                }
            }

            // Write node IDs back as a column
            let values: Vec<Value> = node_ids.into_iter().map(Value::Node).collect();
            batch.write_column(node.alias.id, values);
        }

        // Process relationships: read endpoints via read_columns, write back via write_column
        for rel in pattern.relationships() {
            // Read endpoint IDs using read_columns
            let endpoint_rows = batch.read_columns(&[rel.from.alias.id, rel.to.alias.id]);

            // Extract endpoints — skip validation when both endpoints were
            // created in this same CREATE pattern (guaranteed valid).
            let skip_validation = created_aliases.contains(&rel.from.alias.id)
                && created_aliases.contains(&rel.to.alias.id);

            let mut endpoints = Vec::with_capacity(endpoint_rows.len());
            if skip_validation {
                for row_vals in &endpoint_rows {
                    let Value::Node(from_id) = row_vals[0] else {
                        return Err(String::from("Invalid node id"));
                    };
                    let Value::Node(to_id) = row_vals[1] else {
                        return Err(String::from("Invalid node id"));
                    };
                    endpoints.push((*from_id, *to_id));
                }
            } else {
                let g = self.g.borrow();
                let pending = self.pending.borrow();
                for row_vals in &endpoint_rows {
                    let Value::Node(from_id) = row_vals[0] else {
                        return Err(String::from("Invalid node id"));
                    };
                    let Value::Node(to_id) = row_vals[1] else {
                        return Err(String::from("Invalid node id"));
                    };

                    if (g.is_node_deleted(*from_id) && !pending.is_node_created(*from_id))
                        || pending.is_node_deleted(*from_id)
                        || (g.is_node_deleted(*to_id) && !pending.is_node_created(*to_id))
                        || pending.is_node_deleted(*to_id)
                    {
                        return Err(String::from(
                            "Failed to create relationship; endpoint was not found.",
                        ));
                    }
                    endpoints.push((*from_id, *to_id));
                }
                drop(g);
                drop(pending);
            }

            // Reserve all relationship IDs at once
            let ids = self.g.borrow_mut().reserve_relationships(endpoints.len());

            // Record all created relationships directly into pending (no intermediate Vec)
            let type_name = rel.types.first().unwrap().clone();
            let rel_ids: Vec<_> = ids
                .iter()
                .zip(endpoints.iter())
                .map(|(id, (from, to))| (*id, *from, *to))
                .collect();
            {
                let mut pending = self.pending.borrow_mut();
                for (&id, &(from, to)) in ids.iter().zip(endpoints.iter()) {
                    pending.created_relationship(id, from, to, type_name.clone());
                }
            }

            // Evaluate relationship attributes per row, then batch-insert.
            // Same as nodes: eval() may borrow pending, so separate eval from insert.
            let mut all_rel_attrs: Vec<OrderMap<Arc<String>, Value>> =
                Vec::with_capacity(rel_ids.len());
            for (_i, row) in batch.active_indices().enumerate() {
                let env = batch.env_ref(row);
                let attrs = ExprEval::from_runtime(self).eval(
                    &rel.attrs,
                    rel.attrs.root().idx(),
                    Some(env),
                    None,
                )?;
                match attrs {
                    Value::Map(attrs) => all_rel_attrs.push(Arc::unwrap_or_clone(attrs)),
                    _ => {
                        return Err(String::from("Invalid relationship properties"));
                    }
                }
            }
            {
                let mut pending = self.pending.borrow_mut();
                for (i, attrs) in all_rel_attrs.into_iter().enumerate() {
                    pending.set_relationship_attributes(rel_ids[i].0, attrs)?;
                }
            }

            // Write relationship values back using write_column
            let values: Vec<Value> = rel_ids
                .into_iter()
                .map(|(id, from, to)| Value::Relationship(Box::new((id, from, to))))
                .collect();
            batch.write_column(rel.alias.id, values);
        }

        Ok(())
    }

    pub fn resolve_pattern(
        &self,
        pattern: &QueryGraph<Arc<String>, Arc<String>, Variable>,
    ) -> QueryGraph<Arc<String>, LabelId, Variable> {
        let mut resolved_pattern = QueryGraph::default();
        for node in pattern.nodes() {
            resolved_pattern.add_node(Arc::new(QueryNode::new(
                node.alias.clone(),
                node.labels
                    .iter()
                    .map(|l| self.g.borrow_mut().get_label_id_mut(l.as_str()))
                    .collect(),
                node.attrs.clone(),
            )));
        }
        for rel in pattern.relationships() {
            resolved_pattern.add_relationship(Arc::new(QueryRelationship::new(
                rel.alias.clone(),
                rel.types.clone(),
                rel.attrs.clone(),
                Arc::new(QueryNode::new(
                    rel.from.alias.clone(),
                    rel.from
                        .labels
                        .iter()
                        .map(|l| self.g.borrow_mut().get_label_id_mut(l.as_str()))
                        .collect(),
                    rel.from.attrs.clone(),
                )),
                Arc::new(QueryNode::new(
                    rel.to.alias.clone(),
                    rel.to
                        .labels
                        .iter()
                        .map(|l| self.g.borrow_mut().get_label_id_mut(l.as_str()))
                        .collect(),
                    rel.to.attrs.clone(),
                )),
                rel.bidirectional,
                rel.min_hops,
                rel.max_hops,
            )));
        }
        for path in pattern.paths() {
            resolved_pattern.add_path(path.clone());
        }
        resolved_pattern
    }
}
