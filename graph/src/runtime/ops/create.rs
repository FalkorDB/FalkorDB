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
use crate::parser::ast::{ExprIR, QueryGraph, QueryNode, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::eval::{ExprEval, ExprNode};
use crate::runtime::ordermap::OrderMap;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, DynTree, NodeIdx, NodeRef};

/// Per-entry plan for a map-literal attrs expression: attribute id, position
/// in the id-sorted output, and the pre-resolved value-expression node.
/// Resolving nodes once per batch means each row's evaluation is pure
/// pointer navigation ([`ExprEval::eval_node`]) with no per-row index
/// validation. Entries are in written order so evaluation side effects match
/// Cypher semantics.
type AttrTemplate<'t> = Vec<(u16, usize, ExprNode<'t>)>;

/// Resolve a map-literal attrs expression once per batch: keys are
/// compile-time string constants, so each name→id lookup happens here
/// instead of per row. Returns `None` when the root is not a map literal
/// (e.g. `CREATE (n $props)`), in which case callers fall back to full
/// evaluation.
fn build_attr_template<'t>(
    attrs: &'t DynTree<ExprIR<Variable>>,
    mut resolve: impl FnMut(&Arc<String>) -> u16,
) -> Option<AttrTemplate<'t>> {
    let root = attrs.root();
    if !matches!(root.data(), ExprIR::Map) {
        return None;
    }
    let mut template: AttrTemplate<'t> = Vec::with_capacity(root.num_children());
    for child in root.children() {
        let ExprIR::Constant(Value::String(key)) = child.data() else {
            return None;
        };
        let id = resolve(key);
        let value_node = child.child(0);
        if let Some(entry) = template.iter_mut().find(|(k, _, _)| *k == id) {
            entry.2 = value_node;
        } else {
            template.push((id, 0, value_node));
        }
    }
    let mut order: Vec<usize> = (0..template.len()).collect();
    order.sort_unstable_by_key(|&i| template[i].0);
    for (pos, &i) in order.iter().enumerate() {
        template[i].1 = pos;
    }
    Some(template)
}

fn resolve_map_attrs(
    map: OrderMap<Arc<String>, Value>,
    mut resolve: impl FnMut(&Arc<String>) -> u16,
) -> Vec<(u16, Value)> {
    let mut out: Vec<(u16, Value)> = map.into_iter().map(|(k, v)| (resolve(&k), v)).collect();
    out.sort_unstable_by_key(|(k, _)| *k);
    out
}

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

        let resolved_pattern = self
            .resolved_pattern
            .get_or_init(|| self.runtime.resolve_pattern(&self.pattern));
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
            let node_ids = self.g.borrow_mut().reserve_nodes(active_len)?;

            // Record creations and set labels in batch
            {
                let mut pending = self.pending.borrow_mut();
                pending.created_nodes(&node_ids);
                pending.set_nodes_labels(&node_ids, &node.labels);
            }

            // Evaluate attributes per row into id-sorted `Vec<(u16, Value)>`.
            // NOTE: eval() may borrow pending internally (e.g. property reads),
            // so we cannot hold pending.borrow_mut() across eval calls.
            let template = build_attr_template(&node.attrs, |k| {
                self.g.borrow_mut().get_or_create_node_attr_id(k)
            });
            let mut all_attrs: Vec<Vec<(u16, Value)>> = Vec::new();
            match &template {
                Some(t) if t.is_empty() => {}
                Some(t) => {
                    let eval = ExprEval::from_runtime(self);
                    all_attrs.reserve(active_len);
                    for row in batch.active_indices() {
                        let env = BatchRow::new(batch, row);
                        let mut out = vec![(0u16, Value::Null); t.len()];
                        for (attr_id, pos, value_node) in t {
                            let v = eval.eval_node(value_node, Some(&env), None)?;
                            out[*pos] = (*attr_id, v);
                        }
                        all_attrs.push(out);
                    }
                }
                None => {
                    let root = node.attrs.root();
                    let eval = ExprEval::from_runtime(self);
                    let mut maps: Vec<OrderMap<Arc<String>, Value>> =
                        Vec::with_capacity(active_len);
                    for row in batch.active_indices() {
                        let env = BatchRow::new(batch, row);
                        let attrs = eval.eval_node(&root, Some(&env), None)?;
                        match attrs {
                            Value::Map(attrs) => maps.push(Arc::unwrap_or_clone(attrs)),
                            other => {
                                return Err(format!(
                                    "Expected map for node attributes, got {}",
                                    other.name()
                                ));
                            }
                        }
                    }
                    let mut g = self.g.borrow_mut();
                    all_attrs.extend(
                        maps.into_iter()
                            .map(|m| resolve_map_attrs(m, |k| g.get_or_create_node_attr_id(k))),
                    );
                }
            }
            // Single borrow to insert all evaluated attrs
            if !all_attrs.is_empty() {
                let mut pending = self.pending.borrow_mut();
                for (i, attrs) in all_attrs.into_iter().enumerate() {
                    pending.set_node_attributes(node_ids[i], attrs)?;
                }
            }

            // Write node IDs back as a column
            let values: Vec<Value> = node_ids.into_iter().map(Value::Node).collect();
            batch.write_column(node.alias.id, values);
        }

        // Process relationships: read endpoints columnar, write back via write_column
        for rel in pattern.relationships() {
            // Extract endpoints — skip validation when both endpoints were
            // created in this same CREATE pattern (guaranteed valid).
            let skip_validation = created_aliases.contains(&rel.from.alias.id)
                && created_aliases.contains(&rel.to.alias.id);

            let mut endpoints = Vec::with_capacity(batch.active_len());
            if skip_validation {
                for row in batch.active_indices() {
                    let Some(Value::Node(from_id)) = batch.value_at(rel.from.alias.id, row) else {
                        return Err(String::from("Invalid node id"));
                    };
                    let Some(Value::Node(to_id)) = batch.value_at(rel.to.alias.id, row) else {
                        return Err(String::from("Invalid node id"));
                    };
                    endpoints.push((from_id, to_id));
                }
            } else {
                let g = self.g.borrow();
                let pending = self.pending.borrow();
                for row in batch.active_indices() {
                    let Some(Value::Node(from_id)) = batch.value_at(rel.from.alias.id, row) else {
                        return Err(String::from("Invalid node id"));
                    };
                    let Some(Value::Node(to_id)) = batch.value_at(rel.to.alias.id, row) else {
                        return Err(String::from("Invalid node id"));
                    };

                    if (g.is_node_deleted(from_id) && !pending.is_node_created(from_id))
                        || pending.is_node_deleted(from_id)
                        || (g.is_node_deleted(to_id) && !pending.is_node_created(to_id))
                        || pending.is_node_deleted(to_id)
                    {
                        return Err(String::from(
                            "Failed to create relationship; endpoint was not found.",
                        ));
                    }
                    endpoints.push((from_id, to_id));
                }
                drop(g);
                drop(pending);
            }

            // Reserve all relationship IDs at once
            let ids = self.g.borrow_mut().reserve_relationships(endpoints.len())?;

            // Record all created relationships directly into pending (no intermediate Vec)
            let type_name = rel.types.first().unwrap().clone();
            {
                let mut pending = self.pending.borrow_mut();
                for (&id, &(from, to)) in ids.iter().zip(endpoints.iter()) {
                    pending.created_relationship(id, from, to, type_name.clone());
                }
            }

            // Evaluate relationship attributes per row, then batch-insert.
            // Same as nodes: eval() may borrow pending, so separate eval from insert.
            let template = build_attr_template(&rel.attrs, |k| {
                self.g.borrow_mut().get_or_create_rel_attr_id(k)
            });
            let mut all_rel_attrs: Vec<Vec<(u16, Value)>> = Vec::new();
            match &template {
                Some(t) if t.is_empty() => {}
                Some(t) => {
                    let eval = ExprEval::from_runtime(self);
                    all_rel_attrs.reserve(ids.len());
                    for row in batch.active_indices() {
                        let env = BatchRow::new(batch, row);
                        let mut out = vec![(0u16, Value::Null); t.len()];
                        for (attr_id, pos, value_node) in t {
                            let v = eval.eval_node(value_node, Some(&env), None)?;
                            out[*pos] = (*attr_id, v);
                        }
                        all_rel_attrs.push(out);
                    }
                }
                None => {
                    let root = rel.attrs.root();
                    let eval = ExprEval::from_runtime(self);
                    let mut maps: Vec<OrderMap<Arc<String>, Value>> = Vec::with_capacity(ids.len());
                    for row in batch.active_indices() {
                        let env = BatchRow::new(batch, row);
                        let attrs = eval.eval_node(&root, Some(&env), None)?;
                        match attrs {
                            Value::Map(attrs) => maps.push(Arc::unwrap_or_clone(attrs)),
                            _ => {
                                return Err(String::from("Invalid relationship properties"));
                            }
                        }
                    }
                    let mut g = self.g.borrow_mut();
                    all_rel_attrs.extend(
                        maps.into_iter()
                            .map(|m| resolve_map_attrs(m, |k| g.get_or_create_rel_attr_id(k))),
                    );
                }
            }
            if !all_rel_attrs.is_empty() {
                let mut pending = self.pending.borrow_mut();
                for (i, attrs) in all_rel_attrs.into_iter().enumerate() {
                    pending.set_relationship_attributes(ids[i], attrs)?;
                }
            }

            // Write relationship values back using write_column
            let values: Vec<Value> = ids.into_iter().map(Value::Relationship).collect();
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
