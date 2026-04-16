//! Batch-mode set operator — updates properties and labels on nodes/relationships.
//!
//! For each active row in each input batch, resolves set items (lazily on
//! first row) and calls `Runtime::set` to record property/label changes
//! in the pending batch.
//!
//! Supports three SET forms:
//! - `SET n.prop = expr` — set a single property
//! - `SET n = expr` / `SET n += expr` — replace or merge all properties
//! - `SET n:Label` — add a label to a node
//!
//! Property changes are skipped when the new value equals the existing
//! value (change-detection optimization).

use std::sync::Arc;

use once_cell::unsync::OnceCell;

use crate::graph::graph::LabelId;
use crate::parser::ast::{ExprIR, SetItem, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct SetOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    items: &'a [SetItem<Arc<String>, Variable>],
    resolved_items: OnceCell<Vec<SetItem<LabelId, Variable>>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> SetOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        items: &'a [SetItem<Arc<String>, Variable>],
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            items,
            resolved_items: OnceCell::new(),
            idx,
        }
    }
}

impl<'a> Iterator for SetOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.child.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        let resolved = self.resolved_items.get_or_init(|| {
            let items = self.runtime.resolve_set_items(self.items);
            self.runtime.pending.borrow_mut().resize(
                self.runtime.g.borrow().node_cap(),
                self.runtime.g.borrow().labels_count(),
            );
            items
        });
        if let Err(e) = self.runtime.set_batch(resolved, &batch) {
            return Some(Err(e));
        }

        Some(Ok(batch))
    }
}
impl Runtime<'_> {
    pub fn set_batch(
        &self,
        items: &Vec<SetItem<LabelId, Variable>>,
        batch: &Batch<'_>,
    ) -> Result<(), String> {
        // Pre-check: if no nodes/relationships have been deleted in this
        // transaction, we can skip the per-row deletion checks entirely.
        let has_deleted_nodes = self.deleted_nodes.borrow().len() > 0;
        let has_pending_deleted_nodes = self.pending.borrow().has_deleted_nodes();
        let has_pending_deleted_rels = self.pending.borrow().has_deleted_relationships();
        let skip_delete_checks =
            !has_deleted_nodes && !has_pending_deleted_nodes && !has_pending_deleted_rels;

        for row in batch.active_indices() {
            let env = batch.env_ref(row);
            self.set_inner(items, env, skip_delete_checks)?;
        }
        Ok(())
    }

    pub fn resolve_set_items(
        &self,
        items: &[SetItem<Arc<String>, Variable>],
    ) -> Vec<SetItem<LabelId, Variable>> {
        items
            .iter()
            .map(|item| match item {
                SetItem::Label { var, labels } => SetItem::Label {
                    var: var.clone(),
                    labels: labels
                        .iter()
                        .map(|l| self.g.borrow_mut().get_label_id_mut(l.as_str()))
                        .collect(),
                },
                SetItem::Attribute {
                    target: entity,
                    value,
                    replace,
                } => SetItem::Attribute {
                    target: entity.clone(),
                    value: value.clone(),
                    replace: *replace,
                },
            })
            .collect()
    }

    #[allow(clippy::too_many_lines)]
    pub fn set(
        &self,
        items: &Vec<SetItem<LabelId, Variable>>,
        vars: &Env<'_>,
    ) -> Result<(), String> {
        self.set_inner(items, vars, false)
    }

    #[allow(clippy::too_many_lines)]
    fn set_inner(
        &self,
        items: &Vec<SetItem<LabelId, Variable>>,
        vars: &Env<'_>,
        skip_delete_checks: bool,
    ) -> Result<(), String> {
        for item in items {
            match item {
                SetItem::Attribute {
                    target: entity,
                    value,
                    replace,
                } => {
                    let run_expr = ExprEval::from_runtime(self).eval(
                        value,
                        value.root().idx(),
                        Some(vars),
                        None,
                    )?;
                    let (entity, attr) = match entity.root().data() {
                        ExprIR::Variable(name) => {
                            let entity = vars
                                .get(name)
                                .ok_or_else(|| format!("Variable {} not found", name.as_str()))?
                                .clone();
                            (entity, None)
                        }
                        ExprIR::Property(property) => (
                            {
                                let this = &self;
                                let idx = entity.root().child(0).idx();
                                crate::runtime::eval::ExprEval::from_runtime(this).eval(
                                    entity,
                                    idx,
                                    Some(vars),
                                    None,
                                )
                            }?,
                            Some(property),
                        ),
                        _ => {
                            unreachable!("set target must be Variable or Property");
                        }
                    };
                    match entity {
                        Value::Node(id) => {
                            if !skip_delete_checks
                                && ((self.g.borrow().is_node_deleted(id)
                                    && !self.pending.borrow().is_node_created(id))
                                    || self.pending.borrow().is_node_deleted(id))
                            {
                                continue;
                            }
                            if let Some(attr) = attr {
                                let existing = if skip_delete_checks {
                                    self.get_node_attribute_no_delete_check(id, attr)
                                } else {
                                    self.get_node_attribute(id, attr)
                                };
                                if let Some(v) = existing
                                    && v == run_expr
                                {
                                    continue;
                                }

                                self.pending.borrow_mut().set_node_attribute(
                                    id,
                                    attr.clone(),
                                    run_expr,
                                )?;
                            } else {
                                match run_expr {
                                    Value::Map(map) => {
                                        if *replace {
                                            self.pending.borrow_mut().clear_node_attributes(id);
                                            for key in self.g.borrow().get_node_attrs(id) {
                                                self.pending.borrow_mut().set_node_attribute(
                                                    id,
                                                    key,
                                                    Value::Null,
                                                )?;
                                            }
                                        }
                                        for (key, value) in map.iter() {
                                            let existing = if skip_delete_checks {
                                                self.get_node_attribute_no_delete_check(id, key)
                                            } else {
                                                self.get_node_attribute(id, key)
                                            };
                                            if let Some(v) = existing
                                                && v == *value
                                            {
                                                continue;
                                            }
                                            self.pending.borrow_mut().set_node_attribute(
                                                id,
                                                key.clone(),
                                                value.clone(),
                                            )?;
                                        }
                                    }
                                    Value::Node(tid) => {
                                        if tid == id {
                                            continue;
                                        }
                                        let g = self.g.borrow();
                                        let attrs = self.get_node_attrs(tid);
                                        if *replace {
                                            for key in g.get_node_attrs(id) {
                                                self.pending.borrow_mut().set_node_attribute(
                                                    id,
                                                    key,
                                                    Value::Null,
                                                )?;
                                            }
                                        }
                                        for (key, value) in attrs {
                                            if let Some(v) = self.get_node_attribute(id, &key)
                                                && v == value
                                            {
                                                continue;
                                            }
                                            self.pending
                                                .borrow_mut()
                                                .set_node_attribute(id, key, value)?;
                                        }
                                    }
                                    Value::Relationship(rel) => {
                                        let g = self.g.borrow();
                                        let attrs = self.get_relationship_attrs(rel.0);
                                        if *replace {
                                            for key in g.get_node_attrs(id) {
                                                self.pending.borrow_mut().set_node_attribute(
                                                    id,
                                                    key,
                                                    Value::Null,
                                                )?;
                                            }
                                        }
                                        for (key, value) in attrs {
                                            if let Some(v) = self.get_node_attribute(id, &key)
                                                && v == value
                                            {
                                                continue;
                                            }
                                            self.pending
                                                .borrow_mut()
                                                .set_node_attribute(id, key, value)?;
                                        }
                                    }
                                    _ => {
                                        return Err("Property values can only be of primitive types or arrays of primitive types".to_string());
                                    }
                                }
                            }
                        }
                        Value::Relationship(target_rel) => {
                            if !skip_delete_checks
                                && ((self.g.borrow().is_relationship_deleted(target_rel.0)
                                    && !self
                                        .pending
                                        .borrow()
                                        .is_relationship_created(target_rel.0))
                                    || self.pending.borrow().is_relationship_deleted(
                                        target_rel.0,
                                        target_rel.1,
                                        target_rel.2,
                                    ))
                            {
                                continue;
                            }
                            if let Some(attr) = attr {
                                let existing = if skip_delete_checks {
                                    self.get_relationship_attribute_no_delete_check(
                                        target_rel.0,
                                        attr,
                                    )
                                } else {
                                    self.get_relationship_attribute(target_rel.0, attr)
                                };
                                if let Some(v) = existing
                                    && v == run_expr
                                {
                                    continue;
                                }

                                self.pending.borrow_mut().set_relationship_attribute(
                                    target_rel.0,
                                    attr.clone(),
                                    run_expr,
                                )?;
                            } else {
                                match run_expr {
                                    Value::Map(map) => {
                                        let g = self.g.borrow();
                                        if *replace {
                                            for key in g.get_relationship_attrs(target_rel.0) {
                                                self.pending
                                                    .borrow_mut()
                                                    .set_relationship_attribute(
                                                        target_rel.0,
                                                        key,
                                                        Value::Null,
                                                    )?;
                                            }
                                        }
                                        for (key, value) in map.iter() {
                                            let existing = if skip_delete_checks {
                                                self.get_relationship_attribute_no_delete_check(
                                                    target_rel.0,
                                                    key,
                                                )
                                            } else {
                                                self.get_relationship_attribute(target_rel.0, key)
                                            };
                                            if let Some(v) = existing
                                                && v == *value
                                            {
                                                continue;
                                            }
                                            self.pending.borrow_mut().set_relationship_attribute(
                                                target_rel.0,
                                                key.clone(),
                                                value.clone(),
                                            )?;
                                        }
                                    }
                                    Value::Node(sid) => {
                                        let g = self.g.borrow();
                                        let attrs = self.get_node_attrs(sid);
                                        if *replace {
                                            for key in g.get_relationship_attrs(target_rel.0) {
                                                self.pending
                                                    .borrow_mut()
                                                    .set_relationship_attribute(
                                                        target_rel.0,
                                                        key,
                                                        Value::Null,
                                                    )?;
                                            }
                                        }
                                        for (key, value) in attrs {
                                            if let Some(v) =
                                                self.get_relationship_attribute(target_rel.0, &key)
                                                && v == value
                                            {
                                                continue;
                                            }
                                            self.pending.borrow_mut().set_relationship_attribute(
                                                target_rel.0,
                                                key,
                                                value,
                                            )?;
                                        }
                                    }
                                    Value::Relationship(source_rel) => {
                                        if source_rel.0 == target_rel.0 {
                                            continue;
                                        }
                                        let g = self.g.borrow();
                                        let attrs = self.get_relationship_attrs(source_rel.0);
                                        if *replace {
                                            for key in g.get_relationship_attrs(target_rel.0) {
                                                self.pending
                                                    .borrow_mut()
                                                    .set_relationship_attribute(
                                                        target_rel.0,
                                                        key,
                                                        Value::Null,
                                                    )?;
                                            }
                                        }
                                        for (key, value) in attrs {
                                            if let Some(v) =
                                                self.get_relationship_attribute(target_rel.0, &key)
                                                && v == value
                                            {
                                                continue;
                                            }
                                            self.pending.borrow_mut().set_relationship_attribute(
                                                target_rel.0,
                                                key,
                                                value,
                                            )?;
                                        }
                                    }
                                    _ => {
                                        return Err("Property values can only be of primitive types or arrays of primitive types".to_string());
                                    }
                                }
                            }
                        }
                        // Silently ignore SET on Null and non-entity types
                        // (e.g. Path), matching C FalkorDB behavior.
                        _ => {}
                    }
                }
                SetItem::Label {
                    var: entity,
                    labels,
                } => {
                    let run_expr = vars.get(entity);
                    match run_expr {
                        Some(Value::Node(id)) => {
                            if !skip_delete_checks
                                && ((self.g.borrow().is_node_deleted(*id)
                                    && !self.pending.borrow().is_node_created(*id))
                                    || self.pending.borrow().is_node_deleted(*id))
                            {
                                continue;
                            }
                            self.pending.borrow_mut().set_node_labels(*id, labels);
                        }
                        Some(Value::Null) => {}
                        _ => {
                            return Err(format!(
                                "Type mismatch: expected Node but was {}",
                                run_expr.map_or_else(|| "undefined", Value::name)
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
