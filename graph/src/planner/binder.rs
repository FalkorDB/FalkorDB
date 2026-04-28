//! Semantic analysis and name resolution for Cypher queries.
//!
//! The binder performs the second phase of query processing, converting a raw
//! AST (with string variable names) into a bound AST (with resolved variable
//! IDs and types). This phase:
//!
//! 1. **Resolves variable references** - Links variable uses to their definitions
//! 2. **Manages scopes** - Handles variable visibility across clauses
//! 3. **Infers types** - Determines types for expressions and variables
//! 4. **Validates semantics** - Catches errors like undefined variables
//!
//! ## Scope Rules
//!
//! Cypher has specific scoping rules:
//! - Variables defined in MATCH/CREATE are visible in subsequent clauses
//! - WITH/RETURN create new scopes, only explicitly projected variables carry over
//! - CALL procedures can YIELD variables into scope
//!
//! ## Example
//!
//! ```text
//! MATCH (n:Person)     // Defines variable 'n' with id=0
//! WHERE n.age > 18     // References variable with id=0
//! WITH n, n.name AS name  // Creates new scope, projects n (id=1), defines name (id=2)
//! RETURN name          // References variable with id=2
//! ```

use crate::parser::ast::{
    AllShortestPaths, BoundQueryIR, ExprIR, QueryExpr, QueryGraph, QueryIR, QueryNode, QueryPath,
    QueryRelationship, RawQueryIR, SetItem, SupportAggregation, Variable,
};
use crate::runtime::functions::{FnArguments, FnType, Type};
use crate::runtime::orderset::OrderSet;
use crate::tree;
use orx_tree::{Dfs, Dyn, DynNode, DynTree, NodeRef};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The binder performs semantic analysis on parsed Cypher queries.
///
/// It resolves variable references, manages scope, and converts the raw AST
/// (with string names) into a bound AST (with numeric variable IDs and types).
pub struct Binder {
    /// Stack of variable environments (name → Variable mapping)
    env_stack: Vec<HashMap<Arc<String>, Variable>>,
    /// Whether to look up variables in parent scope
    use_parent_scope: bool,
    /// Variables copied from parent scope to current scope
    parent_to_child_scope: HashMap<Arc<String>, Variable>,
    /// Track which variables need to be copied from parent
    copy_from_parent: HashMap<Arc<String>, (Variable, Variable)>,
    /// Accumulated labels for each node variable across all MATCH clauses.
    /// When `MATCH (n:N) ... MATCH (n:O)` is encountered, this maps
    /// n's (scope_id, variable ID) → {N, O} so the planner can create a single
    /// NodeByLabelScan with the full label set.
    node_labels: HashMap<(u32, u32), OrderSet<Arc<String>>>,
}

impl Default for Binder {
    fn default() -> Self {
        Self {
            env_stack: vec![HashMap::new()],
            use_parent_scope: false,
            parent_to_child_scope: HashMap::new(),
            copy_from_parent: HashMap::new(),
            node_labels: HashMap::new(),
        }
    }
}

/// The type of projection clause (affects scope handling).
#[derive(Clone, Copy)]
enum ProjectionKind {
    With,
    Return,
}

impl Binder {
    /// Binds a raw query IR, resolving all variable references.
    ///
    /// This is the main entry point for semantic analysis.
    /// Returns the bound IR, scope variables, and accumulated node labels.
    pub fn bind(
        mut self,
        ir: RawQueryIR,
    ) -> Result<(BoundQueryIR, Vec<Vec<Variable>>), String> {
        let mut bound = self.bind_ir(ir)?;
        self.update_all_node_labels(&mut bound);
        let scope_vars = self
            .env_stack
            .iter()
            .map(|env| {
                let mut vars = env.values().cloned().collect::<Vec<_>>();
                vars.sort_by_key(|v| v.id);
                vars
            })
            .collect();
        Ok((bound, scope_vars))
    }

    /// Post-process the bound IR: update every QueryNode's labels to the
    /// full accumulated set from `self.node_labels`.  This ensures that
    /// the first MATCH occurrence of a node has labels from all later
    /// MATCH clauses, so the planner can create efficient scans directly.
    fn update_all_node_labels(
        &self,
        ir: &mut BoundQueryIR,
    ) {
        match ir {
            QueryIR::Match { pattern, .. } => {
                Self::update_graph_labels(pattern, &self.node_labels);
            }
            QueryIR::Create(graph) | QueryIR::Merge { pattern: graph, .. } => {
                Self::update_graph_labels(graph, &self.node_labels);
            }
            QueryIR::Query { clauses, .. } => {
                for clause in clauses {
                    self.update_all_node_labels(clause);
                }
            }
            QueryIR::Union { branches, .. } => {
                for branch in branches {
                    self.update_all_node_labels(branch);
                }
            }
            QueryIR::ForEach { body, .. } => {
                for clause in body {
                    self.update_all_node_labels(clause);
                }
            }
            // Labels already applied in bind_call_subquery — skip to
            // avoid overwriting inner labels with colliding outer scope IDs.
            _ => {}
        }
    }

    /// Replace QueryNode labels in the graph with the full accumulated set.
    fn update_graph_labels(
        graph: &mut QueryGraph<Arc<String>, Arc<String>, Variable>,
        node_labels: &HashMap<(u32, u32), OrderSet<Arc<String>>>,
    ) {
        for node in graph.nodes_mut() {
            let key = (node.alias.scope_id, node.alias.id);
            if let Some(labels) = node_labels.get(&key)
                && node.labels != *labels
            {
                *node = Arc::new(QueryNode::new(
                    node.alias.clone(),
                    labels.clone(),
                    node.attrs.clone(),
                ));
            }
        }
        for rel in graph.relationships_mut() {
            let from_key = (rel.from.alias.scope_id, rel.from.alias.id);
            let to_key = (rel.to.alias.scope_id, rel.to.alias.id);
            let from_changed = node_labels
                .get(&from_key)
                .is_some_and(|l| rel.from.labels != *l);
            let to_changed = node_labels
                .get(&to_key)
                .is_some_and(|l| rel.to.labels != *l);
            if from_changed || to_changed {
                let from = if let Some(labels) = node_labels.get(&from_key) {
                    Arc::new(QueryNode::new(
                        rel.from.alias.clone(),
                        labels.clone(),
                        rel.from.attrs.clone(),
                    ))
                } else {
                    rel.from.clone()
                };
                let to = if let Some(labels) = node_labels.get(&to_key) {
                    Arc::new(QueryNode::new(
                        rel.to.alias.clone(),
                        labels.clone(),
                        rel.to.attrs.clone(),
                    ))
                } else {
                    rel.to.clone()
                };
                let mut new_rel = QueryRelationship::new(
                    rel.alias.clone(),
                    rel.types.clone(),
                    rel.attrs.clone(),
                    from,
                    to,
                    rel.bidirectional,
                    rel.min_hops,
                    rel.max_hops,
                );
                new_rel.all_shortest_paths = rel.all_shortest_paths;
                *rel = Arc::new(new_rel);
            }
        }
    }

    fn current_env(&self) -> &HashMap<Arc<String>, Variable> {
        self.env_stack
            .last()
            .expect("env_stack should never be empty")
    }

    fn current_env_mut(&mut self) -> &mut HashMap<Arc<String>, Variable> {
        self.env_stack
            .last_mut()
            .expect("env_stack should never be empty")
    }

    fn push_scope(&mut self) {
        self.env_stack.push(HashMap::new());
        self.use_parent_scope = true;
    }

    fn commit_scope(&mut self) {
        self.use_parent_scope = false;
        self.parent_to_child_scope.clear();
    }

    #[allow(clippy::too_many_lines)]
    fn bind_ir(
        &mut self,
        ir: RawQueryIR,
    ) -> Result<BoundQueryIR, String> {
        match ir {
            QueryIR::Union { branches, all } => {
                // Each UNION branch is an independent sub-query with its own
                // variable scope, so each branch is bound with a fresh Binder.
                let mut bound_branches = Vec::with_capacity(branches.len());
                let mut first_columns: Option<Vec<String>> = None;
                for branch in branches {
                    let binder = Self::default();
                    let (bound, _) = binder.bind(branch)?;
                    let columns = bound.return_column_names();
                    if let Some(ref expected) = first_columns {
                        if columns != *expected {
                            return Err(String::from(
                                "All sub queries in a UNION must have the same column names.",
                            ));
                        }
                    } else {
                        first_columns = Some(columns);
                    }
                    bound_branches.push(bound);
                }
                Ok(QueryIR::Union {
                    branches: bound_branches,
                    all,
                })
            }
            QueryIR::Query { clauses, write } => {
                let mut bound = Vec::with_capacity(clauses.len());
                for clause in clauses {
                    bound.push(self.bind_ir(clause)?);
                }
                Ok(QueryIR::Query {
                    clauses: bound,
                    write,
                })
            }
            QueryIR::Match {
                pattern,
                filter,
                optional,
            } => {
                // Validate allShortestPaths: both source and destination of the overall
                // allShortestPaths pattern must already be resolved. Intermediate nodes
                // (endpoints shared with other relationships in the same pattern) are OK
                // since they'll be bound by preceding CondTraverse ops.
                for rel in pattern.relationships() {
                    if rel.all_shortest_paths != AllShortestPaths::No {
                        let from_name = &rel.from.alias;
                        let to_name = &rel.to.alias;
                        // Collect aliases of nodes that appear as endpoints of other
                        // relationships in the same pattern (i.e., intermediate nodes).
                        let pattern_bound: std::collections::HashSet<_> = pattern
                            .relationships()
                            .iter()
                            .filter(|r| r.all_shortest_paths == AllShortestPaths::No)
                            .flat_map(|r| [r.from.alias.clone(), r.to.alias.clone()])
                            .collect();
                        let from_ok = self.current_env().contains_key(from_name)
                            || pattern_bound.contains(from_name);
                        let to_ok = self.current_env().contains_key(to_name)
                            || pattern_bound.contains(to_name);
                        if !from_ok || !to_ok {
                            return Err(String::from(
                                "Source and destination must already be resolved to call allShortestPaths",
                            ));
                        }
                    }
                }
                let pattern = self.bind_graph(&pattern, false)?;
                let filter = filter.map(|expr| self.bind_expr(&expr)).transpose()?;
                if let Some(ref f) = filter
                    && !Self::expr_may_return_boolean(f.root())
                {
                    return Err(String::from("Expected boolean predicate"));
                }
                Ok(QueryIR::Match {
                    pattern,
                    filter,
                    optional,
                })
            }
            QueryIR::Unwind {
                expr,
                var: var_name,
            } => {
                let expr = self.bind_expr(&expr)?;
                let var = self.define_name_in_scope(var_name, Type::Any, false)?;
                Ok(QueryIR::Unwind { expr, var })
            }
            QueryIR::Merge {
                pattern,
                on_create,
                on_match,
            } => {
                // MERGE may create new entities, so validate that inline attrs
                // don't reference entities being merged.  Entity aliases are
                // not yet in scope, so any such reference is caught here, e.g.:
                //   MERGE (a:L {v: a.v})   → "'a' not defined"
                for node in pattern.nodes() {
                    self.bind_expr(&node.attrs)?;
                }
                for relationship in pattern.relationships() {
                    self.bind_expr(&relationship.attrs)?;
                }
                let pattern = self.bind_graph(&pattern, false)?;
                let on_create = self.bind_set_items(on_create)?;
                let on_match = self.bind_set_items(on_match)?;
                Ok(QueryIR::Merge {
                    pattern,
                    on_create,
                    on_match,
                })
            }
            QueryIR::Create(pattern) => {
                let bound = self.bind_graph_create(&pattern)?;
                Ok(QueryIR::Create(bound))
            }
            QueryIR::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options,
            } => {
                let options = options.map(|expr| self.bind_expr(&expr)).transpose()?;
                Ok(QueryIR::CreateIndex {
                    label,
                    attrs,
                    index_type,
                    entity_type,
                    options,
                })
            }
            QueryIR::DropIndex {
                label,
                attrs,
                index_type,
                entity_type,
            } => Ok(QueryIR::DropIndex {
                label,
                attrs,
                index_type,
                entity_type,
            }),
            QueryIR::Delete { exprs, detach } => {
                let exprs = exprs
                    .iter()
                    .map(|expr| self.bind_expr(expr))
                    .collect::<Result<Vec<_>, _>>()?;
                for expr in &exprs {
                    if !Self::expr_may_return_entity(expr.root()) {
                        return Err(String::from(
                            "DELETE can only be called on nodes, paths and relationships",
                        ));
                    }
                }
                Ok(QueryIR::Delete { exprs, detach })
            }
            QueryIR::Set(items) => Ok(QueryIR::Set(self.bind_set_items(items)?)),
            QueryIR::Remove(items) => {
                let items = items
                    .iter()
                    .map(|expr| self.bind_expr(expr))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(QueryIR::Remove(items))
            }
            QueryIR::LoadCsv {
                file_path,
                headers,
                delimiter,
                var,
            } => {
                let file_path = self.bind_expr(&file_path)?;
                let delimiter = self.bind_expr(&delimiter)?;
                let var = self.define_name_in_scope(var, Type::Any, true)?;
                Ok(QueryIR::LoadCsv {
                    file_path,
                    headers,
                    delimiter,
                    var,
                })
            }
            QueryIR::With {
                distinct,
                all,
                exprs,
                copy_from_parent: _,
                orderby,
                skip,
                limit,
                filter,
                write,
            } => self.bind_projection(
                ProjectionKind::With,
                distinct,
                all,
                &exprs,
                &orderby,
                skip,
                limit,
                filter,
                write,
            ),
            QueryIR::Return {
                distinct,
                all,
                exprs,
                copy_from_parent: _,
                orderby,
                skip,
                limit,
                write,
            } => {
                if all
                    && !self
                        .current_env()
                        .iter()
                        .any(|(key, _)| !key.starts_with('_'))
                {
                    return Err(String::from(
                        "RETURN * is not allowed when there are no variables in scope",
                    ));
                }
                self.bind_projection(
                    ProjectionKind::Return,
                    distinct,
                    all,
                    &exprs,
                    &orderby,
                    skip,
                    limit,
                    None,
                    write,
                )
            }
            QueryIR::Call {
                func,
                args,
                yields: vars,
                yield_aliases: aliases,
                filter,
                explicit_yield: yielded,
            } => {
                let args = args
                    .iter()
                    .map(|expr| self.bind_expr(expr))
                    .collect::<Result<Vec<_>, _>>()?;

                // Validate yield field names against procedure outputs
                if yielded && let FnType::Procedure(ref fields) = func.fn_type {
                    let mut seen_fields = std::collections::HashSet::new();
                    for (i, name) in vars.iter().enumerate() {
                        // The actual field name is the alias (original field) if present,
                        // otherwise the yield name itself
                        let field_name = aliases.get(i).and_then(|a| a.as_ref()).unwrap_or(name);
                        if !fields.iter().any(|f| f.as_str() == field_name.as_str()) {
                            return Err(format!(
                                "Unknown yield field '{}' for procedure '{}'",
                                field_name, func.name
                            ));
                        }
                        if !seen_fields.insert(field_name.as_str().to_lowercase()) {
                            return Err(format!(
                                "Duplicate yield field '{}' for procedure '{}'",
                                field_name, func.name
                            ));
                        }
                    }

                    // The fulltext scan procedures are rewritten to dedicated
                    // scan operators in the planner; the rewrite needs the
                    // entity-field slot to bind the matched node/relationship
                    // into. Reject `YIELD score` without the entity field
                    // up-front so the planner doesn't have to handle it.
                    let required_entity = match func.name.as_str() {
                        "db.idx.fulltext.queryNodes" => Some("node"),
                        "db.idx.fulltext.queryRelationships" => Some("relationship"),
                        _ => None,
                    };
                    if let Some(entity) = required_entity {
                        let yielded_entity = vars.iter().enumerate().any(|(i, name)| {
                            let field_name =
                                aliases.get(i).and_then(|a| a.as_ref()).unwrap_or(name);
                            field_name.as_str() == entity
                        });
                        if !yielded_entity {
                            return Err(format!(
                                "Procedure '{}' requires YIELD of '{}'",
                                func.name, entity
                            ));
                        }
                    }
                }

                let mut bound_vars = Vec::with_capacity(vars.len());
                let mut bound_aliases: Vec<Option<Variable>> = Vec::with_capacity(vars.len());
                for (i, name) in vars.into_iter().enumerate() {
                    let alias = aliases.get(i).and_then(std::clone::Clone::clone);
                    if yielded {
                        if let Some(ref original_field) = alias {
                            // YIELD field AS alias: register in scope under the alias name,
                            // but set Variable.name to the original field for procedure map lookup.
                            let var = self.define_name_in_scope(name.clone(), Type::Any, true)?;
                            let var = Variable {
                                name: Some(original_field.clone()),
                                ..var
                            };
                            // Re-insert under alias name with the updated variable
                            self.current_env_mut().insert(name.clone(), var.clone());
                            bound_vars.push(var.clone());
                            // Record alias: a Variable whose name is the alias
                            bound_aliases.push(Some(Variable {
                                name: Some(name),
                                ..var
                            }));
                        } else {
                            bound_vars.push(self.define_name_in_scope(name, Type::Any, true)?);
                            bound_aliases.push(None);
                        }
                    } else {
                        // Create a variable with the original name (for procedure map lookup)
                        // but store it in scope under a `_`-prefixed key so RETURN * doesn't see it.
                        let var = self.fresh_var(
                            Some(name.clone()),
                            Type::Any,
                            self.env_stack.len() as u32 - 1,
                        );
                        let hidden_name = Arc::new(format!("_hidden_{}_{name}", var.id));
                        self.current_env_mut().insert(hidden_name, var.clone());
                        bound_vars.push(var);
                        bound_aliases.push(None);
                    }
                }
                let filter = filter.map(|expr| self.bind_expr(&expr)).transpose()?;
                if let Some(ref f) = filter
                    && !Self::expr_may_return_boolean(f.root())
                {
                    return Err(String::from("Expected boolean predicate"));
                }
                Ok(QueryIR::Call {
                    func,
                    args,
                    yields: bound_vars,
                    yield_aliases: bound_aliases,
                    filter,
                    explicit_yield: yielded,
                })
            }
            QueryIR::ForEach {
                list: list_expr,
                var: var_name,
                body,
            } => {
                let bound_list = self.bind_expr(&list_expr)?;
                // Save the current env so that names defined inside the
                // FOREACH body (loop variable, CREATE'd nodes, etc.) don't
                // leak into subsequent clauses after the FOREACH.
                let saved_env = self.current_env().clone();
                let var = self.define_name_in_scope(var_name, Type::Any, true)?;
                let mut bound_body = Vec::with_capacity(body.len());
                for clause in body {
                    bound_body.push(self.bind_ir(clause)?);
                }
                // Restore the outer scope.
                *self.current_env_mut() = saved_env;
                Ok(QueryIR::ForEach {
                    list: bound_list,
                    var,
                    body: bound_body,
                })
            }
            QueryIR::CallSubquery {
                body, is_returning, ..
            } => self.bind_call_subquery(*body, is_returning),
        }
    }

    fn bind_call_subquery(
        &mut self,
        body: RawQueryIR,
        is_returning: bool,
    ) -> Result<BoundQueryIR, String> {
        // 1. Save outer scope
        let saved_env = self.current_env().clone();
        let saved_env_stack_len = self.env_stack.len();

        // 2. Bind the inner body with scope isolation (also validates import WITH)
        let mut bound_body = self.bind_call_body(body, &saved_env)?;

        // 2b. Apply node labels to the inner body NOW, before the inner
        // scopes are popped.  This prevents scope_id reuse in the outer
        // query from overwriting the inner body's label entries.
        self.update_all_node_labels(&mut bound_body);

        // 2c. Remove inner-scope entries from node_labels so that a
        // subsequent CALL body (which reuses the same scope_id range)
        // doesn't inherit stale labels from this body.
        let inner_scope_ids: Vec<u32> =
            (saved_env_stack_len as u32..self.env_stack.len() as u32).collect();
        self.node_labels
            .retain(|&(scope_id, _), _| !inner_scope_ids.contains(&scope_id));

        // 3. Capture subquery output, restore outer scope
        let subquery_env = self.current_env().clone();
        while self.env_stack.len() > saved_env_stack_len {
            self.env_stack.pop();
        }
        *self.current_env_mut() = saved_env;

        // 4. If returning, check for variable shadowing, allocate outer IDs,
        //    and build inner→outer remapping for the Planner.
        let mut remap = Vec::new();
        if is_returning {
            for name in subquery_env.keys() {
                // Filter out internal slot reservations
                if name.starts_with("_slot_")
                    || name.starts_with("__agg_placeholder_")
                    || name.starts_with("_quant_")
                    || name.starts_with("_lc_")
                    || name.starts_with("_reduce_")
                {
                    continue;
                }
                if self.current_env().contains_key(name) {
                    return Err(format!("Variable `{name}` already declared in outer scope"));
                }
            }
            for (name, inner_var) in &subquery_env {
                // Skip internal slot reservations
                if name.starts_with("_slot_")
                    || name.starts_with("__agg_placeholder_")
                    || name.starts_with("_quant_")
                    || name.starts_with("_lc_")
                    || name.starts_with("_reduce_")
                {
                    continue;
                }
                // Allocate a fresh outer-scope ID for this returned variable
                let outer_var = self.project_name(name, inner_var.ty.clone());
                remap.push((inner_var.clone(), outer_var));
            }
        }

        Ok(QueryIR::CallSubquery {
            body: Box::new(bound_body),
            is_returning,
            remap,
        })
    }

    /// Bind the inner body of a CALL subquery with scope isolation.
    /// For Query bodies: set env to imported vars, bind normally.
    /// For Union bodies: create binders initialized with imported vars for each branch.
    fn bind_call_body(
        &mut self,
        body: RawQueryIR,
        outer_env: &HashMap<Arc<String>, Variable>,
    ) -> Result<BoundQueryIR, String> {
        match body {
            QueryIR::Query { clauses, write } => {
                let has_import = matches!(clauses.first(), Some(QueryIR::With { .. }));

                // Validate and extract imports
                let imported = if has_import {
                    self.validate_import_with(clauses.first().unwrap(), outer_env)?
                } else {
                    HashMap::new()
                };

                let skip_count = usize::from(has_import);
                let mut bound = Vec::with_capacity(clauses.len());

                // Create a new scope for the CALL body (NOT via push_scope —
                // the CALL body is isolated, not a child scope)
                self.env_stack.push(HashMap::new());

                if !imported.is_empty() {
                    // Allocate fresh inner IDs for imported variables and build
                    // projection pairs that map outer → inner.
                    let projections = self.build_import_projections(&imported);

                    // Emit a bound import WITH as the first clause
                    bound.push(QueryIR::With {
                        distinct: false,
                        all: false,
                        exprs: projections,
                        copy_from_parent: vec![],
                        orderby: vec![],
                        skip: None,
                        limit: None,
                        filter: None,
                        write: false,
                    });
                }

                // Bind remaining clauses (skip raw import WITH)
                for clause in clauses.into_iter().skip(skip_count) {
                    bound.push(self.bind_ir(clause)?);
                }
                Ok(QueryIR::Query {
                    clauses: bound,
                    write,
                })
            }
            QueryIR::Union { branches, all } => {
                // For UNION in CALL subquery, each branch is an independent query
                // that may have its own import WITH. Create binders with imported env.
                let mut bound_branches = Vec::with_capacity(branches.len());
                let mut first_columns: Option<Vec<String>> = None;
                let mut last_binder_env: Option<HashMap<Arc<String>, Variable>> = None;
                for branch in branches {
                    let (has_import, imported) = match &branch {
                        QueryIR::Query { clauses, .. } => {
                            if let Some(first) = clauses.first() {
                                if matches!(first, QueryIR::With { .. }) {
                                    (true, self.validate_import_with(first, outer_env)?)
                                } else {
                                    (false, HashMap::new())
                                }
                            } else {
                                (false, HashMap::new())
                            }
                        }
                        _ => (false, HashMap::new()),
                    };
                    let mut binder = Self {
                        env_stack: vec![HashMap::new()],
                        use_parent_scope: false,
                        parent_to_child_scope: HashMap::new(),
                        copy_from_parent: HashMap::new(),
                        node_labels: HashMap::new(),
                    };

                    // Build bound clauses: explicit import WITH + remaining
                    let mut bound = if let QueryIR::Query { clauses, write } = branch {
                        let skip_count = usize::from(has_import);
                        let mut bound_clauses = Vec::with_capacity(clauses.len());

                        if !imported.is_empty() {
                            let projections = binder.build_import_projections(&imported);
                            bound_clauses.push(QueryIR::With {
                                distinct: false,
                                all: false,
                                exprs: projections,
                                copy_from_parent: vec![],
                                orderby: vec![],
                                skip: None,
                                limit: None,
                                filter: None,
                                write: false,
                            });
                        }

                        for clause in clauses.into_iter().skip(skip_count) {
                            bound_clauses.push(binder.bind_ir(clause)?);
                        }
                        QueryIR::Query {
                            clauses: bound_clauses,
                            write,
                        }
                    } else {
                        binder.bind_ir(branch)?
                    };

                    // Post-process this branch's IR with its own accumulated labels
                    binder.update_all_node_labels(&mut bound);
                    let columns = bound.return_column_names();
                    if let Some(ref expected) = first_columns {
                        if columns != *expected {
                            return Err(String::from(
                                "All sub queries in a UNION must have the same column names.",
                            ));
                        }
                    } else {
                        first_columns = Some(columns);
                    }
                    // Capture the binder's final env (RETURN-projected vars)
                    last_binder_env = Some(binder.current_env().clone());
                    bound_branches.push(bound);
                }
                // Set current env to the returned columns from the last branch
                if let Some(env) = last_binder_env {
                    *self.current_env_mut() = env;
                } else {
                    *self.current_env_mut() = HashMap::new();
                }
                Ok(QueryIR::Union {
                    branches: bound_branches,
                    all,
                })
            }
            other => {
                *self.current_env_mut() = HashMap::new();
                self.bind_ir(other)
            }
        }
    }

    /// Build projection pairs that map outer variables to fresh inner variables.
    /// For each imported variable, allocates a fresh inner ID via `project_name`
    /// and creates `(inner_var, ExprIR::Variable(outer_var))` pairs.
    /// Also carries forward `node_labels` for Node-typed imports.
    fn build_import_projections(
        &mut self,
        imported: &HashMap<Arc<String>, Variable>,
    ) -> Vec<(Variable, QueryExpr<Variable>)> {
        let mut projections = Vec::with_capacity(imported.len());
        for (name, outer_var) in imported {
            let inner_var = self.project_name(name, outer_var.ty.clone());
            // Carry forward node_labels for Node-typed imports
            if outer_var.ty == Type::Node
                && let Some(labels) = self
                    .node_labels
                    .get(&(outer_var.scope_id, outer_var.id))
                    .cloned()
            {
                self.node_labels
                    .insert((inner_var.scope_id, inner_var.id), labels);
            }
            let outer_expr = Arc::new(DynTree::new(ExprIR::Variable(outer_var.clone())));
            projections.push((inner_var, outer_expr));
        }
        projections.sort_by(|(a, _), (b, _)| a.name.cmp(&b.name));
        projections
    }

    fn validate_import_with(
        &self,
        with: &RawQueryIR,
        outer_env: &HashMap<Arc<String>, Variable>,
    ) -> Result<HashMap<Arc<String>, Variable>, String> {
        let import_error =
            "WITH imports in CALL {} must consist of only simple references to outside variables";
        if let QueryIR::With {
            distinct,
            all,
            exprs,
            orderby,
            skip,
            limit,
            filter,
            ..
        } = with
        {
            // No ORDER BY, SKIP, LIMIT, WHERE, DISTINCT allowed on import WITH
            if *distinct
                || !orderby.is_empty()
                || skip.is_some()
                || limit.is_some()
                || filter.is_some()
            {
                return Err(import_error.to_string());
            }
            if *all {
                // WITH * — import all outer variables
                return Ok(outer_env.clone());
            }
            let mut imported = HashMap::new();
            for (alias, expr) in exprs {
                // Must be a simple variable reference where alias == variable name
                let root = expr.root();
                if let ExprIR::Variable(var_name) = root.data() {
                    if root.num_children() == 0 && var_name == alias {
                        if let Some(var) = outer_env.get(var_name) {
                            imported.insert(alias.clone(), var.clone());
                        } else {
                            return Err(format!("'{var_name}' not defined"));
                        }
                    } else {
                        return Err(import_error.to_string());
                    }
                } else {
                    return Err(import_error.to_string());
                }
            }
            Ok(imported)
        } else {
            Ok(HashMap::new())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_projection(
        &mut self,
        kind: ProjectionKind,
        distinct: bool,
        all: bool,
        exprs: &[(Arc<String>, QueryExpr<Arc<String>>)],
        orderby: &[(QueryExpr<Arc<String>>, bool)],
        skip: Option<QueryExpr<Arc<String>>>,
        limit: Option<QueryExpr<Arc<String>>>,
        filter: Option<QueryExpr<Arc<String>>>,
        write: bool,
    ) -> Result<QueryIR<Variable>, String> {
        let bound_exprs = exprs
            .iter()
            .map(|(name, expr)| {
                let bound = self.bind_expr(expr)?;
                Self::validate_boolean_operands(&bound)?;
                Ok((name.clone(), bound))
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut projected = Vec::with_capacity(bound_exprs.len());

        // If `all` is true, project all named (non-anonymous) variables from the current env
        if all {
            // Collect the names of explicitly provided projections so we can
            // skip duplicates when expanding *.
            let explicit_names: HashSet<Arc<String>> =
                bound_exprs.iter().map(|(name, _)| name.clone()).collect();

            let env_copy = self
                .current_env()
                .iter()
                .map(|(a, b)| (a.clone(), b.clone()))
                .collect::<Vec<_>>(); // Clone to avoid borrowing issues
            self.push_scope();
            for (name, var) in env_copy {
                // Skip anonymous variables (names starting with '_')
                if name.starts_with('_') {
                    continue;
                }
                // Skip variables that are explicitly listed after *
                if explicit_names.contains(&name) {
                    continue;
                }
                let bound_var = self.project_name(&name, var.ty.clone());
                // Carry forward labels for projected node variables.
                if var.ty == Type::Node
                    && let Some(labels) = self.node_labels.get(&(var.scope_id, var.id)).cloned()
                {
                    self.node_labels
                        .insert((bound_var.scope_id, bound_var.id), labels);
                }
                let expr = Arc::new(DynTree::new(ExprIR::Variable(var.clone())));
                projected.push((bound_var, expr));
            }
            projected.sort_by(|(name_a, _), (name_b, _)| name_a.name.cmp(&name_b.name));

            // Now add the explicit projections after the star-expanded ones
            let mut seen_aliases: HashSet<Arc<String>> = projected
                .iter()
                .filter_map(|(v, _)| v.name.clone())
                .collect();
            for (name, expr) in bound_exprs {
                if !seen_aliases.insert(name.clone()) {
                    return Err(String::from(
                        "Error: Multiple result columns with the same name are not supported.",
                    ));
                }
                let bound_var = self.project_name(&name, Type::Any);
                if let ExprIR::Variable(var_name) = expr.root().data() {
                    self.parent_to_child_scope
                        .insert(var_name.name.as_ref().unwrap().clone(), bound_var.clone());
                    // Carry forward node_labels for projected node variables.
                    if var_name.ty == Type::Node
                        && let Some(labels) = self
                            .node_labels
                            .get(&(var_name.scope_id, var_name.id))
                            .cloned()
                    {
                        self.node_labels
                            .insert((bound_var.scope_id, bound_var.id), labels);
                    }
                }
                projected.push((bound_var, expr));
            }
        } else {
            // Project explicitly listed expressions
            self.push_scope();
            let mut seen_aliases = HashSet::new();
            for (name, expr) in bound_exprs {
                if !seen_aliases.insert(name.clone()) {
                    return Err(String::from(
                        "Error: Multiple result columns with the same name are not supported.",
                    ));
                }
                let bound_var = self.project_name(&name, Type::Any);
                if let ExprIR::Variable(var_name) = expr.root().data() {
                    self.parent_to_child_scope
                        .insert(var_name.name.as_ref().unwrap().clone(), bound_var.clone());
                    // Carry forward node_labels for projected node variables.
                    if var_name.ty == Type::Node
                        && let Some(labels) = self
                            .node_labels
                            .get(&(var_name.scope_id, var_name.id))
                            .cloned()
                    {
                        self.node_labels
                            .insert((bound_var.scope_id, bound_var.id), labels);
                    }
                }
                projected.push((bound_var, expr));
            }
        }

        // Collect variable names used in non-aggregation (group-by) projected
        // expressions.  These variables are allowed in ORDER BY even in an
        // aggregation scope because they form the grouping keys.
        let groupby_var_names: HashSet<Arc<String>> = projected
            .iter()
            .filter(|(_, e)| !e.is_aggregation())
            .flat_map(|(_, e)| {
                e.root()
                    .indices::<Dfs>()
                    .filter_map(|idx| {
                        if let ExprIR::Variable(v) = e.node(idx).data() {
                            v.name.clone()
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // When an ORDER BY expression (or a sub-expression within it) is an
        // aggregation that matches a projected aggregation, rewrite it to
        // reference the projected alias before binding.
        // E.g. RETURN count(x) AS cnt ORDER BY count(x)  →  ORDER BY cnt
        // E.g. RETURN avg(x.age) AS a ORDER BY $p + avg(x.age) - 1000
        //   →  ORDER BY $p + a - 1000
        let orderby: Vec<_> = orderby
            .iter()
            .map(|(expr, desc)| {
                if expr.is_aggregation() {
                    let replaced = replace_agg_subtrees(&expr.root(), exprs);
                    (Arc::new(replaced), *desc)
                } else {
                    (expr.clone(), *desc)
                }
            })
            .collect();

        let orderby = orderby
            .iter()
            .map(|(expr, desc)| Ok((self.bind_expr(expr)?, *desc)))
            .collect::<Result<Vec<_>, String>>()?;

        // Reject aggregation functions inside ORDER BY expressions
        // that don't match any projected aggregation.
        for (expr, _) in &orderby {
            if expr.is_aggregation() {
                return Err(String::from("failed to map aggregation expression"));
            }
        }

        // When the projection contains aggregation(s), ORDER BY must not
        // reference variables that were not explicitly projected.
        // e.g. WITH count(X) AS cnt ORDER BY X  — X is not projected
        // But ORDER BY t.v is fine when RETURN t.v, count(t.v) — t is
        // used in the group-by key expression t.v.
        let has_aggregation = projected.iter().any(|(_, e)| e.is_aggregation());
        if has_aggregation {
            let has_disallowed = self
                .copy_from_parent
                .keys()
                .any(|k| !groupby_var_names.contains(k));
            if has_disallowed {
                return Err(String::from(
                    "ORDER BY cannot reference variables not projected",
                ));
            }
        }

        // Save env keys before binding filter/orderby/skip/limit so we can
        // remove variables that were only added by those clauses.
        let env_keys_before_filter: HashSet<Arc<String>> =
            self.current_env().keys().cloned().collect();

        let skip = skip.map(|expr| self.bind_expr(&expr)).transpose()?;
        let limit = limit.map(|expr| self.bind_expr(&expr)).transpose()?;
        let filter = filter.map(|expr| self.bind_expr(&expr)).transpose()?;
        if let Some(ref f) = filter
            && !Self::expr_may_return_boolean(f.root())
        {
            return Err(String::from("Expected boolean predicate"));
        }

        // Remove from current env any variables added only by filter/orderby/skip/limit.
        // These should be available for the WHERE evaluation at runtime (kept in
        // copy_from_parent) but not visible to subsequent clauses (RETURN *).
        let filter_only_keys: Vec<Arc<String>> = self
            .copy_from_parent
            .keys()
            .filter(|name| !env_keys_before_filter.contains(*name))
            .cloned()
            .collect();
        for key in &filter_only_keys {
            self.current_env_mut().remove(key);
        }

        let copy_from_parent = self
            .copy_from_parent
            .values()
            .cloned()
            .collect::<Vec<(Variable, Variable)>>();
        self.copy_from_parent.clear();
        self.commit_scope();

        Ok(match kind {
            ProjectionKind::With => QueryIR::With {
                distinct,
                all,
                exprs: projected,
                copy_from_parent,
                orderby,
                skip,
                limit,
                filter,
                write,
            },
            ProjectionKind::Return => QueryIR::Return {
                distinct,
                all,
                exprs: projected,
                copy_from_parent,
                orderby,
                skip,
                limit,
                write,
            },
        })
    }

    /// Binds a parsed graph pattern — resolving string variable names to
    /// numeric `Variable` IDs and checking that every referenced name is in
    /// scope.
    ///
    /// The function walks the pattern in two groups (nodes, then
    /// relationships).  For each entity it:
    ///   1. Defines the alias in the current scope via `define_name_in_scope`.
    ///   2. Binds the inline property expression (`attrs`).
    ///
    /// Because the alias is defined *before* its attrs are bound,
    /// self-referential inline properties resolve correctly in MATCH:
    ///
    /// ```cypher
    /// MATCH (a {age: a.age}) RETURN a.age   -- existential filter
    /// ```
    ///
    /// For CREATE / MERGE the callers perform their own pre-validation
    /// (see `bind_graph_create` and the Merge handler) *before* calling
    /// this function, so cross-entity and self-referential accesses are
    /// caught with the appropriate error messages before we get here.
    ///
    /// `is_create` controls `allow_reuse` when defining aliases:
    ///   - `false` (MATCH / MERGE) — an alias may shadow an outer-scope
    ///     definition (e.g. `MATCH (n) MATCH (n)` re-uses `n`).
    ///   - `true`  — redeclaration is an error.
    fn bind_graph(
        &mut self,
        graph: &QueryGraph<Arc<String>, Arc<String>, Arc<String>>,
        is_create: bool,
    ) -> Result<QueryGraph<Arc<String>, Arc<String>, Variable>, String> {
        let mut bound: QueryGraph<Arc<String>, Arc<String>, Variable> = QueryGraph::default();

        // Pre-register path variables in scope so they can be resolved
        // when binding node/relationship inline properties below.
        for raw_path in graph.paths() {
            self.define_name_in_scope(raw_path.var.clone(), Type::Path, true)?;
        }

        // Bind all nodes in the graph, merging duplicates by alias.
        // Note: node aliases are defined BEFORE binding their attrs so that
        // self-referential inline properties resolve correctly, e.g.:
        //   MATCH (a {age: a.age}) RETURN a.age
        // Here a.age acts as an existential filter — matching nodes that have
        // the "age" property.  The alias "a" must be in scope when we bind
        // the attrs expression {age: a.age}.
        for node in graph.nodes() {
            let alias = self.define_name_in_scope(node.alias.clone(), Type::Node, !is_create)?;
            let attrs = self.bind_expr(&node.attrs)?;

            // Accumulate labels across MATCH clauses for this node variable.
            let labels = self
                .node_labels
                .entry((alias.scope_id, alias.id))
                .or_default();
            labels.extend(node.labels.iter().cloned());
            let all_labels = labels.clone();

            let bound_node = Arc::new(QueryNode::new(alias.clone(), all_labels, attrs));
            bound.add_node(bound_node);
        }

        // Bind relationships, binding any referenced nodes that weren't in the graph
        for relationship in graph.relationships() {
            let alias = self.define_name_in_scope(
                relationship.alias.clone(),
                Type::Relationship,
                !is_create,
            )?;
            let attrs = self.bind_expr(&relationship.attrs)?;

            // Resolve 'from' node by alias, binding if missing and merging labels if present
            let from_bound = if let Some(bound_node) = bound
                .nodes()
                .iter()
                .find(|n| n.alias.name.as_ref().unwrap().clone() == relationship.from.alias)
            {
                // Node already bound.  If the relationship endpoint carries
                // additional inline attrs or labels (e.g. reversed pattern
                // `(c:country)<-[:visited]-(f:person)` where `f` was already
                // bound by an earlier hop), create an enriched clone for the
                // relationship's from/to field so the planner can emit filters.
                let has_extra_labels = relationship
                    .from
                    .labels
                    .iter()
                    .any(|l| !bound_node.labels.contains(l));
                let has_extra_attrs = relationship.from.attrs.root().num_children() > 0;
                if has_extra_labels || has_extra_attrs {
                    let from_attrs = self.bind_expr(&relationship.from.attrs)?;
                    let mut merged_labels = bound_node.labels.clone();
                    for l in relationship.from.labels.iter() {
                        if !merged_labels.contains(l) {
                            merged_labels.insert(l.clone());
                        }
                    }
                    Arc::new(QueryNode::new(
                        bound_node.alias.clone(),
                        merged_labels,
                        from_attrs,
                    ))
                } else {
                    bound_node.clone()
                }
            } else {
                let from_alias = self.define_name_in_scope(
                    relationship.from.alias.clone(),
                    Type::Node,
                    !is_create,
                )?;
                let from_attrs = self.bind_expr(&relationship.from.attrs)?;
                let labels = self
                    .node_labels
                    .entry((from_alias.scope_id, from_alias.id))
                    .or_default();
                labels.extend(relationship.from.labels.iter().cloned());
                let all_labels = labels.clone();
                let bound_node =
                    Arc::new(QueryNode::new(from_alias.clone(), all_labels, from_attrs));
                bound.add_node(bound_node.clone());
                bound_node
            };

            // Resolve 'to' node by alias, binding if missing and merging labels if present
            let to_bound = if let Some(bound_node) = bound
                .nodes()
                .iter()
                .find(|n| n.alias.name.as_ref().unwrap().clone() == relationship.to.alias)
            {
                let has_extra_labels = relationship
                    .to
                    .labels
                    .iter()
                    .any(|l| !bound_node.labels.contains(l));
                let has_extra_attrs = relationship.to.attrs.root().num_children() > 0;
                if has_extra_labels || has_extra_attrs {
                    let to_attrs = self.bind_expr(&relationship.to.attrs)?;
                    let mut merged_labels = bound_node.labels.clone();
                    for l in relationship.to.labels.iter() {
                        if !merged_labels.contains(l) {
                            merged_labels.insert(l.clone());
                        }
                    }
                    Arc::new(QueryNode::new(
                        bound_node.alias.clone(),
                        merged_labels,
                        to_attrs,
                    ))
                } else {
                    bound_node.clone()
                }
            } else {
                let to_alias = self.define_name_in_scope(
                    relationship.to.alias.clone(),
                    Type::Node,
                    !is_create,
                )?;
                let to_attrs = self.bind_expr(&relationship.to.attrs)?;
                let labels = self
                    .node_labels
                    .entry((to_alias.scope_id, to_alias.id))
                    .or_default();
                labels.extend(relationship.to.labels.iter().cloned());
                let all_labels = labels.clone();
                let bound_node = Arc::new(QueryNode::new(to_alias.clone(), all_labels, to_attrs));
                bound.add_node(bound_node.clone());
                bound_node
            };

            // Add to bound graph if not already there
            // Nodes already added via alias_to_bound; no duplicate insert needed

            let mut new_rel = QueryRelationship::new(
                alias.clone(),
                relationship.types.clone(),
                attrs,
                from_bound,
                to_bound,
                relationship.bidirectional,
                relationship.min_hops,
                relationship.max_hops,
            );
            new_rel.all_shortest_paths = relationship.all_shortest_paths;
            let rel = Arc::new(new_rel);
            bound.add_relationship(rel);
        }

        // Bind paths - path vars reference entities by name.
        // Stub paths (empty vars) come from pattern comprehension: derive
        // the ordered element list from the bound relationships.
        for raw_path in graph.paths() {
            let alias = self.define_name_in_scope(raw_path.var.clone(), Type::Path, true)?;

            let vars = if raw_path.vars.is_empty() {
                // Pattern comprehension stub: derive from bound relationships
                let mut v = Vec::new();
                if let Some(first_rel) = bound.relationships().first() {
                    v.push(first_rel.from.alias.clone());
                    for rel in bound.relationships() {
                        v.push(rel.alias.clone());
                        v.push(rel.to.alias.clone());
                    }
                }
                v
            } else {
                // Regular MATCH path: resolve by name
                let mut v = Vec::with_capacity(raw_path.vars.len());
                for name in &raw_path.vars {
                    if let Some(var) = self.current_env().get(name) {
                        v.push(var.clone());
                    } else {
                        v.push(self.resolve_name(name, &[])?);
                    }
                }
                v
            };
            bound.add_path(Arc::new(QueryPath::new(alias, vars)));
        }

        Ok(bound)
    }

    fn bind_graph_create(
        &mut self,
        pattern: &QueryGraph<Arc<String>, Arc<String>, Arc<String>>,
    ) -> Result<QueryGraph<Arc<String>, Arc<String>, Variable>, String> {
        // For CREATE clause validation: we need to detect when a variable is declared
        // multiple times OR when it shadows a previous declaration.
        //
        // Key insight: The parser includes ALL nodes in pattern.nodes(), including those
        // that only appear as endpoints of relationships. We need to distinguish:
        // - Bare node patterns: (a) - should be validated for redeclaration
        // - Endpoint-only nodes: x in (x)-[:R]->(y) - should NOT be validated (it's a reference)
        //
        // Strategy: A node should only be validated if it's NOT an endpoint of any relationship.
        // If a node ONLY appears as relationship endpoints, it's being referenced, not declared.

        // Collect all node aliases that are endpoints of relationships
        let mut endpoint_aliases = HashSet::new();
        for rel in pattern.relationships() {
            endpoint_aliases.insert(rel.from.alias.clone());
            endpoint_aliases.insert(rel.to.alias.clone());
        }

        // Track which names we've already processed in this CREATE clause
        let mut defined_in_create = HashSet::new();

        // Validate nodes - only check nodes that are NOT purely endpoints of relationships
        for node in pattern.nodes() {
            if node.alias.starts_with("_anon") {
                continue; // Skip anonymous nodes
            }

            // If this node is ONLY an endpoint (not a bare pattern), skip validation
            // A node is "only an endpoint" if:
            // 1. It has no labels and no properties (standard bare node)
            // 2. It appears in endpoint_aliases
            // Actually, we should check: is there a bare pattern for this node?
            // The parser adds nodes when they're first encountered, either as:
            // - Bare pattern: (a)
            // - Relationship endpoint: (a) in (a)-[:R]->()
            //
            // We can't distinguish these just from the node. But we can use the fact that
            // if a node appears in a relationship, the relationship reference is likely
            // what added it, not a bare pattern.
            //
            // Actually, let me reconsider: the parser is called with the pattern as it appears
            // in the CREATE clause. If the user wrote (x)-[:R]->(y), then x and y in endpoints
            // are the FIRST and ONLY occurrence of x and y in the parsed pattern.
            // They should NOT be validated against env because they're references to matched vars.
            //
            // So the rule is: if a node is an endpoint of ANY relationship, treat it as a reference
            // and don't validate it against env.
            if endpoint_aliases.contains(&node.alias) {
                // This node is an endpoint - it's a reference, not a new declaration
                defined_in_create.insert(node.alias.clone());
                continue;
            }

            // This is a bare node pattern (not an endpoint)
            // Check if this is the first occurrence in CREATE of this name
            if !defined_in_create.contains(&node.alias) {
                // Check if it shadows an existing variable from a previous clause
                if self.current_env().contains_key(&node.alias) {
                    return Err(format!(
                        "The bound variable '{}' can't be redeclared in a CREATE clause",
                        node.alias
                    ));
                }
                // Mark it as defined in this clause
                defined_in_create.insert(node.alias.clone());
            }
        }

        // Check relationships for redeclaration
        for rel in pattern.relationships() {
            if rel.alias.starts_with("_anon") {
                continue; // Skip anonymous relationships
            }

            // If this is the first occurrence in CREATE of this name
            if !defined_in_create.contains(&rel.alias) {
                // Check if it shadows an existing variable from a previous clause
                if self.current_env().contains_key(&rel.alias) {
                    return Err(format!(
                        "The bound variable '{}' can't be redeclared in a CREATE clause",
                        rel.alias
                    ));
                }
                // Mark it as defined in this clause
                defined_in_create.insert(rel.alias.clone());
            }
            // If it's a subsequent occurrence, it's allowed (it's a reference)
        }

        // Reject self-referential property access BEFORE binding.
        // A node being created has no properties yet, so referencing its own
        // attributes is invalid, e.g.:
        //   CREATE (a:L {v: a.v})   → "undefined attribute" (a.v during creation)
        // Cross-node references are checked separately below.
        for node in pattern.nodes() {
            Self::check_unbound_self_referential(&node.alias, &node.attrs)?;
        }
        for rel in pattern.relationships() {
            Self::check_unbound_self_referential(&rel.alias, &rel.attrs)?;
        }

        // Validate that inline attrs don't reference other entities being
        // created in the same CREATE clause.  Entity aliases are not yet in
        // scope, so any such reference produces "'x' not defined", e.g.:
        //   CREATE (a {v:1}), (z {v:a.v+2})   → "'a' not defined"
        for node in pattern.nodes() {
            self.bind_expr(&node.attrs)?;
        }
        for rel in pattern.relationships() {
            self.bind_expr(&rel.attrs)?;
        }

        // Bind the pattern - allow reuse since we've already validated
        self.bind_graph(pattern, false)
    }

    /// Checks whether an entity's unbound inline attrs contain a property
    /// access on the entity's own alias.  Used to reject self-referential
    /// property access in CREATE clauses before the general binding pass,
    /// e.g.:
    ///   CREATE (a:L {v: a.v})  → Attempted to access undefined attribute
    fn check_unbound_self_referential(
        entity_alias: &Arc<String>,
        attrs: &QueryExpr<Arc<String>>,
    ) -> Result<(), String> {
        if entity_alias.starts_with("_anon") {
            return Ok(());
        }
        for idx in attrs.root().indices::<Dfs>() {
            if let ExprIR::Property(_) = attrs.node(idx).data()
                && let ExprIR::Variable(var_name) = attrs.node(idx).child(0).data()
                && var_name == entity_alias
            {
                return Err(format!("'{entity_alias}' not defined; undefined attribute"));
            }
        }
        Ok(())
    }

    fn bind_set_items(
        &mut self,
        items: Vec<SetItem<Arc<String>, Arc<String>>>,
    ) -> Result<Vec<SetItem<Arc<String>, Variable>>, String> {
        let mut res = Vec::with_capacity(items.len());
        for item in items {
            match item {
                SetItem::Attribute {
                    target,
                    value,
                    replace: strict,
                } => res.push(SetItem::Attribute {
                    target: self.bind_expr(&target)?,
                    value: self.bind_expr(&value)?,
                    replace: strict,
                }),
                SetItem::Label { var: name, labels } => {
                    let var = self.resolve_name(&name, &[])?;
                    res.push(SetItem::Label { var, labels });
                }
            }
        }
        Ok(res)
    }

    fn bind_expr(
        &mut self,
        expr: &QueryExpr<Arc<String>>,
    ) -> Result<QueryExpr<Variable>, String> {
        self.bind_expr_with_locals(expr, &mut vec![])
    }

    fn bind_expr_with_locals(
        &mut self,
        expr: &QueryExpr<Arc<String>>,
        locals: &mut Vec<HashMap<Arc<String>, Variable>>,
    ) -> Result<QueryExpr<Variable>, String> {
        let root = expr.root();
        Ok(Arc::new(self.bind_expr_node(expr, &root, locals)?))
    }

    #[allow(
        clippy::only_used_in_recursion,
        clippy::too_many_lines,
        clippy::needless_pass_by_value
    )]
    fn bind_expr_node(
        &mut self,
        expr: &DynTree<ExprIR<Arc<String>>>,
        node_ref: &DynNode<ExprIR<Arc<String>>>,
        locals: &mut Vec<HashMap<Arc<String>, Variable>>,
    ) -> Result<DynTree<ExprIR<Variable>>, String> {
        match node_ref.data() {
            ExprIR::Variable(name) => {
                let var = self.resolve_name(name, locals)?;
                Ok(tree!(ExprIR::Variable(var)))
            }
            ExprIR::Quantifier {
                quantifier_type: qt,
                var: name,
            } => {
                let scope_id = self.env_stack.len() as u32 - 1;
                let bound_var: Variable = self.fresh_var(Some(name.clone()), Type::Any, scope_id);
                // Reserve the ID slot in env_stack with a unique key.
                let key = Arc::new(format!("_quant_{}_{}", scope_id, bound_var.id));
                self.env_stack[scope_id as usize].insert(key, bound_var.clone());

                // Bind child[0] (the iterable) BEFORE introducing the
                // quantifier variable so it resolves in the outer scope.
                let mut children_iter = node_ref.children();
                let iterable_child = children_iter.next().unwrap();
                let bound_iterable = self.bind_expr_node(expr, &iterable_child, locals)?;

                let mut local = HashMap::new();
                local.insert(name.clone(), bound_var.clone());
                locals.push(local);
                let rest_children = children_iter
                    .map(|child| self.bind_expr_node(expr, &child, locals))
                    .collect::<Result<Vec<_>, _>>()?;
                locals.pop();

                let mut children = vec![bound_iterable];
                children.extend(rest_children);

                let mut new_tree = DynTree::new(ExprIR::Quantifier {
                    quantifier_type: qt.clone(),
                    var: bound_var,
                });
                let mut root = new_tree.root_mut();
                for child in children {
                    root.push_child_tree(child);
                }
                Ok(new_tree)
            }
            ExprIR::ListComprehension(name) => {
                let scope_id = self.env_stack.len() as u32 - 1;
                let bound_var: Variable = self.fresh_var(Some(name.clone()), Type::Any, scope_id);
                // Reserve the ID slot in env_stack with a unique key.
                let key = Arc::new(format!("_lc_{}_{}", scope_id, bound_var.id));
                self.env_stack[scope_id as usize].insert(key, bound_var.clone());

                // Bind child[0] (the iterable) BEFORE introducing the
                // comprehension variable so that e.g. `[x IN nodes(x) | ...]`
                // resolves the iterable's `x` to the outer-scope path, not
                // to the comprehension variable.
                let mut children_iter = node_ref.children();
                let iterable_child = children_iter.next().unwrap();
                let bound_iterable = self.bind_expr_node(expr, &iterable_child, locals)?;

                let mut local = HashMap::new();
                local.insert(name.clone(), bound_var.clone());
                locals.push(local);
                let rest_children = children_iter
                    .map(|child| self.bind_expr_node(expr, &child, locals))
                    .collect::<Result<Vec<_>, _>>()?;
                locals.pop();

                let mut children = vec![bound_iterable];
                children.extend(rest_children);

                // Child 1 is the WHERE condition — validate it returns boolean
                if children.len() > 1 && !Self::expr_may_return_boolean(children[1].root()) {
                    return Err(String::from("Expected boolean predicate"));
                }

                let mut new_tree = DynTree::new(ExprIR::ListComprehension(bound_var));
                let mut root = new_tree.root_mut();
                for child in children {
                    root.push_child_tree(child);
                }
                Ok(new_tree)
            }
            ExprIR::Reduce {
                accumulator: acc_name,
                iterator: iter_name,
            } => {
                if acc_name == iter_name {
                    return Err(format!("Variable `{acc_name}` already declared"));
                }
                let scope_id = self.env_stack.len() as u32 - 1;
                let bound_acc: Variable =
                    self.fresh_var(Some(acc_name.clone()), Type::Any, scope_id);
                let key = Arc::new(format!("_reduce_acc_{}_{}", scope_id, bound_acc.id));
                self.env_stack[scope_id as usize].insert(key, bound_acc.clone());
                let bound_iter: Variable =
                    self.fresh_var(Some(iter_name.clone()), Type::Any, scope_id);
                let key = Arc::new(format!("_reduce_iter_{}_{}", scope_id, bound_iter.id));
                self.env_stack[scope_id as usize].insert(key, bound_iter.clone());

                let mut children_iter = node_ref.children();

                // child[0] = init expression — bind in outer scope
                let init_child = children_iter.next().unwrap();
                let bound_init = self.bind_expr_node(expr, &init_child, locals)?;

                // child[1] = list expression — bind in outer scope
                let list_child = children_iter.next().unwrap();
                let bound_list = self.bind_expr_node(expr, &list_child, locals)?;

                // child[2] = body expression — bind with acc + iter in scope
                let body_child = children_iter.next().unwrap();
                let mut local = HashMap::new();
                local.insert(acc_name.clone(), bound_acc.clone());
                local.insert(iter_name.clone(), bound_iter.clone());
                locals.push(local);
                let bound_body = self.bind_expr_node(expr, &body_child, locals)?;
                locals.pop();

                let mut new_tree = DynTree::new(ExprIR::Reduce {
                    accumulator: bound_acc,
                    iterator: bound_iter,
                });
                let mut root = new_tree.root_mut();
                root.push_child_tree(bound_init);
                root.push_child_tree(bound_list);
                root.push_child_tree(bound_body);
                Ok(new_tree)
            }
            ExprIR::PatternComprehension(graph) => {
                // Snapshot outer scope so pattern-local aliases can be
                // cleaned up after binding (they must not leak outward).
                let outer_scope_names: HashSet<Arc<String>> =
                    self.current_env().keys().cloned().collect();

                // Temporarily inject local comprehension variables into
                // the current env so bind_graph can resolve them (e.g.
                // a list comprehension variable used in the pattern).
                for scope in locals.iter() {
                    for (name, var) in scope {
                        self.current_env_mut().insert(name.clone(), var.clone());
                    }
                }

                // bind_graph uses define_name_in_scope which reuses
                // outer-scope variables (e.g. 'n' from MATCH) and creates
                // fresh variables only for new aliases (anonymous nodes/rels).
                let bound_graph = self.bind_graph(graph, false)?;

                let children = node_ref
                    .children()
                    .map(|child| self.bind_expr_node(expr, &child, locals))
                    .collect::<Result<Vec<_>, _>>()?;

                // Remove pattern-local aliases so they don't leak into outer scope.
                // Keep anonymous variables (_anon_*) since they are always created
                // fresh and their IDs must be visible to scope_vars so the planner
                // can avoid ID collisions.
                self.current_env_mut().retain(|name, _| {
                    outer_scope_names.contains(name) || name.starts_with("_anon")
                });

                let mut new_tree = DynTree::new(ExprIR::PatternComprehension(bound_graph));
                let mut root = new_tree.root_mut();
                for child in children {
                    root.push_child_tree(child);
                }
                Ok(new_tree)
            }
            _ => {
                // For ShortestPath, wrap child binding errors as "requires bound nodes"
                if let ExprIR::ShortestPath { all_paths, .. } = node_ref.data() {
                    let fn_name = if *all_paths {
                        "allShortestPaths"
                    } else {
                        "shortestPath"
                    };
                    let children = node_ref
                        .children()
                        .map(|child| self.bind_expr_node(expr, &child, locals))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| format!("A {fn_name} requires bound nodes"))?;
                    let ExprIR::ShortestPath {
                        rel_types,
                        min_hops,
                        max_hops,
                        directed,
                        all_paths,
                    } = node_ref.data().clone()
                    else {
                        unreachable!();
                    };
                    let mut new_tree = DynTree::new(ExprIR::ShortestPath {
                        rel_types,
                        min_hops,
                        max_hops,
                        directed,
                        all_paths,
                    });
                    let mut root = new_tree.root_mut();
                    for child in children {
                        root.push_child_tree(child);
                    }
                    return Ok(new_tree);
                }

                let children = node_ref
                    .children()
                    .map(|child| self.bind_expr_node(expr, &child, locals))
                    .collect::<Result<Vec<_>, _>>()?;

                let new_data = match node_ref.data().clone() {
                    ExprIR::Null => ExprIR::Null,
                    ExprIR::Bool(b) => ExprIR::Bool(b),
                    ExprIR::Integer(i) => ExprIR::Integer(i),
                    ExprIR::Float(fl) => ExprIR::Float(fl),
                    ExprIR::String(s) => ExprIR::String(s),
                    ExprIR::List => ExprIR::List,
                    ExprIR::Map => ExprIR::Map,
                    ExprIR::MapProjection => ExprIR::MapProjection,
                    ExprIR::Parameter(p) => ExprIR::Parameter(p),
                    ExprIR::Length => ExprIR::Length,
                    ExprIR::GetElement => ExprIR::GetElement,
                    ExprIR::GetElements => ExprIR::GetElements,
                    ExprIR::IsNode => ExprIR::IsNode,
                    ExprIR::IsRelationship => ExprIR::IsRelationship,
                    ExprIR::Or => ExprIR::Or,
                    ExprIR::Xor => ExprIR::Xor,
                    ExprIR::And => ExprIR::And,
                    ExprIR::Not => ExprIR::Not,
                    ExprIR::Negate => ExprIR::Negate,
                    ExprIR::Eq => ExprIR::Eq,
                    ExprIR::Neq => ExprIR::Neq,
                    ExprIR::Lt => ExprIR::Lt,
                    ExprIR::Gt => ExprIR::Gt,
                    ExprIR::Le => ExprIR::Le,
                    ExprIR::Ge => ExprIR::Ge,
                    ExprIR::In => ExprIR::In,
                    ExprIR::Add => ExprIR::Add,
                    ExprIR::Sub => ExprIR::Sub,
                    ExprIR::Mul => ExprIR::Mul,
                    ExprIR::Div => ExprIR::Div,
                    ExprIR::Pow => ExprIR::Pow,
                    ExprIR::Modulo => ExprIR::Modulo,
                    ExprIR::Distinct => ExprIR::Distinct,
                    ExprIR::Property(prop) => {
                        // Property access is not valid on Path types.
                        if let Some(first_child) = children.first() {
                            let root = first_child.root();
                            let inner = if matches!(root.data(), ExprIR::Paren) {
                                root.get_child(0)
                            } else {
                                Some(root)
                            };
                            if inner.is_some_and(
                                |n| matches!(n.data(), ExprIR::Variable(v) if v.ty == Type::Path),
                            ) {
                                return Err("Type mismatch: expected Map, Node, Edge, \
                                             Datetime, Date, Time, Duration, Null, \
                                             or Point but was Path"
                                    .to_string());
                            }
                        }
                        ExprIR::Property(prop)
                    }
                    ExprIR::FuncInvocation(func) => {
                        // Compile-time type check: validate argument types
                        // against the function's declared parameter types.
                        // Skip hasLabels — it is generated internally by the parser
                        // for SET/REMOVE label operations, whose runtime operators
                        // handle type checking with more precise error messages.
                        if func.name != "hasLabels"
                            && let FnArguments::Fixed(arg_types) = &func.args_type
                        {
                            for (i, expected_ty) in arg_types.iter().enumerate() {
                                if let Some(child) = children.get(i)
                                    && let ExprIR::Variable(var) = child.root().data()
                                    && !var.ty.is_compatible_with(expected_ty)
                                {
                                    return Err(format!(
                                        "Type mismatch: expected {expected_ty} but was {}",
                                        var.ty
                                    ));
                                }
                            }
                        }
                        ExprIR::FuncInvocation(func)
                    }
                    ExprIR::Paren => ExprIR::Paren,
                    ExprIR::ShortestPath {
                        rel_types,
                        min_hops,
                        max_hops,
                        directed,
                        all_paths,
                    } => {
                        // Verify children (source/dest vars) are bound
                        for child in &children {
                            if let ExprIR::Variable(var) = child.root().data()
                                && var.name.is_none()
                            {
                                let fn_name = if all_paths {
                                    "allShortestPaths"
                                } else {
                                    "shortestPath"
                                };
                                return Err(format!("A {fn_name} requires bound nodes"));
                            }
                        }
                        ExprIR::ShortestPath {
                            rel_types,
                            min_hops,
                            max_hops,
                            directed,
                            all_paths,
                        }
                    }
                    ExprIR::Variable(_)
                    | ExprIR::Quantifier { .. }
                    | ExprIR::ListComprehension(_)
                    | ExprIR::Reduce { .. }
                    | ExprIR::PatternComprehension(_) => unreachable!("handled above"),
                    ExprIR::Pattern(pattern) => {
                        // Snapshot outer scope so pattern-local aliases can be
                        // cleaned up after binding (they must not leak outward).
                        let outer_scope_names: HashSet<Arc<String>> =
                            self.current_env().keys().cloned().collect();

                        // Temporarily inject local comprehension variables into
                        // the current env so bind_graph can resolve them.
                        for scope in locals.iter() {
                            for (name, var) in scope {
                                self.current_env_mut().insert(name.clone(), var.clone());
                            }
                        }

                        // Save node_labels so labels from pattern predicates
                        // (e.g. in WHERE clause OR branches) don't leak into
                        // the outer MATCH pattern's label accumulation.
                        let saved_labels = self.node_labels.clone();
                        let result = self.bind_graph(&pattern, false);
                        self.node_labels = saved_labels;

                        // Remove pattern-local aliases so they don't leak into
                        // the outer scope.  Keep anonymous variables (_anon_*)
                        // since their IDs must remain visible in scope_vars.
                        self.current_env_mut().retain(|name, _| {
                            outer_scope_names.contains(name) || name.starts_with("_anon")
                        });

                        ExprIR::Pattern(result?)
                    }
                };
                let mut new_tree = DynTree::new(new_data);
                let mut root = new_tree.root_mut();
                for child in children {
                    root.push_child_tree(child);
                }
                Ok(new_tree)
            }
        }
    }

    fn define_name_in_scope(
        &mut self,
        name: Arc<String>,
        ty: Type,
        allow_reuse: bool,
    ) -> Result<Variable, String> {
        // Anonymous variables (_anon_*) should always be fresh, never reused
        if !name.starts_with("_anon")
            && let Some(existing) = self.current_env().get(&name)
        {
            if !allow_reuse {
                return Err(format!("Variable `{name}` already declared"));
            }
            Self::ensure_type(existing, &ty)?;
            return Ok(existing.clone());
        }

        let var = self.fresh_var(Some(name.clone()), ty, self.env_stack.len() as u32 - 1);
        self.current_env_mut().insert(name, var.clone());
        Ok(var)
    }

    fn project_name(
        &mut self,
        name: &Arc<String>,
        ty: Type,
    ) -> Variable {
        let var = self.fresh_var(Some(name.clone()), ty, self.env_stack.len() as u32 - 1);
        self.current_env_mut().insert(name.clone(), var.clone());
        var
    }

    fn resolve_name(
        &mut self,
        name: &Arc<String>,
        locals: &[HashMap<Arc<String>, Variable>],
    ) -> Result<Variable, String> {
        // Special placeholder for aggregate function ordering - always create a fresh variable
        if name.as_str() == "__agg_order_by_placeholder__" {
            let scope_id = self.env_stack.len() as u32 - 1;
            let var = self.fresh_var(Some(name.clone()), Type::Any, scope_id);
            // Insert with a unique key so each placeholder occupies its own
            // slot in the env (keeping env.len() == next available ID).
            let key = Arc::new(format!("__agg_placeholder_{}", var.id));
            self.env_stack[scope_id as usize].insert(key, var.clone());
            return Ok(var);
        }

        // Anonymous variables should also create fresh variables
        if name.starts_with("_anon") {
            let scope_id = self.env_stack.len() as u32 - 1;
            let var = self.fresh_var(Some(name.clone()), Type::Any, scope_id);
            // Insert with the anon name (unique by construction) to reserve
            // the ID slot.
            self.current_env_mut().insert(name.clone(), var.clone());
            return Ok(var);
        }

        for scope in locals.iter().rev() {
            if let Some(var) = scope.get(name) {
                return Ok(var.clone());
            }
        }

        if let Some(var) = self.current_env().get(name) {
            return Ok(var.clone());
        }

        if self.use_parent_scope {
            if let Some(var) = self.parent_to_child_scope.get(name) {
                return Ok(var.clone());
            } else if let Some((_, var)) = self.copy_from_parent.get(name) {
                return Ok(var.clone());
            } else if let Some(var) = self
                .env_stack
                .get(self.env_stack.len().saturating_sub(2))
                .and_then(|parent_env| parent_env.get(name))
                .cloned()
            {
                // Copy variable from parent scope into current scope
                let current_scope_id = self.env_stack.len() as u32 - 1;
                let var_id = self.env_stack[current_scope_id as usize].len() as u32;

                let copied_var = Variable {
                    name: var.name.clone(),
                    id: var_id,
                    scope_id: current_scope_id,
                    ty: var.ty.clone(),
                };
                self.current_env_mut()
                    .insert(name.clone(), copied_var.clone());
                self.copy_from_parent
                    .insert(name.clone(), (var, copied_var.clone()));
                return Ok(copied_var);
            }
        }

        Err(format!("'{}' not defined", name.as_str()))
    }

    fn fresh_var(
        &self,
        name: Option<Arc<String>>,
        ty: Type,
        scope_id: u32,
    ) -> Variable {
        let var_id = self.env_stack[scope_id as usize].len() as u32;

        Variable {
            name,
            id: var_id,
            scope_id,
            ty,
        }
    }

    fn ensure_type(
        existing: &Variable,
        ty: &Type,
    ) -> Result<(), String> {
        if (ty == &Type::Relationship && (existing.ty == Type::Node || existing.ty == Type::Path))
            || (ty == &Type::Node
                && (existing.ty == Type::Relationship || existing.ty == Type::Path))
            || (ty == &Type::Path
                && (existing.ty == Type::Node || existing.ty == Type::Relationship))
        {
            return Err(format!(
                "The alias '{}' was specified for both a node and a relationship.",
                existing.as_str()
            ));
        }
        Ok(())
    }

    /// Walks an expression tree and validates that every AND/OR/XOR/NOT node
    /// has operands that can return boolean.  Produces a "Type mismatch"
    /// error for non-filter contexts (e.g. RETURN expressions).
    fn validate_boolean_operands(expr: &QueryExpr<Variable>) -> Result<(), String> {
        Self::validate_boolean_operands_impl(&expr.root())
    }

    fn validate_boolean_operands_impl(node: &DynNode<ExprIR<Variable>>) -> Result<(), String> {
        if matches!(
            node.data(),
            ExprIR::And | ExprIR::Or | ExprIR::Xor | ExprIR::Not
        ) {
            for child in node.children() {
                if !Self::expr_may_return_boolean(child) {
                    return Err(String::from("Type mismatch: expected Boolean"));
                }
            }
        }
        for child in node.children() {
            Self::validate_boolean_operands_impl(&child)?;
        }
        Ok(())
    }

    /// Recursively determines whether an expression tree node can return a
    /// boolean value at compile time. For nodes whose type cannot be
    /// determined statically (variables, parameters, properties), returns
    /// `true` (deferring to the runtime check).
    #[allow(clippy::needless_pass_by_value)]
    fn expr_may_return_boolean(node: DynNode<ExprIR<Variable>>) -> bool {
        match node.data() {
            // Logical operators – result is boolean, but each child must
            // also return boolean (mirrors the C recursive FilterTree_Valid)
            ExprIR::Or | ExprIR::And | ExprIR::Xor | ExprIR::Not => {
                node.children().all(Self::expr_may_return_boolean)
            }

            // Transparent wrappers – recurse into the single child
            ExprIR::Paren | ExprIR::Distinct => {
                node.get_child(0).is_some_and(Self::expr_may_return_boolean)
            }

            // Function calls – use the registered return type
            ExprIR::FuncInvocation(func) => func.ret_type.can_return_boolean(),

            // Non-boolean: literals, unary arithmetic, subscript, list comprehension
            ExprIR::Integer(_)
            | ExprIR::Float(_)
            | ExprIR::String(_)
            | ExprIR::List
            | ExprIR::Map
            | ExprIR::MapProjection
            | ExprIR::Negate
            | ExprIR::Length
            | ExprIR::GetElement
            | ExprIR::GetElements
            | ExprIR::ListComprehension(_)
            | ExprIR::PatternComprehension(_) => false,

            // Boolean literals, comparisons, predicates, and runtime-typed nodes
            ExprIR::Bool(_)
            | ExprIR::Null
            | ExprIR::Eq
            | ExprIR::Neq
            | ExprIR::Lt
            | ExprIR::Gt
            | ExprIR::Le
            | ExprIR::Ge
            | ExprIR::In
            | ExprIR::Add
            | ExprIR::Sub
            | ExprIR::Mul
            | ExprIR::Div
            | ExprIR::Pow
            | ExprIR::Modulo
            | ExprIR::IsNode
            | ExprIR::IsRelationship
            | ExprIR::Quantifier { .. }
            | ExprIR::Reduce { .. }
            | ExprIR::Variable(_)
            | ExprIR::Parameter(_)
            | ExprIR::Property(_)
            | ExprIR::ShortestPath { .. }
            | ExprIR::Pattern(_) => true,
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn expr_may_return_entity(node: DynNode<ExprIR<Variable>>) -> bool {
        match node.data() {
            // Variables – check the resolved type
            ExprIR::Variable(var) => var.ty.can_return_entity(),

            // Function calls – check the return type
            ExprIR::FuncInvocation(func) => func.ret_type.can_return_entity(),

            // Transparent wrappers – recurse into the single child
            ExprIR::Paren | ExprIR::Distinct => {
                node.get_child(0).is_some_and(Self::expr_may_return_entity)
            }

            // Subscript and property access could produce entities at runtime
            ExprIR::GetElement
            | ExprIR::Property(_)
            | ExprIR::Parameter(_)
            | ExprIR::Null
            | ExprIR::Reduce { .. } => true,

            // Everything else cannot produce a graph entity
            ExprIR::Integer(_)
            | ExprIR::Float(_)
            | ExprIR::String(_)
            | ExprIR::Bool(_)
            | ExprIR::List
            | ExprIR::Map
            | ExprIR::MapProjection
            | ExprIR::Negate
            | ExprIR::Length
            | ExprIR::GetElements
            | ExprIR::ListComprehension(_)
            | ExprIR::PatternComprehension(_)
            | ExprIR::Or
            | ExprIR::And
            | ExprIR::Xor
            | ExprIR::Not
            | ExprIR::Eq
            | ExprIR::Neq
            | ExprIR::Lt
            | ExprIR::Gt
            | ExprIR::Le
            | ExprIR::Ge
            | ExprIR::In
            | ExprIR::Add
            | ExprIR::Sub
            | ExprIR::Mul
            | ExprIR::Div
            | ExprIR::Pow
            | ExprIR::Modulo
            | ExprIR::IsNode
            | ExprIR::IsRelationship
            | ExprIR::Quantifier { .. }
            | ExprIR::ShortestPath { .. }
            | ExprIR::Pattern(_) => false,
        }
    }
}

/// Recursively walk an ORDER BY expression tree and replace any aggregation
/// sub-tree that structurally matches a projected aggregation with a variable
/// reference to the projection alias.
fn replace_agg_subtrees(
    node: &DynNode<ExprIR<Arc<String>>>,
    exprs: &[(Arc<String>, QueryExpr<Arc<String>>)],
) -> DynTree<ExprIR<Arc<String>>> {
    // If this node is an aggregate function, check if the subtree rooted here
    // matches any projected aggregation.
    if let ExprIR::FuncInvocation(func) = node.data()
        && func.is_aggregate()
    {
        let subtree = node.clone_as_tree();
        for (name, proj_expr) in exprs {
            if proj_expr.is_aggregation() && raw_exprs_structurally_equal(&subtree, proj_expr) {
                return DynTree::new(ExprIR::Variable(name.clone()));
            }
        }
    }

    // Not a matching aggregation — reconstruct this node with recursively
    // processed children.
    let mut new_tree = DynTree::new(node.data().clone());
    for child in node.children() {
        let child_tree = replace_agg_subtrees(&child, exprs);
        new_tree.root_mut().push_child_tree(child_tree);
    }
    new_tree
}

/// Compare two raw (pre-bind) expression trees structurally, ignoring
/// internal `__agg_order_by_placeholder__` variables that the parser
/// appends to aggregate functions.
fn raw_exprs_structurally_equal(
    a: &DynTree<ExprIR<Arc<String>>>,
    b: &DynTree<ExprIR<Arc<String>>>,
) -> bool {
    fn is_agg_placeholder(node: &ExprIR<Arc<String>>) -> bool {
        matches!(node, ExprIR::Variable(v) if v.as_str() == "__agg_order_by_placeholder__")
    }

    fn nodes_eq(
        a: &DynTree<ExprIR<Arc<String>>>,
        a_idx: orx_tree::NodeIdx<Dyn<ExprIR<Arc<String>>>>,
        b: &DynTree<ExprIR<Arc<String>>>,
        b_idx: orx_tree::NodeIdx<Dyn<ExprIR<Arc<String>>>>,
    ) -> bool {
        let a_node = a.node(a_idx);
        let b_node = b.node(b_idx);

        let a_children: Vec<_> = (0..a_node.num_children())
            .filter(|&i| !is_agg_placeholder(a_node.child(i).data()))
            .collect();
        let b_children: Vec<_> = (0..b_node.num_children())
            .filter(|&i| !is_agg_placeholder(b_node.child(i).data()))
            .collect();

        if a_children.len() != b_children.len() {
            return false;
        }

        if !expr_ir_eq(a_node.data(), b_node.data()) {
            return false;
        }

        a_children
            .iter()
            .zip(b_children.iter())
            .all(|(&ai, &bi)| nodes_eq(a, a_node.child(ai).idx(), b, b_node.child(bi).idx()))
    }

    fn expr_ir_eq(
        a: &ExprIR<Arc<String>>,
        b: &ExprIR<Arc<String>>,
    ) -> bool {
        match (a, b) {
            (ExprIR::Variable(va), ExprIR::Variable(vb)) => va == vb,
            (ExprIR::FuncInvocation(fa), ExprIR::FuncInvocation(fb)) => fa.name == fb.name,
            (ExprIR::String(sa), ExprIR::String(sb)) => sa == sb,
            (ExprIR::Integer(ia), ExprIR::Integer(ib)) => ia == ib,
            (ExprIR::Float(fa), ExprIR::Float(fb)) => fa == fb,
            (ExprIR::Bool(ba), ExprIR::Bool(bb)) => ba == bb,
            (ExprIR::Property(pa), ExprIR::Property(pb)) => pa == pb,
            (ExprIR::Parameter(pa), ExprIR::Parameter(pb)) => pa == pb,
            _ => std::mem::discriminant(a) == std::mem::discriminant(b),
        }
    }

    nodes_eq(a, a.root().idx(), b, b.root().idx())
}
