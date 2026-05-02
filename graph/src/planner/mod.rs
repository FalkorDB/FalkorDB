//! Query plan generation from bound AST.
//!
//! The planner converts a bound Cypher AST into a logical execution plan (IR tree).
//! This phase determines the order of operations and which algorithms to use for
//! pattern matching.
//!
//! ## Plan Structure
//!
//! The plan is a tree where:
//! - Leaf nodes produce tuples (scans, argument)
//! - Internal nodes transform/filter tuples from children
//! - The root produces the final result
//!
//! ## Key Planning Decisions
//!
//! 1. **Scan selection**: Chooses between label scans, index scans, or ID lookups
//! 2. **Join ordering**: Determines order of pattern matching for efficiency
//! 3. **Projection placement**: Decides when to project/aggregate
//! 4. **Filter pushdown**: Places filters as early as possible
//!
//! ## IR Operators
//!
//! - **NodeByLabelScan**: Scan all nodes with a label
//! - **NodeByIndexScan**: Use an index for node lookup
//! - **CondTraverse**: Traverse relationships conditionally
//! - **ExpandInto**: Check for relationship between known nodes
//! - **Filter**: Apply predicate to filter tuples
//! - **Project**: Compute new values from existing
//! - **Aggregate**: Group and aggregate tuples
//! - **Sort/Skip/Limit**: Order and paginate results

pub mod binder;
pub mod optimizer;
pub mod tree;

use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    sync::Arc,
};

use crate::runtime::functions::Type;
use crate::tree;

use orx_tree::{DynNode, DynTree, NodeRef, Side, Traversal, Traverser};

use crate::{
    entity_type::EntityType,
    index::indexer::{IndexQuery, IndexType},
    parser::ast::{
        AllShortestPaths, BoundQueryIR, ExprIR, QueryExpr, QueryGraph, QueryIR, QueryNode,
        QueryPath, QueryRelationship, SetItem, SupportAggregation, Variable,
    },
    runtime::functions::GraphFn,
    runtime::orderset::OrderSet,
};

/// Intermediate Representation (IR) for execution plan operators.
///
/// Each variant represents a physical operation in the query execution plan.
/// The plan forms a tree where data flows from leaves to root.
#[derive(Clone, Debug)]
pub enum IR {
    /// Receives input from parent operator
    Argument,
    /// OPTIONAL MATCH - returns nulls if no match
    Optional(Vec<Variable>),
    /// CALL procedure with arguments, yielding outputs
    ProcedureCall {
        func: Arc<GraphFn>,
        args: Vec<QueryExpr<Variable>>,
        yields: Vec<Variable>,
    },
    /// UNWIND list AS variable
    Unwind {
        expr: QueryExpr<Variable>,
        var: Variable,
    },
    /// CREATE pattern
    Create(QueryGraph<Arc<String>, Arc<String>, Variable>),
    /// MERGE pattern with ON CREATE/ON MATCH actions
    Merge {
        pattern: QueryGraph<Arc<String>, Arc<String>, Variable>,
        on_create: Vec<SetItem<Arc<String>, Variable>>,
        on_match: Vec<SetItem<Arc<String>, Variable>>,
    },
    /// DELETE entities (detach flag for relationships)
    Delete {
        exprs: Vec<QueryExpr<Variable>>,
        detach: bool,
    },
    /// SET properties/labels
    Set(Vec<SetItem<Arc<String>, Variable>>),
    /// REMOVE properties/labels
    Remove(Vec<QueryExpr<Variable>>),
    /// Scan all nodes (no label filter)
    AllNodeScan(Arc<QueryNode<Arc<String>, Variable>>),
    /// Scan nodes by label
    NodeByLabelScan(Arc<QueryNode<Arc<String>, Variable>>),
    /// Scan nodes using an index
    NodeByIndexScan {
        node: Arc<QueryNode<Arc<String>, Variable>>,
        index: Arc<String>,
        query: Arc<IndexQuery<QueryExpr<Variable>>>,
    },
    /// Scan edges using an index, replacing CondTraverse when a filter
    /// can be pushed into the edge index.
    EdgeByIndexScan {
        relationship: Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>,
        query: Arc<IndexQuery<QueryExpr<Variable>>>,
        transposed: bool,
    },
    /// Scan nodes using a fulltext index
    NodeByFulltextScan {
        node: Variable,
        label: QueryExpr<Variable>,
        query: QueryExpr<Variable>,
        score: Option<Variable>,
    },
    /// Scan edges using a fulltext index
    EdgeByFulltextScan {
        edge: Variable,
        label: QueryExpr<Variable>,
        query: QueryExpr<Variable>,
        score: Option<Variable>,
    },
    /// Scan nodes using a KNN vector index. Mirrors
    /// [`NodeByFulltextScan`] but takes four input expressions
    /// (label, attribute, k, vector) and yields rows ordered by
    /// ascending distance.
    NodeByVectorScan {
        node: Variable,
        label: QueryExpr<Variable>,
        attr: QueryExpr<Variable>,
        k: QueryExpr<Variable>,
        vector: QueryExpr<Variable>,
        score: Option<Variable>,
    },
    /// Scan edges using a KNN vector index. Mirrors
    /// [`EdgeByFulltextScan`] but for vector indexes.
    EdgeByVectorScan {
        edge: Variable,
        label: QueryExpr<Variable>,
        attr: QueryExpr<Variable>,
        k: QueryExpr<Variable>,
        vector: QueryExpr<Variable>,
        score: Option<Variable>,
    },
    /// Lookup node by label and id
    NodeByLabelAndIdScan {
        node: Arc<QueryNode<Arc<String>, Variable>>,
        filter: Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
    },
    /// Lookup node by id only
    NodeByIdSeek {
        node: Arc<QueryNode<Arc<String>, Variable>>,
        filter: Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
    },
    /// Traverse relationships from known nodes.
    /// `emit_relationship`: when false, anonymous edge optimization applies —
    /// only one row per (src, dst) pair is emitted instead of one per edge.
    CondTraverse {
        relationship: Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>,
        emit_relationship: bool,
        /// Alias IDs of other relationship variables in the same MATCH clause
        /// component. Only these are checked for relationship uniqueness.
        sibling_edges: Vec<u32>,
        /// When true, the optimizer has swapped from/to relative to the edge
        /// direction in the graph. The runtime transposes the relationship
        /// scan accordingly.
        transposed: bool,
    },
    /// Variable-length traversal (BFS) from known nodes
    CondVarLenTraverse {
        relationship: Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>,
        /// Optional per-hop edge filter absorbed from a WHERE clause by the optimizer.
        edge_filter: Option<QueryExpr<Variable>>,
    },
    /// All shortest paths between two known nodes
    AllShortestPaths(Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>),
    /// Check relationship between two known nodes.
    /// `emit_relationship`: when false, anonymous edge optimization applies.
    ExpandInto {
        relationship: Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>,
        emit_relationship: bool,
        /// Alias IDs of other relationship variables in the same MATCH clause
        /// component. Only these are checked for relationship uniqueness.
        sibling_edges: Vec<u32>,
    },
    /// Build path objects from matched patterns
    PathBuilder(Vec<Arc<QueryPath<Variable>>>),
    /// Apply filter predicate
    Filter(QueryExpr<Variable>),
    /// Cartesian product of child results
    CartesianProduct,
    /// Value Hash Join: replaces CartesianProduct + equality Filter.
    /// Children: child(0) = left sub-plan, child(1) = right sub-plan.
    /// The join expressions (lhs_exp evaluated on left rows, rhs_exp on right rows)
    /// are stored here so the runtime can build a hash table.
    ValueHashJoin {
        lhs_exp: QueryExpr<Variable>,
        rhs_exp: QueryExpr<Variable>,
    },
    /// Apply = correlated join: for each row from child 0, run child 1
    Apply,
    /// Semi-join: passes through left row when right produces at least one result
    SemiApply,
    /// Anti-semi-join: passes through left row when right produces NO results
    AntiSemiApply,
    /// Or-apply-multiplexer: for each row from child 0 (bound branch),
    /// test condition branches (children 1..N) with short-circuit OR semantics.
    /// Passes through the row if ANY branch succeeds.
    /// `Vec<bool>` has one entry per condition branch: true means invert the
    /// branch result (anti-semi-join semantics for NOT-pattern predicates).
    /// Scalar filter branches are placed before pattern branches for efficiency.
    OrApplyMultiplexer(Vec<bool>),
    /// Load CSV file
    LoadCsv {
        file_path: QueryExpr<Variable>,
        headers: bool,
        delimiter: QueryExpr<Variable>,
        var: Variable,
    },
    /// Sort by expressions (bool = descending)
    Sort(Vec<(QueryExpr<Variable>, bool)>),
    /// Skip first N rows
    Skip(QueryExpr<Variable>),
    /// Limit to N rows
    Limit(QueryExpr<Variable>),
    /// Aggregate with grouping keys, aggregations, copy_from_parent, and projections
    Aggregate {
        names: Vec<Variable>,
        keys: Vec<(Variable, QueryExpr<Variable>)>,
        aggregations: Vec<(Variable, QueryExpr<Variable>)>,
        projections: Vec<(Variable, Variable)>,
    },
    /// Project expressions to new variables
    Project {
        exprs: Vec<(Variable, QueryExpr<Variable>)>,
        copies: Vec<(Variable, Variable)>,
    },
    /// Remove duplicate rows
    Distinct,
    /// UNION of multiple sub-query branches.
    /// Each child is a fully-planned branch.
    Union,
    /// Commit write operations to graph
    Commit,
    /// FOREACH(var IN list | body_plan)
    /// Children: child(0) = body sub-plan
    ForEach {
        list: QueryExpr<Variable>,
        var: Variable,
    },
    /// CREATE INDEX operation
    CreateIndex {
        label: Arc<String>,
        attrs: Vec<Arc<String>>,
        index_type: IndexType,
        entity_type: EntityType,
        options: Option<QueryExpr<Variable>>,
    },
    /// DROP INDEX operation
    DropIndex {
        label: Arc<String>,
        attrs: Vec<Arc<String>>,
        index_type: IndexType,
        entity_type: EntityType,
    },
}

/// Returns true if the subtree rooted at `idx` contains any node matching `predicate`.
pub fn subtree_contains(
    plan: &DynTree<IR>,
    idx: orx_tree::NodeIdx<orx_tree::Dyn<IR>>,
    predicate: fn(&IR) -> bool,
) -> bool {
    plan.node(idx)
        .walk_with(&mut Traversal.bfs().over_nodes())
        .any(|n| predicate(n.data()))
}

/// Returns true if a QueryExpr tree contains any non-deterministic function call.
fn expr_has_non_deterministic(expr: &DynTree<ExprIR<Variable>>) -> bool {
    expr.root()
        .walk_with(&mut Traversal.bfs().over_nodes())
        .any(|n| matches!(n.data(), ExprIR::FuncInvocation(func) if func.non_deterministic))
}

/// Returns true if a SetItem references any non-deterministic expression.
fn set_item_has_non_deterministic(item: &SetItem<Arc<String>, Variable>) -> bool {
    match item {
        SetItem::Attribute { target, value, .. } => {
            expr_has_non_deterministic(target) || expr_has_non_deterministic(value)
        }
        SetItem::Label { .. } => false,
    }
}

/// Returns true if a QueryGraph (CREATE/MERGE pattern) contains non-deterministic expressions.
fn query_graph_has_non_deterministic(qg: &QueryGraph<Arc<String>, Arc<String>, Variable>) -> bool {
    for node in qg.nodes() {
        if expr_has_non_deterministic(&node.attrs) {
            return true;
        }
    }
    for rel in qg.relationships() {
        if expr_has_non_deterministic(&rel.attrs) {
            return true;
        }
    }
    false
}

/// Returns true if an IndexQuery tree contains any non-deterministic function call.
fn index_query_has_non_deterministic(query: &IndexQuery<QueryExpr<Variable>>) -> bool {
    match query {
        IndexQuery::Range { min, max, .. } => {
            min.as_ref().is_some_and(|e| expr_has_non_deterministic(e))
                || max.as_ref().is_some_and(|e| expr_has_non_deterministic(e))
        }
        IndexQuery::And(queries) | IndexQuery::Or(queries) => {
            queries.iter().any(index_query_has_non_deterministic)
        }
        IndexQuery::Point { point, radius, .. } => {
            expr_has_non_deterministic(point) || expr_has_non_deterministic(radius)
        }
        IndexQuery::InList { list, .. } => expr_has_non_deterministic(list),
        IndexQuery::Equal { value, .. } | IndexQuery::ArrayContains { value, .. } => {
            expr_has_non_deterministic(value)
        }
    }
}

/// Returns true if the execution plan contains any non-deterministic function call.
#[must_use]
pub fn plan_is_non_deterministic(plan: &DynTree<IR>) -> bool {
    plan.root()
        .walk_with(&mut Traversal.bfs().over_nodes())
        .any(|node| match node.data() {
            IR::Create(qg) => query_graph_has_non_deterministic(qg),
            IR::Merge {
                pattern,
                on_create,
                on_match,
            } => {
                query_graph_has_non_deterministic(pattern)
                    || on_create.iter().any(set_item_has_non_deterministic)
                    || on_match.iter().any(set_item_has_non_deterministic)
            }
            IR::Set(items) => items.iter().any(set_item_has_non_deterministic),
            IR::Remove(exprs) | IR::Delete { exprs, .. } => {
                exprs.iter().any(|e| expr_has_non_deterministic(e))
            }
            IR::Unwind { expr, .. }
            | IR::Filter(expr)
            | IR::Skip(expr)
            | IR::Limit(expr)
            | IR::ForEach { list: expr, .. } => expr_has_non_deterministic(expr),
            IR::Sort(exprs) => exprs.iter().any(|(e, _)| expr_has_non_deterministic(e)),
            IR::Project { exprs, .. } => exprs.iter().any(|(_, e)| expr_has_non_deterministic(e)),
            IR::Aggregate {
                keys, aggregations, ..
            } => {
                keys.iter().any(|(_, e)| expr_has_non_deterministic(e))
                    || aggregations
                        .iter()
                        .any(|(_, e)| expr_has_non_deterministic(e))
            }
            IR::ProcedureCall { args, .. } => args.iter().any(|e| expr_has_non_deterministic(e)),
            IR::LoadCsv {
                file_path,
                delimiter,
                ..
            } => expr_has_non_deterministic(file_path) || expr_has_non_deterministic(delimiter),
            IR::ValueHashJoin { lhs_exp, rhs_exp } => {
                expr_has_non_deterministic(lhs_exp) || expr_has_non_deterministic(rhs_exp)
            }
            IR::NodeByIndexScan { query, .. } | IR::EdgeByIndexScan { query, .. } => {
                index_query_has_non_deterministic(query)
            }
            IR::NodeByFulltextScan { label, query, .. }
            | IR::EdgeByFulltextScan { label, query, .. } => {
                expr_has_non_deterministic(label) || expr_has_non_deterministic(query)
            }
            IR::NodeByVectorScan {
                label,
                attr,
                k,
                vector,
                ..
            }
            | IR::EdgeByVectorScan {
                label,
                attr,
                k,
                vector,
                ..
            } => {
                expr_has_non_deterministic(label)
                    || expr_has_non_deterministic(attr)
                    || expr_has_non_deterministic(k)
                    || expr_has_non_deterministic(vector)
            }
            IR::NodeByLabelAndIdScan { filter, .. } | IR::NodeByIdSeek { filter, .. } => {
                filter.iter().any(|(e, _)| expr_has_non_deterministic(e))
            }
            _ => false,
        })
}

/// Formats a relationship for CondTraverse/ExpandInto display.
/// Shows node labels and hides anonymous edge aliases.
fn fmt_rel_with_labels(rel: &QueryRelationship<Arc<String>, Arc<String>, Variable>) -> String {
    use itertools::Itertools;

    let direction = if rel.bidirectional { "" } else { ">" };

    let fmt_node = |node: &QueryNode<Arc<String>, Variable>| -> String {
        if node.labels.is_empty() {
            node.alias.to_string()
        } else {
            format!("{}:{}", node.alias, node.labels.iter().join(":"))
        }
    };
    let from_str = rel.from.alias.to_string();
    let to_str = fmt_node(&rel.to);

    let alias_str = rel.alias.to_string();
    let is_anon = alias_str.starts_with("_anon");

    if is_anon {
        format!("({from_str})-{direction}({to_str})")
    } else if rel.types.is_empty() {
        let alias = &rel.alias;
        format!("({from_str})-[{alias}]-{direction}({to_str})")
    } else {
        let alias = &rel.alias;
        let types = rel.types.iter().join("|");
        format!("({from_str})-[{alias}:{types}]-{direction}({to_str})")
    }
}

#[cfg_attr(tarpaulin, skip)]
impl Display for IR {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Argument => write!(f, "Argument"),
            Self::Optional(_) => write!(f, "Optional"),
            Self::ProcedureCall { .. } => write!(f, "ProcedureCall"),
            Self::Unwind { .. } => {
                write!(f, "Unwind")
            }
            Self::Create(pattern) => write!(f, "Create | {pattern}"),
            Self::Merge { pattern, .. } => write!(f, "Merge | {pattern}"),
            Self::Delete { .. } => write!(f, "Delete"),
            Self::Set(_) | Self::Remove(_) => write!(f, "Update"),
            Self::AllNodeScan(node) => {
                write!(f, "All Node Scan | {node}")
            }
            Self::NodeByLabelScan(node) => {
                write!(f, "Node By Label Scan | {node}")
            }
            Self::NodeByIndexScan { node, .. } => {
                write!(f, "Node By Index Scan | {node}")
            }
            Self::EdgeByIndexScan {
                relationship: rel, ..
            } => {
                write!(f, "Edge By Index Scan | {}", fmt_rel_with_labels(rel))
            }
            Self::NodeByFulltextScan { .. } => {
                write!(f, "Node By Fulltext Index Scan")
            }
            Self::EdgeByFulltextScan { .. } => {
                write!(f, "Edge By Fulltext Index Scan")
            }
            Self::NodeByVectorScan { .. } => {
                write!(f, "Node By Vector Index Scan")
            }
            Self::EdgeByVectorScan { .. } => {
                write!(f, "Edge By Vector Index Scan")
            }
            Self::NodeByLabelAndIdScan { node, .. } => {
                write!(f, "Node By Label and ID Scan | {node}")
            }
            Self::NodeByIdSeek { .. } => write!(f, "NodeByIdSeek"),
            Self::CondTraverse {
                relationship: rel, ..
            } => {
                write!(f, "Conditional Traverse | {}", fmt_rel_with_labels(rel))
            }
            Self::CondVarLenTraverse {
                relationship: rel, ..
            } => write!(f, "Conditional Variable Length Traverse | {rel}"),
            Self::AllShortestPaths(rel) => write!(f, "All Shortest Paths | {rel}"),
            Self::ExpandInto {
                relationship: rel, ..
            } => {
                write!(f, "Expand Into | {}", fmt_rel_with_labels(rel))
            }
            Self::PathBuilder(_) => write!(f, "PathBuilder"),
            Self::Filter(_) => write!(f, "Filter"),
            Self::CartesianProduct => write!(f, "Cartesian Product"),
            Self::ValueHashJoin { .. } => write!(f, "Value Hash Join"),
            Self::Apply => write!(f, "Apply"),
            Self::SemiApply => write!(f, "Semi Apply"),
            Self::AntiSemiApply => write!(f, "Anti Semi Apply"),
            Self::OrApplyMultiplexer(_) => write!(f, "Or Apply Multiplexer"),
            Self::LoadCsv { .. } => write!(f, "Load CSV"),
            Self::Sort(_) => write!(f, "Sort"),
            Self::Skip(_) => write!(f, "Skip"),
            Self::Limit(_) => write!(f, "Limit"),
            Self::Aggregate { .. } => write!(f, "Aggregate"),
            Self::Project { .. } => write!(f, "Project"),
            Self::Commit => write!(f, "Commit"),
            Self::ForEach { var, .. } => write!(f, "ForEach | {var}"),
            Self::Union => write!(f, "Union"),
            Self::Distinct => write!(f, "Distinct"),
            Self::CreateIndex { label, attrs, .. } => {
                write!(f, "Create Index | :{label}({attrs:?})")
            }
            Self::DropIndex { label, attrs, .. } => {
                write!(f, "Drop Index | :{label}({attrs:?})")
            }
        }
    }
}

/// Extracts inline attributes from a node pattern into a filter expression.
///
/// Given a node like `(n:Person {name: 'Alice', age: 30})`, returns a new node
/// with empty attrs and an equivalent filter expression `n.name = 'Alice' AND n.age = 30`.
/// Returns `None` for the filter if the node has no inline attributes.
pub(super) fn inline_node_attrs_to_filter(
    node: &Arc<QueryNode<Arc<String>, Variable>>
) -> (
    Arc<QueryNode<Arc<String>, Variable>>,
    Option<DynTree<ExprIR<Variable>>>,
) {
    let mut filters: Vec<DynTree<ExprIR<Variable>>> = vec![];

    for attr in node.attrs.root().children() {
        let ExprIR::String(attr_str) = attr.data() else {
            unreachable!("inline attrs map children must be ExprIR::String keys");
        };
        let eq = tree!(
            ExprIR::Eq,
            tree!(
                ExprIR::Property(attr_str.clone()),
                tree!(ExprIR::Variable(node.alias.clone()))
            ),
            attr.child(0).as_cloned_subtree()
        );
        filters.push(eq);
    }

    let filter = if filters.is_empty() {
        None
    } else if filters.len() == 1 {
        Some(filters.pop().unwrap())
    } else {
        Some(tree!(ExprIR::And; filters))
    };

    filter.map_or_else(
        || (node.clone(), None),
        |f| {
            let clean_node = Arc::new(QueryNode::new(
                node.alias.clone(),
                node.labels.clone(),
                Arc::new(tree!(ExprIR::Map)),
            ));
            (clean_node, Some(f))
        },
    )
}

/// Converts a bound Cypher AST into a logical execution plan (IR tree).
///
/// The planner maintains state across clauses:
/// - `visited` tracks which variable IDs have already been bound by earlier
///   scans or traversals, so we know whether a node needs a fresh scan
///   or can be referenced from the existing stream.
/// - `scope_vars` holds the binder-assigned variables grouped by scope ID,
///   used to mint fresh variable IDs for synthetic variables introduced
///   during pattern-predicate decomposition without collisions.
#[derive(Default)]
pub struct Planner {
    /// Variable (id, scope_id) pairs that are already bound in the current
    /// execution stream.  Tracking scope alongside id prevents inner-scope
    /// variables from shadowing outer-scope variables with the same id.
    visited: HashSet<(u32, u32)>,
    /// Binder-assigned variables grouped by scope ID.
    /// Used to derive fresh variable IDs within each scope.
    scope_vars: Vec<Vec<Variable>>,
}

impl Planner {
    #[must_use]
    pub fn new(scope_vars: Vec<Vec<Variable>>) -> Self {
        Self {
            visited: HashSet::new(),
            scope_vars,
        }
    }

    /// Mint a fresh variable with an ID unique within the given scope.
    fn fresh_var(
        &mut self,
        scope_id: u32,
        ty: Type,
    ) -> Variable {
        let id = self.scope_vars[scope_id as usize].len() as u32;
        let var = Variable {
            name: None,
            id,
            scope_id,
            ty,
        };
        self.scope_vars[scope_id as usize].push(var.clone());
        var
    }

    /// Attach `Argument` nodes to every leaf in the plan tree.
    ///
    /// When a sub-plan is used inside a correlated join (Apply, SemiApply, etc.),
    /// its leaves must receive the current row from the outer stream.  `Argument`
    /// is the operator that feeds the outer row into the sub-plan.
    ///
    /// MERGE nodes are treated specially: their last child is the match sub-plan
    /// which has its own Argument taps managed by MERGE planning. We must NOT
    /// descend into it. If MERGE has 2+ children, child(0) is the input pipeline
    /// and we descend into that. If MERGE has only 1 child (match branch), the
    /// runtime creates an inline Argument for the input.
    fn add_argument_to_leaves(tree: &mut DynTree<IR>) {
        let mut leaves = Vec::new();

        // DFS walk, but skip MERGE's internal match-branch sub-plan.
        let mut stack = vec![tree.root().idx()];
        while let Some(idx) = stack.pop() {
            let node = tree.node(idx);
            if matches!(node.data(), IR::Merge { .. }) {
                // Only descend into the input pipeline (child 0), not the
                // match branch (last child). If MERGE has only 1 child
                // (match-only), skip entirely — the runtime creates an
                // inline Argument for its input.
                if node.num_children() > 1 {
                    stack.push(node.child(0).idx());
                }
                continue;
            }
            if node.is_leaf() && !matches!(node.data(), IR::Argument) {
                leaves.push(idx);
            } else {
                for i in 0..node.num_children() {
                    stack.push(node.child(i).idx());
                }
            }
        }

        // Add Argument node as a child to each leaf.
        for leaf_idx in leaves {
            tree.node_mut(leaf_idx).push_child(IR::Argument);
        }
    }

    /// Check whether a rebuilt expression subtree references any of the given
    /// inline-pattern variable IDs.
    fn contains_inline_var(
        node: &DynNode<ExprIR<Variable>>,
        inline_var_ids: &HashSet<u32>,
    ) -> bool {
        let mut tr = Traversal.bfs().over_nodes();
        node.walk_with(&mut tr)
            .any(|n| matches!(n.data(), ExprIR::Variable(v) if inline_var_ids.contains(&v.id)))
    }

    /// Build a pattern sub-plan for a graph, saving and restoring visited state.
    fn build_pattern_sub_plan(
        &mut self,
        graph: &QueryGraph<Arc<String>, Arc<String>, Variable>,
    ) -> DynTree<IR> {
        let saved = self.visited.clone();
        let mut sub_plan = self.plan_match(graph, None);
        self.visited = saved;
        Self::add_argument_to_leaves(&mut sub_plan);
        sub_plan
    }

    /// Walk an expression tree and replace `PatternComprehension` / `Pattern`
    /// nodes with fresh variable references.  Returns the rebuilt expression
    /// and a list of extracted comprehensions ready for plan building.
    fn extract_pattern_comprehensions(
        &mut self,
        node: &DynNode<ExprIR<Variable>>,
        scope_id: u32,
        extracted: &mut Vec<(
            Variable,
            QueryGraph<Arc<String>, Arc<String>, Variable>,
            Option<Arc<DynTree<ExprIR<Variable>>>>,
            Arc<DynTree<ExprIR<Variable>>>,
            Vec<Arc<QueryPath<Variable>>>,
        )>,
    ) -> DynTree<ExprIR<Variable>> {
        match node.data() {
            ExprIR::PatternComprehension(graph) => {
                let var = self.fresh_var(scope_id, Type::List(Box::new(Type::Any)));

                let where_tree = {
                    let t =
                        self.extract_pattern_comprehensions(&node.child(0), scope_id, extracted);
                    if matches!(t.root().data(), ExprIR::Bool(true)) {
                        None
                    } else {
                        Some(Arc::new(t))
                    }
                };
                let result_tree = Arc::new(self.extract_pattern_comprehensions(
                    &node.child(1),
                    scope_id,
                    extracted,
                ));

                extracted.push((var.clone(), graph.clone(), where_tree, result_tree, vec![]));
                DynTree::new(ExprIR::Variable(var))
            }
            ExprIR::Pattern(graph) => {
                let var = self.fresh_var(scope_id, Type::List(Box::new(Type::Any)));

                // Build a path variable and path component variables from the
                // graph's nodes and relationships in pattern order so the
                // sub-plan collects actual Path values instead of a
                // placeholder integer.
                let path_var = self.fresh_var(scope_id, Type::Path);
                let mut path_component_vars = Vec::new();
                let nodes = graph.nodes();
                let rels = graph.relationships();
                for i in 0..nodes.len() {
                    path_component_vars.push(nodes[i].alias.clone());
                    if i < rels.len() {
                        path_component_vars.push(rels[i].alias.clone());
                    }
                }
                let query_path = Arc::new(QueryPath::new(path_var.clone(), path_component_vars));

                extracted.push((
                    var.clone(),
                    graph.clone(),
                    None,
                    Arc::new(DynTree::new(ExprIR::Variable(path_var))),
                    vec![query_path],
                ));
                DynTree::new(ExprIR::Variable(var))
            }
            _ => {
                let mut new_tree = DynTree::new(node.data().clone());
                for child in node.children() {
                    let child_tree =
                        self.extract_pattern_comprehensions(&child, scope_id, extracted);
                    new_tree.root_mut().push_child_tree(child_tree);
                }
                new_tree
            }
        }
    }

    /// Build the Apply + Aggregate sub-plan for a single pattern comprehension.
    ///
    /// Returns a plan tree:  `Aggregate(collect(result_expr)) -> traversal -> Argument`
    fn build_pattern_comprehension_plan(
        &mut self,
        var: &Variable,
        graph: &QueryGraph<Arc<String>, Arc<String>, Variable>,
        where_filter: Option<&Arc<DynTree<ExprIR<Variable>>>>,
        result_expr: &Arc<DynTree<ExprIR<Variable>>>,
        paths: &[Arc<QueryPath<Variable>>],
    ) -> DynTree<IR> {
        use crate::runtime::functions::{FnType, get_functions};
        use crate::runtime::value::Value;

        let saved = self.visited.clone();
        let mut sub_plan = self.plan_match(graph, None);
        self.visited = saved;

        // Add PathBuilder to construct Path values from matched variables.
        if !paths.is_empty() {
            sub_plan = tree!(IR::PathBuilder(paths.to_vec()), sub_plan);
        }

        // Add WHERE filter if present
        if let Some(filter) = where_filter {
            sub_plan = tree!(IR::Filter(filter.clone()), sub_plan);
        }

        Self::add_argument_to_leaves(&mut sub_plan);

        // Build collect(result_expr) aggregation expression
        let collect_fn = get_functions()
            .get(
                "collect",
                &FnType::Aggregation {
                    initial: Value::Null,
                    finalizer: None,
                },
            )
            .expect("collect function not registered");

        // Mint a fresh variable for the aggregation accumulator slot.
        // The aggregate runtime expects the last child of a FuncInvocation
        // (for aggregate functions) to be a Variable node that stores the
        // running accumulator value.
        let scope_id = var.scope_id;
        let agg_acc_var = self.fresh_var(scope_id, Type::Any);

        let mut collect_expr = DynTree::new(ExprIR::FuncInvocation(collect_fn));
        collect_expr
            .root_mut()
            .push_child_tree(result_expr.as_ref().clone());
        collect_expr
            .root_mut()
            .push_child_tree(DynTree::new(ExprIR::Variable(agg_acc_var)));
        let collect_expr = Arc::new(collect_expr);

        // Create Aggregate node: names=[var], group_by_keys=[], aggregations=[(var, collect(expr))]
        tree!(
            IR::Aggregate {
                names: vec![var.clone()],
                keys: vec![],
                aggregations: vec![(var.clone(), collect_expr)],
                projections: vec![]
            },
            sub_plan
        )
    }

    /// Recursively decompose an expression (that may contain inline-pattern
    /// variables) into an IR sub-plan.  `input` is the upstream data stream.
    /// The returned plan filters rows: only rows for which the expression
    /// evaluates to true are passed through.
    fn expr_to_plan(
        &mut self,
        node: &DynNode<ExprIR<Variable>>,
        inline_map: &HashMap<u32, QueryGraph<Arc<String>, Arc<String>, Variable>>,
        input: DynTree<IR>,
    ) -> DynTree<IR> {
        let inline_var_ids: HashSet<u32> = inline_map.keys().copied().collect();

        // Unwrap Paren nodes transparently
        if matches!(node.data(), ExprIR::Paren)
            && let Some(child) = node.get_child(0)
        {
            return self.expr_to_plan(&child, inline_map, input);
        }

        // Inline-pattern variable → SemiApply (pass if pattern exists)
        if let ExprIR::Variable(v) = node.data()
            && let Some(graph) = inline_map.get(&v.id)
        {
            let sub_plan = self.build_pattern_sub_plan(graph);
            return tree!(IR::SemiApply, input, sub_plan);
        }

        // NOT(inline-pattern variable) → AntiSemiApply
        if matches!(node.data(), ExprIR::Not)
            && let Some(child) = node.get_child(0)
            && let ExprIR::Variable(v) = child.data()
            && let Some(graph) = inline_map.get(&v.id)
        {
            let sub_plan = self.build_pattern_sub_plan(graph);
            return tree!(IR::AntiSemiApply, input, sub_plan);
        }

        // Pure scalar (no inline var refs) → Filter
        if !Self::contains_inline_var(node, &inline_var_ids) {
            let expr_tree = node.clone_as_tree();
            return tree!(IR::Filter(Arc::new(expr_tree)), input);
        }

        // OR → OrApplyMultiplexer
        if matches!(node.data(), ExprIR::Or) {
            return self.or_expr_to_plan(node, inline_map, input);
        }

        // AND → chain conditions sequentially
        if matches!(node.data(), ExprIR::And) {
            return self.and_expr_to_plan(node, inline_map, input);
        }

        // NOT(complex_expr) → AntiSemiApply(input, inner_plan)
        // inner_plan passes rows when the inner expression is true,
        // so AntiSemiApply inverts: passes when inner is false.
        if matches!(node.data(), ExprIR::Not)
            && let Some(child) = node.get_child(0)
        {
            let inner = self.expr_to_plan(&child, inline_map, tree!(IR::Argument));
            return tree!(IR::AntiSemiApply, input, inner);
        }

        // Fallback for other operators (XOR etc.) with inline vars:
        // shouldn't happen in practice; treat as opaque filter.
        let expr_tree = node.clone_as_tree();
        tree!(IR::Filter(Arc::new(expr_tree)), input)
    }

    /// Build an `OrApplyMultiplexer` for an OR expression.
    /// `input` becomes the bound branch (child 0).
    fn or_expr_to_plan(
        &mut self,
        or_node: &DynNode<ExprIR<Variable>>,
        inline_map: &HashMap<u32, QueryGraph<Arc<String>, Arc<String>, Variable>>,
        input: DynTree<IR>,
    ) -> DynTree<IR> {
        let inline_var_ids: HashSet<u32> = inline_map.keys().copied().collect();

        // Collect OR children into owned trees so we can call &mut self freely.
        let child_trees: Vec<DynTree<ExprIR<Variable>>> =
            or_node.children().map(|c| c.clone_as_tree()).collect();

        // Classify branches: scalars first (cheap), then non-scalars.
        let mut scalar_branches: Vec<DynTree<IR>> = vec![];
        let mut other_branches: Vec<(DynTree<IR>, bool)> = vec![]; // (plan, is_anti)

        for child_tree in &child_trees {
            let child = child_tree.root();
            // Bare pattern variable → raw pattern sub-plan (anti=false)
            if let ExprIR::Variable(v) = child.data()
                && let Some(graph) = inline_map.get(&v.id)
            {
                let sub_plan = self.build_pattern_sub_plan(graph);
                other_branches.push((sub_plan, false));
                continue;
            }
            // NOT(pattern variable) → raw pattern sub-plan (anti=true)
            if matches!(child.data(), ExprIR::Not)
                && let Some(grandchild) = child.get_child(0)
                && let ExprIR::Variable(v) = grandchild.data()
                && let Some(graph) = inline_map.get(&v.id)
            {
                let sub_plan = self.build_pattern_sub_plan(graph);
                other_branches.push((sub_plan, true));
                continue;
            }
            // Pure scalar → Filter(expr, Argument)
            if !Self::contains_inline_var(&child, &inline_var_ids) {
                let expr_tree = child.clone_as_tree();
                let branch = tree!(IR::Filter(Arc::new(expr_tree)), tree!(IR::Argument));
                scalar_branches.push(branch);
                continue;
            }
            // Complex child (AND with patterns, nested OR, etc.):
            // Recursively build a sub-plan starting from Argument.
            let branch = self.expr_to_plan(&child, inline_map, tree!(IR::Argument));
            other_branches.push((branch, false));
        }

        // Assemble: bound branch first, then scalars, then others.
        let mut anti_flags = Vec::with_capacity(scalar_branches.len() + other_branches.len());
        let mut children = Vec::with_capacity(1 + scalar_branches.len() + other_branches.len());
        children.push(input);
        for branch in scalar_branches {
            anti_flags.push(false);
            children.push(branch);
        }
        for (branch, is_anti) in other_branches {
            anti_flags.push(is_anti);
            children.push(branch);
        }
        tree!(IR::OrApplyMultiplexer(anti_flags); children)
    }

    /// Build a chained plan for an AND expression.
    /// Scalars are applied first (cheap), then pattern / complex conditions.
    fn and_expr_to_plan(
        &mut self,
        and_node: &DynNode<ExprIR<Variable>>,
        inline_map: &HashMap<u32, QueryGraph<Arc<String>, Arc<String>, Variable>>,
        input: DynTree<IR>,
    ) -> DynTree<IR> {
        let inline_var_ids: HashSet<u32> = inline_map.keys().copied().collect();

        // Collect children into owned trees for borrow-safety.
        let child_trees: Vec<DynTree<ExprIR<Variable>>> =
            and_node.children().map(|c| c.clone_as_tree()).collect();

        let mut scalar_parts: Vec<DynTree<ExprIR<Variable>>> = vec![];
        let mut non_scalar_trees: Vec<&DynTree<ExprIR<Variable>>> = vec![];

        for child_tree in &child_trees {
            let child = child_tree.root();
            if Self::contains_inline_var(&child, &inline_var_ids) {
                non_scalar_trees.push(child_tree);
            } else {
                // Skip trivial Bool(true) from extractable pattern replacement
                if !matches!(child.data(), ExprIR::Bool(true)) {
                    scalar_parts.push(child.clone_as_tree());
                }
            }
        }

        let mut plan = input;

        // Apply scalar filter first (cheapest).
        if !scalar_parts.is_empty() {
            let filter_expr = if scalar_parts.len() == 1 {
                Arc::new(scalar_parts.into_iter().next().unwrap())
            } else {
                Arc::new(tree!(ExprIR::And; scalar_parts))
            };
            plan = tree!(IR::Filter(filter_expr), plan);
        }

        // Apply non-scalar conditions sequentially (each filters the stream).
        for child_tree in non_scalar_trees {
            plan = self.expr_to_plan(&child_tree.root(), inline_map, plan);
        }

        plan
    }

    /// Walk a WHERE-clause expression tree and separate out pattern predicates
    /// (e.g. `WHERE EXISTS { (a)-[:KNOWS]->(b) }`) from scalar predicates.
    ///
    /// Pattern predicates cannot be evaluated as simple filters -- they require
    /// building a sub-plan (SemiApply / AntiSemiApply).  This function rebuilds
    /// the expression tree with patterns replaced by either:
    ///
    /// - **Extractable** patterns (`can_extract = true`): removed from the
    ///   expression and collected in `extractable`.  These are top-level AND
    ///   conjuncts that can each become their own SemiApply/AntiSemiApply.
    ///   The expression slot is replaced with `Bool(true)` (identity for AND).
    ///
    /// - **Inline** patterns (`can_extract = false`): replaced with a fresh
    ///   synthetic variable and collected in `inline`.  These appear under OR
    ///   or other operators where they cannot be independently extracted and
    ///   must be handled by `expr_to_plan` (OrApplyMultiplexer, etc.).
    ///
    /// `can_extract` propagates through AND (conjuncts are independently
    /// extractable) but resets to `false` under OR, NOT, and other operators.
    fn collect_patterns_and_rebuild(
        &mut self,
        node: &DynNode<ExprIR<Variable>>,
        extractable: &mut Vec<(QueryGraph<Arc<String>, Arc<String>, Variable>, bool)>,
        inline: &mut HashMap<u32, QueryGraph<Arc<String>, Arc<String>, Variable>>,
        can_extract: bool,
    ) -> DynTree<ExprIR<Variable>> {
        match node.data() {
            // Bare pattern: `EXISTS { ... }` or similar.
            ExprIR::Pattern(graph) => {
                if can_extract {
                    // Top-level conjunct: extract for SemiApply, replace with true.
                    extractable.push((graph.clone(), false));
                    DynTree::new(ExprIR::Bool(true))
                } else {
                    // Under OR/NOT: replace with a fresh boolean variable and
                    // record for inline handling via expr_to_plan.
                    let current_scope = graph.variables().next().unwrap().scope_id;
                    let var = Variable {
                        name: None,
                        id: self.scope_vars[current_scope as usize].len() as u32,
                        scope_id: current_scope,
                        ty: Type::Bool,
                    };
                    self.scope_vars[current_scope as usize].push(var.clone());
                    inline.insert(var.id, graph.clone());
                    DynTree::new(ExprIR::Variable(var))
                }
            }
            // NOT(pattern): `NOT EXISTS { ... }`.
            ExprIR::Not => {
                // Special-case: NOT directly wrapping a pattern.
                if let Some(child) = node.get_child(0)
                    && let ExprIR::Pattern(graph) = child.data()
                {
                    if can_extract {
                        // Extract for AntiSemiApply (is_anti = true).
                        extractable.push((graph.clone(), true));
                        return DynTree::new(ExprIR::Bool(true));
                    }
                    // Inline: create NOT(synth_var) so expr_to_plan can
                    // recognize the negation and use AntiSemiApply.
                    let current_scope = graph.variables().next().unwrap().scope_id;
                    let var = Variable {
                        name: None,
                        id: self.scope_vars[current_scope as usize].len() as u32,
                        scope_id: current_scope,
                        ty: Type::Bool,
                    };
                    self.scope_vars[current_scope as usize].push(var.clone());
                    inline.insert(var.id, graph.clone());
                    let mut new_tree = DynTree::new(ExprIR::Not);
                    new_tree
                        .root_mut()
                        .push_child_tree(DynTree::new(ExprIR::Variable(var)));
                    return new_tree;
                }
                // NOT wrapping something other than a bare pattern:
                // recurse into children with can_extract = false (NOT
                // blocks extraction since the pattern is negated).
                let mut new_tree = DynTree::new(node.data().clone());
                for child in node.children() {
                    let child_tree =
                        self.collect_patterns_and_rebuild(&child, extractable, inline, false);
                    new_tree.root_mut().push_child_tree(child_tree);
                }
                new_tree
            }
            // AND: propagate can_extract to children since each conjunct
            // can be independently extracted as its own SemiApply.
            ExprIR::And => {
                let mut new_tree = DynTree::new(ExprIR::And);
                for child in node.children() {
                    let child_tree =
                        self.collect_patterns_and_rebuild(&child, extractable, inline, can_extract);
                    new_tree.root_mut().push_child_tree(child_tree);
                }
                new_tree
            }
            // Any other expression node (comparisons, function calls, OR, etc.):
            // recurse with can_extract = false since patterns under these
            // operators cannot be independently extracted.
            _ => {
                let mut new_tree = DynTree::new(node.data().clone());
                for child in node.children() {
                    let child_tree =
                        self.collect_patterns_and_rebuild(&child, extractable, inline, false);
                    new_tree.root_mut().push_child_tree(child_tree);
                }
                new_tree
            }
        }
    }

    /// Build an execution plan for a MATCH clause.
    ///
    /// The pattern graph is decomposed into connected components.  Each component
    /// produces a sub-plan (scan + traversals), and disconnected components are
    /// joined with a CartesianProduct.  The optional WHERE filter is then applied
    /// on top, with pattern predicates decomposed into SemiApply/AntiSemiApply.
    ///
    /// The `visited` set is updated as variables are bound, so subsequent clauses
    /// know which variables are already available in the stream.
    fn plan_match(
        &mut self,
        pattern: &QueryGraph<Arc<String>, Arc<String>, Variable>,
        filter: Option<QueryExpr<Variable>>,
    ) -> DynTree<IR> {
        // Each connected component of the pattern becomes a separate sub-plan.
        let mut vec = vec![];
        // Collect extra filters for bound variables with new constraints
        // (labels or inline properties). These are applied as top-level
        // filters rather than as plan components, so they don't interfere
        // with stitching.
        let mut bound_filters: Vec<DynTree<ExprIR<Variable>>> = vec![];
        for component in pattern.connected_components() {
            let relationships = component.relationships();
            // Reorder relationships: put variable-length and AllShortestPaths
            // relationships after fixed-length ones so that fixed-length
            // traversals (especially self-loops / ExpandInto) are planned as
            // leaves and variable-length traversals sit above them.
            let mut sorted_rels: Vec<_> = relationships.to_vec();
            sorted_rels.sort_by_key(|r| {
                let base = if r.all_shortest_paths == AllShortestPaths::No {
                    i32::from(r.min_hops.is_some())
                } else {
                    2
                };
                // Within the same hop-type tier, prefer relationships that
                // touch an already-bound variable so that traversals start
                // from the known node (matches FalkorDB C behaviour).
                let has_bound = i32::from(
                    !(self
                        .visited
                        .contains(&(r.from.alias.id, r.from.alias.scope_id))
                        || self.visited.contains(&(r.to.alias.id, r.to.alias.scope_id))),
                );
                (base, has_bound)
            });
            let mut iter = sorted_rels.iter();
            let Some(relationship) = iter.next() else {
                // Node-only component (no relationships).
                let nodes = component.nodes();
                debug_assert_eq!(nodes.len(), 1);
                let node = nodes[0].clone();
                if self.visited.contains(&(node.alias.id, node.alias.scope_id)) {
                    // Already bound: check for inline property constraints and
                    // additional labels that need verifying.
                    let (_, attr_filter) = inline_node_attrs_to_filter(&node);
                    if let Some(filter_expr) = attr_filter {
                        bound_filters.push(filter_expr);
                    }
                    if node.labels.is_empty() {
                        vec.push(tree!(IR::Argument));
                    } else {
                        // Additional labels on an already-bound node: create a
                        // synthetic self-loop ExpandInto to verify them.
                        let from_node = Arc::new(QueryNode::new(
                            node.alias.clone(),
                            OrderSet::default(),
                            Arc::new(tree!(ExprIR::Map)),
                        ));
                        let to_node = Arc::new(QueryNode::new(
                            node.alias.clone(),
                            node.labels.clone(),
                            Arc::new(tree!(ExprIR::Map)),
                        ));
                        let edge_alias = Variable {
                            name: None,
                            id: u32::MAX - node.alias.id,
                            scope_id: node.alias.scope_id,
                            ty: Type::Relationship,
                        };
                        let rel = Arc::new(QueryRelationship::new(
                            edge_alias,
                            vec![],
                            Arc::new(tree!(ExprIR::Map)),
                            from_node,
                            to_node,
                            false,
                            None,
                            None,
                        ));
                        vec.push(tree!(
                            IR::ExpandInto {
                                relationship: rel,
                                emit_relationship: false,
                                sibling_edges: vec![]
                            },
                            tree!(IR::Argument)
                        ));
                    }
                } else {
                    let (clean_node, attr_filter) = inline_node_attrs_to_filter(&node);
                    // The binder's post-processing already set the full
                    // accumulated label set on each QueryNode directly.
                    let mut res = if clean_node.labels.is_empty() {
                        tree!(IR::AllNodeScan(clean_node))
                    } else {
                        // Multi-label node: the runtime's get_nodes()
                        // intersects all label matrices, so we can pass
                        // all labels directly to NodeByLabelScan.
                        tree!(IR::NodeByLabelScan(clean_node))
                    };
                    if let Some(filter_expr) = attr_filter {
                        res = tree!(IR::Filter(Arc::new(filter_expr)), res);
                    }
                    self.visited.insert((node.alias.id, node.alias.scope_id));
                    let paths = component.paths();
                    if !paths.is_empty() {
                        res = tree!(IR::PathBuilder(paths.to_vec()), res);
                    }
                    vec.push(res);
                }
                continue;
            };
            // Plan the first relationship in this connected component.
            // The choice of operator depends on which endpoints are already bound:
            //   - Self-loop (from == to): scan the node, then ExpandInto
            //   - Both endpoints bound: ExpandInto (just check the edge exists)
            //   - Variable-length path: CondVarLenTraverse (BFS)
            //   - Otherwise: CondTraverse (fixed-length traversal)
            //
            // emit_relationship: true when the edge must be bound per-edge
            // (named edge or edge referenced in a named path). When false,
            // the runtime may collapse multi-edges into one row per (src, dst).
            let emit_rel = |rel: &QueryRelationship<Arc<String>, Arc<String>, Variable>| -> bool {
                !rel.alias
                    .name
                    .as_ref()
                    .is_some_and(|n| n.starts_with("_anon"))
                    || component
                        .paths()
                        .iter()
                        .any(|p| p.vars.iter().any(|v| v.id == rel.alias.id))
            };
            // Collect edge alias IDs for relationship uniqueness checking.
            // Only named (non-anonymous) edges participate in uniqueness constraints;
            // anonymous edges (`_anon*`) are excluded so the same physical edge
            // can be traversed by different anonymous bindings (matches FalkorDB C).
            let sibling_edges: Vec<u32> = relationships
                .iter()
                .filter(|r| {
                    !r.alias
                        .name
                        .as_ref()
                        .is_some_and(|n| n.starts_with("_anon"))
                })
                .map(|r| r.alias.id)
                .collect();
            let mut res = if relationship.all_shortest_paths != AllShortestPaths::No {
                tree!(IR::AllShortestPaths(relationship.clone()))
            } else if relationship.min_hops.is_some() {
                // Variable-length path — must use CVLT even for self-loops (a)-[*0]->(a).
                // Build scan child for the from-node when it's not yet visited
                // (i.e. this CVLT is the leaf of a component).
                let scan_child = if self
                    .visited
                    .contains(&(relationship.from.alias.id, relationship.from.alias.scope_id))
                {
                    None
                } else {
                    let (clean_node, from_attr_filter) =
                        inline_node_attrs_to_filter(&relationship.from);
                    let mut scan = if clean_node.labels.is_empty() {
                        tree!(IR::AllNodeScan(clean_node))
                    } else {
                        tree!(IR::NodeByLabelScan(clean_node))
                    };
                    if let Some(filter_expr) = from_attr_filter {
                        scan = tree!(IR::Filter(Arc::new(filter_expr)), scan);
                    }
                    Some(scan)
                };
                scan_child.map_or_else(
                    || {
                        tree!(IR::CondVarLenTraverse {
                            relationship: relationship.clone(),
                            edge_filter: None
                        })
                    },
                    |scan| {
                        tree!(
                            IR::CondVarLenTraverse {
                                relationship: relationship.clone(),
                                edge_filter: None
                            },
                            scan
                        )
                    },
                )
            } else if relationship.from.alias.id == relationship.to.alias.id {
                // Self-loop with fixed-length edge: scan + ExpandInto.
                let (clean_node, attr_filter) = inline_node_attrs_to_filter(&relationship.from);
                let mut scan = if clean_node.labels.is_empty() {
                    tree!(IR::AllNodeScan(clean_node))
                } else {
                    tree!(IR::NodeByLabelScan(clean_node))
                };
                if let Some(filter_expr) = attr_filter {
                    scan = tree!(IR::Filter(Arc::new(filter_expr)), scan);
                }
                tree!(
                    IR::ExpandInto {
                        relationship: relationship.clone(),
                        emit_relationship: emit_rel(relationship),
                        sibling_edges: sibling_edges.clone()
                    },
                    scan
                )
            } else if self
                .visited
                .contains(&(relationship.from.alias.id, relationship.from.alias.scope_id))
                && self
                    .visited
                    .contains(&(relationship.to.alias.id, relationship.to.alias.scope_id))
            {
                let mut ei = tree!(IR::ExpandInto {
                    relationship: relationship.clone(),
                    emit_relationship: emit_rel(relationship),
                    sibling_edges: sibling_edges.clone()
                });
                // Both endpoints already bound — check for inline attrs
                // that need filtering (e.g. reversed patterns where attrs
                // appear on a later occurrence of an already-bound node).
                let (_, from_attr_filter) = inline_node_attrs_to_filter(&relationship.from);
                if let Some(filter_expr) = from_attr_filter {
                    ei = tree!(IR::Filter(Arc::new(filter_expr)), ei);
                }
                let (_, to_attr_filter) = inline_node_attrs_to_filter(&relationship.to);
                if let Some(filter_expr) = to_attr_filter {
                    ei = tree!(IR::Filter(Arc::new(filter_expr)), ei);
                }
                ei
            } else {
                tree!(IR::CondTraverse {
                    relationship: relationship.clone(),
                    emit_relationship: emit_rel(relationship),
                    sibling_edges: sibling_edges.clone(),
                    transposed: false
                })
            };
            // Check destination node for inline attributes (e.g., (b {val: 'v2'}))
            // and add a Filter if present.
            if !self
                .visited
                .contains(&(relationship.to.alias.id, relationship.to.alias.scope_id))
            {
                let (_, to_attr_filter) = inline_node_attrs_to_filter(&relationship.to);
                if let Some(filter_expr) = to_attr_filter {
                    res = tree!(IR::Filter(Arc::new(filter_expr)), res);
                }
            }
            // Check source node for inline attributes and add a Filter above
            // the CT. This is needed so the optimizer's chain reversal can
            // see and reposition these filters properly.
            if !self
                .visited
                .contains(&(relationship.from.alias.id, relationship.from.alias.scope_id))
            {
                let (_, from_attr_filter) = inline_node_attrs_to_filter(&relationship.from);
                if let Some(filter_expr) = from_attr_filter {
                    res = tree!(IR::Filter(Arc::new(filter_expr)), res);
                }
            }
            self.visited
                .insert((relationship.from.alias.id, relationship.from.alias.scope_id));
            self.visited
                .insert((relationship.to.alias.id, relationship.to.alias.scope_id));
            self.visited
                .insert((relationship.alias.id, relationship.alias.scope_id));
            // Chain remaining relationships in the component, each one
            // stacking on top of the previous result using the same logic.
            for relationship in iter {
                res = if relationship.all_shortest_paths != AllShortestPaths::No {
                    tree!(IR::AllShortestPaths(relationship.clone()), res)
                } else if relationship.min_hops.is_some() {
                    let mut cvlt = tree!(
                        IR::CondVarLenTraverse {
                            relationship: relationship.clone(),
                            edge_filter: None
                        },
                        res
                    );
                    if !self
                        .visited
                        .contains(&(relationship.from.alias.id, relationship.from.alias.scope_id))
                    {
                        let (_, from_attr_filter) = inline_node_attrs_to_filter(&relationship.from);
                        if let Some(filter_expr) = from_attr_filter {
                            cvlt = tree!(IR::Filter(Arc::new(filter_expr)), cvlt);
                        }
                    }
                    cvlt
                } else if relationship.from.alias.id == relationship.to.alias.id {
                    let (clean_node, attr_filter) = inline_node_attrs_to_filter(&relationship.from);
                    let mut scan = if clean_node.labels.is_empty() {
                        tree!(IR::AllNodeScan(clean_node))
                    } else {
                        tree!(IR::NodeByLabelScan(clean_node))
                    };
                    if let Some(filter_expr) = attr_filter {
                        scan = tree!(IR::Filter(Arc::new(filter_expr)), scan);
                    }
                    tree!(
                        IR::ExpandInto {
                            relationship: relationship.clone(),
                            emit_relationship: emit_rel(relationship),
                            sibling_edges: sibling_edges.clone()
                        },
                        scan,
                        res
                    )
                } else if self
                    .visited
                    .contains(&(relationship.from.alias.id, relationship.from.alias.scope_id))
                    && self
                        .visited
                        .contains(&(relationship.to.alias.id, relationship.to.alias.scope_id))
                {
                    let mut ei = tree!(
                        IR::ExpandInto {
                            relationship: relationship.clone(),
                            emit_relationship: emit_rel(relationship),
                            sibling_edges: sibling_edges.clone()
                        },
                        res
                    );
                    let (_, from_attr_filter) = inline_node_attrs_to_filter(&relationship.from);
                    if let Some(filter_expr) = from_attr_filter {
                        ei = tree!(IR::Filter(Arc::new(filter_expr)), ei);
                    }
                    let (_, to_attr_filter) = inline_node_attrs_to_filter(&relationship.to);
                    if let Some(filter_expr) = to_attr_filter {
                        ei = tree!(IR::Filter(Arc::new(filter_expr)), ei);
                    }
                    ei
                } else {
                    tree!(
                        IR::CondTraverse {
                            relationship: relationship.clone(),
                            emit_relationship: emit_rel(relationship),
                            sibling_edges: sibling_edges.clone(),
                            transposed: false
                        },
                        res
                    )
                };
                // Check destination node for inline attributes (e.g., (b {val: 'v2'}))
                // and add a Filter if present.
                if !self
                    .visited
                    .contains(&(relationship.to.alias.id, relationship.to.alias.scope_id))
                {
                    let (_, to_attr_filter) = inline_node_attrs_to_filter(&relationship.to);
                    if let Some(filter_expr) = to_attr_filter {
                        res = tree!(IR::Filter(Arc::new(filter_expr)), res);
                    }
                }
                self.visited
                    .insert((relationship.from.alias.id, relationship.from.alias.scope_id));
                self.visited
                    .insert((relationship.to.alias.id, relationship.to.alias.scope_id));
                self.visited
                    .insert((relationship.alias.id, relationship.alias.scope_id));
            }
            let paths = component.paths();
            if !paths.is_empty() {
                res = tree!(IR::PathBuilder(paths.to_vec()), res);
            }
            vec.push(res);
        }
        // Join disconnected components: single component uses its plan directly,
        // multiple components are joined via CartesianProduct.
        let mut res = if vec.len() == 1 {
            vec.pop().unwrap()
        } else {
            tree!(IR::CartesianProduct; vec)
        };
        // Apply the WHERE filter.  Pattern predicates are separated from scalar
        // predicates by collect_patterns_and_rebuild:
        //   - "extractable" patterns become SemiApply / AntiSemiApply wrappers
        //   - "inline" patterns (under OR, etc.) are handled by expr_to_plan
        //   - remaining scalar predicates become a Filter node
        if let Some(filter) = filter {
            let mut extractable = vec![];
            let mut inline = HashMap::new();
            let rebuilt = self.collect_patterns_and_rebuild(
                &filter.root(),
                &mut extractable,
                &mut inline,
                true,
            );

            // When there are inline patterns, recursively decompose the
            // rebuilt expression into multiplexer / semi-apply / filter nodes.
            if !inline.is_empty() {
                res = self.expr_to_plan(&rebuilt.root(), &inline, res);
            } else if !matches!(rebuilt.root().data(), ExprIR::Bool(true)) {
                res = tree!(IR::Filter(Arc::new(rebuilt)), res);
            }

            // Apply SemiApply/AntiSemiApply for each extractable pattern
            for (graph, is_anti) in extractable {
                let saved = self.visited.clone();
                let mut sub_plan = self.plan_match(&graph, None);
                self.visited = saved;
                Self::add_argument_to_leaves(&mut sub_plan);
                if is_anti {
                    res = tree!(IR::AntiSemiApply, res, sub_plan);
                } else {
                    res = tree!(IR::SemiApply, res, sub_plan);
                }
            }
        }
        // Apply filters for bound variables with new label/property
        // constraints. These are collected during component planning and
        // applied here so they sit above the stitching target, ensuring
        // the bound variable's env is available when the filter runs.
        if !bound_filters.is_empty() {
            let filter_expr = if bound_filters.len() == 1 {
                bound_filters.pop().unwrap()
            } else {
                tree!(ExprIR::And; bound_filters)
            };
            res = tree!(IR::Filter(Arc::new(filter_expr)), res);
        }
        res
    }

    /// Build a plan for WITH or RETURN clauses (projection / aggregation).
    ///
    /// This handles: projection, aggregation, DISTINCT, ORDER BY, SKIP, LIMIT,
    /// and an optional WHERE filter (only for WITH, not RETURN).
    ///
    /// The plan tree is built top-down: the root is Project/Aggregate, with
    /// Commit, Distinct, Sort, Skip, Limit, and Filter layered above as needed.
    /// The `visited` set is reset to only the projected variables, since
    /// WITH/RETURN starts a new scope.
    #[allow(clippy::too_many_arguments)]
    fn plan_project(
        &mut self,
        exprs: Vec<(Variable, QueryExpr<Variable>)>,
        copy_from_parent: Vec<(Variable, Variable)>,
        orderby: Vec<(QueryExpr<Variable>, bool)>,
        skip: Option<QueryExpr<Variable>>,
        limit: Option<QueryExpr<Variable>>,
        filter: Option<QueryExpr<Variable>>,
        distinct: bool,
        write: bool,
    ) -> DynTree<IR> {
        // Check if any expressions contain pattern comprehensions or patterns.
        // Only rebuild expressions if patterns need to be extracted.
        fn has_patterns(node: &DynNode<ExprIR<Variable>>) -> bool {
            match node.data() {
                ExprIR::PatternComprehension(_) | ExprIR::Pattern(_) => true,
                _ => node.children().any(|c| has_patterns(&c)),
            }
        }
        let needs_extraction = exprs.iter().any(|(_, e)| has_patterns(&e.root()))
            || orderby.iter().any(|(e, _)| has_patterns(&e.root()));

        // Extract pattern comprehensions from all projection expressions BEFORE
        // clearing visited — the sub-plans need to know which variables are
        // already bound by preceding clauses (e.g., MATCH).
        let scope_id = exprs.first().map_or(0, |e| e.0.scope_id);
        // Pattern comprehension variables live in the pre-projection scope
        // because the Apply sub-plans execute and merge results into that
        // scope's Env (before the projection creates a new Env).
        let pre_scope_id = scope_id.saturating_sub(1);
        let mut all_extracted = Vec::new();
        let exprs: Vec<_> = if needs_extraction {
            exprs
                .into_iter()
                .map(|(var, expr)| {
                    let rebuilt = self.extract_pattern_comprehensions(
                        &expr.root(),
                        pre_scope_id,
                        &mut all_extracted,
                    );
                    (var, Arc::new(rebuilt) as QueryExpr<Variable>)
                })
                .collect()
        } else {
            exprs
        };
        // Also extract from orderby expressions
        let orderby: Vec<_> = if needs_extraction {
            orderby
                .into_iter()
                .map(|(expr, desc)| {
                    let rebuilt = self.extract_pattern_comprehensions(
                        &expr.root(),
                        pre_scope_id,
                        &mut all_extracted,
                    );
                    (Arc::new(rebuilt) as QueryExpr<Variable>, desc)
                })
                .collect()
        } else {
            orderby
        };

        // Build Apply + Aggregate sub-plans for each extracted pattern comprehension.
        // This uses the CURRENT (pre-clear) visited set so plan_match knows which
        // variables are already bound by the outer stream.
        let mut apply_plans = Vec::new();
        for (var, graph, where_filter, result_expr, paths) in &all_extracted {
            let sub_plan = self.build_pattern_comprehension_plan(
                var,
                graph,
                where_filter.as_ref(),
                result_expr,
                paths,
            );
            apply_plans.push((var.clone(), sub_plan));
        }

        // Now clear visited set for the new scope — after WITH/RETURN, only the
        // projected (and copied) variables are in scope.
        self.visited.clear();
        for expr in &exprs {
            self.visited.insert((expr.0.id, expr.0.scope_id));
        }
        for (new_var, _) in &copy_from_parent {
            self.visited.insert((new_var.id, new_var.scope_id));
        }
        for (var, _) in &apply_plans {
            self.visited.insert((var.id, var.scope_id));
        }

        // If any expression uses an aggregation function, produce an
        // Aggregate node that separates group-by keys from aggregations.
        // Otherwise, produce a simple Project node.
        let mut res = if exprs.iter().any(|e| e.1.is_aggregation()) {
            let mut group_by_keys = Vec::new();
            let mut aggregations = Vec::new();
            let mut names = Vec::new();
            for (name, expr) in exprs {
                names.push(name.clone());
                if expr.is_aggregation() {
                    aggregations.push((name, expr));
                } else {
                    group_by_keys.push((name, expr));
                }
            }
            tree!(IR::Aggregate {
                names,
                keys: group_by_keys,
                aggregations,
                projections: copy_from_parent
            })
        } else {
            tree!(IR::Project {
                exprs,
                copies: copy_from_parent
            })
        };
        // If this clause follows write operations, insert a Commit node
        // so mutations are flushed before the projection reads results.
        if write {
            res.root_mut().push_child(IR::Commit);
        }

        // Insert Apply + Aggregate sub-plans below the Project/Aggregate.
        // Each Apply wraps the input stream with one sub-plan: Apply(input, sub_plan).
        // Multiple pattern comprehensions chain:
        //   Project -> Apply_outer(Apply_inner(input, sub2), sub1)
        // Build bottom-up: last sub-plan is innermost (closest to input).
        if !apply_plans.is_empty() {
            // Find the deepest child slot: if res has a Commit child, go below it.
            let mut insert_idx = res.root().idx();
            if res.node(insert_idx).num_children() > 0
                && matches!(res.node(insert_idx).child(0).data(), IR::Commit)
            {
                insert_idx = res.node(insert_idx).child(0).idx();
            }
            // Build the innermost Apply first (last sub_plan), then wrap outward.
            // The innermost Apply starts with sub_plan as its sole child;
            // plan_query stitching inserts the preceding clause as child(0),
            // giving the standard 2-child layout: Apply(input, sub_plan).
            let mut apply_chain: Option<DynTree<IR>> = None;
            for (_var, sub_plan) in apply_plans.into_iter().rev() {
                let apply = if let Some(inner) = apply_chain {
                    tree!(IR::Apply, inner, sub_plan)
                } else {
                    tree!(IR::Apply, sub_plan)
                };
                apply_chain = Some(apply);
            }
            if let Some(chain) = apply_chain {
                res.node_mut(insert_idx).push_child_tree(chain);
            }
        }
        if distinct && !matches!(res.root().data(), IR::Aggregate { .. }) {
            res = tree!(IR::Distinct, res);
        }
        if !orderby.is_empty() {
            res = tree!(IR::Sort(orderby), res);
        }
        if let Some(skip_expr) = skip {
            res = tree!(IR::Skip(skip_expr), res);
        }
        if let Some(limit_expr) = limit {
            res = tree!(IR::Limit(limit_expr), res);
        }
        // WITH ... WHERE filter (not applicable to RETURN, which passes None).
        // Same pattern-predicate decomposition as in plan_match.
        if let Some(filter) = filter {
            let mut extractable = vec![];
            let mut inline = HashMap::new();
            let rebuilt = self.collect_patterns_and_rebuild(
                &filter.root(),
                &mut extractable,
                &mut inline,
                true,
            );

            if !matches!(rebuilt.root().data(), ExprIR::Bool(true)) {
                if inline.is_empty() {
                    res = tree!(IR::Filter(Arc::new(rebuilt)), res);
                } else {
                    res = self.expr_to_plan(&rebuilt.root(), &inline, res);
                }
            }

            for (graph, is_anti) in extractable {
                let saved = self.visited.clone();
                let mut sub_plan = self.plan_match(&graph, None);
                self.visited = saved;
                Self::add_argument_to_leaves(&mut sub_plan);
                if is_anti {
                    res = tree!(IR::AntiSemiApply, res, sub_plan);
                } else {
                    res = tree!(IR::SemiApply, res, sub_plan);
                }
            }
        }
        res
    }

    /// Assemble a multi-clause query plan from individual clause plans.
    ///
    /// Each Cypher clause (MATCH, WITH, RETURN, CREATE, etc.) is planned
    /// independently first.  The resulting plan trees are then stitched
    /// together in reverse order: the last clause (typically RETURN) becomes
    /// the root, and earlier clauses are inserted as its deepest input.
    ///
    /// The "insertion point" (`idx`) walks past post-processing operators
    /// (Sort, Skip, Limit, Distinct, Filter, semi-apply variants) and past
    /// Project/Aggregate → Commit, to find the spot where the preceding
    /// clause's output should feed in.
    fn plan_query(
        &mut self,
        q: Vec<QueryIR<Variable>>,
        write: bool,
    ) -> DynTree<IR> {
        // Plan each clause independently.
        let mut plans = Vec::with_capacity(q.len());
        for ir in q {
            plans.push(self.plan(ir));
        }
        // Stitch plans together in reverse: start from the last clause's plan
        // (the root), then insert each preceding plan at the deepest input slot.
        let mut iter = plans.into_iter().rev();
        let mut res = iter.next().unwrap();
        // Walk down to find the insertion point past post-processing operators.
        let mut idx = res.root().idx();
        while matches!(res.node(idx).data(), |IR::Sort(_)| IR::Skip(_)
            | IR::Limit(_)
            | IR::Distinct
            | IR::Filter(_)
            | IR::SemiApply
            | IR::AntiSemiApply
            | IR::OrApplyMultiplexer(_))
        {
            idx = res.node(idx).child(0).idx();
        }
        // If we landed on a Project/Aggregate, walk past Commit and Apply
        // children — the preceding clause feeds below them.
        if matches!(
            res.node(idx).data(),
            |IR::Project { .. }| IR::Aggregate { .. }
        ) && res.node(idx).num_children() > 0
        {
            if matches!(res.node(idx).child(0).data(), IR::Commit) {
                idx = res.node(idx).child(0).idx();
            }
            while res.node(idx).num_children() > 0
                && matches!(res.node(idx).child(0).data(), IR::Apply)
            {
                idx = res.node(idx).child(0).idx();
            }
        }
        // Insert each remaining clause plan (in reverse order) at the
        // current insertion point, then walk down again to find the next
        // insertion point for the clause before it.
        for n in iter {
            if matches!(
                res.node(idx).data(),
                IR::CartesianProduct | IR::ValueHashJoin { .. }
            ) && Self::needs_apply_wrapping(&n)
            {
                // When stitching a data-producing clause (LOAD CSV, UNWIND,
                // WITH, etc.) into a CartesianProduct, wrap the CartesianProduct
                // in Apply so that bound variables from the preceding clause
                // propagate via Argument leaves.
                // This matches the FalkorDB C project's approach.
                let cp_children: Vec<_> = res.node(idx).children().map(|c| c.idx()).collect();
                let mut leaves = Vec::new();
                let mut stack: Vec<_> = cp_children;
                while let Some(n) = stack.pop() {
                    let node = res.node(n);
                    if matches!(node.data(), IR::Merge { .. }) {
                        if node.num_children() > 1 {
                            stack.push(node.child(0).idx());
                        }
                        continue;
                    }
                    if node.is_leaf() && !matches!(node.data(), IR::Argument) {
                        leaves.push(n);
                    } else {
                        for i in 0..node.num_children() {
                            stack.push(node.child(i).idx());
                        }
                    }
                }
                for leaf in leaves {
                    res.node_mut(leaf).push_child(IR::Argument);
                }
                res.node_mut(idx).push_parent(IR::Apply);
                idx = res.node_mut(idx).push_sibling_tree(Side::Left, n);
            } else if res.node(idx).num_children() > 0 {
                idx = res
                    .node_mut(idx)
                    .child_mut(0)
                    .push_sibling_tree(Side::Left, n);
            } else {
                idx = res.node_mut(idx).push_child_tree(n);
            }
            while res.node(idx).num_children() > 0
                && matches!(res.node(idx).data(), |IR::Sort(_)| IR::Skip(_)
                    | IR::Limit(_)
                    | IR::Distinct
                    | IR::Filter(_)
                    | IR::SemiApply
                    | IR::AntiSemiApply
                    | IR::OrApplyMultiplexer(_)
                    | IR::CondTraverse { .. }
                    | IR::CondVarLenTraverse { .. }
                    | IR::AllShortestPaths(_)
                    | IR::ExpandInto { .. }
                    | IR::EdgeByIndexScan { .. }
                    | IR::PathBuilder(_))
            {
                idx = res.node(idx).child(0).idx();
            }
            if matches!(
                res.node(idx).data(),
                |IR::Project { .. }| IR::Aggregate { .. }
            ) && res.node(idx).num_children() > 0
            {
                if matches!(res.node(idx).child(0).data(), IR::Commit) {
                    idx = res.node(idx).child(0).idx();
                }
                while res.node(idx).num_children() > 0
                    && matches!(res.node(idx).child(0).data(), IR::Apply)
                {
                    idx = res.node(idx).child(0).idx();
                }
            }
        }

        // For write queries without an explicit WITH/RETURN commit, wrap
        // the entire plan in a top-level Commit.
        if write {
            res = tree!(IR::Commit, res);
        }

        // Ensure every Apply node has exactly 2 children.  The innermost Apply
        // in a pattern-comprehension chain starts with only the sub-plan
        // (1 child) and relies on stitching to insert the preceding clause
        // as child(0).  When there is no preceding clause (bare RETURN), the
        // Apply stays single-child; add an Argument to supply one empty row.
        Self::ensure_apply_has_input(&mut res);

        res
    }

    /// Returns true if a plan `n` being stitched into a CartesianProduct
    /// requires Apply wrapping. Plans from data-producing clauses (LOAD CSV,
    /// UNWIND, WITH/RETURN projections) produce variables that may be referenced
    /// inside the CartesianProduct — these need Apply + Argument propagation.
    /// Plans from MATCH components (scans, traversals) are just additional
    /// cross-product branches and should be inserted as CartesianProduct children.
    fn needs_apply_wrapping(n: &DynTree<IR>) -> bool {
        // Walk to the root of n and check its type.
        // Match-produced plans have scan/traversal/filter/argument at root.
        let mut idx = n.root().idx();
        loop {
            match n.node(idx).data() {
                // Scan/traversal nodes come from MATCH — add as CP child
                IR::NodeByLabelScan(_)
                | IR::AllNodeScan(_)
                | IR::NodeByIndexScan { .. }
                | IR::NodeByIdSeek { .. }
                | IR::NodeByLabelAndIdScan { .. }
                | IR::CondTraverse { .. }
                | IR::CondVarLenTraverse { .. }
                | IR::AllShortestPaths(_)
                | IR::ExpandInto { .. }
                | IR::EdgeByIndexScan { .. }
                | IR::CartesianProduct
                | IR::ValueHashJoin { .. }
                | IR::Argument
                | IR::PathBuilder(_) => return false,
                // Filter, SemiApply, etc. wrap scans — walk through
                IR::Filter(_) | IR::SemiApply | IR::AntiSemiApply | IR::OrApplyMultiplexer(_) => {
                    if n.node(idx).num_children() > 0 {
                        idx = n.node(idx).child(0).idx();
                    } else {
                        return false;
                    }
                }
                // Data-producing clauses need Apply wrapping
                _ => return true,
            }
        }
    }

    /// Walk the plan tree and insert an `Argument` node as child(0) of any
    /// `Apply` that only has one child (the sub-plan).
    fn ensure_apply_has_input(tree: &mut DynTree<IR>) {
        let apply_idxs: Vec<_> = {
            let mut tr = orx_tree::Traversal.bfs().over_nodes();
            tree.root()
                .walk_with(&mut tr)
                .filter(|n| matches!(n.data(), IR::Apply) && n.num_children() == 1)
                .map(|n| n.idx())
                .collect()
        };
        for idx in apply_idxs {
            let sub_plan_idx = tree.node(idx).child(0).idx();
            tree.node_mut(sub_plan_idx)
                .push_sibling_tree(Side::Left, DynTree::new(IR::Argument));
        }
    }

    /// Main entry point: convert a single bound query IR node into an execution plan.
    ///
    /// Each `QueryIR` variant maps to one or more IR operators.  Compound
    /// queries (`QueryIR::Query`) are handled by `plan_query`, which stitches
    /// multiple clause plans together.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn plan(
        &mut self,
        ir: BoundQueryIR,
    ) -> DynTree<IR> {
        match ir {
            // CALL procedure: special-case fulltext index procedures into
            // native CreateIndex/DropIndex IR nodes.
            QueryIR::Call {
                func: proc,
                args: exprs,
                yields: named_outputs,
                yield_aliases: _,
                filter,
                explicit_yield: _yielded,
            } => {
                if proc.name == "db.idx.fulltext.drop" {
                    let ExprIR::String(label) = exprs[0].root().data() else {
                        unreachable!()
                    };
                    return tree!(IR::DropIndex {
                        label: label.clone(),
                        attrs: vec![],
                        index_type: IndexType::Fulltext,
                        entity_type: EntityType::Node,
                    });
                }
                // Resolve a yield slot by its canonical procedure-field name.
                // Variable.name carries the original field name (the alias-
                // before-AS) regardless of `YIELD … AS …` renaming, so this
                // lookup is order- and alias-independent. The binder
                // guarantees the entity field is yielded for these
                // procedures, so `node` / `relationship` are always present.
                let yield_by_field = |field: &str| -> Option<Variable> {
                    named_outputs
                        .iter()
                        .find(|v| v.name.as_ref().is_some_and(|n| n.as_str() == field))
                        .cloned()
                };
                if proc.name == "db.idx.fulltext.queryNodes" {
                    let scan = tree!(IR::NodeByFulltextScan {
                        node: yield_by_field("node")
                            .expect("binder ensures 'node' is yielded for queryNodes"),
                        label: exprs[0].clone(),
                        query: exprs[1].clone(),
                        score: yield_by_field("score"),
                    });
                    return if let Some(filter) = filter {
                        tree!(IR::Filter(filter), scan)
                    } else {
                        scan
                    };
                }
                if proc.name == "db.idx.fulltext.queryRelationships" {
                    let scan = tree!(IR::EdgeByFulltextScan {
                        edge: yield_by_field("relationship").expect(
                            "binder ensures 'relationship' is yielded for queryRelationships"
                        ),
                        label: exprs[0].clone(),
                        query: exprs[1].clone(),
                        score: yield_by_field("score"),
                    });
                    return if let Some(filter) = filter {
                        tree!(IR::Filter(filter), scan)
                    } else {
                        scan
                    };
                }
                if proc.name == "db.idx.vector.queryNodes" {
                    let scan = tree!(IR::NodeByVectorScan {
                        node: yield_by_field("node")
                            .expect("binder ensures 'node' is yielded for queryNodes"),
                        label: exprs[0].clone(),
                        attr: exprs[1].clone(),
                        k: exprs[2].clone(),
                        vector: exprs[3].clone(),
                        score: yield_by_field("score"),
                    });
                    return if let Some(filter) = filter {
                        tree!(IR::Filter(filter), scan)
                    } else {
                        scan
                    };
                }
                if proc.name == "db.idx.vector.queryRelationships" {
                    let scan = tree!(IR::EdgeByVectorScan {
                        edge: yield_by_field("relationship").expect(
                            "binder ensures 'relationship' is yielded for queryRelationships"
                        ),
                        label: exprs[0].clone(),
                        attr: exprs[1].clone(),
                        k: exprs[2].clone(),
                        vector: exprs[3].clone(),
                        score: yield_by_field("score"),
                    });
                    return if let Some(filter) = filter {
                        tree!(IR::Filter(filter), scan)
                    } else {
                        scan
                    };
                }
                if let Some(filter) = filter {
                    return tree!(
                        IR::Filter(filter),
                        tree!(IR::ProcedureCall {
                            func: proc,
                            args: exprs,
                            yields: named_outputs
                        })
                    );
                }
                tree!(IR::ProcedureCall {
                    func: proc,
                    args: exprs,
                    yields: named_outputs
                })
            }
            // MATCH / OPTIONAL MATCH
            QueryIR::Match {
                pattern,
                filter,
                optional,
            } => {
                if optional {
                    // Compute optional variables BEFORE plan_match adds them to visited,
                    // so we know which variables to null-pad when no match is found.
                    let optional_vars: Vec<Variable> = pattern
                        .variables()
                        .filter(|v| !self.visited.contains(&(v.id, v.scope_id)))
                        .collect();
                    let all_visited = pattern
                        .variables()
                        .all(|v| self.visited.contains(&(v.id, v.scope_id)));
                    let mut match_plan = self.plan_match(&pattern, filter);
                    Self::add_argument_to_leaves(&mut match_plan);
                    // If all pattern variables are already bound from a prior clause,
                    // we need an Apply (correlated join) so the inner plan re-evaluates
                    // the pattern for each incoming row.  Otherwise, the Optional node
                    // directly wraps the match plan and handles null-padding.
                    if all_visited {
                        tree!(IR::Apply, tree!(IR::Optional(optional_vars), match_plan))
                    } else {
                        tree!(IR::Optional(optional_vars), match_plan)
                    }
                } else {
                    let all_visited = pattern
                        .variables()
                        .all(|v| self.visited.contains(&(v.id, v.scope_id)));
                    let match_plan = self.plan_match(&pattern, filter);
                    // If all pattern variables are already bound, we need
                    // Apply so each incoming row feeds the inner plan via
                    // set_argument_batch (Argument leaves are runtime-only
                    // leaf nodes that don't pull from children).
                    if all_visited {
                        let mut inner = match_plan;
                        Self::add_argument_to_leaves(&mut inner);
                        tree!(IR::Apply, inner)
                    } else {
                        match_plan
                    }
                }
            }
            QueryIR::Unwind { expr, var: alias } => {
                self.visited.insert((alias.id, alias.scope_id));
                tree!(IR::Unwind { expr, var: alias })
            }
            // MERGE: try to match the full pattern first; the Merge IR node
            // decides at runtime whether to create the missing parts.
            // filter_visited strips already-bound entities from the create pattern.
            QueryIR::Merge {
                pattern,
                on_create: on_create_set_items,
                on_match: on_match_set_items,
            } => {
                let create_pattern = pattern.filter_visited(&self.visited);
                let mut match_branch = self.plan_match(&pattern, None);
                Self::add_argument_to_leaves(&mut match_branch);

                tree!(
                    IR::Merge {
                        pattern: create_pattern,
                        on_create: on_create_set_items,
                        on_match: on_match_set_items
                    },
                    match_branch
                )
            }
            // CREATE: only create entities not already bound.
            QueryIR::Create(pattern) => {
                let filtered = pattern.filter_visited(&self.visited);
                // Add created variables to visited so subsequent clauses
                // (e.g. FOREACH body) know they're already bound.
                for v in pattern.variables() {
                    self.visited.insert((v.id, v.scope_id));
                }
                tree!(IR::Create(filtered))
            }
            QueryIR::Delete {
                exprs,
                detach: is_detach,
            } => tree!(IR::Delete {
                exprs,
                detach: is_detach
            }),
            QueryIR::Set(items) => tree!(IR::Set(items)),
            QueryIR::Remove(items) => tree!(IR::Remove(items)),
            QueryIR::LoadCsv {
                file_path,
                headers,
                delimiter,
                var,
            } => {
                self.visited.insert((var.id, var.scope_id));
                tree!(IR::LoadCsv {
                    file_path,
                    headers,
                    delimiter,
                    var,
                })
            }
            // WITH clause: projection that also introduces a new scope.
            // May include WHERE filter with pattern predicates.
            QueryIR::With {
                distinct,
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                filter,
                write,
                ..
            } => self.plan_project(
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                filter,
                distinct,
                write,
            ),
            // RETURN clause: final projection (no WHERE filter).
            QueryIR::Return {
                distinct,
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                write,
                ..
            } => self.plan_project(
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                None,
                distinct,
                write,
            ),
            QueryIR::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options,
            } => tree!(IR::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options
            }),
            QueryIR::DropIndex {
                label,
                attrs,
                index_type,
                entity_type,
            } => {
                tree!(IR::DropIndex {
                    label,
                    attrs,
                    index_type,
                    entity_type
                })
            }
            // Multi-clause query: plan each clause and stitch together.
            QueryIR::Query { clauses: q, write } => self.plan_query(q, write),
            QueryIR::Union { branches, all } => {
                let mut res = tree!(IR::Union; branches.into_iter().map(|branch| {
                    let mut planner = Self::default();
                    planner.plan(branch)
                }));
                if !all {
                    res = tree!(IR::Distinct, res);
                }
                res
            }
            QueryIR::CallSubquery {
                body,
                is_returning,
                remap,
            } => {
                let saved_visited = self.visited.clone();

                // Plan the inner body
                let mut inner_plan = self.plan(*body);

                // Add Argument leaves for correlated execution
                Self::add_argument_to_leaves(&mut inner_plan);

                // Restore visited, then add returned var IDs so subsequent
                // clauses know these variables are bound.
                self.visited = saved_visited;
                if is_returning {
                    if remap.is_empty() {
                        // No remapping needed (shouldn't happen for returning subqueries)
                        match inner_plan.root().data() {
                            IR::Project { exprs: vars, .. } => {
                                for (var, _) in vars {
                                    self.visited.insert((var.id, var.scope_id));
                                }
                            }
                            IR::Aggregate { names, .. } => {
                                for var in names {
                                    self.visited.insert((var.id, var.scope_id));
                                }
                            }
                            _ => {}
                        }
                        tree!(IR::Apply, inner_plan)
                    } else {
                        // Build a Project that remaps inner return IDs to outer IDs.
                        // This prevents inner scope IDs from colliding with outer
                        // variable IDs in the Apply merge.
                        let exprs: Vec<(Variable, QueryExpr<Variable>)> = remap
                            .iter()
                            .map(|(inner_var, outer_var)| {
                                let expr =
                                    Arc::new(DynTree::new(ExprIR::Variable(inner_var.clone())));
                                (outer_var.clone(), expr)
                            })
                            .collect();
                        let project = tree!(
                            IR::Project {
                                exprs,
                                copies: vec![]
                            },
                            inner_plan
                        );
                        for (_, outer_var) in &remap {
                            self.visited.insert((outer_var.id, outer_var.scope_id));
                        }
                        tree!(IR::Apply, project)
                    }
                } else {
                    // Non-returning: side-effect only, wrap in Optional so
                    // outer row survives even if inner produces nothing
                    tree!(IR::Apply, tree!(IR::Optional(vec![]), inner_plan))
                }
            }
            QueryIR::ForEach {
                list: list_expr,
                var,
                body,
            } => {
                // Add the loop variable to visited so body clauses (MERGE, CREATE)
                // know it's already bound and don't create new entities for it.
                let saved_visited = self.visited.clone();
                self.visited.insert((var.id, var.scope_id));

                // Plan the body clauses as a sub-plan
                let body_plans: Vec<DynTree<IR>> =
                    body.into_iter().map(|clause| self.plan(clause)).collect();

                // Restore visited to pre-FOREACH state
                self.visited = saved_visited;

                // Stitch body plans together (same as plan_query stitching)
                let mut body_iter = body_plans.into_iter().rev();
                let mut body_plan = body_iter.next().unwrap();
                let mut idx = body_plan.root().idx();
                for n in body_iter {
                    if body_plan.node(idx).num_children() > 0 {
                        idx = body_plan
                            .node_mut(idx)
                            .child_mut(0)
                            .push_sibling_tree(Side::Left, n);
                    } else {
                        idx = body_plan.node_mut(idx).push_child_tree(n);
                    }
                }
                // Do NOT wrap in Commit — mutations accumulate in pending
                // across all iterations and are committed by the outer Commit
                // after the entire FOREACH completes.
                // Add Argument leaves so the body gets the loop env
                Self::add_argument_to_leaves(&mut body_plan);
                tree!(
                    IR::ForEach {
                        list: list_expr,
                        var
                    },
                    body_plan
                )
            }
        }
    }
}
