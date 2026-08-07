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

use crate::planner::optimizer::collect_expr_variables;
use crate::runtime::functions::{FnType, Type, get_functions};
use crate::runtime::value::Value;
use crate::tree;

use orx_tree::{Bfs, Dyn, DynNode, DynTree, NodeIdx, NodeRef, Side, Traversal, Traverser};

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
    /// Receives input from parent operator.
    ///
    /// The payload lists the variables the argument rows are known to bind,
    /// as `(id, scope_id)` pairs: `Some(vars)` means the incoming rows bind
    /// exactly these variables, `None` means unknown (optimizers must stay
    /// conservative). The scope must be carried — `fresh_var` numbers ids
    /// per scope (`id = scope_vars[scope_id].len()`), so a bare id is
    /// ambiguous across scopes.
    Argument(Option<Vec<(u32, u32)>>),
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
    /// CREATE pattern. Boxed: `QueryGraph` is 72 bytes inline.
    Create(Box<QueryGraph<Arc<String>, Arc<String>, Variable>>),
    /// MERGE pattern with ON CREATE/ON MATCH actions.
    /// Pattern boxed: with the inline 72-byte `QueryGraph` plus the two
    /// `Vec`s this variant was 120 bytes — the size cap of the whole `IR`
    /// enum, which every plan-tree node carries.
    Merge {
        pattern: Box<QueryGraph<Arc<String>, Arc<String>, Variable>>,
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
    /// Scan nodes by label.
    /// Scan all nodes with a given label.
    NodeByLabelScan {
        node: Arc<QueryNode<Arc<String>, Variable>>,
    },
    /// Wraps a scan to include pending-created nodes and exclude pending-deleted
    /// nodes. Used inside MERGE match sub-plans so they see in-flight mutations.
    /// The optimizer is free to rewrite the child scan (e.g. to an index scan).
    IncludePending {
        node: Arc<QueryNode<Arc<String>, Variable>>,
    },
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
        /// Additional hops fused by `fuse_anonymous_traverse`, in traversal
        /// order. Empty for a single-hop CondTraverse. Each chain hop is an
        /// anonymous-edge, anonymous-intermediate-node traversal — only the
        /// final hop's `to` alias is bound at runtime.
        chain: Vec<Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>>,
        /// When true this traverse carries OPTIONAL MATCH semantics (fused
        /// from an `Optional` wrapper by `fuse_optional_traverse`): input rows
        /// producing no expansion are emitted with the traverse-introduced
        /// variables (edge + destination) bound to NULL.
        optional: bool,
    },
    /// Variable-length traversal (BFS) from known nodes
    CondVarLenTraverse {
        relationship: Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>,
        /// Optional per-hop edge filter absorbed from a WHERE clause by the optimizer.
        edge_filter: Option<QueryExpr<Variable>>,
        /// When false, the path/relationship-list binding (`relationship.alias`)
        /// is not consumed by any ancestor, so the operator skips materializing
        /// the per-row `Value::Path`. Conservatively `true` at planning time;
        /// lowered to `false` by the `reduce_var_len_path` optimizer pass.
        emit_path: bool,
        /// Named-path variable this traverse binds directly (in addition to the
        /// relationship alias) when the whole named path is exactly this one
        /// var-len pattern, letting the planner skip the `PathBuilder` op.
        path_var: Option<Variable>,
        /// True when both endpoints were already bound at planning time, so the
        /// traverse only verifies reachability between two known nodes
        /// ("Expand Into" in EXPLAIN output).
        expand_into: bool,
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
pub(super) fn expr_has_non_deterministic(expr: &DynTree<ExprIR<Variable>>) -> bool {
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

/// Formats a relationship for variable-length traverse display, e.g.
/// `(n)-[e:R*1..INF]->(m)`.
///
/// Unlike the fixed-length traversals the hop range is part of the printed
/// pattern, and the arrow always points along the traversal direction — an
/// undirected pattern is still expanded from `from` towards `to`.
fn fmt_var_len_rel(rel: &QueryRelationship<Arc<String>, Arc<String>, Variable>) -> String {
    use itertools::Itertools;

    let alias = &rel.alias;
    let types = if rel.types.is_empty() {
        String::new()
    } else {
        format!(":{}", rel.types.iter().join("|"))
    };

    let min_hops = rel.min_hops.unwrap_or(1);
    let hops = if min_hops == 1 && rel.max_hops == Some(1) {
        String::new()
    } else {
        let max_hops = rel
            .max_hops
            .map_or_else(|| "INF".to_owned(), |m| m.to_string());
        format!("*{min_hops}..{max_hops}")
    };

    format!(
        "({})-[{alias}{types}{hops}]->({})",
        rel.from.alias, rel.to.alias
    )
}

/// Formats a relationship for CondTraverse/ExpandInto display.
/// Shows node labels and hides anonymous edge aliases.
fn fmt_rel_with_labels(rel: &QueryRelationship<Arc<String>, Arc<String>, Variable>) -> String {
    fmt_rel_with_labels_dir(rel, false)
}

fn fmt_rel_with_labels_dir(
    rel: &QueryRelationship<Arc<String>, Arc<String>, Variable>,
    transposed: bool,
) -> String {
    use itertools::Itertools;

    let (left_arrow, right_arrow) = if rel.bidirectional {
        ("", "")
    } else if transposed {
        ("<", "")
    } else {
        ("", ">")
    };

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
        format!("({from_str}){left_arrow}-{right_arrow}({to_str})")
    } else if rel.types.is_empty() {
        let alias = &rel.alias;
        format!("({from_str}){left_arrow}-[{alias}]-{right_arrow}({to_str})")
    } else {
        let alias = &rel.alias;
        let types = rel.types.iter().join("|");
        format!("({from_str}){left_arrow}-[{alias}:{types}]-{right_arrow}({to_str})")
    }
}

#[cfg_attr(tarpaulin, skip)]
impl Display for IR {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Argument(_) => write!(f, "Argument"),
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
            Self::NodeByLabelScan { node, .. } => {
                write!(f, "Node By Label Scan | {node}")
            }
            Self::IncludePending { node } => {
                write!(f, "Include Pending | {node}")
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
                relationship: rel,
                transposed,
                chain,
                optional,
                ..
            } => {
                let name = if *optional {
                    "Optional Conditional Traverse"
                } else {
                    "Conditional Traverse"
                };
                if chain.is_empty() {
                    write!(f, "{name} | {}", fmt_rel_with_labels_dir(rel, *transposed))
                } else {
                    write!(
                        f,
                        "{name} | {} (+ {} fused hop{})",
                        fmt_rel_with_labels_dir(rel, *transposed),
                        chain.len(),
                        if chain.len() == 1 { "" } else { "s" }
                    )
                }
            }
            Self::CondVarLenTraverse {
                relationship: rel,
                expand_into,
                ..
            } => {
                let name = if *expand_into {
                    "Conditional Variable Length Traverse (Expand Into)"
                } else {
                    "Conditional Variable Length Traverse"
                };
                write!(f, "{name} | {}", fmt_var_len_rel(rel))
            }
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

/// Builds an equality filter expression from inline attributes on a pattern.
///
/// Given an alias and attrs like `{name: 'Alice', age: 30}`, returns
/// `alias.name = 'Alice' AND alias.age = 30`. Returns `None` if empty.
pub(super) fn inline_attrs_to_filter(
    alias: &Variable,
    attrs: &DynTree<ExprIR<Variable>>,
) -> Option<DynTree<ExprIR<Variable>>> {
    let mut filters: Vec<DynTree<ExprIR<Variable>>> = vec![];

    for attr in attrs.root().children() {
        let ExprIR::Constant(Value::String(attr_str)) = attr.data() else {
            unreachable!("inline attrs map children must be ExprIR::Constant(Value::String) keys");
        };
        let eq = tree!(
            ExprIR::Eq,
            tree!(
                ExprIR::Property(attr_str.clone()),
                tree!(ExprIR::Variable(alias.clone()))
            ),
            attr.child(0).as_cloned_subtree()
        );
        filters.push(eq);
    }

    if filters.is_empty() {
        None
    } else if filters.len() == 1 {
        Some(filters.pop().unwrap())
    } else {
        Some(tree!(ExprIR::And; filters))
    }
}

/// Build a `hasLabels(var, [label1, label2, ...])` filter expression.
fn has_labels_filter(
    var: &Variable,
    labels: &[Arc<String>],
) -> DynTree<ExprIR<Variable>> {
    let has_labels_fn = get_functions()
        .get("hasLabels", &FnType::Function)
        .expect("hasLabels function must exist");
    tree!(
        ExprIR::FuncInvocation(has_labels_fn),
        tree!(ExprIR::Variable(var.clone())),
        tree!(ExprIR::List; labels.iter().map(|l| tree!(ExprIR::Constant(Value::String(l.clone())))))
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
    /// Labels already enforced for each bound variable (by a scan, a
    /// traversal's label mask, or an explicit hasLabels filter).  Labels on
    /// an already-bound variable that are missing from this set (e.g.
    /// clause-local labels from an OPTIONAL MATCH on a bound alias) must be
    /// re-verified with a hasLabels filter.
    verified_labels: HashMap<(u32, u32), OrderSet<Arc<String>>>,
}

/// A pattern comprehension (or inline pattern) hoisted out of a projection
/// expression, together with the comprehensions nested inside its own predicate
/// and result expression.
struct ExtractedComprehension {
    /// Variable the comprehension's collected list is bound to.
    var: Variable,
    /// Pattern the comprehension iterates over.
    graph: QueryGraph<Arc<String>, Arc<String>, Variable>,
    /// Optional `WHERE` predicate.
    where_filter: Option<Arc<DynTree<ExprIR<Variable>>>>,
    /// Expression collected for each match.
    result_expr: Arc<DynTree<ExprIR<Variable>>>,
    /// Paths to materialize before evaluating `result_expr`.
    paths: Vec<Arc<QueryPath<Variable>>>,
    /// Comprehensions nested within `where_filter` / `result_expr`, which may
    /// reference variables bound by `graph` and so must be applied inside this
    /// comprehension's sub-plan.
    nested: Vec<ExtractedComprehension>,
}

/// What `extract_pattern_comprehensions` does with a bare existential
/// `Pattern` node.  Pattern comprehensions are always extracted; only
/// existential patterns are ambiguous, because they are a boolean in a
/// predicate but a list of paths anywhere else.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PatternMode {
    /// Collect the matched paths into a list variable.  Used where the
    /// expression's value is what matters: projections, UNWIND, FOREACH.
    Collect,
    /// Leave the pattern in place for `collect_patterns_and_rebuild`, which
    /// turns it into a SemiApply / AntiSemiApply.  This is the preferred
    /// form in a WHERE predicate: a semi-join stops at the first match
    /// instead of materializing every path.
    SemiApply,
    /// Collect as in `Collect`, but yield `size(paths) > 0` so the result
    /// still reads as a boolean.  A `SemiApply` filters the whole row, so it
    /// cannot stand in for one operand of an operator like `XOR`; `Predicate`
    /// degrades to this once the walk descends past such an operator.
    Exists,
}

impl PatternMode {
    /// The mode to use for the children of `parent`.
    ///
    /// `SemiApply` only survives under the operators `expr_to_plan` can take
    /// apart.  Anywhere else the pattern is an ordinary operand and has to
    /// become a value.
    fn descend(
        self,
        parent: &ExprIR<Variable>,
    ) -> Self {
        if self == Self::SemiApply
            && !matches!(
                parent,
                ExprIR::And | ExprIR::Or | ExprIR::Not | ExprIR::Paren
            )
        {
            Self::Exists
        } else {
            self
        }
    }
}

impl Planner {
    #[must_use]
    pub fn new(scope_vars: Vec<Vec<Variable>>) -> Self {
        Self {
            visited: HashSet::new(),
            scope_vars,
            verified_labels: HashMap::new(),
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
    fn add_argument_to_leaves(
        tree: &mut DynTree<IR>,
        bound_vars: Option<Vec<(u32, u32)>>,
    ) {
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
            if node.is_leaf() && !matches!(node.data(), IR::Argument(_)) {
                leaves.push(idx);
            } else {
                for i in 0..node.num_children() {
                    stack.push(node.child(i).idx());
                }
            }
        }

        // Add Argument node as a child to each leaf.
        for leaf_idx in leaves {
            tree.node_mut(leaf_idx)
                .push_child(IR::Argument(bound_vars.clone()));
        }
    }

    /// Wrap every `NodeByLabelScan` (and `AllNodeScan`) in `tree` with an
    /// `IncludePending` parent node. Must be called BEFORE `add_argument_to_leaves`
    /// so scans are still leaves when we restructure them.
    fn set_include_pending_on_scans(tree: &mut DynTree<IR>) {
        let indices = tree.root().indices::<Bfs>().collect::<Vec<_>>();
        for idx in indices {
            if !matches!(
                tree.node(idx).data(),
                IR::NodeByLabelScan { .. } | IR::AllNodeScan(_)
            ) {
                continue;
            }
            let original_data = std::mem::replace(
                tree.node_mut(idx).data_mut(),
                IR::Argument(None), // temporary placeholder
            );
            let node = match &original_data {
                IR::NodeByLabelScan { node } | IR::AllNodeScan(node) => node.clone(),
                _ => unreachable!(),
            };
            *tree.node_mut(idx).data_mut() = IR::IncludePending { node };
            tree.node_mut(idx).push_child(original_data);
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

    /// Does this expression tree contain a pattern comprehension or an
    /// existential pattern that has to be planned as a sub-plan?
    fn has_pattern_expr(node: &DynNode<ExprIR<Variable>>) -> bool {
        match node.data() {
            ExprIR::PatternComprehension(_) | ExprIR::Pattern(_) => true,
            _ => node.children().any(|c| Self::has_pattern_expr(&c)),
        }
    }

    /// Is this an `Apply` that already holds its input stream?
    ///
    /// A WHERE-predicate pattern comprehension is applied as
    /// `Apply(input, sub_plan)`, so stitching has to descend past it to reach
    /// the input's leaves.  A single-child `Apply` is the opposite case — it is
    /// still waiting for stitching to supply child(0) — so it must be left
    /// alone.
    fn is_saturated_apply(
        tree: &DynTree<IR>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> bool {
        matches!(tree.node(idx).data(), IR::Apply) && tree.node(idx).num_children() > 1
    }

    /// Walk past the Apply chain a `ForEach` or `Unwind` carries for pattern
    /// comprehensions in its list expression, so the preceding clause is
    /// stitched below the sub-plans rather than as an extra child.
    ///
    /// A freshly planned `ForEach` holds its body as the last child, so only a
    /// second child can be the chain; a freshly planned `Unwind` has no child
    /// at all, so any child it has is the chain.
    fn descend_list_expr_applies(
        tree: &DynTree<IR>,
        mut idx: NodeIdx<Dyn<IR>>,
    ) -> NodeIdx<Dyn<IR>> {
        let min_children = match tree.node(idx).data() {
            IR::ForEach { .. } => 2,
            IR::Unwind { .. } => 1,
            _ => return idx,
        };
        if tree.node(idx).num_children() >= min_children
            && matches!(tree.node(idx).child(0).data(), IR::Apply)
        {
            idx = tree.node(idx).child(0).idx();
            while Self::is_saturated_apply(tree, idx) {
                idx = tree.node(idx).child(0).idx();
            }
        }
        idx
    }

    /// Build a pattern sub-plan for a graph, saving and restoring visited state.
    fn build_pattern_sub_plan(
        &mut self,
        graph: &QueryGraph<Arc<String>, Arc<String>, Variable>,
    ) -> DynTree<IR> {
        let saved = self.visited.clone();
        let saved_verified = self.verified_labels.clone();
        let mut sub_plan = self.plan_match(graph, None);
        self.visited = saved;
        self.verified_labels = saved_verified;
        Self::add_argument_to_leaves(&mut sub_plan, None);
        sub_plan
    }

    /// Walk an expression tree and replace `PatternComprehension` / `Pattern`
    /// nodes with fresh variable references.  Returns the rebuilt expression
    /// and a list of extracted comprehensions ready for plan building.
    ///
    /// Comprehensions found inside another comprehension's predicate or result
    /// expression are collected into that comprehension's `nested` list rather
    /// than the caller's, because their patterns may reference variables the
    /// enclosing comprehension binds (e.g. `p` in
    /// `[(i)-->(p) | [(p)-->(o) | o.name]]`).  Those sub-plans therefore have to
    /// be applied inside the enclosing comprehension's sub-plan.
    ///
    /// `mode` decides what happens to bare existential `Pattern` nodes; see
    /// [`PatternMode`].  Pattern comprehensions are always extracted.
    fn extract_pattern_comprehensions(
        &mut self,
        node: &DynNode<ExprIR<Variable>>,
        scope_id: u32,
        extracted: &mut Vec<ExtractedComprehension>,
        mode: PatternMode,
    ) -> DynTree<ExprIR<Variable>> {
        match node.data() {
            ExprIR::PatternComprehension(graph) => {
                let var = self.fresh_var(scope_id, Type::List(Box::new(Type::Any)));

                let mut nested = Vec::new();
                let where_tree = {
                    let t = self.extract_pattern_comprehensions(
                        &node.child(0),
                        scope_id,
                        &mut nested,
                        mode,
                    );
                    if matches!(t.root().data(), ExprIR::Constant(Value::Bool(true))) {
                        None
                    } else {
                        Some(Arc::new(t))
                    }
                };
                let result_tree = Arc::new(self.extract_pattern_comprehensions(
                    &node.child(1),
                    scope_id,
                    &mut nested,
                    mode,
                ));

                extracted.push(ExtractedComprehension {
                    var: var.clone(),
                    graph: graph.as_ref().clone(),
                    where_filter: where_tree,
                    result_expr: result_tree,
                    paths: vec![],
                    nested,
                });
                DynTree::new(ExprIR::Variable(var))
            }
            ExprIR::Pattern(graph) if mode != PatternMode::SemiApply => {
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

                extracted.push(ExtractedComprehension {
                    var: var.clone(),
                    graph: graph.as_ref().clone(),
                    where_filter: None,
                    result_expr: Arc::new(DynTree::new(ExprIR::Variable(path_var))),
                    paths: vec![query_path],
                    nested: vec![],
                });
                if mode == PatternMode::Exists {
                    // The pattern was a predicate, so hand the caller a
                    // boolean rather than the list of matched paths.
                    let mut length = DynTree::new(ExprIR::Length);
                    length
                        .root_mut()
                        .push_child_tree(DynTree::new(ExprIR::Variable(var)));
                    let mut gt = DynTree::new(ExprIR::Gt);
                    gt.root_mut().push_child_tree(length);
                    gt.root_mut().push_child(ExprIR::Constant(Value::Int(0)));
                    gt
                } else {
                    DynTree::new(ExprIR::Variable(var))
                }
            }
            _ => {
                let child_mode = mode.descend(node.data());
                let mut new_tree = DynTree::new(node.data().clone());
                for child in node.children() {
                    let child_tree = self
                        .extract_pattern_comprehensions(&child, scope_id, extracted, child_mode);
                    new_tree.root_mut().push_child_tree(child_tree);
                }
                new_tree
            }
        }
    }

    /// Extract the pattern comprehensions of a clause's list expression
    /// (`UNWIND`, `FOREACH`) into their own sub-plans.
    ///
    /// Returns the rewritten expression — each comprehension replaced by the
    /// variable holding its collected list — and an Apply chain to hang below
    /// the clause's operator.  The innermost Apply is deliberately left
    /// single-child: `plan_query` stitching inserts the preceding clause there
    /// as child(0) (see `descend_list_expr_applies`).
    fn extract_list_expr_comprehensions(
        &mut self,
        expr: QueryExpr<Variable>,
        scope_id: u32,
    ) -> (QueryExpr<Variable>, Option<DynTree<IR>>) {
        if !Self::has_pattern_expr(&expr.root()) {
            return (expr, None);
        }

        let mut extracted = Vec::new();
        let rebuilt = self.extract_pattern_comprehensions(
            &expr.root(),
            scope_id,
            &mut extracted,
            PatternMode::Collect,
        );
        // Build the innermost Apply first (last comprehension), then wrap
        // outward, so the chain reads Apply(Apply(input, sub2), sub1).
        let mut chain: Option<DynTree<IR>> = None;
        for comprehension in extracted.iter().rev() {
            let sub_plan = self.build_pattern_comprehension_plan(comprehension);
            chain = Some(match chain {
                Some(inner) => tree!(IR::Apply, inner, sub_plan),
                None => tree!(IR::Apply, sub_plan),
            });
        }
        for c in &extracted {
            self.visited.insert((c.var.id, c.var.scope_id));
        }
        (Arc::new(rebuilt), chain)
    }

    /// Extract the parts of a WHERE predicate that need their own sub-plan:
    /// every pattern comprehension, plus any existential pattern sitting where
    /// `expr_to_plan` cannot reach it with a SemiApply (an `XOR` operand, a
    /// comparison, a function argument, …).  Each is replaced in the predicate
    /// by an expression over a variable, and returned as a sub-plan the caller
    /// must `Apply` below the `Filter` so that variable is bound by the time
    /// the predicate runs.
    ///
    /// Existential patterns the decomposer *can* reach are left alone, so the
    /// common `WHERE (a)-->(b)` shapes keep their cheaper semi-join plan.
    fn extract_filter_comprehensions(
        &mut self,
        filter: QueryExpr<Variable>,
        scope_id: u32,
    ) -> (QueryExpr<Variable>, Vec<DynTree<IR>>) {
        fn needs_extraction(
            node: &DynNode<ExprIR<Variable>>,
            mode: PatternMode,
        ) -> bool {
            match node.data() {
                ExprIR::PatternComprehension(_) => true,
                ExprIR::Pattern(_) => mode == PatternMode::Exists,
                other => {
                    let child_mode = mode.descend(other);
                    node.children().any(|c| needs_extraction(&c, child_mode))
                }
            }
        }
        if !needs_extraction(&filter.root(), PatternMode::SemiApply) {
            return (filter, vec![]);
        }

        let mut extracted = Vec::new();
        let rebuilt = self.extract_pattern_comprehensions(
            &filter.root(),
            scope_id,
            &mut extracted,
            PatternMode::SemiApply,
        );
        let sub_plans: Vec<DynTree<IR>> = extracted
            .iter()
            .map(|c| self.build_pattern_comprehension_plan(c))
            .collect();
        for c in &extracted {
            self.visited.insert((c.var.id, c.var.scope_id));
        }
        (Arc::new(rebuilt), sub_plans)
    }

    /// Build the Apply + Aggregate sub-plan for a single pattern comprehension.
    ///
    /// Returns a plan tree:  `Aggregate(collect(result_expr)) -> traversal -> Argument`
    fn build_pattern_comprehension_plan(
        &mut self,
        comprehension: &ExtractedComprehension,
    ) -> DynTree<IR> {
        use crate::runtime::functions::{FnType, get_functions};

        let ExtractedComprehension {
            var,
            graph,
            where_filter,
            result_expr,
            paths,
            nested,
        } = comprehension;

        let saved = self.visited.clone();
        let saved_verified = self.verified_labels.clone();
        let mut sub_plan = self.plan_match(graph, None);

        // Add PathBuilder to construct Path values from matched variables.
        if !paths.is_empty() {
            sub_plan = tree!(IR::PathBuilder(paths.to_vec()), sub_plan);
        }

        // Nested comprehensions are planned while this comprehension's own
        // pattern variables are still marked visited, so their patterns bind to
        // them instead of re-scanning, and are applied above the traversal (but
        // below the WHERE filter, which may itself reference them).
        for inner in nested {
            let inner_plan = self.build_pattern_comprehension_plan(inner);
            sub_plan = tree!(IR::Apply, sub_plan, inner_plan);
        }

        self.visited = saved;
        self.verified_labels = saved_verified;

        // Add WHERE filter if present
        if let Some(filter) = where_filter {
            sub_plan = tree!(IR::Filter(filter.clone()), sub_plan);
        }

        Self::add_argument_to_leaves(&mut sub_plan, None);

        // Build collect(result_expr) aggregation expression
        let collect_fn = get_functions()
            .get(
                "collect",
                &FnType::Aggregation {
                    initial: Value::Null,
                    finalizer: None,
                    batch_agg: None,
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
            let inner = self.expr_to_plan(&child, inline_map, tree!(IR::Argument(None)));
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
                let branch = tree!(IR::Filter(Arc::new(expr_tree)), tree!(IR::Argument(None)));
                scalar_branches.push(branch);
                continue;
            }
            // Complex child (AND with patterns, nested OR, etc.):
            // Recursively build a sub-plan starting from Argument.
            let branch = self.expr_to_plan(&child, inline_map, tree!(IR::Argument(None)));
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
                if !matches!(child.data(), ExprIR::Constant(Value::Bool(true))) {
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
                    extractable.push((graph.as_ref().clone(), false));
                    DynTree::new(ExprIR::Constant(Value::Bool(true)))
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
                    inline.insert(var.id, graph.as_ref().clone());
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
                        extractable.push((graph.as_ref().clone(), true));
                        return DynTree::new(ExprIR::Constant(Value::Bool(true)));
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
                    inline.insert(var.id, graph.as_ref().clone());
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

    /// Record that `labels` have been enforced for `var` in the current stream.
    fn mark_labels_verified<'l>(
        &mut self,
        var: &Variable,
        labels: impl Iterator<Item = &'l Arc<String>>,
    ) {
        let mut labels = labels.peekable();
        if labels.peek().is_none() {
            return;
        }
        let entry = self
            .verified_labels
            .entry((var.id, var.scope_id))
            .or_default();
        for l in labels {
            if !entry.contains(l) {
                entry.insert(l.clone());
            }
        }
    }

    /// Labels on `node` that have not yet been enforced for its
    /// (already-bound) variable.
    fn unverified_labels(
        &self,
        node: &QueryNode<Arc<String>, Variable>,
    ) -> Vec<Arc<String>> {
        let verified = self
            .verified_labels
            .get(&(node.alias.id, node.alias.scope_id));
        node.labels
            .iter()
            .filter(|l| !verified.is_some_and(|v| v.contains(*l)))
            .cloned()
            .collect()
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
        // Variables the WHERE predicate constrains. Used below to order the
        // hops of a component so constrained endpoints are reached first.
        let filter_vars: HashSet<u32> = filter
            .as_ref()
            .map(|f| collect_expr_variables(f))
            .unwrap_or_default();
        // Collect extra filters for bound variables with new constraints
        // (labels or inline properties). These are applied as top-level
        // filters rather than as plan components, so they don't interfere
        // with stitching.
        let mut bound_filters: Vec<DynTree<ExprIR<Variable>>> = vec![];
        for component in pattern.connected_components() {
            let relationships = component.relationships();
            // Endpoints already bound by earlier clauses may carry labels
            // that were never enforced by a scan or traversal (e.g.
            // clause-local labels from an OPTIONAL MATCH on a bound alias).
            // Verify them with an explicit hasLabels filter.
            for rel in relationships {
                for endpoint in [&rel.from, &rel.to] {
                    if self
                        .visited
                        .contains(&(endpoint.alias.id, endpoint.alias.scope_id))
                    {
                        let missing = self.unverified_labels(endpoint);
                        if !missing.is_empty() {
                            bound_filters.push(has_labels_filter(&endpoint.alias, &missing));
                            self.mark_labels_verified(&endpoint.alias, missing.iter());
                        }
                    }
                }
            }
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
                // Among the hops leaving that known node, prefer the one that
                // reaches a constrained endpoint, so its filter runs right
                // after the first hop instead of after the whole pattern has
                // been traversed (matches FalkorDB C behaviour). Components
                // with no bound entry point are left alone: there
                // `select_scan_node` picks the starting endpoint and reverses
                // the chain, so the order chosen here does not decide where
                // the filters land.
                let unfiltered = if has_bound == 0 {
                    i32::from(![&r.from, &r.to].iter().any(|n| {
                        n.attrs.root().num_children() > 0 || filter_vars.contains(&n.alias.id)
                    }))
                } else {
                    0
                };
                (base, has_bound, unfiltered)
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
                    let attr_filter = inline_attrs_to_filter(&node.alias, &node.attrs);
                    if let Some(filter_expr) = attr_filter {
                        bound_filters.push(filter_expr);
                    }
                    if node.labels.is_empty() {
                        vec.push(tree!(IR::Argument(None)));
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
                            tree!(IR::Argument(None))
                        ));
                        self.mark_labels_verified(&node.alias, node.labels.iter());
                    }
                } else {
                    let attr_filter = inline_attrs_to_filter(&node.alias, &node.attrs);
                    // The binder's post-processing already set the full
                    // accumulated label set on each QueryNode directly.
                    let mut res = if node.labels.is_empty() {
                        tree!(IR::AllNodeScan(node.clone()))
                    } else {
                        // Multi-label node: the runtime's get_nodes()
                        // intersects all label matrices, so we can pass
                        // all labels directly to NodeByLabelScan.
                        tree!(IR::NodeByLabelScan { node: node.clone() })
                    };
                    if let Some(filter_expr) = attr_filter {
                        res = tree!(IR::Filter(Arc::new(filter_expr)), res);
                    }
                    self.visited.insert((node.alias.id, node.alias.scope_id));
                    self.mark_labels_verified(&node.alias, node.labels.iter());
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
                    let from_attr_filter =
                        inline_attrs_to_filter(&relationship.from.alias, &relationship.from.attrs);
                    let mut scan = if relationship.from.clone().labels.is_empty() {
                        tree!(IR::AllNodeScan(relationship.from.clone()))
                    } else {
                        tree!(IR::NodeByLabelScan {
                            node: relationship.from.clone(),
                        })
                    };
                    if let Some(filter_expr) = from_attr_filter {
                        scan = tree!(IR::Filter(Arc::new(filter_expr)), scan);
                    }
                    Some(scan)
                };
                // Both endpoints already bound (and distinct): the traverse only
                // verifies reachability between two known nodes.
                let expand_into = scan_child.is_none()
                    && relationship.from.alias.id != relationship.to.alias.id
                    && self
                        .visited
                        .contains(&(relationship.to.alias.id, relationship.to.alias.scope_id));
                scan_child.map_or_else(
                    || {
                        tree!(IR::CondVarLenTraverse {
                            relationship: relationship.clone(),
                            edge_filter: None,
                            emit_path: true,
                            path_var: None,
                            expand_into,
                        })
                    },
                    |scan| {
                        tree!(
                            IR::CondVarLenTraverse {
                                relationship: relationship.clone(),
                                edge_filter: None,
                                emit_path: true,
                                path_var: None,
                                expand_into,
                            },
                            scan
                        )
                    },
                )
            } else if relationship.from.alias.id == relationship.to.alias.id {
                // Self-loop with fixed-length edge: scan + ExpandInto.
                // If the node is already bound (visited), don't rescan — the
                // Argument from the surrounding Apply will provide the binding.
                let already_bound = self
                    .visited
                    .contains(&(relationship.from.alias.id, relationship.from.alias.scope_id));
                if already_bound {
                    let attr_filter =
                        inline_attrs_to_filter(&relationship.from.alias, &relationship.from.attrs);
                    let mut ei = tree!(IR::ExpandInto {
                        relationship: relationship.clone(),
                        emit_relationship: emit_rel(relationship),
                        sibling_edges: sibling_edges.clone()
                    });
                    if let Some(filter_expr) = attr_filter {
                        ei = tree!(IR::Filter(Arc::new(filter_expr)), ei);
                    }
                    ei
                } else {
                    let attr_filter =
                        inline_attrs_to_filter(&relationship.from.alias, &relationship.from.attrs);
                    let mut scan = if relationship.from.clone().labels.is_empty() {
                        tree!(IR::AllNodeScan(relationship.from.clone()))
                    } else {
                        tree!(IR::NodeByLabelScan {
                            node: relationship.from.clone(),
                        })
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
                }
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
                let from_attr_filter =
                    inline_attrs_to_filter(&relationship.from.alias, &relationship.from.attrs);
                if let Some(filter_expr) = from_attr_filter {
                    ei = tree!(IR::Filter(Arc::new(filter_expr)), ei);
                }
                let to_attr_filter =
                    inline_attrs_to_filter(&relationship.to.alias, &relationship.to.attrs);
                if let Some(filter_expr) = to_attr_filter {
                    ei = tree!(IR::Filter(Arc::new(filter_expr)), ei);
                }
                ei
            } else {
                let edge_attr_filter =
                    inline_attrs_to_filter(&relationship.alias, &relationship.attrs);
                let mut ct = tree!(IR::CondTraverse {
                    relationship: relationship.clone(),
                    emit_relationship: emit_rel(relationship),
                    sibling_edges: sibling_edges.clone(),
                    transposed: false,
                    chain: Vec::new(),
                    optional: false,
                });
                if let Some(filter_expr) = edge_attr_filter {
                    ct = tree!(IR::Filter(Arc::new(filter_expr)), ct);
                }
                ct
            };
            // Check destination node for inline attributes (e.g., (b {val: 'v2'}))
            // and add a Filter if present.
            if !self
                .visited
                .contains(&(relationship.to.alias.id, relationship.to.alias.scope_id))
            {
                let to_attr_filter =
                    inline_attrs_to_filter(&relationship.to.alias, &relationship.to.attrs);
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
                let from_attr_filter =
                    inline_attrs_to_filter(&relationship.from.alias, &relationship.from.attrs);
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
            self.mark_labels_verified(&relationship.from.alias, relationship.from.labels.iter());
            self.mark_labels_verified(&relationship.to.alias, relationship.to.labels.iter());
            // Chain remaining relationships in the component, each one
            // stacking on top of the previous result using the same logic.
            for relationship in iter {
                res = if relationship.all_shortest_paths != AllShortestPaths::No {
                    tree!(IR::AllShortestPaths(relationship.clone()), res)
                } else if relationship.min_hops.is_some() {
                    let expand_into = relationship.from.alias.id != relationship.to.alias.id
                        && self.visited.contains(&(
                            relationship.from.alias.id,
                            relationship.from.alias.scope_id,
                        ))
                        && self
                            .visited
                            .contains(&(relationship.to.alias.id, relationship.to.alias.scope_id));
                    let mut cvlt = tree!(
                        IR::CondVarLenTraverse {
                            relationship: relationship.clone(),
                            edge_filter: None,
                            emit_path: true,
                            path_var: None,
                            expand_into,
                        },
                        res
                    );
                    if !self
                        .visited
                        .contains(&(relationship.from.alias.id, relationship.from.alias.scope_id))
                    {
                        let from_attr_filter = inline_attrs_to_filter(
                            &relationship.from.alias,
                            &relationship.from.attrs,
                        );
                        if let Some(filter_expr) = from_attr_filter {
                            cvlt = tree!(IR::Filter(Arc::new(filter_expr)), cvlt);
                        }
                    }
                    cvlt
                } else if relationship.from.alias.id == relationship.to.alias.id {
                    let already_bound = self
                        .visited
                        .contains(&(relationship.from.alias.id, relationship.from.alias.scope_id));
                    if already_bound {
                        let attr_filter = inline_attrs_to_filter(
                            &relationship.from.alias,
                            &relationship.from.attrs,
                        );
                        let mut ei = tree!(
                            IR::ExpandInto {
                                relationship: relationship.clone(),
                                emit_relationship: emit_rel(relationship),
                                sibling_edges: sibling_edges.clone()
                            },
                            res
                        );
                        if let Some(filter_expr) = attr_filter {
                            ei = tree!(IR::Filter(Arc::new(filter_expr)), ei);
                        }
                        ei
                    } else {
                        let attr_filter = inline_attrs_to_filter(
                            &relationship.from.alias,
                            &relationship.from.attrs,
                        );
                        let mut scan = if relationship.from.clone().labels.is_empty() {
                            tree!(IR::AllNodeScan(relationship.from.clone()))
                        } else {
                            tree!(IR::NodeByLabelScan {
                                node: relationship.from.clone(),
                            })
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
                    }
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
                    let from_attr_filter =
                        inline_attrs_to_filter(&relationship.from.alias, &relationship.from.attrs);
                    if let Some(filter_expr) = from_attr_filter {
                        ei = tree!(IR::Filter(Arc::new(filter_expr)), ei);
                    }
                    let to_attr_filter =
                        inline_attrs_to_filter(&relationship.to.alias, &relationship.to.attrs);
                    if let Some(filter_expr) = to_attr_filter {
                        ei = tree!(IR::Filter(Arc::new(filter_expr)), ei);
                    }
                    ei
                } else {
                    let edge_attr_filter =
                        inline_attrs_to_filter(&relationship.alias, &relationship.attrs);
                    let mut ct = tree!(
                        IR::CondTraverse {
                            relationship: relationship.clone(),
                            emit_relationship: emit_rel(relationship),
                            sibling_edges: sibling_edges.clone(),
                            transposed: false,
                            chain: Vec::new(),
                            optional: false,
                        },
                        res
                    );
                    if let Some(filter_expr) = edge_attr_filter {
                        ct = tree!(IR::Filter(Arc::new(filter_expr)), ct);
                    }
                    ct
                };
                // Check destination node for inline attributes (e.g., (b {val: 'v2'}))
                // and add a Filter if present.
                if !self
                    .visited
                    .contains(&(relationship.to.alias.id, relationship.to.alias.scope_id))
                {
                    let to_attr_filter =
                        inline_attrs_to_filter(&relationship.to.alias, &relationship.to.attrs);
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
                self.mark_labels_verified(
                    &relationship.from.alias,
                    relationship.from.labels.iter(),
                );
                self.mark_labels_verified(&relationship.to.alias, relationship.to.labels.iter());
            }
            let paths = component.paths();
            if !paths.is_empty() {
                // A named path covering exactly one forward var-len pattern is
                // bound directly by the CondVarLenTraverse (which already
                // materializes the full path value), skipping PathBuilder.
                let mut remaining = Vec::with_capacity(paths.len());
                for path in paths {
                    let rel = relationships.first();
                    let elidable = relationships.len() == 1
                        && rel.is_some_and(|rel| {
                            rel.min_hops.is_some()
                                && rel.all_shortest_paths == AllShortestPaths::No
                                && path.vars.len() == 3
                                && path.vars[0].id == rel.from.alias.id
                                && path.vars[1].id == rel.alias.id
                                && path.vars[2].id == rel.to.alias.id
                        });
                    if elidable {
                        let mut bound = false;
                        let indices: Vec<_> = res.root().indices::<Bfs>().collect();
                        for idx in indices {
                            if let IR::CondVarLenTraverse {
                                relationship,
                                path_var: path_var @ None,
                                ..
                            } = res.node_mut(idx).data_mut()
                                && relationship.alias.id == path.vars[1].id
                            {
                                *path_var = Some(path.var.clone());
                                bound = true;
                                break;
                            }
                        }
                        if bound {
                            continue;
                        }
                    }
                    remaining.push(path.clone());
                }
                if !remaining.is_empty() {
                    res = tree!(IR::PathBuilder(remaining), res);
                }
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
            // Pattern comprehensions in the predicate get their own sub-plans,
            // applied below the Filter so their result lists are bound first.
            let scope_id = pattern.variables().next().map_or(0, |v| v.scope_id);
            let (filter, comprehension_plans) =
                self.extract_filter_comprehensions(filter, scope_id);
            for sub_plan in comprehension_plans {
                res = tree!(IR::Apply, res, sub_plan);
            }

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
            } else if !matches!(rebuilt.root().data(), ExprIR::Constant(Value::Bool(true))) {
                res = tree!(IR::Filter(Arc::new(rebuilt)), res);
            }

            // Apply SemiApply/AntiSemiApply for each extractable pattern
            for (graph, is_anti) in extractable {
                let saved = self.visited.clone();
                let saved_verified = self.verified_labels.clone();
                let mut sub_plan = self.plan_match(&graph, None);
                self.visited = saved;
                self.verified_labels = saved_verified;
                Self::add_argument_to_leaves(&mut sub_plan, None);
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
        let needs_extraction = exprs.iter().any(|(_, e)| Self::has_pattern_expr(&e.root()))
            || orderby
                .iter()
                .any(|(e, _)| Self::has_pattern_expr(&e.root()));

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
                        PatternMode::Collect,
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
                        PatternMode::Collect,
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
        for comprehension in &all_extracted {
            let sub_plan = self.build_pattern_comprehension_plan(comprehension);
            apply_plans.push((comprehension.var.clone(), sub_plan));
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

        // Carry verified labels through the projection for node variables
        // projected as-is; everything else belongs to the old scope.
        let mut carried_verified = HashMap::new();
        for (new_var, expr) in &exprs {
            if let ExprIR::Variable(old) = expr.root().data()
                && let Some(v) = self.verified_labels.get(&(old.id, old.scope_id))
            {
                carried_verified.insert((new_var.id, new_var.scope_id), v.clone());
            }
        }
        for (new_var, old_var) in &copy_from_parent {
            if let Some(v) = self.verified_labels.get(&(old_var.id, old_var.scope_id)) {
                carried_verified.insert((new_var.id, new_var.scope_id), v.clone());
            }
        }
        self.verified_labels = carried_verified;

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
            // The predicate runs after the projection, so its pattern
            // comprehensions resolve against the projected scope.
            let (filter, comprehension_plans) =
                self.extract_filter_comprehensions(filter, scope_id);
            for sub_plan in comprehension_plans {
                res = tree!(IR::Apply, res, sub_plan);
            }

            let mut extractable = vec![];
            let mut inline = HashMap::new();
            let rebuilt = self.collect_patterns_and_rebuild(
                &filter.root(),
                &mut extractable,
                &mut inline,
                true,
            );

            if !matches!(rebuilt.root().data(), ExprIR::Constant(Value::Bool(true))) {
                if inline.is_empty() {
                    res = tree!(IR::Filter(Arc::new(rebuilt)), res);
                } else {
                    res = self.expr_to_plan(&rebuilt.root(), &inline, res);
                }
            }

            for (graph, is_anti) in extractable {
                let saved = self.visited.clone();
                let saved_verified = self.verified_labels.clone();
                let mut sub_plan = self.plan_match(&graph, None);
                self.visited = saved;
                self.verified_labels = saved_verified;
                Self::add_argument_to_leaves(&mut sub_plan, None);
                if is_anti {
                    res = tree!(IR::AntiSemiApply, res, sub_plan);
                } else {
                    res = tree!(IR::SemiApply, res, sub_plan);
                }
            }
        }
        res
    }

    /// Renames the output variables of a subquery body's root `Project` to the
    /// outer scope's variables, so no extra `Project` is needed on top of it.
    ///
    /// Only the root operator is rewritten, so nothing above it can still refer
    /// to the inner names. Returns `false` — leaving `plan` untouched — when the
    /// root is not a plain `Project`, or when `remap` doesn't cover every one of
    /// its outputs; the caller then falls back to a dedicated remapping
    /// `Project`.
    fn rename_projection_outputs(
        plan: &mut DynTree<IR>,
        remap: &[(Variable, Variable)],
    ) -> bool {
        let outer = |v: &Variable| {
            remap
                .iter()
                .find(|(inner, _)| inner.id == v.id && inner.scope_id == v.scope_id)
                .map(|(_, outer)| outer.clone())
        };
        let idx = plan.root().idx();
        let IR::Project { exprs, copies } = plan.node(idx).data() else {
            return false;
        };
        if !exprs.iter().all(|(v, _)| outer(v).is_some())
            || !copies.iter().all(|(_, v)| outer(v).is_some())
        {
            return false;
        }
        let mut node = plan.node_mut(idx);
        let IR::Project { exprs, copies } = node.data_mut() else {
            unreachable!()
        };
        for (var, _) in exprs.iter_mut() {
            *var = outer(var).expect("checked above");
        }
        for (_, var) in copies.iter_mut() {
            *var = outer(var).expect("checked above");
        }
        true
    }

    /// True when an `OPTIONAL MATCH` clause contributes nothing to the plan.
    ///
    /// A pattern made only of already-bound nodes — no relationships, no named
    /// paths — cannot introduce a new variable, and `OPTIONAL MATCH` never
    /// drops an input row: with nothing to null-pad, every row passes through
    /// unchanged. Labels, inline properties and the clause's `WHERE` are
    /// therefore unobservable, so the whole clause is dropped rather than
    /// planned as an `Apply` over an `Optional` that always matches. Mirrors
    /// the early exit in FalkorDB C's `ExecutionPlan_ProcessPattern`.
    fn is_redundant_optional_match(
        &self,
        ir: &QueryIR<Variable>,
    ) -> bool {
        let QueryIR::Match {
            pattern,
            optional: true,
            ..
        } = ir
        else {
            return false;
        };
        pattern.relationships().is_empty()
            && pattern.paths().is_empty()
            && !pattern.nodes().is_empty()
            && pattern
                .nodes()
                .iter()
                .all(|n| self.visited.contains(&(n.alias.id, n.alias.scope_id)))
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
        // Plan each clause independently. A redundant OPTIONAL MATCH is
        // skipped so no operators are emitted for it; `plans` still ends up
        // non-empty because a query can never conclude with a MATCH clause
        // (enforced by `QueryIR::inner_validate`). Skipping also leaves
        // `visited` untouched, which is correct: every variable the clause
        // mentions is already bound.
        let mut plans = Vec::with_capacity(q.len());
        for ir in q {
            if self.is_redundant_optional_match(&ir) {
                continue;
            }
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
            || Self::is_saturated_apply(&res, idx)
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
        idx = Self::descend_list_expr_applies(&res, idx);
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
                    if node.is_leaf() && !matches!(node.data(), IR::Argument(_)) {
                        leaves.push(n);
                    } else {
                        for i in 0..node.num_children() {
                            stack.push(node.child(i).idx());
                        }
                    }
                }
                for leaf in leaves {
                    res.node_mut(leaf).push_child(IR::Argument(None));
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
            while (res.node(idx).num_children() > 0
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
                    | IR::PathBuilder(_)))
                || Self::is_saturated_apply(&res, idx)
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
            idx = Self::descend_list_expr_applies(&res, idx);
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
                IR::NodeByLabelScan { .. }
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
                | IR::Argument(_)
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
                .push_sibling_tree(Side::Left, DynTree::new(IR::Argument(None)));
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
                    let ExprIR::Constant(Value::String(label)) = exprs[0].root().data() else {
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
                    let any_visited = pattern
                        .variables()
                        .any(|v| self.visited.contains(&(v.id, v.scope_id)));
                    // Label checks made inside an OPTIONAL MATCH only hold
                    // within its subplan (unmatched rows flow past them), so
                    // they are discarded once the clause is planned.
                    let saved_verified = self.verified_labels.clone();
                    let mut match_plan = self.plan_match(&pattern, filter);
                    self.verified_labels = saved_verified;
                    // If any pattern variable is already bound from a prior clause,
                    Self::add_argument_to_leaves(&mut match_plan, None);
                    // If all pattern variables are already bound from a prior clause,
                    // we need an Apply (correlated join) so the inner plan re-evaluates
                    // the pattern for each incoming row.  Otherwise, the Optional node
                    // directly wraps the match plan and handles null-padding.
                    if any_visited {
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
                        Self::add_argument_to_leaves(&mut inner, None);
                        tree!(IR::Apply, inner)
                    } else {
                        match_plan
                    }
                }
            }
            QueryIR::Unwind { expr, var: alias } => {
                // A pattern comprehension in the unwound expression needs its
                // own traversal sub-plan; extract it before the alias is bound,
                // since the comprehension resolves against the incoming stream.
                let (expr, apply_chain) =
                    self.extract_list_expr_comprehensions(expr, alias.scope_id);
                self.visited.insert((alias.id, alias.scope_id));
                let mut res = tree!(IR::Unwind { expr, var: alias });
                if let Some(chain) = apply_chain {
                    res.root_mut().push_child_tree(chain);
                }
                res
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
                // Snapshot before plan_match: it adds the pattern's own
                // variables to `visited`, which are NOT bound by the
                // incoming Argument rows. Keep the scope on each entry —
                // ids are only unique within a scope.
                let bound: Vec<(u32, u32)> = self.visited.iter().copied().collect();
                let mut match_branch = self.plan_match(&pattern, None);
                Self::set_include_pending_on_scans(&mut match_branch);
                // The Argument rows bind exactly the variables visited so
                // far; passing them lets the scan-selection optimizer prove
                // when the match branch is uncorrelated and add a scan.
                Self::add_argument_to_leaves(&mut match_branch, Some(bound));

                let paths = pattern.paths();
                let merge = tree!(
                    IR::Merge {
                        pattern: Box::new(create_pattern),
                        on_create: on_create_set_items,
                        on_match: on_match_set_items
                    },
                    match_branch
                );

                if paths.is_empty() {
                    merge
                } else {
                    tree!(IR::PathBuilder(paths.to_vec()), merge)
                }
            }
            // CREATE: only create entities not already bound.
            QueryIR::Create(pattern) => {
                let filtered = pattern.filter_visited(&self.visited);
                // Add created variables to visited so subsequent clauses
                // (e.g. FOREACH body) know they're already bound.
                for v in pattern.variables() {
                    self.visited.insert((v.id, v.scope_id));
                }
                tree!(IR::Create(Box::new(filtered)))
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
                    // Each branch is an independent stream (fresh `visited`),
                    // but must share the binder's scope table: pattern-predicate
                    // decomposition mints synthetic variables by indexing
                    // `scope_vars`, which panics on an empty Default table.
                    let mut planner = Self::new(self.scope_vars.clone());
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
                Self::add_argument_to_leaves(&mut inner_plan, None);

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
                        // The inner return IDs must become outer IDs, otherwise
                        // inner scope IDs collide with outer variable IDs in the
                        // Apply merge. Prefer renaming the body's own RETURN
                        // projection in place; only stack a dedicated remapping
                        // Project on top when that isn't possible (matches
                        // FalkorDB C's plan shape).
                        let renamed = Self::rename_projection_outputs(&mut inner_plan, &remap);
                        for (_, outer_var) in &remap {
                            self.visited.insert((outer_var.id, outer_var.scope_id));
                        }
                        if renamed {
                            tree!(IR::Apply, inner_plan)
                        } else {
                            let exprs: Vec<(Variable, QueryExpr<Variable>)> = remap
                                .iter()
                                .map(|(inner_var, outer_var)| {
                                    let expr =
                                        Arc::new(DynTree::new(ExprIR::Variable(inner_var.clone())));
                                    (outer_var.clone(), expr)
                                })
                                .collect();
                            tree!(
                                IR::Apply,
                                tree!(
                                    IR::Project {
                                        exprs,
                                        copies: vec![]
                                    },
                                    inner_plan
                                )
                            )
                        }
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
                // A pattern comprehension in the list expression can't be
                // evaluated inline — it needs its own traversal sub-plan.
                // Extract it while `visited` still reflects the outer scope
                // (the loop variable isn't bound in the list expression), so
                // the sub-plan binds to the variables the preceding clauses
                // produced.  Each extraction becomes an Apply below the
                // ForEach; the list expression is left referencing the
                // collected result variable.
                let (list_expr, apply_chain) =
                    self.extract_list_expr_comprehensions(list_expr, var.scope_id);

                // Add the loop variable to visited so body clauses (MERGE, CREATE)
                // know it's already bound and don't create new entities for it.
                // The comprehension variables are already in `visited`, and
                // stay there past the restore: their Apply sub-plans run below
                // the ForEach, on the outer stream.
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
                let mut idx = Self::descend_list_expr_applies(&body_plan, body_plan.root().idx());
                for n in body_iter {
                    if body_plan.node(idx).num_children() > 0 {
                        idx = body_plan
                            .node_mut(idx)
                            .child_mut(0)
                            .push_sibling_tree(Side::Left, n);
                    } else {
                        idx = body_plan.node_mut(idx).push_child_tree(n);
                    }
                    idx = Self::descend_list_expr_applies(&body_plan, idx);
                }
                // Do NOT wrap in Commit — mutations accumulate in pending
                // across all iterations and are committed by the outer Commit
                // after the entire FOREACH completes.
                // Add Argument leaves so the body gets the loop env
                Self::add_argument_to_leaves(&mut body_plan, None);

                let mut res = tree!(IR::ForEach {
                    list: list_expr,
                    var
                });
                // The comprehension sub-plans go below the ForEach, on the
                // input stream, and must precede the body child.
                if let Some(chain) = apply_chain {
                    res.root_mut().push_child_tree(chain);
                }
                res.root_mut().push_child_tree(body_plan);
                res
            }
        }
    }
}
