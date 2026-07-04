//! Abstract Syntax Tree (AST) definitions for Cypher queries.
//!
//! This module defines the intermediate representation (IR) for parsed Cypher
//! queries. The AST is produced by the parser ([`crate::parser::cypher`]) and
//! consumed by the binder ([`crate::planner::binder`]) and planner
//! ([`crate::planner`]).
//!
//! ## Overall Structure
//!
//! A parsed Cypher query is a tree of `QueryIR` clause nodes, each of which
//! may contain expression trees (`DynTree<ExprIR<TVar>>`) and graph pattern
//! structures (`QueryGraph`):
//!
//! ```text
//! QueryIR::Query
//!  |-- QueryIR::Match
//!  |     |-- QueryGraph
//!  |     |     |-- QueryNode  ("n", labels: [Person])
//!  |     |     |-- QueryNode  ("m", labels: [])
//!  |     |     '-- QueryRelationship  ("r", types: [KNOWS], from: n, to: m)
//!  |     '-- filter: DynTree<ExprIR>   (expression tree for WHERE clause)
//!  |
//!  '-- QueryIR::Return
//!        '-- exprs: [("name", DynTree<ExprIR>)]
//!                                |
//!                           Property("name")
//!                                |
//!                           Variable("m")
//! ```
//!
//! ## Key Types
//!
//! - [`Variable`]: A named or anonymous variable with a unique binding ID
//! - [`ExprIR`]: Expression nodes (literals, operators, function calls, etc.)
//! - [`QueryIR`]: Query clause nodes (MATCH, CREATE, RETURN, WITH, etc.)
//! - [`QueryGraph`]: Pattern graph structure containing nodes, relationships,
//!   and named paths
//! - [`QueryExpr`]: Type alias for `Arc<DynTree<ExprIR<TVar>>>` -- a
//!   reference-counted expression tree
//!
//! ## Type Parameters
//!
//! AST types are generic over `TVar` (variable type) to support two stages:
//! - `Arc<String>`: Raw AST before binding -- variables are just names
//!   (type alias: [`RawQueryIR`])
//! - [`Variable`]: Bound AST with resolved variable IDs, scopes, and types
//!   (type alias: [`BoundQueryIR`])
//!
//! ## Expression Trees
//!
//! Expressions are stored as trees using `DynTree<ExprIR<TVar>>` from
//! `orx-tree`. Operators are internal nodes with operands as children,
//! supporting arbitrary expression nesting. For example, `a.age + b.age * 2`:
//!
//! ```text
//!          Add
//!         /   \
//!   Property   Mul
//!   ("age")   /   \
//!     |    Property  Integer(2)
//!  Var("a") ("age")
//!              |
//!           Var("b")
//! ```

use std::{collections::HashSet, fmt::Display, hash::Hash, sync::Arc};

use itertools::Itertools;
use orx_tree::{Dfs, DynTree, NodeRef};

use crate::{
    entity_type::EntityType,
    index::indexer::IndexType,
    runtime::{
        functions::{GraphFn, Type},
        orderset::OrderSet,
        value::Value,
    },
};

/// A variable in a Cypher query, either named or anonymous.
///
/// Variables are assigned unique IDs during binding to distinguish between
/// variables with the same name in different scopes.
///
/// # Fields
/// - `name`: The variable name as it appears in the query (None for anonymous)
/// - `id`: Unique identifier assigned during binding
/// - `scope_id`: The scope in which this variable was defined
/// - `ty`: The inferred or declared type of the variable
#[derive(Clone, Debug)]
pub struct Variable {
    pub name: Option<Arc<String>>,
    pub id: u32,
    pub scope_id: u32,
    pub ty: Type,
}

impl Display for Variable {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{name}")
        } else {
            write!(f, "?{}", self.id)
        }
    }
}

impl PartialEq for Variable {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.id == other.id
    }
}

impl Eq for Variable {}

impl Hash for Variable {
    fn hash<H: std::hash::Hasher>(
        &self,
        state: &mut H,
    ) {
        self.id.hash(state);
    }
}

impl Variable {
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.name.as_ref().map_or("?", |n| n.as_str())
    }
}

/// Expression IR nodes for the Cypher expression tree.
///
/// Expressions form a tree structure where operators are internal nodes and
/// their operands are children. For example, `a + b * c` becomes:
///
/// ```text
///       Add
///      /   \
///     a    Mul
///         /   \
///        b     c
/// ```
///
/// # Type Parameter
/// - `TVar`: Variable type (`Arc<String>` before binding, `Variable` after)
#[derive(Clone, Debug)]
pub enum ExprIR<TVar> {
    /// Literal/constant value. Carries any runtime [`Value`] including
    /// Null/Bool/Int/Float/String literals as well as folded values like
    /// Date, Duration, Map, etc.
    Constant(Value),
    /// List constructor - children are list elements
    List,
    /// Map constructor - children are key-value pairs
    Map,
    /// Variable reference
    Variable(TVar),
    /// Query parameter reference ($param)
    Parameter(String),
    /// Length/size of a list or string
    Length,
    /// Element access (list[index] or map.key)
    GetElement,
    /// Slice access (list[start..end])
    GetElements,
    /// Type check: is value a node?
    IsNode,
    /// Type check: is value a relationship?
    IsRelationship,
    /// Logical OR
    Or,
    /// Logical XOR
    Xor,
    /// Logical AND
    And,
    /// Logical NOT
    Not,
    /// Numeric negation
    Negate,
    /// Equality comparison
    Eq,
    /// Inequality comparison
    Neq,
    /// Less than
    Lt,
    /// Greater than
    Gt,
    /// Less than or equal
    Le,
    /// Greater than or equal
    Ge,
    /// IN operator (element in list)
    In,
    /// Addition or string concatenation
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Power/exponentiation
    Pow,
    /// Modulo
    Modulo,
    /// DISTINCT modifier for expressions
    Distinct,
    /// Property access (e.g., n.prop)
    Property(Arc<String>),
    /// Function call with function definition
    FuncInvocation(Arc<GraphFn>),
    /// List quantifier (all/any/none/single)
    Quantifier {
        quantifier_type: QuantifierType,
        var: TVar,
    },
    /// List comprehension [x IN list | expr]
    ListComprehension(TVar),
    /// Reduce expression: reduce(acc = init, var IN list | expr)
    /// Children: [init_expr, list_expr, body_expr]
    Reduce { accumulator: TVar, iterator: TVar },
    /// Pattern comprehension [(pattern) WHERE cond | expr]
    /// Stores the graph pattern for runtime traversal.
    /// Children: [where_condition, result_expression]
    ///
    /// Boxed: `QueryGraph` is 72 bytes but this variant is rare, so inlining
    /// it would bloat every node of every expression tree.
    PatternComprehension(Box<QueryGraph<Arc<String>, Arc<String>, TVar>>),
    /// Parenthesized expression (for precedence)
    Paren,
    /// Pattern predicate should be rewritten in planner (boxed; see
    /// `PatternComprehension`).
    Pattern(Box<QueryGraph<Arc<String>, Arc<String>, TVar>>),
    /// shortestPath((a)-[*]->(b)) or allShortestPaths((a)-[*]->(b))
    /// Children: [source_var_expr, dest_var_expr]
    ShortestPath {
        rel_types: Vec<Arc<String>>,
        min_hops: u32,
        max_hops: Option<u32>,
        directed: bool,
        all_paths: bool,
    },
    /// Map projection: base { .prop, .*, key: expr, var }
    /// First child is the base expression, remaining children are projection items
    MapProjection,
}

#[cfg_attr(tarpaulin, skip)]
impl<TVar: Display + std::fmt::Debug> Display for ExprIR<TVar> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Constant(v) => match v {
                Value::Null => write!(f, "null"),
                Value::Bool(b) => write!(f, "{b}"),
                Value::Int(i) => write!(f, "{i}"),
                Value::Float(fl) => write!(f, "{fl}"),
                Value::String(s) => write!(f, "{s}"),
                _ => write!(f, "const({v:?})"),
            },
            Self::List => write!(f, "[]"),
            Self::Map => write!(f, "{{}}"),
            Self::Variable(id) => write!(f, "{id}"),
            Self::Parameter(p) => write!(f, "@{p}"),
            Self::Length => write!(f, "length()"),
            Self::GetElement => write!(f, "get_element()"),
            Self::GetElements => write!(f, "get_elements()"),
            Self::IsNode => write!(f, "is_node()"),
            Self::IsRelationship => write!(f, "is_relationship()"),
            Self::Or => write!(f, "or()"),
            Self::Xor => write!(f, "xor()"),
            Self::And => write!(f, "and()"),
            Self::Not => write!(f, "not()"),
            Self::Negate => write!(f, "-negate()"),
            Self::Eq => write!(f, "="),
            Self::Neq => write!(f, "<>"),
            Self::Lt => write!(f, "<"),
            Self::Gt => write!(f, ">"),
            Self::Le => write!(f, "<="),
            Self::Ge => write!(f, ">="),
            Self::In => write!(f, "in()"),
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::Mul => write!(f, "*"),
            Self::Div => write!(f, "/"),
            Self::Pow => write!(f, "^"),
            Self::Modulo => write!(f, "%"),
            Self::Distinct => write!(f, "distinct"),
            Self::Property(prop) => write!(f, "property({prop})"),
            Self::FuncInvocation(func) => write!(f, "{}()", func.name),
            Self::Quantifier {
                quantifier_type,
                var,
            } => {
                write!(f, "{quantifier_type} {var}")
            }
            Self::ListComprehension(var) => {
                write!(f, "list comp({var})")
            }
            Self::Reduce {
                accumulator,
                iterator,
            } => {
                write!(f, "reduce({accumulator}, {iterator})")
            }
            Self::PatternComprehension(_) => {
                write!(f, "pattern comp")
            }
            Self::Paren => write!(f, "()"),
            Self::Pattern(_) => write!(f, "<pattern>"),
            Self::ShortestPath { all_paths, .. } => {
                if *all_paths {
                    write!(f, "allShortestPaths()")
                } else {
                    write!(f, "shortestPath()")
                }
            }
            Self::MapProjection => write!(f, "map_projection"),
        }
    }
}

/// Quantifier types for list predicates (all, any, none, single).
#[derive(Clone, Debug)]
pub enum QuantifierType {
    All,
    Any,
    None,
    Single,
}

#[cfg_attr(tarpaulin, skip)]
impl Display for QuantifierType {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::All => write!(f, "all"),
            Self::Any => write!(f, "any"),
            Self::None => write!(f, "none"),
            Self::Single => write!(f, "single"),
        }
    }
}

/// Trait for checking if an expression contains aggregation functions.
pub trait SupportAggregation {
    /// Returns true if this expression tree contains any aggregation function
    /// (e.g., count, sum, avg, collect).
    fn is_aggregation(&self) -> bool;
}

impl SupportAggregation for DynTree<ExprIR<Variable>> {
    fn is_aggregation(&self) -> bool {
        self.root().indices::<Dfs>().any(|idx| {
            matches!(
                self.node(idx).data(),
                ExprIR::FuncInvocation(func) if func.is_aggregate()
            )
        })
    }
}

impl SupportAggregation for DynTree<ExprIR<Arc<String>>> {
    fn is_aggregation(&self) -> bool {
        self.root().indices::<Dfs>().any(|idx| {
            matches!(
                self.node(idx).data(),
                ExprIR::FuncInvocation(func) if func.is_aggregate()
            )
        })
    }
}

/// A node pattern in a MATCH or CREATE clause.
///
/// Represents patterns like `(n:Person {name: 'Alice'})` where:
/// - `alias` is the variable `n`
/// - `labels` contains `Person`
/// - `attrs` contains the property filter expression
#[derive(Debug)]
pub struct QueryNode<L, TVar> {
    pub alias: TVar,
    pub labels: OrderSet<L>,
    pub attrs: QueryExpr<TVar>,
}

#[cfg_attr(tarpaulin, skip)]
impl<L: Display + PartialEq, TVar: Display + PartialEq> Display for QueryNode<L, TVar> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if self.labels.is_empty() {
            return write!(f, "({})", self.alias);
        }
        write!(f, "({}:{})", self.alias, self.labels.iter().join(":"))
    }
}

impl<L, TVar> QueryNode<L, TVar> {
    #[must_use]
    pub const fn new(
        alias: TVar,
        labels: OrderSet<L>,
        attrs: QueryExpr<TVar>,
    ) -> Self {
        Self {
            alias,
            labels,
            attrs,
        }
    }
}

/// A relationship pattern in a MATCH or CREATE clause.
///
/// Represents patterns like `(a)-[r:KNOWS]->(b)` where:
/// - `alias` is the variable `r`
/// - `types` contains `KNOWS` (can have multiple for OR: `[:A|B]`)
/// - `from` and `to` are the connected nodes
/// - `bidirectional` is true for undirected patterns `-[]-`
/// - `min_hops`/`max_hops` are set for variable-length patterns like `[*1..3]`
#[derive(Debug)]
pub struct QueryRelationship<T, L, TVar> {
    pub alias: TVar,
    pub types: Vec<T>,
    pub attrs: QueryExpr<TVar>,
    pub from: Arc<QueryNode<L, TVar>>,
    pub to: Arc<QueryNode<L, TVar>>,
    pub bidirectional: bool,
    pub min_hops: Option<u32>,
    pub max_hops: Option<u32>,
    pub all_shortest_paths: AllShortestPaths,
}

/// Whether this relationship is part of an allShortestPaths pattern,
/// and if so, whether the result paths need to be reversed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllShortestPaths {
    /// Not an allShortestPaths pattern.
    No,
    /// allShortestPaths with edges in traversal order.
    Forward,
    /// allShortestPaths with edges reversed (incoming pattern like `(a)<-[*]-(b)`).
    Reversed,
}

#[cfg_attr(tarpaulin, skip)]
impl<T: Display, L: Display, TVar: Display> Display for QueryRelationship<T, L, TVar> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let direction = if self.bidirectional { "" } else { ">" };
        if self.types.is_empty() {
            write!(
                f,
                "({})-[{}]-{}({})",
                self.from.alias, self.alias, direction, self.to.alias
            )
        } else {
            write!(
                f,
                "({})-[{}:{}]-{}({})",
                self.from.alias,
                self.alias,
                self.types.iter().join("|"),
                direction,
                self.to.alias
            )
        }
    }
}

impl<T, L, TVar> QueryRelationship<T, L, TVar> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        alias: TVar,
        types: Vec<T>,
        attrs: QueryExpr<TVar>,
        from: Arc<QueryNode<L, TVar>>,
        to: Arc<QueryNode<L, TVar>>,
        bidirectional: bool,
        min_hops: Option<u32>,
        max_hops: Option<u32>,
    ) -> Self {
        Self {
            alias,
            types,
            attrs,
            from,
            to,
            bidirectional,
            min_hops,
            max_hops,
            all_shortest_paths: AllShortestPaths::No,
        }
    }
}

/// A named path pattern in a MATCH clause.
///
/// Represents patterns like `p = (a)-[*]->(b)` where:
/// - `var` is the path variable `p`
/// - `vars` contains all variables in the path pattern
#[derive(Debug)]
pub struct QueryPath<TVar> {
    pub var: TVar,
    pub vars: Vec<TVar>,
}

impl<TVar> QueryPath<TVar> {
    #[must_use]
    pub const fn new(
        var: TVar,
        vars: Vec<TVar>,
    ) -> Self {
        Self { var, vars }
    }
}

/// A graph pattern containing nodes, relationships, and paths.
///
/// This represents the pattern portion of MATCH, CREATE, and MERGE clauses.
/// The graph can be decomposed into connected components for query optimization.
///
/// Uses `Arc` for sharing patterns between different parts of the query plan,
/// avoiding expensive cloning of complex patterns.
#[derive(Clone, Debug)]
pub struct QueryGraph<T, L, TVar> {
    nodes: Vec<Arc<QueryNode<L, TVar>>>,
    relationships: Vec<Arc<QueryRelationship<T, L, TVar>>>,
    paths: Vec<Arc<QueryPath<TVar>>>,
}

impl<T, L, TVar> Default for QueryGraph<T, L, TVar> {
    fn default() -> Self {
        Self {
            nodes: Vec::default(),
            relationships: Vec::default(),
            paths: Vec::default(),
        }
    }
}

#[cfg_attr(tarpaulin, skip)]
impl<T: Display + PartialEq, L: Display + PartialEq, TVar: Display + PartialEq + Eq + Hash> Display
    for QueryGraph<T, L, TVar>
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        for node in &self.nodes {
            write!(f, "{node}, ")?;
        }
        for relationship in &self.relationships {
            write!(f, "{relationship}, ")?;
        }
        for path in &self.paths {
            write!(f, "{}, ", path.var)?;
        }
        Ok(())
    }
}

impl<T, L, TVar: Clone + Hash + Eq> QueryGraph<T, L, TVar> {
    pub fn add_node(
        &mut self,
        node: Arc<QueryNode<L, TVar>>,
    ) -> bool {
        if self.nodes.iter().any(|n| n.alias == node.alias) {
            return false;
        }
        self.nodes.push(node);
        true
    }

    pub fn replace_node(
        &mut self,
        alias: &TVar,
        node: Arc<QueryNode<L, TVar>>,
    ) {
        if let Some(pos) = self.nodes.iter().position(|n| n.alias == *alias) {
            self.nodes[pos] = node;
        }
    }

    pub fn add_relationship(
        &mut self,
        relationship: Arc<QueryRelationship<T, L, TVar>>,
    ) -> bool {
        if self
            .relationships
            .iter()
            .any(|r| r.alias == relationship.alias)
        {
            false
        } else {
            self.relationships.push(relationship);
            true
        }
    }

    pub fn add_path(
        &mut self,
        path: Arc<QueryPath<TVar>>,
    ) -> bool {
        if self.paths.iter().any(|p| p.var == path.var) {
            false
        } else {
            self.paths.push(path);
            true
        }
    }

    pub fn variables(&self) -> impl Iterator<Item = TVar> + '_ {
        self.nodes
            .iter()
            .map(|n| n.alias.clone())
            .chain(self.relationships.iter().map(|r| r.alias.clone()))
            .chain(self.paths.iter().map(|p| p.var.clone()))
    }

    #[must_use]
    pub fn nodes(&self) -> &[Arc<QueryNode<L, TVar>>] {
        &self.nodes
    }

    pub const fn nodes_mut(&mut self) -> &mut Vec<Arc<QueryNode<L, TVar>>> {
        &mut self.nodes
    }

    #[must_use]
    pub fn relationships(&self) -> &[Arc<QueryRelationship<T, L, TVar>>] {
        &self.relationships
    }

    pub const fn relationships_mut(&mut self) -> &mut Vec<Arc<QueryRelationship<T, L, TVar>>> {
        &mut self.relationships
    }

    #[must_use]
    pub fn paths(&self) -> &[Arc<QueryPath<TVar>>] {
        &self.paths
    }
}

impl<T, L> QueryGraph<T, L, Variable> {
    #[must_use]
    pub fn filter_visited(
        &self,
        visited: &HashSet<(u32, u32)>,
    ) -> Self
    where
        T: Default,
        L: Default,
    {
        let mut res = Self::default();
        for node in &self.nodes {
            if !visited.contains(&(node.alias.id, node.alias.scope_id)) {
                res.add_node(node.clone());
            }
        }
        for relationship in &self.relationships {
            if !visited.contains(&(relationship.alias.id, relationship.alias.scope_id)) {
                res.add_relationship(relationship.clone());
            }
        }
        for path in &self.paths {
            if !visited.contains(&(path.var.id, path.var.scope_id)) {
                res.add_path(path.clone());
            }
        }
        res
    }

    #[must_use]
    pub fn connected_components(&self) -> Vec<Self>
    where
        T: Default,
        L: Default,
    {
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for node in &self.nodes {
            if !visited.contains(&node.alias.id) {
                let mut component = Self::default();

                self.dfs(node, &mut visited, &mut component);

                components.push(component);
            }
        }

        components
    }

    fn dfs(
        &self,
        node: &Arc<QueryNode<L, Variable>>,
        visited: &mut HashSet<u32>,
        component: &mut Self,
    ) {
        visited.insert(node.alias.id);
        component.add_node(node.clone());

        for relationship in &self.relationships {
            if relationship.from.alias.id == node.alias.id {
                if visited.insert(relationship.alias.id) {
                    component.add_relationship(relationship.clone());
                }
                if !visited.contains(&relationship.to.alias.id) {
                    self.dfs(&relationship.to, visited, component);
                }
            } else if relationship.to.alias.id == node.alias.id {
                if visited.insert(relationship.alias.id) {
                    component.add_relationship(relationship.clone());
                }
                if !visited.contains(&relationship.from.alias.id) {
                    self.dfs(&relationship.from, visited, component);
                }
            }
        }

        for path in &self.paths {
            if path.vars.iter().all(|id| visited.contains(&id.id)) && visited.insert(path.var.id) {
                component.add_path(path.clone());
            }
        }
    }
}

/// Type alias for expression trees.
pub type QueryExpr<TVar> = Arc<DynTree<ExprIR<TVar>>>;

/// An item in a SET clause - either property assignment or label modification.
#[derive(Clone, Debug)]
pub enum SetItem<L, TVar> {
    /// Property assignment: `n.prop = value` (replace=true) or `n += {props}` (replace=false)
    Attribute {
        target: QueryExpr<TVar>,
        value: QueryExpr<TVar>,
        replace: bool,
    },
    /// Label assignment: `SET n:Label`
    Label { var: TVar, labels: OrderSet<L> },
}

#[cfg_attr(tarpaulin, skip)]
impl<L: Display + PartialEq, TVar: Display + std::fmt::Debug> Display for SetItem<L, TVar> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Attribute {
                target,
                value,
                replace,
            } => {
                let op = if *replace { "=" } else { "+=" };
                write!(f, "{target} {op} {value}")
            }
            Self::Label { var, labels } => {
                write!(f, "{var}:")?;
                let mut first = true;
                for i in 0..labels.len() {
                    if !first {
                        write!(f, ":")?;
                    }
                    first = false;
                    write!(f, "{}", labels[i])?;
                }
                Ok(())
            }
        }
    }
}

/// Query clause IR - represents each clause type in a Cypher query.
///
/// A complete query is a sequence of these clauses. The planner converts
/// this AST into an execution plan.
#[derive(Debug)]
pub enum QueryIR<TVar> {
    /// CALL procedure(args) YIELD outputs WHERE filter
    /// The bool indicates whether YIELD was explicitly written (true) or default outputs are used (false).
    /// yield_aliases stores the original field names when AS aliasing is used;
    /// yields then holds the alias names (scope-visible).
    Call {
        func: Arc<GraphFn>,
        args: Vec<QueryExpr<TVar>>,
        yields: Vec<TVar>,
        yield_aliases: Vec<Option<TVar>>,
        filter: Option<QueryExpr<TVar>>,
        explicit_yield: bool,
    },
    /// MATCH pattern WHERE filter (optional flag for OPTIONAL MATCH)
    Match {
        pattern: QueryGraph<Arc<String>, Arc<String>, TVar>,
        filter: Option<QueryExpr<TVar>>,
        optional: bool,
    },
    /// UNWIND list AS var
    Unwind {
        expr: QueryExpr<TVar>,
        var: TVar,
    },
    /// MERGE pattern ON CREATE SET ... ON MATCH SET ...
    Merge {
        pattern: QueryGraph<Arc<String>, Arc<String>, TVar>,
        on_create: Vec<SetItem<Arc<String>, TVar>>,
        on_match: Vec<SetItem<Arc<String>, TVar>>,
    },
    /// CREATE pattern
    Create(QueryGraph<Arc<String>, Arc<String>, TVar>),
    /// DELETE exprs (detach flag for DETACH DELETE)
    Delete {
        exprs: Vec<QueryExpr<TVar>>,
        detach: bool,
    },
    /// SET items
    Set(Vec<SetItem<Arc<String>, TVar>>),
    /// REMOVE items (properties or labels)
    Remove(Vec<QueryExpr<TVar>>),
    /// LOAD CSV FROM path AS var
    LoadCsv {
        file_path: QueryExpr<TVar>,
        headers: bool,
        delimiter: QueryExpr<TVar>,
        var: TVar,
    },
    /// WITH clause for intermediate projections and aggregations
    With {
        distinct: bool,
        all: bool,
        exprs: Vec<(TVar, QueryExpr<TVar>)>,
        copy_from_parent: Vec<(Variable, Variable)>,
        orderby: Vec<(QueryExpr<TVar>, bool)>,
        skip: Option<QueryExpr<TVar>>,
        limit: Option<QueryExpr<TVar>>,
        filter: Option<QueryExpr<TVar>>,
        write: bool,
    },
    Return {
        distinct: bool,
        all: bool,
        exprs: Vec<(TVar, QueryExpr<TVar>)>,
        copy_from_parent: Vec<(Variable, Variable)>,
        orderby: Vec<(QueryExpr<TVar>, bool)>,
        skip: Option<QueryExpr<TVar>>,
        limit: Option<QueryExpr<TVar>>,
        write: bool,
    },
    CreateIndex {
        label: Arc<String>,
        attrs: Vec<Arc<String>>,
        index_type: IndexType,
        entity_type: EntityType,
        options: Option<QueryExpr<TVar>>,
    },
    DropIndex {
        label: Arc<String>,
        attrs: Vec<Arc<String>>,
        index_type: IndexType,
        entity_type: EntityType,
    },
    /// UNION of multiple sub-query branches.
    /// `all` is true for UNION ALL (keep duplicates), false for UNION (deduplicate).
    Union {
        branches: Vec<Self>,
        all: bool,
    },
    Query {
        clauses: Vec<Self>,
        write: bool,
    },
    /// FOREACH(var IN list_expr | body_clauses)
    ForEach {
        list: QueryExpr<TVar>,
        var: TVar,
        body: Vec<Self>,
    },
    /// CALL { subquery_body }
    /// is_returning = body ends with RETURN
    /// remap = return remapping: (inner_var, outer_var) pairs
    CallSubquery {
        body: Box<Self>,
        is_returning: bool,
        remap: Vec<(TVar, TVar)>,
    },
}

#[cfg_attr(tarpaulin, skip)]
impl<TVar: Display + std::fmt::Debug + Eq + Hash> Display for QueryIR<TVar> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Call { func, args, .. } => {
                writeln!(f, "{}():", func.name)?;
                for arg in args {
                    write!(f, "{arg}")?;
                }
                Ok(())
            }
            Self::Match { pattern, .. } => writeln!(f, "MATCH {pattern}"),
            Self::Unwind { expr, var } => {
                writeln!(f, "UNWIND {var}:")?;
                write!(f, "{expr}")
            }
            Self::Merge { pattern, .. } => writeln!(f, "MERGE {pattern}"),
            Self::Create(p) => write!(f, "CREATE {p}"),
            Self::Delete { exprs, .. } => {
                writeln!(f, "DELETE:")?;
                for expr in exprs {
                    write!(f, "{expr}")?;
                }
                Ok(())
            }
            Self::Set(items) => {
                writeln!(f, "SET:")?;
                for item in items {
                    write!(f, "{item}")?;
                }
                Ok(())
            }
            Self::Remove(items) => {
                writeln!(f, "REMOVE:")?;
                for item in items {
                    write!(f, "{item}")?;
                }
                Ok(())
            }
            Self::LoadCsv { file_path, var, .. } => {
                writeln!(f, "LOAD CSV FROM {file_path} AS {var:}:")
            }
            Self::With { exprs, .. } => {
                writeln!(f, "WITH:")?;
                for (name, _) in exprs {
                    write!(f, "{name}")?;
                }
                Ok(())
            }
            Self::Return { exprs, .. } => {
                writeln!(f, "RETURN:")?;
                for (name, _) in exprs {
                    write!(f, "{name}")?;
                }
                Ok(())
            }
            Self::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options: _options,
            } => {
                writeln!(
                    f,
                    "CREATE {index_type:?} {entity_type:?} INDEX ON :{label}({attrs:?})"
                )
            }
            Self::DropIndex {
                label,
                attrs,
                index_type,
                entity_type,
            } => {
                writeln!(
                    f,
                    "DROP {index_type:?} {entity_type:?} INDEX ON :{label}({attrs:?})"
                )
            }
            Self::Query { clauses, .. } => {
                for q in clauses {
                    write!(f, "{q}")?;
                }
                Ok(())
            }
            Self::Union { branches, all } => {
                let keyword = if *all { "UNION ALL" } else { "UNION" };
                for (i, branch) in branches.iter().enumerate() {
                    if i > 0 {
                        writeln!(f, "{keyword}")?;
                    }
                    write!(f, "{branch}")?;
                }
                Ok(())
            }
            Self::ForEach { list, var, body } => {
                write!(f, "FOREACH({var} IN {list} | ")?;
                for clause in body {
                    write!(f, "{clause}")?;
                }
                write!(f, ")")
            }
            Self::CallSubquery { body, .. } => write!(f, "CALL {{ {body} }}"),
        }
    }
}

impl<TVar: Eq + Hash + Display> QueryIR<TVar> {
    pub fn validate(&self) -> Result<(), String> {
        self.inner_validate(std::iter::empty())
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cognitive_complexity)]
    fn inner_validate<'a, T>(
        &self,
        mut iter: T,
    ) -> Result<(), String>
    where
        T: Iterator<Item = &'a Self>,
        TVar: 'a,
    {
        match self {
            Self::Call {
                func: proc,
                args,
                ..
            } => {
                if proc.name == "db.idx.fulltext.createNodeIndex" {
                    match args[0].root().data() {
                        ExprIR::Constant(Value::String(_)) => {}
                        ExprIR::Map => {
                            let mut has_labels = false;
                            for child in args[0].root().children() {
                                if let ExprIR::Constant(Value::String(label)) = child.data()
                                    && label.as_str() == "label"
                                {
                                    has_labels = true;
                                    break;
                                }
                            }
                            if !has_labels {
                                return Err(String::from("Label is missing"));
                            }
                        }
                        _ => {
                            return Err(String::from(
                                "The first argument of a procedure call must be a string or a map with a 'label' key",
                            ));
                        }
                    }
                }
                Ok(())
            }
            Self::Match { pattern, .. } => {
                Self::validate_inlined_properties(pattern)?;
                iter.next().map_or_else(|| Err(String::from(
                        "Query cannot conclude with MATCH (must be a RETURN clause, an update clause, a procedure call or a non-returning subquery)",
                    )), |first| first.inner_validate(iter))
            }
            Self::Unwind { .. } => {
                iter.next().map_or_else(|| Err(String::from(
                        "Query cannot conclude with UNWIND (must be a RETURN clause, an update clause, a procedure call or a non-returning subquery)",
                    )), |first| first.inner_validate(iter))
            }
            Self::Merge {
                pattern,
                on_create: on_create_set_items,
                on_match: on_match_set_items,
            } => {
                Self::validate_inlined_properties(pattern)?;
                for relationship in &pattern.relationships {
                    if relationship.types.len() != 1 {
                        return Err(String::from(
                            "Exactly one relationship type must be specified for each relation in a MERGE pattern.",
                        ));
                    }
                }
                Self::validate_set_items(on_create_set_items)?;
                Self::validate_set_items(on_match_set_items)?;
                iter.next()
                    .map_or(Ok(()), |first| first.inner_validate(iter))
            }
            Self::Create(p) => {
                Self::validate_inlined_properties(p)?;
                for relationship in &p.relationships {
                    if relationship.types.len() != 1 {
                        return Err(String::from(
                            "Exactly one relationship type must be specified for each relation in a CREATE pattern.",
                        ));
                    }
                }
                iter.next()
                    .map_or(Ok(()), |first| first.inner_validate(iter))
            }
            Self::Delete { .. } => {
                iter.next()
                    .map_or(Ok(()), |first| first.inner_validate(iter))
            }
            Self::Set(items) => {
                Self::validate_set_items(items)?;
                iter.next()
                    .map_or(Ok(()), |first| first.inner_validate(iter))
            }
            Self::Remove(items) => {
                for item in items {
                    if  matches!(item.root().data(), ExprIR::Property(_)) && matches!(item.root().child(0).data(), ExprIR::Constant(Value::Null)) {
                        return Err("Type mismatch: expected Node or Relationship but was Null".to_string());
                    }
                }
                iter.next()
                    .map_or(Ok(()), |first| first.inner_validate(iter))
            }
            Self::LoadCsv { .. } => {
                iter.next()
                    .map_or(Ok(()), |first| first.inner_validate(iter))
            }
            Self::With { exprs: _, orderby: _orderby, .. } | Self::Return { exprs: _, orderby: _orderby, .. } => {
                iter.next()
                    .map_or(Ok(()), |first| first.inner_validate(iter))
            }
            Self::CreateIndex { .. } => iter
                .next()
                .map_or(Ok(()), |first| first.inner_validate(iter)),
            Self::DropIndex { .. } => iter
                .next()
                .map_or(Ok(()), |first| first.inner_validate(iter)),
            Self::Query { clauses, .. } => {
                let mut iter = clauses.iter();
                let first = iter.next().ok_or("Error: empty query.")?;
                first.inner_validate(iter)
            }
            Self::Union { branches, .. } => {
                let mut first_columns: Option<Vec<String>> = None;
                for branch in branches {
                    branch.validate()?;
                    let columns = branch.return_column_names();
                    if let Some(ref expected) = first_columns {
                        if columns != *expected {
                            return Err(String::from(
                                "All sub queries in a UNION must have the same column names.",
                            ));
                        }
                    } else {
                        first_columns = Some(columns);
                    }
                }
                Ok(())
            }
            Self::ForEach { body, .. } => {
                for clause in body {
                    clause.validate()?;
                }
                iter.next()
                    .map_or(Ok(()), |first| first.inner_validate(iter))
            }
            Self::CallSubquery {
                body,
                is_returning,
                ..
            } => {
                body.validate()?;
                if *is_returning {
                    iter.next().map_or_else(
                        || Err("Query cannot conclude with a returning subquery (must be a RETURN clause, an update clause, a procedure call or a non-returning subquery)".into()),
                        |first| first.inner_validate(iter),
                    )
                } else {
                    iter.next()
                        .map_or(Ok(()), |first| first.inner_validate(iter))
                }
            }
        }
    }

    fn validate_set_items(items: &Vec<SetItem<Arc<String>, TVar>>) -> Result<(), String> {
        for item in items {
            if let SetItem::Attribute { target, .. } = item {
                if let ExprIR::Property(_) = target.root().data()
                    && let ExprIR::Variable(_) = target.root().child(0).data()
                {
                } else if let ExprIR::Variable(_) = target.root().data() {
                } else {
                    return Err(String::from(
                        "FalkorDB does not currently support non-alias references on the left-hand side of SET expressions",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Validates that inlined properties in node/relationship patterns are maps.
    /// Mirrors `_ValidateInlinedProperties` in the C `ast_validations.c`.
    fn validate_inlined_properties(
        p: &QueryGraph<Arc<String>, Arc<String>, TVar>
    ) -> Result<(), String> {
        for node in &p.nodes {
            if !matches!(node.attrs.root().data(), ExprIR::Map) {
                return Err(String::from(
                    "Encountered unhandled type in inlined properties.",
                ));
            }
        }
        for rel in &p.relationships {
            if !matches!(rel.attrs.root().data(), ExprIR::Map) {
                return Err(String::from(
                    "Encountered unhandled type in inlined properties.",
                ));
            }
        }
        Ok(())
    }

    /// Extracts RETURN (or CALL/YIELD) column names from a UNION branch
    /// for cross-branch validation.
    pub fn return_column_names(&self) -> Vec<String> {
        if let Self::Query { clauses, .. } = self {
            for clause in clauses.iter().rev() {
                match clause {
                    Self::Return { exprs, .. } => {
                        return exprs.iter().map(|(var, _)| var.to_string()).collect();
                    }
                    Self::Call { yields, .. } => {
                        return yields.iter().map(ToString::to_string).collect();
                    }
                    Self::CallSubquery {
                        body,
                        is_returning: true,
                        ..
                    } => {
                        return body.return_column_names();
                    }
                    _ => {}
                }
            }
        }
        Vec::new()
    }
}

/// Type alias for unbound query IR (variables are just string names).
pub type RawQueryIR = QueryIR<Arc<String>>;

/// Type alias for bound query IR (variables have resolved IDs and types).
pub type BoundQueryIR = QueryIR<Variable>;
