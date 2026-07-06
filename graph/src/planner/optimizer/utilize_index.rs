//! Index utilization optimizer pass.
//!
//! Scans the execution plan for `NodeByLabelScan` operators that sit below a
//! `Filter` on an indexed property, and replaces the pair with a single
//! `NodeByIndexScan` that pushes the predicate into the index engine.
//!
//! ## Supported Patterns
//!
//! **Single comparison filter:**
//!
//! ```text
//! Before:                       After:
//!
//! Filter(n.age = 30)            NodeByIndexScan(:Person, age, Equal(30))
//!   |
//!   v
//! NodeByLabelScan(:Person)
//! ```
//!
//! **AND filter with multiple indexed conjuncts:**
//!
//! When a Filter contains `AND(n.year >= 1980, n.year < 1990)`, the pass
//! merges both conjuncts into a single `Range` index query:
//!
//! ```text
//! Before:                         After:
//!
//! Filter(AND(year>=1980,          NodeByIndexScan(:Movie, year,
//!            year<1990))            Range{min:1980, max:1990})
//!   |
//!   v
//! NodeByLabelScan(:Movie)
//! ```
//!
//! If only some AND conjuncts are indexable, the indexable ones are merged
//! into the scan and the remaining conjuncts stay as a reduced Filter.
//!
//! **Inline node attributes:**
//!
//! Also converts `NodeByLabelScan` nodes that carry inline property attributes
//! (e.g. `(n:Person {name: 'Alice'})`) into `NodeByIndexScan` when the
//! attribute is indexed.
//!
//! ## Supported operators and index types
//!
//! - Equality (`=`), less-than (`<`, `<=`), greater-than (`>`, `>=`)
//! - `distance()` function for point indexes
//! - Range indexes only (fulltext indexes are handled separately)

use std::sync::Arc;

use orx_tree::{Bfs, Dyn, DynTree, NodeIdx, NodeRef};

use crate::{
    graph::graph::Graph,
    index::{
        Index,
        indexer::{IndexQuery, IndexType},
    },
    parser::ast::{ExprIR, QueryExpr, QueryNode, Variable},
    runtime::functions::{FnType, get_functions},
    tree,
};

use super::super::IR;

use crate::parser::ast::QueryRelationship;
use crate::runtime::orderset::OrderSet;
use crate::runtime::value::Value;

/// Build a `hasLabels(variable, [label1, label2, ...])` filter expression.
fn build_has_labels_filter(
    var: &Variable,
    labels: impl Iterator<Item = Arc<String>>,
) -> QueryExpr<Variable> {
    let has_labels_fn = get_functions()
        .get("hasLabels", &FnType::Function)
        .expect("hasLabels function must exist");
    Arc::new(tree!(
        ExprIR::FuncInvocation(has_labels_fn),
        tree!(ExprIR::Variable(var.clone())),
        tree!(ExprIR::List; labels.map(|l| tree!(ExprIR::Constant(Value::String(l)))))
    ))
}

/// Result of a single-predicate index-scan attempt: the pattern
/// subject (node or relationship), the label/type the index lives on,
/// and the query pushed into the index engine.
type Scan<T> = Option<(T, Arc<String>, IndexQuery<QueryExpr<Variable>>)>;

/// Shared interface for the two pattern kinds that can back an index
/// scan (node labels and relationship types). Every kind-specific
/// decision in the utilization passes — which IR variant to match,
/// which IR variant to emit, which indexer to query, and whether
/// function-based predicates like `distance()` are supported — lives
/// here so the passes themselves are uniform.
trait IndexSubject: Clone {
    /// Kind-specific metadata threaded through a rewrite: `()` for
    /// nodes, `bool` (the `transposed` flag) for edges.
    type Metadata: Copy;

    fn alias(&self) -> &Variable;

    /// All labels/types on the pattern — iterated to find an indexed
    /// label/type match.
    fn all_labels(&self) -> Box<dyn Iterator<Item = &Arc<String>> + '_>;

    /// Inline property attributes (e.g. `{age: 30}` on the pattern).
    fn inline_attrs(&self) -> &DynTree<ExprIR<Variable>>;

    /// Look up the appropriate indexer (node vs edge) on the graph.
    fn is_indexed(
        graph: &Graph,
        label: &Arc<String>,
        attr: &Arc<String>,
        ty: &IndexType,
    ) -> bool;

    /// Subject-specific hook for function-based predicates. Node impl
    /// delegates to `try_distance_index_scan`; edge impl returns `None`
    /// (no edge-distance path is exercised today).
    fn try_func_scan(
        subject: &Self,
        attr: &Arc<String>,
        filter: &DynTree<ExprIR<Variable>>,
        attr_side: NodeIdx<Dyn<ExprIR<Variable>>>,
        constant_node: DynTree<ExprIR<Variable>>,
        label: Arc<String>,
    ) -> Scan<Self>;

    /// Recognize the IR variant that's a candidate for an index scan
    /// (`NodeByLabelScan` for nodes, `CondTraverse` for edges). Returns
    /// `None` if the variant doesn't match or if `labels` / `types` is
    /// empty (no scan target to push the filter into).
    fn match_scan_source(ir: &IR) -> Option<(Self, Self::Metadata)>;

    /// Build the replacement IR after a successful pushdown
    /// (`NodeByIndexScan` or `EdgeByIndexScan`).
    fn build_scan_ir(
        self,
        index: Arc<String>,
        query: Arc<IndexQuery<QueryExpr<Variable>>>,
        metadata: Self::Metadata,
    ) -> IR;

    /// Returns a copy of the subject whose labels/types list places
    /// `label` first. The runtime index-scan ops treat `labels[0]` /
    /// `types[0]` as the index key and post-filter the rest.
    fn with_primary_label(
        &self,
        label: &Arc<String>,
    ) -> Self;
}

impl IndexSubject for Arc<QueryNode<Arc<String>, Variable>> {
    type Metadata = ();

    fn alias(&self) -> &Variable {
        &self.alias
    }
    fn all_labels(&self) -> Box<dyn Iterator<Item = &Arc<String>> + '_> {
        Box::new(self.labels.iter())
    }
    fn inline_attrs(&self) -> &DynTree<ExprIR<Variable>> {
        &self.attrs
    }
    fn is_indexed(
        graph: &Graph,
        label: &Arc<String>,
        attr: &Arc<String>,
        ty: &IndexType,
    ) -> bool {
        graph.is_indexed(label, attr, ty)
    }
    fn try_func_scan(
        subject: &Self,
        attr: &Arc<String>,
        filter: &DynTree<ExprIR<Variable>>,
        attr_side: NodeIdx<Dyn<ExprIR<Variable>>>,
        constant_node: DynTree<ExprIR<Variable>>,
        label: Arc<String>,
    ) -> Scan<Self> {
        try_distance_index_scan(subject, attr, filter, attr_side, constant_node, label)
    }
    fn match_scan_source(ir: &IR) -> Option<(Self, Self::Metadata)> {
        let IR::NodeByLabelScan { node } = ir else {
            return None;
        };
        if node.labels.is_empty() {
            return None;
        }
        Some((node.clone(), ()))
    }
    fn build_scan_ir(
        self,
        index: Arc<String>,
        query: Arc<IndexQuery<QueryExpr<Variable>>>,
        _metadata: (),
    ) -> IR {
        IR::NodeByIndexScan {
            node: self,
            index,
            query,
        }
    }
    fn with_primary_label(
        &self,
        label: &Arc<String>,
    ) -> Self {
        let mut reordered = OrderSet::default();
        reordered.insert(label.clone());
        for l in self.labels.iter() {
            if l != label {
                reordered.insert(l.clone());
            }
        }
        Self::new(QueryNode::new(
            self.alias.clone(),
            reordered,
            self.attrs.clone(),
        ))
    }
}

impl IndexSubject for Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>> {
    type Metadata = bool; // transposed

    fn alias(&self) -> &Variable {
        &self.alias
    }
    fn all_labels(&self) -> Box<dyn Iterator<Item = &Arc<String>> + '_> {
        Box::new(self.types.iter())
    }
    fn inline_attrs(&self) -> &DynTree<ExprIR<Variable>> {
        &self.attrs
    }
    fn is_indexed(
        graph: &Graph,
        label: &Arc<String>,
        attr: &Arc<String>,
        ty: &IndexType,
    ) -> bool {
        graph.is_edge_indexed(label, attr, ty)
    }
    fn try_func_scan(
        _subject: &Self,
        _attr: &Arc<String>,
        _filter: &DynTree<ExprIR<Variable>>,
        _attr_side: NodeIdx<Dyn<ExprIR<Variable>>>,
        _constant_node: DynTree<ExprIR<Variable>>,
        _label: Arc<String>,
    ) -> Scan<Self> {
        None
    }
    fn match_scan_source(ir: &IR) -> Option<(Self, Self::Metadata)> {
        let IR::CondTraverse {
            relationship,
            transposed,
            sibling_edges,
            ..
        } = ir
        else {
            return None;
        };
        // EdgeByIndexScan can only faithfully replace CondTraverse when
        // the pattern has exactly one relationship type — matches
        // FalkorDB C's `utilize_indices.c:reduce_cond_op` gate
        // (`QGEdge_RelationCount(e) != 1`). A multi-type `[:A|B]`
        // pattern would require a UNION over two separate indexes
        // which this operator doesn't implement.
        if relationship.types.len() != 1 {
            return None;
        }
        // `sibling_edges` always includes the edge's own alias for
        // named edges; a real uniqueness constraint means at least one
        // sibling alias that *isn't* this edge's own alias. When that
        // constraint is present, CondTraverseOp enforces it — our op
        // doesn't, so bail.
        if sibling_edges.iter().any(|&id| id != relationship.alias.id) {
            return None;
        }
        Some((relationship.clone(), *transposed))
    }
    fn build_scan_ir(
        self,
        _index: Arc<String>,
        query: Arc<IndexQuery<QueryExpr<Variable>>>,
        transposed: bool,
    ) -> IR {
        IR::EdgeByIndexScan {
            relationship: self,
            query,
            transposed,
        }
    }
    fn with_primary_label(
        &self,
        label: &Arc<String>,
    ) -> Self {
        let mut reordered = Vec::with_capacity(self.types.len());
        reordered.push(label.clone());
        for t in &self.types {
            if t != label {
                reordered.push(t.clone());
            }
        }
        let mut new_rel = QueryRelationship::new(
            self.alias.clone(),
            reordered,
            self.attrs.clone(),
            self.from.clone(),
            self.to.clone(),
            self.bidirectional,
            self.min_hops,
            self.max_hops,
        );
        new_rel.all_shortest_paths = self.all_shortest_paths;
        Self::new(new_rel)
    }
}

/// Build an `IndexQuery` for a single comparison operator over a
/// (attr, constant) pair. Used by both node and edge index-scan paths
/// once the subject has resolved its label/type.
fn build_op_query(
    attr: &Arc<String>,
    op: &ExprIR<Variable>,
    constant_node: DynTree<ExprIR<Variable>>,
) -> Option<IndexQuery<QueryExpr<Variable>>> {
    match op {
        ExprIR::Eq => Some(IndexQuery::Equal {
            key: attr.clone(),
            value: Arc::new(constant_node),
        }),
        ExprIR::Gt => Some(IndexQuery::Range {
            key: attr.clone(),
            min: Some(Arc::new(constant_node)),
            max: None,
            include_min: false,
            include_max: false,
        }),
        ExprIR::Ge => Some(IndexQuery::Range {
            key: attr.clone(),
            min: Some(Arc::new(constant_node)),
            max: None,
            include_min: true,
            include_max: false,
        }),
        ExprIR::Lt => Some(IndexQuery::Range {
            key: attr.clone(),
            min: None,
            max: Some(Arc::new(constant_node)),
            include_min: false,
            include_max: false,
        }),
        ExprIR::Le => Some(IndexQuery::Range {
            key: attr.clone(),
            min: None,
            max: Some(Arc::new(constant_node)),
            include_min: false,
            include_max: true,
        }),
        _ => None,
    }
}

fn extract_attribute_from_subtree(
    tree: &DynTree<ExprIR<Variable>>,
    root_idx: NodeIdx<Dyn<ExprIR<Variable>>>,
) -> Option<Arc<String>> {
    for idx in tree.node(root_idx).indices::<Bfs>() {
        let node = tree.node(idx);
        if let ExprIR::Property(attr) = node.data() {
            return Some(attr.clone());
        }
    }
    None
}

/// Extract the attribute of the target node and the expression from the filter.
///
/// Looks for a Property access on the target node (identified by alias) in either
/// child. The other child is treated as the constant/expression side, even if it
/// contains properties of other (already-bound) variables. This enables runtime
/// index utilization for queries like `WHERE b.age > a.age` when `a` is already bound.
fn extract_attribute_and_expression_from_filter(
    filter: &DynTree<ExprIR<Variable>>,
    target_alias: Option<&Variable>,
) -> Option<(
    Arc<String>,
    NodeIdx<Dyn<ExprIR<Variable>>>,
    NodeIdx<Dyn<ExprIR<Variable>>>,
)> {
    let lhs_idx = filter.root().child(0).idx();
    let rhs_idx = filter.root().child(1).idx();

    if let Some(alias) = target_alias {
        // Check which side has a property of the target node.
        // If both sides reference the target (e.g. `n.age > n.salary`),
        // we can't use the index — it can only accelerate one property lookup.
        let lhs_has_target_property = subtree_has_property_of(filter, lhs_idx, alias);
        let rhs_has_target_property = subtree_has_property_of(filter, rhs_idx, alias);

        match (lhs_has_target_property, rhs_has_target_property) {
            (true, true) => None,
            (true, _) => {
                let attr = extract_attribute_from_subtree(filter, lhs_idx)?;
                Some((attr, lhs_idx, rhs_idx))
            }
            (_, true) => {
                let attr = extract_attribute_from_subtree(filter, rhs_idx)?;
                Some((attr, rhs_idx, lhs_idx))
            }
            _ => None,
        }
    } else {
        // Fallback: no target alias, use original logic
        match (
            extract_attribute_from_subtree(filter, lhs_idx),
            extract_attribute_from_subtree(filter, rhs_idx),
        ) {
            (Some(_), Some(_)) => None,
            (Some(attr), _) => Some((attr, lhs_idx, rhs_idx)),
            (_, Some(attr)) => Some((attr, rhs_idx, lhs_idx)),
            _ => None,
        }
    }
}

/// Builds an index scan for a distance filter (e.g. `distance(n.loc, point(...)) < 100`).
fn try_distance_index_scan(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    attr: &Arc<String>,
    filter: &DynTree<ExprIR<Variable>>,
    attribute_side: NodeIdx<Dyn<ExprIR<Variable>>>,
    constant_node: DynTree<ExprIR<Variable>>,
    label: Arc<String>,
) -> Scan<Arc<QueryNode<Arc<String>, Variable>>> {
    let operand = filter.root().data();
    // distance() must be on the "less than" side of the comparison:
    // distance(...) < x  or  distance(...) <= x  (attribute_side == child(0))
    // x > distance(...)  or  x >= distance(...)  (attribute_side == child(1))
    match operand {
        ExprIR::Lt | ExprIR::Le => {
            if filter.root().child(0).idx() != attribute_side {
                return None;
            }
        }
        ExprIR::Gt | ExprIR::Ge => {
            if filter.root().child(1).idx() != attribute_side {
                return None;
            }
        }
        _ => return None,
    }
    let child_0_idx = filter.node(attribute_side).child(0).idx();
    let child_1_idx = filter.node(attribute_side).child(1).idx();
    match (
        extract_attribute_from_subtree(filter, child_0_idx),
        extract_attribute_from_subtree(filter, child_1_idx),
    ) {
        (Some(_), None) => Some((
            node.clone(),
            label,
            IndexQuery::Point {
                key: attr.clone(),
                point: Arc::new(filter.node(child_1_idx).clone_as_tree()),
                radius: Arc::new(constant_node),
            },
        )),
        (None, Some(_)) => Some((
            node.clone(),
            label,
            IndexQuery::Point {
                key: attr.clone(),
                point: Arc::new(filter.node(child_0_idx).clone_as_tree()),
                radius: Arc::new(constant_node),
            },
        )),
        _ => None,
    }
}

/// Merges two index queries on the same attribute into a single Range query.
///
/// For example, `year >= 1980` and `year < 1990` become `Range { min: 1980, max: 1990, include_min: true, include_max: false }`.
///
/// When both queries specify the same bound (both min or both max), we cannot
/// determine at plan time which is stricter (the values are expression trees),
/// so we fall back to `And` and let the index engine intersect them.
fn merge_range_queries(
    a: IndexQuery<QueryExpr<Variable>>,
    b: IndexQuery<QueryExpr<Variable>>,
) -> IndexQuery<QueryExpr<Variable>> {
    match (a, b) {
        (
            IndexQuery::Range {
                key,
                min: min_a,
                max: max_a,
                include_min: inc_min_a,
                include_max: inc_max_a,
            },
            IndexQuery::Range {
                key: key_b,
                min: min_b,
                max: max_b,
                include_min: inc_min_b,
                include_max: inc_max_b,
            },
        ) if key == key_b => {
            // If both specify the same bound, we can't compare expression
            // values at plan time to pick the stricter one — fall back to And.
            if min_a.is_some() && min_b.is_some() || max_a.is_some() && max_b.is_some() {
                return IndexQuery::And(vec![
                    IndexQuery::Range {
                        key: key.clone(),
                        min: min_a,
                        max: max_a,
                        include_min: inc_min_a,
                        include_max: inc_max_a,
                    },
                    IndexQuery::Range {
                        key,
                        min: min_b,
                        max: max_b,
                        include_min: inc_min_b,
                        include_max: inc_max_b,
                    },
                ]);
            }
            // Complementary bounds: one provides min, the other max.
            // The unused include flag is always false, so or/select works.
            let (min, include_min) = if min_a.is_some() {
                (min_a, inc_min_a)
            } else {
                (min_b, inc_min_b)
            };
            let (max, include_max) = if max_a.is_some() {
                (max_a, inc_max_a)
            } else {
                (max_b, inc_max_b)
            };
            IndexQuery::Range {
                key,
                min,
                max,
                include_min,
                include_max,
            }
        }
        (a, b) => IndexQuery::And(vec![a, b]),
    }
}

/// Checks if a subtree contains any Property node referencing the given node alias.
fn subtree_has_property_of(
    tree: &DynTree<ExprIR<Variable>>,
    root_idx: NodeIdx<Dyn<ExprIR<Variable>>>,
    alias: &Variable,
) -> bool {
    for idx in tree.node(root_idx).indices::<Bfs>() {
        let node = tree.node(idx);
        if let ExprIR::Property(_) = node.data() {
            // Check if the Variable child of this Property matches the alias
            for child in node.children() {
                if let ExprIR::Variable(v) = child.data()
                    && v == alias
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Checks if a subtree has any graph entity dependency (any Property access at all).
fn subtree_has_any_property(
    tree: &DynTree<ExprIR<Variable>>,
    root_idx: NodeIdx<Dyn<ExprIR<Variable>>>,
) -> bool {
    for idx in tree.node(root_idx).indices::<Bfs>() {
        if let ExprIR::Property(_) = tree.node(idx).data() {
            return true;
        }
    }
    false
}

/// Checks if a list expression subtree contains any nested List (non-indexable).
fn list_has_nested_list(
    tree: &DynTree<ExprIR<Variable>>,
    root_idx: NodeIdx<Dyn<ExprIR<Variable>>>,
) -> bool {
    if !matches!(tree.node(root_idx).data(), ExprIR::List) {
        return false;
    }
    for child in tree.node(root_idx).children() {
        if matches!(child.data(), ExprIR::List) {
            return true;
        }
    }
    false
}

/// Tries to convert an `IN` filter into an index scan against `subject`.
///
/// Handles two patterns:
/// 1. `property IN [list]` — converts to InList index query
/// 2. `value IN property` — converts to ArrayContains index query
fn try_in_filter_scan<T: IndexSubject>(
    subject: &T,
    filter: &DynTree<ExprIR<Variable>>,
    graph: &Graph,
) -> Scan<T> {
    if !matches!(filter.root().data(), ExprIR::In) {
        return None;
    }

    let lhs_idx = filter.root().child(0).idx();
    let rhs_idx = filter.root().child(1).idx();

    let lhs_has_target = subtree_has_property_of(filter, lhs_idx, subject.alias());
    let rhs_has_target = subtree_has_property_of(filter, rhs_idx, subject.alias());

    // Exactly one side must reference the target subject for index utilization:
    //   property IN [list]   →  attr on left  (is_property_in_list = true)
    //   value IN property    →  attr on right (is_property_in_list = false)
    let (attr_side, expr_side, is_property_in_list) = match (lhs_has_target, rhs_has_target) {
        (true, false) => (lhs_idx, rhs_idx, true),
        (false, true) => (rhs_idx, lhs_idx, false),
        _ => return None,
    };

    let attr = extract_attribute_from_subtree(filter, attr_side)?;
    let label = subject
        .all_labels()
        .find(|l| T::is_indexed(graph, l, &attr, &IndexType::Range))?
        .clone();

    let query = if is_property_in_list {
        // Pattern: p.age IN [1, 2, 3]
        // Don't index if the list contains nested arrays or if the
        // expression has graph-entity dependencies.
        if list_has_nested_list(filter, expr_side) {
            return None;
        }
        if subtree_has_any_property(filter, expr_side) {
            return None;
        }
        IndexQuery::InList {
            key: attr,
            list: Arc::new(filter.node(expr_side).clone_as_tree()),
        }
    } else {
        // Pattern: $x IN p.samples
        IndexQuery::ArrayContains {
            key: attr,
            value: Arc::new(filter.node(expr_side).clone_as_tree()),
        }
    };

    Some((subject.clone(), label, query))
}

/// Tries to convert a single comparison filter into an index scan against `subject`.
fn try_single_filter_scan<T: IndexSubject>(
    subject: &T,
    filter: &DynTree<ExprIR<Variable>>,
    graph: &Graph,
) -> Scan<T> {
    if matches!(filter.root().data(), ExprIR::In) {
        return try_in_filter_scan(subject, filter, graph);
    }
    if !matches!(
        filter.root().data(),
        ExprIR::Eq | ExprIR::Gt | ExprIR::Ge | ExprIR::Lt | ExprIR::Le
    ) {
        return None;
    }
    let (attr, attr_side, constant_side) =
        extract_attribute_and_expression_from_filter(filter, Some(subject.alias()))?;
    let label = subject
        .all_labels()
        .find(|l| T::is_indexed(graph, l, &attr, &IndexType::Range))?
        .clone();
    match filter.node(attr_side).data() {
        ExprIR::FuncInvocation(func) if func.name.as_str() == "distance" => {
            let constant_node = filter.node(constant_side).clone_as_tree();
            T::try_func_scan(subject, &attr, filter, attr_side, constant_node, label)
        }
        ExprIR::Property(attr_ref) => {
            let constant_node = filter.node(constant_side).clone_as_tree();
            // If the property is on the right side (e.g., `1980 <= m.year`),
            // flip the operator so the index scan uses the correct direction.
            let op = if attr_side == filter.root().child(0).idx() {
                filter.root().data().clone()
            } else {
                match filter.root().data() {
                    ExprIR::Eq => ExprIR::Eq,
                    ExprIR::Gt => ExprIR::Lt,
                    ExprIR::Ge => ExprIR::Le,
                    ExprIR::Lt => ExprIR::Gt,
                    ExprIR::Le => ExprIR::Ge,
                    _ => unreachable!(),
                }
            };
            let query = build_op_query(attr_ref, &op, constant_node)?;
            Some((subject.clone(), label, query))
        }
        _ => None,
    }
}

/// Checks whether an inline property attribute on a pattern
/// (e.g. `(n:Person {name: 'Alice'})` or `[r:KNOWS {since: 2020}]`) is
/// covered by a range index and, if so, returns the subject, the label
/// or type that carries the index, the indexed attribute, and an
/// equivalent `attr = value` filter tree for the index scan.
fn get_inline_attr_index<T: IndexSubject>(
    graph: &Graph,
    subject: &T,
) -> Option<(T, Arc<String>, Arc<String>, DynTree<ExprIR<Variable>>)> {
    for label in subject.all_labels() {
        for attr in subject.inline_attrs().root().children() {
            if let ExprIR::Constant(Value::String(attr_str)) = attr.data()
                && T::is_indexed(graph, label, attr_str, &IndexType::Range)
            {
                return Some((
                    subject.clone(),
                    label.clone(),
                    attr_str.clone(),
                    tree!(
                        ExprIR::Eq,
                        tree!(
                            ExprIR::Property(attr_str.clone()),
                            tree!(ExprIR::Variable(subject.alias().clone()))
                        ),
                        attr.child(0).as_cloned_subtree()
                    ),
                ));
            }
        }
    }
    None
}

/// Result of pushing a whole `Filter` predicate into a single index
/// scan: the label/type the index lives on, the merged index query and
/// any conjuncts that couldn't be indexed and must stay as a reduced
/// post-filter.
type FilterPushdown = (
    Arc<String>,
    IndexQuery<QueryExpr<Variable>>,
    Vec<DynTree<ExprIR<Variable>>>,
);

/// Tries to push a `Filter` predicate into a single index query on
/// `subject`, handling the three top-level shapes uniformly:
///
/// - `AND(...)` — merge every indexable conjunct into one `Range` query
///   and leave the rest as post-filter conjuncts.
/// - `OR(...)` — convert every branch to an index query or bail
///   (partial conversion would produce wrong results).
/// - Any other comparison / `IN` — delegate to `try_single_filter_scan`
///   and keep the original filter when it's a "value IN property"
///   array-contains (index may return false positives for non-indexable
///   list elements).
fn try_filter_pushdown<T: IndexSubject>(
    subject: &T,
    filter: &DynTree<ExprIR<Variable>>,
    graph: &Graph,
) -> Option<FilterPushdown> {
    match filter.root().data() {
        ExprIR::And => {
            let mut merged: Option<(Arc<String>, IndexQuery<QueryExpr<Variable>>)> = None;
            let mut remaining = Vec::new();
            for child in filter.root().children() {
                let conjunct = child.clone_as_tree();
                if let Some((_, label, query)) = try_single_filter_scan(subject, &conjunct, graph) {
                    merged = Some(match merged {
                        None => (label, query),
                        Some((prev_label, prev_q)) => {
                            (prev_label, merge_range_queries(prev_q, query))
                        }
                    });
                } else {
                    remaining.push(conjunct);
                }
            }
            merged.map(|(label, q)| (label, q, remaining))
        }
        ExprIR::Or => {
            let mut or_queries = Vec::new();
            let mut or_label: Option<Arc<String>> = None;
            for child in filter.root().children() {
                let branch = child.clone_as_tree();
                if let Some((_, label, q)) = try_single_filter_scan(subject, &branch, graph) {
                    if or_label.is_none() {
                        or_label = Some(label);
                    }
                    or_queries.push(q);
                } else {
                    return None;
                }
            }
            or_label.and_then(|label| {
                (!or_queries.is_empty()).then(|| (label, IndexQuery::Or(or_queries), Vec::new()))
            })
        }
        _ => try_single_filter_scan(subject, filter, graph).map(|(_, label, q)| {
            // For "value IN property" (array-contains), keep the filter
            // as a post-filter — the index may return false positives
            // for non-indexable array elements.
            let is_array_contains = matches!(filter.root().data(), ExprIR::In)
                && !subtree_has_property_of(filter, filter.root().child(0).idx(), subject.alias())
                && subtree_has_property_of(filter, filter.root().child(1).idx(), subject.alias());
            if is_array_contains {
                (label, q, vec![filter.root().clone_as_tree()])
            } else {
                (label, q, Vec::new())
            }
        }),
    }
}

/// Whether a filter contains runtime values that might evaluate to
/// non-indexable types (variables from other scans, parameters,
/// int-precision-losing literals, or non-scalar literals like lists,
/// maps, and function calls whose return type can't be statically
/// proven indexable), requiring a post-filter safety net even after
/// the filter has been pushed into the index.
///
/// If the runtime's `can_utilize_index` rejects the evaluated query
/// (e.g. a list or a `date(...)` value), the scan op falls back to
/// iterating all entities of the label/type; the retained Filter
/// above it re-establishes correctness. Mirrors the C
/// `unresolved_filters` path in `op_edge_by_index_scan.c`.
///
/// Structural walker conservatively flags any compound / function
/// subexpression on either side of the filter, *except* when it's the
/// RHS (list expression) of an `IN` operator — `InList` index queries
/// handle the list natively, so that subtree is whitelisted.
fn needs_post_filter(
    filter: &QueryExpr<Variable>,
    scan_alias_id: u32,
) -> bool {
    // Whitelist the RHS subtree of `property IN [literal, ...]` only
    // when the list is non-empty *and* every direct element is a
    // bare scalar literal (Null / Bool / Integer / Float / String).
    // Those are the exact cases the `InList` runtime path can push
    // into the index without fallback.
    //
    // Everything else — empty lists, lists with `$param`,
    // `date(…)`, sub-expressions, nested lists — must keep the
    // filter as a post-filter safety net: the runtime can filter
    // out non-indexable elements and hand the index an `Or([])`
    // which some backends treat as match-all.
    let rhs = filter.root().child(1);
    let rhs_is_scalar_literal_list = matches!(filter.root().data(), ExprIR::In)
        && matches!(filter.root().child(0).data(), ExprIR::Property(_))
        && matches!(rhs.data(), ExprIR::List)
        && rhs.num_children() > 0
        && rhs.children().all(|child| match child.data() {
            ExprIR::Constant(Value::Bool(_) | Value::Float(_) | Value::String(_)) => true,
            // Large int64s can't round-trip through f64 exactly,
            // so the runtime will reject them and fall back to a
            // full scan — the Filter has to stay above to
            // re-establish correctness in that case.
            ExprIR::Constant(Value::Int(v)) => !Index::int_loses_f64_precision(*v),
            // `Null` is intentionally *not* whitelisted: the runtime
            // drops `Null` from the IN list when building the index
            // query, which can collapse to an empty `Or([])` that the
            // index backend treats as match-all. Keeping the filter
            // keeps Cypher semantics (`v IN [NULL]` → unknown → row
            // filtered out).
            _ => false,
        });
    let skip_descendants: std::collections::HashSet<_> = if rhs_is_scalar_literal_list {
        filter.node(rhs.idx()).indices::<Bfs>().collect()
    } else {
        std::collections::HashSet::new()
    };
    filter.root().indices::<Bfs>().any(|i| {
        if skip_descendants.contains(&i) {
            return false;
        }
        is_non_indexable_subexpr(filter.node(i).data(), Some(scan_alias_id))
    })
}

/// Same as `needs_post_filter`, but applied to the value subtree of an
/// inline-attribute equality filter (the RHS of `attr = value`). For a
/// bare constant literal the index handles it exactly and no filter is
/// needed.
fn needs_inline_post_filter(filter: &DynTree<ExprIR<Variable>>) -> bool {
    let value_idx = filter.root().child(1).idx();
    filter
        .node(value_idx)
        .indices::<Bfs>()
        .any(|i| is_non_indexable_subexpr(filter.node(i).data(), None))
}

/// Returns true when the given `ExprIR` describes a value that the
/// index may not be able to resolve. `scan_alias_id`, when `Some`,
/// tolerates `Variable` references to the scan target itself (the
/// property-access side), so only *other* variables count as runtime
/// dependencies.
fn is_non_indexable_subexpr(
    expr: &ExprIR<Variable>,
    scan_alias_id: Option<u32>,
) -> bool {
    #[allow(clippy::match_same_arms)]
    match expr {
        ExprIR::Variable(v) => scan_alias_id.is_none_or(|id| v.id != id),
        ExprIR::Parameter(_) => true,
        ExprIR::Constant(Value::Int(v)) => Index::int_loses_f64_precision(*v),
        // Compound / non-primitive literals: the index backing store
        // only handles numeric, string, bool, and point scalars.
        ExprIR::List | ExprIR::Map => true,
        // Function calls can return any type, including non-indexable
        // temporal values (`date()`, `datetime()`, `duration()`…).
        // Conservative: keep the filter as a safety net.
        ExprIR::FuncInvocation(_) => true,
        _ => false,
    }
}

/// Drives a local rewrite to a fixed point: walks the plan in BFS
/// order and applies `try_rewrite` at each index; when a rewrite
/// succeeds (returns `true`), restart the walk against the new plan
/// because the NodeIdx list may be invalidated by structural changes
/// (orx-tree's Auto memory policy). Stop once a full pass produces no
/// change.
fn rewrite_until_stable<F>(
    plan: &mut DynTree<IR>,
    mut try_rewrite: F,
) where
    F: FnMut(&mut DynTree<IR>, NodeIdx<Dyn<IR>>) -> bool,
{
    loop {
        let mut changed = false;
        let indices = plan.root().indices::<Bfs>().collect::<Vec<_>>();
        for idx in indices {
            if try_rewrite(plan, idx) {
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
}

/// Match `Filter(expr) → scan-source` at `idx`. Returns the subject,
/// the filter expression, and the subject's metadata. Returns `None`
/// when the scan source doesn't match, when its labels/types are empty,
/// or when there's no `Filter` parent (including when the scan is at
/// the plan root — parent is absent rather than panicking).
fn match_scan_with_filter<T: IndexSubject>(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
) -> Option<(T, QueryExpr<Variable>, T::Metadata)> {
    let (subject, metadata) = T::match_scan_source(plan.node(idx).data())?;
    let IR::Filter(filter) = plan.node(idx).parent()?.data() else {
        return None;
    };
    Some((subject, filter.clone(), metadata))
}

/// Reorders the subject's labels so the indexed label is first.
/// `NodeByIndexScanOp` and `EdgeByIndexScanOp` both use the first
/// label/type as the index's primary and post-filter the rest.
fn reorder_subject_labels<T: IndexSubject>(
    subject: T,
    index_label: &Arc<String>,
) -> T {
    if subject
        .all_labels()
        .next()
        .is_some_and(|l| l == index_label)
    {
        return subject;
    }
    subject.with_primary_label(index_label)
}

/// Apply a successful filter pushdown in place: replace the scan with
/// an index scan, then either drop the parent `Filter` entirely, keep
/// it as a runtime safety net, or narrow it to the conjuncts that
/// weren't pushed down.
#[allow(clippy::too_many_arguments)]
fn apply_filter_pushdown<T: IndexSubject>(
    plan: &mut DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
    subject: T,
    index: Arc<String>,
    query: IndexQuery<QueryExpr<Variable>>,
    remaining: Vec<DynTree<ExprIR<Variable>>>,
    original_filter: &QueryExpr<Variable>,
    metadata: T::Metadata,
) {
    let keep_filter = needs_post_filter(original_filter, subject.alias().id);
    let subject = reorder_subject_labels(subject, &index);
    let scan_ir = subject.build_scan_ir(index, Arc::new(query), metadata);
    let mut op = plan.node_mut(idx);
    *op.data_mut() = scan_ir;

    if remaining.is_empty() {
        if !keep_filter {
            op.parent_mut().unwrap().take_out();
        }
        // else: leave the original filter as a runtime safety net.
    } else if keep_filter {
        // Some conjuncts were pushed, but the filter contains runtime
        // values the index may not resolve. Keep the *full* original
        // filter above so the pushed conjuncts act as a safety net
        // when the scan falls back to a label/type iterator — not
        // just the unpushed conjuncts (which would let false
        // positives through).
        *op.parent_mut().unwrap().data_mut() = IR::Filter(original_filter.clone());
    } else {
        let remaining_filter = if remaining.len() == 1 {
            Arc::new(remaining.into_iter().next().unwrap())
        } else {
            Arc::new(tree!(ExprIR::And; remaining))
        };
        *op.parent_mut().unwrap().data_mut() = IR::Filter(remaining_filter);
    }
}

/// Apply an inline-attr rewrite: replace the scan with an equality
/// index scan, optionally prefixed with a post-filter when the inline
/// value expression isn't a bare constant.
#[allow(clippy::needless_pass_by_value)]
fn apply_inline_rewrite<T: IndexSubject>(
    plan: &mut DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
    subject: T,
    label: Arc<String>,
    attr: Arc<String>,
    inline_filter: DynTree<ExprIR<Variable>>,
    metadata: T::Metadata,
) {
    if needs_inline_post_filter(&inline_filter) {
        plan.node_mut(idx)
            .push_parent(IR::Filter(Arc::new(inline_filter.clone())));
    }
    let query = Arc::new(IndexQuery::Equal {
        key: attr,
        value: Arc::new(inline_filter.root().child(1).clone_as_tree()),
    });
    let subject = reorder_subject_labels(subject, &label);
    *plan.node_mut(idx).data_mut() = subject.build_scan_ir(label, query, metadata);
}

/// Attempt an index rewrite at `idx` for subject kind `T`. Tries the
/// filter-pushdown path first (scan under a `Filter`), then the
/// inline-attr path (pattern with inline `{attr: value}`). Returns
/// `true` if either path modified the plan.
fn try_index_rewrite<T: IndexSubject>(
    plan: &mut DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
    graph: &Graph,
) -> bool {
    if let Some((subject, filter, metadata)) = match_scan_with_filter::<T>(plan, idx)
        && let Some((label, query, remaining)) = try_filter_pushdown(&subject, &filter, graph)
    {
        apply_filter_pushdown(
            plan, idx, subject, label, query, remaining, &filter, metadata,
        );
        return true;
    }

    if let Some((subject, metadata)) = T::match_scan_source(plan.node(idx).data())
        && let Some((_, label, attr, inline_filter)) = get_inline_attr_index(graph, &subject)
    {
        apply_inline_rewrite(plan, idx, subject, label, attr, inline_filter, metadata);
        return true;
    }

    false
}

/// Cleanup: prune a redundant `AllNodeScan` under an `EdgeByIndexScan`.
/// When the edge scan has no source-node constraint, the child scan
/// does nothing — the index directly yields edge endpoints.
///
/// Must not fire when the index query references variables bound by
/// the child scan: `MATCH (n)-[r:T]->() WHERE r.p = n.q` would leave
/// `n` unbound at runtime and fail expression evaluation.
fn prune_all_node_scan_child(
    plan: &mut DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
) -> bool {
    let IR::EdgeByIndexScan { query, .. } = plan.node(idx).data() else {
        return false;
    };
    let query = query.clone();
    let Some(child) = plan.node(idx).get_child(0) else {
        return false;
    };
    let IR::AllNodeScan(child_node) = child.data() else {
        return false;
    };
    let child_alias_id = child_node.alias.id;
    let child_idx = child.idx();
    // Safety: don't drop the scan that binds `child_alias_id` if the
    // edge-index query still depends on it.
    if index_query_references_var(&query, child_alias_id) {
        return false;
    }
    plan.node_mut(child_idx).prune();
    true
}

/// Walks every expression subtree inside an `IndexQuery` looking for a
/// `Variable` reference with the given alias id. Used by
/// `prune_all_node_scan_child` to avoid pruning a scan whose output is
/// still needed by the index query.
fn index_query_references_var(
    q: &IndexQuery<QueryExpr<Variable>>,
    alias_id: u32,
) -> bool {
    let expr_refs = |e: &QueryExpr<Variable>| -> bool {
        e.root()
            .indices::<Bfs>()
            .any(|i| matches!(e.node(i).data(), ExprIR::Variable(v) if v.id == alias_id))
    };
    match q {
        IndexQuery::Equal { value, .. } | IndexQuery::ArrayContains { value, .. } => {
            expr_refs(value)
        }
        IndexQuery::Range { min, max, .. } => {
            min.as_ref().is_some_and(expr_refs) || max.as_ref().is_some_and(expr_refs)
        }
        IndexQuery::Point { point, radius, .. } => expr_refs(point) || expr_refs(radius),
        IndexQuery::InList { list, .. } => expr_refs(list),
        IndexQuery::And(children) | IndexQuery::Or(children) => children
            .iter()
            .any(|c| index_query_references_var(c, alias_id)),
    }
}

/// Cleanup: add a `hasLabels` filter above an `EdgeByIndexScan` whose
/// `to` endpoint carries a label constraint. The edge index doesn't
/// enforce endpoint labels, so we filter them post-scan. Skipped when
/// the parent is already such a filter (prevents infinite re-addition).
fn add_to_labels_filter(
    plan: &mut DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
) -> bool {
    let IR::EdgeByIndexScan {
        relationship: rel, ..
    } = plan.node(idx).data()
    else {
        return false;
    };
    if rel.to.labels.is_empty() {
        return false;
    }
    let already_filtered = plan.node(idx).parent().is_some_and(|p| {
        if let IR::Filter(f) = p.data() {
            is_has_labels_for(f, rel.to.alias.id)
        } else {
            false
        }
    });
    if already_filtered {
        return false;
    }
    let to_alias = rel.to.alias.clone();
    let to_labels = rel.to.labels.clone();
    let filter_expr = build_has_labels_filter(&to_alias, to_labels.into_iter());
    plan.node_mut(idx).push_parent(IR::Filter(filter_expr));
    true
}

/// Replaces label scans with index scans where applicable and cleans
/// up the plan after edge-index rewrites.
///
/// Four passes, each driven to a fixed point independently to avoid
/// NodeIdx invalidation under orx-tree's Auto memory policy:
///
/// 1. Push property filters into `NodeByIndexScan`.
/// 2. Push property filters into `EdgeByIndexScan`.
/// 3. Prune redundant `AllNodeScan` children of `EdgeByIndexScan`.
/// 4. Re-add endpoint label filters the edge index doesn't enforce.
pub(super) fn utilize_index(
    optimized_plan: &mut DynTree<IR>,
    graph: &Graph,
) {
    rewrite_until_stable(optimized_plan, |plan, idx| {
        try_index_rewrite::<Arc<QueryNode<Arc<String>, Variable>>>(plan, idx, graph)
    });
    rewrite_until_stable(optimized_plan, |plan, idx| {
        try_index_rewrite::<Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>>(
            plan, idx, graph,
        )
    });
    rewrite_until_stable(optimized_plan, prune_all_node_scan_child);
    rewrite_until_stable(optimized_plan, add_to_labels_filter);
}

/// Returns true when `filter` is a `hasLabels(variable, [...])` call whose
/// first argument is a `Variable` with the given id.
fn is_has_labels_for(
    filter: &QueryExpr<Variable>,
    var_id: u32,
) -> bool {
    let root = filter.root();
    if let ExprIR::FuncInvocation(func) = root.data()
        && func.name == "hasLabels"
        && root.num_children() >= 1
        && let ExprIR::Variable(v) = root.child(0).data()
    {
        v.id == var_id
    } else {
        false
    }
}
