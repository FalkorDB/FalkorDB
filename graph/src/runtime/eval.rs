//! Standalone expression evaluator.
//!
//! [`ExprEval`] encapsulates all expression evaluation logic. It is used by
//! both the runtime (via `ExprEval::from_runtime`) and the optimizer (via
//! `ExprEval::constant()` for compile-time constant folding).
//!
//! ```text
//!  Expression tree (ExprIR)          Evaluation
//!  ========================          ==========
//!
//!        Add                     eval(Add)
//!       /   \                       |
//!    Mul     Int(1)              eval(Mul) + eval(Int(1))
//!   /   \                          |
//! Var(a) Var(b)               env[a] * env[b]  +  1
//! ```
//!
//! ## Two Evaluation Modes
//!
//! - **Runtime mode** (`ExprEval::from_runtime`): Full evaluation with access
//!   to the graph, environment bindings, parameters, and function calls.
//!   Used during query execution.
//!
//! - **Constant mode** (`ExprEval::constant`): No graph, no env, no functions.
//!   Used by the optimizer to fold constant sub-expressions at plan time.
//!   Returns `Err` if the expression references variables or functions.
//!
//! ## Evaluation Strategy
//!
//! The evaluator uses a hybrid approach:
//! - **Leaf nodes** (Null, Bool, Int, Float, String, Variable, Parameter, Map)
//!   are handled via fast-path early returns at the top of `eval()`.
//! - **Compound nodes** (Add, Sub, Mul, And, Or, List, etc.) use a stack-based
//!   iterative loop to avoid deep recursion. Children are pushed onto a stack
//!   and results accumulate in a `Vec<Value>`.
//! - **Complex nodes** (Quantifier, ListComprehension, Reduce, ShortestPath)
//!   fall back to recursive `eval()` calls since they need scoped env mutations.
//!
//! ## Iterable expressions
//!
//! [`ExprEval::eval_iter_expr`] evaluates a list-valued expression into an
//! optional [`RowIter`], the row-expanding shape consumed by `UNWIND` (via
//! the batched emitter) and the list-comprehension decomposition. It optimizes
//! `range()` and list literals by producing values on the fly — a lazy
//! [`RangeIter`] or an inline spread — without materializing the whole list.

use std::cmp::Ordering;
use std::sync::Arc;

use orx_tree::{Dyn, DynNode, DynTree, NodeIdx, NodeRef};
use smallvec::SmallVec;
use thin_vec::{ThinVec, thin_vec};

use crate::{
    graph::graph::NodeId,
    parser::ast::{ExprIR, QuantifierType, Variable},
    runtime::{
        batch::{Batch, BatchRow, Column, FloatLane, NullBitmap, classify_numeric},
        functions::{FnType, apply_pow},
        ops::batched_result_emitter::RowIter,
        ordermap::OrderMap,
        row::RowView,
        runtime::Runtime,
        value::{CompareValue, Contains, DisjointOrNull, Value},
    },
};

// ---------------------------------------------------------------------------
// RangeIter
// ---------------------------------------------------------------------------

/// Convenience `None` for the `env` argument of [`ExprEval::eval`] in
/// constant-evaluation contexts, where the row type cannot be inferred.
pub const NO_ROW: Option<&'static crate::runtime::row::Row> = None;

/// Classify bulk-evaluated values into a *lossless* column suitable for
/// equality / join-key use.
///
/// Produces [`Column::Ints`] only when every non-null value is an integer (the
/// common, fast case — it lets the join key directly on `i64` with no per-row
/// `Value`); anything else stays an exact [`Column::Values`]. Unlike
/// [`classify_column`](crate::runtime::batch::classify_column), it never coerces
/// a *mixed* int/float column to all-float: that conversion would round an
/// integer past 2^53 to the nearest `f64` and silently change which keys
/// compare equal, diverging from per-row evaluation.
fn classify_join_keys(values: Vec<Value>) -> (Column, NullBitmap) {
    let nulls = NullBitmap::from_values(&values);
    // Lossless: stop at all-`Int` (or null); never promote a mixed int/float
    // column to `f64`, which would round integers past 2^53 and silently change
    // which keys compare equal. `FloatLane::None` keeps `classify_numeric` from
    // ever yielding a float column, so the result is `Ints` or `Values` only.
    let column = classify_numeric(values, true, FloatLane::None);
    (column, nulls)
}

/// Lazy integer range iterator backing `UNWIND range(a, b, step)`: yields
/// `Value::Int`s on the fly without materialising an intermediate
/// `Value::List`. `step` is stored as a positive magnitude and `up` selects the
/// direction (ascending when the original step was positive). Boxed into a
/// [`RowIter::many`] by [`ExprEval::eval_iter_expr`].
struct RangeIter {
    current: i64,
    end: i64,
    step: i64,
    up: bool,
}

impl Iterator for RangeIter {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        if self.up {
            if self.current > self.end {
                return None;
            }
        } else if self.current < self.end {
            return None;
        }
        let val = self.current;
        self.current += if self.up { self.step } else { -self.step };
        Some(Value::Int(val))
    }
}

// ---------------------------------------------------------------------------
// ExprEval
// ---------------------------------------------------------------------------

/// Lazily yields the neighbours of a node by scanning single rows of the
/// graph's *versioned* adjacency / relationship matrices on demand, reusing
/// row iterators and one scratch buffer across calls.
///
/// Used by the `shortestPath` BFS so the per-query cost is proportional to the
/// edges actually traversed by the search rather than the whole graph's edge
/// count. Scanning the versioned matrices directly (base + delta-plus, minus
/// delta-minus) avoids materializing a merged adjacency matrix per call —
/// `VersionedMatrix::to_matrix()` duplicates the entire base matrix, which
/// dominated the shortestPath profile. Correct because BFS visits each node at
/// most once, so every row is scanned at most once.
struct NeighborIter {
    /// Forward (src → dst) row iterators, one per source matrix.
    fwd: Vec<crate::graph::graphblas::versioned_matrix::Iter>,
    /// Backward (dst → src) row iterators; empty for directed traversal.
    bwd: Vec<crate::graph::graphblas::versioned_matrix::Iter>,
    /// True when neighbours can repeat across iterators (multiple relationship
    /// types, or undirected traversal with reciprocal edges) and the buffer
    /// must be deduplicated.
    dedup: bool,
    buf: Vec<u64>,
}

impl NeighborIter {
    fn new(
        g: &crate::graph::graph::Graph,
        rel_types: &[Arc<String>],
        directed: bool,
    ) -> Self {
        let mut fwd = Vec::new();
        let mut bwd = Vec::new();
        if rel_types.is_empty() {
            // The adjacency matrix is the pair-level union of all relationship
            // types, so a single forward iterator suffices.
            fwd.push(g.adjacency_matrix().iter(0, 0));
            if !directed {
                for t in g.relationship_matrices_iter() {
                    bwd.push(t.matrix_t().iter(0, 0));
                }
            }
        } else {
            for t in rel_types
                .iter()
                .filter_map(|t| g.get_relationship_matrix(t))
            {
                fwd.push(t.structural_iter(0, 0));
                if !directed {
                    bwd.push(t.matrix_t().iter(0, 0));
                }
            }
        }
        let dedup = fwd.len() + bwd.len() > 1;
        Self {
            fwd,
            bwd,
            dedup,
            buf: Vec::new(),
        }
    }

    /// Iterator over *incoming* neighbours (predecessors) for a directed
    /// traversal, used as the backward side of the bidirectional BFS.
    /// Scans the transposed relationship matrices: row `n` of `matrix_t`
    /// holds the sources of edges pointing at `n`.
    fn new_reversed(
        g: &crate::graph::graph::Graph,
        rel_types: &[Arc<String>],
    ) -> Self {
        let mut fwd = Vec::new();
        if rel_types.is_empty() {
            for t in g.relationship_matrices_iter() {
                fwd.push(t.matrix_t().iter(0, 0));
            }
        } else {
            for t in rel_types
                .iter()
                .filter_map(|t| g.get_relationship_matrix(t))
            {
                fwd.push(t.matrix_t().iter(0, 0));
            }
        }
        let dedup = fwd.len() > 1;
        Self {
            fwd,
            bwd: Vec::new(),
            dedup,
            buf: Vec::new(),
        }
    }

    /// Returns the neighbours of `node`. The returned slice is valid until
    /// the next call to `neighbors`.
    fn neighbors(
        &mut self,
        node: u64,
    ) -> &[u64] {
        self.buf.clear();
        for it in &mut self.fwd {
            it.seek(node, node);
            for (_, col) in it.by_ref() {
                self.buf.push(col);
            }
        }
        for it in &mut self.bwd {
            it.seek(node, node);
            for (_, col) in it.by_ref() {
                self.buf.push(col);
            }
        }
        if self.dedup {
            self.buf.sort_unstable();
            self.buf.dedup();
        }
        &self.buf
    }
}

/// A resolved node of an expression tree.
///
/// Navigating children through a node (`child`, `children`) is pure pointer
/// chasing, while resolving a `NodeIdx` via `tree.node(idx)` validates the
/// index on every call (memory-state check plus a scan of the tree's pinned
/// memory fragments). The evaluator therefore validates an idx once at the
/// [`ExprEval::eval`] boundary and recurses on `ExprNode`s; hot per-row
/// callers can pre-resolve nodes once and call [`ExprEval::eval_node`]
/// directly, paying no validation at all.
pub type ExprNode<'t> = DynNode<'t, ExprIR<Variable>>;

/// Shared expression evaluator used by both the runtime and the optimizer.
pub struct ExprEval<'a> {
    /// Full runtime context. `None` when evaluating constant expressions at
    /// plan time (optimizer).
    runtime: Option<&'a Runtime<'a>>,
}

impl<'a> ExprEval<'a> {
    /// Full evaluation context backed by a [`Runtime`].
    pub const fn from_runtime(rt: &'a Runtime<'a>) -> Self {
        Self { runtime: Some(rt) }
    }

    /// Constant-only evaluation — no graph, no env, no functions.
    /// Any non-constant branch returns `Err`.
    #[must_use]
    pub const fn constant() -> Self {
        Self { runtime: None }
    }

    /// Convenience: unwrap the runtime or return a descriptive error.
    fn rt(&self) -> Result<&'a Runtime<'a>, String> {
        self.runtime
            .ok_or_else(|| String::from("not a constant expression"))
    }

    /// Resolve an environment variable.
    // Hot path: forced inlining measurably matters here.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn resolve_var<R: RowView + ?Sized>(
        env: Option<&R>,
        x: &Variable,
    ) -> Result<Value, String> {
        env.and_then(|e| e.value_at(x.id))
            .ok_or_else(|| format!("Variable {} not found", x.as_str()))
    }

    // -------------------------------------------------------------------
    // Main evaluator
    // -------------------------------------------------------------------

    pub fn eval<R: RowView + ?Sized>(
        &self,
        ir: &DynTree<ExprIR<Variable>>,
        idx: NodeIdx<Dyn<ExprIR<Variable>>>,
        env: Option<&R>,
        agg_group_key: Option<u64>,
    ) -> Result<Value, String> {
        // The only idx validation on the whole evaluation path; everything
        // below recurses on `ExprNode`s (see the type's docs).
        self.eval_node(&ir.node(idx), env, agg_group_key)
    }

    /// Evaluate a child expression that is statistically almost always a leaf
    /// (`Constant` / `Variable`) without the overhead of a nested `eval_node`
    /// call: the leaf cases resolve inline at the call site; anything else
    /// falls back to the full evaluator.
    #[inline]
    fn eval_operand<R: RowView + ?Sized>(
        &self,
        child: &ExprNode<'_>,
        env: Option<&R>,
        agg_group_key: Option<u64>,
    ) -> Result<Value, String> {
        match child.data() {
            ExprIR::Constant(v) => Ok(v.clone()),
            ExprIR::Variable(x) => Self::resolve_var(env, x),
            _ => self.eval_node(child, env, agg_group_key),
        }
    }

    #[inline]
    pub fn eval_node<R: RowView + ?Sized>(
        &self,
        node: &ExprNode<'_>,
        env: Option<&R>,
        agg_group_key: Option<u64>,
    ) -> Result<Value, String> {
        // Fast-path early returns for leaf / simple nodes.
        match node.data() {
            ExprIR::Constant(v) => return Ok(v.clone()),
            ExprIR::Variable(x) => {
                return Self::resolve_var(env, x);
            }
            ExprIR::Parameter(x) => {
                let rt = self.rt()?;
                return rt.parameters.get(x).map_or_else(
                    || Err(format!("Parameter {x} not found")),
                    |v| Ok(v.clone()),
                );
            }
            ExprIR::Map => {
                let mut map = OrderMap::with_capacity(node.num_children());
                for child in node.children() {
                    let ExprIR::Constant(Value::String(key)) = child.data() else {
                        return Err("Map key must be a string".into());
                    };
                    let value = self.eval_node(&child.child(0), env, agg_group_key)?;
                    map.insert(key.clone(), value);
                }
                return Ok(Value::Map(Arc::new(map)));
            }
            ExprIR::MapProjection => {
                return self.eval_map_projection(node, env, agg_group_key);
            }
            ExprIR::ShortestPath(info) => {
                return self.eval_shortest_path(
                    node,
                    env,
                    agg_group_key,
                    &info.rel_types,
                    info.min_hops,
                    info.max_hops,
                    info.directed,
                );
            }
            // Property access (`n.prop`) is the most common non-leaf expression
            // (join keys, filters, projections, sort keys). Handle it here so it
            // never falls through to the allocating stack-based loop below, which
            // would heap-allocate `stack` + `res` on every call. The child is
            // usually a `Variable` (its own fast path), so this resolves with no
            // scratch allocation. Mirrors the in-loop `Property` handler.
            ExprIR::Property(attr) => {
                let obj = self.eval_operand(&node.child(0), env, agg_group_key)?;
                return match obj {
                    Value::Node(id) => {
                        let rt = self.rt()?;
                        Ok(rt.get_node_attribute(id, attr).unwrap_or(Value::Null))
                    }
                    Value::Relationship(rel) => {
                        let rt = self.rt()?;
                        Ok(rt
                            .get_relationship_attribute(rel, attr)
                            .unwrap_or(Value::Null))
                    }
                    other => other.get_attr(attr),
                };
            }
            // Binary `a + b` (e.g. `x + 1` in CREATE/SET attribute maps) is
            // common enough to keep out of the allocating stack loop; the
            // operands are usually leaves that `eval_operand` resolves inline.
            ExprIR::Add if node.num_children() == 2 => {
                let lhs = self.eval_operand(&node.child(0), env, agg_group_key)?;
                let rhs = self.eval_operand(&node.child(1), env, agg_group_key)?;
                return lhs + rhs;
            }
            _ => {}
        }

        self.eval_compound(node, env, agg_group_key)
    }

    /// Out-of-line continuation of [`eval_node`](Self::eval_node) for
    /// compound expressions. Kept separate so `eval_node`'s leaf/simple
    /// dispatch stays small enough to inline into hot per-row loops without
    /// dragging the stack machine's code (and stack frame) along.
    fn eval_compound<R: RowView + ?Sized>(
        &self,
        node: &ExprNode<'_>,
        env: Option<&R>,
        agg_group_key: Option<u64>,
    ) -> Result<Value, String> {
        // Stack-based iterative evaluation. Both scratch buffers are stack-inline
        // SmallVecs, so typical expressions evaluate with zero heap allocation;
        // unusually large ones (long list literals / many-arg functions) spill to
        // the heap exactly like a Vec, so this is never an allocation regression.
        let mut res_owned: SmallVec<[Value; 4]> = SmallVec::new();
        let res = &mut res_owned;

        let mut stack: SmallVec<[_; 4]> = SmallVec::new();
        stack.push((node.clone(), false));
        while let Some((node, reenter)) = stack.pop() {
            match node.data() {
                ExprIR::Constant(v) => res.push(v.clone()),
                ExprIR::Variable(x) => res.push(Self::resolve_var(env, x)?),
                ExprIR::Parameter(x) => res.push(self.rt()?.parameters.get(x).map_or_else(
                    || Err(format!("Parameter {x} not found")),
                    |v| Ok(v.clone()),
                )?),
                ExprIR::List => {
                    if reenter {
                        let mut list = thin_vec![];
                        for _ in 0..node.num_children() {
                            list.push(res.pop().unwrap());
                        }
                        res.push(Value::List(Arc::new(list)));
                    } else if node.num_children() > 0 {
                        stack.push((node.clone(), true));
                        for child in node.children() {
                            stack.push((child, false));
                        }
                    } else {
                        res.push(Value::List(Arc::new(thin_vec![])));
                    }
                }
                ExprIR::Length => match self.eval_node(&node.child(0), env, agg_group_key)? {
                    Value::List(arr) => res.push(Value::Int(arr.len() as _)),
                    _ => return Err(String::from("Length operator requires a list")),
                },
                ExprIR::GetElement => {
                    let arr = self.eval_node(&node.child(0), env, agg_group_key)?;
                    let i = self.eval_node(&node.child(1), env, agg_group_key)?;
                    match (arr, i) {
                        (Value::List(values), Value::Int(i)) => {
                            let len = values.len() as i64;
                            let normalized_index = if i < 0 { len + i } else { i };
                            if normalized_index >= 0 && normalized_index < len {
                                res.push(values[normalized_index as usize].clone());
                            } else {
                                res.push(Value::Null);
                            }
                        }
                        (Value::List(_), Value::Null) => {
                            res.push(Value::Null);
                        }
                        (Value::List(_), v) => {
                            return Err(format!(
                                "Type mismatch: expected Integer but was {}",
                                v.name()
                            ));
                        }
                        (Value::Node(id), Value::String(key)) => {
                            let rt = self.rt()?;
                            res.push(rt.get_node_attribute(id, &key).unwrap_or(Value::Null));
                        }
                        (Value::Relationship(rel), Value::String(key)) => {
                            let rt = self.rt()?;
                            res.push(
                                rt.get_relationship_attribute(rel, &key)
                                    .unwrap_or(Value::Null),
                            );
                        }
                        (Value::Map(map), Value::String(key)) => {
                            res.push(map.get(&key).map_or(Value::Null, std::clone::Clone::clone));
                        }
                        (Value::Map(_), Value::Null) | (Value::Null, _) => res.push(Value::Null),
                        v => return Err(format!("Type mismatch: unexpected types {v:?}")),
                    }
                }
                ExprIR::GetElements => {
                    let arr = self.eval_node(&node.child(0), env, agg_group_key)?;
                    let a = self.eval_node(&node.child(1), env, agg_group_key)?;
                    let b = self.eval_node(&node.child(2), env, agg_group_key)?;
                    res.push(get_elements(&arr, &a, &b)?);
                }
                ExprIR::IsNode => match self.eval_node(&node.child(0), env, agg_group_key)? {
                    Value::Node(_) => res.push(Value::Bool(true)),
                    _ => res.push(Value::Bool(false)),
                },
                ExprIR::IsRelationship => {
                    match self.eval_node(&node.child(0), env, agg_group_key)? {
                        Value::Relationship(_) => res.push(Value::Bool(true)),
                        _ => res.push(Value::Bool(false)),
                    }
                }
                ExprIR::Or => {
                    let mut is_null = false;
                    let mut found = false;
                    for child in node.children() {
                        match self.eval_node(&child, env, agg_group_key)? {
                            Value::Bool(true) => {
                                found = true;
                                res.push(Value::Bool(true));
                                break;
                            }
                            Value::Bool(false) => {}
                            Value::Null => is_null = true,
                            ir => {
                                return Err(format!("Type mismatch: expected Bool but was {ir:?}"));
                            }
                        }
                    }
                    if !found {
                        if is_null {
                            res.push(Value::Null);
                        } else {
                            res.push(Value::Bool(false));
                        }
                    }
                }
                ExprIR::Xor => {
                    let mut last = None;
                    let mut found = false;
                    for child in node.children() {
                        match self.eval_node(&child, env, agg_group_key)? {
                            Value::Bool(b) => last = Some(last.map_or(b, |l| logical_xor(l, b))),
                            Value::Null => {
                                found = true;
                                res.push(Value::Null);
                                break;
                            }
                            ir => {
                                return Err(format!("Type mismatch: expected Bool but was {ir:?}"));
                            }
                        }
                    }
                    if !found {
                        res.push(Value::Bool(last.unwrap_or(false)));
                    }
                }
                ExprIR::And => {
                    let mut is_null = false;
                    let mut found = false;
                    for child in node.children() {
                        match self.eval_node(&child, env, agg_group_key)? {
                            Value::Bool(false) => {
                                found = true;
                                res.push(Value::Bool(false));
                                break;
                            }
                            Value::Bool(true) => {}
                            Value::Null => is_null = true,
                            ir => {
                                return Err(format!("Type mismatch: expected Bool but was {ir:?}"));
                            }
                        }
                    }
                    if !found {
                        if is_null {
                            res.push(Value::Null);
                        } else {
                            res.push(Value::Bool(true));
                        }
                    }
                }
                ExprIR::Not => match self.eval_node(&node.child(0), env, agg_group_key)? {
                    Value::Bool(b) => res.push(Value::Bool(!b)),
                    Value::Null => res.push(Value::Null),
                    v => {
                        return Err(format!(
                            "Type mismatch: expected Boolean or Null but was {}",
                            v.name()
                        ));
                    }
                },
                ExprIR::Negate => match self.eval_node(&node.child(0), env, agg_group_key)? {
                    Value::Int(i) => res.push(Value::Int(i.checked_neg().ok_or_else(|| {
                        String::from("ArgumentError: integer overflow in unary minus")
                    })?)),
                    Value::Float(f) => res.push(Value::Float(-f)),
                    Value::Null => res.push(Value::Null),
                    v => {
                        return Err(format!(
                            "Type mismatch: expected Integer, Float, or Null but was {}",
                            v.name()
                        ));
                    }
                },
                ExprIR::Eq => res.push(all_equals(
                    node.children()
                        .map(|child| self.eval_node(&child, env, agg_group_key)),
                )?),
                ExprIR::Neq => res.push(all_not_equals(
                    node.children()
                        .map(|child| self.eval_node(&child, env, agg_group_key)),
                )?),
                ExprIR::Lt => match self
                    .eval_node(&node.child(0), env, agg_group_key)?
                    .compare_value(&self.eval_node(&node.child(1), env, agg_group_key)?)
                {
                    (_, DisjointOrNull::ComparedNull | DisjointOrNull::Disjoint) => {
                        res.push(Value::Null);
                    }
                    (_, DisjointOrNull::NaN) => res.push(Value::Bool(false)),
                    (Ordering::Less, _) => res.push(Value::Bool(true)),
                    _ => res.push(Value::Bool(false)),
                },
                ExprIR::Gt => match self
                    .eval_node(&node.child(0), env, agg_group_key)?
                    .compare_value(&self.eval_node(&node.child(1), env, agg_group_key)?)
                {
                    (_, DisjointOrNull::ComparedNull | DisjointOrNull::Disjoint) => {
                        res.push(Value::Null);
                    }
                    (_, DisjointOrNull::NaN) => res.push(Value::Bool(false)),
                    (Ordering::Greater, _) => res.push(Value::Bool(true)),
                    _ => res.push(Value::Bool(false)),
                },
                ExprIR::Le => match self
                    .eval_node(&node.child(0), env, agg_group_key)?
                    .compare_value(&self.eval_node(&node.child(1), env, agg_group_key)?)
                {
                    (_, DisjointOrNull::ComparedNull | DisjointOrNull::Disjoint) => {
                        res.push(Value::Null);
                    }
                    (_, DisjointOrNull::NaN) => res.push(Value::Bool(false)),
                    (Ordering::Less | Ordering::Equal, _) => res.push(Value::Bool(true)),
                    _ => res.push(Value::Bool(false)),
                },
                ExprIR::Ge => match self
                    .eval_node(&node.child(0), env, agg_group_key)?
                    .compare_value(&self.eval_node(&node.child(1), env, agg_group_key)?)
                {
                    (_, DisjointOrNull::ComparedNull | DisjointOrNull::Disjoint) => {
                        res.push(Value::Null);
                    }
                    (_, DisjointOrNull::NaN) => res.push(Value::Bool(false)),
                    (Ordering::Greater | Ordering::Equal, _) => res.push(Value::Bool(true)),
                    _ => res.push(Value::Bool(false)),
                },
                ExprIR::In => {
                    let value = self.eval_node(&node.child(0), env, agg_group_key)?;
                    let list = self.eval_node(&node.child(1), env, agg_group_key)?;
                    res.push(list_contains(&list, value)?);
                }
                ExprIR::Add => res.push(
                    node.children()
                        .map(|child| self.eval_node(&child, env, agg_group_key))
                        .reduce(|acc, value| acc? + value?)
                        .ok_or_else(|| {
                            String::from("Add operator requires at least one operand")
                        })??,
                ),
                ExprIR::Sub => res.push(
                    node.children()
                        .map(|child| self.eval_node(&child, env, agg_group_key))
                        .reduce(|acc, value| acc? - value?)
                        .ok_or_else(|| {
                            String::from("Sub operator requires at least one argument")
                        })??,
                ),
                ExprIR::Mul => res.push(
                    node.children()
                        .map(|child| self.eval_node(&child, env, agg_group_key))
                        .reduce(|acc, value| acc? * value?)
                        .ok_or_else(|| {
                            String::from("Mul operator requires at least one argument")
                        })??,
                ),
                ExprIR::Div => res.push(
                    node.children()
                        .map(|child| self.eval_node(&child, env, agg_group_key))
                        .reduce(|acc, value| acc? / value?)
                        .ok_or_else(|| {
                            String::from("Div operator requires at least one argument")
                        })??,
                ),
                ExprIR::Modulo => res.push(
                    node.children()
                        .map(|child| self.eval_node(&child, env, agg_group_key))
                        .reduce(|acc, value| acc? % value?)
                        .ok_or_else(|| {
                            String::from("Modulo operator requires at least one argument")
                        })??,
                ),
                ExprIR::Pow => res.push(
                    node.children()
                        .flat_map(|child| self.eval_node(&child, env, agg_group_key))
                        .reduce(apply_pow)
                        .ok_or_else(|| {
                            String::from("Pow operator requires at least one argument")
                        })?,
                ),
                ExprIR::Distinct => {
                    let rt = self.rt()?;
                    let group_id = agg_group_key.unwrap();
                    let values = node
                        .children()
                        .map(|child| self.eval_node(&child, env, agg_group_key))
                        .collect::<Result<ThinVec<_>, _>>()?;
                    let mut value_dedupers = rt.value_dedupers.borrow_mut();
                    let value_deduper = value_dedupers.entry((node.idx(), group_id)).or_default();
                    if value_deduper.is_seen(&values) {
                        res.push(Value::List(Arc::new(thin_vec![Value::Null])));
                    } else {
                        res.push(Value::List(Arc::new(values)));
                    }
                }
                ExprIR::Property(attr) => {
                    let obj = self.eval_node(&node.child(0), env, agg_group_key)?;
                    match obj {
                        Value::Node(id) => {
                            let rt = self.rt()?;
                            res.push(rt.get_node_attribute(id, attr).unwrap_or(Value::Null));
                        }
                        Value::Relationship(rel) => {
                            let rt = self.rt()?;
                            res.push(
                                rt.get_relationship_attribute(rel, attr)
                                    .unwrap_or(Value::Null),
                            );
                        }
                        other => {
                            res.push(other.get_attr(attr)?);
                        }
                    }
                }
                ExprIR::FuncInvocation(func) => {
                    let rt = self.rt()?;
                    if agg_group_key.is_none()
                        && let FnType::Aggregation {
                            finalizer: finalize,
                            ..
                        } = &func.fn_type
                        && let ExprIR::Variable(key) = node.child(node.num_children() - 1).data()
                    {
                        let acc = env
                            .and_then(|e| e.value_at(key.id))
                            .ok_or_else(|| String::from("Variable not found"))?;

                        return match finalize {
                            Some(func) => Ok((func)(acc)),
                            None => Ok(acc),
                        };
                    }
                    // Fast path for struct-constructor functions (duration,
                    // date, point, ...): when the binder rewrote a
                    // `{key: value}` map call into positional children, write
                    // child values into a stack array and dispatch the
                    // slice-based struct_fn directly. Bypasses the per-row
                    // ThinVec<Value> heap alloc, the validate_args_type walk,
                    // and the Arc<RuntimeFn> dispatch hop. The
                    // `num_children > 1` guard distinguishes the rewritten
                    // positional form from the regular `duration({...})` call
                    // with a single Map argument. Max 10 slots
                    // (matches `localdatetime`).
                    if let Some(struct_fn) = func.struct_fn
                        && node.num_children() > 1
                    {
                        let n = node.num_children();
                        debug_assert!(n <= 10);
                        let mut slots: [Value; 10] = std::array::from_fn(|_| Value::Null);
                        for (i, child) in node.children().enumerate() {
                            if !matches!(child.data(), ExprIR::Constant(Value::Null)) {
                                slots[i] = self.eval_node(&child, env, agg_group_key)?;
                            }
                        }
                        res.push(struct_fn(&slots[..n])?);
                        continue;
                    }
                    // Distinct-prefixed aggregate (e.g. `count(DISTINCT x, acc)`):
                    // the first child carries the already-deduplicated list;
                    // flatten it with the accumulator. Needs owned ownership of
                    // the inner list to avoid a copy, so it stays on the
                    // ThinVec path.
                    if node.num_children() == 2 && matches!(node.child(0).data(), ExprIR::Distinct)
                    {
                        let mut args = node
                            .children()
                            .map(|child| self.eval_node(&child, env, agg_group_key))
                            .collect::<Result<ThinVec<_>, _>>()?;
                        match args.remove(0) {
                            Value::List(values) => {
                                let mut values = Arc::unwrap_or_clone(values);
                                values.append(&mut args);
                                args = values;
                            }
                            _ => unreachable!(),
                        }
                        func.validate_args_type(&args)?;
                        if !rt.write && func.write {
                            return Err(String::from(
                                "graph.RO_QUERY is to be executed only on read-only queries",
                            ));
                        }
                        res.push(func.func.call(rt, &args)?);
                        continue;
                    }

                    // Common path: evaluate children directly onto the shared
                    // `res` stack and pass a slice to the function. Eliminates
                    // the per-call ThinVec<Value> heap allocation that an
                    // intermediate `args` collection would do — same trick the
                    // struct_fn fast path above uses with a stack array.
                    let base = res.len();
                    for child in node.children() {
                        let v = self.eval_node(&child, env, agg_group_key)?;
                        res.push(v);
                    }
                    func.validate_args_type(&res[base..])?;
                    if !rt.write && func.write {
                        return Err(String::from(
                            "graph.RO_QUERY is to be executed only on read-only queries",
                        ));
                    }
                    let out = func.func.call(rt, &res[base..])?;
                    res.truncate(base);
                    res.push(out);
                }
                ExprIR::Map => {
                    let mut map = OrderMap::with_capacity(node.num_children());
                    for child in node.children() {
                        let ExprIR::Constant(Value::String(key)) = child.data() else {
                            return Err("Map key must be a string".into());
                        };
                        let value = self.eval_node(&child.child(0), env, agg_group_key)?;
                        map.insert(key.clone(), value);
                    }
                    res.push(Value::Map(Arc::new(map)));
                }
                ExprIR::MapProjection => {
                    res.push(self.eval_map_projection(&node, env, agg_group_key)?);
                }
                ExprIR::Quantifier {
                    quantifier_type: quantifier,
                    var,
                } => {
                    let list = self.eval_node(&node.child(0), env, agg_group_key)?;
                    match list {
                        Value::List(values) => {
                            let e = env.ok_or_else(|| String::from("Variable not found"))?;
                            let mut row = e.to_owned_row();
                            let mut t = 0;
                            let mut f = 0;
                            let mut n = 0;
                            for value in values.iter().cloned() {
                                row.insert(var, value);

                                match self.eval_node(&node.child(1), Some(&row), agg_group_key)? {
                                    Value::Bool(true) => t += 1,
                                    Value::Bool(false) => f += 1,
                                    Value::Null => n += 1,
                                    value => {
                                        return Err(format!(
                                            "Type mismatch: expected Boolean but was {}",
                                            value.name()
                                        ));
                                    }
                                }
                            }

                            res.push(eval_quantifier(quantifier, t, f, n));
                        }
                        Value::Null => res.push(Value::Null),
                        value => {
                            return Err(format!(
                                "Type mismatch: expected List but was {}",
                                value.name()
                            ));
                        }
                    }
                }
                ExprIR::ListComprehension(var) => {
                    let e = env.ok_or_else(|| String::from("Variable not found"))?;
                    let mut row = e.to_owned_row();
                    let mut acc = thin_vec![];
                    if let Some(result) = self.eval_iter_expr(&node.child(0), env)? {
                        for value in result {
                            row.insert(var, value);
                            match self.eval_node(&node.child(1), Some(&row), agg_group_key)? {
                                Value::Bool(true) => {}
                                _ => continue,
                            }
                            acc.push(self.eval_node(&node.child(2), Some(&row), agg_group_key)?);
                        }
                    }

                    res.push(Value::List(Arc::new(acc)));
                }
                ExprIR::Reduce(vars) => {
                    let (acc_var, iter_var) = (&vars.accumulator, &vars.iterator);
                    // child[0] = init, child[1] = list, child[2] = body
                    let init = self.eval_node(&node.child(0), env, agg_group_key)?;
                    let list = self.eval_node(&node.child(1), env, agg_group_key)?;
                    match list {
                        Value::List(values) => {
                            let e = env.ok_or_else(|| String::from("Variable not found"))?;
                            let mut row = e.to_owned_row();
                            let mut accumulator = init;
                            for value in values.iter().cloned() {
                                row.insert(acc_var, accumulator);
                                row.insert(iter_var, value);
                                accumulator =
                                    self.eval_node(&node.child(2), Some(&row), agg_group_key)?;
                            }
                            res.push(accumulator);
                        }
                        Value::Null => res.push(Value::Null),
                        value => {
                            return Err(format!(
                                "Type mismatch: expected List but was {}",
                                value.name()
                            ));
                        }
                    }
                }
                ExprIR::PatternComprehension(_) => {
                    unreachable!("PatternComprehension should be handled by the planner")
                }
                ExprIR::Paren => {
                    res.push(self.eval_node(&node.child(0), env, agg_group_key)?);
                }
                ExprIR::Pattern(_) => {
                    unreachable!("Pattern should be handled by the planner")
                }
                ExprIR::ShortestPath(_) => {
                    unreachable!("ShortestPath should be handled in the early-return section")
                }
            }
        }
        debug_assert_eq!(res.len(), 1);
        let result = res.pop().unwrap();
        Ok(result)
    }

    /// Bulk-evaluate `tree` over the `active` rows of `batch`, returning a
    /// lossless typed [`Column`] plus a [`NullBitmap`] (entry `i` corresponds to
    /// row `active[i]`).
    ///
    /// This is the columnar counterpart to [`eval`](Self::eval): it avoids the
    /// per-row interpreter dispatch for the shape that dominates hot paths.
    /// `var.attr`, where `var` is a bound node column, is served by a single
    /// bulk attribute fetch yielding a primitive [`Column::Ints`] (or exact
    /// [`Column::Values`]) with no per-row `Value` allocation; every other shape
    /// falls back to per-row `eval` collected into a column. Either way the
    /// result is *lossless* — an all-integer column stays `Ints`, anything else
    /// stays exact `Values` (never coerced to all-float) — so null handling,
    /// error propagation, and which values compare equal match evaluating each
    /// row individually.
    pub fn eval_batch(
        &self,
        tree: &DynTree<ExprIR<Variable>>,
        batch: &Batch<'_>,
        active: &[usize],
    ) -> Result<(Column, NullBitmap), String> {
        if let Some(result) = self.try_eval_batch_property(tree, batch, active)? {
            return Ok(result);
        }
        self.eval_batch_per_row(tree, batch, active)
    }

    /// If `tree` is a node property access `var.attr` over a bound node column,
    /// bulk-fetch the property for all active rows in one shot and classify it
    /// losslessly. Returns `None` for any other shape, so the caller falls back
    /// to per-row evaluation.
    fn try_eval_batch_property(
        &self,
        tree: &DynTree<ExprIR<Variable>>,
        batch: &Batch<'_>,
        active: &[usize],
    ) -> Result<Option<(Column, NullBitmap)>, String> {
        let root = tree.root();
        let ExprIR::Property(attr) = root.data() else {
            return Ok(None);
        };
        if root.num_children() != 1 {
            return Ok(None);
        }
        let ExprIR::Variable(var) = root.child(0).data() else {
            return Ok(None);
        };
        let Some(node_ids) = batch.extract_node_ids(var.id) else {
            return Ok(None);
        };
        let active_ids: Vec<NodeId> = active.iter().map(|&i| node_ids[i]).collect();
        let values = self
            .rt()?
            .materialize_node_property_values(&active_ids, attr);
        Ok(Some(classify_join_keys(values)))
    }

    /// Per-row fallback for [`eval_batch`](Self::eval_batch): evaluate `tree`
    /// against each active row through a borrowed [`BatchRow`] and classify the
    /// collected results losslessly. Used for any shape the bulk fast path does
    /// not cover (arithmetic, function calls, non-node properties).
    fn eval_batch_per_row(
        &self,
        tree: &DynTree<ExprIR<Variable>>,
        batch: &Batch<'_>,
        active: &[usize],
    ) -> Result<(Column, NullBitmap), String> {
        let root = tree.root();
        let mut values = Vec::with_capacity(active.len());
        for &row in active {
            let view = BatchRow::new(batch, row);
            values.push(self.eval_node(&root, Some(&view), None)?);
        }
        Ok(classify_join_keys(values))
    }

    // -------------------------------------------------------------------
    // Companion methods
    // -------------------------------------------------------------------

    /// Evaluate a list-valued expression into an optional [`RowIter`] — the
    /// row-expanding shape consumed by `UNWIND` (via the batched emitter) and
    /// the list-comprehension decomposition. `range(..)` and list literals get
    /// dedicated lazy/inline shapes so neither materializes an intermediate
    /// `Value::List`:
    ///
    /// - an empty `range(..)` / a `null` / a `null` scalar -> `None` (the row
    ///   produces no output rows),
    /// - a single non-null scalar -> [`RowIter::one`],
    /// - a list *literal* -> [`RowIter::spread`] (inline smallvec, no boxing),
    /// - a lazy `range(..)` or a materialized list value -> [`RowIter::many`].
    pub(crate) fn eval_iter_expr<R: RowView + ?Sized>(
        &self,
        node: &ExprNode<'_>,
        env: Option<&R>,
    ) -> Result<Option<RowIter<'a, Value>>, String> {
        match node.data() {
            ExprIR::FuncInvocation(func) if func.name == "range" => {
                let start = self.eval_node(&node.child(0), env, None)?;
                let end = self.eval_node(&node.child(1), env, None)?;
                let step = node
                    .get_child(2)
                    .map_or_else(|| Ok(Value::Int(1)), |c| self.eval_node(&c, env, None))?;
                func.validate_args_type(&[&start, &end, &step])?;
                let (Value::Int(start), Value::Int(end), Value::Int(step)) = (start, end, step)
                else {
                    unreachable!();
                };
                if step == 0 {
                    return Err(String::from(
                        "ArgumentError: step argument to range() can't be 0",
                    ));
                }
                if (start > end && step > 0) || (start < end && step < 0) {
                    return Ok(None);
                }
                let span = if end >= start {
                    (end as i128) - (start as i128)
                } else {
                    (start as i128) - (end as i128)
                };
                let abs_step = (step as i128).unsigned_abs();
                let length = (span.unsigned_abs() / abs_step)
                    .checked_add(1)
                    .ok_or_else(|| String::from("Range too large"))?;
                if length > u32::MAX as u128 {
                    return Err(String::from("Range too large"));
                }
                let iter = RangeIter {
                    current: start,
                    end,
                    step: step.unsigned_abs() as i64,
                    up: step > 0,
                };
                Ok(Some(RowIter::many(Box::new(iter))))
            }
            ExprIR::List => {
                // Fuse `UNWIND [a, b, c]`: evaluate the element expressions
                // directly into inline storage instead of building a
                // `Value::List(Arc<ThinVec>)` only to immediately unwrap and
                // iterate it. This avoids the per-row `Arc` + `ThinVec`
                // allocation for the list literal.
                let mut values: SmallVec<[Value; 4]> = SmallVec::with_capacity(node.num_children());
                for child in node.children() {
                    values.push(self.eval_node(&child, env, None)?);
                }
                Ok(Some(RowIter::spread(values.into_iter())))
            }
            _ => {
                let res = self.eval_node(node, env, None)?;
                Ok(match res {
                    Value::List(arr) => Some(RowIter::many(Box::new(
                        Arc::unwrap_or_clone(arr).into_iter(),
                    ))),
                    Value::Null => None,
                    other => Some(RowIter::one(other)),
                })
            }
        }
    }

    /// Evaluate a `shortestPath()` or `allShortestPaths()` expression.
    ///
    /// Children: [source_var_expr, dest_var_expr]
    /// Returns a `Path` value (alternating nodes and edges) or `Null`.
    #[allow(clippy::too_many_arguments)]
    fn eval_shortest_path<R: RowView + ?Sized>(
        &self,
        node: &ExprNode<'_>,
        env: Option<&R>,
        agg_group_key: Option<u64>,
        rel_types: &[Arc<String>],
        min_hops: u32,
        max_hops: Option<u32>,
        directed: bool,
    ) -> Result<Value, String> {
        let src_val = self.eval_node(&node.child(0), env, agg_group_key)?;
        let dst_val = self.eval_node(&node.child(1), env, agg_group_key)?;

        let src_id = match &src_val {
            Value::Node(id) => *id,
            Value::Null => return Ok(Value::Null),
            _ => return Err("A shortestPath requires bound nodes".into()),
        };
        let dst_id = match &dst_val {
            Value::Node(id) => *id,
            Value::Null => return Ok(Value::Null),
            _ => return Err("A shortestPath requires bound nodes".into()),
        };

        let rt = self.rt()?;
        let g = rt.g.borrow();

        // min_hops == 0: if src == dest, return single-node path
        if min_hops == 0 && src_id == dst_id {
            let path: ThinVec<Value> = thin_vec![Value::Node(src_id)];
            return Ok(Value::Path(Arc::new(path)));
        }

        // Fetch neighbours lazily, one matrix row at a time, scanning the
        // versioned adjacency / relationship matrices directly. This avoids
        // materializing a merged adjacency matrix per call (a full duplicate
        // of the base matrix) and keeps the per-query cost proportional to
        // the edges actually traversed by the search. For undirected
        // traversal the backward (transposed) tensor matrices supply incoming
        // neighbours; duplicates are removed in NeighborIter.
        let max_level = max_hops.map_or(u64::MAX, |m| m as u64);
        let node_cap = g.node_cap();

        // Single shortest path via bidirectional BFS. The backward side
        // follows incoming edges for directed traversal; for undirected
        // traversal neighbours are symmetric so both sides use the same
        // (separately-stateful) iterator construction.
        let mut fwd_nbrs = NeighborIter::new(&g, rel_types, directed);
        let mut bwd_nbrs = if directed {
            NeighborIter::new_reversed(&g, rel_types)
        } else {
            NeighborIter::new(&g, rel_types, false)
        };
        Ok(self.bfs_shortest_path(
            &g,
            &mut fwd_nbrs,
            &mut bwd_nbrs,
            src_id,
            dst_id,
            max_level,
            node_cap,
            rel_types,
            min_hops,
        ))
    }

    /// Bidirectional BFS to find a single shortest path between two nodes.
    ///
    /// Expands the smaller frontier one full level at a time, alternating
    /// between a forward search from `src` (outgoing edges) and a backward
    /// search from `dst` (incoming edges; symmetric neighbours when
    /// undirected). This bounds the explored set by roughly
    /// `O(b^(d/2))` instead of `O(b^d)`, and lets no-path queries terminate
    /// as soon as the smaller side's reachable set is exhausted — the
    /// pathological cases where a unidirectional scalar BFS had to visit the
    /// entire graph.
    ///
    /// Correctness of the level-synchronised stop rule: after completing
    /// levels `(df, db)` with no meeting node, any path must be longer than
    /// `df + db` (a shortest path of length `D <= df + db` would have a node
    /// visited by both sides, detected when the second side discovered it).
    /// Hence after the first level whose expansion produces meeting
    /// candidates, the minimum candidate total equals the true shortest
    /// distance.
    #[allow(clippy::too_many_arguments)]
    fn bfs_shortest_path(
        &self,
        g: &crate::graph::graph::Graph,
        fwd_nbrs: &mut NeighborIter,
        bwd_nbrs: &mut NeighborIter,
        src_id: crate::graph::graph::NodeId,
        dst_id: crate::graph::graph::NodeId,
        max_level: u64,
        _node_cap: u64,
        rel_types: &[Arc<String>],
        min_hops: u32,
    ) -> Value {
        use crate::graph::graph::{NodeId, RelationshipId};

        let src = u64::from(src_id);
        let dst = u64::from(dst_id);

        // src == dst with min_hops > 0 (the zero-hop case is handled by the
        // caller): a cyclic path back to the same node is not matched,
        // mirroring the prior unidirectional behaviour.
        if src == dst {
            return Value::Null;
        }

        // (parent, depth) per visited node, keyed only by visited nodes
        // (SEC-3: bounded by visited count, not node_cap).
        let mut f_map: rustc_hash::FxHashMap<u64, (u64, u64)> = rustc_hash::FxHashMap::default();
        let mut b_map: rustc_hash::FxHashMap<u64, (u64, u64)> = rustc_hash::FxHashMap::default();
        f_map.insert(src, (src, 0));
        b_map.insert(dst, (dst, 0));

        let mut f_front: Vec<u64> = vec![src];
        let mut b_front: Vec<u64> = vec![dst];
        let mut next: Vec<u64> = Vec::new();
        let mut df: u64 = 0; // completed forward depth
        let mut db: u64 = 0; // completed backward depth

        // Best meeting found in the level being expanded:
        // (other side's depth, meeting node).
        let mut meet: Option<(u64, u64)> = None;

        while meet.is_none() {
            if f_front.is_empty() || b_front.is_empty() || df + db >= max_level {
                return Value::Null;
            }
            let expand_fwd = f_front.len() <= b_front.len();
            let (front, own, other, nbrs) = if expand_fwd {
                (&f_front, &mut f_map, &b_map, &mut *fwd_nbrs)
            } else {
                (&b_front, &mut b_map, &f_map, &mut *bwd_nbrs)
            };
            let depth = if expand_fwd { df } else { db } + 1;
            next.clear();
            for &cur in front {
                for &nb in nbrs.neighbors(cur) {
                    // Not using the Entry API: the common case is an
                    // already-visited neighbour, which Entry would allocate
                    // a hash slot probe closure for; contains+insert reads
                    // cleaner with the meet check in between.
                    #[allow(clippy::map_entry)]
                    if !own.contains_key(&nb) {
                        own.insert(nb, (cur, depth));
                        if let Some(&(_, od)) = other.get(&nb) {
                            if meet.is_none_or(|(best_od, _)| od < best_od) {
                                meet = Some((od, nb));
                            }
                        } else {
                            next.push(nb);
                        }
                    }
                }
            }
            std::mem::swap(
                if expand_fwd {
                    &mut f_front
                } else {
                    &mut b_front
                },
                &mut next,
            );
            if expand_fwd {
                df = depth;
            } else {
                db = depth;
            }
        }

        let (_, meet_node) = meet.expect("loop exits only with a meet");

        // Reconstruct: src -> meet via forward parents, then meet -> dst via
        // backward parents (each backward parent is one step closer to dst).
        let mut path_nodes: Vec<u64> = vec![meet_node];
        let mut cur = meet_node;
        while cur != src {
            cur = f_map.get(&cur).expect("BFS parent chain broken").0;
            path_nodes.push(cur);
        }
        path_nodes.reverse();
        cur = meet_node;
        while cur != dst {
            cur = b_map.get(&cur).expect("BFS parent chain broken").0;
            path_nodes.push(cur);
        }

        // Enforce min_hops: path must have at least min_hops edges
        if (path_nodes.len() - 1) < min_hops as usize {
            return Value::Null;
        }

        // Build alternating node/relationship path
        let mut path: ThinVec<Value> = ThinVec::with_capacity(path_nodes.len() * 2 - 1);
        path.push(Value::Node(NodeId::from(path_nodes[0])));
        for i in 0..path_nodes.len() - 1 {
            let from = NodeId::from(path_nodes[i]);
            let to = NodeId::from(path_nodes[i + 1]);
            // Find the relationship between consecutive path nodes
            let rel_id: Option<RelationshipId> = g
                .get_src_dest_relationships(from, to, rel_types)
                .next()
                .or_else(|| g.get_src_dest_relationships(to, from, rel_types).next());
            if let Some(rid) = rel_id {
                path.push(Value::Relationship(rid));
            }
            path.push(Value::Node(to));
        }

        Value::Path(Arc::new(path))
    }

    pub(crate) fn eval_map_projection<R: RowView + ?Sized>(
        &self,
        node: &ExprNode<'_>,
        env: Option<&R>,
        agg_group_key: Option<u64>,
    ) -> Result<Value, String> {
        let rt = self.rt()?;
        let base = self.eval_node(&node.child(0), env, agg_group_key)?;

        if matches!(base, Value::Null) {
            return Ok(Value::Null);
        }

        if !matches!(
            &base,
            Value::Node(_) | Value::Relationship(_) | Value::Map(_)
        ) {
            return Err("Encountered unhandled type evaluating map projection".to_string());
        }

        let mut result = OrderMap::default();

        for i in 1..node.num_children() {
            let item = node.child(i);
            match item.data() {
                ExprIR::MapProjection => match &base {
                    Value::Node(id) => {
                        for (k, v) in rt.get_node_attrs(*id) {
                            result.insert(k, v);
                        }
                    }
                    Value::Relationship(rel) => {
                        for (k, v) in rt.get_relationship_attrs(*rel) {
                            result.insert(k, v);
                        }
                    }
                    Value::Map(map) => {
                        for (k, v) in map.iter() {
                            result.insert(k.clone(), v.clone());
                        }
                    }
                    _ => {
                        return Err(
                            "Encountered unhandled type evaluating map projection".to_string()
                        );
                    }
                },
                ExprIR::Property(prop_name) => {
                    let value = match &base {
                        Value::Node(id) => {
                            rt.get_node_attribute(*id, prop_name).unwrap_or(Value::Null)
                        }
                        Value::Relationship(rel) => rt
                            .get_relationship_attribute(*rel, prop_name)
                            .unwrap_or(Value::Null),
                        Value::Map(map) => map.get(prop_name).cloned().unwrap_or(Value::Null),
                        _ => {
                            return Err(
                                "Encountered unhandled type evaluating map projection".to_string()
                            );
                        }
                    };
                    result.insert(prop_name.clone(), value);
                }
                ExprIR::Constant(Value::String(_)) => {
                    let key = if let ExprIR::Constant(Value::String(k)) = item.data() {
                        k.clone()
                    } else {
                        unreachable!();
                    };
                    let value = self.eval_node(&item.child(0), env, agg_group_key)?;
                    result.insert(key, value);
                }
                _ => {
                    return Err("Encountered unhandled type evaluating map projection".to_string());
                }
            }
        }

        Ok(Value::Map(Arc::new(result)))
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions (no Runtime dependency)
// ---------------------------------------------------------------------------

pub(crate) const fn eval_quantifier(
    quantifier_type: &QuantifierType,
    true_count: usize,
    false_count: usize,
    null_count: usize,
) -> Value {
    match quantifier_type {
        QuantifierType::All => {
            if false_count > 0 {
                Value::Bool(false)
            } else if null_count > 0 {
                Value::Null
            } else {
                Value::Bool(true)
            }
        }
        QuantifierType::Any => {
            if true_count > 0 {
                Value::Bool(true)
            } else if null_count > 0 {
                Value::Null
            } else {
                Value::Bool(false)
            }
        }
        QuantifierType::None => {
            if true_count > 0 {
                Value::Bool(false)
            } else if null_count > 0 {
                Value::Null
            } else {
                Value::Bool(true)
            }
        }
        QuantifierType::Single => {
            if true_count == 1 && null_count == 0 {
                Value::Bool(true)
            } else if true_count > 1 {
                Value::Bool(false)
            } else if null_count > 0 {
                Value::Null
            } else {
                Value::Bool(false)
            }
        }
    }
}

// the semantic of Eq [1, 2, 3] is: 1 EQ 2 AND 2 EQ 3
pub(crate) fn all_equals<I>(mut iter: I) -> Result<Value, String>
where
    I: Iterator<Item = Result<Value, String>>,
{
    if let Some(first) = iter.next() {
        let prev = first?;
        for next in iter {
            let next = next?;
            match prev.compare_value(&next) {
                (_, DisjointOrNull::ComparedNull) => return Ok(Value::Null),
                (_, DisjointOrNull::NaN | DisjointOrNull::Disjoint) => {
                    return Ok(Value::Bool(false));
                }
                (Ordering::Equal, _) => {}
                _ => return Ok(Value::Bool(false)),
            }
        }
        Ok(Value::Bool(true))
    } else {
        Err(String::from("Eq operator requires at least two arguments"))
    }
}

pub(crate) fn all_not_equals<I>(mut iter: I) -> Result<Value, String>
where
    I: Iterator<Item = Result<Value, String>>,
{
    if let Some(first) = iter.next() {
        let prev = first?;
        for next in iter {
            let next = next?;
            match prev.partial_cmp(&next) {
                None => return Ok(Value::Null),
                Some(Ordering::Less | Ordering::Greater) => {}
                Some(Ordering::Equal) => return Ok(Value::Bool(false)),
            }
        }
        Ok(Value::Bool(true))
    } else {
        Err(String::from("Eq operator requires at least two arguments"))
    }
}

pub(crate) fn list_contains(
    list: &Value,
    value: Value,
) -> Result<Value, String> {
    match list {
        Value::List(l) => Ok(Contains::contains(l.as_ref(), value)),
        Value::Null => Ok(Value::Null),
        _ => Err(format!(
            "Type mismatch: expected List or Null but was {}",
            list.name()
        )),
    }
}

pub(crate) fn get_elements(
    arr: &Value,
    start: &Value,
    end: &Value,
) -> Result<Value, String> {
    match (arr, start, end) {
        (Value::List(values), Value::Int(start), Value::Int(end)) => {
            let mut start = *start;
            let mut end = *end;
            if start < 0 {
                start = (values.len() as i64 + start).max(0);
            }
            if end < 0 {
                end = (values.len() as i64 + end).max(0);
            } else {
                end = end.min(values.len() as i64);
            }
            if start > end {
                return Ok(Value::List(Arc::new(thin_vec![])));
            }
            Ok(Value::List(Arc::new(
                values[start as usize..end as usize]
                    .iter()
                    .cloned()
                    .collect::<ThinVec<_>>(),
            )))
        }
        (Value::Null, _, _) | (_, Value::Null, _) | (_, _, Value::Null) => Ok(Value::Null),
        _ => Err(String::from("Invalid array range parameters.")),
    }
}

#[inline]
pub(crate) const fn logical_xor(
    a: bool,
    b: bool,
) -> bool {
    (a && !b) || (!a && b)
}

pub fn evaluate_param(expr: &DynNode<ExprIR<Arc<String>>>) -> Result<Value, String> {
    match expr.data() {
        ExprIR::Constant(v) => Ok(v.clone()),
        ExprIR::List => Ok(Value::List(Arc::new(
            expr.children()
                .map(|c| evaluate_param(&c))
                .collect::<Result<ThinVec<_>, _>>()?,
        ))),
        ExprIR::Map => Ok(Value::Map(Arc::new(
            expr.children()
                .map(|ir| match ir.data() {
                    ExprIR::Constant(Value::String(key)) => {
                        Ok::<_, String>((key.clone(), evaluate_param(&ir.child(0))?))
                    }
                    _ => Err("Map parameter key must be a string".into()),
                })
                .collect::<Result<OrderMap<_, _>, _>>()?,
        ))),
        ExprIR::Negate => {
            let v = evaluate_param(&expr.child(0))?;
            match v {
                Value::Int(i) => Ok(Value::Int(i.checked_neg().ok_or_else(|| {
                    String::from("ArgumentError: integer overflow in unary minus")
                })?)),
                Value::Float(f) => Ok(Value::Float(-f)),
                _ => Ok(Value::Null),
            }
        }
        _ => Err(String::from("Invalid parameter expression.")),
    }
}
