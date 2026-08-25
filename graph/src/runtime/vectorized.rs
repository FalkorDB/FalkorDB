//! Vectorized operations on typed columns.
//!
//! This module provides bulk comparison kernels that operate on entire columns
//! of homogeneous values at once, enabling LLVM auto-vectorization for filter
//! predicates like `n.age > 30`.
//!
//! ```text
//!  Scalar (per-row) filter          Vectorized filter
//!  =========================        ==================
//!
//!  for each row:                    1. Materialize property column
//!    eval(n.age > 30)                  ages = [25, 42, 18, 55, ...]
//!    if true -> keep row            2. compare_i64_column(ages, Gt, 30)
//!                                      mask = [F, T, F, T, ...]
//!  O(rows * expr_depth)            3. keep the rows the mask kept
//!                                      sel  = [1, 3, ...]
//!                                   O(rows) with SIMD lanes
//! ```
//!
//! ## Components
//!
//! - [`CmpOp`] -- comparison operator enum (Eq, Neq, Lt, Le, Gt, Ge)
//! - Comparison kernels: [`compare_i64_column`] and [`compare_f64_column`] --
//!   tight indexed loops for auto-vectorization over primitive columns, plus
//!   [`compare_value_column`] -- the general lane for every other column shape
//! - [`SimplePredicate`] / [`VectorizablePredicate`] -- detected filter patterns
//!   that can use the bulk path instead of per-row expression evaluation
//! - [`try_extract_vectorizable_predicate`] -- analyzes a filter expression tree
//!   to detect `entity.property <cmp> constant` patterns
//!
//! ## Agreement with the scalar evaluator
//!
//! Every kernel answers exactly what the scalar evaluator would answer for the
//! same row. The primitive lanes are entered only for column/constant pairs
//! whose primitive comparison *is* the `Value` comparison (`Int` against `Int`,
//! and the `f64` promotion `Value::compare_value` itself performs for mixed
//! numerics); every other shape goes through [`compare_value_column`], which
//! calls the same `compare_value` / `partial_cmp` the per-row path uses. A
//! filter and a projection of the same predicate therefore cannot disagree —
//! the divergence issue #2582 reported, where `WHERE p <> 'x'` dropped rows
//! that `RETURN p <> 'x'` called `true`.
//!
//! The primitive kernels are written as tight indexed loops to enable
//! LLVM auto-vectorization on all target platforms (x86_64 SSE/AVX, ARM NEON).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use crate::parser::ast::{ExprIR, Variable};
use crate::runtime::batch::NullBitmap;
use crate::runtime::value::{CompareValue, DisjointOrNull, Value};

use orx_tree::{Dyn, DynTree, NodeIdx, NodeRef};

// ---------------------------------------------------------------------------
// CmpOp — comparison operator enum
// ---------------------------------------------------------------------------

/// Comparison operator for vectorized kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    /// Converts from an `ExprIR` comparison node to a `CmpOp`.
    pub const fn from_expr_ir<T>(ir: &ExprIR<T>) -> Option<Self> {
        match ir {
            ExprIR::Eq => Some(Self::Eq),
            ExprIR::Neq => Some(Self::Neq),
            ExprIR::Lt => Some(Self::Lt),
            ExprIR::Le => Some(Self::Le),
            ExprIR::Gt => Some(Self::Gt),
            ExprIR::Ge => Some(Self::Ge),
            _ => None,
        }
    }

    /// Returns the flipped operator (for when operands are swapped).
    #[must_use]
    pub const fn flip(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Neq => Self::Neq,
            Self::Lt => Self::Gt,
            Self::Le => Self::Ge,
            Self::Gt => Self::Lt,
            Self::Ge => Self::Le,
        }
    }
}

// ---------------------------------------------------------------------------
// Comparison kernels — tight loops for auto-vectorization
// ---------------------------------------------------------------------------

/// Compares each element of `data` against `threshold` using `op`.
/// Null rows (per `nulls` bitmap) always produce `false`.
#[allow(clippy::needless_range_loop)]
#[inline]
#[must_use]
pub fn compare_i64_column(
    data: &[i64],
    op: CmpOp,
    threshold: i64,
    nulls: &NullBitmap,
) -> Vec<bool> {
    let len = data.len();
    let mut result = vec![false; len];
    match op {
        CmpOp::Eq => {
            for i in 0..len {
                result[i] = data[i] == threshold;
            }
        }
        CmpOp::Neq => {
            for i in 0..len {
                result[i] = data[i] != threshold;
            }
        }
        CmpOp::Lt => {
            for i in 0..len {
                result[i] = data[i] < threshold;
            }
        }
        CmpOp::Le => {
            for i in 0..len {
                result[i] = data[i] <= threshold;
            }
        }
        CmpOp::Gt => {
            for i in 0..len {
                result[i] = data[i] > threshold;
            }
        }
        CmpOp::Ge => {
            for i in 0..len {
                result[i] = data[i] >= threshold;
            }
        }
    }
    // Mask out nulls in a separate pass to avoid polluting the inner loop
    if nulls.any_null() {
        for i in 0..len {
            if nulls.is_null(i) {
                result[i] = false;
            }
        }
    }
    result
}

/// Compares each element of `data` against `threshold` using `op`.
/// NaN comparisons naturally return false, matching Cypher semantics.
/// Null rows (per `nulls` bitmap) always produce `false`.
#[allow(clippy::needless_range_loop)]
#[inline]
#[must_use]
pub fn compare_f64_column(
    data: &[f64],
    op: CmpOp,
    threshold: f64,
    nulls: &NullBitmap,
) -> Vec<bool> {
    let len = data.len();
    let mut result = vec![false; len];
    match op {
        CmpOp::Eq => {
            for i in 0..len {
                result[i] = data[i] == threshold;
            }
        }
        CmpOp::Neq => {
            for i in 0..len {
                result[i] = data[i] != threshold;
            }
        }
        CmpOp::Lt => {
            for i in 0..len {
                result[i] = data[i] < threshold;
            }
        }
        CmpOp::Le => {
            for i in 0..len {
                result[i] = data[i] <= threshold;
            }
        }
        CmpOp::Gt => {
            for i in 0..len {
                result[i] = data[i] > threshold;
            }
        }
        CmpOp::Ge => {
            for i in 0..len {
                result[i] = data[i] >= threshold;
            }
        }
    }
    if nulls.any_null() {
        for i in 0..len {
            if nulls.is_null(i) {
                result[i] = false;
            }
        }
    }
    result
}

/// Compares a heterogeneous `Value` column against `constant`, row by row, with
/// exactly the semantics the scalar evaluator applies to the same comparison.
///
/// This is the general lane: it backs every column the typed kernels above do
/// not cover (strings, points, lists, maps, booleans, mixed int/float). A row
/// passes only when the comparison yields `true` — a `null` result drops the
/// row, matching how [`FilterOp`](crate::runtime::ops::filter::FilterOp) treats
/// `Value::Null` on the per-row path.
///
/// Mirrors `eval.rs`:
/// - `=` is `all_equals`: disjoint types, `NaN` and `null` are all *not* equal.
/// - `<>` is `all_not_equals`, i.e. `Value::partial_cmp`, which is `None` only
///   when a `null` is involved. Two values of disjoint types are therefore
///   **unequal**, not `false` — `point(...) <> 'Dhaka'` is `true`.
/// - `<`, `<=`, `>`, `>=` are `null` (so: drop) for disjoint types and `null`,
///   and `false` for `NaN`.
#[must_use]
pub fn compare_value_column(
    data: &[Value],
    op: CmpOp,
    constant: &Value,
) -> Vec<bool> {
    data.iter()
        .map(|v| {
            let (ord, flag) = v.compare_value(constant);
            match op {
                // Equality is total across types: only an exact match passes,
                // and a null operand makes the whole comparison null.
                CmpOp::Eq => flag == DisjointOrNull::None && ord == Ordering::Equal,
                // `partial_cmp` is `None` only for a null operand; every other
                // pair — including disjoint types — is ordered, hence unequal.
                CmpOp::Neq => flag != DisjointOrNull::ComparedNull && ord != Ordering::Equal,
                // Ordering across disjoint types (or with a null, or a NaN) is
                // not a truth value, so the row drops.
                _ => {
                    flag == DisjointOrNull::None
                        && match op {
                            CmpOp::Lt => ord == Ordering::Less,
                            CmpOp::Le => ord != Ordering::Greater,
                            CmpOp::Gt => ord == Ordering::Greater,
                            _ => ord != Ordering::Less,
                        }
                }
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Simple predicate detection
// ---------------------------------------------------------------------------

/// A simple predicate that can be evaluated in vectorized mode.
/// Represents: `entity_variable.property <op> constant`
#[derive(Debug)]
pub struct SimplePredicate {
    /// The variable whose property is being compared (e.g., `n` in `n.age > 30`).
    pub var: Variable,
    /// The property name (e.g., "age").
    pub attr: Arc<String>,
    /// The comparison operator.
    pub op: CmpOp,
    /// The constant value on the other side.
    pub constant: Value,
}

/// A vectorizable predicate — either a single comparison or a conjunction.
#[derive(Debug)]
pub enum VectorizablePredicate {
    Single(SimplePredicate),
    Conjunction(Vec<SimplePredicate>),
}

/// Tries to extract a vectorizable predicate from a filter expression tree.
///
/// Detects patterns like:
/// - `n.age > 30` → `Single(SimplePredicate { var: n, attr: "age", op: Gt, constant: Int(30) })`
/// - `n.age > 30 AND n.name = 'Alice'` → `Conjunction([...])`
///
/// Returns `None` for complex predicates that cannot be vectorized.
#[allow(clippy::implicit_hasher)]
pub fn try_extract_vectorizable_predicate(
    tree: &DynTree<ExprIR<Variable>>,
    params: &HashMap<String, Value>,
) -> Option<VectorizablePredicate> {
    let root = tree.root();
    let root_data = root.data();

    // Check for AND (conjunction of simple predicates)
    if matches!(root_data, ExprIR::And) {
        let mut preds = Vec::new();
        for child in root.children() {
            let child_tree = child.clone_as_tree();
            preds.push(try_extract_single_predicate(&child_tree, params)?);
        }
        if preds.is_empty() {
            return None;
        }
        return Some(VectorizablePredicate::Conjunction(preds));
    }

    // Single predicate
    try_extract_single_predicate(tree, params).map(VectorizablePredicate::Single)
}

/// Tries to extract a single `SimplePredicate` from a comparison expression.
fn try_extract_single_predicate(
    tree: &DynTree<ExprIR<Variable>>,
    params: &HashMap<String, Value>,
) -> Option<SimplePredicate> {
    let root = tree.root();
    let op = CmpOp::from_expr_ir(root.data())?;

    if root.num_children() != 2 {
        return None;
    }

    let lhs_idx = root.child(0).idx();
    let rhs_idx = root.child(1).idx();

    // Try: Property(attr) -> Variable(var)  <op>  Constant
    if let Some(pred) = try_property_vs_constant(tree, lhs_idx, rhs_idx, op, params) {
        return Some(pred);
    }
    // Try: Constant  <op>  Property(attr) -> Variable(var) (flip operator)
    try_property_vs_constant(tree, rhs_idx, lhs_idx, op.flip(), params)
}

/// Checks if `prop_side` is `Property(attr) -> Variable(var)` and
/// `const_side` is a literal constant or a resolvable query parameter.
fn try_property_vs_constant(
    tree: &DynTree<ExprIR<Variable>>,
    prop_idx: NodeIdx<Dyn<ExprIR<Variable>>>,
    const_idx: NodeIdx<Dyn<ExprIR<Variable>>>,
    op: CmpOp,
    params: &HashMap<String, Value>,
) -> Option<SimplePredicate> {
    let prop_node = tree.node(prop_idx);
    let ExprIR::Property(attr) = prop_node.data() else {
        return None;
    };
    if prop_node.num_children() != 1 {
        return None;
    }
    let ExprIR::Variable(var) = prop_node.child(0).data() else {
        return None;
    };

    // const_side must be a leaf literal or a query parameter that resolves to
    // a literal value (parameters are not substituted into the cached plan, so
    // `MATCH (n {id: $id})` keeps `$id` as an `ExprIR::Parameter` node).
    let const_node = tree.node(const_idx);
    if const_node.num_children() != 0 {
        return None;
    }
    let constant = match const_node.data() {
        ExprIR::Constant(v) => v.clone(),
        ExprIR::Parameter(name) => params.get(name)?.clone(),
        _ => return None,
    };

    Some(SimplePredicate {
        var: var.clone(),
        attr: attr.clone(),
        op,
        constant,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::runtime::eval::{ExprEval, NO_ROW};
    use crate::runtime::value::Point;
    use orx_tree::NodeMut;

    #[test]
    fn test_cmp_op_flip() {
        assert_eq!(CmpOp::Eq.flip(), CmpOp::Eq);
        assert_eq!(CmpOp::Neq.flip(), CmpOp::Neq);
        assert_eq!(CmpOp::Lt.flip(), CmpOp::Gt);
        assert_eq!(CmpOp::Le.flip(), CmpOp::Ge);
        assert_eq!(CmpOp::Gt.flip(), CmpOp::Lt);
        assert_eq!(CmpOp::Ge.flip(), CmpOp::Le);
    }

    #[test]
    fn test_compare_i64_basic() {
        let data = vec![10, 20, 30, 40, 50];
        let nulls = NullBitmap::none(5);
        assert_eq!(
            compare_i64_column(&data, CmpOp::Gt, 25, &nulls),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            compare_i64_column(&data, CmpOp::Eq, 30, &nulls),
            vec![false, false, true, false, false]
        );
        assert_eq!(
            compare_i64_column(&data, CmpOp::Le, 30, &nulls),
            vec![true, true, true, false, false]
        );
    }

    #[test]
    fn test_compare_i64_with_nulls() {
        let data = vec![10, 0, 30, 0, 50]; // indices 1 and 3 are null
        let nulls = NullBitmap::from_values(&[
            Value::Int(10),
            Value::Null,
            Value::Int(30),
            Value::Null,
            Value::Int(50),
        ]);
        let result = compare_i64_column(&data, CmpOp::Gt, 5, &nulls);
        assert_eq!(result, vec![true, false, true, false, true]);
    }

    #[test]
    fn test_compare_f64_basic() {
        let data = vec![1.5, 2.5, 3.5];
        let nulls = NullBitmap::none(3);
        assert_eq!(
            compare_f64_column(&data, CmpOp::Lt, 3.0, &nulls),
            vec![true, true, false]
        );
    }

    #[test]
    fn test_compare_f64_nan() {
        let data = vec![1.0, f64::NAN, 3.0];
        let nulls = NullBitmap::none(3);
        // NaN comparisons return false for all operators
        let result = compare_f64_column(&data, CmpOp::Gt, 0.0, &nulls);
        assert_eq!(result, vec![true, false, true]);
    }

    #[test]
    fn test_compare_empty() {
        let data: Vec<i64> = vec![];
        let nulls = NullBitmap::none(0);
        assert_eq!(
            compare_i64_column(&data, CmpOp::Eq, 0, &nulls),
            Vec::<bool>::new()
        );
    }

    /// Every kernel answer must equal what the scalar evaluator produces for
    /// the same pair, so a filter and a projection can never disagree. `null`
    /// is the one asymmetry: the evaluator yields `Value::Null`, which the
    /// filter drops — i.e. `false` in the mask.
    fn scalar_says(
        value: &Value,
        op: CmpOp,
        constant: &Value,
    ) -> bool {
        let tree: DynTree<ExprIR<Variable>> = DynTree::new(match op {
            CmpOp::Eq => ExprIR::Eq,
            CmpOp::Neq => ExprIR::Neq,
            CmpOp::Lt => ExprIR::Lt,
            CmpOp::Le => ExprIR::Le,
            CmpOp::Gt => ExprIR::Gt,
            CmpOp::Ge => ExprIR::Ge,
        });
        let mut tree = tree;
        let mut root = tree.root_mut();
        root.push_child(ExprIR::Constant(value.clone()));
        root.push_child(ExprIR::Constant(constant.clone()));
        let root = tree.root();
        matches!(
            ExprEval::constant().eval(&tree, root.idx(), NO_ROW, None),
            Ok(Value::Bool(true))
        )
    }

    fn sample_values() -> Vec<Value> {
        vec![
            Value::Null,
            Value::Bool(true),
            Value::Int(7),
            Value::Float(7.0),
            Value::Float(f64::NAN),
            Value::String(Arc::new("Dhaka".to_string())),
            Value::String(Arc::new("Alice".to_string())),
            Value::List(Arc::new(vec![Value::Int(1), Value::Int(2)].into())),
            Value::Point(Point {
                latitude: 31.18,
                longitude: 34.22,
            }),
        ]
    }

    #[test]
    fn test_compare_value_column_matches_scalar_eval() {
        let ops = [
            CmpOp::Eq,
            CmpOp::Neq,
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Gt,
            CmpOp::Ge,
        ];
        let data = sample_values();
        for constant in sample_values() {
            for op in ops {
                let kernel = compare_value_column(&data, op, &constant);
                let scalar: Vec<bool> =
                    data.iter().map(|v| scalar_says(v, op, &constant)).collect();
                assert_eq!(
                    kernel, scalar,
                    "kernel and scalar disagree for {op:?} against {constant:?}"
                );
            }
        }
    }

    #[test]
    fn test_compare_value_column_neq_keeps_disjoint_types() {
        // Issue #2582: every non-string is `<> 'Dhaka'`, not `false`.
        let data = sample_values();
        let dhaka = Value::String(Arc::new("Dhaka".to_string()));
        assert_eq!(
            compare_value_column(&data, CmpOp::Neq, &dhaka),
            vec![false, true, true, true, true, false, true, true, true]
        );
        // `=` is unaffected: only the matching string passes.
        assert_eq!(
            compare_value_column(&data, CmpOp::Eq, &dhaka),
            vec![false, false, false, false, false, true, false, false, false]
        );
        // Ordering against a disjoint type stays `null`, so no row survives.
        assert!(
            compare_value_column(&data, CmpOp::Lt, &dhaka)
                .iter()
                .zip(&data)
                .all(|(&pass, v)| pass == matches!(v, Value::String(s) if s.as_str() < "Dhaka"))
        );
    }
}
