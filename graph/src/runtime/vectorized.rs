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
//!  O(rows * expr_depth)            3. mask_to_selection(mask)
//!                                      sel  = [1, 3, ...]
//!                                   O(rows) with SIMD lanes
//! ```
//!
//! ## Components
//!
//! - [`CmpOp`] -- comparison operator enum (Eq, Neq, Lt, Le, Gt, Ge)
//! - Comparison kernels: [`compare_i64_column`], [`compare_f64_column`],
//!   [`compare_string_column`] -- tight indexed loops for auto-vectorization
//! - [`SimplePredicate`] / [`VectorizablePredicate`] -- detected filter patterns
//!   that can use the bulk path instead of per-row expression evaluation
//! - [`try_extract_vectorizable_predicate`] -- analyzes a filter expression tree
//!   to detect `entity.property <cmp> constant` patterns
//! - [`mask_to_selection`] / [`mask_intersect_selection`] -- convert boolean
//!   masks to/from the selection vector used by [`Batch`](super::batch::Batch)
//!
//! The comparison kernels are written as tight indexed loops to enable
//! LLVM auto-vectorization on all target platforms (x86_64 SSE/AVX, ARM NEON).

use std::collections::HashMap;
use std::sync::Arc;

use crate::parser::ast::{ExprIR, Variable};
use crate::runtime::batch::NullBitmap;
use crate::runtime::value::Value;

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

/// Compares string values in a `Value` slice against `threshold`.
/// Non-string and Null values produce `false`.
#[must_use]
pub fn compare_string_column(
    data: &[Value],
    op: CmpOp,
    threshold: &str,
) -> Vec<bool> {
    let len = data.len();
    let mut result = vec![false; len];
    for i in 0..len {
        if let Value::String(s) = &data[i] {
            result[i] = match op {
                CmpOp::Eq => s.as_str() == threshold,
                CmpOp::Neq => s.as_str() != threshold,
                CmpOp::Lt => s.as_str() < threshold,
                CmpOp::Le => s.as_str() <= threshold,
                CmpOp::Gt => s.as_str() > threshold,
                CmpOp::Ge => s.as_str() >= threshold,
            };
        }
    }
    result
}

/// Converts a boolean mask to a selection vector of passing row indices.
#[must_use]
pub fn mask_to_selection(mask: &[bool]) -> Vec<u16> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, &pass)| if pass { Some(i as u16) } else { None })
        .collect()
}

/// Intersects a boolean mask with an existing selection vector.
/// Only rows present in both the mask AND the existing selection pass.
#[must_use]
pub fn mask_intersect_selection(
    mask: &[bool],
    existing: &[u16],
) -> Vec<u16> {
    existing
        .iter()
        .copied()
        .filter(|&i| mask[i as usize])
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
    fn test_mask_to_selection() {
        let mask = vec![true, false, true, false, true];
        assert_eq!(mask_to_selection(&mask), vec![0, 2, 4]);
    }

    #[test]
    fn test_mask_intersect_selection() {
        let mask = vec![true, false, true, true, false];
        let existing = vec![0, 2, 3, 4];
        assert_eq!(mask_intersect_selection(&mask, &existing), vec![0, 2, 3]);
    }

    #[test]
    fn test_compare_string_column() {
        let data = vec![
            Value::String(Arc::new("Alice".to_string())),
            Value::String(Arc::new("Bob".to_string())),
            Value::Null,
            Value::String(Arc::new("Alice".to_string())),
        ];
        let result = compare_string_column(&data, CmpOp::Eq, "Alice");
        assert_eq!(result, vec![true, false, false, true]);
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
}
