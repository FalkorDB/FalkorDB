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
//!
//! Callers do not choose between these and the per-row evaluator: they are the
//! leaves of [`vector_expr`](super::vector_expr), which evaluates whole
//! expression trees columnarly and reaches for a kernel whenever a comparison's
//! operands happen to be a typed column and a constant.
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

use crate::parser::ast::ExprIR;
use crate::runtime::batch::NullBitmap;
use crate::runtime::value::{CompareValue, DisjointOrNull, Value};

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
    compare_value_column_3vl(data, op, constant).0
}

#[must_use]
/// One comparison with the per-row evaluator's semantics: `None` is `null`.
pub fn compare_values(
    lhs: &Value,
    rhs: &Value,
    op: CmpOp,
) -> Option<bool> {
    let (ord, flag) = lhs.compare_value(rhs);
    match op {
        // `=` is `all_equals`: null makes it null, disjoint types and NaN are
        // simply not equal.
        CmpOp::Eq => match flag {
            DisjointOrNull::ComparedNull => None,
            DisjointOrNull::NaN | DisjointOrNull::Disjoint => Some(false),
            DisjointOrNull::None => Some(ord == Ordering::Equal),
        },
        // `<>` is `all_not_equals`, i.e. `partial_cmp`, which is `None` only
        // for a null operand — disjoint types are ordered, hence unequal.
        CmpOp::Neq => match flag {
            DisjointOrNull::ComparedNull => None,
            _ => Some(ord != Ordering::Equal),
        },
        // Ordering across disjoint types or with a null is null; NaN is false.
        _ => match flag {
            DisjointOrNull::ComparedNull | DisjointOrNull::Disjoint => None,
            DisjointOrNull::NaN => Some(false),
            DisjointOrNull::None => Some(match op {
                CmpOp::Lt => ord == Ordering::Less,
                CmpOp::Le => ord != Ordering::Greater,
                CmpOp::Gt => ord == Ordering::Greater,
                _ => ord != Ordering::Less,
            }),
        },
    }
}

/// [`compare_value_column`] keeping `null` distinct from `false`.
///
/// A filter drops both, so it uses the mask alone; a predicate nested under
/// `NOT` / `AND` / `OR` needs the difference, since `NOT null` is `null` while
/// `NOT false` is `true`.
#[must_use]
pub fn compare_value_column_3vl(
    data: &[Value],
    op: CmpOp,
    constant: &Value,
) -> (Vec<bool>, NullBitmap) {
    let mut mask = vec![false; data.len()];
    let mut nulls = NullBitmap::none(data.len());
    for (i, v) in data.iter().enumerate() {
        match compare_values(v, constant, op) {
            Some(pass) => mask[i] = pass,
            None => nulls.set(i),
        }
    }
    (mask, nulls)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::parser::ast::Variable;
    use orx_tree::{DynTree, NodeRef};

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
