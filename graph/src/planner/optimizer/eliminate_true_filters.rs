//! Constant-true filter elimination pass.
//!
//! Removes `Filter` nodes whose predicate evaluates to a constant `true` at
//! plan time. These trivial filters are introduced by the planner when
//! pattern predicates are extracted from WHERE clauses (the planner replaces
//! extracted patterns with `Bool(true)` as an AND identity element).
//!
//! ## Transformations
//!
//! **Entire filter is constant true:**
//!
//! ```text
//! Before:           After:
//!
//! Filter(true)      ChildA
//!   |
//!   v
//! ChildA
//! ```
//!
//! **AND filter with some constant-true conjuncts:**
//!
//! ```text
//! Before:                          After:
//!
//! Filter(AND(true, n.age > 18))    Filter(n.age > 18)
//!   |                                |
//!   v                                v
//! ChildA                           ChildA
//! ```
//!
//! If all AND conjuncts are constant true, the entire Filter is removed.

use std::{collections::HashMap, sync::Arc};

use orx_tree::{Bfs, DynTree, NodeRef};

use crate::{
    parser::ast::{ExprIR, Variable},
    runtime::{eval::ExprEval, eval::NO_ROW, value::Value},
    tree,
};

use super::super::IR;

/// Returns true if the filter expression evaluates to a constant `true`.
/// First tries constant-only evaluation. If that fails (e.g. parameter
/// references), substitutes parameter values and retries.
fn is_constant_true(
    filter: &DynTree<ExprIR<Variable>>,
    params: &HashMap<String, Value>,
) -> bool {
    // Fast path: pure constant (no params needed).
    if matches!(
        ExprEval::constant().eval(filter, filter.root().idx(), NO_ROW, None),
        Ok(Value::Bool(true))
    ) {
        return true;
    }

    // Slow path: substitute parameters into a cloned expression, then
    // re-evaluate as constant.
    if !params.is_empty() && expr_has_parameter(filter) {
        let substituted = substitute_params(filter, params);
        return matches!(
            ExprEval::constant().eval(&substituted, substituted.root().idx(), NO_ROW, None),
            Ok(Value::Bool(true))
        );
    }

    false
}

/// Check if the expression tree contains any parameter references.
fn expr_has_parameter(expr: &DynTree<ExprIR<Variable>>) -> bool {
    expr.root()
        .indices::<Bfs>()
        .any(|idx| matches!(expr.node(idx).data(), ExprIR::Parameter(_)))
}

/// Clone the expression tree, replacing `ExprIR::Parameter(name)` nodes
/// with their evaluated constant values from `params`.
fn substitute_params(
    expr: &DynTree<ExprIR<Variable>>,
    params: &HashMap<String, Value>,
) -> DynTree<ExprIR<Variable>> {
    let mut result = expr.clone();
    let indices: Vec<_> = result.root().indices::<Bfs>().collect();
    for idx in indices {
        if let ExprIR::Parameter(name) = result.node(idx).data()
            && let Some(val) = params.get(name)
        {
            *result.node_mut(idx).data_mut() = ExprIR::Constant(val.clone());
        }
    }
    result
}

/// Eliminates filter nodes whose expression evaluates to constant `true`.
///
/// Also removes `Bool(true)` conjuncts from AND filters, and removes the
/// entire filter if all conjuncts are true.
pub(super) fn eliminate_true_filters(
    optimized_plan: &mut DynTree<IR>,
    params: &HashMap<String, Value>,
) {
    loop {
        let mut changed = false;
        let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();

        for idx in indices {
            let IR::Filter(filter) = optimized_plan.node(idx).data() else {
                continue;
            };

            if is_constant_true(filter, params) {
                optimized_plan.node_mut(idx).take_out();
                changed = true;
                break;
            }

            // For AND filters, remove constant-true conjuncts.
            if matches!(filter.root().data(), ExprIR::And) {
                let remaining: Vec<DynTree<ExprIR<Variable>>> = filter
                    .root()
                    .children()
                    .filter(|c| !is_constant_true(&c.clone_as_tree(), params))
                    .map(|c| c.clone_as_tree())
                    .collect();

                if remaining.len() < filter.root().num_children() {
                    if remaining.is_empty() {
                        optimized_plan.node_mut(idx).take_out();
                    } else if remaining.len() == 1 {
                        *optimized_plan.node_mut(idx).data_mut() =
                            IR::Filter(Arc::new(remaining.into_iter().next().unwrap()));
                    } else {
                        *optimized_plan.node_mut(idx).data_mut() =
                            IR::Filter(Arc::new(tree!(ExprIR::And; remaining)));
                    }
                    changed = true;
                    break;
                }
            }
        }

        if !changed {
            break;
        }
    }
}
