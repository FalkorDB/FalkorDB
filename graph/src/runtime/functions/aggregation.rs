//! Aggregation functions.
//!
//! These functions accumulate values across multiple rows.  The runtime
//! drives them through a two-phase protocol:
//!
//! ```text
//!  Phase 1 -- accumulate          Phase 2 -- finalize
//!  ┌──────────────────────┐       ┌────────────────────┐
//!  │ for each row:        │       │ finalize(acc) ->    │
//!  │   acc = fn(val, acc) │──────>│   final Value       │
//!  └──────────────────────┘       └────────────────────┘
//! ```
//!
//! Each aggregation is registered with an initial accumulator value
//! (`FnType::Aggregation { initial, finalizer }`).  The runtime calls the
//! function once per row with `(current_value, accumulator)` and
//! replaces the accumulator with the return value.  After all rows,
//! the optional finalizer transforms the accumulator into the result.
//!
//! | Cypher           | Accumulator fn   | Finalizer               |
//! |------------------|------------------|-------------------------|
//! | `collect(x)`     | `collect()`      | --                      |
//! | `count(x)`       | `count()`        | --                      |
//! | `sum(x)`         | `sum()`          | --                      |
//! | `max(x)`         | `max()`          | --                      |
//! | `min(x)`         | `min()`          | --                      |
//! | `avg(x)`         | `avg()`          | `finalize_avg()`        |
//! | `percentileDisc` | `percentile()`   | `finalize_percentile_disc()` |
//! | `percentileCont` | `percentile()`   | `finalize_percentile_cont()` |
//! | `stDev(x)`       | `stdev()`        | `finalize_stdev()`      |
//! | `stDevP(x)`      | `stdev()`        | `finalize_stdevp()`     |
//!
//! `avg` uses an overflow-safe incremental algorithm: when the running
//! sum approaches `f64::MAX`, it switches to incremental averaging to
//! avoid infinite values.

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]

use super::{FnType, Functions, Type};
use crate::runtime::{
    runtime::Runtime,
    value::{CompareValue, DisjointOrNull, Value},
};
use std::{cmp::Ordering, sync::Arc};
use thin_vec::thin_vec;

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "collect",
        args: [Type::Any],
        ret: Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
        agg_init: Value::List(Arc::new(thin_vec![])),
        batch_agg: collect_batch,
        fn collect(_, args) {
            let mut iter = args.iter().cloned();
            match (iter.next(), iter.next()) {
                (Some(a), Some(Value::Null)) => Ok(Value::List(Arc::new(thin_vec![a]))),
                (Some(a), Some(Value::List(mut l))) => {
                    if a == Value::Null {
                        return Ok(Value::List(l));
                    }
                    Arc::make_mut(&mut l).push(a);
                    Ok(Value::List(l))
                }

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "count",
        args: [Type::Any],
        ret: Type::Int,
        agg_init: Value::Int(0),
        batch_agg: count_batch,
        fn count(_, args) {
            let mut iter = args.iter().cloned();
            let first = iter.next();
            let sec = iter.next();
            match (first, sec) {
                (Some(Value::Null), Some(sec)) => Ok(sec),
                (Some(_), Some(Value::Int(a))) | (Some(Value::Int(a)), None) => Ok(Value::Int(a + 1)),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "sum",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        agg_init: Value::Float(0.0),
        batch_agg: sum_batch,
        fn sum(_, args) {
            let mut iter = args.iter().cloned();
            let first = iter.next();
            let second = iter.next();

            match (first, second) {
                // Skip null values - return accumulator unchanged
                (Some(Value::Null), Some(acc)) => Ok(acc),

                // sum() always returns Float per Cypher spec
                (Some(Value::Int(a)), Some(Value::Float(b))) => Ok(Value::Float(a as f64 + b)),
                (Some(Value::Float(a)), Some(Value::Int(b))) => Ok(Value::Float(a + b as f64)),
                (Some(Value::Float(a)), Some(Value::Float(b))) => Ok(Value::Float(a + b)),

                _ => unreachable!("sum expects Integer, Float, or Null (validation done before call)"),
            }
        }
    );

    cypher_fn!(funcs, "max",
        args: [Type::Any],
        ret: Type::Any,
        agg_init: Value::Null,
        batch_agg: max_batch,
        fn max(_, args) {
            let mut iter = args.iter().cloned();
            match (iter.next(), iter.next()) {
                (Some(a), Some(b)) => {
                    if let (ord, cmp) = b.compare_value(&a) &&
                    (ord == Ordering::Less || cmp == DisjointOrNull::ComparedNull) {
                        return Ok(a);
                    }
                    Ok(b)
                }

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "min",
        args: [Type::Any],
        ret: Type::Any,
        agg_init: Value::Null,
        batch_agg: min_batch,
        fn min(_, args) {
            let mut iter = args.iter().cloned();
            match (iter.next(), iter.next()) {
                (Some(a), Some(b)) => {
                    if let (ord, cmp) = b.compare_value(&a) &&
                    (ord == Ordering::Greater || cmp == DisjointOrNull::ComparedNull) {
                        return Ok(a);
                    }
                    Ok(b)
                }

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "avg",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        agg_init: Value::List(Arc::new(thin_vec![Value::Float(0.0), Value::Int(0), Value::Bool(false)])),
        finalizer: finalize_avg,
        fn avg(_, args) {
            let mut iter = args.iter().cloned();
            let val = iter.next().unwrap();
            let ctx = iter.next().unwrap();
            match (val, ctx) {
                // distinct may pass null as a way to skip the value
                (Value::Null, ctx) => {
                    // If the first value is null, return the accumulator unchanged
                    Ok(ctx)
                }
                (val @ (Value::Int(_) | Value::Float(_)), Value::List(mut vec)) => {
                    let val = val.get_numeric();

                    // Use split_at_mut to get mutable references to all three elements safely
                    // vec = [sum, count, had_overflow]
                    let vec_mut = Arc::make_mut(&mut vec);
                    let (first, rest) = vec_mut.split_at_mut(1);
                    let (second, third) = rest.split_at_mut(1);

                    let (Value::Float(sum), Value::Int(count), Value::Bool(had_overflow)) =
                        (&mut first[0], &mut second[0], &mut third[0])
                    else {
                        unreachable!("avg accumulator should be [sum, count, overflow]");
                    };

                    *count += 1;

                    // Check for overflow condition
                    if *had_overflow || about_to_overflow(*sum, val) {
                        // Use incremental averaging algorithm
                        // Divide the total by the new count (in-place mutation like C)
                        *sum /= *count as f64;

                        // If we were already in overflow mode, multiply back by previous count
                        if *had_overflow {
                            *sum *= (*count - 1) as f64;
                        }

                        // Add the new value contribution
                        *sum += val / *count as f64;

                        // Mark that we're now in overflow mode
                        *had_overflow = true;
                    } else {
                        // Normal accumulation - sum stores total
                        *sum += val;
                    }

                    Ok(Value::List(vec))
                }
                _ => unreachable!("avg expects Integer, Float, or Null (validation done before call)"),
            }
        }
    );

    cypher_fn!(funcs, "percentileDisc",
        args: [
            Type::Union(vec![Type::Int, Type::Float, Type::Null]),
            Type::Union(vec![Type::Int, Type::Float]),
        ],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        agg_init: Value::List(Arc::new(thin_vec![Value::Float(0.0), Value::List(Arc::new(thin_vec![]))])),
        finalizer: finalize_percentile_disc,
        fn percentile(_, args) {
            let val = args[0].clone();
            let percentile_val = args[1].clone();

            // Domain validation is now done in PHASE 3.5, so these checks are removed
            // (Or kept as defensive programming - they should never fail)

            let percentile = percentile_val.get_numeric();

            let ctx = args[2].clone();
            if matches!(val, Value::Null) {
                return Ok(ctx);
            }

            let Value::List(mut state) = ctx else {
                unreachable!("Context must be a List");
            };

            let state_mut = Arc::make_mut(&mut state);
            let (first, rest) = state_mut.split_at_mut(1);
            let Value::Float(stored_percentile) = &mut first[0] else {
                unreachable!("First element of state must be the percentile")
            };
            let Value::List(collected_values) = &mut rest[0] else {
                unreachable!("Second element of state must be a List")
            };

            *stored_percentile = percentile;
            Arc::make_mut(collected_values).push(Value::Float(val.get_numeric()));

            Ok(Value::List(state))
        }
    );

    funcs.add(
        "percentileCont",
        percentile,
        false,
        false,
        vec![
            Type::Union(vec![Type::Int, Type::Float, Type::Null]),
            Type::Union(vec![Type::Int, Type::Float]),
        ],
        FnType::Aggregation {
            initial: Value::List(Arc::new(thin_vec![
                Value::Float(0.0),
                Value::List(Arc::new(thin_vec![]))
            ])),
            finalizer: Some(Box::new(finalize_percentile_cont)),
            batch_agg: None,
        },
        Type::Union(vec![Type::Float, Type::Null]),
    );

    cypher_fn!(funcs, "stDev",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        agg_init: Value::List(Arc::new(thin_vec![Value::Float(0.0), Value::List(Arc::new(thin_vec![]))])),
        finalizer: finalize_stdev,
        fn stdev(_, args) {
            let mut iter = args.iter().cloned();
            let val = iter.next().unwrap();
            let ctx = iter.next().unwrap();
            match (val, ctx) {
                (Value::Null, ctx) => Ok(ctx),
                (val @ (Value::Int(_) | Value::Float(_)), Value::List(mut vec)) => {
                    let val = val.get_numeric();

                    // Use split_at_mut to get mutable references to both elements safely
                    let vec_mut = Arc::make_mut(&mut vec);
                    let (first, rest) = vec_mut.split_at_mut(1);
                    let (Value::Float(sum), Value::List(values)) = (&mut first[0], &mut rest[0]) else {
                        unreachable!("stdev accumulator should be [sum, values]")
                    };

                    // Mutate in-place:  update sum and push value to list (avoids O(n²) cloning)
                    *sum += val;
                    Arc::make_mut(values).push(Value::Float(val));

                    Ok(Value::List(vec))
                }
                _ => unreachable!("stdev expects Integer, Float, or Null (validation done before call)"),
            }
        }
    );

    funcs.add(
        "stDevP",
        stdev,
        false,
        false,
        vec![Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        FnType::Aggregation {
            initial: Value::List(Arc::new(thin_vec![
                Value::Float(0.0),
                Value::List(Arc::new(thin_vec![]))
            ])),
            finalizer: Some(Box::new(finalize_stdevp)),
            batch_agg: None,
        },
        Type::Union(vec![Type::Float, Type::Null]),
    );
}

fn about_to_overflow(
    a: f64,
    b: f64,
) -> bool {
    a.signum() == b.signum() && a.abs() >= (f64::MAX - b.abs())
}

/// Bulk `count` aggregator. When `inputs` is empty (no input column —
/// `count(*)`) every one of the `num_rows` rows contributes; otherwise
/// only non-null inputs are counted. Cypher spec: `count(x)` ignores
/// nulls; `count(*)` counts every row.
fn count_batch(
    _: &Runtime,
    inputs: &[Value],
    num_rows: usize,
    acc: Value,
) -> Result<Value, String> {
    let prev = match acc {
        Value::Int(n) => n,
        Value::Null => 0,
        _ => unreachable!("count accumulator must be Int"),
    };
    let added = if inputs.is_empty() {
        num_rows as i64
    } else {
        inputs.iter().filter(|v| !matches!(v, Value::Null)).count() as i64
    };
    Ok(Value::Int(prev + added))
}

/// Bulk `collect` aggregator. Appends all non-null inputs into the
/// accumulator list in a single pass, avoiding per-row `Arc::make_mut`.
fn collect_batch(
    _: &Runtime,
    inputs: &[Value],
    _num_rows: usize,
    acc: Value,
) -> Result<Value, String> {
    let mut list = match acc {
        Value::List(l) => l,
        Value::Null => Arc::new(thin_vec![]),
        _ => unreachable!("collect accumulator must be a List"),
    };
    let v = Arc::make_mut(&mut list);
    v.reserve(inputs.len());
    for val in inputs {
        if !matches!(val, Value::Null) {
            v.push(val.clone());
        }
    }
    Ok(Value::List(list))
}

/// Bulk `sum` aggregator. Sums non-null numeric inputs in one pass.
fn sum_batch(
    _: &Runtime,
    inputs: &[Value],
    _num_rows: usize,
    acc: Value,
) -> Result<Value, String> {
    let mut total = match acc {
        Value::Float(f) => f,
        Value::Int(i) => i as f64,
        Value::Null => 0.0,
        _ => unreachable!("sum accumulator must be numeric"),
    };
    for val in inputs {
        match val {
            Value::Null => {}
            Value::Int(i) => total += *i as f64,
            Value::Float(f) => total += *f,
            _ => unreachable!("sum expects Integer, Float, or Null"),
        }
    }
    Ok(Value::Float(total))
}

/// Bulk `max` aggregator. Walks all inputs once; keeps the max via
/// `compare_value`, ignoring null/disjoint comparisons.
fn max_batch(
    _: &Runtime,
    inputs: &[Value],
    _num_rows: usize,
    acc: Value,
) -> Result<Value, String> {
    let mut best = acc;
    for val in inputs {
        if matches!(val, Value::Null) {
            continue;
        }
        if matches!(best, Value::Null) {
            best = val.clone();
            continue;
        }
        let (ord, cmp) = val.compare_value(&best);
        if ord == Ordering::Greater && cmp != DisjointOrNull::ComparedNull {
            best = val.clone();
        }
    }
    Ok(best)
}

/// Bulk `min` aggregator. Mirror of `max_batch`.
fn min_batch(
    _: &Runtime,
    inputs: &[Value],
    _num_rows: usize,
    acc: Value,
) -> Result<Value, String> {
    let mut best = acc;
    for val in inputs {
        if matches!(val, Value::Null) {
            continue;
        }
        if matches!(best, Value::Null) {
            best = val.clone();
            continue;
        }
        let (ord, cmp) = val.compare_value(&best);
        if ord == Ordering::Less && cmp != DisjointOrNull::ComparedNull {
            best = val.clone();
        }
    }
    Ok(best)
}

pub fn finalize_avg(value: Value) -> Value {
    let Value::List(vec) = value else {
        unreachable!("finalize_avg expects a list");
    };
    let (Value::Float(sum), Value::Int(count), Value::Bool(overflow)) = (&vec[0], &vec[1], &vec[2])
    else {
        unreachable!("avg function should have [sum, count, overflow] format");
    };
    if *count == 0 {
        Value::Null
    } else if *overflow {
        Value::Float(*sum)
    } else {
        Value::Float(sum / *count as f64)
    }
}

#[allow(clippy::needless_pass_by_value)]
pub fn finalize_percentile_disc(ctx: Value) -> Value {
    let Value::List(mut state) = ctx else {
        unreachable!()
    };

    let [Value::Float(percentile), Value::List(values)] = Arc::make_mut(&mut state).as_mut_slice()
    else {
        unreachable!()
    };

    if values.is_empty() {
        return Value::Null;
    }

    Arc::make_mut(values).sort_by(|a, b| {
        a.get_numeric()
            .partial_cmp(&b.get_numeric())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let index = if *percentile > 0.0 {
        (values.len() as f64 * *percentile).ceil() as usize - 1
    } else {
        0
    };

    Value::Float(values[index].get_numeric())
}

#[allow(clippy::needless_pass_by_value)]
pub fn finalize_percentile_cont(ctx: Value) -> Value {
    let Value::List(mut state) = ctx else {
        unreachable!()
    };

    let [Value::Float(percentile), Value::List(values)] = Arc::make_mut(&mut state).as_mut_slice()
    else {
        unreachable!()
    };

    if values.is_empty() {
        return Value::Null;
    }

    Arc::make_mut(values).sort_by(|a, b| {
        a.get_numeric()
            .partial_cmp(&b.get_numeric())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    #[allow(clippy::float_cmp)]
    if *percentile == 1.0 || values.len() == 1 {
        return Value::Float(values[values.len() - 1].get_numeric());
    }

    let float_idx = (values.len() - 1) as f64 * *percentile;

    let (fraction_val, int_val) = modf(float_idx);
    let index = int_val as usize;

    if fraction_val == 0.0 {
        return Value::Float(values[index].get_numeric());
    }
    let lhs = values[index].get_numeric() * (1.0 - fraction_val);
    let rhs = values[index + 1].get_numeric() * fraction_val;
    Value::Float(lhs + rhs)
}

const fn modf(x: f64) -> (f64, f64) {
    let int_part = x.trunc();
    let frac_part = x.fract();
    (frac_part, int_part)
}

pub fn finalize_stdev(ctx: Value) -> Value {
    let Value::List(vec) = ctx else {
        unreachable!("finalize_stdev expects a list");
    };
    let (Value::Float(sum), Value::List(values)) = (&vec[0], &vec[1]) else {
        unreachable!("stdev function should have [sum, values] format");
    };
    if values.is_empty() || values.len() == 1 {
        return Value::Float(0.0);
    }
    let mean = sum / values.len() as f64;
    let variance: f64 = values
        .iter()
        .map(|v| {
            let diff = v.get_numeric() - mean;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Value::Float(variance.sqrt())
}

pub fn finalize_stdevp(ctx: Value) -> Value {
    let Value::List(vec) = ctx else {
        unreachable!("finalize_stdev expects a list");
    };
    let (Value::Float(sum), Value::List(values)) = (&vec[0], &vec[1]) else {
        unreachable!("stdev function should have [sum, values] format");
    };
    if values.is_empty() {
        return Value::Float(0.0);
    }
    let mean = sum / values.len() as f64;
    let variance: f64 = values
        .iter()
        .map(|v| {
            let diff = v.get_numeric() - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    Value::Float(variance.sqrt())
}
