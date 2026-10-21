//! Trigonometric functions.
//!
//! Standard trigonometric operations following IEEE 754 semantics.
//! Inputs are `Int` or `Float`; all outputs are `Float`.  `Null`
//! propagates unchanged.
//!
//! ```text
//!  Cypher          Function     Notes
//! ──────────────────────────────────────────────
//!  sin(x)          sin()        radians in
//!  cos(x)          cos()        radians in
//!  tan(x)          tan()        radians in
//!  cot(x)          cot()        cos/sin
//!  asin(x)         asin()       radians out
//!  acos(x)         acos()       radians out
//!  atan(x)         atan()       radians out
//!  atan2(y, x)     atan2()      two-argument
//!  degrees(x)      degrees()    radians -> degrees
//!  radians(x)      radians()    degrees -> radians
//!  pi()            pi()         constant (3.14159...)
//!  haversin(x)     haversin()   (1 - cos(x)) / 2
//! ```

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]

use super::{FnType, Functions, Type};
use crate::runtime::{runtime::Runtime, value::Value};

/// Apply a unary `f64 -> f64` function to a single numeric-or-null argument.
fn apply_unary_float(
    args: &[Value],
    f: fn(f64) -> f64,
) -> Result<Value, String> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Float(f(*n as f64))),
        Value::Float(v) => Ok(Value::Float(f(*v))),
        Value::Null => Ok(Value::Null),
        _ => unreachable!("trig functions expect Int, Float, or Null"),
    }
}

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "sin",
        args: [Type::union([Type::Int, Type::Float, Type::Null])],
        ret: Type::union([Type::Float, Type::Null]),
        fn sin(_, args) { apply_unary_float(args, f64::sin) }
    );

    cypher_fn!(funcs, "cos",
        args: [Type::union([Type::Int, Type::Float, Type::Null])],
        ret: Type::union([Type::Float, Type::Null]),
        fn cos(_, args) { apply_unary_float(args, f64::cos) }
    );

    cypher_fn!(funcs, "tan",
        args: [Type::union([Type::Int, Type::Float, Type::Null])],
        ret: Type::union([Type::Float, Type::Null]),
        fn tan(_, args) { apply_unary_float(args, f64::tan) }
    );

    cypher_fn!(funcs, "cot",
        args: [Type::union([Type::Int, Type::Float, Type::Null])],
        ret: Type::union([Type::Float, Type::Null]),
        fn cot(_, args) {
            apply_unary_float(args, |x| x.cos() / x.sin())
        }
    );

    cypher_fn!(funcs, "asin",
        args: [Type::union([Type::Int, Type::Float, Type::Null])],
        ret: Type::union([Type::Float, Type::Null]),
        fn asin(_, args) { apply_unary_float(args, f64::asin) }
    );

    cypher_fn!(funcs, "acos",
        args: [Type::union([Type::Int, Type::Float, Type::Null])],
        ret: Type::union([Type::Float, Type::Null]),
        fn acos(_, args) { apply_unary_float(args, f64::acos) }
    );

    cypher_fn!(funcs, "atan",
        args: [Type::union([Type::Int, Type::Float, Type::Null])],
        ret: Type::union([Type::Float, Type::Null]),
        fn atan(_, args) { apply_unary_float(args, f64::atan) }
    );

    cypher_fn!(funcs, "atan2",
        args: [
            Type::union([Type::Int, Type::Float, Type::Null]),
            Type::union([Type::Int, Type::Float, Type::Null]),
        ],
        ret: Type::union([Type::Float, Type::Null]),
        fn atan2(_, args) {
            match (&args[0], &args[1]) {
                (Value::Int(y), Value::Int(x)) => Ok(Value::Float((*y as f64).atan2(*x as f64))),
                (Value::Float(y), Value::Float(x)) => Ok(Value::Float(y.atan2(*x))),
                (Value::Int(y), Value::Float(x)) => Ok(Value::Float((*y as f64).atan2(*x))),
                (Value::Float(y), Value::Int(x)) => Ok(Value::Float(y.atan2(*x as f64))),
                (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                _ => unreachable!("atan2 expects two numeric-or-null arguments"),
            }
        }
    );

    cypher_fn!(funcs, "degrees",
        args: [Type::union([Type::Int, Type::Float, Type::Null])],
        ret: Type::union([Type::Float, Type::Null]),
        fn degrees(_, args) { apply_unary_float(args, f64::to_degrees) }
    );

    cypher_fn!(funcs, "radians",
        args: [Type::union([Type::Int, Type::Float, Type::Null])],
        ret: Type::union([Type::Float, Type::Null]),
        fn radians(_, args) { apply_unary_float(args, f64::to_radians) }
    );

    cypher_fn!(funcs, "pi",
        args: [],
        ret: Type::Float,
        fn pi(_, args) {
            debug_assert!(args.is_empty());
            Ok(Value::Float(std::f64::consts::PI))
        }
    );

    cypher_fn!(funcs, "haversin",
        args: [Type::union([Type::Int, Type::Float, Type::Null])],
        ret: Type::union([Type::Float, Type::Null]),
        fn haversin(_, args) {
            apply_unary_float(args, |x| (1.0 - x.cos()) / 2.0)
        }
    );
}
