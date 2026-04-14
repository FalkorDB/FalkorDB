//! Numeric and general math functions.
//!
//! Standard Cypher math operations plus utility functions like `range()`
//! and `coalesce()`.  All functions propagate `Null` following
//! three-valued logic.
//!
//! ```text
//!  Cypher           Function       Returns
//! ──────────────────────────────────────────────
//!  abs(x)           abs()          Int | Float
//!  ceil(x)          ceil()         Int | Float
//!  e()              e()            Float (Euler's number)
//!  exp(x)           exp()          Float
//!  floor(x)         floor()        Int | Float
//!  log(x)           log()          Float (natural log)
//!  log10(x)         log10()        Float
//!  randomUUID()     random_uuid()  String (v4 UUID)
//!  x ^ y            apply_pow()    Float  (also used by expr eval)
//!  pow(x, y)        pow()          Float  (wraps apply_pow)
//!  rand()           rand()         Float  [0, 1)
//!  round(x)         round()        Int | Float
//!  sign(x)          sign()         Int
//!  sqrt(x)          sqrt()         Float
//!  range(s, e [,step]) range()     [Int]  (Arc-wrapped)
//!  coalesce(...)    coalesce()     first non-Null arg
//! ```
//!
//! `apply_pow` is the only `pub` item (re-exported from `mod.rs`) because
//! the runtime expression evaluator uses it directly for the `^` operator.

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]

use super::{FnType, Functions, Type};
use crate::runtime::{runtime::Runtime, value::Value};
use rand::RngExt;
use std::sync::Arc;
use thin_vec::{ThinVec, thin_vec};

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "abs",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Int, Type::Float, Type::Null]),
        fn abs(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Int(n.abs())),
                Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "ceil",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Int, Type::Float, Type::Null]),
        fn ceil(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Int(n)),
                Some(Value::Float(f)) => Ok(Value::Float(f.ceil())),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "e",
        args: [],
        ret: Type::Float,
        fn e(_, args) {
            match args.into_iter().next() {
                None => Ok(Value::Float(std::f64::consts::E)),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "exp",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn exp(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Float((n as f64).exp())),
                Some(Value::Float(f)) => Ok(Value::Float(f.exp())),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "floor",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Int, Type::Float, Type::Null]),
        fn floor(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Int(n)),
                Some(Value::Float(f)) => Ok(Value::Float(f.floor())),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "log",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn log(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Float((n as f64).ln())),
                Some(Value::Float(f)) => Ok(Value::Float(f.ln())),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "log10",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn log10(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Float((n as f64).log10())),
                Some(Value::Float(f)) => Ok(Value::Float(f.log10())),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "randomUUID",
        args: [],
        ret: Type::String,
        non_deterministic,
        fn random_uuid(_, _args) {
            // Generate 16 random bytes (128 bits)
            let mut rng = rand::rng();
            let mut bytes = [0u8; 16];
            rng.fill(&mut bytes);

            // Set version to 4 (random UUID) - set bits 12-15 of byte 6 to 0100
            bytes[6] = (bytes[6] & 0x0F) | 0x40;

            // Set variant to RFC 4122 - set bits 6-7 of byte 8 to 10
            bytes[8] = (bytes[8] & 0x3F) | 0x80;

            // Format as UUID string:  xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
            let uuid = format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:04x}{:08x}",
                u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                u16::from_be_bytes([bytes[4], bytes[5]]),
                u16::from_be_bytes([bytes[6], bytes[7]]),
                u16::from_be_bytes([bytes[8], bytes[9]]),
                u16::from_be_bytes([bytes[10], bytes[11]]),
                u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            );

            Ok(Value::String(Arc::new(uuid)))
        }
    );

    cypher_fn!(funcs, "pow",
        args: [
            Type::Union(vec![Type::Int, Type::Float, Type::Null]),
            Type::Union(vec![Type::Int, Type::Float, Type::Null]),
        ],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn pow(_, args) {
            let mut iter = args.into_iter();
            match (iter.next(), iter.next()) {
                (Some(a), Some(b)) => Ok(apply_pow(a, b)),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "rand",
        args: [],
        ret: Type::Float,
        non_deterministic,
        #[allow(clippy::needless_pass_by_value)]
        fn rand(_, args) {
            debug_assert!(args.is_empty());
            let mut rng = rand::rng();
            Ok(Value::Float(rng.random_range(0.0..1.0)))
        }
    );

    cypher_fn!(funcs, "round",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Int, Type::Float, Type::Null]),
        fn round(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Int(n)),
                Some(Value::Float(f)) => Ok(Value::Float(f.round())),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "sign",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Int, Type::Null]),
        fn sign(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => Ok(Value::Int(n.signum())),
                Some(Value::Float(f)) => Ok(if f == 0.0 {
                    Value::Int(0)
                } else {
                    Value::Float(f.signum().round())
                }),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "sqrt",
        args: [Type::Union(vec![Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn sqrt(_, args) {
            match args.into_iter().next() {
                Some(Value::Int(n)) => {
                    if n < 0 {
                        Ok(Value::Float(f64::NAN))
                    } else {
                        Ok(Value::Float((n as f64).sqrt()))
                    }
                }
                Some(Value::Float(f)) => {
                    if f >= 0f64 {
                        Ok(Value::Float(f.sqrt()))
                    } else {
                        Ok(Value::Float(f64::NAN))
                    }
                }
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "range",
        args: [Type::Int, Type::Int, Type::Optional(Box::new(Type::Int))],
        ret: Type::Union(vec![Type::List(Box::new(Type::Int)), Type::Null]),
        fn range(_, args) {
            let mut iter = args.into_iter();
            let start = iter.next().ok_or("Missing start value")?;
            let end = iter.next().ok_or("Missing end value")?;
            let step = iter.next().unwrap_or_else(|| Value::Int(1));
            match (start, end, step) {
                (Value::Int(start), Value::Int(end), Value::Int(step)) => {
                    if step == 0 {
                        return Err(String::from(
                            "ArgumentError: step argument to range() can't be 0",
                        ));
                    }
                    if (start > end && step > 0) || (start < end && step < 0) {
                        return Ok(Value::List(Arc::new(thin_vec![])));
                    }

                    let length = (end - start) / step + 1;
                    #[allow(clippy::cast_lossless)]
                    if length > u32::MAX as i64 {
                        return Err(String::from("Range too large"));
                    }
                    if step > 0 {
                        return Ok(Value::List(Arc::new(
                            (start..=end)
                                .step_by(step as usize)
                                .map(Value::Int)
                                .collect(),
                        )));
                    }
                    Ok(Value::List(Arc::new(
                        (end..=start)
                            .rev()
                            .step_by((-step) as usize)
                            .map(Value::Int)
                            .collect(),
                    )))
                }

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "coalesce",
        var_arg: Type::Any,
        ret: Type::Any,
        fn coalesce(_, args) {
            let iter = args.into_iter();
            for arg in iter {
                if arg == Value::Null {
                    continue;
                }
                return Ok(arg);
            }
            Ok(Value::Null)
        }
    );
}

// called from fn pow and expr pow (^)
#[inline]
#[must_use]
pub fn apply_pow(
    base: Value,
    exponent: Value,
) -> Value {
    match (base, exponent) {
        // Convert all numeric types to f64 and use powf
        // This matches C's behavior:  pow(SI_GET_NUMERIC(base), SI_GET_NUMERIC(exp))
        (Value::Int(a), Value::Int(b)) => Value::Float((a as f64).powf(b as f64)),
        (Value::Float(a), Value::Float(b)) => Value::Float(a.powf(b)),
        (Value::Int(a), Value::Float(b)) => Value::Float((a as f64).powf(b)),
        (Value::Float(a), Value::Int(b)) => Value::Float(a.powf(b as f64)),
        _ => Value::Null,
    }
}
