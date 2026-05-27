//! Type-conversion and type-checking functions.
//!
//! Handles casting between Cypher value types and the boolean
//! predicate `isEmpty`.  Conversion rules follow the C reference
//! implementation closely:
//!
//! ```text
//!  Cypher                  Function            Conversion path
//! ─────────────────────────────────────────────────────────────────
//!  toInteger(x)            value_to_integer()  String -> parse i64/f64 -> floor
//!                                              Float  -> floor (saturating)
//!                                              Bool   -> 0 / 1
//!  toFloat(x)              value_to_float()    String -> parse f64
//!                                              Int    -> cast
//!  toString(x)             value_to_string()   Int    -> decimal
//!                                              Float  -> 6 decimal places
//!                                              Point  -> "point({...})"
//!                                              Temporal -> ISO 8601
//!  toJSON(x)               to_json()           any value -> JSON string
//!  toBoolean(x)            to_boolean()        String -> "true"/"false"
//!                                              Int    -> != 0
//!  isEmpty(x)              is_empty()          String / List / Map -> Bool
//! ```
//!
//! The `*OrNull` variants (`toIntegerOrNull`, `toFloatOrNull`, etc.)
//! reuse the same implementation functions but accept `Type::Any` so
//! unsupported input types return `Null` instead of a type error.

#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use super::{FnType, Functions, Type};
use crate::runtime::{runtime::Runtime, value::Value};
use std::sync::Arc;
use thin_vec::ThinVec;

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "tointeger",
        args: [Type::Union(vec![Type::String, Type::Bool, Type::Int, Type::Float, Type::Null])],
        ret: Type::Union(vec![Type::Int, Type::Null]),
        fn value_to_integer(_runtime, args) {
            match args.first() {
                Some(Value::String(s)) => {
                    if s.is_empty() {
                        return Ok(Value::Null);
                    }
                    if let Ok(i) = s.parse::<i64>() {
                        return Ok(Value::Int(i));
                    }
                    match s.parse::<f64>() {
                        Ok(f) if f.is_finite() => {
                            let floored = f.floor();
                            #[allow(clippy::cast_precision_loss)]
                            let i64_max_as_f64 = i64::MAX as f64;
                            #[allow(clippy::cast_precision_loss)]
                            let i64_min_as_f64 = i64::MIN as f64;
                            if floored >= i64_max_as_f64 || floored < i64_min_as_f64 {
                                return Ok(Value::Null);
                            }
                            #[allow(clippy::cast_possible_truncation)]
                            Ok(Value::Int(floored as i64))
                        }
                        _ => Ok(Value::Null),
                    }
                }
                Some(&Value::Int(i)) => Ok(Value::Int(i)),
                Some(&Value::Float(f)) => {
                    if !f.is_finite() {
                        return Ok(Value::Null);
                    }
                    let floored = f.floor();
                    #[allow(clippy::cast_possible_truncation)]
                    Ok(Value::Int(floored as i64))
                }
                Some(&Value::Bool(b)) => Ok(Value::Int(i64::from(b))),
                _ => Ok(Value::Null),
            }
        }
    );
    funcs.add(
        "toIntegerOrNull",
        value_to_integer,
        false,
        false,
        vec![Type::Any],
        FnType::Function,
        Type::Union(vec![Type::Int, Type::Null]),
    );

    cypher_fn!(funcs, "tofloat",
        args: [Type::Union(vec![Type::String, Type::Float, Type::Int, Type::Null])],
        ret: Type::Union(vec![Type::Float, Type::Null]),
        fn value_to_float(_runtime, args) {
            match args.first() {
                Some(Value::String(s)) => s.parse::<f64>().map(Value::Float).or(Ok(Value::Null)),
                Some(&Value::Float(f)) => Ok(Value::Float(f)),
                Some(&Value::Int(i)) => Ok(Value::Float(i as f64)),
                _ => Ok(Value::Null),
            }
        }
    );
    funcs.add(
        "toFloatOrNull",
        value_to_float,
        false,
        false,
        vec![Type::Any],
        FnType::Function,
        Type::Union(vec![Type::Float, Type::Null]),
    );

    cypher_fn!(funcs, "tostring",
        args: [Type::Union(vec![Type::Datetime, Type::Date, Type::Time, Type::Duration, Type::String, Type::Bool, Type::Int, Type::Float, Type::Null, Type::Point])],
        ret: Type::Union(vec![Type::String, Type::Null]),
        fn value_to_string(_runtime, args) {
            match args.first() {
                Some(Value::String(s)) => Ok(Value::String(Arc::clone(s))),
                Some(Value::Int(i)) => Ok(Value::String(Arc::new(i.to_string()))),
                Some(Value::Float(f)) => {
                    // Format without trailing zeros; ensure at least one decimal
                    let s = format!("{f}");
                    Ok(Value::String(Arc::new(s)))
                }
                Some(Value::Bool(b)) => Ok(Value::String(Arc::new(b.to_string()))),
                Some(Value::Point(p)) => Ok(Value::String(Arc::new(format!(
                    "point({{latitude: {:.6}, longitude: {:.6}}})",
                    p.latitude, p.longitude
                )))),
                Some(&Value::Datetime(ts)) => Ok(Value::String(Arc::new(Value::format_datetime(ts)))),
                Some(&Value::Date(ts)) => Ok(Value::String(Arc::new(Value::format_date(ts)))),
                Some(&Value::Time(ts)) => Ok(Value::String(Arc::new(Value::format_time(ts)))),
                Some(&Value::Duration(dur)) => Ok(Value::String(Arc::new(Value::format_duration(dur)))),
                Some(_) => Ok(Value::Null),
                None => unreachable!(),
            }
        }
    );
    funcs.add(
        "tostringornull",
        value_to_string,
        false,
        false,
        vec![Type::Any],
        FnType::Function,
        Type::Union(vec![Type::String, Type::Null]),
    );

    cypher_fn!(funcs, "tojson",
        args: [Type::Any],
        ret: Type::Union(vec![Type::String, Type::Null]),
        fn to_json(runtime, args) {
            args.first().map_or_else(
                || unreachable!(),
                |v| {
                    let json_string = v.to_json_string(runtime);
                    Ok(Value::String(Arc::new(json_string)))
                },
            )
        }
    );

    cypher_fn!(funcs, "isEmpty",
        args: [Type::Union(vec![Type::Map, Type::List(Box::new(Type::Any)), Type::String, Type::Null])],
        ret: Type::Union(vec![Type::Bool, Type::Null]),
        fn is_empty(_, args) {
            match args.first() {
                Some(Value::Null) => Ok(Value::Null),
                Some(Value::String(s)) => Ok(Value::Bool(s.is_empty())),
                Some(Value::List(v)) => Ok(Value::Bool(v.is_empty())),
                Some(Value::Map(m)) => Ok(Value::Bool(m.is_empty())),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "toBoolean",
        args: [Type::Union(vec![Type::String, Type::Bool, Type::Int, Type::Null])],
        ret: Type::Union(vec![Type::Bool, Type::Null]),
        fn to_boolean(_, args) {
            match args.first() {
                Some(&Value::Bool(b)) => Ok(Value::Bool(b)),
                Some(Value::String(s)) => {
                    if s.eq_ignore_ascii_case("true") {
                        Ok(Value::Bool(true))
                    } else if s.eq_ignore_ascii_case("false") {
                        Ok(Value::Bool(false))
                    } else {
                        Ok(Value::Null)
                    }
                }
                Some(&Value::Int(n)) => Ok(Value::Bool(n != 0)),
                _ => Ok(Value::Null),
            }
        }
    );
    funcs.add(
        "toBooleanOrNull",
        to_boolean,
        false,
        false,
        vec![Type::Any],
        FnType::Function,
        Type::Union(vec![Type::Bool, Type::Null]),
    );

    cypher_fn!(funcs, "toBooleanList",
        args: [Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null])],
        ret: Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
        fn to_boolean_list(_, args) {
            match args.first() {
                Some(Value::List(vs)) => {
                    let result: ThinVec<Value> = vs.iter().map(|v| match v {
                        Value::Bool(b) => Value::Bool(*b),
                        Value::String(s) => {
                            if s.eq_ignore_ascii_case("true") {
                                Value::Bool(true)
                            } else if s.eq_ignore_ascii_case("false") {
                                Value::Bool(false)
                            } else {
                                Value::Null
                            }
                        }
                        Value::Int(n) => Value::Bool(*n != 0),
                        _ => Value::Null,
                    }).collect();
                    Ok(Value::List(Arc::new(result)))
                }
                _ => Ok(Value::Null),
            }
        }
    );

    cypher_fn!(funcs, "toFloatList",
        args: [Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null])],
        ret: Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
        fn to_float_list(_, args) {
            match args.first() {
                Some(Value::List(vs)) => {
                    let result: ThinVec<Value> = vs.iter().map(|v| match v {
                        Value::Float(f) => Value::Float(*f),
                        Value::Int(i) => Value::Float(*i as f64),
                        Value::String(s) => s.parse::<f64>().map_or(Value::Null, Value::Float),
                        _ => Value::Null,
                    }).collect();
                    Ok(Value::List(Arc::new(result)))
                }
                _ => Ok(Value::Null),
            }
        }
    );

    cypher_fn!(funcs, "toIntegerList",
        args: [Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null])],
        ret: Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
        fn to_integer_list(runtime, args) {
            match args.first() {
                Some(Value::List(vs)) => {
                    let result: ThinVec<Value> = vs.iter().map(|v| {
                        value_to_integer(runtime, &[v.clone()]).unwrap_or(Value::Null)
                    }).collect();
                    Ok(Value::List(Arc::new(result)))
                }
                _ => Ok(Value::Null),
            }
        }
    );

    cypher_fn!(funcs, "toStringList",
        args: [Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null])],
        ret: Type::Union(vec![Type::List(Box::new(Type::Any)), Type::Null]),
        fn to_string_list(runtime, args) {
            match args.first() {
                Some(Value::List(vs)) => {
                    let result: ThinVec<Value> = vs.iter().map(|v| {
                        value_to_string(runtime, &[v.clone()]).unwrap_or(Value::Null)
                    }).collect();
                    Ok(Value::List(Arc::new(result)))
                }
                _ => Ok(Value::Null),
            }
        }
    );
}
