//! Internal-only operator functions.
//!
//! These functions are **not** exposed to users via Cypher syntax
//! directly.  Instead, the parser rewires higher-level Cypher
//! constructs into calls to these internal helpers:
//!
//! ```text
//!  Cypher syntax            Internal function       Registered as
//! ────────────────────────────────────────────────────────────────
//!  x STARTS WITH y          internal_starts_with()  FnType::Internal
//!  x ENDS WITH y            internal_ends_with()    FnType::Internal
//!  x CONTAINS y             internal_contains()     FnType::Internal
//!  x IS [NOT] NULL          internal_is_null()      FnType::Internal
//!  x =~ pattern             internal_regex_matches() FnType::Internal
//!  CASE ... WHEN ... END    internal_case()         FnType::Internal
//! ```
//!
//! Because they are registered with `FnType::Internal`, they cannot
//! be invoked by name in user queries (the parser's function lookup
//! filters by `FnType`).
//!
//! `internal_case` supports both simple (`CASE expr WHEN v1 THEN ...`)
//! and generic (`CASE WHEN cond THEN ...`) forms, encoded as a list
//! of `[condition, result]` pairs plus an optional else clause.

#![allow(clippy::unnecessary_wraps)]

use super::{FnType, Functions, Type};
use crate::runtime::{runtime::Runtime, value::Value};

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "starts_with",
        args: [
            Type::Any,
            Type::Any,
        ],
        ret: Type::union([Type::Bool, Type::Null]),
        internal,
        fn internal_starts_with(_, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(Value::String(s)), Some(Value::String(prefix))) => {
                    Ok(Value::Bool(s.starts_with(prefix.as_str())))
                }
                _ => Ok(Value::Null),
            }
        }
    );

    cypher_fn!(funcs, "ends_with",
        args: [
            Type::Any,
            Type::Any,
        ],
        ret: Type::union([Type::Bool, Type::Null]),
        internal,
        fn internal_ends_with(_, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(Value::String(s)), Some(Value::String(suffix))) => {
                    Ok(Value::Bool(s.ends_with(suffix.as_str())))
                }
                _ => Ok(Value::Null),
            }
        }
    );

    cypher_fn!(funcs, "contains",
        args: [
            Type::Any,
            Type::Any,
        ],
        ret: Type::union([Type::Bool, Type::Null]),
        internal,
        fn internal_contains(_, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(Value::String(s)), Some(Value::String(substring))) => {
                    Ok(Value::Bool(s.contains(substring.as_str())))
                }
                _ => Ok(Value::Null),
            }
        }
    );

    cypher_fn!(funcs, "is_null",
        args: [Type::union([Type::Bool]), Type::Any],
        ret: Type::union([Type::Bool, Type::Null]),
        internal,
        fn internal_is_null(_, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(&Value::Bool(is_not)), Some(Value::Null)) => Ok(Value::Bool(!is_not)),
                (Some(&Value::Bool(is_not)), Some(_)) => Ok(Value::Bool(is_not)),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "regex_matches",
        args: [
            Type::union([Type::String, Type::Null]),
            Type::union([Type::String, Type::Null]),
        ],
        ret: Type::union([Type::Bool, Type::Null]),
        internal,
        fn internal_regex_matches(_, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(Value::String(s)), Some(Value::String(pattern))) => {
                    // openCypher `=~` is a whole-string match; \A/\z anchor
                    // without matching before a trailing newline (#2603).
                    let anchored = format!(r"\A(?:{pattern})\z");
                    match regex::Regex::new(anchored.as_str()) {
                        Ok(re) => Ok(Value::Bool(re.is_match(s.as_str()))),
                        Err(_) => match regex::Regex::new(pattern.as_str()) {
                            // Report the original pattern's error, not the
                            // wrapped form's.
                            Ok(_) => unreachable!("anchored compile failed but raw compiled"),
                            Err(e) => Err(format!("Invalid regex pattern: {e}")),
                        },
                    }
                }
                (Some(Value::Null), _) | (_, Some(Value::Null)) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    // ── case (internal, for dbms.functions() enumeration) ─────────────
    // CASE is handled by ExprIR::Case in the evaluator, which short-circuits
    // instead of evaluating every branch into a list first. This registration
    // exists so dbms.functions() can enumerate it.
    cypher_fn!(funcs, "case",
        var_arg: Type::Any,
        ret: Type::Any,
        internal,
        fn internal_case(_, _args) {
            Err(String::from("Internal function 'case' should not be called directly"))
        }
    );

    // ── add (internal, for dbms.functions() enumeration) ──────────────
    // The actual `+` operator is handled by ExprIR::Add in the evaluator.
    // This registration exists so dbms.functions() can enumerate it.
    cypher_fn!(funcs, "add",
        args: [
            Type::union([Type::Map, Type::List(Box::new(Type::Any)), Type::Datetime, Type::Date, Type::Time, Type::Duration, Type::String, Type::Bool, Type::Int, Type::Float, Type::Null]),
            Type::union([Type::Map, Type::List(Box::new(Type::Any)), Type::Datetime, Type::Date, Type::Time, Type::Duration, Type::String, Type::Bool, Type::Int, Type::Float, Type::Null]),
        ],
        ret: Type::union([Type::Map, Type::List(Box::new(Type::Any)), Type::Datetime, Type::Date, Type::Time, Type::Duration, Type::String, Type::Bool, Type::Int, Type::Float, Type::Null]),
        internal,
        fn internal_add(_, _args) {
            Err(String::from("Internal function 'add' should not be called directly"))
        }
    );
}
