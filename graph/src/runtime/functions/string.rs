//! String manipulation functions.
//!
//! Implements the standard Cypher string functions.  All functions
//! propagate `Null` inputs as `Null` outputs (three-valued logic).
//!
//! ```text
//!  Cypher                     Function               Notes
//! ────────────────────────────────────────────────────────────────
//!  substring(s, start [,len]) substring()            0-based index
//!  split(s, delim)            split()                returns [String]
//!  toLower(s)                 string_to_lower()      rejects U+FFFD
//!  toUpper(s)                 string_to_upper()      rejects U+FFFD
//!  replace(s, search, repl)   string_replace()
//!  left(s, n)                 string_left()          first n chars
//!  lTrim(s)                   string_ltrim()         trims spaces
//!  rTrim(s)                   string_rtrim()         trims spaces
//!  trim(s)                    string_trim()          trims spaces
//!  right(s, n)                string_right()         last n chars
//!  string.join(list [,delim]) string_join()          pre-allocates
//!  string.matchRegEx(s, re)   string_match_reg_ex()  returns [[String]]
//!  string.replaceRegEx(...)   string_replace_reg_ex()
//! ```
//!
//! `toLower` / `toUpper` detect the Unicode replacement character
//! (`U+FFFD`) and return an error, mirroring the C implementation's
//! behaviour on invalid UTF-8 input.

#![allow(clippy::unnecessary_wraps)]

use super::{FnType, Functions, Type};
use crate::runtime::{runtime::Runtime, string_pool, value::Value};
use std::sync::Arc;
use thin_vec::thin_vec;

pub fn register(funcs: &mut Functions) {
    cypher_fn!(funcs, "intern",
        args: [Type::union([Type::String, Type::Null])],
        ret: Type::union([Type::String, Type::Null]),
        non_deterministic,
        fn intern(_runtime, args) {
            match args.first() {
                Some(Value::String(s)) => {
                    Ok(Value::String(string_pool::global().intern(Arc::clone(s))))
                }
                Some(Value::Null) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "substring",
        args: [
            Type::union([Type::String, Type::Null]),
            Type::Int,
            Type::Optional(Box::new(Type::Int)),
        ],
        ret: Type::union([Type::String, Type::Null]),
        fn substring(_runtime, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next(), iter.next()) {
                // Handle NULL input case
                (Some(Value::Null), _, _) => Ok(Value::Null),
                // Two-argument version: (string, start)
                (Some(Value::String(s)), Some(&Value::Int(start)), None) => {
                    if start < 0 {
                        return Err("start must be a non-negative integer".into());
                    }
                    if start >= s.len() as _ {
                        return Ok(Value::String(Arc::new(String::new())));
                    }
                    let start = start as usize;

                    Ok(Value::String(Arc::new(s.chars().skip(start).collect())))
                }

                // Three-argument version: (string, start, length)
                (Some(Value::String(s)), Some(&Value::Int(start)), Some(&Value::Int(length))) => {
                    if start < 0 {
                        return Err("start must be a non-negative integer".into());
                    }

                    let start = start as usize;
                    if start >= s.len() {
                        return Ok(Value::String(Arc::new(String::new())));
                    }

                    if length < 0 {
                        return Err("length must be a non-negative integer".into());
                    }

                    let length = length as usize;

                    Ok(Value::String(Arc::new(
                        s.chars().skip(start).take(length).collect(),
                    )))
                }

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "split",
        args: [
            Type::union([Type::String, Type::Null]),
            Type::union([Type::String, Type::Null]),
        ],
        ret: Type::union([Type::List(Box::new(Type::String)), Type::Null]),
        fn split(_runtime, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(Value::String(string)), Some(Value::String(delimiter))) => {
                    if string.is_empty() {
                        Ok(Value::List(Arc::new(thin_vec![Value::String(Arc::new(
                            String::new()
                        ))])))
                    } else if delimiter.is_empty() {
                        // split string to characters
                        let parts = string
                            .chars()
                            .map(|c| Value::String(Arc::new(String::from(c))))
                            .collect();
                        Ok(Value::List(Arc::new(parts)))
                    } else {
                        let parts = string
                            .split(delimiter.as_str())
                            .map(|s| Value::String(Arc::new(String::from(s))))
                            .collect();
                        Ok(Value::List(Arc::new(parts)))
                    }
                }
                (Some(Value::Null), Some(_)) | (Some(_), Some(Value::Null)) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "tolower",
        args: [Type::union([Type::String, Type::Null])],
        ret: Type::union([Type::String, Type::Null]),
        fn string_to_lower(_runtime, args) {
            match args.first() {
                Some(Value::String(s)) => {
                    // Match C behavior: detect replacement character which indicates invalid UTF-8
                    // In the C version, str_tolower returns NULL on invalid UTF-8 (c == -1)
                    // In Rust, we check for the replacement character
                    if s.contains('\u{FFFD}') {
                        return Err(String::from("Invalid UTF8 string"));
                    }
                    let lower = s.to_lowercase();
                    Ok(Value::String(Arc::new(lower)))
                }
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "toupper",
        args: [Type::union([Type::String, Type::Null])],
        ret: Type::union([Type::String, Type::Null]),
        fn string_to_upper(_runtime, args) {
            match args.first() {
                Some(Value::String(s)) => {
                    // Match C behavior: detect replacement character which indicates invalid UTF-8
                    // In the C version, str_toupper returns NULL on invalid UTF-8 (c == -1)
                    // In Rust, we check for the replacement character
                    if s.contains('\u{FFFD}') {
                        return Err(String::from("Invalid UTF8 string"));
                    }
                    let upper = s.to_uppercase();
                    Ok(Value::String(Arc::new(upper)))
                }
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "replace",
        args: [
            Type::union([Type::String, Type::Null]),
            Type::union([Type::String, Type::Null]),
            Type::union([Type::String, Type::Null]),
        ],
        ret: Type::union([Type::String, Type::Null]),
        fn string_replace(_runtime, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next(), iter.next()) {
                (Some(Value::String(s)), Some(Value::String(search)), Some(Value::String(replacement))) => {
                    Ok(Value::String(Arc::new(
                        s.replace(search.as_str(), replacement.as_str()),
                    )))
                }
                (Some(Value::Null), _, _) | (_, Some(Value::Null), _) | (_, _, Some(Value::Null)) => {
                    Ok(Value::Null)
                }

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "left",
        args: [
            Type::union([Type::String, Type::Null]),
            Type::union([Type::Int, Type::Null]),
        ],
        ret: Type::union([Type::String, Type::Null]),
        fn string_left(_runtime, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(Value::String(s)), Some(&Value::Int(n))) => {
                    if n < 0 {
                        Err(String::from("length must be a non-negative integer"))
                    } else {
                        Ok(Value::String(Arc::new(
                            s.chars().take(n as usize).collect(),
                        )))
                    }
                }
                (Some(Value::Null), _) => Ok(Value::Null),
                (_, Some(Value::Null)) => Err(String::from("length must be a non-negative integer")),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "ltrim",
        args: [Type::union([Type::String, Type::Null])],
        ret: Type::union([Type::String, Type::Null]),
        fn string_ltrim(_runtime, args) {
            match args.first() {
                Some(Value::String(s)) => Ok(Value::String(Arc::new(String::from(
                    s.trim_start_matches(' '),
                )))),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "rtrim",
        args: [Type::union([Type::String, Type::Null])],
        ret: Type::union([Type::String, Type::Null]),
        fn string_rtrim(_runtime, args) {
            match args.first() {
                Some(Value::String(s)) => Ok(Value::String(Arc::new(String::from(
                    s.trim_end_matches(' '),
                )))),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "trim",
        args: [Type::union([Type::String, Type::Null])],
        ret: Type::union([Type::String, Type::Null]),
        fn string_trim(_runtime, args) {
            match args.first() {
                Some(Value::String(s)) => Ok(Value::String(Arc::new(String::from(s.trim_matches(' '))))),
                Some(Value::Null) => Ok(Value::Null),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "right",
        args: [
            Type::union([Type::String, Type::Null]),
            Type::union([Type::Int, Type::Null]),
        ],
        ret: Type::union([Type::String, Type::Null]),
        fn string_right(_runtime, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(Value::String(s)), Some(&Value::Int(n))) => {
                    if n < 0 {
                        Err(String::from("length must be a non-negative integer"))
                    } else {
                        let start = s.chars().count().saturating_sub(n as usize);
                        Ok(Value::String(Arc::new(s.chars().skip(start).collect())))
                    }
                }
                (Some(Value::Null), _) => Ok(Value::Null),
                (_, Some(Value::Null)) => Err(String::from("length must be a non-negative integer")),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "string.join",
        args: [
            Type::union([Type::List(Box::new(Type::Any)), Type::Null]),
            Type::Optional(Box::new(Type::String)),
        ],
        ret: Type::union([Type::String, Type::Null]),
        fn string_join(_runtime, args) {
            /// Convert Value list to Arc<String> vector, checking types
            fn to_string_vec(values: &[Value]) -> Result<Vec<Arc<String>>, String> {
                values
                    .iter()
                    .map(|value| match value {
                        Value::String(s) => Ok(Arc::clone(s)),
                        _ => Err(format!(
                            "Type mismatch: expected String but was {}",
                            value.name()
                        )),
                    })
                    .collect()
            }

            /// Compute the total length needed, with overflow detection
            /// Uses i32 for calculations to match C implementation behavior
            fn compute_join_length(
                strings: &[Arc<String>],
                delimiter: &str,
            ) -> Result<usize, String> {
                if strings.is_empty() {
                    return Ok(0);
                }

                let delimiter_len =
                    i32::try_from(delimiter.len()).map_err(|_| String::from("String overflow"))?;
                let n = i32::try_from(strings.len()).map_err(|_| String::from("String overflow"))?;
                let mut str_len: i32 = 0;

                if n >= 2 {
                    let delimiter_contribution = delimiter_len
                        .checked_mul(n - 1)
                        .ok_or_else(|| String::from("String overflow"))?;

                    str_len = str_len
                        .checked_add(delimiter_contribution)
                        .ok_or_else(|| String::from("String overflow"))?;
                }

                for s in strings {
                    let s_len = i32::try_from(s.len()).map_err(|_| String::from("String overflow"))?;

                    str_len = str_len
                        .checked_add(s_len)
                        .ok_or_else(|| String::from("String overflow"))?;
                }

                str_len = str_len
                    .checked_add(1)
                    .ok_or_else(|| String::from("String overflow"))?;

                let capacity = (str_len - 1) as usize;
                Ok(capacity)
            }

            /// Join strings with pre-allocated buffer
            fn join_with_preallocate(
                strings: &[Arc<String>],
                delimiter: &str,
                capacity: usize,
            ) -> String {
                if strings.is_empty() {
                    return String::new();
                }

                let mut result = String::with_capacity(capacity);
                let mut first = true;

                for s in strings {
                    if !first {
                        result.push_str(delimiter);
                    }
                    result.push_str(s.as_str());
                    first = false;
                }

                debug_assert_eq!(result.len(), capacity, "String join calculation mismatch");
                result
            }

            let mut iter = args.iter();

            let first = iter.next().unwrap();

            match (first, iter.next()) {
                (Value::List(vec), Some(Value::String(s))) => {
                    let strings = to_string_vec(vec)?;
                    let size = compute_join_length(&strings, s.as_str())?;
                    let joined = join_with_preallocate(&strings, s.as_str(), size);
                    Ok(Value::String(Arc::new(joined)))
                }
                (Value::List(vec), None) => {
                    let strings = to_string_vec(vec)?;
                    let size = compute_join_length(&strings, "")?;
                    let joined = join_with_preallocate(&strings, "", size);
                    Ok(Value::String(Arc::new(joined)))
                }
                (Value::List(_), Some(Value::Null)) | (Value::Null, _) => Ok(Value::Null),
                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "string.matchRegEx",
        args: [
            Type::union([Type::String, Type::Null]),
            Type::union([Type::String, Type::Null]),
        ],
        ret: Type::union([Type::List(Box::new(Type::Any)), Type::Null]),
        fn string_match_reg_ex(_runtime, args) {
            let mut iter = args.iter();
            match (iter.next(), iter.next()) {
                (Some(Value::String(text)), Some(Value::String(pattern))) => {
                    match regex::Regex::new(pattern.as_str()) {
                        Ok(re) => {
                            let mut all_matches = thin_vec![];
                            // For each match, create a sub-list containing the full match and all capture groups
                            for caps in re.captures_iter(text.as_str()) {
                                let mut match_list = thin_vec![];
                                // Iterate through all capture groups (0 = full match, 1+ = capture groups)
                                // Include NULL for non-participating optional groups to maintain index consistency
                                for i in 0..caps.len() {
                                    if let Some(m) = caps.get(i) {
                                        match_list.push(Value::String(Arc::new(String::from(m.as_str()))));
                                    } else {
                                        // Non-participating optional group - use NULL to preserve position
                                        match_list.push(Value::Null);
                                    }
                                }
                                // Add this match's captures as a sub-list
                                all_matches.push(Value::List(Arc::new(match_list)));
                            }
                            Ok(Value::List(Arc::new(all_matches)))
                        }
                        Err(e) => Err(format!("Invalid regex, {e}")),
                    }
                }
                (Some(Value::Null), Some(_)) | (Some(_), Some(Value::Null)) => Ok(Value::List(Arc::new(thin_vec![]))),

                _ => unreachable!(),
            }
        }
    );

    cypher_fn!(funcs, "string.replaceRegEx",
        args: [
            Type::union([Type::String, Type::Null]),
            Type::union([Type::String, Type::Null]),
            Type::Optional(Box::new(Type::union([Type::String, Type::Null]))),
        ],
        ret: Type::union([Type::String, Type::Null]),
        fn string_replace_reg_ex(_runtime, args) {
            let mut iter = args.iter();
            let text = iter.next();
            let pattern = iter.next();
            let replacement = iter.next(); // May be None (optional third argument)

            match (text, pattern) {
                // NULL text or pattern returns NULL
                (Some(Value::Null), _) | (_, Some(Value::Null)) => Ok(Value::Null),

                // Valid text and pattern
                (Some(Value::String(text)), Some(Value::String(pattern))) => {
                    // Compile the regex first (before handling replacement)
                    let re = match regex::Regex::new(pattern.as_str()) {
                        Ok(re) => re,
                        Err(e) => return Err(format!("Invalid regex, {e}")),
                    };

                    // Now handle replacement and perform replacement in one step
                    match replacement {
                        // No third argument provided, default to empty string
                        None => {
                            let replaced_text = re.replace_all(text.as_str(), "").into_owned();
                            Ok(Value::String(Arc::new(replaced_text)))
                        }
                        // NULL replacement returns NULL
                        Some(Value::Null) => Ok(Value::Null),
                        // Use provided string
                        Some(Value::String(repl)) => {
                            let replaced_text = re.replace_all(text.as_str(), repl.as_str()).into_owned();
                            Ok(Value::String(Arc::new(replaced_text)))
                        }
                        // All other types should have been caught by type checking
                        Some(v) => Err(format!(
                            "Type mismatch: expected String or Null but was {}",
                            v.name()
                        )),
                    }
                }

                // All other type combinations should have been caught by type checking
                _ => unreachable!(),
            }
        }
    );
}
