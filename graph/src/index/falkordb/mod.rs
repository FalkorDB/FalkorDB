//! The FalkorDB index (the in-repo replacement for RediSearch).
//!
//! [`data_structures`] is the always-compiled substrate (the CoW B⁺-tree). The
//! index proper — the numeric key [`encode`]r, the tree-backed index, and the
//! hooks that turn graph writes into index changes — lives behind the
//! `index-falkordb` feature, off by default.

pub mod data_structures;

#[cfg(feature = "index-falkordb")]
pub mod encode;

#[cfg(feature = "index-falkordb")]
pub mod falkordb_index;

#[cfg(feature = "index-falkordb")]
pub mod graph_writes;

#[cfg(feature = "index-falkordb")]
pub mod numeric;

#[cfg(feature = "index-falkordb")]
pub mod online_build;

#[cfg(feature = "index-falkordb")]
pub mod pending_writes;

#[cfg(feature = "index-falkordb")]
pub mod versioned;

/// The error a scan raises when the native index cannot serve a predicate.
///
/// Under `index-falkordb` the native index is the only index — there is no RediSearch to fall
/// through to — so an unsupported predicate has to surface. It is deliberately an error and not
/// an empty result: a silent zero-row answer is indistinguishable from "nothing matched", which
/// would let an unimplemented kind masquerade as correct behaviour in every test that asserts a
/// count. Failing names the gap instead of hiding it.
///
/// The message carries the entity kind, the label/type, and the predicate shape, so a CI failure
/// or a ledger entry says *what* is missing rather than only *where*.
#[cfg(feature = "index-falkordb")]
#[must_use]
pub fn unsupported_by_native_index(
    entity: &str,
    label: &std::sync::Arc<String>,
    query: &crate::index::IndexQuery<crate::runtime::value::Value>,
) -> String {
    format!(
        "native index cannot serve this predicate on {entity} :{label} — {}. \
         The `index-falkordb` build has no RediSearch fallback; this index kind or predicate \
         shape is not implemented yet.",
        describe_predicate(query)
    )
}

/// A short, stable description of a predicate's *shape* — enough to identify the gap without
/// leaking user values into a log line.
#[cfg(feature = "index-falkordb")]
fn describe_predicate(query: &crate::index::IndexQuery<crate::runtime::value::Value>) -> String {
    use crate::index::IndexQuery as Q;
    match query {
        // Test encodability rather than assuming it. At top level a declined `Equal` is
        // necessarily non-numeric, but the same arm is reached through `And`/`Or`, where the
        // child may be perfectly servable and the *combination* is what is not — reporting it as
        // "non-numeric" there sends the reader after a missing value kind that is not the problem.
        Q::Equal { key, value } => {
            if encode::encode_numeric(value).is_some() {
                format!("equality on `{key}`")
            } else {
                format!("equality on `{key}` with a non-numeric value")
            }
        }
        Q::Range { key, min, max, .. } => {
            let encodable = |v: &Option<crate::runtime::value::Value>| {
                v.as_ref()
                    .is_none_or(|v| encode::encode_numeric(v).is_some())
            };
            if encodable(min) && encodable(max) {
                format!("range on `{key}`")
            } else {
                format!("range on `{key}` with a non-numeric bound")
            }
        }
        // Three reasons a union is declined, and naming the wrong one sends whoever reads the
        // ledger entry after work that isn't the problem. A union spanning attributes needs a real
        // cross-column merge; one with a member that is not an equality or `IN` list (`x = 1 OR
        // x > 5`) needs the union to serve that member kind; a union of equalities and `IN` lists
        // on one attribute has a non-numeric value, which needs that value's index kind.
        Q::Or(children) => {
            let mut members = Vec::new();
            union_members(children, true, &mut members);
            let n = children.len();
            let plural = if n == 1 { "" } else { "s" };
            let one_key = members
                .first()
                .map(|&(key, _)| key)
                .filter(|&first| members.iter().all(|&(key, _)| key == first));
            match one_key {
                None => format!("a union of {n} member{plural} spanning more than one attribute"),
                Some(key) if members.iter().all(|&(_, served)| served) => {
                    format!("a union of {n} member{plural} on `{key}` that is not all-numeric")
                }
                Some(key) => format!(
                    "a union of {n} member{plural} on `{key}` with a member that is not an \
                     equality or IN list"
                ),
            }
        }
        // Name the children, not just the count. "a conjunction of 2 predicates" cannot
        // distinguish two ranges on one key (which wants an arithmetic collapse) from two
        // predicates on different keys (which wants a real intersection) — and it hides the case
        // where the two are the *same* predicate twice, which wants neither.
        Q::And(children) => format!(
            "a conjunction of {} predicates [{}]",
            children.len(),
            children
                .iter()
                .map(describe_predicate)
                .collect::<Vec<_>>()
                .join("; ")
        ),
        // Since #2383 `IN` reaches the index whole, so it is declined here on its own values.
        Q::InList { key, list } => {
            let numeric = matches!(
                list,
                crate::runtime::value::Value::List(items)
                    if items.iter().all(|v| encode::encode_numeric(v).is_some())
            );
            if numeric {
                format!("an IN list on `{key}`")
            } else {
                format!("an IN list on `{key}` with a non-numeric value")
            }
        }
        Q::Point { key, .. } => format!("a geo predicate on `{key}`"),
        Q::ArrayContains { key, .. } => format!("an array-contains on `{key}`"),
    }
}

/// Collect the attribute of every member of a union, paired with whether the native union serves
/// that member kind: an `Equal` or an `IN` list, directly in the union. `NumericIndex::union`
/// declines a nested combinator, so the members of a nested union or conjunction count by their
/// attributes but are not served (`top_level` is `false` for them).
#[cfg(feature = "index-falkordb")]
fn union_members<'a>(
    children: &'a [crate::index::IndexQuery<crate::runtime::value::Value>],
    top_level: bool,
    out: &mut Vec<(&'a std::sync::Arc<String>, bool)>,
) {
    use crate::index::IndexQuery as Q;
    for child in children {
        match child {
            Q::Equal { key, .. } | Q::InList { key, .. } => out.push((key, top_level)),
            Q::Or(nested) | Q::And(nested) => union_members(nested, false, out),
            Q::Range { key, .. } | Q::Point { key, .. } | Q::ArrayContains { key, .. } => {
                out.push((key, false));
            }
        }
    }
}

#[cfg(all(test, feature = "index-falkordb"))]
mod tests {
    use std::sync::Arc;

    use super::describe_predicate;
    use crate::index::IndexQuery as Q;
    use crate::runtime::value::Value;

    fn eq(
        attr: &str,
        v: i64,
    ) -> Q<Value> {
        Q::Equal {
            key: Arc::new(attr.to_string()),
            value: Value::Int(v),
        }
    }

    fn gt(
        attr: &str,
        v: i64,
    ) -> Q<Value> {
        Q::Range {
            key: Arc::new(attr.to_string()),
            min: Some(Value::Int(v)),
            max: None,
            include_min: false,
            include_max: false,
        }
    }

    fn in_list(
        attr: &str,
        values: Vec<Value>,
    ) -> Q<Value> {
        Q::InList {
            key: Arc::new(attr.to_string()),
            list: Value::List(Arc::new(values.into_iter().collect())),
        }
    }

    /// `IN` reaches the index whole, so its message tests the values rather than assuming them.
    #[test]
    fn a_declined_in_list_says_whether_a_value_is_non_numeric() {
        let mixed = in_list(
            "a",
            vec![Value::Int(1), Value::String(Arc::new("x".to_string()))],
        );
        assert_eq!(
            describe_predicate(&mixed),
            "an IN list on `a` with a non-numeric value"
        );
        let numeric = in_list("a", vec![Value::Int(1), Value::Int(2)]);
        assert_eq!(describe_predicate(&numeric), "an IN list on `a`");
    }

    /// The ledger's whole value is that the message names the gap. A union is declined for three
    /// different reasons and they need different work to close — a missing index kind, a member
    /// kind the union does not serve, or a real cross-column merge — so reporting one as another
    /// sends the reader after the wrong thing.
    #[test]
    fn a_declined_union_says_which_reason_applies() {
        let mixed = Q::Or(vec![
            eq("a", 1),
            Q::Equal {
                key: Arc::new("a".to_string()),
                value: Value::String(Arc::new("x".to_string())),
            },
        ]);
        let msg = describe_predicate(&mixed);
        assert!(msg.contains("not all-numeric"), "{msg}");
        assert!(msg.contains('a'), "names the attribute: {msg}");

        let cross = Q::Or(vec![eq("a", 1), eq("b", 2)]);
        let msg = describe_predicate(&cross);
        assert!(
            msg.contains("more than one attribute"),
            "a cross-attribute union must not be reported as non-numeric: {msg}"
        );

        // Nested unions are flattened for this purpose, the same way the router flattens them.
        let nested_cross = Q::Or(vec![Q::Or(vec![eq("a", 1)]), Q::Or(vec![eq("b", 2)])]);
        assert!(
            describe_predicate(&nested_cross).contains("more than one attribute"),
            "nesting must not hide the second attribute"
        );

        // `a = 1 OR a > 5`: one attribute, declined because the union serves only equalities
        // and IN lists.
        let with_range = Q::Or(vec![eq("a", 1), gt("a", 5)]);
        let msg = describe_predicate(&with_range);
        assert!(msg.contains("not an equality or IN list"), "{msg}");
        assert!(msg.contains("`a`"), "names the attribute: {msg}");

        // An IN list is a member kind the union serves; its string value is what declines it.
        let mixed_in = Q::Or(vec![
            in_list(
                "a",
                vec![Value::Int(1), Value::String(Arc::new("x".to_string()))],
            ),
            eq("a", 2),
        ]);
        let msg = describe_predicate(&mixed_in);
        assert!(msg.contains("not all-numeric"), "{msg}");

        // A range on a second attribute still makes it a cross-attribute union.
        let range_cross = Q::Or(vec![eq("a", 1), gt("b", 5)]);
        assert!(
            describe_predicate(&range_cross).contains("more than one attribute"),
            "{}",
            describe_predicate(&range_cross)
        );
    }
}
