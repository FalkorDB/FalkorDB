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
        // cross-column merge; one with a range or other non-equality member (`x = 1 OR x > 5`)
        // needs the union to serve that member kind; a union of equalities on one attribute has a
        // non-numeric member, which needs that value's index kind.
        Q::Or(children) => {
            let mut members = Vec::new();
            union_members(children, false, &mut members);
            let n = children.len();
            let plural = if n == 1 { "" } else { "s" };
            let one_key = members
                .first()
                .map(|&(key, _)| key)
                .filter(|&first| members.iter().all(|&(key, _)| key == first));
            match one_key {
                None => format!("a union of {n} member{plural} spanning more than one attribute"),
                Some(key) if members.iter().all(|&(_, is_equality)| is_equality) => {
                    format!("a union of {n} member{plural} on `{key}` that is not all-numeric")
                }
                Some(key) => {
                    format!("a union of {n} member{plural} on `{key}` with a non-equality member")
                }
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
        Q::InList { key, .. } => format!("an IN list on `{key}`"),
        Q::Point { key, .. } => format!("a geo predicate on `{key}`"),
        Q::ArrayContains { key, .. } => format!("an array-contains on `{key}`"),
    }
}

/// Collect the attribute of every member of a (possibly nested) union, paired with whether the
/// member is an `Equal`, the only member kind the native union serves. Nested unions are
/// flattened the same way the router flattens them; the children of a conjunction inside the
/// union count as non-equality members.
#[cfg(feature = "index-falkordb")]
fn union_members<'a>(
    children: &'a [crate::index::IndexQuery<crate::runtime::value::Value>],
    in_conjunction: bool,
    out: &mut Vec<(&'a std::sync::Arc<String>, bool)>,
) {
    use crate::index::IndexQuery as Q;
    for child in children {
        match child {
            Q::Equal { key, .. } => out.push((key, !in_conjunction)),
            Q::Or(nested) => union_members(nested, in_conjunction, out),
            Q::And(nested) => union_members(nested, true, out),
            Q::Range { key, .. }
            | Q::Point { key, .. }
            | Q::InList { key, .. }
            | Q::ArrayContains { key, .. } => out.push((key, false)),
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

        // `a = 1 OR a > 5`: one attribute, declined because the union serves only equalities.
        let with_range = Q::Or(vec![eq("a", 1), gt("a", 5)]);
        let msg = describe_predicate(&with_range);
        assert!(msg.contains("non-equality member"), "{msg}");
        assert!(msg.contains("`a`"), "names the attribute: {msg}");

        // A range on a second attribute still makes it a cross-attribute union.
        let range_cross = Q::Or(vec![eq("a", 1), gt("b", 5)]);
        assert!(
            describe_predicate(&range_cross).contains("more than one attribute"),
            "{}",
            describe_predicate(&range_cross)
        );
    }
}
