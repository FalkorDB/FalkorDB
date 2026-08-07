//! The FalkorDB index (the in-repo replacement for RediSearch).
//!
//! [`data_structures`] is the always-compiled substrate (the CoW B⁺-tree). The
//! index proper — the numeric key [`encode`]r, the tree-backed index, and its
//! runtime wiring — lives behind the `index-falkordb` feature, off by default.

pub mod data_structures;

#[cfg(feature = "index-falkordb")]
pub mod build_registry;

#[cfg(feature = "index-falkordb")]
pub mod encode;

#[cfg(feature = "index-falkordb")]
pub mod falkordb_index;

#[cfg(feature = "index-falkordb")]
pub mod numeric;

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
        Q::Equal { key, .. } => format!("equality on `{key}` with a non-numeric value"),
        Q::Range { key, .. } => format!("range on `{key}` with a non-numeric bound"),
        // Two reasons a union is declined, and naming the wrong one sends whoever reads the
        // ledger entry looking for a missing value kind that isn't the problem. A union spanning
        // attributes needs a real cross-column intersection/merge; one with a non-numeric member
        // needs that member's index kind.
        Q::Or(children) => {
            let mut keys = Vec::new();
            let all_same_key = union_keys(children, &mut keys)
                && keys.first().is_some_and(|f| keys.iter().all(|k| k == f));
            let n = children.len();
            let plural = if n == 1 { "" } else { "s" };
            if all_same_key {
                format!("a union of {n} member{plural} on `{}` that is not all-numeric", keys[0])
            } else {
                format!("a union of {n} member{plural} spanning more than one attribute")
            }
        }
        Q::And(children) => format!("a conjunction of {} predicates", children.len()),
        Q::InList { key, .. } => format!("an IN list on `{key}`"),
        Q::Point { key, .. } => format!("a geo predicate on `{key}`"),
        Q::ArrayContains { key, .. } => format!("an array-contains on `{key}`"),
    }
}

/// Collect the attributes a (possibly nested) union's `Equal` members target. `false` when a
/// member is neither an `Equal` nor a nested union, in which case the keys collected so far say
/// nothing about the shape.
#[cfg(feature = "index-falkordb")]
fn union_keys<'a>(
    children: &'a [crate::index::IndexQuery<crate::runtime::value::Value>],
    out: &mut Vec<&'a std::sync::Arc<String>>,
) -> bool {
    use crate::index::IndexQuery as Q;
    children.iter().all(|child| match child {
        Q::Equal { key, .. } => {
            out.push(key);
            true
        }
        Q::Or(nested) => union_keys(nested, out),
        _ => false,
    })
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

    /// The ledger's whole value is that the message names the gap. A union is declined for two
    /// different reasons and they need different work to close — a missing index kind, or a real
    /// cross-column merge — so reporting one as the other sends the reader after the wrong thing.
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
    }
}
