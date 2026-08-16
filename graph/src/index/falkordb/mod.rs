//! The FalkorDB index (the in-repo replacement for RediSearch).
//!
//! [`data_structures`] is the always-compiled substrate (the CoW B⁺-tree). The
//! index proper — the numeric key [`encode`]r, the tree-backed index, and its
//! runtime wiring — lives behind the `index-falkordb` feature, off by default.

pub mod data_structures;

#[cfg(feature = "index-falkordb")]
pub mod build_registry;

#[cfg(feature = "index-falkordb")]
pub mod doc_iter;

#[cfg(feature = "index-falkordb")]
pub mod encode;

#[cfg(feature = "index-falkordb")]
pub mod falkordb_index;

#[cfg(feature = "index-falkordb")]
pub mod geo;

#[cfg(feature = "index-falkordb")]
pub mod numeric;

#[cfg(feature = "index-falkordb")]
pub mod range;

#[cfg(feature = "index-falkordb")]
pub mod tag;

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
        // Name the operand's type rather than calling it "non-numeric": numbers, strings and
        // points all have a kind now, so what is left is a type no kind holds (a list, a map, a
        // vector). At top level a declined `Equal` is necessarily one of those, but the same arm
        // is reached through `And`/`Or`, where the child may be perfectly servable and the
        // *combination* is what is not — reporting the child as unindexable there sends the reader
        // after a missing value kind that is not the problem.
        Q::Equal { key, value } => match indexable_type(value) {
            true => format!("equality on `{key}`"),
            false => format!("equality on `{key}` with a {} value", value.name()),
        },
        Q::Range { key, min, max, .. } => {
            let unindexable = [min, max]
                .into_iter()
                .flatten()
                .find(|v| !indexable_type(v));
            match unindexable {
                None => format!("range on `{key}`"),
                Some(v) => format!("range on `{key}` with a {} bound", v.name()),
            }
        }
        // Two reasons a union is declined, and naming the wrong one sends whoever reads the
        // ledger entry looking for a missing value kind that isn't the problem. A union spanning
        // attributes needs a real cross-column merge; one with an unindexable member needs a kind
        // that can hold that type. A union that has neither problem is servable on its own and is
        // only being described because it sits inside something that is not — so it must not be
        // reported as broken.
        Q::Or(children) => {
            let mut keys = Vec::new();
            let all_same_key = union_keys(children, &mut keys)
                && keys.first().is_some_and(|f| keys.iter().all(|k| k == f));
            let n = children.len();
            let plural = if n == 1 { "" } else { "s" };
            if !all_same_key {
                return format!("a union of {n} member{plural} spanning more than one attribute");
            }
            match union_unindexable(children) {
                Some(v) => format!(
                    "a union of {n} member{plural} on `{}` including a {} value",
                    keys[0],
                    v.name()
                ),
                None => format!("a union of {n} member{plural} on `{}`", keys[0]),
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
        // The array trees hold numbers and strings, mirroring RediSearch's `numeric:arr` and
        // `string:arr` sub-fields. Nothing indexes a point (or a nested list) inside a list, so
        // that probe names its own type.
        Q::ArrayContains { key, value } => {
            if matches!(
                value,
                crate::runtime::value::Value::Int(_)
                    | crate::runtime::value::Value::Float(_)
                    | crate::runtime::value::Value::Bool(_)
                    | crate::runtime::value::Value::String(_)
            ) {
                format!("an array-contains on `{key}`")
            } else {
                format!("an array-contains on `{key}` with a {} value", value.name())
            }
        }
    }
}

/// The first member of a (possibly nested) union whose value no kind can hold — the thing that
/// makes the union itself unservable, as opposed to its context.
#[cfg(feature = "index-falkordb")]
fn union_unindexable(
    children: &[crate::index::IndexQuery<crate::runtime::value::Value>]
) -> Option<&crate::runtime::value::Value> {
    use crate::index::IndexQuery as Q;
    children.iter().find_map(|child| match child {
        // A `NULL` member is not a gap: it matches nothing and drops out of the union.
        Q::Equal { value, .. }
            if !indexable_type(value) && !matches!(value, crate::runtime::value::Value::Null) =>
        {
            Some(value)
        }
        Q::Or(nested) => union_unindexable(nested),
        _ => None,
    })
}

/// Whether some kind in a Range column can hold a value of this type: numbers (including the
/// temporals and booleans that coerce to one), strings, and points. `NULL`, lists, maps and
/// vectors have no kind — a predicate over one is what a decline names.
#[cfg(feature = "index-falkordb")]
fn indexable_type(value: &crate::runtime::value::Value) -> bool {
    use crate::runtime::value::Value as V;
    matches!(
        value,
        V::Int(_)
            | V::Float(_)
            | V::Bool(_)
            | V::Datetime(_)
            | V::Date(_)
            | V::Time(_)
            | V::Duration(_)
            | V::String(_)
            | V::Point(_)
    )
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
        // A member of a type no kind holds. A *string* member no longer belongs here: the tag
        // kind holds those, and `n.v IN [1, 'a']` is served from both kinds at once.
        let unindexable = Q::Or(vec![
            eq("a", 1),
            Q::Equal {
                key: Arc::new("a".to_string()),
                value: Value::List(Arc::new(Default::default())),
            },
        ]);
        let msg = describe_predicate(&unindexable);
        assert!(msg.contains("including a List value"), "{msg}");
        assert!(msg.contains('a'), "names the attribute: {msg}");

        // A union of servable members is described as what it is. It only shows up in a message at
        // all because something *around* it was declined — saying it holds an unindexable value
        // would send the reader hunting for a missing kind that is not missing.
        let servable = Q::Or(vec![
            eq("a", 1),
            Q::Equal {
                key: Arc::new("a".to_string()),
                value: Value::String(Arc::new("x".to_string())),
            },
        ]);
        let msg = describe_predicate(&servable);
        assert_eq!(msg, "a union of 2 members on `a`", "{msg}");

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
