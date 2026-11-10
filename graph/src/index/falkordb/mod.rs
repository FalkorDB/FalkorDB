//! The FalkorDB index (the in-repo replacement for RediSearch).
//!
//! [`data_structures`] is the always-compiled substrate (the CoW B⁺-tree). The
//! index proper — the [`range`] column with its numeric ([`encode`]) and [`tag`] kinds,
//! and the hooks that turn graph writes into index changes — lives behind the
//! `index-falkordb` feature, off by default. So does the [`cch`] shortest-path index, whose
//! state is a field of the same [`FalkorDbIndex`](falkordb_index::FalkorDbIndex).

pub mod data_structures;

#[cfg(feature = "index-falkordb")]
pub mod cch;

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
pub mod range;

#[cfg(feature = "index-falkordb")]
pub mod tag;

#[cfg(feature = "index-falkordb")]
pub mod unique;

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
        // Name the operand's type rather than calling it "non-numeric": numbers and strings both
        // have a kind, so what is left is a type no kind holds (a point, a list, a map, a
        // temporal). At top level a declined `Equal` is necessarily one of those, but the same arm
        // is reached through `And`/`Or`, where the child may be perfectly servable and the
        // *combination* is what is not — reporting the child as unindexable there sends the reader
        // after a missing value kind that is not the problem. A `NULL` operand is not one either:
        // it is answered with no rows, never declined.
        Q::Equal { key, value } => {
            if indexable(value) || matches!(value, crate::runtime::value::Value::Null) {
                format!("equality on `{key}`")
            } else {
                format!("equality on `{key}` with a {} value", value.name())
            }
        }
        Q::Range { key, min, max, .. } => {
            match [min, max]
                .into_iter()
                .flatten()
                .find(|v| !indexable(v) && !matches!(v, crate::runtime::value::Value::Null))
            {
                None => format!("range on `{key}`"),
                Some(v) => format!("range on `{key}` with a {} bound", v.name()),
            }
        }
        // Three reasons a union is declined, and naming the wrong one sends whoever reads the
        // ledger entry after work that isn't the problem. A union with a member that is not an
        // equality or `IN` list (`x = 1 OR x > 5`) needs the union to serve that member kind; one
        // with a value of a type no kind holds needs that kind; a union spanning attributes is
        // served one column per attribute, so it is declined for one of the other two reasons in a
        // member, or for a missing column.
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
                    match union_unindexable(children) {
                        Some(v) => format!(
                            "a union of {n} member{plural} on `{key}` including a {} value",
                            v.name()
                        ),
                        // Servable on its own: only described because something around it is not.
                        None => format!("a union of {n} member{plural} on `{key}`"),
                    }
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
            let unindexable = match list {
                crate::runtime::value::Value::List(items) => items
                    .iter()
                    .find(|v| !indexable(v) && !matches!(v, crate::runtime::value::Value::Null)),
                other => Some(other),
            };
            match unindexable {
                None => format!("an IN list on `{key}`"),
                Some(v) => format!("an IN list on `{key}` with a {} value", v.name()),
            }
        }
        Q::Point { key, .. } => format!("a geo predicate on `{key}`"),
        // The array trees hold numbers and strings, mirroring RediSearch's `numeric:arr` and
        // `string:arr` sub-fields. Nothing indexes a point (or a nested list) inside a list, so
        // that probe names its own type.
        Q::ArrayContains { key, value } => {
            use crate::runtime::value::Value as V;
            if matches!(value, V::Int(_) | V::Float(_) | V::Bool(_) | V::String(_)) {
                format!("an array-contains on `{key}`")
            } else {
                format!("an array-contains on `{key}` with a {} value", value.name())
            }
        }
    }
}

/// Whether some kind in a Range column holds a value of this type: numbers (and the booleans
/// that coerce to one) and strings. `NULL`, points, temporals, lists, maps and vectors have no
/// kind — a predicate over one is what a decline names.
#[cfg(feature = "index-falkordb")]
fn indexable(value: &crate::runtime::value::Value) -> bool {
    use crate::runtime::value::Value as V;
    matches!(value, V::Int(_) | V::Float(_) | V::Bool(_) | V::String(_))
}

/// The first value in a (possibly nested) union of equalities and `IN` lists whose type no kind
/// holds — what makes the union itself unservable, as opposed to its context. A `NULL` is not one:
/// it matches nothing and drops out of the union.
#[cfg(feature = "index-falkordb")]
fn union_unindexable(
    children: &[crate::index::IndexQuery<crate::runtime::value::Value>]
) -> Option<&crate::runtime::value::Value> {
    use crate::index::IndexQuery as Q;
    use crate::runtime::value::Value as V;
    let unindexable = |v: &&V| !indexable(v) && !matches!(v, V::Null);
    children.iter().find_map(|child| match child {
        Q::Equal { value, .. } => Some(value).filter(unindexable),
        Q::InList {
            list: V::List(items),
            ..
        } => items.iter().find(unindexable),
        Q::Or(nested) => union_unindexable(nested),
        _ => None,
    })
}

/// Collect the attribute of every member of a union, paired with whether it is a member kind the
/// native union serves — an `Equal` or an `IN` list — directly in the union. The members of a
/// nested union or conjunction count by their attributes but not as served (`top_level` is `false`
/// for them).
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

    fn list() -> Value {
        Value::List(Arc::new(vec![Value::Int(1)].into_iter().collect()))
    }

    /// `IN` reaches the index whole, so its message tests the values rather than assuming them,
    /// and names the type no kind holds. A string has a kind now, so it is not one.
    #[test]
    fn a_declined_in_list_names_the_unindexable_type() {
        let unindexable = in_list("a", vec![Value::Int(1), list()]);
        assert_eq!(
            describe_predicate(&unindexable),
            "an IN list on `a` with a List value"
        );
        let mixed = in_list(
            "a",
            vec![Value::Int(1), Value::String(Arc::new("x".to_string()))],
        );
        assert_eq!(describe_predicate(&mixed), "an IN list on `a`");
    }

    /// The ledger's whole value is that the message names the gap. A union is declined for three
    /// different reasons and they need different work to close — a missing index kind, a member
    /// kind the union does not serve, or a real cross-column merge — so reporting one as another
    /// sends the reader after the wrong thing.
    /// A `NULL` operand is answered with no rows, never declined, so a message about a
    /// combination it sits in does not name it as the unindexable value.
    #[test]
    fn a_null_operand_is_not_named_as_the_gap() {
        let null_eq = Q::Equal {
            key: Arc::new("a".to_string()),
            value: Value::Null,
        };
        assert_eq!(describe_predicate(&null_eq), "equality on `a`");
        let null_range = Q::Range {
            key: Arc::new("a".to_string()),
            min: Some(Value::Null),
            max: None,
            include_min: true,
            include_max: false,
        };
        assert_eq!(describe_predicate(&null_range), "range on `a`");
    }

    #[test]
    fn a_declined_union_says_which_reason_applies() {
        // A member of a type no kind holds.
        let unindexable = Q::Or(vec![
            eq("a", 1),
            Q::Equal {
                key: Arc::new("a".to_string()),
                value: list(),
            },
        ]);
        let msg = describe_predicate(&unindexable);
        assert!(msg.contains("including a List value"), "{msg}");
        assert!(msg.contains("`a`"), "names the attribute: {msg}");

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
        assert_eq!(describe_predicate(&servable), "a union of 2 members on `a`");

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

        // An IN list is a member kind the union serves; its list value is what declines it.
        let mixed_in = Q::Or(vec![in_list("a", vec![Value::Int(1), list()]), eq("a", 2)]);
        let msg = describe_predicate(&mixed_in);
        assert!(msg.contains("including a List value"), "{msg}");

        // A range on a second attribute still makes it a cross-attribute union.
        let range_cross = Q::Or(vec![eq("a", 1), gt("b", 5)]);
        assert!(
            describe_predicate(&range_cross).contains("more than one attribute"),
            "{}",
            describe_predicate(&range_cross)
        );
    }
}
