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
        Q::Or(children) => format!(
            "a union of {} members that is not all-numeric",
            children.len()
        ),
        Q::And(children) => format!("a conjunction of {} predicates", children.len()),
        Q::InList { key, .. } => format!("an IN list on `{key}`"),
        Q::Point { key, .. } => format!("a geo predicate on `{key}`"),
        Q::ArrayContains { key, .. } => format!("an array-contains on `{key}`"),
    }
}
