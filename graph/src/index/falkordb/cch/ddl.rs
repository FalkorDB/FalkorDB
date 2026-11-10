//! `CREATE CCH INDEX` and `DROP CCH INDEX`: the runtime side of the DDL.
//!
//! The parser, binder and planner handle the statements in every build; only executing them
//! needs this module, so a build without the `index-falkordb` feature reports a clear error from
//! the runtime instead of a syntax error. A build with it always accepts them: the feature is
//! the switch.

use std::sync::Arc;

use super::{CchError, CchIndexDef};
use crate::{
    graph::graph::Graph,
    parser::ast::CchIndexSpec,
    runtime::{batch::BatchOp, index_ddl::reject_read_only_query, runtime::Runtime},
};

impl CchError {
    /// The query error for a failed CCH create/drop, worded like C's where C has a message.
    ///
    /// Not the `Display` text: that is the error's own wording, and the C wording differs
    /// (`AlreadyExists`/`NotFound` name the definition, `Internal` drops "internal").
    #[must_use]
    fn query_message(&self) -> String {
        match self {
            Self::AlreadyExists => String::from(
                "a CCH index already exists over these relationship types and weight attribute",
            ),
            Self::NotFound => {
                String::from("no CCH index over these relationship types and weight attribute")
            }
            Self::UnknownRelType => String::from("CCH index error: unknown relationship type"),
            Self::UnknownAttribute => String::from("CCH index error: unknown weight attribute"),
            Self::InvalidWeight => String::from(
                "CCH index error: edge weights must be >= 0 (+inf marks a closed edge)",
            ),
            Self::NeedsRebuild => format!("CCH index error: {self}"),
            Self::Internal(message) => format!("CCH index error: {message}"),
            Self::CsrExport(error) => format!("CCH index error: CSR export: {error}"),
            Self::Ordering(error) => format!("CCH index error: {error}"),
        }
    }
}

/// Why a [`CchIndexSpec`] names no definition in this graph.
enum Unresolved<'a> {
    /// No relationship type has this name.
    RelType(&'a Arc<String>),
    /// No attribute has the weight attribute's name.
    Attribute,
    /// A bug, not bad input.
    Internal(CchError),
}

/// The definition `spec` names, resolving names to ids without creating any.
fn resolve_cch_index<'a>(
    graph: &Graph,
    spec: &'a CchIndexSpec,
) -> Result<CchIndexDef, Unresolved<'a>> {
    let rel_types = spec
        .rel_types
        .iter()
        .map(|name| graph.get_type_id(name).ok_or(Unresolved::RelType(name)))
        .collect::<Result<_, _>>()?;
    let attribute_id = graph
        .get_relationship_attribute_id(&spec.weight_attr)
        .ok_or(Unresolved::Attribute)?;
    // Attribute ids stay below `MAX_ATTRIBUTES`, which fits a `u16`.
    let weight_attr = u16::try_from(attribute_id).map_err(|_| {
        Unresolved::Internal(CchError::Internal(
            "the weight attribute's id does not fit in u16",
        ))
    })?;
    Ok(CchIndexDef::new(rel_types, weight_attr))
}

/// `CREATE CCH INDEX`.
///
/// Unlike `CREATE INDEX` (and C), the relationship types and the weight attribute must already
/// exist: this statement never creates schema. CCH DDL emits no effects yet, so a type or
/// attribute created here would never reach a replica or the AOF, and the next write naming it
/// would carry an id the replica cannot resolve. Requiring existing schema keeps every id a
/// replica sees one an earlier, replicated write announced. Proper replication (C-compatible
/// CREATE_CCH/DROP_CCH effects, preceded by the schema effects they need) replaces this check.
pub(crate) fn create_cch_index<'a>(
    runtime: &'a Runtime<'a>,
    spec: &CchIndexSpec,
) -> Result<BatchOp<'a>, String> {
    reject_read_only_query(runtime)?;
    // Publishing this version depends on being a writer (`commit_and_replicate`
    // commits whenever the session is in writer mode), not on the stats.
    runtime.upgrade_to_write()?;
    {
        let mut graph = runtime.graph_mut();
        let definition =
            resolve_cch_index(&graph, spec).map_err(|unresolved| match unresolved {
                Unresolved::RelType(name) => format!(
                    "CCH index error: relationship type '{name}' does not exist; create it \
                     first (create a relationship of that type), then create the index"
                ),
                Unresolved::Attribute => format!(
                    "CCH index error: attribute '{}' does not exist; create it \
                     first (set it on a relationship), then create the index",
                    spec.weight_attr
                ),
                Unresolved::Internal(error) => error.query_message(),
            })?;
        graph
            .create_cch_index(definition)
            .map_err(|error| error.query_message())?;
    }
    // Not replicated: the bump makes the write count as modified, so `commit_and_replicate` logs
    // "modified but produced no effects payload" (there is no verbatim-query fallback on main)
    // and shows "Indexes created" in the reply. Replicas and the AOF do not get the index: no
    // effect carries it, and the RDB does not save CCH indexes yet.
    runtime.stats.borrow_mut().indexes_created += 1;
    Ok(BatchOp::Once(None))
}

/// `DROP CCH INDEX`. Resolves only: everything must already exist to match a live index.
pub(crate) fn drop_cch_index<'a>(
    runtime: &'a Runtime<'a>,
    spec: &CchIndexSpec,
) -> Result<BatchOp<'a>, String> {
    reject_read_only_query(runtime)?;
    runtime.upgrade_to_write()?;
    {
        let mut graph = runtime.graph_mut();
        let definition =
            resolve_cch_index(&graph, spec).map_err(|unresolved| match unresolved {
                Unresolved::RelType(_) | Unresolved::Attribute => {
                    CchError::NotFound.query_message()
                }
                Unresolved::Internal(error) => error.query_message(),
            })?;
        graph
            .drop_cch_index(&definition)
            .map_err(|error| error.query_message())?;
    }
    // Not replicated either; see `create_cch_index`.
    runtime.stats.borrow_mut().indexes_dropped += 1;
    Ok(BatchOp::Once(None))
}
