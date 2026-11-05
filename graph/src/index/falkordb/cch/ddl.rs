//! `CREATE CCH INDEX` and `DROP CCH INDEX`: the runtime side of the DDL.
//!
//! The parser, binder and planner handle the statements in every build; only executing them
//! needs this module, so a build without the `index-falkordb` feature reports a clear error from
//! the runtime instead of a syntax error. A build with it always accepts them: the feature is
//! the switch.

use std::sync::Arc;

use super::{CchError, CchIndexDef};
use crate::runtime::{batch::BatchOp, runtime::Runtime};

/// The query error for a failed CCH create/drop, worded like C's where C has a message.
#[must_use]
pub(super) fn cch_error_message(e: &CchError) -> String {
    match e {
        CchError::AlreadyExists => String::from(
            "a CCH index already exists over these relationship types and weight attribute",
        ),
        CchError::NotFound => {
            String::from("no CCH index over these relationship types and weight attribute")
        }
        CchError::UnknownRelType => String::from("CCH index error: unknown relationship type"),
        CchError::UnknownAttribute => String::from("CCH index error: unknown weight attribute"),
        CchError::InvalidWeight => {
            String::from("CCH index error: edge weights must be >= 0 (+inf marks a closed edge)")
        }
        CchError::Internal(msg) => format!("CCH index error: {msg}"),
    }
}

fn check_ddl_allowed(runtime: &Runtime<'_>) -> Result<(), String> {
    if !runtime.write {
        return Err(String::from(
            "graph.RO_QUERY is to be executed only on read-only queries",
        ));
    }
    Ok(())
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
    rel_types: &[Arc<String>],
    weight_attr: &Arc<String>,
) -> Result<BatchOp<'a>, String> {
    check_ddl_allowed(runtime)?;
    // Publishing this version depends on being a writer (`commit_and_replicate`
    // commits whenever the session is in writer mode), not on the stats.
    runtime.write_escalation().upgrade_to_write()?;
    {
        let mut g = runtime.graph_mut();
        let types = rel_types
            .iter()
            .map(|t| {
                g.get_type_id(t).ok_or_else(|| {
                    format!(
                        "CCH index error: relationship type '{t}' does not exist; create it \
                         first (create a relationship of that type), then create the index"
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        let attr = g
            .get_relationship_attribute_id(weight_attr)
            .and_then(|a| u16::try_from(a).ok())
            .ok_or_else(|| {
                format!(
                    "CCH index error: attribute '{weight_attr}' does not exist; create it \
                     first (set it on a relationship), then create the index"
                )
            })?;
        g.create_cch_index(CchIndexDef::new(types, attr))
            .map_err(|e| cch_error_message(&e))?;
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
    rel_types: &[Arc<String>],
    weight_attr: &Arc<String>,
) -> Result<BatchOp<'a>, String> {
    check_ddl_allowed(runtime)?;
    runtime.write_escalation().upgrade_to_write()?;
    {
        let mut g = runtime.graph_mut();
        let types: Option<Vec<_>> = rel_types.iter().map(|t| g.get_type_id(t)).collect();
        let attr = g
            .get_relationship_attribute_id(weight_attr)
            .and_then(|a| u16::try_from(a).ok());
        let (Some(types), Some(attr)) = (types, attr) else {
            return Err(cch_error_message(&CchError::NotFound));
        };
        g.drop_cch_index(&CchIndexDef::new(types, attr))
            .map_err(|e| cch_error_message(&e))?;
    }
    // Not replicated either; see `create_cch_index`.
    runtime.stats.borrow_mut().indexes_dropped += 1;
    Ok(BatchOp::Once(None))
}
