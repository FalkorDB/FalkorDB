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
pub fn cch_error_message(e: &CchError) -> String {
    match e {
        CchError::AlreadyExists => String::from(
            "a CCH index already exists over these relationship types and weight attribute",
        ),
        CchError::NotFound => {
            String::from("no CCH index over these relationship types and weight attribute")
        }
        CchError::UnknownRelType => String::from("CCH index error: unknown relationship type"),
        CchError::UnknownAttribute => String::from("CCH index error: unknown weight attribute"),
        CchError::GraphTooLarge => {
            String::from("CCH index error: graph is too large for a CCH index")
        }
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

#[cfg(test)]
mod tests {
    //! CCH DDL through real Cypher on an `MvccGraph` write slot. That a replica keeps applying
    //! writes after it is covered end to end by the effects flow suite (`test_effects_ddl.py`).

    use std::collections::HashMap;

    use crate::graph::{graph::Graph, graphblas::test_init, mvcc_graph::MvccGraph};
    use crate::locks::WriteEscalation;
    use crate::runtime::{
        functions::{init_functions, init_udf_functions},
        runtime::Runtime,
    };

    struct NoLocks;

    impl WriteEscalation for NoLocks {
        fn upgrade_to_write(&self) -> Result<(), String> {
            Ok(())
        }
    }

    struct Db(MvccGraph);

    impl Db {
        fn new(name: &str) -> Self {
            test_init::ensure_init();
            let _ = init_functions();
            init_udf_functions();
            Self(MvccGraph::from_graph(Graph::new(16, 16, 1, 0, name)))
        }

        /// One write query, committed on success and rolled back on error.
        fn write(
            &mut self,
            query: &str,
        ) -> Result<(), String> {
            let plan = self.0.read().borrow().get_plan(query)?.plan;
            let next = self.0.write().expect("the write slot is free");
            let outcome = Runtime::new(
                next.clone(),
                HashMap::new(),
                true,
                plan,
                false,
                String::new(),
                -1,
                false,
                None,
                0,
                None,
                &NoLocks,
            )
            .query()
            .map(|_| ());
            match outcome {
                Ok(()) => self.0.commit(next).expect("the version validates"),
                Err(_) => self.0.rollback(),
            }
            outcome
        }

        /// Relationship types, relationship attribute names and CCH index count.
        fn schema(&self) -> (Vec<String>, Vec<String>, usize) {
            let g = self.0.read();
            let g = g.borrow();
            (
                g.get_types().iter().map(|t| t.to_string()).collect(),
                g.get_relationship_attribute_names()
                    .iter()
                    .map(|a| a.to_string())
                    .collect(),
                g.cch_indexes().count(),
            )
        }
    }

    /// CCH DDL is not replicated yet, so it must never create schema: a type or attribute it
    /// created would be unknown to a replica, which then fails the next write naming it.
    #[test]
    fn cch_ddl_refuses_to_create_schema_and_works_over_existing_schema() {
        let mut db = Db::new("cch_ddl_existing_schema_only");

        let err = db
            .write("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
            .expect_err("an unknown type is refused");
        assert!(
            err.contains("relationship type 'ROAD' does not exist") && err.contains("first"),
            "{err}"
        );
        assert_eq!(db.schema(), (vec![], vec![], 0), "nothing was created");

        db.write("CREATE (:N)-[:ROAD {time: 1.0}]->(:N)").unwrap();

        let err = db
            .write("CREATE CCH INDEX FOR ()-[e:ROAD|FERRY]->() ON (e.time)")
            .expect_err("one unknown type is enough");
        assert!(err.contains("'FERRY' does not exist"), "{err}");
        let err = db
            .write("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.length)")
            .expect_err("an unknown attribute is refused");
        assert!(
            err.contains("attribute 'length' does not exist") && err.contains("first"),
            "{err}"
        );
        assert_eq!(
            db.schema(),
            (vec!["ROAD".into()], vec!["time".into()], 0),
            "nothing was created"
        );

        db.write("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
            .expect("existing schema is indexed");
        assert_eq!(db.schema(), (vec!["ROAD".into()], vec!["time".into()], 1));
    }
}
