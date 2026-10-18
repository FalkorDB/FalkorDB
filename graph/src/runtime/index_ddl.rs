//! Index DDL — `CREATE INDEX` and `DROP INDEX`.
//!
//! Not an operator, which is why this sits beside `runtime` rather than under
//! `ops`: DDL is an immediate side effect with no rows to pull, so there is
//! nothing to iterate and no `BatchOp` variant for it. These are plain functions
//! the runtime's dispatcher delegates to — out of that dispatcher because
//! effects emission belongs beside the mutation that caused it.
//!
//! Unlike every other mutation, index DDL does not go through `Pending`: it
//! writes shared, non-MVCC RediSearch state directly. So it also cannot ride
//! `CommitOp`'s effects emission, and emits its own.

use std::sync::Arc;

use crate::parser::ast::{QueryExpr, Variable};
use crate::{
    effects::{EffectsBuffer, announce::AnnouncedIndex},
    entity_type::EntityType,
    index::{
        cch::{CchError, CchIndexDef},
        indexer::IndexType,
    },
    runtime::{
        batch::BatchOp,
        eval::{ExprEval, NO_ROW},
        runtime::{Runtime, cch_error_message, map_to_index_options},
        value::Value,
    },
};
use orx_tree::NodeRef;

/// `CREATE INDEX`, with its `OPTIONS {...}` map if it carried one.
pub(crate) fn create_index<'a>(
    runtime: &'a Runtime<'a>,
    label: &Arc<String>,
    attrs: &Vec<Arc<String>>,
    index_type: &IndexType,
    entity_type: &EntityType,
    options: Option<&QueryExpr<Variable>>,
) -> Result<BatchOp<'a>, String> {
    if !runtime.write {
        return Err(String::from(
            "graph.RO_QUERY is to be executed only on read-only queries",
        ));
    }
    // The evaluated map is kept, not just the parsed options: it is what goes on
    // the wire for v3, and this is the only place it exists. A plan scan cannot
    // rebuild it — the expression has to be evaluated to know what it says.
    let options_value = match options {
        Some(expr) => {
            let idx = expr.root().idx();
            let val = ExprEval::from_runtime(runtime).eval(expr, idx, NO_ROW, None)?;
            if !matches!(val, Value::Map(_)) {
                return Err("Index options must be a map".into());
            }
            Some(val)
        }
        None => None,
    };
    let index_options = match &options_value {
        Some(Value::Map(map)) => map_to_index_options(index_type, map)?,
        _ => None,
    };
    // A vector index needs its dimension and similarity function (C rejects a
    // missing OPTIONS block the same way).
    if *index_type == IndexType::Vector && index_options.is_none() {
        return Err("Invalid vector index configuration".into());
    }
    // Index DDL mutates the shared, non-MVCC index directly (not via `pending`)
    // and calls host FFI that needs the global lock, so become a writer first —
    // same contract as `CommitOp`.
    runtime.write_escalation().upgrade_to_write()?;
    runtime
        .g
        .borrow_mut()
        .create_index(index_type, entity_type, label, attrs, index_options)?;
    runtime.stats.borrow_mut().indexes_created += attrs.len();
    emit_effect(
        runtime,
        true,
        label,
        attrs,
        index_type,
        entity_type,
        options_value.as_ref(),
    )?;
    Ok(BatchOp::Once(None))
}

/// `DROP INDEX`.
pub(crate) fn drop_index<'a>(
    runtime: &'a Runtime<'a>,
    label: &Arc<String>,
    attrs: &Vec<Arc<String>>,
    index_type: &IndexType,
    entity_type: &EntityType,
) -> Result<BatchOp<'a>, String> {
    if !runtime.write {
        return Err(String::from(
            "graph.RO_QUERY is to be executed only on read-only queries",
        ));
    }
    // See `create_index`: DDL runs in writer mode.
    runtime.write_escalation().upgrade_to_write()?;
    let dropped = runtime
        .g
        .borrow_mut()
        .drop_index(index_type, entity_type, label, attrs)?;
    runtime.stats.borrow_mut().indexes_dropped += dropped;
    emit_effect(runtime, false, label, attrs, index_type, entity_type, None)?;
    Ok(BatchOp::Once(None))
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
    runtime.check_cch_ddl_allowed()?;
    // Publishing this version depends on being a writer (`commit_and_replicate`
    // commits whenever the session is in writer mode), not on the stats.
    runtime.write_escalation().upgrade_to_write()?;
    {
        let mut g = runtime.g.borrow_mut();
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
    // and shows "Indexes created" in the reply. Replicas and the AOF get the index only through
    // an RDB until CCH DDL emits effects.
    runtime.stats.borrow_mut().indexes_created += 1;
    Ok(BatchOp::Once(None))
}

/// `DROP CCH INDEX`. Resolves only: everything must already exist to match a live index.
pub(crate) fn drop_cch_index<'a>(
    runtime: &'a Runtime<'a>,
    rel_types: &[Arc<String>],
    weight_attr: &Arc<String>,
) -> Result<BatchOp<'a>, String> {
    runtime.check_cch_ddl_allowed()?;
    runtime.write_escalation().upgrade_to_write()?;
    {
        let mut g = runtime.g.borrow_mut();
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

/// Append this statement to the query's effects buffer.
///
/// Index DDL does not go through `Pending`, so it cannot ride `CommitOp`'s
/// emission and appends here instead — in plan order, from the one place the
/// evaluated `OPTIONS` map exists.
fn emit_effect(
    runtime: &Runtime<'_>,
    create: bool,
    label: &Arc<String>,
    attrs: &Vec<Arc<String>>,
    index_type: &IndexType,
    entity_type: &EntityType,
    options: Option<&Value>,
) -> Result<(), String> {
    if !runtime.build_effects.get() {
        return Ok(());
    }
    let mut buf_ref = runtime.effects_buffer.borrow_mut();
    let buf = buf_ref.get_or_insert_with(EffectsBuffer::new);
    buf.build_index(
        &runtime.pending.borrow(),
        &runtime.g,
        create,
        &AnnouncedIndex {
            entity_type: *entity_type,
            index_type,
            label,
            fields: attrs,
            options,
        },
    )?;
    runtime.effects_count.set(runtime.effects_count.get() + 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    //! CCH DDL through real Cypher on an `MvccGraph` write slot. That a replica keeps applying
    //! writes after it is covered end to end by the effects flow suite (`test_effects_ddl.py`).

    use std::collections::HashMap;
    use std::sync::atomic::Ordering;

    use crate::graph::{graph::Graph, graphblas::test_init, mvcc_graph::MvccGraph};
    use crate::locks::WriteEscalation;
    use crate::runtime::{
        functions::{init_functions, init_udf_functions},
        runtime::{CCH_DDL_ENABLED, Runtime},
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
            CCH_DDL_ENABLED.store(true, Ordering::Relaxed);
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
