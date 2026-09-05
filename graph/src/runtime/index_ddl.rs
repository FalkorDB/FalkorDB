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
    effects::{EffectsPayload, announce::AnnouncedIndex},
    entity_type::EntityType,
    index::indexer::IndexType,
    runtime::{
        batch::BatchOp,
        eval::{ExprEval, NO_ROW},
        runtime::{Runtime, map_to_index_options},
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
    let buf = buf_ref.get_or_insert_with(Vec::new);
    EffectsPayload::build_index(
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
        buf,
    )?;
    runtime.effects_count.set(runtime.effects_count.get() + 1);
    Ok(())
}
