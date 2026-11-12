//! Index DDL — `CREATE INDEX` and `DROP INDEX`, and with the `index-falkordb`
//! feature `CREATE CCH INDEX` and `DROP CCH INDEX`.
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
#[cfg(feature = "index-falkordb")]
use crate::{
    effects::announce::AnnouncedCchIndex,
    graph::attribute_store::{ATTRIBUTE_ID_NONE, MAX_ATTRIBUTES},
    graph::graph::Graph,
    index::falkordb::cch::{CchError, CchIndexDef},
    parser::ast::CchIndexSpec,
};
use crate::{
    effects::{EffectsBuffer, announce::AnnouncedIndex},
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
    reject_read_only_query(runtime)?;
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
    runtime.upgrade_to_write()?;
    runtime
        .graph_mut()
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
    reject_read_only_query(runtime)?;
    // See `create_index`: DDL runs in writer mode.
    runtime.upgrade_to_write()?;
    let dropped = runtime
        .graph_mut()
        .drop_index(index_type, entity_type, label, attrs)?;
    runtime.stats.borrow_mut().indexes_dropped += dropped;
    emit_effect(runtime, false, label, attrs, index_type, entity_type, None)?;
    Ok(BatchOp::Once(None))
}

/// Index DDL is a write: refuse it in a `GRAPH.RO_QUERY`, with C's message.
fn reject_read_only_query(runtime: &Runtime<'_>) -> Result<(), String> {
    if runtime.write {
        Ok(())
    } else {
        Err(String::from(
            "graph.RO_QUERY is to be executed only on read-only queries",
        ))
    }
}

/// Refuse a CCH statement naming more relationship types than its index record can carry.
///
/// The record keeps every reference as written, repeats included, behind a `u16` count (C's
/// layout), so `:ROAD|ROAD|…` 65,536 times cannot be replicated even though it defines a
/// one-type index. Checked first and whether or not this query builds effects, so a
/// statement is accepted or refused the same way with or without AOF and replicas, and
/// before anything is registered.
#[cfg(feature = "index-falkordb")]
fn check_type_reference_count(spec: &CchIndexSpec) -> Result<(), String> {
    let count = spec.rel_types.len();
    if count > usize::from(u16::MAX) {
        return Err(format!(
            "CCH index error: a statement names at most {} relationship types, repeats \
             included, but this one names {count}",
            u16::MAX
        ));
    }
    Ok(())
}

/// The CCH index definition `spec` names, registering any relationship type or
/// weight attribute it does not find — as C's `cch_index_create` does, and as
/// `CREATE INDEX` does. The registrations reach a replica as `ADD_SCHEMA` /
/// `ADD_ATTRIBUTE` ahead of the index record ([`emit_cch_effect`]).
///
/// # Errors
/// The attribute dictionary is full, so the weight attribute cannot be registered.
#[cfg(feature = "index-falkordb")]
fn resolve_or_create_cch_index(
    graph: &mut Graph,
    spec: &CchIndexSpec,
) -> Result<CchIndexDef, String> {
    let rel_types = spec
        .rel_types
        .iter()
        .map(|name| graph.get_type_id_mut(name))
        .collect();
    let weight_attr = graph.get_or_create_rel_attr_id(&spec.weight_attr);
    if weight_attr == ATTRIBUTE_ID_NONE {
        return Err(format!(
            "CCH index error: the attribute dictionary is full ({MAX_ATTRIBUTES} names), so \
             weight attribute '{}' cannot be registered",
            spec.weight_attr
        ));
    }
    Ok(CchIndexDef::new(rel_types, weight_attr))
}

/// The CCH index definition `spec` names, or `None` if a relationship type or the
/// weight attribute does not exist: then no index can match it.
#[cfg(feature = "index-falkordb")]
fn resolve_cch_index(
    graph: &Graph,
    spec: &CchIndexSpec,
) -> Option<CchIndexDef> {
    let rel_types = spec
        .rel_types
        .iter()
        .map(|name| graph.get_type_id(name))
        .collect::<Option<_>>()?;
    let weight_attr = graph
        .get_relationship_attribute_id(&spec.weight_attr)
        .and_then(|id| u16::try_from(id).ok())?;
    Some(CchIndexDef::new(rel_types, weight_attr))
}

/// `CREATE CCH INDEX`. Creates a missing relationship type or weight attribute,
/// like C; replicated as those schema records, then the index record.
#[cfg(feature = "index-falkordb")]
pub(crate) fn create_cch_index<'a>(
    runtime: &'a Runtime<'a>,
    spec: &CchIndexSpec,
) -> Result<BatchOp<'a>, String> {
    reject_read_only_query(runtime)?;
    check_type_reference_count(spec)?;
    // See `create_index`: DDL runs in writer mode.
    runtime.upgrade_to_write()?;
    {
        let mut graph = runtime.graph_mut();
        let definition = resolve_or_create_cch_index(&mut graph, spec)?;
        graph
            .create_cch_index(definition)
            .map_err(|error| error.query_message())?;
    }
    // Not saved in the RDB yet.
    runtime.stats.borrow_mut().indexes_created += 1;
    emit_cch_effect(runtime, true, spec)?;
    Ok(BatchOp::Once(None))
}

/// `DROP CCH INDEX`. Resolves only: everything must already exist to match a held index.
#[cfg(feature = "index-falkordb")]
pub(crate) fn drop_cch_index<'a>(
    runtime: &'a Runtime<'a>,
    spec: &CchIndexSpec,
) -> Result<BatchOp<'a>, String> {
    reject_read_only_query(runtime)?;
    check_type_reference_count(spec)?;
    runtime.upgrade_to_write()?;
    {
        let mut graph = runtime.graph_mut();
        let definition =
            resolve_cch_index(&graph, spec).ok_or_else(|| CchError::NotFound.query_message())?;
        graph
            .drop_cch_index(&definition)
            .map_err(|error| error.query_message())?;
    }
    runtime.stats.borrow_mut().indexes_dropped += 1;
    emit_cch_effect(runtime, false, spec)?;
    Ok(BatchOp::Once(None))
}

/// Append this CCH statement to the query's effects buffer, after the schema it
/// registered. The C counterpart is `EffectsBuffer_AddCreateCCHEffect` /
/// `EffectsBuffer_AddDropCCHEffect` in `cch_index_create` / `cch_index_drop`.
#[cfg(feature = "index-falkordb")]
fn emit_cch_effect(
    runtime: &Runtime<'_>,
    create: bool,
    spec: &CchIndexSpec,
) -> Result<(), String> {
    if !runtime.build_effects.get() {
        return Ok(());
    }
    let mut buf_ref = runtime.effects_buffer.borrow_mut();
    let buf = buf_ref.get_or_insert_with(EffectsBuffer::new);
    buf.build_cch_index(
        &runtime.pending.borrow(),
        &runtime.graph(),
        create,
        &AnnouncedCchIndex {
            relationship_types: &spec.rel_types,
            weight_attribute: &spec.weight_attr,
        },
    )?;
    runtime.effects_count.set(runtime.effects_count.get() + 1);
    Ok(())
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
        &runtime.graph(),
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

#[cfg(all(test, feature = "index-falkordb"))]
mod cch_effects_tests;
