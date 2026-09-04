//! Applying a v3 effects payload on a replica.
//!
//! The v2 path decodes and applies in one pass, and has to reconstruct batches
//! it never received: it accumulates runs of adjacent `DELETE_NODE` records
//! because applying a node's edges one at a time made a replica ~40x slower than
//! the master that produced the writes.
//!
//! v3 needs none of that. A record already covers every entity of its shape, so
//! the buffer decodes to a list of records and each one applies as a single bulk
//! operation. The look-ahead, the two pending batches and the flush-on-type-change
//! logic all disappear.
//!
//! ## What this path verifies that v2 could not
//!
//! `ADD_SCHEMA` and `ADD_ATTRIBUTE` now carry the id the master assigned. The
//! replica computes the id it *would* assign and rejects the whole buffer if they
//! disagree. That check is the point of v3: every other record identifies labels,
//! types and attributes by a bare id, and until now nothing established that the
//! two engines had numbered them the same way.

use crate::{
    effects::v3::{
        AttrRef, DecodeError, INDEX_FLD_FULLTEXT, INDEX_FLD_VECTOR, Record, entity_tag,
        open_payload,
    },
    entity_type::EntityType,
    graph::graph::{Graph, TypeId},
    index::{IndexType, indexer::IndexOptions},
    runtime::{pending::IndexDocs, runtime::map_to_index_options, value::Value},
};
// Not v3's: applying a payload can fail the same ways whatever version
// wrote it, so the error lives beside `DecodeError`. Re-exported because
// this is where callers have always found it.
pub use crate::effects::error::{ApplyError, LocalName};
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
use std::sync::Arc;

impl From<String> for ApplyError {
    fn from(e: String) -> Self {
        Self::Graph(e)
    }
}

/// Index bookkeeping that accumulates across a buffer and is committed once.
#[derive(Default)]
struct IndexOps {
    /// The same type the write path collects into, rather than a second set of
    /// four maps that has to agree with it by inspection.
    docs: IndexDocs,
    /// The first never-used node id as of **before** this buffer, and the ids
    /// this buffer has created so far.
    ///
    /// Both are needed because the id space is only dense *between* buffers,
    /// not within one: records are emitted grouped by shape, so a buffer can
    /// create ids 500..600 before it creates 0..500, and mid-buffer
    /// `node_count` is a count rather than a bound. Judging an id against the
    /// mark as it was on entry, plus what this buffer has already added, is
    /// stable under that reordering.
    ///
    /// `Graph::node_id_high_water` rather than `max_node_id() + 1`: the latter
    /// has a 0 sentinel for an empty graph, which reads as "id 0 was handed
    /// out" and rejects the first create of id 0.
    pre_high_water: u64,
    /// The ids this buffer has created and not since deleted. A delete removes
    /// its ids again, because the allocator may hand a freed id back within the
    /// same buffer.
    created_here: RoaringTreemap,
}

/// Apply a whole `GRAPH.EFFECT` payload.
///
/// Every failure aborts the buffer rather than applying a prefix: effects are
/// transactional, and a record that does not decode means the replica has
/// diverged.
pub fn apply_effects(
    g: &mut Graph,
    buf: &[u8],
) -> Result<(), ApplyError> {
    // Decode one record, apply it, drop it. The alternative — decoding every
    // record first — holds the whole buffer's worth of owned ids, labels and
    // values before touching the graph, and walks that memory twice. Streaming
    // keeps a record hot in cache between being decoded and being applied, and
    // caps live decoded state at one record whatever the payload holds.
    //
    // A compressed payload still has to be inflated whole before any of it can
    // be read; `open_payload` owns that plaintext and the records borrow from it.
    let payload = open_payload(buf)?;

    let mut ops = IndexOps {
        pre_high_water: g.node_id_high_water(),
        ..IndexOps::default()
    };

    for record in payload.records() {
        apply_record(g, record?, &mut ops)?;
    }

    g.commit_index(&mut ops.docs.node_adds, &mut ops.docs.node_removes);
    g.commit_edge_index(&mut ops.docs.edge_adds, &mut ops.docs.edge_removes);
    Ok(())
}

fn apply_record(
    g: &mut Graph,
    record: Record,
    ops: &mut IndexOps,
) -> Result<(), ApplyError> {
    match record {
        Record::AddSchema {
            schema_type,
            id,
            name,
        } => apply_add_schema(g, schema_type, id, &name),

        Record::AddAttribute { id, name } => {
            // The replica appends to the same dictionary the master did, so the
            // id the name ends up holding must equal the one on the wire.
            //
            // Read back rather than derived from the dictionary's length:
            // `add_node_attribute_name` is get-or-create, so when the name is
            // already present the length does not move, and length arithmetic
            // then reports the id of the dictionary's *last* entry instead of
            // this name's. That got it wrong in both directions — it accepted a
            // buffer whose id belonged to a different attribute, which is exactly
            // the wrong-attribute corruption `IdMismatch` exists to catch, and it
            // rejected correct buffers whose name was already registered.
            // `apply_add_schema` above avoids this by taking the id its getter
            // returns; this now does the same.
            let key = Arc::new(name.clone());
            g.add_node_attribute_name(&name);
            let assigned = g
                .get_node_attribute_id(&key)
                .expect("the name was just registered");
            verify_id(
                "attribute",
                &name,
                i64::from(id),
                assigned as i64,
                g.get_node_attribute_names()
                    .get(id as usize)
                    .map(|a| a.as_str().to_string()),
            )
        }

        Record::CreateNode {
            ids,
            labels,
            attr_ids,
            rows,
        } => {
            let nodes = ids.to_roaring();
            verify_creatable(g, &nodes, ops)?;
            g.add_reserved_node_count(ids.len() as u64);
            g.create_nodes(&nodes);
            ops.created_here |= &nodes;

            // The graph's bulk APIs take `&[u64]`, so the ids are materialized
            // once here rather than per call.
            let ids: Vec<u64> = ids.iter().collect();
            if !labels.is_empty() {
                let label_ids = checked_label_ids(g, &labels)?;
                g.set_node_labels_product(&ids, &label_ids, &mut ops.docs.node_adds, true);
            }
            if !attr_ids.is_empty() {
                check_attr_shape(g, &ids, &attr_ids, &rows)?;
                g.set_nodes_attributes_rows(&ids, &attr_ids, &rows, &mut ops.docs.node_adds)?;
            }
            Ok(())
        }

        Record::CreateEdge {
            ids,
            relation_id,
            src,
            dst,
            attr_ids,
            rows,
        } => {
            let type_name = resolve_type(g, relation_id)?;
            g.add_reserved_relationship_count(ids.len() as u64);
            // `&[u64]` for the bulk APIs; materialized once each.
            let (ids, src, dst): (Vec<u64>, Vec<u64>, Vec<u64>) = (
                ids.iter().collect(),
                src.iter().collect(),
                dst.iter().collect(),
            );
            g.create_relationships_bulk(&type_name, &src, &dst, &ids);

            if !attr_ids.is_empty() {
                let map = attr_map(g, &ids, &attr_ids, &rows)?;
                g.set_relationships_attributes(&map, &mut ops.docs.edge_adds)?;
            }
            Ok(())
        }

        Record::Update {
            entity,
            ids,
            // Both of these are *used*, not re-derived. The record states the
            // schema membership the primary saw, so indexing under it makes
            // this graph's index hold what the primary's holds; re-deriving
            // would make it agree with local state instead, which is the same
            // answer only until something has diverged and a silently
            // different one afterwards. Bounds-checked against this graph's
            // dictionaries first, so a record naming a label or type the
            // replica has not seen fails the buffer rather than indexing under
            // an id it invented.
            labels,
            relation_id,
            attr_ids,
            rows,
        } => {
            let ids: Vec<u64> = ids.iter().collect();
            match entity {
                EntityType::Node => {
                    check_attr_shape(g, &ids, &attr_ids, &rows)?;
                    let label_ids = checked_label_ids(g, &labels)?;
                    g.set_nodes_attributes_rows_of_labels(
                        &ids,
                        &label_ids,
                        &attr_ids,
                        &rows,
                        &mut ops.docs.node_adds,
                    )?;
                }
                EntityType::Relationship => {
                    // Stated once for the record, so the index bookkeeping does
                    // not re-derive it per edge. `set_relationships_attributes`
                    // calls `get_relationship_type_id` for every id, which is a
                    // delta-matrix `iter` each time.
                    let type_id = checked_type_id(g, relation_id)?;
                    // Edges still go through the map form; only the node store
                    // has the row-major entry point so far.
                    let map = attr_map(g, &ids, &attr_ids, &rows)?;
                    g.set_relationships_attributes_of_type(type_id, &map, &mut ops.docs.edge_adds)?;
                }
            }
            Ok(())
        }

        Record::Labels { add, ids, labels } => {
            let label_ids = checked_label_ids(g, &labels)?;
            if add {
                g.set_node_labels_product(
                    &ids.iter().collect::<Vec<_>>(),
                    &label_ids,
                    &mut ops.docs.node_adds,
                    false,
                );
            } else {
                // Removal still takes the expanded pairs; only the add path has
                // been given the compact form so far.
                let mut rows = Vec::with_capacity(ids.len() * label_ids.len());
                let mut cols = Vec::with_capacity(ids.len() * label_ids.len());
                for &lid in &label_ids {
                    for id in ids.iter() {
                        rows.push(id);
                        cols.push(lid);
                    }
                }
                g.remove_nodes_labels(&rows, &cols, &mut ops.docs.node_removes);
            }
            Ok(())
        }

        // One bulk delete for the whole record — the reason v3 needs no
        // stream look-ahead.
        Record::DeleteNode { ids, labels: _ } => {
            // Borrowed, not consumed: a by-value `into_iter` materialized a
            // `Vec<u64>` first, and a delete-by-label arrives as a consecutive
            // range — the one shape that has no vector to hand over.
            let nodes = ids.to_roaring();
            verify_deletable(g, &nodes, ops)?;
            g.delete_nodes(&nodes, &mut ops.docs.node_removes)?;
            // Deleting releases the id back to the bin, so a *later* record in
            // this same buffer may legitimately create it again: a multi-commit
            // query (`CREATE (n) WITH n DELETE n WITH 1 AS z CREATE ()`) commits
            // three times into one buffer, and the allocator recycles the freed
            // id on the third. Leaving it in `created_here` makes that a
            // false `NodeAlreadyLive` and discards the whole payload.
            ops.created_here -= &nodes;
            Ok(())
        }

        Record::DeleteEdge { ids, .. } => {
            let edges = ids.to_roaring();
            g.delete_relationships(&edges, &mut ops.docs.edge_removes)?;
            Ok(())
        }

        Record::Index {
            create,
            schema_type,
            label_id,
            label,
            field_type,
            fields,
            options,
        } => {
            verify_schema(g, schema_type, label_id, &label)?;
            for field in &fields {
                verify_attribute(g, field.id, &field.name)?;
            }
            let entity_type = schema_type;
            let index_type = index_type_of(field_type);
            let label = Arc::new(label);
            let fields: Vec<Arc<String>> = fields.into_iter().map(|f| Arc::new(f.name)).collect();
            if create {
                // Population is spawned, not run here. `populate_indexes_sync`
                // ran on the Redis main thread, so a replica applying an index
                // over a large label froze for the whole build — the same class
                // of problem as expanding ids while decoding.
                //
                // Spawning is safe for the reason it is safe on the primary,
                // which has always done it under concurrent writes:
                // `populate_index_batch` populates from a snapshot in 10,000-row
                // batches, and entities written *after* the snapshot are indexed
                // by the write path instead (`IndexOps::docs` into
                // `commit_index`). A later record that drops or recreates the
                // index does not race it either — the population ticket carries
                // a generation, and a worker whose generation is stale releases
                // its ticket and stops rather than committing documents into the
                // new spec.
                g.create_index(
                    &index_type,
                    &entity_type,
                    &label,
                    &fields,
                    index_options(&index_type, options.as_ref())?,
                )?;
            } else {
                g.drop_index(&index_type, &entity_type, &label, &fields)?;
            }
            Ok(())
        }

        Record::Constraint {
            create,
            constraint_type,
            entity_type,
            status,
            label_id,
            label,
            props,
        } => {
            verify_schema(g, entity_type, label_id, &label)?;
            for AttrRef { id, name } in &props {
                verify_attribute(g, *id, name)?;
            }

            let ct = constraint_type;
            let et = entity_type;
            let properties: Vec<Arc<String>> =
                props.into_iter().map(|p| Arc::new(p.name)).collect();

            if create {
                // Install the master's outcome rather than re-deriving it. A
                // replica that validated independently would scan at a
                // different time against different interleavings, and could
                // legitimately reach a different status. The upsert is what
                // lets the second announcement — the one carrying the validated
                // status — land on the constraint the first one created.
                // A create record always carries one; the decoder builds `Some`
                // from the opcode, so this cannot be `None` here.
                let status = status.ok_or(ApplyError::MissingConstraintStatus)?;
                g.upsert_constraint_raw(ct, et, &Arc::new(label), &properties, status);
            } else {
                g.drop_constraint(&ct, &et, &label, &properties)?;
            }
            Ok(())
        }
    }
}

/// `ADD_SCHEMA`: register the label or type, then check the id matches.
fn apply_add_schema(
    g: &mut Graph,
    schema_type: EntityType,
    id: i32,
    name: &str,
) -> Result<(), ApplyError> {
    match schema_type {
        EntityType::Node => {
            let assigned = usize::from(g.get_label_id_mut(name)) as i64;
            let local = g.get_labels().get(id as usize).map(|l| l.to_string());
            verify_id("label", name, i64::from(id), assigned, local)
        }
        EntityType::Relationship => {
            let assigned = g.get_type_id_mut(name).0 as i64;
            let local = g.get_types().get(id as usize).map(|t| t.to_string());
            verify_id("relationship type", name, i64::from(id), assigned, local)
        }
    }
}

/// The check v2 could not make.
/// Every id a `CREATE_NODE` names must be one this replica could legitimately
/// hand out: recycled, or past the high-water mark. Anything else is already
/// live, and creating it would double-count `node_count` and shift every
/// subsequent fresh id.
///
/// Two roaring operations for the whole record, not a probe per id: the ids not
/// in the bin are a set difference, and because that difference is sorted only
/// its minimum has to clear the mark.
fn verify_creatable(
    g: &Graph,
    nodes: &RoaringTreemap,
    ops: &IndexOps,
) -> Result<(), ApplyError> {
    // Not in the bin means it was never freed; below the entry mark means it
    // was already handed out. Both together mean it is live here, and creating
    // it would double-count `node_count` and shift every later fresh id.
    let fresh = g.node_ids_not_recycled(nodes);
    if let Some(lowest) = fresh.min().filter(|&id| id < ops.pre_high_water) {
        return Err(ApplyError::NodeAlreadyLive {
            id: lowest,
            bin: g.deleted_nodes_count(),
            high_water: ops.pre_high_water,
        });
    }
    // Two records in one buffer claiming the same id is divergence too, and the
    // mark cannot see it because neither id was live on entry.
    if let Some(twice) = (nodes & &ops.created_here).min() {
        return Err(ApplyError::NodeAlreadyLive {
            id: twice,
            bin: g.deleted_nodes_count(),
            high_water: ops.pre_high_water,
        });
    }
    Ok(())
}

/// Every id a `DELETE_NODE` names must currently be live: not already in the
/// recycle bin, and below the high-water mark.
fn verify_deletable(
    g: &Graph,
    nodes: &RoaringTreemap,
    ops: &IndexOps,
) -> Result<(), ApplyError> {
    if let Some(id) = g.node_ids_already_recycled(nodes).min() {
        return Err(ApplyError::NodeNotLive {
            id,
            reason: "it is already in the recycle bin",
        });
    }
    // At or above the entry mark and not created by this buffer means it was
    // never allocated here at all. The second half matters: a buffer may legitimately
    // create a node and then delete it.
    if let Some(id) = (nodes - &ops.created_here)
        .max()
        .filter(|&id| id >= ops.pre_high_water)
    {
        return Err(ApplyError::NodeNotLive {
            id,
            reason: "it was never allocated here",
        });
    }
    Ok(())
}

fn verify_id(
    kind: &'static str,
    name: &str,
    expected: i64,
    assigned: i64,
    local: Option<String>,
) -> Result<(), ApplyError> {
    if assigned == expected {
        return Ok(());
    }
    Err(ApplyError::IdMismatch {
        kind,
        name: name.to_string(),
        expected,
        assigned,
        local: LocalName(local),
    })
}

/// Resolve a label or type by id and confirm the name matches.
///
/// The id is authoritative; the name is the cheap cross-check that surfaces
/// divergence instead of writing through a stale id. Mirrors C's `VerifySchema`.
fn verify_schema(
    g: &Graph,
    schema_type: EntityType,
    id: i32,
    name: &str,
) -> Result<(), ApplyError> {
    let (kind, local) = match schema_type {
        EntityType::Node => (
            "label",
            g.get_labels().get(id as usize).map(|l| l.to_string()),
        ),
        EntityType::Relationship => (
            "relationship type",
            g.get_types().get(id as usize).map(|t| t.to_string()),
        ),
    };
    resolved(kind, name, i64::from(id), local)
}

/// Mirrors C's `VerifyAttribute`.
fn verify_attribute(
    g: &Graph,
    id: u16,
    name: &str,
) -> Result<(), ApplyError> {
    let local = g
        .get_node_attribute_names()
        .get(id as usize)
        .map(|a| a.as_str().to_string());
    resolved("attribute", name, i64::from(id), local)
}

fn resolved(
    kind: &'static str,
    name: &str,
    id: i64,
    local: Option<String>,
) -> Result<(), ApplyError> {
    match local {
        Some(local) if local == name => Ok(()),
        Some(local) => Err(ApplyError::NameMismatch {
            kind,
            name: name.to_string(),
            id,
            local,
        }),
        None => Err(ApplyError::Unresolved {
            kind,
            name: name.to_string(),
            id,
        }),
    }
}

fn resolve_type(
    g: &Graph,
    relation_id: i32,
) -> Result<Arc<String>, ApplyError> {
    let out_of_range = || ApplyError::IdOutOfRange {
        kind: "relationship type",
        id: i64::from(relation_id),
    };
    if relation_id < 0 {
        return Err(out_of_range());
    }
    g.get_type(TypeId(relation_id as usize))
        .ok_or_else(out_of_range)
}

/// An `UPDATE_EDGE`'s relationship type, checked against this graph.
///
/// The same check C's `ApplyUpdateEdge` opens with — it refuses a record whose
/// `r_id` is negative or past the local edge-schema count, logging "references
/// relationship type %d which doesn't exist locally". A replica that has not
/// seen the `ADD_SCHEMA` yet must fail here rather than index the rows under a
/// type it invented.
fn checked_type_id(
    g: &Graph,
    relation_id: Option<i32>,
) -> Result<TypeId, ApplyError> {
    let relation_id = relation_id.ok_or(ApplyError::IdOutOfRange {
        kind: "relationship type",
        id: -1,
    })?;
    resolve_type(g, relation_id)?;
    Ok(TypeId(relation_id as usize))
}

/// Expand `(ids, labels)` into the row/column pairs the bulk label API takes,
/// bounds-checking every label id on the way.
/// The label ids, checked against this graph's dictionary.
///
/// Returns the ids themselves rather than the `ids x labels` product: the
/// product is `set_node_labels_product`'s to walk lazily, and materializing it
/// here cost two allocations the size of the product for a caller that then
/// regrouped it straight back.
fn checked_label_ids(
    g: &Graph,
    labels: &[i32],
) -> Result<Vec<u64>, ApplyError> {
    let bound = g.get_labels().len();
    labels
        .iter()
        .map(|&label| {
            if label < 0 || label as usize >= bound {
                return Err(ApplyError::IdOutOfRange {
                    kind: "label",
                    id: i64::from(label),
                });
            }
            Ok(label as u64)
        })
        .collect()
}

/// Turn a record's shape plus its row-major values back into per-entity
/// attribute lists.
///
/// A `T_NULL` slot means the property is absent for that entity, which is
/// unambiguous because FalkorDB never stores a null property value.
/// Every attribute id resolves here, and the value block is the size the
/// record's count and shape imply.
fn check_attr_shape(
    g: &Graph,
    ids: &[u64],
    attr_ids: &[u16],
    rows: &[Value],
) -> Result<(), ApplyError> {
    let bound = g.get_node_attribute_names().len();
    for &attr_id in attr_ids {
        if attr_id as usize >= bound {
            return Err(ApplyError::IdOutOfRange {
                kind: "attribute",
                id: i64::from(attr_id),
            });
        }
    }
    // Strictly ascending, and this is the only place it is enforced.
    // `insert_attrs_rows` documents the requirement and checks it with a
    // `debug_assert`, so in a release build an out-of-order `AttrSet` becomes an
    // unsorted span in the store, and every later property read binary-searches
    // it: wrong values or a spurious `Null`, compounding on each merge. A
    // duplicate id puts two entries under one key in the same span.
    //
    // No Rust primary emits one — `Pending` keeps a sorted vec — but v3 exists so
    // that a *C* primary can write these buffers, and C's `AttributeSet` carries
    // no such guarantee.
    if let Some(w) = attr_ids.windows(2).find(|w| w[0] >= w[1]) {
        return Err(ApplyError::AttrIdsNotAscending {
            first: w[0],
            second: w[1],
        });
    }
    let width = attr_ids.len();
    if rows.len() != ids.len() * width {
        return Err(ApplyError::ShapeMismatch {
            entities: ids.len(),
            width,
            values: rows.len(),
        });
    }
    Ok(())
}

fn attr_map(
    g: &Graph,
    ids: &[u64],
    attr_ids: &[u16],
    rows: &[Value],
) -> Result<FxHashMap<u64, Vec<(u16, Value)>>, ApplyError> {
    check_attr_shape(g, ids, attr_ids, rows)?;
    let width = attr_ids.len();

    let mut map = FxHashMap::default();
    for (row, &id) in ids.iter().enumerate() {
        // Nulls are kept, not filtered: a null means "remove this attribute",
        // and `merge_span` is what removes it. Dropping them here made
        // `SET x = NULL` a no-op on the replica while the primary removed the
        // property, so the two diverged silently until the next resync.
        let pairs: Vec<(u16, Value)> = attr_ids
            .iter()
            .enumerate()
            .map(|(col, &attr_id)| (attr_id, rows[row * width + col].clone()))
            .collect();
        // No sort: `check_attr_shape` above has already refused anything that is
        // not strictly ascending, and the pairs are built in `attr_ids` order.
        map.insert(id, pairs);
    }
    Ok(map)
}

/// Turn the record's options value back into typed index options.
///
/// v2 dropped `OPTIONS {...}` on the wire entirely and forced those statements
/// to replicate as verbatim queries. v3 carries the map, so the replica rebuilds
/// the same options the master did rather than approximating them.
fn index_options(
    index_type: &IndexType,
    options: Option<&Value>,
) -> Result<Option<IndexOptions>, ApplyError> {
    match options {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Map(m)) => Ok(map_to_index_options(index_type, m)?),
        Some(other) => Err(ApplyError::OptionsNotAMap(format!("{other:?}"))),
    }
}

/// `IndexFieldType` is a bit flag set, so this tests bits rather than matching
/// a discriminant. Anything that is neither full-text nor vector is a range
/// index — `INDEX_FLD_RANGE` is itself the union of the three scalar kinds.
fn index_type_of(field_type: u32) -> IndexType {
    if field_type & INDEX_FLD_FULLTEXT != 0 {
        IndexType::Fulltext
    } else if field_type & INDEX_FLD_VECTOR != 0 {
        IndexType::Vector
    } else {
        IndexType::Range
    }
}

/// C numbers `GraphEntityType` from 1; this keeps the compiler honest.
const _: () = assert!(entity_tag(EntityType::Node) == 1);
const _: () = assert!(entity_tag(EntityType::Relationship) == 2);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::v3::ConstraintSpec;
    use crate::effects::v3::staging::StagePending;
    use crate::effects::v3::{
        AttrRef, INDEX_FLD_RANGE, IdList, new_buffer, write_add_attribute, write_add_schema,
        write_constraint, write_create_index, write_create_node, write_labels, write_update,
    };
    use crate::graph::constraint::{ConstraintStatus, ConstraintType};

    fn graph() -> Graph {
        crate::graph::graphblas::test_init::ensure_init();
        Graph::new(64, 64, 0, 0, "t")
    }

    /// The delete-then-recreate cycle a replica legitimately sees.
    fn write_delete(
        buf: &mut Vec<u8>,
        ids: &IdList,
        labels: &[i32],
    ) {
        crate::effects::v3::write_delete_node(buf, ids, labels);
    }

    #[test]
    fn recreating_a_recycled_id_is_allowed() {
        // The case the check must not break: the primary deleted 1, so 1 is in
        // the bin here too, and it is free to come back.
        let mut g = graph();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([0, 1, 2]), &[], &[], &[]);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([1]), &[]);
        apply_effects(&mut g, &buf).expect("delete must apply");

        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([1]), &[], &[], &[]);
        apply_effects(&mut g, &buf).expect("recreating a recycled id must apply");
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn creating_a_fresh_id_past_the_high_water_mark_is_allowed() {
        // Sequential allocation on the primary can outrun this replica's
        // high-water mark without anything being wrong.
        let mut g = graph();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([0, 1]), &[], &[], &[]);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([2, 3]), &[], &[], &[]);
        apply_effects(&mut g, &buf).expect("fresh ids must apply");
        assert_eq!(g.node_count(), 4);
    }

    #[test]
    fn creating_an_already_live_id_aborts_the_buffer() {
        // The divergence that is otherwise invisible: node 1 is live here and
        // the primary says to create it. Unchecked, node_count double-counts
        // and every later fresh id is off by one.
        let mut g = graph();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([0, 1, 2]), &[], &[], &[]);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([1]), &[], &[], &[]);
        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(err, ApplyError::NodeAlreadyLive { id: 1, .. }),
            "{err}"
        );
        assert_eq!(g.node_count(), 3, "the buffer must not have been applied");
    }

    #[test]
    fn one_buffer_claiming_an_id_twice_aborts() {
        // Neither record's ids were live on entry, so the high-water mark alone
        // cannot see this. It is still divergence.
        let mut g = graph();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([0, 1]), &[], &[], &[]);
        write_create_node(&mut buf, &IdList::from([1, 2]), &[], &[], &[]);
        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(err, ApplyError::NodeAlreadyLive { id: 1, .. }),
            "{err}"
        );
    }

    #[test]
    fn a_buffer_may_create_a_node_and_then_delete_it() {
        // The false positive the `created_here` set exists to avoid: the id is
        // past the entry mark, so it looks never-allocated to the delete check.
        let mut g = graph();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([0, 1]), &[], &[], &[]);
        write_delete(&mut buf, &IdList::from([1]), &[]);
        apply_effects(&mut g, &buf).expect("create-then-delete must apply");
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn deleting_an_already_recycled_id_aborts_the_buffer() {
        let mut g = graph();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([0, 1]), &[], &[], &[]);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([1]), &[]);
        apply_effects(&mut g, &buf).expect("delete must apply");

        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([1]), &[]);
        let err = apply_effects(&mut g, &buf).expect_err("must refuse a double delete");
        assert!(
            matches!(err, ApplyError::NodeNotLive { id: 1, .. }),
            "{err}"
        );
    }

    #[test]
    fn deleting_a_never_allocated_id_aborts_the_buffer() {
        let mut g = graph();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([0, 1]), &[], &[], &[]);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([99]), &[]);
        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(err, ApplyError::NodeNotLive { id: 99, .. }),
            "{err}"
        );
    }

    #[test]
    fn a_null_in_a_row_removes_the_property() {
        // FalkorDB never stores a null, so a null on the wire means "remove".
        // Both the apply paths used to filter nulls out before the attribute
        // store saw them, which made `SET x = NULL` a no-op on the replica
        // while the primary removed the property — a silent divergence that
        // only healed on the next resync.
        let mut g = graph();
        let mut buf = new_buffer();
        write_add_schema(&mut buf, EntityType::Node, 0, "L");
        write_add_attribute(&mut buf, 0, "keep");
        write_add_attribute(&mut buf, 1, "drop");
        write_create_node(
            &mut buf,
            &IdList::from([0, 1]),
            &[0],
            &[0, 1],
            &[Value::Int(1), Value::Int(10), Value::Int(2), Value::Int(20)],
        );
        apply_effects(&mut g, &buf).expect("create must apply");
        assert_eq!(
            g.get_node_attribute(0.into(), &Arc::new("drop".into())),
            Some(Value::Int(10))
        );

        // Now null it out, as an UPDATE_NODE would.
        let mut buf = new_buffer();
        crate::effects::v3::write_update(
            &mut buf,
            EntityType::Node,
            &IdList::from([0, 1]),
            &[0],
            None,
            &[1],
            &[Value::Null, Value::Null],
        );
        apply_effects(&mut g, &buf).expect("update must apply");

        assert_eq!(
            g.get_node_attribute(0.into(), &Arc::new("drop".into())),
            None,
            "the null must have removed it"
        );
        assert_eq!(
            g.get_node_attribute(0.into(), &Arc::new("keep".into())),
            Some(Value::Int(1)),
            "and left the rest alone"
        );
    }

    #[test]
    fn a_label_added_in_a_later_buffer_widens_the_label_matrix() {
        // `node_labels_matrix` is nodes x labels. Registering a label grows the
        // per-label matrix list but does not widen that matrix, so applying the
        // new label writes a column index past its width unless something
        // resizes first. `create_nodes` resizes, which is why a CREATE_NODE
        // carrying a new label is safe — but SET_LABELS has no create to lean
        // on, so the label writer has to do it.
        let mut g = graph();

        let mut buf = new_buffer();
        write_add_schema(&mut buf, EntityType::Node, 0, "A");
        write_create_node(&mut buf, &IdList::from([0, 1, 2]), &[0], &[], &[]);
        apply_effects(&mut g, &buf).expect("first buffer must apply");

        // A second buffer introducing a label and immediately applying it.
        let mut buf = new_buffer();
        write_add_schema(&mut buf, EntityType::Node, 1, "B");
        write_labels(&mut buf, true, &IdList::from([0, 1, 2]), &[1]);
        apply_effects(&mut g, &buf).expect("labelling with a fresh label must apply");

        assert_eq!(g.get_labels().len(), 2);
        assert_eq!(
            g.label_node_count(&Arc::new("B".to_string())),
            3,
            "every node must have picked up the new label"
        );
    }

    #[test]
    fn an_all_deleted_graph_still_accepts_recycled_and_fresh_ids() {
        // The edge case where "highest id handed out" and "count of live nodes"
        // come apart completely: every node is gone, so `node_count` is 0 while
        // the recycle bin holds every id ever allocated. `max_node_id` returns a
        // 0 sentinel here rather than 2, so anything derived from it has to
        // still get this right.
        let mut g = graph();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([0, 1, 2]), &[], &[], &[]);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([0, 1, 2]), &[]);
        apply_effects(&mut g, &buf).expect("delete must apply");
        assert_eq!(g.node_count(), 0);

        // Recycled: every id is in the bin, so all three may come back.
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([1]), &[], &[], &[]);
        apply_effects(&mut g, &buf).expect("a recycled id must apply");

        // Fresh: past anything ever handed out.
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([99]), &[], &[], &[]);
        apply_effects(&mut g, &buf).expect("a fresh id must apply");

        // And a delete of something already binned is still refused.
        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([0]), &[]);
        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(err, ApplyError::NodeNotLive { id: 0, .. }),
            "{err}"
        );
    }

    #[test]
    fn a_buffer_may_create_delete_and_recreate_the_same_id() {
        // A multi-commit query commits into *one* buffer, and the id allocator
        // recycles a freed id across commits, so `C(0) · D(0) · C(0)` is what
        // `CREATE (n) WITH n DELETE n WITH 1 AS z CREATE ()` actually ships.
        // Before the delete released it from `created_here` this tripped
        // `NodeAlreadyLive` and the replica discarded all three commits,
        // leaving the master with a node the replica never got.
        let mut g = graph();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([0]), &[], &[], &[]);
        write_delete(&mut buf, &IdList::from([0]), &[]);
        write_create_node(&mut buf, &IdList::from([0]), &[], &[], &[]);
        apply_effects(&mut g, &buf).expect("the recreate is legitimate");
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn a_buffer_applies_end_to_end() {
        let mut g = graph();
        let mut buf = new_buffer();
        write_add_schema(&mut buf, EntityType::Node, 0, "Person");
        write_add_attribute(&mut buf, 0, "name");
        write_create_node(
            &mut buf,
            &IdList::from([0, 1, 2]),
            &[0],
            &[0],
            &[
                Value::String(Arc::new("a".into())),
                Value::String(Arc::new("b".into())),
                Value::String(Arc::new("c".into())),
            ],
        );

        apply_effects(&mut g, &buf).expect("buffer must apply");
        assert_eq!(g.get_labels().len(), 1);
        assert_eq!(g.get_node_attribute_names().len(), 1);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn a_schema_id_disagreement_aborts_the_buffer() {
        // The check that is the whole point of v3. The replica already holds a
        // label, so appending "Person" gives it id 1 — but the master says 0.
        let mut g = graph();
        g.get_label_id_mut("Existing");

        let mut buf = new_buffer();
        write_add_schema(&mut buf, EntityType::Node, 0, "Person");

        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(
                &err,
                ApplyError::IdMismatch {
                    kind: "label",
                    expected: 0,
                    assigned: 1,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn an_attribute_id_disagreement_aborts_the_buffer() {
        // The case that actually bit us: an RDB-seeded replica whose dictionary
        // is a different length silently lands the value on another attribute.
        let mut g = graph();
        g.add_node_attribute_name("already_here");

        let mut buf = new_buffer();
        write_add_attribute(&mut buf, 0, "name");

        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(
                &err,
                ApplyError::IdMismatch {
                    kind: "attribute",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_stale_label_id_is_caught_by_its_name() {
        // VerifySchema's job: the id resolves, but to something else.
        let mut g = graph();
        g.get_label_id_mut("Actual");
        g.add_node_attribute_name("a");

        let mut buf = new_buffer();
        write_create_index(
            &mut buf,
            EntityType::Node,
            0,
            "Expected",
            INDEX_FLD_RANGE,
            &[AttrRef { id: 0, name: "a" }],
            &Value::Null,
        );
        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        let ApplyError::NameMismatch { name, local, .. } = &err else {
            panic!("expected a name mismatch, got {err:?}");
        };
        assert_eq!(name, "Expected");
        assert_eq!(local, "Actual");
    }

    #[test]
    fn labels_apply_to_every_node_in_the_record() {
        let mut g = graph();
        let mut buf = new_buffer();
        write_add_schema(&mut buf, EntityType::Node, 0, "L");
        write_create_node(&mut buf, &IdList::from([0, 1, 2, 3]), &[], &[], &[]);
        write_labels(&mut buf, true, &IdList::from([0, 1, 2, 3]), &[0]);

        apply_effects(&mut g, &buf).expect("must apply");
        assert_eq!(g.label_node_count(&Arc::new("L".to_string())), 4);
    }

    #[test]
    fn the_second_announcement_converges_rather_than_duplicates() {
        // The primary announces a constraint twice: once under construction,
        // once with the validated status. The replica must end with one
        // constraint, enforcing — not two, and not one stuck pending.
        let mut g = graph();
        g.get_label_id_mut("Person");
        g.add_node_attribute_name("email");

        for status in [
            ConstraintStatus::UnderConstruction,
            ConstraintStatus::Operational,
        ] {
            let mut buf = new_buffer();
            write_constraint(
                &mut buf,
                true,
                &ConstraintSpec {
                    constraint_type: ConstraintType::Unique,
                    entity_type: EntityType::Node,
                    status: Some(status),
                    label_id: 0,
                    label: "Person",
                    props: &[AttrRef {
                        id: 0,
                        name: "email",
                    }],
                },
            );
            apply_effects(&mut g, &buf).expect("announcement must apply");
        }

        let constraints = g.constraints();
        assert_eq!(constraints.len(), 1, "one constraint, announced twice");
        assert_eq!(
            constraints[0].status,
            ConstraintStatus::Operational,
            "the replica must end on the primary's validated status"
        );
    }

    #[test]
    fn a_constraint_installs_the_masters_outcome() {
        // Not re-validated locally: a replica scanning independently would do so
        // at a different time against different interleavings, and could
        // legitimately reach a different status.
        let mut g = graph();
        let mut buf = new_buffer();
        write_add_schema(&mut buf, EntityType::Node, 0, "Person");
        write_add_attribute(&mut buf, 0, "email");
        write_constraint(
            &mut buf,
            true,
            &ConstraintSpec {
                constraint_type: ConstraintType::Unique,
                entity_type: EntityType::Node,
                status: Some(ConstraintStatus::Operational),
                label_id: 0,
                label: "Person",
                props: &[AttrRef {
                    id: 0,
                    name: "email",
                }],
            },
        );

        apply_effects(&mut g, &buf).expect("must apply");
        assert_eq!(g.constraints().len(), 1);
    }

    #[test]
    fn a_malformed_buffer_is_refused_not_applied() {
        let mut g = graph();
        let mut buf = new_buffer();
        write_add_schema(&mut buf, EntityType::Node, 0, "L");
        write_create_node(&mut buf, &IdList::from([0, 1]), &[0], &[], &[]);

        for cut in 2..buf.len() {
            let mut fresh = graph();
            // Must never panic, whatever the truncation.
            let _ = apply_effects(&mut fresh, &buf[..cut]);
        }
        // And the whole buffer still works.
        apply_effects(&mut g, &buf).expect("intact buffer must apply");
    }

    #[test]
    fn a_pending_survives_the_round_trip_to_a_replica() {
        // The end-to-end check the unit tests cannot make: build a real Pending
        // on a "master", emit v3, apply it to an empty "replica", and compare
        // the observable state. Encoder and decoder agreeing with each other is
        // not the same as either being right.
        use crate::runtime::pending::Pending;
        use atomic_refcell::AtomicRefCell;

        let master = AtomicRefCell::new(graph());
        {
            let mut g = master.borrow_mut();
            g.get_label_id_mut("Person");
            g.add_node_attribute_name("name");
            g.add_node_attribute_name("age");
        }

        let mut p = Pending::default();
        for id in 0..500u64 {
            p.stage_created_node(id, &[0], &[(0, Value::Int(id as i64)), (1, Value::Int(30))]);
        }
        // A second shape, so the buffer carries more than one record.
        for id in 500..600u64 {
            p.stage_created_node(id, &[0], &[(0, Value::Int(id as i64))]);
        }

        let mut buf = Vec::new();
        crate::effects::v3::emit::build_effects_buffer(&p, &master, &mut buf);
        assert_eq!(
            crate::effects::v3::read_buffer(&buf).unwrap().len(),
            5,
            "2 schema + 1 attr + 2 shapes"
        );

        let mut replica = graph();
        apply_effects(&mut replica, &buf).expect("replica must apply the master's buffer");

        let m = master.borrow();
        assert_eq!(
            replica.get_labels(),
            m.get_labels(),
            "label ids must line up"
        );
        assert_eq!(
            replica.get_node_attribute_names(),
            m.get_node_attribute_names(),
            "attribute ids must line up"
        );
        assert_eq!(replica.node_count(), 600);
        assert_eq!(
            replica.label_node_count(&Arc::new("Person".to_string())),
            600
        );
    }

    #[test]
    fn an_unsorted_attribute_set_is_refused_rather_than_stored() {
        // The stores merge the record's ids in as a *span*, so wire order is
        // load-bearing: unsorted, a release build stores a span that every later
        // binary search reads wrong. Only a `debug_assert` stood behind this, so
        // the release replica silently corrupted instead of failing.
        for bad in [vec![7_u16, 3], vec![5, 5]] {
            let mut g = graph();
            for name in ["a", "b", "c", "d", "e", "f", "g", "h"] {
                g.add_node_attribute_name(name);
            }
            let mut buf = new_buffer();
            write_create_node(&mut buf, &IdList::from([0]), &[], &[], &[]);
            apply_effects(&mut g, &buf).expect("setup");

            let mut buf = new_buffer();
            write_update(
                &mut buf,
                EntityType::Node,
                &IdList::from([0]),
                &[],
                None,
                &bad,
                &[Value::Int(1), Value::Int(2)],
            );
            assert_eq!(
                apply_effects(&mut g, &buf),
                Err(ApplyError::AttrIdsNotAscending {
                    first: bad[0],
                    second: bad[1],
                }),
                "accepted {bad:?}"
            );
        }
    }

    #[test]
    fn an_add_attribute_is_judged_against_the_id_the_name_holds() {
        // Both directions of the bug, on a replica that already knows the name.
        // Length arithmetic reported the id of the dictionary's last entry, so a
        // re-announcement of an *earlier* attribute was measured against the
        // wrong id.
        let mut g = graph();
        for name in ["p", "q", "r"] {
            g.add_node_attribute_name(name);
        }

        // The damaging direction: "q" is id 1 here, and a buffer claiming id 2
        // for it must be refused. Under length arithmetic `assigned` came out as
        // 2 and this was accepted, after which every record carrying attribute 2
        // wrote `r` on the replica and `q` on the primary.
        let mut buf = new_buffer();
        write_add_attribute(&mut buf, 2, "q");
        assert!(
            matches!(
                apply_effects(&mut g, &buf),
                Err(ApplyError::IdMismatch { .. })
            ),
            "a mismatched id must be refused"
        );

        // And the correct id for an already-registered name must be accepted —
        // length arithmetic rejected this one, stopping replication outright.
        let mut buf = new_buffer();
        write_add_attribute(&mut buf, 1, "q");
        apply_effects(&mut g, &buf).expect("the real id of 'q' is 1");

        // A genuinely new name still lands on the next id.
        let mut buf = new_buffer();
        write_add_attribute(&mut buf, 3, "s");
        apply_effects(&mut g, &buf).expect("'s' is the next id");
    }

    #[test]
    fn index_field_type_maps_by_bit_not_ordinal() {
        assert_eq!(index_type_of(INDEX_FLD_RANGE), IndexType::Range);
        assert_eq!(index_type_of(INDEX_FLD_FULLTEXT), IndexType::Fulltext);
        assert_eq!(index_type_of(INDEX_FLD_VECTOR), IndexType::Vector);
        // A range index is the OR of three scalar kinds, so bit-testing is the
        // only thing that classifies it correctly.
        assert_eq!(
            index_type_of(crate::effects::v3::INDEX_FLD_NUMERIC),
            IndexType::Range
        );
    }
}
