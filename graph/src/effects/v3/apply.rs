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

use super::records::IndexOptions as WireIndexOptions;
use crate::index::text_index_options::TextIndexOptions;
use crate::index::vector_index_options::VectorIndexOptions;
use crate::{
    effects::v3::{
        AttrRef, INDEX_FLD_FULLTEXT, INDEX_FLD_VECTOR, Record, entity_tag, open_payload,
    },
    entity_type::EntityType,
    graph::{
        graph::{Graph, NodeOpError, TypeId},
        id_space::{IdSpace, IdSpaceError},
    },
    index::{IndexType, indexer::IndexOptions},
    runtime::{pending::IndexDocs, value::Value},
};
// Not v3's: applying a payload can fail the same ways whatever version
// wrote it, so the error lives beside `DecodeError`. Re-exported because
// this is where callers have always found it.
pub use crate::effects::error::{ApplyError, LocalName};
use rustc_hash::FxHashMap;
use std::sync::Arc;

impl From<String> for ApplyError {
    fn from(e: String) -> Self {
        Self::Graph(e)
    }
}

/// What accumulates across a buffer and is settled once, at the end.
struct BufferOps {
    /// The same type the write path collects into, rather than a second set of
    /// four maps that has to agree with it by inspection.
    docs: IndexDocs,
    /// The node id space this buffer is building.
    ///
    /// Held here rather than on the graph because its lifetime is the buffer's,
    /// and a buffer is a thing only this file knows about. What the graph does
    /// know is how to maintain one: `create_nodes` and `delete_nodes` take it and
    /// feed it themselves, so nothing here can create a node and forget to
    /// account for it — and nothing can be handed a graph without saying what to
    /// do about validation, because the argument is not optional to supply.
    nodes: IdSpace,
    /// And the relationship one. Same type, same checks — the id spaces are two
    /// counted ranges with recycle bins, and nothing about the invariant differs.
    edges: IdSpace,
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

    let mut ops = BufferOps {
        docs: IndexDocs::default(),
        nodes: IdSpace::at(g.node_id_bound()),
        edges: IdSpace::at(g.relationship_id_bound()),
    };

    for record in payload.records() {
        apply_record(g, record?, &mut ops)?;
    }

    // Only now: records are grouped by shape rather than ordered by id, so the
    // id space is legitimately fragmented partway through a buffer and only has
    // to be whole at the end. A buffer that fails earlier never reaches this,
    // which is right — it has not finished building the thing being checked.
    ops.nodes
        .verify(g.node_id_bound())
        .map_err(|e| id_space_error_map("node", e))?;
    ops.edges
        .verify(g.relationship_id_bound())
        .map_err(|e| id_space_error_map("relationship", e))?;

    g.commit_index(&mut ops.docs.node_adds, &mut ops.docs.node_removes);
    g.commit_edge_index(&mut ops.docs.edge_adds, &mut ops.docs.edge_removes);
    Ok(())
}

/// A refusal from a bulk node operation, as the divergence it is.
///
/// Two arms, and only one of them is really the graph's: whatever it reported
/// while doing the work. Every judgement about liveness belongs to
/// [`IdSpace`] — it decides, and its refusals arrive wrapped, to be unwrapped
/// straight back into the rendering [`id_space_error_map`] gives them.
fn node_op(
    kind: &'static str,
    e: NodeOpError,
) -> ApplyError {
    match e {
        NodeOpError::Graph(e) => ApplyError::Graph(e),
        NodeOpError::IdSpace(e) => id_space_error_map(kind, e),
    }
}

/// A refusal from the id-space check, as the divergence it is.
///
/// The graph states these in its own terms — it knows nothing of buffers or of a
/// peer engine — so the wording that names both sides is added here, where there
/// is a peer to name.
fn id_space_error_map(
    kind: &'static str,
    e: IdSpaceError,
) -> ApplyError {
    match e {
        IdSpaceError::AlreadyLive { id, entry_bound } => ApplyError::AlreadyLive {
            kind,
            id,
            first_unallocated: entry_bound,
        },
        IdSpaceError::AlreadyRecycled(id) => ApplyError::NotLive {
            kind,
            id,
            reason: "it is already in the recycle bin",
        },
        IdSpaceError::NeverCreated(id) => ApplyError::NotLive {
            kind,
            id,
            reason: "it was never allocated here",
        },
        IdSpaceError::Hole {
            entry_bound,
            highest,
            created,
        } => ApplyError::IdsHaveAHole {
            kind,
            entry_bound,
            highest,
            created,
        },
        IdSpaceError::Miscounted {
            graph_bound,
            expected,
        } => ApplyError::CountMiscounted {
            kind,
            graph_bound,
            expected,
        },
        IdSpaceError::IdOutOfRange(id) => ApplyError::IdPastEndOfSpace { kind, id },
    }
}

fn apply_record(
    g: &mut Graph,
    record: Record,
    ops: &mut BufferOps,
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
            // No reservation to make: these ids came from the master, not from
            // this graph's allocator. The graph refuses rather than
            // double-counting, so there is no separate check here to keep in step
            // with it either.
            g.create_nodes(&nodes, &mut ops.nodes)
                .map_err(|e| node_op("node", e))?;

            // The graph's bulk APIs take `&[u64]`, so the ids are materialized
            // once here rather than per call.
            let ids: Vec<u64> = ids.iter().collect();
            // Resolved once and used twice. The attribute call needs the same
            // label set to know which label-scoped indexes an indexed property
            // belongs to, and deriving it there instead meant a
            // `node_labels_matrix.iter(id, id)` per created node — a
            // delta-matrix iteration for a question this record has already
            // answered, on the path where a bulk create makes it a million of
            // them.
            let label_ids = checked_label_ids(g, &labels)?;
            if !label_ids.is_empty() {
                g.set_node_labels_product(&ids, &label_ids, &mut ops.docs.node_adds, true);
            }
            if !attr_ids.is_empty() {
                check_attr_shape(g, &ids, &attr_ids, &rows)?;
                g.set_nodes_attributes_rows_of_labels(
                    &ids,
                    &label_ids,
                    &attr_ids,
                    &rows,
                    &mut ops.docs.node_adds,
                )?;
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
            // `&[u64]` for the bulk APIs; materialized once each.
            let (ids, src, dst): (Vec<u64>, Vec<u64>, Vec<u64>) = (
                ids.iter().collect(),
                src.iter().collect(),
                dst.iter().collect(),
            );
            g.create_relationships_bulk(&type_name, &src, &dst, &ids, Some(&mut ops.edges))
                .map_err(|e| node_op("relationship", e))?;

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
            // Two ways a delete can name something that is not live, and the
            // graph answers one of them by itself: an id already in the recycle
            // bin. The other — at or above the boundary this buffer started from
            // and never created by it, so nothing has ever held it — needs the
            // batch, which is why it is handed over here.
            g.delete_nodes(&nodes, &mut ops.docs.node_removes, Some(&ops.nodes))
                .map_err(|e| node_op("node", e))?;
            Ok(())
        }

        Record::DeleteEdge { ids, .. } => {
            let edges = ids.to_roaring();
            g.delete_relationships(&edges, &mut ops.docs.edge_removes, Some(&ops.edges))
                .map_err(|e| node_op("relationship", e))?;
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
    id: u32,
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
    id: u32,
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
    relation_id: u32,
) -> Result<Arc<String>, ApplyError> {
    g.get_type(TypeId(relation_id as usize))
        .ok_or(ApplyError::IdOutOfRange {
            kind: "relationship type",
            id: i64::from(relation_id),
        })
}

/// An `UPDATE_EDGE`'s relationship type, checked against this graph.
///
/// The same check C's `ApplyUpdateEdge` makes — after reading the record, it
/// refuses one whose `r_id` is negative or past the local edge-schema count,
/// logging "references relationship type %d which doesn't exist locally"
/// (`src/effects/update_edge_effect.c`). A replica that has not
/// seen the `ADD_SCHEMA` yet must fail here rather than index the rows under a
/// type it invented.
fn checked_type_id(
    g: &Graph,
    relation_id: Option<u32>,
) -> Result<TypeId, ApplyError> {
    let relation_id = relation_id.ok_or(ApplyError::MissingRelType)?;
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
    labels: &[u32],
) -> Result<Vec<u64>, ApplyError> {
    let bound = g.get_labels().len();
    labels
        .iter()
        .map(|&label| {
            // No `< 0` arm: the wire field is unsigned, so a sentinel cannot
            // arrive as one. A peer that wrote C's -1 into these four bytes
            // arrives here as 4294967295 and fails the same bound, which
            // `IdOutOfRange` renders both ways.
            if label as usize >= bound {
                return Err(ApplyError::IdOutOfRange {
                    kind: "label",
                    id: i64::from(label),
                });
            }
            Ok(u64::from(label))
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
    // no such guarantee: `AttributeSet_Add` appends, and `AttributeSet_Contains`
    // finds a key by scanning the ids linearly, under a standing
    // `// TODO: use SIMD or support sort` (`src/graph/entities/attribute_set.c`).
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
    options: Option<&WireIndexOptions>,
) -> Result<Option<IndexOptions>, ApplyError> {
    // No string matching and no type checking left here. The wire block is
    // already typed, so what used to be `map_to_index_options` — looking up
    // "dim", asserting it is an integer, rejecting an unknown key — happened on
    // the primary at parse time. A malformed option is now a decode error on a
    // payload rather than an apply error on a graph.
    let Some(o) = options else {
        return Ok(None);
    };
    Ok(match index_type {
        IndexType::Vector => match o.vector {
            None => None,
            Some(v) => {
                // The wire holds `dimension` as a `u64` because it is the one
                // vector option with no absent form, not because either engine
                // is that wide: C types it `uint32_t` (`src/index/index_field.h`
                // line 45), and so does this one. The `size_t` in that struct
                // belongs to the next three fields, `M`/`efConstruction`/
                // `efRuntime` — which is why those cast plainly below and this
                // does not. The wire can therefore carry a dimension neither
                // engine can hold, and `as u32` would turn 2^32 into 0 and
                // index against it, so the narrowing is checked and the payload
                // refused.
                let dimension = u32::try_from(v.dimension).map_err(|_| {
                    ApplyError::UnsupportedIndexOption(format!(
                        "vector dimension {} exceeds this engine's limit of {}",
                        v.dimension,
                        u32::MAX
                    ))
                })?;
                // The VecSimMetric numbering the RDB persists. An unrecognised
                // discriminant is refused rather than read as L2: silently
                // choosing a metric would build an index that answers the
                // wrong queries and never say so.
                let similarity_function = match v.sim_func {
                    None => None,
                    Some(0) => Some("euclidean".to_owned()),
                    Some(1) => Some("ip".to_owned()),
                    Some(2) => Some("cosine".to_owned()),
                    Some(other) => {
                        return Err(ApplyError::UnsupportedIndexOption(format!(
                            "vector similarity function {other}"
                        )));
                    }
                };
                Some(IndexOptions::Vector(VectorIndexOptions {
                    dimension,
                    m: v.m.map(|x| x as usize),
                    ef_construction: v.ef_construction.map(|x| x as usize),
                    ef_runtime: v.ef_runtime.map(|x| x as usize),
                    similarity_function,
                }))
            }
        },
        IndexType::Fulltext => {
            // Absent stays absent. Materialising a default here is what
            // diverged a replica: an explicit language is refused when one is
            // already set for the label, so "nothing said" has to survive the
            // wire as nothing said.
            let phonetic = match o.phonetic.as_deref() {
                None => None,
                Some(s) if s.eq_ignore_ascii_case("dm:en") => Some(true),
                Some("") => Some(false),
                Some(s) => {
                    return Err(ApplyError::UnsupportedIndexOption(format!(
                        "phonetic algorithm '{s}'"
                    )));
                }
            };
            Some(IndexOptions::Text(TextIndexOptions {
                weight: o.weight,
                nostem: o.nostem,
                phonetic,
                language: o.language.as_ref().map(|l| Arc::new(l.clone())),
                stopwords: o
                    .stopwords
                    .as_ref()
                    .map(|sw| sw.iter().map(|s| Arc::new(s.clone())).collect()),
            }))
        }
        // A range index takes none of these.
        IndexType::Range => None,
    })
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
    use crate::effects::EffectEncode;
    use crate::effects::v3::records::VectorOptions as WireVectorOptions;
    use crate::effects::v3::staging::StagePending;
    use crate::effects::v3::test_aux::graph;
    use crate::effects::v3::{AttrRef, INDEX_FLD_RANGE, IdList, Record, new_buffer};
    use crate::graph::constraint::{ConstraintStatus, ConstraintType};

    /// The delete-then-recreate cycle a replica legitimately sees.
    fn write_delete(
        buf: &mut Vec<u8>,
        ids: &IdList,
        labels: &[u32],
    ) {
        Record::DeleteNode {
            ids: ids.clone(),
            labels: labels.to_vec(),
        }
        .encode(buf);
    }

    #[test]
    fn recreating_a_recycled_id_is_allowed() {
        // The case the check must not break: the primary deleted 1, so 1 is in
        // the bin here too, and it is free to come back.
        let mut g = graph();
        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([0, 1, 2]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([1]), &[]);
        apply_effects(&mut g, &buf).expect("delete must apply");

        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([1]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("recreating a recycled id must apply");
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn creating_a_fresh_id_past_the_last_allocated_one_is_allowed() {
        // Sequential allocation on the primary can outrun this replica's
        // last allocated id without anything being wrong.
        let mut g = graph();
        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([0, 1]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([2, 3]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
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
        Record::CreateNode {
            ids: IdList::from([0, 1, 2]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([1]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(
                err,
                ApplyError::AlreadyLive {
                    kind: "node",
                    id: 1,
                    ..
                }
            ),
            "{err}"
        );
        assert_eq!(g.node_count(), 3, "the buffer must not have been applied");
    }

    #[test]
    fn one_buffer_claiming_an_id_twice_aborts() {
        // Neither record's ids were live on entry, so the entry boundary alone
        // cannot see this — id 1 is above it and reads as fresh both times. What
        // sees it is the intersection with what the buffer has already created,
        // and because the recycle bin has been removed from the candidates first,
        // it does not fire on the legitimate delete-then-recreate that
        // `a_buffer_may_create_delete_and_recreate_the_same_id` covers.
        let mut g = graph();
        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([0, 1]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        Record::CreateNode {
            ids: IdList::from([1, 2]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(
                err,
                ApplyError::AlreadyLive {
                    kind: "node",
                    id: 1,
                    ..
                }
            ),
            "{err}"
        );
        assert_eq!(
            g.node_count(),
            2,
            "refused at the record, so the second one applied nothing"
        );
    }

    #[test]
    fn a_buffer_may_create_a_node_and_then_delete_it() {
        // The false positive the delete check has to avoid: id 1 is at or above
        // the boundary the buffer started from, so "never allocated here" is what
        // it looks like to anything that only knows that boundary. What makes it
        // legitimate is that this buffer created it, which is exactly what the
        // ingested set remembers.
        let mut g = graph();
        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([0, 1]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        write_delete(&mut buf, &IdList::from([1]), &[]);
        apply_effects(&mut g, &buf).expect("create-then-delete must apply");
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn deleting_an_already_recycled_id_aborts_the_buffer() {
        let mut g = graph();
        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([0, 1]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([1]), &[]);
        apply_effects(&mut g, &buf).expect("delete must apply");

        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([1]), &[]);
        let err = apply_effects(&mut g, &buf).expect_err("must refuse a double delete");
        assert!(
            matches!(
                err,
                ApplyError::NotLive {
                    kind: "node",
                    id: 1,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn deleting_a_never_allocated_id_aborts_the_buffer() {
        let mut g = graph();
        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([0, 1]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([99]), &[]);
        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(
                err,
                ApplyError::NotLive {
                    kind: "node",
                    id: 99,
                    ..
                }
            ),
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
        Record::AddSchema {
            schema_type: EntityType::Node,
            id: 0,
            name: "L".to_owned(),
        }
        .encode(&mut buf);
        Record::AddAttribute {
            id: 0,
            name: "keep".to_owned(),
        }
        .encode(&mut buf);
        Record::AddAttribute {
            id: 1,
            name: "drop".to_owned(),
        }
        .encode(&mut buf);
        Record::CreateNode {
            ids: IdList::from([0, 1]),
            labels: vec![0],
            attr_ids: vec![0, 1],
            rows: vec![Value::Int(1), Value::Int(10), Value::Int(2), Value::Int(20)],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("create must apply");
        assert_eq!(
            g.get_node_attribute(0.into(), &Arc::new("drop".into())),
            Some(Value::Int(10))
        );

        // Now null it out, as an UPDATE_NODE would.
        let mut buf = new_buffer();
        crate::effects::v3::Record::Update {
            entity: EntityType::Node,
            ids: IdList::from([0, 1]),
            labels: vec![0],
            relation_id: None,
            attr_ids: vec![1],
            rows: vec![Value::Null, Value::Null],
        }
        .encode(&mut buf);
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
        Record::AddSchema {
            schema_type: EntityType::Node,
            id: 0,
            name: "A".to_owned(),
        }
        .encode(&mut buf);
        Record::CreateNode {
            ids: IdList::from([0, 1, 2]),
            labels: vec![0],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("first buffer must apply");

        // A second buffer introducing a label and immediately applying it.
        let mut buf = new_buffer();
        Record::AddSchema {
            schema_type: EntityType::Node,
            id: 1,
            name: "B".to_owned(),
        }
        .encode(&mut buf);
        Record::Labels {
            add: true,
            ids: IdList::from([0, 1, 2]),
            labels: vec![1],
        }
        .encode(&mut buf);
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
        Record::CreateNode {
            ids: IdList::from([0, 1, 2]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("create must apply");

        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([0, 1, 2]), &[]);
        apply_effects(&mut g, &buf).expect("delete must apply");
        assert_eq!(g.node_count(), 0);

        // Recycled: every id is in the bin, so all three may come back.
        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([1]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("a recycled id must apply");

        // Fresh: the next id never handed out. Id 3 rather than an arbitrary
        // high one — a master allocates the lowest free id, so it cannot reach
        // 99 without having handed out everything below it, and a buffer that
        // claims otherwise is divergence rather than a fresh create. That case
        // is `a_buffer_that_jumps_the_bound_is_refused`.
        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([3]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("a fresh id must apply");

        // And a delete of something already binned is still refused.
        let mut buf = new_buffer();
        write_delete(&mut buf, &IdList::from([0]), &[]);
        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(
                err,
                ApplyError::NotLive {
                    kind: "node",
                    id: 0,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn a_buffer_that_jumps_the_bound_is_refused() {
        // Creating id 9 on a graph that has handed out nothing leaves 0..8
        // allocated and never created. No master produces that: it allocates the
        // lowest free id, so reaching 9 means it created everything below, and a
        // replica that accepted this would hold an id space its master does not
        // have. Refused as the divergence it is, and the whole buffer is dropped.
        let mut g = graph();
        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([9]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(
                err,
                ApplyError::IdsHaveAHole {
                    kind: "node",
                    entry_bound: 0,
                    highest: 9,
                    created: 1,
                }
            ),
            "{err}"
        );
    }

    /// A buffer that creates two nodes and one edge between them, as setup.
    fn edge_setup(buf: &mut Vec<u8>) {
        Record::AddSchema {
            schema_type: EntityType::Relationship,
            id: 0,
            name: "R".to_owned(),
        }
        .encode(buf);
        Record::CreateNode {
            ids: IdList::from([0, 1]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(buf);
    }

    fn write_create_edge(
        buf: &mut Vec<u8>,
        edge_ids: &[u64],
    ) {
        Record::CreateEdge {
            ids: edge_ids.iter().copied().collect(),
            relation_id: 0,
            src: std::iter::repeat_n(0u64, edge_ids.len()).collect(),
            dst: std::iter::repeat_n(1u64, edge_ids.len()).collect(),
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(buf);
    }

    #[test]
    fn an_edge_id_claimed_twice_in_one_buffer_aborts() {
        // Edges get the same id space as nodes. Before they did, this buffer was
        // applied without complaint and the replica's edge ids drifted from the
        // master's — invisible until a promotion, because edges enumerate from
        // the tensor rather than from a counter, so a wrong count cannot fuse two
        // of them the way it fuses two nodes.
        let mut g = graph();
        let mut buf = new_buffer();
        edge_setup(&mut buf);
        write_create_edge(&mut buf, &[0, 1]);
        write_create_edge(&mut buf, &[1, 2]);

        let err = apply_effects(&mut g, &buf).expect_err("edge 1 is claimed twice");
        assert!(
            matches!(
                err,
                ApplyError::AlreadyLive {
                    kind: "relationship",
                    id: 1,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn an_edge_buffer_that_jumps_the_bound_is_refused() {
        // The replica-missed-a-buffer case, on the edge side: reaching edge id 9
        // means the master handed out 0..8, and this replica was told about none
        // of them.
        let mut g = graph();
        let mut buf = new_buffer();
        edge_setup(&mut buf);
        write_create_edge(&mut buf, &[9]);

        let err = apply_effects(&mut g, &buf).expect_err("must refuse");
        assert!(
            matches!(
                err,
                ApplyError::IdsHaveAHole {
                    kind: "relationship",
                    entry_bound: 0,
                    highest: 9,
                    created: 1,
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn deleting_an_edge_never_allocated_aborts_the_buffer() {
        let mut g = graph();
        let mut buf = new_buffer();
        edge_setup(&mut buf);
        write_create_edge(&mut buf, &[0]);
        apply_effects(&mut g, &buf).expect("setup must apply");

        let mut buf = new_buffer();
        Record::DeleteEdge {
            ids: IdList::from([7]),
            relation_id: 0,
            src: IdList::from([0]),
            dst: IdList::from([1]),
        }
        .encode(&mut buf);
        let err = apply_effects(&mut g, &buf).expect_err("edge 7 was never allocated");
        assert!(
            matches!(
                err,
                ApplyError::NotLive {
                    kind: "relationship",
                    id: 7,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn a_buffer_may_create_delete_and_recreate_the_same_id() {
        // A multi-commit query commits into *one* buffer, and the id allocator
        // recycles a freed id across commits, so `C(0) · D(0) · C(0)` is what
        // `CREATE (n) WITH n DELETE n WITH 1 AS z CREATE ()` actually ships.
        //
        // This is why the ingested set does not shrink on a delete: id 0 was
        // handed out once, and the recreate is the allocator reusing it rather
        // than the buffer claiming it twice. A set that dropped it on the delete
        // would have to decide which of those it was looking at, and cannot.
        let mut g = graph();
        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([0]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        write_delete(&mut buf, &IdList::from([0]), &[]);
        Record::CreateNode {
            ids: IdList::from([0]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("the recreate is legitimate");
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn a_buffer_applies_end_to_end() {
        let mut g = graph();
        let mut buf = new_buffer();
        Record::AddSchema {
            schema_type: EntityType::Node,
            id: 0,
            name: "Person".to_owned(),
        }
        .encode(&mut buf);
        Record::AddAttribute {
            id: 0,
            name: "name".to_owned(),
        }
        .encode(&mut buf);
        Record::CreateNode {
            ids: IdList::from([0, 1, 2]),
            labels: vec![0],
            attr_ids: vec![0],
            rows: vec![
                Value::String(Arc::new("a".into())),
                Value::String(Arc::new("b".into())),
                Value::String(Arc::new("c".into())),
            ],
        }
        .encode(&mut buf);

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
        Record::AddSchema {
            schema_type: EntityType::Node,
            id: 0,
            name: "Person".to_owned(),
        }
        .encode(&mut buf);

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
        Record::AddAttribute {
            id: 0,
            name: "name".to_owned(),
        }
        .encode(&mut buf);

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
        Record::Index {
            create: true,
            schema_type: EntityType::Node,
            label_id: 0,
            label: "Expected".to_owned(),
            field_type: INDEX_FLD_RANGE,
            fields: vec![AttrRef {
                id: 0,
                name: "a".to_owned(),
            }],
            options: Some(WireIndexOptions::none_given(None)),
        }
        .encode(&mut buf);
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
        Record::AddSchema {
            schema_type: EntityType::Node,
            id: 0,
            name: "L".to_owned(),
        }
        .encode(&mut buf);
        Record::CreateNode {
            ids: IdList::from([0, 1, 2, 3]),
            labels: vec![],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);
        Record::Labels {
            add: true,
            ids: IdList::from([0, 1, 2, 3]),
            labels: vec![0],
        }
        .encode(&mut buf);

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
            Record::Constraint {
                create: true,
                constraint_type: ConstraintType::Unique,
                entity_type: EntityType::Node,
                status: Some(status),
                label_id: 0,
                label: "Person".to_owned(),
                props: vec![AttrRef {
                    id: 0,
                    name: "email".to_owned(),
                }],
            }
            .encode(&mut buf);
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
        Record::AddSchema {
            schema_type: EntityType::Node,
            id: 0,
            name: "Person".to_owned(),
        }
        .encode(&mut buf);
        Record::AddAttribute {
            id: 0,
            name: "email".to_owned(),
        }
        .encode(&mut buf);
        Record::Constraint {
            create: true,
            constraint_type: ConstraintType::Unique,
            entity_type: EntityType::Node,
            status: Some(ConstraintStatus::Operational),
            label_id: 0,
            label: "Person".to_owned(),
            props: vec![AttrRef {
                id: 0,
                name: "email".to_owned(),
            }],
        }
        .encode(&mut buf);

        apply_effects(&mut g, &buf).expect("must apply");
        assert_eq!(g.constraints().len(), 1);
    }

    #[test]
    fn a_malformed_buffer_is_refused_not_applied() {
        let mut g = graph();
        let mut buf = new_buffer();
        Record::AddSchema {
            schema_type: EntityType::Node,
            id: 0,
            name: "L".to_owned(),
        }
        .encode(&mut buf);
        Record::CreateNode {
            ids: IdList::from([0, 1]),
            labels: vec![0],
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);

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

        let mut buf = crate::effects::v3::new_buffer();
        crate::effects::v3::emit::for_each_record(&p, &master, |r| r.encode(&mut buf));
        assert_eq!(
            crate::effects::v3::test_aux::read_buffer(&buf)
                .unwrap()
                .len(),
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
            Record::CreateNode {
                ids: IdList::from([0]),
                labels: vec![],
                attr_ids: vec![],
                rows: vec![],
            }
            .encode(&mut buf);
            apply_effects(&mut g, &buf).expect("setup");

            let mut buf = new_buffer();
            Record::Update {
                entity: EntityType::Node,
                ids: IdList::from([0]),
                labels: vec![],
                relation_id: None,
                attr_ids: bad.to_vec(),
                rows: vec![Value::Int(1), Value::Int(2)],
            }
            .encode(&mut buf);
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
        Record::AddAttribute {
            id: 2,
            name: "q".to_owned(),
        }
        .encode(&mut buf);
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
        Record::AddAttribute {
            id: 1,
            name: "q".to_owned(),
        }
        .encode(&mut buf);
        apply_effects(&mut g, &buf).expect("the real id of 'q' is 1");

        // A genuinely new name still lands on the next id.
        let mut buf = new_buffer();
        Record::AddAttribute {
            id: 3,
            name: "s".to_owned(),
        }
        .encode(&mut buf);
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

    /// An option nobody stated must not arrive as one somebody did.
    ///
    /// This is what `test_CRUD_replication` caught when the wire wrote the
    /// RDB's defaults for absent options: `create_index` refuses an explicit
    /// language once one is set for the label, so a materialised "english"
    /// diverged a replica that was otherwise in step.
    #[test]
    fn an_absent_index_option_stays_absent_through_apply() {
        let opts = index_options(
            &IndexType::Fulltext,
            Some(&WireIndexOptions::none_given(None)),
        )
        .expect("nothing stated is not an error")
        .expect("a fulltext index carries a text block");
        let IndexOptions::Text(t) = opts else {
            panic!("expected the text half of a fulltext index");
        };
        assert_eq!(
            (&t.weight, &t.nostem, &t.phonetic, &t.language, &t.stopwords),
            (&None, &None, &None, &None, &None),
            "no option may be materialised on the way in"
        );
    }

    /// The wire holds `dimension` as a `u64`; both engines hold 32 bits.
    #[test]
    fn a_vector_dimension_too_large_for_this_engine_is_refused() {
        // Not `as u32`, which would make this 0 and index against it.
        let over = u64::from(u32::MAX) + 1;
        let Err(err) = index_options(
            &IndexType::Vector,
            Some(&WireIndexOptions::none_given(Some(
                WireVectorOptions::of_dimension(over),
            ))),
        ) else {
            panic!("a dimension this engine cannot hold must be refused");
        };
        assert!(
            matches!(err, ApplyError::UnsupportedIndexOption(ref m) if m.contains("4294967296")),
            "{err}"
        );
    }

    /// An unknown metric is refused, not read as L2.
    #[test]
    fn an_unrecognised_similarity_function_is_refused() {
        // Choosing a metric on the replica's behalf would build an index that
        // answers different queries from the primary's and never say so.
        let mut v = WireVectorOptions::of_dimension(4);
        v.sim_func = Some(7);
        let Err(err) = index_options(
            &IndexType::Vector,
            Some(&WireIndexOptions::none_given(Some(v))),
        ) else {
            panic!("an unknown VecSimMetric must be refused");
        };
        assert!(
            matches!(err, ApplyError::UnsupportedIndexOption(ref m) if m.contains('7')),
            "{err}"
        );

        // And the three this engine does implement still map.
        for (code, name) in [(0, "euclidean"), (1, "ip"), (2, "cosine")] {
            let mut v = WireVectorOptions::of_dimension(4);
            v.sim_func = Some(code);
            let opts = index_options(
                &IndexType::Vector,
                Some(&WireIndexOptions::none_given(Some(v))),
            )
            .unwrap()
            .unwrap();
            let IndexOptions::Vector(got) = opts else {
                panic!("expected the vector half");
            };
            assert_eq!(got.similarity_function.as_deref(), Some(name));
        }
    }
}
