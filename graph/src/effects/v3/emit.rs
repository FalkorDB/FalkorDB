//! Turning a committed `Pending` into v3 effect records.
//!
//! The mirror of `effects_v3_apply`, and deliberately shaped like it:
//!
//! ```text
//!   emit    Pending  --digest-->  Record  --encode-->  bytes
//!   apply   bytes    --decode-->  Record  --apply-->   Graph
//! ```
//!
//! `Record` is the pivot both directions turn on. Digesting into it rather than
//! writing bytes straight out of `Pending` costs nothing — the grouping already
//! allocated the ids, labels, attribute ids and row values that a `Record`
//! holds — and it buys the grouping its own testable surface: the tests below
//! assert on records, not on decoded buffers.
//!
//! Keeping this out of `pending.rs` is the other half. A mutation accumulator
//! should not know what a replication frame looks like, so `Pending`'s fields
//! are `pub(crate)` and the wire knowledge lives here.

use super::records::IndexFieldOptions;
use crate::index::indexer::IndexOptions as EngineIndexOptions;
use crate::runtime::runtime::map_to_index_options;
use atomic_refcell::AtomicRefCell;
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;

use crate::{
    effects::EffectWrite,
    effects::v3::{self as v3, EffectEncode, IdList, Record},
    entity_type::EntityType,
    graph::graph::{DeletedEdge, Graph, NodeId, RelationshipId},
    index::IndexType,
    runtime::{pending::Pending, value::Value},
};

use crate::effects::announce::{AnnouncedConstraint, AnnouncedIndex, SchemaBaseline};

/// A schema id as the wire carries it: C's `LabelID`/`RelationID`, which is a
/// signed `int`.
///
/// Checked rather than `as i32`, and the sentinels are why. C reserves the
/// negatives (`src/graph/graph.h` lines 20-23) — `GRAPH_NO_LABEL` and
/// `GRAPH_NO_RELATION` are -1, `GRAPH_UNKNOWN_LABEL` and
/// `GRAPH_UNKNOWN_RELATION` are -2 — so a truncating
/// cast of a `usize` past `i32::MAX` would not merely be a wrong id, it would
/// be one of C's "there is no label here" values, and the replica would act on
/// it rather than reject it.
///
/// Unreachable in practice: a graph does not hold two billion labels. That is
/// exactly the argument that was made for `props.len() as u8`, which silently
/// wrote 0 for a 256th property, so it is made loudly here instead.
fn schema_id(id: usize) -> u32 {
    u32::try_from(id).expect("a schema id must fit four bytes; the dictionary cannot be this large")
}

/// Announce every schema and attribute registered since `baseline`.
pub fn emit_schema_additions(
    g: &Graph,
    baseline: &SchemaBaseline,
    out: &mut impl FnMut(Record),
) {
    for (offset, label) in g.get_labels().iter().enumerate().skip(baseline.labels) {
        out(Record::AddLabel {
            id: schema_id(offset),
            name: label.to_string(),
        });
    }
    for (offset, rel_type) in g.get_types().iter().enumerate().skip(baseline.types) {
        out(Record::AddRelType {
            id: schema_id(offset),
            name: rel_type.to_string(),
        });
    }
    // Once, not once per entity kind. #2459 unified the two attribute
    // dictionaries, so both getters return the same store — and unlike v2's,
    // v3's ADD_ATTRIBUTE has no node/relationship discriminator, so announcing
    // per kind would send every new attribute twice under the same id.
    for (offset, attr) in g
        .get_node_attribute_names()
        .iter()
        .enumerate()
        .skip(baseline.attrs)
    {
        out(Record::AddAttribute {
            id: offset as u16,
            name: attr.to_string(),
        });
    }
}

/// Build the payload announcing one constraint.
///
/// A constraint is announced **twice** — once when it is created, still under
/// construction, and again when validation finishes and it is operational or
/// failed — so this is called with the same constraint and a different status.
/// The apply side upserts, so the second announcement converges on the first
/// rather than duplicating it.
///
/// `baseline` is the schema state from before the constraint was created:
/// `create_constraint` registers the label and the property names, and those
/// registrations have to travel ahead of the record whose ids depend on them.
///
/// # Errors
///
/// Returns an error if a property is not registered, which would mean
/// `create_constraint` did not run or did not register it.
pub fn build_constraint_buffer<W: EffectWrite + ?Sized>(
    g: &Graph,
    create: bool,
    c: &AnnouncedConstraint<'_>,
    baseline: &SchemaBaseline,
    buf: &mut W,
) -> Result<(), String> {
    let AnnouncedConstraint {
        ct,
        entity_type,
        status,
        label,
        properties,
    } = *c;
    let label_id = match entity_type {
        EntityType::Node => g.get_label_id(label).map(|l| l.0),
        EntityType::Relationship => g.get_type_id(label).map(|t| t.0),
    }
    .map(schema_id)
    .ok_or_else(|| format!("constraint label '{label}' is not registered"))?;

    let names = g.get_node_attribute_names();
    let props: Vec<v3::AttrRef<String>> = properties
        .iter()
        .map(|p| {
            names
                .iter()
                .position(|n| n == p)
                .map(|i| v3::AttrRef {
                    id: i as u16,
                    name: p.as_str().to_owned(),
                })
                .ok_or_else(|| format!("constraint property '{p}' is not registered"))
        })
        .collect::<Result<_, _>>()?;

    // Through `Record`, like everything else. This was the one production path
    // that reached past the pivot into a record writer, which is also why it
    // was the one that needed a `ConstraintSpec` to keep its argument list
    // legible.
    //
    // Built before anything is written. `encode` is fallible now, and a record
    // refused halfway leaves a prefix behind for the next one to be appended
    // after — so every reason to refuse is settled here, while the buffer is
    // still untouched. A create carries the primary's status and a drop has no
    // field for one, which is the one thing this can get wrong.
    let record = if create {
        Record::CreateConstraint {
            constraint_type: ct,
            entity_type,
            status: status.ok_or_else(|| {
                format!("constraint create on '{label}' carries no status to replicate")
            })?,
            label_id,
            label: label.to_owned(),
            props,
        }
    } else {
        Record::DropConstraint {
            constraint_type: ct,
            entity_type,
            label_id,
            label: label.to_owned(),
            props,
        }
    };

    encode_all(buf, g, baseline)?;
    record.encode(buf).map_err(|e| e.to_string())
}

/// Encode the schema announcements, stopping at the first refusal.
///
/// A helper rather than a closure at each call site because `encode` is
/// fallible and `emit_schema_additions` hands records to a `FnMut(Record)`
/// that cannot return one.
fn encode_all<W: EffectWrite + ?Sized>(
    buf: &mut W,
    g: &Graph,
    baseline: &SchemaBaseline,
) -> Result<(), String> {
    let mut failed = None;
    emit_schema_additions(g, baseline, &mut |record| {
        if failed.is_none()
            && let Err(e) = record.encode(buf)
        {
            failed = Some(e);
        }
    });
    failed.map_or(Ok(()), |e| Err(e.to_string()))
}

/// `IndexFieldType` is a bit flag set rather than a discriminant, so this is a
/// union not a cast. The inverse is `apply::index_type_of`.
const fn index_field_flags(t: &IndexType) -> u32 {
    match t {
        IndexType::Range => v3::INDEX_FLD_RANGE,
        IndexType::Fulltext => v3::INDEX_FLD_FULLTEXT,
        IndexType::Vector => v3::INDEX_FLD_VECTOR,
    }
}

/// Encode one index DDL statement, schema announcements included.
///
/// One record per *statement*, not per field. C sends one per field
/// (`src/effects/create_index_effect.c`) and gets away with it by making the
/// index-level work idempotent, as that file says: `Index_SetLanguage` tolerates
/// being re-set to the same value, and `Index_SetStopwords` is guarded by
/// `Index_ContainsStopwords` because it does not.
///
/// This engine has no such guard, so two single-field records are not equivalent
/// to one two-field statement here: applying the second is refused with "Can not
/// override index configuration", because index-level options belong to the
/// index and cannot be set twice. `field_type` therefore sits at the statement
/// level — a drop sends an empty field list, and a per-field type would vanish
/// along with the fields.
pub fn build_index_buffer<W: EffectWrite + ?Sized>(
    p: &Pending,
    g: &AtomicRefCell<Graph>,
    create: bool,
    ix: &AnnouncedIndex<'_>,
    buf: &mut W,
) -> Result<(), String> {
    let g = &g.borrow();
    // Derived here rather than by the caller, for the same reason
    // `build_effects_buffer` derives it: which schema entries still need
    // announcing is this module's business, not the runtime's.
    let baseline = &SchemaBaseline {
        labels: p.schema_label_count,
        types: p.schema_rel_type_count,
        attrs: p.schema_node_attr_count,
    };
    let label_id = match ix.entity_type {
        EntityType::Node => g.get_label_id(ix.label).map(|l| l.0),
        EntityType::Relationship => g.get_type_id(ix.label).map(|t| t.0),
    }
    .map(schema_id)
    .ok_or_else(|| format!("index label '{}' is not registered", ix.label))?;

    let names = g.get_node_attribute_names();
    let fields: Vec<v3::AttrRef<String>> = ix
        .fields
        .iter()
        .map(|f| {
            names
                .iter()
                .position(|n| n == f)
                .map(|i| v3::AttrRef {
                    id: i as u16,
                    name: f.as_str().to_owned(),
                })
                .ok_or_else(|| format!("index field '{f}' is not registered"))
        })
        .collect::<Result<_, _>>()?;

    // Built before anything is written, so a refusal leaves the buffer
    // untouched rather than torn. A drop has no options field at all; a create
    // always states its block, with a presence byte per option, so "nothing
    // said" survives the wire as nothing said.
    let record = if create {
        Record::CreateIndex {
            schema_type: ix.entity_type,
            label_id,
            label: ix.label.to_owned(),
            field_type: index_field_flags(ix.index_type),
            fields,
            options: wire_index_options(ix),
        }
    } else {
        Record::DropIndex {
            schema_type: ix.entity_type,
            label_id,
            label: ix.label.to_owned(),
            field_type: index_field_flags(ix.index_type),
            fields,
        }
    };

    encode_all(buf, g, baseline)?;
    record.encode(buf).map_err(|e| e.to_string())
}

/// A record's partition key: its label set and its attribute ids.
type Shape = (Vec<u32>, Vec<u16>);

/// Every record a committed `Pending` implies, in apply order, handed to `out`
/// one at a time.
///
/// "Apply order" is the sequence of the calls below, and it is the only
/// ordering the format requires: schema announcements before the records whose
/// ids depend on them, and edges deleted before the nodes they hang off. Within
/// one of those stages the records are disjoint — every entity belongs to
/// exactly one group — so their order among themselves carries no meaning and
/// is not imposed.
///
/// Eager in the grouping — a shape's members are only known once every entity
/// has been seen — but records go out as they are built, so peak memory is one
/// stage's group map plus one record rather than every record at once. The
/// mirror of the apply path, which decodes one record and applies it rather
/// than materializing them all.
///
/// The shapes are **not** pre-grouped inside `Pending`, deliberately.
/// Attributes accumulate incrementally — `set_node_attribute` inserts into a
/// sorted vec — so `CREATE (n) SET n.a=1 SET n.b=2` changes that node's shape
/// twice, and pre-bucketing would mean re-bucketing on every attribute write.
/// It would also charge every write for a grouping most never use, since
/// effects are only built when something is actually replicating.
pub fn for_each_record(
    p: &Pending,
    g: &AtomicRefCell<Graph>,
    mut out: impl FnMut(Record),
) {
    let out = &mut out;

    // Schema and attribute additions first, each carrying its id. That ordering
    // is normative in v3, not incidental: the ids on every later record are
    // only meaningful once the replica has agreed on the numbering.
    {
        let graph = g.borrow();
        emit_schema_additions(
            &graph,
            &SchemaBaseline {
                labels: p.schema_label_count,
                types: p.schema_rel_type_count,
                attrs: p.schema_node_attr_count,
            },
            out,
        );
    }

    digest_created_nodes(p, out);
    digest_created_edges(p, g, out);
    {
        // The entities still exist here — effects are encoded after `commit`
        // applies — so a node's labels and an edge's type are read straight off
        // the graph.
        let graph = g.borrow();
        digest_updates(
            &p.existing_nodes_attrs,
            &p.deleted_nodes,
            EntityType::Node,
            &graph,
            out,
        );
        digest_updates(
            &p.existing_relationships_attrs,
            &p.deleted_relationships,
            EntityType::Relationship,
            &graph,
            out,
        );
    }
    // A created node carries its labels in its CREATE_NODE record already.
    digest_labels(&p.set_labels, Some(&p.created_nodes), true, out);
    digest_labels(&p.remove_labels, None, false, out);
    // Edges before nodes, so the replica unhooks first.
    digest_deleted_edges(p, out);
    digest_deleted_nodes(p, out);
    digest_cancelled_nodes(p, out);
}

/// A node created and deleted inside one segment, as the create and the delete
/// the master skipped.
///
/// The master takes a shortcut for these — `delete_pending_node` unwinds the
/// node out of `Pending` and `return_node_id` hands the id back — so nothing
/// it commits mentions the id at all. The buffer said nothing either, and the
/// replica's id space came out one short: its next allocation after a
/// promotion landed on a node that was already live, fusing two entities.
///
/// Emitting the pair is what C's payload carries for the same query — C has no
/// such shortcut, so its `CREATE (a:A), (b:B) DELETE a` puts two CREATE_NODEs
/// and a DELETE_NODE on the wire — and it needs no record type v3 does not
/// already have. The replica creates the node and deletes it, ending with the
/// id in its recycle bin exactly as the master has it.
///
/// Last, and as a pair. A later commit in the same query may reclaim this id
/// from the bin, and its `CREATE_NODE` is appended after these — so the delete
/// has to be in the buffer before that create, not merely somewhere in it.
fn digest_cancelled_nodes(
    p: &Pending,
    out: &mut impl FnMut(Record),
) {
    if p.cancelled_nodes.is_empty() {
        return;
    }
    // Ids and nothing else.
    //
    // The pair is here to move the replica's allocator over an id the master
    // handed out and then took back — not to reproduce a node that did not
    // survive its own transaction. Carrying its labels and attributes would
    // have the replica build a node and then immediately unbuild it: the
    // create adds index documents, the delete a record later removes them, and
    // the net effect on the graph is the id space and nothing besides.
    //
    // `DeleteNode` ignores its label list on the way in regardless — `apply`
    // binds it as `labels: _` — so on that half they were never read.
    //
    // No grouping by shape either. A shape only earns its grouping when the
    // content travels with it; with no content, every cancelled node in the
    // buffer is one record pair however many labels they had between them.
    let mut ids: Vec<u64> = p.cancelled_nodes.iter().map(|n| n.id).collect();
    ids.sort_unstable();
    let ids: IdList = ids.into_iter().collect();
    out(Record::CreateNode {
        ids: ids.clone(),
        labels: Vec::new(),
        attr_ids: Vec::new(),
        rows: Vec::new(),
    });
    out(Record::DeleteNode {
        ids,
        labels: Vec::new(),
    });
}

fn digest_created_nodes(
    p: &Pending,
    out: &mut impl FnMut(Record),
) {
    // Shape key: the node's labels and its attribute ids. Labels are sorted
    // because `set_labels` keeps caller order — [7,8] and [8,7] are the same
    // shape, and emitting them ascending is what makes two engines that agree
    // on the set produce the same bytes.
    //
    // The key is built into a scratch pair that is cleared and refilled per
    // node, and only cloned when the shape turns out to be new. Collecting a
    // fresh `Vec` per node instead cost two heap allocations for every node —
    // 200,000 of them for 100,000 nodes, and 39% of the whole encode — for keys
    // that are overwhelmingly duplicates of one already in the map.
    // Removing the per-node allocation alone moved little: the cost is hashing
    // the key, not building it. So the map is consulted only when the shape
    // actually changes. Entities of one shape arrive in runs — the overwhelming
    // case is a query that creates one shape throughout, where this hashes once
    // and then compares a two-element slice per node.
    // The shape and its members sit together in one `Vec`, with the map holding
    // only a position into it — so there is no second collection to keep in
    // step and no zip at the end. `last` cannot be the map entry itself, much
    // as that would read better: a `&mut` into the map cannot be held across
    // the next lookup, which is the whole point of the memo. That indirection
    // is what the memo costs, and it is what buys ~20% on the single-shape
    // bulk create — a map holding the ids directly hashes once per node.
    let mut slots: Vec<(Shape, IdList)> = Vec::new();
    let mut index: FxHashMap<Shape, usize> = FxHashMap::default();
    let mut last: Option<usize> = None;

    let mut key: Shape = (Vec::new(), Vec::new());
    for id in &p.created_nodes {
        let (labels, attr_ids) = &mut key;
        labels.clear();
        if let Some(l) = p.set_labels.get(&id) {
            labels.extend(l.iter().map(|&v| schema_id(v as usize)));
            labels.sort_unstable();
            labels.dedup();
        }
        attr_ids.clear();
        if let Some(a) = p.new_nodes_attrs.get(&id) {
            attr_ids.extend(a.iter().map(|(aid, _)| *aid));
        }

        let slot = match last {
            Some(i) if slots[i].0 == key => i,
            _ => {
                // `get` then `insert`, not `entry(key.clone())`. `entry` takes
                // an owned key, so it cloned both halves of the shape on every
                // memo *miss* rather than only when the shape was new — two
                // allocations per node once shapes alternate, which measured
                // at 54% of this grouping over 100,000 nodes in four shapes.
                let i = if let Some(&i) = index.get(&key) {
                    i
                } else {
                    slots.push((key.clone(), IdList::new()));
                    index.insert(key.clone(), slots.len() - 1);
                    slots.len() - 1
                };
                last = Some(i);
                i
            }
        };
        slots[slot].1.push(id);
    }

    for ((labels, attr_ids), ids) in slots {
        let rows = gather_rows(&ids, &attr_ids, &p.new_nodes_attrs);
        out(Record::CreateNode {
            ids,
            labels,
            attr_ids,
            rows,
        });
    }
}

fn digest_created_edges(
    p: &Pending,
    g: &AtomicRefCell<Graph>,
    out: &mut impl FnMut(Record),
) {
    if p.created_rels_by_type.is_empty() {
        return;
    }
    let graph = g.borrow();
    for (type_name, entries) in &p.created_rels_by_type {
        // Not `expect`, for the same reason `digest_updates` does not use the
        // panicking type lookup: a type resolves only once an edge of it has
        // actually been committed, and a transaction can create an edge and
        // then cascade it away — `CREATE (a)-[:R]->(b) DELETE a` deletes the
        // edge along with the node. That left `{"R": []}` behind and took the
        // server down on a legitimate query. The group is dropped rather than
        // sent: there is no such edge to replicate.
        let Some(relation_id) = graph.get_type_id(type_name).map(|t| schema_id(t.0)) else {
            debug_assert!(
                entries.is_empty(),
                "a type with surviving edges must have been registered by commit"
            );
            continue;
        };
        // Already partitioned by type; split further by attribute shape.
        let mut groups: FxHashMap<Vec<u16>, Vec<(u64, u64, u64)>> = FxHashMap::default();
        for &(rel_id, from, to) in entries {
            let id = u64::from(rel_id);
            let attr_ids: Vec<u16> = p
                .new_relationships_attrs
                .get(&id)
                .map(|a| a.iter().map(|(aid, _)| *aid).collect())
                .unwrap_or_default();
            groups
                .entry(attr_ids)
                .or_default()
                .push((id, u64::from(from), u64::from(to)));
        }
        for (attr_ids, mut rows) in groups {
            // Sort the triples together so the edge ids come out ascending and
            // the endpoints stay aligned with them. Ascending is what makes the
            // run encodings eligible, and edge ids are allocated sequentially,
            // so the input is usually already in this order and the sort costs
            // one detection pass.
            rows.sort_unstable();
            let ids: IdList = rows.iter().map(|r| r.0).collect();
            let src: IdList = rows.iter().map(|r| r.1).collect();
            let dst: IdList = rows.iter().map(|r| r.2).collect();
            let values = gather_rows(&ids, &attr_ids, &p.new_relationships_attrs);
            out(Record::CreateEdge {
                ids,
                relation_id,
                src,
                dst,
                attr_ids,
                rows: values,
            });
        }
    }
}

fn digest_deleted_edges(
    p: &Pending,
    out: &mut impl FnMut(Record),
) {
    let mut groups: FxHashMap<u64, Vec<(u64, u64, u64)>> = FxHashMap::default();
    for &DeletedEdge {
        id: rel_id,
        type_id,
        src: from,
        dst: to,
    } in &p.deleted_endpoints
    {
        groups.entry(type_id).or_default().push((
            u64::from(rel_id),
            u64::from(from),
            u64::from(to),
        ));
    }
    for (type_id, mut rows) in groups {
        rows.sort_unstable();
        out(Record::DeleteEdge {
            ids: rows.iter().map(|r| r.0).collect(),
            relation_id: schema_id(type_id as usize),
            src: rows.iter().map(|r| r.1).collect(),
            dst: rows.iter().map(|r| r.2).collect(),
        });
    }
}

/// `UPDATE_NODE` / `UPDATE_EDGE`, grouped by attribute shape.
/// `UPDATE_NODE` / `UPDATE_EDGE`, grouped by shape.
///
/// For nodes the shape is `(derived labels, attribute ids)`. The labels are not
/// decoration: an index is `(label, property)` and a node may hold several
/// labels, so they are what says which label-scoped indexes the replica must
/// touch — and the pattern that matched the node under-reports them, since
/// `MATCH (n:A) SET n.p = 1` says nothing about an `(:A:B)` node's `:B`. Pass
/// `graph` for nodes; edges pass `None` and group on attribute ids alone,
/// because an edge has exactly one type and nothing on the apply side looks it
/// up.
///
/// `deleted` is the set of ids this transaction removes. An update to an
/// entity that does not outlive the transaction is not part of the payload —
/// the replica has nothing to apply it to, and the `DELETE_*` record lands it
/// in the same place regardless. Filtering here rather than at the source
/// keeps the query statistics honest: `SET e.x = 1 DELETE e` did set a
/// property, and still reports it, exactly as the node form always has.
///
/// It is also what lets the type lookup below be the ordinary panicking
/// accessor. `commit` clears the matrices before effects are built, so a
/// deleted edge has no type to read; excluded here, no such id reaches it.
fn digest_updates(
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
    deleted: &RoaringTreemap,
    entity: EntityType,
    graph: &Graph,
    out: &mut impl FnMut(Record),
) {
    // The shape is the entity's schema membership plus its attribute ids —
    // labels for a node, the one relationship type for an edge. Both go in the
    // key, because both go on the wire, and a record can only state one of
    // them for the whole group.
    let mut groups: FxHashMap<(Vec<u32>, Option<u32>, Vec<u16>), Vec<u64>> = FxHashMap::default();
    for (id, pairs) in attrs {
        if deleted.contains(*id) {
            continue;
        }
        let attr_ids: Vec<u16> = pairs.iter().map(|(aid, _)| *aid).collect();
        let mut labels: Vec<u32> = Vec::new();
        let mut relation_id: Option<u32> = None;
        match entity {
            EntityType::Node => {
                labels.extend(
                    graph
                        .get_node_label_ids(NodeId::from(*id))
                        .map(|l| schema_id(l.0)),
                );
                // Ascending and deduplicated, matching `digest_created_nodes`:
                // the same label set has to hash to the same shape and
                // serialize to the same bytes on both engines.
                labels.sort_unstable();
                labels.dedup();
            }
            // One lookup per updated edge, on the primary, so the replica does
            // not repeat it per edge inside `track_edge_index_updates`. The
            // primary already pays this in its own commit whenever an edge
            // index exists; the replica pays it unconditionally today.
            //
            // Every id here still has a row in the type matrix: `Pending`
            // drops an edge's staged update when the edge is deleted, both
            // explicitly and when it is cascaded away with its endpoint. So
            // this is the ordinary accessor, as everywhere else in the tree —
            // an update for an entity that did not survive the transaction is
            // not filtered out here, it never reaches here.
            EntityType::Relationship => {
                let type_id = graph.get_relationship_type_id(RelationshipId::from(*id));
                relation_id = Some(schema_id(type_id.0));
            }
        }
        groups
            .entry((labels, relation_id, attr_ids))
            .or_default()
            .push(*id);
    }
    for ((labels, relation_id, attr_ids), mut ids) in groups {
        // Ascending, so the rows below are gathered in that order and the run
        // encodings stay eligible. This is about the ids *within* a record;
        // the order of records among themselves carries no meaning.
        ids.sort_unstable();
        let ids: IdList = ids.into_iter().collect();
        let rows = gather_rows(&ids, &attr_ids, attrs);
        // A pad and a removal are the same byte, so an update group whose members
        // do not all carry the shape's exact attributes would silently delete a
        // property the primary still holds. The group key is each member's own
        // attribute-id vector, so this holds by construction — this is what fails
        // if that ever changes. Counting rather than comparing shapes keeps it to
        // one pass over values already in cache.
        debug_assert_eq!(
            rows.iter().filter(|v| matches!(v, Value::Null)).count(),
            ids.iter()
                .filter_map(|id| attrs.get(&id))
                .flat_map(|pairs| pairs.iter())
                .filter(|(_, v)| matches!(v, Value::Null))
                .count(),
            "an UPDATE row was padded, and a pad is indistinguishable from a removal",
        );
        // The two forms fill the same slot with different things — a node's
        // label set, an edge's one relationship type — so they are separate
        // variants and neither can be built carrying the other's key. Which one
        // is settled by `entity`, and `relation_id` is `Some` exactly when that
        // says relationship.
        out(match entity {
            EntityType::Node => Record::UpdateNode {
                ids,
                labels,
                attr_ids,
                rows,
            },
            EntityType::Relationship => Record::UpdateEdge {
                ids,
                relation_id: relation_id
                    .expect("the relationship arm above sets a type id for every edge"),
                attr_ids,
                rows,
            },
        });
    }
}

/// `DELETE_NODE`, grouped by the labels each node actually carried.
///
/// The labels come from `Pending::deleted_node_labels` rather than the graph,
/// because `delete_nodes` has already cleared the label matrices by the time
/// this runs. They tell the replica which label-scoped indexes to clear, which
/// the query's pattern cannot: `MATCH (n:A) DELETE n` over an `(:A:B)` node
/// must clear `:B`'s indexes too.
fn digest_deleted_nodes(
    p: &Pending,
    out: &mut impl FnMut(Record),
) {
    if p.deleted_nodes.is_empty() {
        return;
    }

    // Both sides are in ascending node order, so this walks them in step rather
    // than looking each node up. A node with no labels contributes no pair and
    // simply lands in the empty-label-set group.
    //
    // Scratch key plus a run memo, and one `Vec` of pairs rather than two kept
    // in step, exactly as in `digest_created_nodes`: a bulk delete is
    // overwhelmingly one label set throughout, so the map is consulted only
    // when the set actually changes and the key is cloned only when it is new.
    let pairs = &p.deleted_node_labels;
    let mut cursor = 0;

    let mut slots: Vec<(Vec<u32>, IdList)> = Vec::new();
    let mut index: FxHashMap<Vec<u32>, usize> = FxHashMap::default();
    let mut last: Option<usize> = None;
    let mut labels: Vec<u32> = Vec::new();

    for id in &p.deleted_nodes {
        while cursor < pairs.len() && pairs[cursor].node < id {
            cursor += 1;
        }
        labels.clear();
        while cursor < pairs.len() && pairs[cursor].node == id {
            labels.push(schema_id(pairs[cursor].label as usize));
            cursor += 1;
        }
        labels.sort_unstable();
        labels.dedup();

        let slot = match last {
            Some(i) if slots[i].0 == labels => i,
            _ => {
                let i = if let Some(&i) = index.get(&labels) {
                    i
                } else {
                    slots.push((labels.clone(), IdList::new()));
                    index.insert(labels.clone(), slots.len() - 1);
                    slots.len() - 1
                };
                last = Some(i);
                i
            }
        };
        slots[slot].1.push(id);
    }

    for (labels, ids) in slots {
        out(Record::DeleteNode { ids, labels });
    }
}

/// `SET_LABELS` / `REMOVE_LABELS`, grouped by label set.
///
/// `skip` excludes nodes whose labels already travel inside their `CREATE_NODE`
/// record.
fn digest_labels(
    labels: &FxHashMap<u64, Vec<u64>>,
    skip: Option<&RoaringTreemap>,
    add: bool,
    out: &mut impl FnMut(Record),
) {
    let mut groups: FxHashMap<Vec<u32>, Vec<u64>> = FxHashMap::default();
    for (&id, label_ids) in labels {
        if skip.is_some_and(|s| s.contains(id)) {
            continue;
        }
        // A label record's whole payload is its label set, so with no labels it
        // states nothing about the nodes it names — it is an instruction to do
        // nothing. `MATCH (n) SET n:Foo REMOVE n:Foo` reaches here with an empty
        // vec, because `remove_node_labels` retains over the staged set and
        // leaves the key behind when it empties.
        //
        // Suppressed rather than tolerated, for the same reason
        // `set_node_attributes` refuses to stage an empty attribute map: the
        // emitter declines to say something that carries no information. An
        // empty label set is still legal *subordinately* — a `DELETE_NODE`
        // stating that its nodes carried no labels is a fact about them — which
        // is why this is a rule about this record and not about the block.
        if label_ids.is_empty() {
            continue;
        }
        let mut shape: Vec<u32> = label_ids.iter().map(|&v| schema_id(v as usize)).collect();
        shape.sort_unstable();
        shape.dedup();
        groups.entry(shape).or_default().push(id);
    }
    for (labels, mut ids) in groups {
        debug_assert!(
            !labels.is_empty(),
            "a label record with no labels states nothing; it is suppressed above"
        );
        // Ascending, which keeps the run encodings eligible.
        ids.sort_unstable();
        let ids: IdList = ids.into_iter().collect();
        // `SetLabels`, not `AddLabels` — the opcode's own name, and one letter
        // from `AddLabel`, which registers a label in the schema dictionary
        // rather than putting one on a node.
        out(if add {
            Record::SetLabels { ids, labels }
        } else {
            Record::RemoveLabels { ids, labels }
        });
    }
}

/// The `count x n` row-major values for one shape.
///
/// An entity missing one of the shape's attributes gets `T_NULL` in that slot.
/// **That pad is byte-identical to a removal**, because `T_NULL` in a value slot
/// is how `SET n.x = NULL` replicates — so padding is only safe where there is
/// nothing to remove, i.e. on a freshly created entity.
///
/// Every caller passes a shape built from its members' own exact attribute-id
/// vectors, so no pad is emitted today and the branches below are the defined
/// behaviour rather than a live path. Widening a partition key to group
/// different-but-overlapping shapes — which is what would cut the record count —
/// requires carrying presence separately from value first; `digest_updates`
/// asserts the absence of pads for exactly this reason.
fn gather_rows(
    ids: &IdList,
    attr_ids: &[u16],
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
) -> Vec<Value> {
    let mut rows = Vec::with_capacity(ids.len() * attr_ids.len());
    for id in ids.iter() {
        let Some(pairs) = attrs.get(&id) else {
            rows.resize(rows.len() + attr_ids.len(), Value::Null);
            continue;
        };
        // Both sides are sorted by attribute id — the shape was built from these
        // very vectors — so this is a merge, not a search per cell. Scanning for
        // each cell made the cost quadratic in the shape's width: unnoticeable
        // at two attributes, not at twenty.
        let mut j = 0;
        for &attr_id in attr_ids {
            while j < pairs.len() && pairs[j].0 < attr_id {
                j += 1;
            }
            if j < pairs.len() && pairs[j].0 == attr_id {
                rows.push(pairs[j].1.clone());
                j += 1;
            } else {
                rows.push(Value::Null);
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::v3::Record;
    use crate::effects::v3::staging::StagePending;
    use crate::effects::v3::test_aux::{
        digest, encode_all, graph_cell as graph, live_node, read_buffer, round_trip as build,
        with_attrs, with_edge,
    };
    use crate::entity_type::EntityType;
    use crate::runtime::pending::Pending;
    // The module itself no longer needs these — the announced types moved to
    // `effects::announce` — but the tests still build the values they carry.
    use std::sync::Arc;

    /// The `AnnouncedIndex` the two index tests announce.
    fn announced<'a>(
        label: &'a str,
        fields: &'a [Arc<String>],
        index_type: &'a IndexType,
        options: Option<&'a Value>,
    ) -> AnnouncedIndex<'a> {
        AnnouncedIndex {
            entity_type: EntityType::Node,
            index_type,
            label,
            fields,
            options,
        }
    }

    #[test]
    fn an_index_statement_carries_its_ids_names_and_options() {
        let g = graph();
        with_attrs(&g, &["body"]);
        {
            let mut graph = g.borrow_mut();
            graph.get_label_id_mut("D");
        }
        let fields = vec![Arc::new(String::from("body"))];
        let options = Value::Map(Arc::new(
            [(
                Arc::new(String::from("language")),
                Value::String(Arc::new(String::from("german"))),
            )]
            .into_iter()
            .collect(),
        ));
        let mut buf = v3::new_buffer();
        // A default `Pending` carries zero schema counts, so the label and the
        // attribute are announced ahead of the record that names their ids.
        build_index_buffer(
            &Pending::default(),
            &g,
            true,
            &announced("D", &fields, &IndexType::Fulltext, Some(&options)),
            &mut buf,
        )
        .unwrap();

        let records = read_buffer(&buf).unwrap();
        // schema first: the ids on the index record are only meaningful after it
        assert!(matches!(records[0], Record::AddLabel { id: 0, .. }));
        assert!(matches!(records[1], Record::AddAttribute { id: 0, .. }));
        let Record::CreateIndex {
            label_id,
            ref label,
            field_type,
            ref fields,
            ref options,
            ..
        } = records[2]
        else {
            panic!("expected a create-index record, got {:?}", records[2]);
        };
        assert_eq!((label_id, label.as_str()), (0, "D"));
        assert_eq!(field_type, v3::INDEX_FLD_FULLTEXT);
        assert_eq!(fields.len(), 1);
        assert_eq!((fields[0].id, fields[0].name.as_str()), (0, "body"));
        // The options travel typed — and as the engine's own type now, so this
        // asserts the values rather than the container.
        let o = options;
        // The statement said `language: 'german'`, so that is what travels —
        // this is the assertion the old `matches!(Some(Value::Map(_)))` could
        // not make: it proved a map was present, not that its contents
        // survived.
        assert_eq!(o.text.language.as_ref().map(|l| l.as_str()), Some("german"));
        // And the keys the statement omitted travel as omitted. Filling them
        // with the RDB's defaults here is what diverged a replica: applying
        // `language` a second time is refused once one is set for the label,
        // so an option nobody asked for must not arrive as one somebody did.
        assert!(
            o.text.weight.is_none()
                && o.text.nostem.is_none()
                && o.text.phonetic.is_none()
                && o.text.stopwords.is_none(),
            "an omitted option travels omitted, not defaulted"
        );
        assert!(
            o.vector.is_none(),
            "a fulltext statement has no vector half"
        );
    }

    #[test]
    fn an_index_over_an_unregistered_name_is_refused_rather_than_encoded() {
        // A bare id on the wire is only safe if the replica agrees on the
        // numbering, so an unresolvable name must not become a guessed id.
        let g = graph();
        let fields = vec![Arc::new(String::from("nope"))];
        let mut buf = v3::new_buffer();
        let err = build_index_buffer(
            &Pending::default(),
            &g,
            true,
            &announced("Missing", &fields, &IndexType::Range, None),
            &mut buf,
        )
        .unwrap_err();
        assert!(err.contains("is not registered"), "{err}");
        assert_eq!(
            buf,
            v3::new_buffer(),
            "nothing may be written before the ids resolve"
        );
    }

    /// Register `n` node attributes so the ids the tests use resolve.
    /// A committed edge of the named type, so the emitter has a type to read.
    /// The loop `format::build` runs, so the tests still go through the bytes
    /// rather than stopping at the records.
    #[test]
    fn nodes_of_one_shape_become_one_record() {
        let g = graph();
        with_attrs(&g, &["a"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        for id in 0..100u64 {
            p.stage_created_node(id, &[0], &[(0, Value::Int(id as i64))]);
        }

        let records = build(&p, &g);
        assert_eq!(records.len(), 1, "100 nodes, one shape, one record");
        let Record::CreateNode {
            ids,
            labels,
            attr_ids,
            rows,
        } = &records[0]
        else {
            panic!("wrong record: {:?}", records[0]);
        };
        assert_eq!(ids.len(), 100);
        assert_eq!(labels, &[0]);
        assert_eq!(attr_ids, &[0]);
        assert_eq!(rows.len(), 100);
        // Row k belongs to the k-th smallest id.
        assert_eq!(rows[7], Value::Int(7));
    }

    /// Put `id` in the graph carrying `labels`, so the emitter can derive them.
    #[test]
    fn a_multi_labeled_node_is_its_own_partition() {
        // CREATE (:A), (:B), (:A:B) — three label sets, so three records. The
        // multi-labeled node is not a member of either single-label shape.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(1, &[0], &[]);
        p.stage_created_node(2, &[1], &[]);
        p.stage_created_node(3, &[0, 1], &[]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 3, "{records:#?}");
        let mut seen: Vec<(Vec<u32>, Vec<u64>)> = records
            .iter()
            .map(|r| {
                let Record::CreateNode { ids, labels, .. } = r else {
                    panic!("wrong record: {r:?}");
                };
                (labels.clone(), ids.iter().collect::<Vec<_>>())
            })
            .collect();
        seen.sort();
        assert_eq!(
            seen,
            vec![
                (vec![0], vec![1]),
                (vec![0, 1], vec![3]),
                (vec![1], vec![2]),
            ]
        );
    }

    #[test]
    fn a_multi_attribute_node_is_its_own_partition() {
        // The same rule on the other half of the key: {x}, {y} and {x,y} are
        // three shapes, not two.
        let g = graph();
        with_attrs(&g, &["x", "y"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(1, &[0], &[(0, Value::Int(1))]);
        p.stage_created_node(2, &[0], &[(1, Value::Int(2))]);
        p.stage_created_node(3, &[0], &[(0, Value::Int(3)), (1, Value::Int(4))]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 3, "{records:#?}");
        let mut shapes: Vec<Vec<u16>> = records
            .iter()
            .map(|r| {
                let Record::CreateNode { attr_ids, .. } = r else {
                    panic!("wrong record: {r:?}");
                };
                attr_ids.clone()
            })
            .collect();
        shapes.sort();
        assert_eq!(shapes, vec![vec![0], vec![0, 1], vec![1]]);
    }

    #[test]
    fn an_update_partitions_on_labels_the_query_never_named() {
        // MATCH (n:A) SET n.x = ... over an (:A) and an (:A:B) node. Both share
        // an attribute shape, but they sit in different indexes, so they are
        // two partitions — and the :B is derived, since the pattern said :A.
        let g = graph();
        with_attrs(&g, &["x"]);
        live_node(&g, 1, &["A"]);
        live_node(&g, 2, &["A", "B"]);

        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_updated_node(1, &[(0, Value::Int(10))]);
        p.stage_updated_node(2, &[(0, Value::Int(20))]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 2, "{records:#?}");
        let mut seen: Vec<(Vec<u32>, Vec<u64>)> = records
            .iter()
            .map(|r| {
                // `UpdateNode` by construction now: an edge update is a
                // different variant and cannot carry a label set at all.
                let Record::UpdateNode { ids, labels, .. } = r else {
                    panic!("wrong record: {r:?}");
                };
                (labels.clone(), ids.iter().collect::<Vec<_>>())
            })
            .collect();
        seen.sort();
        assert_eq!(seen, vec![(vec![0], vec![1]), (vec![0, 1], vec![2])]);
    }

    #[test]
    fn a_delete_carries_the_labels_captured_at_deletion() {
        // MATCH (n:A) DELETE n over (:A) and (:A:B). The (:A:B) node must carry
        // :B or the replica leaves a stale entry in :B's index.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_deleted_node(1, &[0]);
        p.stage_deleted_node(2, &[0, 1]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 2, "{records:#?}");
        let mut seen: Vec<(Vec<u32>, Vec<u64>)> = records
            .iter()
            .map(|r| {
                let Record::DeleteNode { ids, labels } = r else {
                    panic!("wrong record: {r:?}");
                };
                (labels.clone(), ids.iter().collect::<Vec<_>>())
            })
            .collect();
        seen.sort();
        assert_eq!(seen, vec![(vec![0], vec![1]), (vec![0, 1], vec![2])]);
    }

    #[test]
    fn a_label_less_delete_is_one_partition() {
        // Nodes with no labels contribute no pair to the captured list; they
        // must still land in a record, under the empty label set.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_deleted_node(1, &[]);
        p.stage_deleted_node(2, &[3]);
        p.stage_deleted_node(3, &[]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 2, "{records:#?}");
        let mut seen: Vec<(Vec<u32>, Vec<u64>)> = records
            .iter()
            .map(|r| {
                let Record::DeleteNode { ids, labels } = r else {
                    panic!("wrong record: {r:?}");
                };
                (labels.clone(), ids.iter().collect::<Vec<_>>())
            })
            .collect();
        seen.sort();
        assert_eq!(seen, vec![(vec![], vec![1, 3]), (vec![3], vec![2])]);
    }

    #[test]
    fn an_edge_update_carries_its_type_not_labels() {
        // The two forms fill the same slot differently: a node's labels, an
        // edge's one relationship type. C's own UPDATE_EDGE has always carried
        // the relation id — see `update_edge_effect.c`, which refuses a record
        // naming a type it does not have.
        let g = graph();
        with_attrs(&g, &["x"]);
        with_edge(&g, "R", 5);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_updated_edge(5, &[(0, Value::Int(1))]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 1, "{records:#?}");
        let Record::UpdateEdge {
            ids, relation_id, ..
        } = &records[0]
        else {
            panic!("wrong record: {:?}", records[0]);
        };
        assert_eq!(ids, &[5]);
        // "carries no label set" is now a fact about the type rather than an
        // assertion: `UpdateEdge` has no labels field to be empty.
        assert_eq!(*relation_id, 0, "it carries its type instead");
    }

    #[test]
    fn two_types_do_not_share_one_update_record() {
        // The type is stated once for the whole record, so it has to be part of
        // the group key. Sharing the shape but not the type must split.
        let g = graph();
        with_attrs(&g, &["x"]);
        with_edge(&g, "R", 5);
        with_edge(&g, "S", 6);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_updated_edge(5, &[(0, Value::Int(1))]);
        p.stage_updated_edge(6, &[(0, Value::Int(2))]);

        let records = build(&p, &g);
        let mut types: Vec<u32> = records
            .iter()
            .map(|r| match r {
                Record::UpdateEdge { relation_id, .. } => *relation_id,
                other => panic!("wrong record: {other:?}"),
            })
            .collect();
        types.sort_unstable();
        assert_eq!(types, vec![0, 1], "{records:#?}");
    }

    #[test]
    fn an_update_is_not_emitted_for_an_edge_the_transaction_deletes() {
        // `MATCH ()-[e]->() SET e.x = 1 DELETE e`. The SET really did set a
        // property and the query still reports `Properties set: 1` — that is
        // not what this is about. The payload is a different question: the
        // replica has no edge to apply the update to, and the DELETE_EDGE
        // record puts it in the same place regardless.
        //
        // It is also what makes the type lookup in `digest_updates` safe with
        // the ordinary panicking accessor. `commit` clears the matrices before
        // effects are built, so a deleted edge has no type left to read;
        // excluded here, no such id ever reaches that lookup.
        let g = graph();
        with_attrs(&g, &["x"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);

        p.stage_updated_edge(5, &[(0, Value::Int(1))]);
        p.deleted_relationship(RelationshipId::from(5_u64));

        let records = build(&p, &g);
        assert!(
            records
                .iter()
                .all(|r| !matches!(r, Record::UpdateNode { .. } | Record::UpdateEdge { .. })),
            "an update for an edge the transaction deletes must not be emitted: \
             {records:#?}"
        );
    }

    #[test]
    fn label_order_does_not_split_a_shape() {
        // set_labels keeps caller order, so [0,1] and [1,0] arrive differently
        // for the same set. They must land in one record, sorted.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(1, &[0, 1], &[]);
        p.stage_created_node(2, &[1, 0], &[]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 1, "same label set, so one record");
        let Record::CreateNode { ids, labels, .. } = &records[0] else {
            panic!("wrong record");
        };
        assert_eq!(ids, &[1, 2]);
        assert_eq!(labels, &[0, 1], "emitted ascending, so bytes are canonical");
    }

    #[test]
    fn differing_shapes_split_into_separate_records() {
        let g = graph();
        with_attrs(&g, &["a", "b"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(1, &[], &[(0, Value::Int(1))]);
        p.stage_created_node(2, &[], &[(1, Value::Int(2))]);

        let records = build(&p, &g);
        assert_eq!(
            records.len(),
            2,
            "different attribute ids are different shapes"
        );
    }

    #[test]
    fn a_created_node_carries_its_labels_and_emits_no_set_labels() {
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(1, &[3], &[]);
        // A node that already existed does need a SET_LABELS record.
        p.stage_label_change(99, &[3]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 2);
        assert!(matches!(records[0], Record::CreateNode { .. }));
        // `SetLabels` by construction: a removal is a different variant now,
        // so "it was an add" is a fact about the type rather than a flag to
        // assert.
        let Record::SetLabels { ids, labels } = &records[1] else {
            panic!("expected SET_LABELS for the pre-existing node");
        };
        assert_eq!(ids, &[99], "the created node is not repeated here");
        assert_eq!(labels, &[3]);
    }

    #[test]
    fn updates_group_by_attribute_shape() {
        let g = graph();
        with_attrs(&g, &["a", "b"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_updated_node(1, &[(0, Value::Int(1))]);
        p.stage_updated_node(2, &[(0, Value::Int(2))]);
        p.stage_updated_node(3, &[(0, Value::Int(3)), (1, Value::Int(4))]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 2, "two shapes");
        // By shape, not by position — see `deleted_edges_group_by_relationship_type`.
        let mut seen: Vec<(Vec<u16>, Vec<u64>)> = records
            .iter()
            .map(|r| {
                let Record::UpdateNode { ids, attr_ids, .. } = r else {
                    panic!("wrong record: {r:?}");
                };
                (attr_ids.clone(), ids.iter().collect())
            })
            .collect();
        seen.sort();
        assert_eq!(seen, vec![(vec![0], vec![1, 2]), (vec![0, 1], vec![3])]);
    }

    #[test]
    fn schema_records_precede_the_records_that_use_them() {
        // Normative ordering: an id on the wire means nothing until the replica
        // has agreed on the numbering that produced it.
        let g = graph();
        {
            let mut graph = g.borrow_mut();
            graph.get_label_id_mut("L");
            graph.add_node_attribute_name("a");
        }
        // Default counts are already zero, which is what "the schema is new"
        // means: everything the graph holds gets announced.
        let mut p = Pending::default();
        p.stage_created_node(1, &[0], &[(0, Value::Int(1))]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 3);
        assert_eq!(
            records[0],
            Record::AddLabel {
                id: 0,
                name: "L".into()
            }
        );
        assert_eq!(
            records[1],
            Record::AddAttribute {
                id: 0,
                name: "a".into()
            }
        );
        assert!(matches!(records[2], Record::CreateNode { .. }));
    }

    #[test]
    fn grouping_is_deterministic() {
        // Two Pendings built in different insertion orders must produce
        // byte-identical buffers, or the cross-engine comparison is worthless.
        let g = graph();
        with_attrs(&g, &["a", "b"]);
        let mut forward = Pending::default();
        let mut backward = Pending::default();
        forward.set_schema_baseline(&g);
        backward.set_schema_baseline(&g);
        for id in 0..50u64 {
            forward.stage_created_node(id, &[], &[((id % 2) as u16, Value::Int(id as i64))]);
        }
        for id in (0..50u64).rev() {
            backward.stage_created_node(id, &[], &[((id % 2) as u16, Value::Int(id as i64))]);
        }

        let mut a = v3::new_buffer();
        encode_all(&forward, &g, &mut a);
        let mut b = v3::new_buffer();
        encode_all(&backward, &g, &mut b);
        assert_eq!(a, b, "insertion order must not reach the wire");
    }

    #[test]
    fn a_new_attribute_is_announced_once() {
        // #2459 unified the node and relationship attribute dictionaries, so
        // the two getters return the same store. v3's ADD_ATTRIBUTE carries no
        // discriminator, so announcing per entity kind would send the same
        // attribute twice under the same id.
        let g = graph();
        {
            let mut graph = g.borrow_mut();
            graph.add_node_attribute_name("a");
            graph.add_node_attribute_name("b");
        }
        // Default counts are zero, so both attributes are new.
        let p = Pending::default();

        let records = build(&p, &g);
        assert_eq!(
            records,
            vec![
                Record::AddAttribute {
                    id: 0,
                    name: "a".into()
                },
                Record::AddAttribute {
                    id: 1,
                    name: "b".into()
                },
            ]
        );
    }

    #[test]
    fn deleted_edges_group_by_relationship_type() {
        // The type has to be captured at deletion time — the edge is gone by
        // the time effects are built, and index_remove_edge_docs only knows
        // about types that carry an index.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_deleted_edges(vec![
            DeletedEdge {
                id: 1.into(),
                type_id: 0,
                src: 10.into(),
                dst: 11.into(),
            },
            DeletedEdge {
                id: 2.into(),
                type_id: 1,
                src: 12.into(),
                dst: 13.into(),
            },
            DeletedEdge {
                id: 3.into(),
                type_id: 0,
                src: 14.into(),
                dst: 15.into(),
            },
        ]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 2, "two types, two records");
        // Found by type rather than by position: records within one stage are
        // disjoint, so nothing orders them and nothing should depend on it.
        let Some(Record::DeleteEdge {
            ids,
            relation_id,
            src,
            dst,
        }) = records
            .iter()
            .find(|r| matches!(r, Record::DeleteEdge { relation_id: 0, .. }))
        else {
            panic!("no record for type 0: {records:#?}");
        };
        assert_eq!(*relation_id, 0);
        assert_eq!(ids, &[1, 3]);
        assert_eq!(src, &[10, 14]);
        assert_eq!(dst, &[11, 15]);
    }

    #[test]
    fn gather_rows_merges_rather_than_searching() {
        // Within a group every entity has exactly the shape's attributes, so the
        // merge normally walks in lockstep. These are the cases that are not
        // that, because the merge has to behave like the scan it replaced: an
        // entity missing a middle attribute, one missing a trailing attribute,
        // and one absent from the map entirely.
        let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        attrs.insert(1, vec![(0, Value::Int(10)), (2, Value::Int(12))]); // no attr 1
        attrs.insert(2, vec![(0, Value::Int(20))]); // no attr 1 or 2
        attrs.insert(
            3,
            vec![
                (0, Value::Int(30)),
                (1, Value::Int(31)),
                (2, Value::Int(32)),
            ],
        );

        let rows = gather_rows(&IdList::from([1, 2, 3, 4]), &[0, 1, 2], &attrs);
        assert_eq!(
            rows,
            vec![
                Value::Int(10),
                Value::Null,
                Value::Int(12),
                Value::Int(20),
                Value::Null,
                Value::Null,
                Value::Int(30),
                Value::Int(31),
                Value::Int(32),
                // id 4 is not in the map at all
                Value::Null,
                Value::Null,
                Value::Null,
            ]
        );
    }

    #[test]
    fn the_scratch_shape_key_groups_exactly_as_a_fresh_one_would() {
        // The key is a reused scratch pair now, cloned only when the shape is
        // new. Clearing it wrongly would smear one node's shape onto the next,
        // which shows up as too few groups.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        // Alternating widths, so a stale scratch would merge them.
        for id in 0..100u64 {
            if id % 2 == 0 {
                p.stage_created_node(id, &[0], &[(0, Value::Int(id as i64))]);
            } else {
                p.stage_created_node(
                    id,
                    &[0, 1],
                    &[(0, Value::Int(id as i64)), (1, Value::Int(1))],
                );
            }
        }
        let records = digest(&p, &g);
        assert_eq!(records.len(), 2, "two shapes, two records");
        for r in &records {
            let Record::CreateNode {
                ids,
                labels,
                attr_ids,
                rows,
            } = r
            else {
                panic!("wrong record");
            };
            assert_eq!(ids.len(), 50);
            assert_eq!(rows.len(), ids.len() * attr_ids.len());
            assert_eq!(labels.len(), attr_ids.len(), "1 label/1 attr, or 2 and 2");
        }
    }

    #[test]
    fn the_digest_is_what_the_bytes_decode_back_to() {
        // The check the old shape could not make. Emit is now
        // Pending -> Record -> bytes and apply is bytes -> Record -> Graph, so
        // the two Record sets must be equal — a far stronger statement than
        // "the buffer is the expected length", and it compares values rather
        // than a byte count that could be right for the wrong reason.
        let g = graph();
        {
            let mut graph = g.borrow_mut();
            graph.get_label_id_mut("Person");
            graph.add_node_attribute_name("name");
            graph.add_node_attribute_name("age");
        }
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        for id in 0..200u64 {
            p.stage_created_node(id, &[0], &[(0, Value::Int(id as i64)), (1, Value::Int(30))]);
        }
        for id in 200..250u64 {
            p.stage_created_node(id, &[0], &[(0, Value::Int(id as i64))]);
        }
        p.stage_updated_node(500, &[(1, Value::Int(41))]);
        p.stage_label_change(600, &[0]);

        let digested = digest(&p, &g);
        let mut buf = v3::new_buffer();
        encode_all(&p, &g, &mut buf);

        assert_eq!(read_buffer(&buf).unwrap(), digested);
    }

    #[test]
    fn grouping_is_testable_without_bytes() {
        // Shapes split on attribute ids, and the digest says so directly
        // instead of the test having to decode a buffer to find out.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(1, &[], &[(0, Value::Int(1))]);
        p.stage_created_node(2, &[], &[(1, Value::Int(2))]);

        let records = digest(&p, &g);
        assert_eq!(records.len(), 2);
        for record in &records {
            let Record::CreateNode { ids, attr_ids, .. } = record else {
                panic!("expected creates, got {record:?}");
            };
            assert_eq!(ids.len(), 1, "one node per shape here");
            assert_eq!(attr_ids.len(), 1);
        }
    }

    #[test]
    fn the_motivating_query_stays_one_record() {
        // `UNWIND range(0, 9999) AS i CREATE (:Person {v: i})` — one shape, so
        // one record however many nodes, which is the whole point of grouping
        // by shape.
        //
        // Pinned as an absolute size now that there is no v2 to measure
        // against. For the record, v2 encoded this same query in 260,001 bytes,
        // one record per node; the ratio was what justified the format.
        let g = graph();
        with_attrs(&g, &["v"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        for id in 0..10_000u64 {
            p.stage_created_node(id, &[0], &[(0, Value::Int(id as i64))]);
        }

        let mut buf = v3::new_buffer();
        encode_all(&p, &g, &mut buf);
        assert_eq!(read_buffer(&buf).unwrap().len(), 1, "one shape, one record");
        assert!(
            buf.len() < 130_000,
            "10,000 nodes should encode in ~120 KB, got {}",
            buf.len()
        );
    }
}

#[cfg(test)]
mod cancelled {
    use super::*;
    use crate::effects::v3::staging::StagePending;
    use crate::effects::v3::test_aux::{digest, graph_cell, with_attrs};
    use crate::graph::graph::NodeId;
    use crate::runtime::pending::Pending;

    /// `CREATE (a:A {v:1}), (b:B) DELETE a` — a's id is handed back and the
    /// master commits nothing for it.
    fn cancelled(g: &AtomicRefCell<Graph>) -> Pending {
        let mut p = Pending::default();
        p.set_schema_baseline(g);
        p.stage_created_node(0, &[0], &[(0, Value::Int(1))]);
        p.stage_created_node(1, &[1], &[]);
        p.delete_pending_node(NodeId::from(0_u64));
        p
    }

    #[test]
    fn a_cancelled_node_is_a_create_and_a_delete() {
        // Without this the buffer says nothing about id 0, and the replica's
        // id space comes out one short — its next allocation after a promotion
        // lands on a live node.
        let g = graph_cell();
        with_attrs(&g, &["v"]);
        {
            let mut graph = g.borrow_mut();
            graph.get_label_id_mut("A");
            graph.get_label_id_mut("B");
        }
        let records = digest(&cancelled(&g), &g);

        let create = records
            .iter()
            .position(|r| matches!(r, Record::CreateNode { labels, .. } if labels.is_empty()))
            .expect("the cancelled node's create");
        let delete = records
            .iter()
            .position(|r| matches!(r, Record::DeleteNode { .. }))
            .expect("the cancelled node's delete");
        assert!(create < delete, "the create has to precede the delete");

        let Record::CreateNode {
            ids,
            labels,
            attr_ids,
            rows,
        } = &records[create]
        else {
            unreachable!()
        };
        assert_eq!(ids.iter().collect::<Vec<_>>(), vec![0]);
        // Ids only. The node is deleted a record later, so its content would be
        // applied and then undone — and `apply` never reads a `DeleteNode`'s
        // labels in the first place.
        assert!(labels.is_empty(), "a cancelled node carries no labels");
        assert!(attr_ids.is_empty(), "nor attributes");
        assert!(rows.is_empty(), "nor their values");

        let Record::DeleteNode { ids, labels } = &records[delete] else {
            unreachable!()
        };
        assert_eq!(ids.iter().collect::<Vec<_>>(), vec![0]);
        assert!(labels.is_empty());
    }

    #[test]
    fn the_survivor_is_still_its_own_record() {
        // The cancelled node must not be folded into the live create: they are
        // different shapes, and one of them is deleted a record later.
        let g = graph_cell();
        with_attrs(&g, &["v"]);
        {
            let mut graph = g.borrow_mut();
            graph.get_label_id_mut("A");
            graph.get_label_id_mut("B");
        }
        let records = digest(&cancelled(&g), &g);
        let survivor = records
            .iter()
            .find(|r| matches!(r, Record::CreateNode { labels, .. } if labels == &[1]))
            .expect("the surviving node's create");
        let Record::CreateNode { ids, .. } = survivor else {
            unreachable!()
        };
        assert_eq!(ids.iter().collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn the_pair_comes_last_so_a_reuse_orders_after_it() {
        // A later commit in the same query can reclaim the id from the bin, and
        // its create is appended after these records. The delete has to be in
        // the buffer *before* that create, not merely somewhere in it.
        let g = graph_cell();
        {
            let mut graph = g.borrow_mut();
            graph.get_label_id_mut("A");
        }
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(0, &[0], &[]);
        p.stage_deleted_node(7, &[0]);
        p.delete_pending_node(NodeId::from(0_u64));

        let records = digest(&p, &g);
        let last_two = &records[records.len() - 2..];
        assert!(
            matches!(last_two[0], Record::CreateNode { .. })
                && matches!(last_two[1], Record::DeleteNode { .. }),
            "{records:#?}"
        );
    }
}

#[cfg(test)]
mod cascade {
    use super::*;
    use crate::effects::v3::staging::StagePending;
    use crate::effects::v3::test_aux::{digest, graph_cell};
    use crate::graph::graph::{NodeId, RelationshipId};
    use crate::runtime::pending::Pending;
    use std::sync::Arc;

    /// `CREATE (a)-[:R]->(b) DELETE a` — the delete cascades to the edge before
    /// either reaches the graph.
    fn cascaded() -> Pending {
        let mut p = Pending::default();
        p.stage_created_node(0, &[], &[]);
        p.stage_created_node(1, &[], &[]);
        p.created_relationship(
            RelationshipId::from(0_u64),
            NodeId::from(0_u64),
            NodeId::from(1_u64),
            Arc::new("R".to_owned()),
        );
        p.remove_pending_relationships_for_node(NodeId::from(0_u64));
        p
    }

    #[test]
    fn the_cascade_leaves_no_empty_group_behind() {
        // The key has to go with its last entry. A type whose every edge was
        // cascaded away was never registered on the graph — nothing of it
        // reached `create_relationships_bulk` — so a key left behind is a type
        // name the emitter cannot resolve.
        let p = cascaded();
        assert!(
            p.created_rels_by_type.is_empty(),
            "{:?}",
            p.created_rels_by_type.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn an_unregistered_type_is_dropped_rather_than_panicking() {
        // The emitter's own guard, tested against the state the cascade used
        // to leave: this took the server down with "created relationship type
        // must be registered" on a legitimate query.
        let g = graph_cell();
        let mut p = cascaded();
        p.created_rels_by_type
            .insert(Arc::new("R".to_owned()), Vec::new());

        let records = digest(&p, &g);
        assert!(
            !records
                .iter()
                .any(|r| matches!(r, Record::CreateEdge { .. })),
            "no such edge exists to replicate: {records:#?}"
        );
    }
}

/// The `OPTIONS` block a `CREATE_INDEX` record carries.
///
/// The query's `OPTIONS {...}` is a Cypher map literal, so a map exists at the
/// query layer and always will. What changed is that it stops there: this
/// converts it to the typed block the wire carries, materialising the RDB's
/// defaults for anything the statement did not say.
fn wire_index_options(ix: &AnnouncedIndex<'_>) -> IndexFieldOptions {
    let parsed = ix.options.and_then(|v| match v {
        Value::Map(m) => map_to_index_options(ix.index_type, m).ok().flatten(),
        _ => None,
    });
    // The record holds the engine's own option types, so there is nothing to
    // convert into — only which half of the pair a statement filled in. That is
    // what `21f37287d` bought: the field-by-field copy this used to be could
    // narrow a value on the way past, and did, twice.
    match parsed {
        // Nothing said. Not "everything at its default" — an effect adds a
        // field to an index that may already exist, so "no language given" and
        // "language english" are different instructions, and the second is
        // refused once a language is set for the label.
        None => IndexFieldOptions::none_given(None),
        Some(EngineIndexOptions::Text(text)) => IndexFieldOptions { text, vector: None },
        Some(EngineIndexOptions::Vector(v)) => IndexFieldOptions::none_given(Some(v)),
    }
}
