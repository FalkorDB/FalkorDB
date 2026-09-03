//! The records themselves, and reading a whole payload.

use crate::runtime::value::Value;

use super::*;

// ── records ──

/// Open a batchable record: its opcode, then how many entities it covers.
///
/// Every block that follows takes its length from this one count — it is never
/// repeated. `count == 1` and `count == 10_000` are the same record, which is
/// why no record type is left un-batchable and the decoder has one shape per
/// opcode.
fn write_header(
    buf: &mut Vec<u8>,
    opcode: Opcode,
    count: Option<u32>,
) {
    debug_assert_eq!(
        count.is_some(),
        opcode.is_batchable(),
        "{opcode:?} was given the wrong kind of header"
    );
    write_u32(buf, opcode as u32);
    if let Some(count) = count {
        write_u32(buf, count);
    }
}

/// `3 CREATE_NODE` — `count · IdList · LabelSet · AttrSet`.
///
/// `ids` must be ascending and unique; `rows` is `ids.len() × attr_ids.len()`
/// values in row-major order. Row *k* belongs to the k-th smallest id.
pub fn write_create_node(
    buf: &mut Vec<u8>,
    ids: &IdList,
    labels: &[i32],
    attr_ids: &[u16],
    rows: &[Value],
) {
    write_header(buf, Opcode::CreateNode, Some(ids.count()));
    write_label_set(buf, labels);
    write_attr_ids(buf, attr_ids);
    ids.encode(buf);
    write_attr_values(buf, attr_ids.len(), rows);
}

/// `4 CREATE_EDGE` — `count · IdList · RelType · IdList(src) · IdList(dst) · AttrSet`.
///
/// Endpoints repeat — many edges share a source — and each must stay
/// positionally aligned with its edge id, which is why nothing here may be
/// re-sorted into a set.
pub fn write_create_edge(
    buf: &mut Vec<u8>,
    ids: &IdList,
    relation_id: i32,
    src: &IdList,
    dst: &IdList,
    attr_ids: &[u16],
    rows: &[Value],
) {
    debug_assert_eq!(ids.len(), src.len(), "one source per edge");
    debug_assert_eq!(ids.len(), dst.len(), "one destination per edge");
    write_header(buf, Opcode::CreateEdge, Some(ids.count()));
    write_rel_type(buf, relation_id);
    write_attr_ids(buf, attr_ids);
    ids.encode(buf);
    src.encode(buf);
    dst.encode(buf);
    write_attr_values(buf, attr_ids.len(), rows);
}

/// `1 UPDATE_NODE` — `count · IdList · LabelSet · AttrSet`.
/// `2 UPDATE_EDGE` — `count · IdList · AttrSet`.
///
/// v2 wrote one record per *(entity, attribute)*; v3 writes one per shape. A
/// property being removed is `T_NULL` in its slot — the tag's only meaning here.
///
/// The node form carries a `LabelSet` and the edge form does not, because a
/// node's labels are what say which label-scoped indexes to touch and a node
/// may hold several — `set_nodes_attributes` walks `node_labels_matrix` per
/// node to work that out. An edge has exactly one type and
/// `set_relationships_attributes` never looks it up, so a type here would be a
/// field nothing reads. `labels` is ignored for [`EntityType::Relationship`].
pub fn write_update(
    buf: &mut Vec<u8>,
    entity: EntityType,
    ids: &IdList,
    labels: &[i32],
    attr_ids: &[u16],
    rows: &[Value],
) {
    write_header(buf, update_opcode(entity), Some(ids.count()));
    if entity == EntityType::Node {
        write_label_set(buf, labels);
    }
    write_attr_ids(buf, attr_ids);
    ids.encode(buf);
    write_attr_values(buf, attr_ids.len(), rows);
}

/// `5 DELETE_NODE` — `count · IdList · LabelSet`.
///
/// The labels are the node's **actual** labels, captured when it was deleted —
/// not the ones the query's pattern named. `MATCH (n:A) DELETE n` over an
/// `(:A:B)` node must clear `:B`'s indexes too, and the pattern cannot say so.
pub fn write_delete_node(
    buf: &mut Vec<u8>,
    ids: &IdList,
    labels: &[i32],
) {
    write_header(buf, Opcode::DeleteNode, Some(ids.count()));
    write_label_set(buf, labels);
    ids.encode(buf);
}

/// `6 DELETE_EDGE` — `count · IdList · RelType · IdList(src) · IdList(dst)`.
///
/// C needs the endpoints to locate the edge in its adjacency matrices, so they
/// travel even though the edge id alone identifies it.
pub fn write_delete_edge(
    buf: &mut Vec<u8>,
    ids: &IdList,
    relation_id: i32,
    src: &IdList,
    dst: &IdList,
) {
    debug_assert_eq!(ids.len(), src.len(), "one source per edge");
    debug_assert_eq!(ids.len(), dst.len(), "one destination per edge");
    write_header(buf, Opcode::DeleteEdge, Some(ids.count()));
    write_rel_type(buf, relation_id);
    ids.encode(buf);
    src.encode(buf);
    dst.encode(buf);
}

/// `7 SET_LABELS` / `8 REMOVE_LABELS` — `count · IdList · LabelSet`.
///
/// The labels, and all their nodes — not one `(node, label)` pair per node.
/// Grouping is what lets the node ids be one contiguous run: 10,000 nodes
/// gaining one label is 46 bytes grouped against 120,008 as pairs. It is sound
/// because label add and remove are idempotent set operations, so order within
/// the record carries nothing.
pub fn write_labels(
    buf: &mut Vec<u8>,
    add: bool,
    ids: &IdList,
    labels: &[i32],
) {
    let opcode = if add {
        Opcode::SetLabels
    } else {
        Opcode::RemoveLabels
    };
    write_header(buf, opcode, Some(ids.count()));
    write_label_set(buf, labels);
    ids.encode(buf);
}

/// `9 ADD_SCHEMA` — `EntityType · id · name`. Singular, so no count.
///
/// **v3 adds the id.** v2 sent the name alone and let the replica infer the id
/// from its own append order, which is the one assumption every other record's
/// bare ids rest on — and the only one that could not be checked. The replica
/// computes the id it would assign and rejects the buffer if it disagrees.
pub fn write_add_schema(
    buf: &mut Vec<u8>,
    schema_type: EntityType,
    id: i32,
    name: &str,
) {
    write_header(buf, Opcode::AddSchema, None);
    write_u32(buf, schema_tag(schema_type));
    write_label_id(buf, id);
    write_string(buf, name);
}

/// `10 ADD_ATTRIBUTE` — `id · name`. Singular, so no count.
///
/// **v3 adds the id**, for the reason on [`write_add_schema`]. This is the
/// record whose missing id let a property effect carrying a bare `u16` land on
/// the wrong attribute of an RDB-seeded replica.
///
/// No node/relationship discriminator: C has one attribute dictionary, and
/// #2459 unified Rust's two to match.
pub fn write_add_attribute(
    buf: &mut Vec<u8>,
    id: u16,
    name: &str,
) {
    write_header(buf, Opcode::AddAttribute, None);
    write_u16(buf, id);
    write_string(buf, name);
}

/// One indexed field: which attribute, and how it is indexed.
///
/// Grouped rather than passed as three loose arguments because `label_id` and
/// `attr_id` are both small integers next to each other on the wire —
/// transposing them is precisely the kind of mistake that does not fail on the
/// far side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexField<S> {
    pub attr_id: u16,
    pub attr: S,
}

/// `11 CREATE_INDEX` — **one record per field**, matching C.
///
/// Unchanged from v2 and deliberately not batched: index DDL is rare, and C
/// applies it idempotently per field (`Index_SetLanguage` tolerates a re-set,
/// `Index_SetStopwords` is guarded). Splitting a multi-field index into one
/// record per field is what lets a replica apply the fields in any order.
///
/// Both the label and the attribute travel as `(id, name)`. The id is
/// authoritative and the name is the cross-check that turns a numbering
/// disagreement into a loud failure rather than a write to the wrong attribute.
pub fn write_create_index(
    buf: &mut Vec<u8>,
    schema_type: EntityType,
    label_id: i32,
    label: &str,
    field_type: u32,
    fields: &[IndexField<&str>],
    options: &Value,
) {
    write_header(buf, Opcode::CreateIndex, None);
    write_u32(buf, schema_tag(schema_type));
    write_label_id(buf, label_id);
    write_string(buf, label);
    write_u32(buf, field_type);
    write_index_fields(buf, fields);
    options.encode(buf);
}

/// Every field of one index statement: `u16 n` then `(attr_id, attr) x n`.
///
/// A statement, not a field — C sends one record per field, and this
/// deliberately does not. Applying two single-field records is not equivalent to
/// one two-field statement: the second is refused with `Can not override index
/// configuration: Language is already set for label 'D'`, because index-level
/// options like `language` and `stopwords` belong to the index rather than the
/// field and cannot be set twice. Rust's `create_index` is multi-field, so one
/// record reproduces the primary's call exactly.
///
/// The field *type* sits outside this list, in the record, for the same reason:
/// it describes the statement. It also has to, because the list can be empty —
/// `db.idx.fulltext.drop('L')` names no attributes and lets the far side expand
/// them from the index — and a per-field type would vanish with the fields.
fn write_index_fields(
    buf: &mut Vec<u8>,
    fields: &[IndexField<&str>],
) {
    // Floor: 2 bytes of id and an 8-byte length per field, the same minimum
    // `read_index_fields` guards the count against.
    buf.reserve(2 + fields.len() * 10);
    write_u16(buf, fields.len() as u16);
    for field in fields {
        write_u16(buf, field.attr_id);
        write_string(buf, field.attr);
    }
}

fn read_index_fields(r: &mut Reader<'_>) -> Result<Vec<IndexField<String>>, DecodeError> {
    let n = r.u16()?;
    // Each field is at least 2 bytes of id plus an 8-byte length.
    let n = r.guard_count(u64::from(n), 10)?;
    let mut fields = Vec::with_capacity(n);
    for _ in 0..n {
        fields.push(IndexField {
            attr_id: r.u16()?,
            attr: r.string()?,
        });
    }
    Ok(fields)
}

/// `12 DROP_INDEX` — mirrors [`write_create_index`] without the options.
pub fn write_drop_index(
    buf: &mut Vec<u8>,
    schema_type: EntityType,
    label_id: i32,
    label: &str,
    field_type: u32,
    fields: &[IndexField<&str>],
) {
    write_header(buf, Opcode::DropIndex, None);
    write_u32(buf, schema_tag(schema_type));
    write_label_id(buf, label_id);
    write_string(buf, label);
    write_u32(buf, field_type);
    write_index_fields(buf, fields);
}

/// `13 CREATE_CONSTRAINT` / `14 DROP_CONSTRAINT`.
///
/// The property count is a **`u8`**, not the `u16` used elsewhere — C's
/// `uint8_t n`. Each property travels as `(id, name)` for the same reason the
/// index records do.
/// A constraint as a record carries it.
///
/// Grouped for the same reason as [`IndexField`]: the writer took eight
/// positional arguments, six of which described one constraint.
pub struct ConstraintSpec<'a> {
    pub constraint_type: ConstraintType,
    pub entity_type: EntityType,
    /// The primary's outcome, and `None` for a drop. See [`write_constraint`].
    pub status: Option<ConstraintStatus>,
    pub label_id: i32,
    pub label: &'a str,
    pub props: &'a [(u16, &'a str)],
}

pub fn write_constraint(
    buf: &mut Vec<u8>,
    create: bool,
    c: &ConstraintSpec<'_>,
) {
    let ConstraintSpec {
        constraint_type,
        entity_type,
        status,
        label_id,
        label,
        props,
    } = *c;
    debug_assert!(
        props.len() <= u8::MAX as usize,
        "C reads the count as uint8"
    );
    write_header(
        buf,
        if create {
            Opcode::CreateConstraint
        } else {
            Opcode::DropConstraint
        },
        None,
    );
    debug_assert_eq!(
        create,
        status.is_some(),
        "status belongs to a create record and only to a create record"
    );
    write_u32(buf, constraint_tag(constraint_type));
    write_u32(buf, entity_tag(entity_type));
    // Create only, and the one place v3 deliberately carries more than C: C's
    // `EffectsBuffer_AddCreateConstraintEffect` sends no status, so its replica
    // cannot tell an enforcing constraint from one still being built. A replica
    // does not validate, so this is the only thing that can tell it — and it is
    // what makes the second announcement, after validation finishes, converge
    // rather than duplicate.
    //
    // A *drop* has no such need: the apply path calls `drop_constraint` and
    // never reads the field. Sending one anyway was a `u32` C does not send, on
    // a record that had no use for it.
    if let Some(status) = status {
        write_u32(buf, constraint_status_tag(status));
    }
    write_label_id(buf, label_id);
    write_string(buf, label);
    // Floor, as above: 2 bytes of id and an 8-byte length per property.
    buf.reserve(1 + props.len() * 10);
    write_u8(buf, props.len() as u8);
    for (attr_id, name) in props {
        write_u16(buf, *attr_id);
        write_string(buf, name);
    }
}

/// One decoded record.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    Update {
        entity: EntityType,
        ids: IdList,
        /// The nodes' derived labels. Always empty for [`EntityType::Relationship`],
        /// which carries none on the wire.
        labels: Vec<i32>,
        attr_ids: Vec<u16>,
        rows: Vec<Value>,
    },
    CreateNode {
        ids: IdList,
        labels: Vec<i32>,
        attr_ids: Vec<u16>,
        rows: Vec<Value>,
    },
    CreateEdge {
        ids: IdList,
        relation_id: i32,
        src: IdList,
        dst: IdList,
        attr_ids: Vec<u16>,
        rows: Vec<Value>,
    },
    DeleteNode {
        ids: IdList,
        labels: Vec<i32>,
    },
    DeleteEdge {
        ids: IdList,
        relation_id: i32,
        src: IdList,
        dst: IdList,
    },
    Labels {
        add: bool,
        ids: IdList,
        labels: Vec<i32>,
    },
    AddSchema {
        schema_type: EntityType,
        id: i32,
        name: String,
    },
    AddAttribute {
        id: u16,
        name: String,
    },
    Index {
        create: bool,
        schema_type: EntityType,
        label_id: i32,
        label: String,
        /// C's index-field flags. A property of the statement, not of a field.
        field_type: u32,
        /// Every field of the statement. See [`write_index_fields`].
        fields: Vec<IndexField<String>>,
        /// `None` on a drop, which carries no options.
        options: Option<Value>,
    },
    Constraint {
        create: bool,
        constraint_type: ConstraintType,
        entity_type: EntityType,
        /// The primary's outcome; `None` on a drop, which carries none.
        status: Option<ConstraintStatus>,
        label_id: i32,
        label: String,
        props: Vec<(u16, String)>,
    },
}

/// Read one record, whatever its opcode.
pub fn read_record(r: &mut Reader<'_>) -> Result<Record, DecodeError> {
    let opcode = Opcode::try_from(r.u32()?)?;

    // The singular records carry no count, so one is only read for the records
    // that have it — the same predicate the writer heads them with. Every
    // opcode is then handled by the single match below, so a record cannot be
    // added to the enum without the compiler asking what it decodes to.
    let count = if opcode.is_batchable() { r.u32()? } else { 0 };
    let record = match opcode {
        Opcode::UpdateNode | Opcode::UpdateEdge => {
            let entity = if opcode == Opcode::UpdateNode {
                EntityType::Node
            } else {
                EntityType::Relationship
            };
            let labels = if entity == EntityType::Node {
                read_label_set(r)?
            } else {
                Vec::new()
            };
            let attr_ids = read_attr_ids(r)?;
            let ids = IdList::decode(r, count)?;
            let rows = read_attr_values(r, count, attr_ids.len())?;
            Record::Update {
                entity,
                ids,
                labels,
                attr_ids,
                rows,
            }
        }
        Opcode::CreateNode => {
            let labels = read_label_set(r)?;
            let attr_ids = read_attr_ids(r)?;
            let ids = IdList::decode(r, count)?;
            let rows = read_attr_values(r, count, attr_ids.len())?;
            Record::CreateNode {
                ids,
                labels,
                attr_ids,
                rows,
            }
        }
        Opcode::CreateEdge => {
            let relation_id = read_rel_type(r)?;
            let attr_ids = read_attr_ids(r)?;
            let ids = IdList::decode(r, count)?;
            let src = IdList::decode(r, count)?;
            let dst = IdList::decode(r, count)?;
            let rows = read_attr_values(r, count, attr_ids.len())?;
            Record::CreateEdge {
                ids,
                relation_id,
                src,
                dst,
                attr_ids,
                rows,
            }
        }
        Opcode::DeleteNode => {
            let labels = read_label_set(r)?;
            Record::DeleteNode {
                labels,
                ids: IdList::decode(r, count)?,
            }
        }
        Opcode::DeleteEdge => {
            let relation_id = read_rel_type(r)?;
            let ids = IdList::decode(r, count)?;
            let src = IdList::decode(r, count)?;
            let dst = IdList::decode(r, count)?;
            Record::DeleteEdge {
                ids,
                relation_id,
                src,
                dst,
            }
        }
        Opcode::SetLabels | Opcode::RemoveLabels => {
            // Field order in a struct literal is evaluation order, and the
            // label set now precedes the ids on the wire — so these must not be
            // written in declaration order.
            let labels = read_label_set(r)?;
            Record::Labels {
                add: opcode == Opcode::SetLabels,
                ids: IdList::decode(r, count)?,
                labels,
            }
        }
        Opcode::AddSchema => {
            let schema_type = entity_from_schema_tag(r.u32()?)?;
            let id = r.i32()?;
            Record::AddSchema {
                schema_type,
                id,
                name: r.string()?,
            }
        }
        Opcode::AddAttribute => {
            let id = r.u16()?;
            Record::AddAttribute {
                id,
                name: r.string()?,
            }
        }
        Opcode::CreateIndex | Opcode::DropIndex => {
            let create = opcode == Opcode::CreateIndex;
            let schema_type = entity_from_schema_tag(r.u32()?)?;
            let label_id = r.i32()?;
            let label = r.string()?;
            let field_type = r.u32()?;
            let fields = read_index_fields(r)?;
            let options = if create {
                Some(Value::decode(r)?)
            } else {
                None
            };
            Record::Index {
                create,
                schema_type,
                label_id,
                label,
                field_type,
                fields,
                options,
            }
        }
        Opcode::CreateConstraint | Opcode::DropConstraint => {
            let create = opcode == Opcode::CreateConstraint;
            let constraint_type = constraint_from_tag(r.u32()?)?;
            let entity_type = entity_from_tag(r.u32()?)?;
            // Present on a create only — see `write_constraint`.
            let status = if create {
                Some(constraint_status_from_tag(r.u32()?)?)
            } else {
                None
            };
            let label_id = r.i32()?;
            let label = r.string()?;
            let n = r.u8()?;
            // Each pair is at least 2 bytes of id plus an 8-byte length.
            let n = r.guard_count(u64::from(n), 10)?;
            let mut props = Vec::with_capacity(n);
            for _ in 0..n {
                let attr_id = r.u16()?;
                props.push((attr_id, r.string()?));
            }
            Record::Constraint {
                create,
                constraint_type,
                entity_type,
                status,
                label_id,
                label,
                props,
            }
        }
    };
    Ok(record)
}

/// Write one record, whatever its opcode.
impl EffectEncode<3> for Record {
    fn encode(
        &self,
        buf: &mut Vec<u8>,
    ) {
        match self {
            Record::AddSchema {
                schema_type,
                id,
                name,
            } => write_add_schema(buf, *schema_type, *id, name),
            Record::AddAttribute { id, name } => write_add_attribute(buf, *id, name),
            Record::CreateNode {
                ids,
                labels,
                attr_ids,
                rows,
            } => write_create_node(buf, ids, labels, attr_ids, rows),
            Record::CreateEdge {
                ids,
                relation_id,
                src,
                dst,
                attr_ids,
                rows,
            } => write_create_edge(buf, ids, *relation_id, src, dst, attr_ids, rows),
            Record::Update {
                entity,
                ids,
                labels,
                attr_ids,
                rows,
            } => write_update(buf, *entity, ids, labels, attr_ids, rows),
            Record::Labels { add, ids, labels } => write_labels(buf, *add, ids, labels),
            Record::DeleteNode { ids, labels } => write_delete_node(buf, ids, labels),
            Record::DeleteEdge {
                ids,
                relation_id,
                src,
                dst,
            } => write_delete_edge(buf, ids, *relation_id, src, dst),
            Record::Index {
                create,
                schema_type,
                label_id,
                label,
                field_type,
                fields,
                options,
            } => {
                let borrowed: Vec<IndexField<&str>> = fields
                    .iter()
                    .map(|f| IndexField {
                        attr_id: f.attr_id,
                        attr: f.attr.as_str(),
                    })
                    .collect();
                if *create {
                    write_create_index(
                        buf,
                        *schema_type,
                        *label_id,
                        label,
                        *field_type,
                        &borrowed,
                        options.as_ref().unwrap_or(&Value::Null),
                    );
                } else {
                    write_drop_index(buf, *schema_type, *label_id, label, *field_type, &borrowed);
                }
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
                let props: Vec<(u16, &str)> = props.iter().map(|(i, n)| (*i, n.as_str())).collect();
                write_constraint(
                    buf,
                    *create,
                    &ConstraintSpec {
                        constraint_type: *constraint_type,
                        entity_type: *entity_type,
                        status: *status,
                        label_id: *label_id,
                        label,
                        props: &props,
                    },
                );
            }
        }
    }
}

/// The effects encoding of one record.
///
/// `encode` is total — every variant has a shape — while `decode` reads the
/// opcode first and dispatches, so the pair is not symmetric in signature even
/// though it is in effect.
impl EffectDecode<3> for Record {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        read_record(r)
    }
}

/// A payload with its header consumed, ready to yield records.
///
/// Owns the plaintext only when it had to decompress; an uncompressed payload
/// is borrowed straight from the caller's buffer.
pub enum Payload<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl Payload<'_> {
    /// The record stream, decoded one at a time.
    #[must_use]
    pub fn records(&self) -> Records<'_> {
        Records {
            failed: false,
            r: Reader::new(match self {
                Self::Borrowed(b) => b,
                Self::Owned(v) => v,
            }),
        }
    }
}

/// Records decoded on demand.
///
/// The item is a `Result`, so a malformed record is *yielded* as an error
/// rather than ending the stream — the caller still has to handle it, and
/// `collect::<Result<Vec<_>, _>>()` or a `?` in the loop body does. What the
/// iterator must not do is keep going afterwards: the reader is mid-record and
/// everything after it is nonsense, so it fuses.
pub struct Records<'a> {
    r: Reader<'a>,
    failed: bool,
}

impl Iterator for Records<'_> {
    type Item = Result<Record, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.r.is_empty() {
            return None;
        }
        let record = <Record as EffectDecode<3>>::decode(&mut self.r);
        self.failed = record.is_err();
        Some(record)
    }
}

/// Consume a `GRAPH.EFFECT` payload's header, leaving the records.
///
/// One command carries one payload carries many records — batching removes
/// per-record framing, it never multiplies commands.
///
/// # Errors
///
/// Returns [`DecodeError`] if the version is unreadable, a flag is unknown, or
/// a compressed frame is malformed.
pub fn open_payload(buf: &[u8]) -> Result<Payload<'_>, DecodeError> {
    let mut r = Reader::new(buf);
    let version = r.u8()?;
    if version != EFFECTS_VERSION {
        return Err(DecodeError::UnsupportedVersion(version));
    }
    let flags = r.u8()?;
    if flags & !KNOWN_FLAGS != 0 {
        return Err(DecodeError::UnknownFlags(flags));
    }

    if flags & FLAG_COMPRESSED == 0 {
        return Ok(Payload::Borrowed(r.rest()));
    }

    // A frame has to be whole before any of it can be read, so this is the one
    // place the payload cannot be streamed.
    let plain_len = r.u32()? as usize;
    let checksum = r.u32()?;
    let frame = r.take(r.remaining())?;
    // `bulk::decompress` rather than `stream::decode_all`, because it takes the
    // declared length as the allocation *ceiling* and refuses a frame that wants
    // more. `decode_all` grows to whatever the frame expands to, and zstd's
    // ratio on repetitive input is unbounded in practice: a ~100-byte frame of
    // zeros inflates to gigabytes, and the length check below only ran after
    // that allocation had already happened.
    let plain = zstd::bulk::decompress(frame, plain_len)
        .map_err(|e| DecodeError::BadCompression(e.to_string()))?;
    // Still cross-checked, because the ceiling is an upper bound: a frame that
    // expands to *less* than declared means the header and the payload disagree,
    // and the records after it cannot be trusted.
    if plain.len() != plain_len {
        return Err(DecodeError::CompressedLengthMismatch {
            declared: plain_len,
            actual: plain.len(),
        });
    }
    // Checked over the *plaintext*, not the frame: it is the bytes the records
    // are actually parsed from, and it stays reproducible across zstd versions,
    // which may frame the same input differently. CRC-32/ISO-HDLC — the zlib
    // polynomial — so C can match it without vendoring anything.
    let actual = crc32fast::hash(&plain);
    if actual != checksum {
        return Err(DecodeError::ChecksumMismatch {
            declared: checksum,
            actual,
        });
    }
    Ok(Payload::Owned(plain))
}

/// Every record in a payload, materialized.
///
/// Convenient for tests and for comparing two payloads; the apply path streams
/// instead, so it never holds more than one record at a time.
///
/// # Errors
///
/// Returns [`DecodeError`] if the payload or any record in it is malformed.
pub fn read_buffer(buf: &[u8]) -> Result<Vec<Record>, DecodeError> {
    let payload = open_payload(buf)?;
    let mut records = Vec::new();
    for record in payload.records() {
        records.push(record?);
    }
    Ok(records)
}

/// Compress a finished payload in place, if that makes it smaller.
///
/// **Default off.** Compression is a bandwidth trade, not a CPU one: measured
/// at 3.245 cycles/byte to compress against 0.067 to copy into the replica
/// output buffer, so on a fast link it spends far more than the bytes are
/// worth. `min_bytes` of 0 disables it entirely, which is what
/// `EFFECTS_COMPRESSION` defaults to.
///
/// Even when enabled the smaller form wins: zstd inflates an already-minimal
/// buffer, and the batched records are already most of the way there.
///
/// Returns whether the payload ended up compressed.
/// zstd level for effect payloads.
///
/// Level 1. Effects are latency-sensitive — the payload is built on the write
/// thread while it holds the GIL — and the corpus is highly repetitive, so the
/// cheapest level already gets most of the ratio. Not configurable because no
/// measurement has yet shown a level worth choosing between.
/// zstd level for the payload frame.
///
/// **Level 1's ratio on effects payloads is a coin flip on alignment.** Its fast
/// match-finder phase-locks onto the record period, so shifting a payload by one
/// byte swings the output: measured on 10,000 `CREATE_NODE` value rows, the same
/// bytes compress to 10,399 at one alignment and ~31,100 at the other eight.
/// Level 3 gives 27,250 at every alignment — worse than level 1's lucky case,
/// better than its common one, and stable, which a wire format wants more than
/// it wants an occasional 3x.
///
/// Left at 1 for now because raising it is a CPU trade that has not been
/// measured on the write thread; tracked separately.
const COMPRESSION_LEVEL: i32 = 1;

pub fn maybe_compress(
    buf: &mut Vec<u8>,
    min_bytes: usize,
) -> bool {
    const HEADER: usize = 2;
    if min_bytes == 0 || buf.len() < HEADER || buf.len() - HEADER < min_bytes {
        return false;
    }
    // Compressing twice produces a payload nothing can read: the second pass
    // would swallow the first one's length prefix and checksum as if they were
    // records, and the reader inflates once. This must be called exactly once,
    // on a finished payload; the guard is here because "exactly once" is an
    // easy thing for a caller to get wrong when a query commits more than once.
    // No `debug_assert` here on purpose: refusing *is* the safety property, and
    // asserting would make it untestable in the builds the tests run in.
    if buf[1] & FLAG_COMPRESSED != 0 {
        return false;
    }
    let Ok(frame) = zstd::stream::encode_all(&buf[HEADER..], COMPRESSION_LEVEL) else {
        // Compression failing is not a reason to fail the write; the
        // uncompressed payload is still correct.
        return false;
    };
    // The declared length and the checksum ride along, so they count.
    if frame.len() + 8 >= buf.len() - HEADER {
        return false;
    }

    let plain_len = (buf.len() - HEADER) as u32;
    let checksum = crc32fast::hash(&buf[HEADER..]);
    buf.truncate(HEADER);
    buf[1] |= FLAG_COMPRESSED;
    buf.extend_from_slice(&plain_len.to_le_bytes());
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf.extend_from_slice(&frame);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::testing::hex;

    // ── records ──

    #[test]
    fn create_node_pins_its_bytes_at_count_one() {
        // One node, label 7, one int property. The smallest possible record —
        // and the order is the invariant: schema first, data last.
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([0]), &[7], &[0], &[Value::Int(1)]);
        assert_eq!(
            hex(&buf),
            concat!(
                "03 00 ",                              // version, flags
                "03 00 00 00 01 00 00 00 ",            // CREATE_NODE, count = 1
                "01 00 07 00 00 00 ",                  // LabelSet: n = 1, label 7
                "01 00 00 00 ",                        // AttrIds: n = 1, attr 0
                "01 00 00 00 00 00 01 ",               // IdList: 1 segment; range, base 0, len 1
                "00 20 00 00 01 00 00 00 00 00 00 00"  // AttrValues: T_INT64, 1
            )
        );
    }

    #[test]
    fn create_node_is_the_same_record_at_ten_thousand() {
        // Same opcode, same blocks, one count field apart — which is the whole
        // point of folding the count in rather than adding a batch opcode.
        let ids: IdList = (0..10_000).collect();
        let rows: Vec<Value> = (0..10_000).map(|i| Value::Int(i as i64)).collect();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &ids, &[7], &[0], &rows);

        assert_eq!(&buf[2..6], &(Opcode::CreateNode as u32).to_le_bytes());
        assert_eq!(&buf[6..10], &10_000_u32.to_le_bytes());
        // Schema first: header ends at 10, LabelSet (6 B) then AttrIds (4 B),
        // so the id block starts at 20. Sequentially allocated ids are a
        // consecutive run, so all ten thousand of them are three bytes.
        // One segment: count, header byte carrying both widths, then the base
        // and the length. Flat in the id count — a million ids cost what ten do.
        assert_eq!(&buf[20..28], &[1, 0, 0, 0, 0x08, 0, 0x10, 0x27]);

        let records = read_buffer(&buf).unwrap();
        assert_eq!(records.len(), 1);
        let Record::CreateNode {
            ids: got, rows: r, ..
        } = &records[0]
        else {
            panic!("wrong record: {:?}", records[0]);
        };
        assert_eq!(got, &ids);
        assert_eq!(r.len(), 10_000);
    }

    #[test]
    fn create_edge_keeps_endpoints_aligned_with_their_edges() {
        // Three edges out of one source: an IdSet would have collapsed the
        // sources to one entry and misaligned every row after the first.
        let ids = IdList::from([10_u64, 11, 12]);
        let src = IdList::from([7_u64, 7, 7]);
        let dst = IdList::from([1_u64, 2, 3]);
        let mut buf = new_buffer();
        write_create_edge(&mut buf, &ids, 4, &src, &dst, &[], &[]);

        let records = read_buffer(&buf).unwrap();
        let Record::CreateEdge {
            ids: gi,
            relation_id,
            src: gs,
            dst: gd,
            ..
        } = &records[0]
        else {
            panic!("wrong record");
        };
        assert_eq!(gi, &ids);
        assert_eq!(*relation_id, 4);
        assert_eq!(gs, &src, "duplicate sources must survive");
        assert_eq!(gd, &dst);
    }

    #[test]
    fn set_labels_carries_the_labels_and_all_their_nodes() {
        // Grouped, not one (node, label) pair per node: 10,000 nodes gaining
        // one label fit in a few dozen bytes because the ids are one run.
        let ids: IdList = (0..10_000).collect();
        let mut buf = new_buffer();
        write_labels(&mut buf, true, &ids, &[5]);
        assert!(buf.len() < 100, "grouped should be tiny, got {}", buf.len());

        let records = read_buffer(&buf).unwrap();
        assert_eq!(
            records[0],
            Record::Labels {
                add: true,
                ids,
                labels: vec![5],
            }
        );
    }

    #[test]
    fn schema_records_carry_their_id() {
        // The v3 change: v2 sent the name alone and let the replica infer the
        // id from its own append order — the assumption every bare id rests on,
        // and the only one that could not be checked.
        let mut buf = new_buffer();
        write_add_schema(&mut buf, EntityType::Node, 3, "Person");
        write_add_attribute(&mut buf, 9, "name");

        let records = read_buffer(&buf).unwrap();
        assert_eq!(
            records,
            vec![
                Record::AddSchema {
                    schema_type: EntityType::Node,
                    id: 3,
                    name: "Person".into(),
                },
                Record::AddAttribute {
                    id: 9,
                    name: "name".into(),
                },
            ]
        );
    }

    #[test]
    fn schema_records_carry_no_count() {
        // They are inherently singular, so they are the documented exception to
        // "every record has a count".
        let mut buf = Vec::new();
        write_add_attribute(&mut buf, 9, "n");
        // opcode, then the id straight away — no 4-byte count in between.
        assert_eq!(&buf[..4], &(Opcode::AddAttribute as u32).to_le_bytes());
        assert_eq!(&buf[4..6], &9_u16.to_le_bytes());
    }

    #[test]
    fn every_record_type_round_trips() {
        let mut buf = new_buffer();
        write_add_schema(&mut buf, EntityType::Relationship, 1, "KNOWS");
        write_add_attribute(&mut buf, 0, "since");
        write_create_node(
            &mut buf,
            &IdList::from([1, 2]),
            &[7],
            &[0],
            &[Value::Int(1), Value::Null],
        );
        write_create_edge(
            &mut buf,
            &IdList::from([5, 6]),
            1,
            &IdList::from([1, 1]),
            &IdList::from([2, 2]),
            &[0],
            &[Value::Int(2020), Value::Int(2021)],
        );
        write_update(
            &mut buf,
            EntityType::Node,
            &IdList::from([1]),
            &[7],
            &[0],
            &[Value::Int(9)],
        );
        write_update(
            &mut buf,
            EntityType::Relationship,
            &IdList::from([5]),
            &[],
            &[0],
            &[Value::Null],
        );
        write_labels(&mut buf, true, &IdList::from([1, 2]), &[7, 8]);
        write_labels(&mut buf, false, &IdList::from([1]), &[8]);
        write_delete_edge(
            &mut buf,
            &IdList::from([5, 6]),
            1,
            &IdList::from([1, 1]),
            &IdList::from([2, 2]),
        );
        write_delete_node(&mut buf, &IdList::from([1, 2]), &[7]);
        write_create_index(
            &mut buf,
            EntityType::Node,
            7,
            "L",
            INDEX_FLD_RANGE,
            &[IndexField {
                attr_id: 0,
                attr: "since",
            }],
            &Value::Null,
        );
        write_drop_index(
            &mut buf,
            EntityType::Node,
            7,
            "L",
            INDEX_FLD_RANGE,
            &[IndexField {
                attr_id: 0,
                attr: "since",
            }],
        );
        write_constraint(
            &mut buf,
            true,
            &ConstraintSpec {
                constraint_type: ConstraintType::Unique,
                entity_type: EntityType::Node,
                status: Some(ConstraintStatus::Operational),
                label_id: 7,
                label: "L",
                props: &[(0, "since")],
            },
        );
        write_constraint(
            &mut buf,
            false,
            &ConstraintSpec {
                constraint_type: ConstraintType::Mandatory,
                entity_type: EntityType::Relationship,
                status: None,
                label_id: 1,
                label: "KNOWS",
                props: &[],
            },
        );

        let records = read_buffer(&buf).unwrap();
        assert_eq!(records.len(), 14, "one command, fourteen records");

        // Re-encoding what was decoded must reproduce the buffer exactly.
        let mut again = new_buffer();
        for record in &records {
            match record {
                Record::AddSchema {
                    schema_type,
                    id,
                    name,
                } => {
                    write_add_schema(&mut again, *schema_type, *id, name);
                }
                Record::AddAttribute { id, name } => write_add_attribute(&mut again, *id, name),
                Record::CreateNode {
                    ids,
                    labels,
                    attr_ids,
                    rows,
                } => {
                    write_create_node(&mut again, ids, labels, attr_ids, rows);
                }
                Record::CreateEdge {
                    ids,
                    relation_id,
                    src,
                    dst,
                    attr_ids,
                    rows,
                } => {
                    write_create_edge(&mut again, ids, *relation_id, src, dst, attr_ids, rows);
                }
                Record::Update {
                    entity,
                    ids,
                    labels,
                    attr_ids,
                    rows,
                } => {
                    write_update(&mut again, *entity, ids, labels, attr_ids, rows);
                }
                Record::Labels { add, ids, labels } => {
                    write_labels(&mut again, *add, ids, labels);
                }
                Record::DeleteEdge {
                    ids,
                    relation_id,
                    src,
                    dst,
                } => {
                    write_delete_edge(&mut again, ids, *relation_id, src, dst);
                }
                Record::DeleteNode { ids, labels } => {
                    write_delete_node(&mut again, ids, labels);
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
                    let borrowed: Vec<IndexField<&str>> = fields
                        .iter()
                        .map(|f| IndexField {
                            attr_id: f.attr_id,
                            attr: f.attr.as_str(),
                        })
                        .collect();
                    if *create {
                        write_create_index(
                            &mut again,
                            *schema_type,
                            *label_id,
                            label,
                            *field_type,
                            &borrowed,
                            options.as_ref().expect("a create carries options"),
                        );
                    } else {
                        write_drop_index(
                            &mut again,
                            *schema_type,
                            *label_id,
                            label,
                            *field_type,
                            &borrowed,
                        );
                    }
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
                    let props: Vec<(u16, &str)> =
                        props.iter().map(|(id, n)| (*id, n.as_str())).collect();
                    write_constraint(
                        &mut again,
                        *create,
                        &ConstraintSpec {
                            constraint_type: *constraint_type,
                            entity_type: *entity_type,
                            status: *status,
                            label_id: *label_id,
                            label,
                            props: &props,
                        },
                    );
                }
            }
        }
        assert_eq!(hex(&again), hex(&buf), "encode(decode(x)) must equal x");
    }

    /// One `CREATE_NODE` record per distinct shape, for 10,000 nodes split
    /// `shapes` ways.
    fn create_10k_in_shapes(shapes: usize) -> Vec<u8> {
        let per = 10_000 / shapes;
        let mut buf = new_buffer();
        for s in 0..shapes {
            let ids: IdList = (0..per).map(|j| (j * shapes + s) as u64).collect();
            let rows: Vec<Value> = (0..per).map(|j| Value::Int(j as i64)).collect();
            write_create_node(&mut buf, &ids, &[7], &[s as u16], &rows);
        }
        buf
    }

    #[test]
    fn the_motivating_query_end_to_end() {
        // UNWIND range(0,9999) AS i CREATE (:L {v:i})
        // v2 emitted one record per node, 34 bytes each: 340,001 bytes.
        const V2: f64 = 340_001.0;
        println!("\n  10,000 nodes            shapes     v3 bytes    vs v2");
        for shapes in [1_usize, 20, 500, 2_000, 10_000] {
            let buf = create_10k_in_shapes(shapes);
            let n = read_buffer(&buf).unwrap().len();
            assert_eq!(n, shapes, "one record per shape");
            println!(
                "  {:<22} {shapes:6} {:12} {:+7.1}%",
                if shapes == 1 {
                    "all share one shape"
                } else if shapes == 10_000 {
                    "every node its own"
                } else {
                    "split"
                },
                buf.len(),
                100.0 * buf.len() as f64 / V2 - 100.0
            );
        }

        // The floor moved deliberately, and this records by how much. A segment
        // list states both its segment count and every segment's length, so
        // that it is well-formed on its own rather than only inside the record
        // carrying it. That is five bytes per single-id record — four for the
        // count, one for the length — which is the whole difference between
        // this bound and the 340,001 that preceded the segment format.
        let worst = create_10k_in_shapes(10_000);
        assert!(
            worst.len() <= 380_001,
            "singleton floor regressed: {}",
            worst.len()
        );

        // Two shapes, ids interleaved — the alternating-label query. Stride 2
        // breaks the single run, so each IdSet falls back to a bitset container.
        let even: IdList = (0..5_000).map(|i| i * 2).collect();
        let odd: IdList = (0..5_000).map(|i| i * 2 + 1).collect();
        let half: Vec<Value> = (0..5_000).map(|i| Value::Int(i as i64)).collect();
        let mut alt = new_buffer();
        write_create_node(&mut alt, &even, &[7], &[0], &half);
        write_create_node(&mut alt, &odd, &[8], &[0], &half);
        println!("\n  alternating labels, 2 records: {} B", alt.len());
        assert_eq!(read_buffer(&alt).unwrap().len(), 2);

        // 10,000 edges out of one supernode.
        let eids: IdList = (0..10_000).collect();
        let src: IdList = std::iter::repeat_n(4_000_000_000_u64, 10_000).collect();
        let dst: IdList = (0..10_000).collect();
        let mut edges = new_buffer();
        write_create_edge(&mut edges, &eids, 1, &src, &dst, &[], &[]);
        println!("  10,000 supernode edges:        {} B", edges.len());
        assert_eq!(read_buffer(&edges).unwrap().len(), 1);
    }

    #[test]
    fn a_singleton_costs_about_what_v2_charged() {
        // The price of one uniform record shape. v2 spent 34 bytes on this
        // record; segments spend 36, and the three-byte difference is the
        // stated segment count — a decoder cannot otherwise tell a truncated
        // segment list from a complete one. The id itself still narrows, and the
        // final segment's length is implied by the record's count, so a lone id
        // is a header byte and a base.
        //
        //   4 opcode + 4 count + IdList + 6 LabelSet + 16 AttrSet
        let mut small = Vec::new();
        write_create_node(&mut small, &IdList::from([0]), &[7], &[0], &[Value::Int(1)]);
        assert_eq!(small.len(), 37, "id 0 narrows to one byte");

        let mut mid = Vec::new();
        write_create_node(
            &mut mid,
            &IdList::from([5_000]),
            &[7],
            &[0],
            &[Value::Int(1)],
        );
        assert_eq!(mid.len(), 38, "a mid-size graph's id takes two");

        let mut large = Vec::new();
        write_create_node(
            &mut large,
            &IdList::from([5_000_000]),
            &[7],
            &[0],
            &[Value::Int(1)],
        );
        assert_eq!(large.len(), 40, "past 2^16 the id takes four");
    }

    // ── index and constraint records ──

    #[test]
    fn index_field_types_are_c_bit_flags() {
        // A range index is the union of the three scalar kinds, not a
        // discriminant of its own — 0x0E, which reads as garbage if treated as
        // an ordinal.
        assert_eq!(INDEX_FLD_RANGE, 0x0E);
        assert_eq!(INDEX_FLD_FULLTEXT, 0x01);
        assert_eq!(INDEX_FLD_VECTOR, 0x10);
        // And the two numberings C uses for the same node-or-edge enum: the
        // schema dictionary is 0-based, GraphEntityType is 1-based because
        // GETYPE_UNKNOWN takes 0.
        assert_eq!(
            (
                schema_tag(EntityType::Node),
                schema_tag(EntityType::Relationship)
            ),
            (0, 1)
        );
        assert_eq!(
            (
                entity_tag(EntityType::Node),
                entity_tag(EntityType::Relationship)
            ),
            (1, 2)
        );
    }

    #[test]
    fn create_index_keeps_the_name_beside_each_id() {
        let mut buf = new_buffer();
        write_create_index(
            &mut buf,
            EntityType::Node,
            3,
            "Person",
            INDEX_FLD_RANGE,
            &[IndexField {
                attr_id: 9,
                attr: "name",
            }],
            &Value::Null,
        );
        let records = read_buffer(&buf).unwrap();
        assert_eq!(
            records[0],
            Record::Index {
                create: true,
                schema_type: EntityType::Node,
                label_id: 3,
                label: "Person".into(),
                field_type: INDEX_FLD_RANGE,
                fields: vec![IndexField {
                    attr_id: 9,
                    attr: "name".into(),
                }],
                options: Some(Value::Null),
            }
        );
    }

    #[test]
    fn a_multi_field_index_is_one_record() {
        // One record per *statement*, not per field — and deliberately unlike
        // C, which sends one per field. Two single-field records are not
        // equivalent to one two-field statement: applying the second is refused
        // with "Can not override index configuration: Language is already set
        // for label 'D'", because index-level options belong to the index and
        // cannot be set twice. Verified against a live server.
        let mut buf = new_buffer();
        let fields: Vec<IndexField<&str>> = [(0_u16, "a"), (1, "b"), (2, "c")]
            .into_iter()
            .map(|(attr_id, attr)| IndexField { attr_id, attr })
            .collect();
        write_create_index(
            &mut buf,
            EntityType::Node,
            1,
            "L",
            INDEX_FLD_RANGE,
            &fields,
            &Value::Null,
        );

        let records = read_buffer(&buf).unwrap();
        assert_eq!(records.len(), 1, "one statement, one record");
        let Record::Index { fields, .. } = &records[0] else {
            panic!("wrong record: {:?}", records[0]);
        };
        assert_eq!(
            fields
                .iter()
                .map(|f| (f.attr_id, f.attr.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, "a"), (1, "b"), (2, "c")]
        );
    }

    #[test]
    fn drop_index_carries_no_options() {
        let mut buf = new_buffer();
        write_drop_index(
            &mut buf,
            EntityType::Relationship,
            1,
            "KNOWS",
            INDEX_FLD_RANGE,
            &[IndexField {
                attr_id: 0,
                attr: "since",
            }],
        );
        let records = read_buffer(&buf).unwrap();
        let Record::Index {
            create, options, ..
        } = &records[0]
        else {
            panic!("wrong record");
        };
        assert!(!create);
        assert_eq!(*options, None);
    }

    #[test]
    fn constraint_status_uses_cs_numbering_not_rusts() {
        // The one mapping in the format that is not the discriminant. C has
        // CT_ACTIVE = 0, CT_PENDING = 1 (constraint.h:39); Rust's enum reads
        // UnderConstruction, Operational, so the first two are the other way
        // round. Casting would send an enforcing constraint as pending, which
        // nothing would report — the replica would just believe the wrong
        // thing about whether it is enforcing.
        assert_eq!(constraint_status_tag(ConstraintStatus::Operational), 0);
        assert_eq!(
            constraint_status_tag(ConstraintStatus::UnderConstruction),
            1
        );
        assert_eq!(constraint_status_tag(ConstraintStatus::Failed), 2);

        for status in [
            ConstraintStatus::Operational,
            ConstraintStatus::UnderConstruction,
            ConstraintStatus::Failed,
        ] {
            assert_eq!(
                constraint_status_from_tag(constraint_status_tag(status)),
                Ok(status)
            );
        }
        assert_eq!(
            constraint_status_from_tag(3),
            Err(DecodeError::BadConstraintStatus(3))
        );
    }

    #[test]
    fn constraint_property_count_is_one_byte() {
        // C reads it as uint8_t, not the u16 used elsewhere in the format.
        let mut buf = Vec::new();
        write_constraint(
            &mut buf,
            true,
            &ConstraintSpec {
                constraint_type: ConstraintType::Unique,
                entity_type: EntityType::Node,
                status: Some(ConstraintStatus::Operational),
                label_id: 3,
                label: "Person",
                props: &[(0, "first"), (1, "last")],
            },
        );
        // opcode, ct, et, status, label_id, then the string, then a single
        // count byte.
        let after_label = 4 + 4 + 4 + 4 + 4 + (8 + "Person".len() + 1);
        assert_eq!(buf[after_label], 2, "the count is one byte wide");
    }

    #[test]
    fn constraints_round_trip_with_their_property_names() {
        let mut buf = new_buffer();
        write_constraint(
            &mut buf,
            true,
            &ConstraintSpec {
                constraint_type: ConstraintType::Unique,
                entity_type: EntityType::Node,
                status: Some(ConstraintStatus::Operational),
                label_id: 3,
                label: "Person",
                props: &[(0, "first"), (1, "last")],
            },
        );
        write_constraint(
            &mut buf,
            false,
            &ConstraintSpec {
                constraint_type: ConstraintType::Mandatory,
                entity_type: EntityType::Relationship,
                status: None,
                label_id: 1,
                label: "KNOWS",
                props: &[(2, "since")],
            },
        );

        let records = read_buffer(&buf).unwrap();
        assert_eq!(
            records[0],
            Record::Constraint {
                create: true,
                constraint_type: ConstraintType::Unique,
                entity_type: EntityType::Node,
                status: Some(ConstraintStatus::Operational),
                label_id: 3,
                label: "Person".into(),
                props: vec![(0, "first".into()), (1, "last".into())],
            }
        );
        let Record::Constraint {
            create,
            entity_type,
            props,
            ..
        } = &records[1]
        else {
            panic!("wrong record");
        };
        assert!(!create);
        assert_eq!(*entity_type, EntityType::Relationship);
        assert_eq!(props, &vec![(2_u16, "since".to_string())]);
    }

    #[test]
    fn ddl_records_carry_no_count() {
        // Like the schema records, these are singular.
        let mut buf = Vec::new();
        write_drop_index(
            &mut buf,
            EntityType::Node,
            1,
            "L",
            INDEX_FLD_RANGE,
            &[IndexField {
                attr_id: 0,
                attr: "a",
            }],
        );
        assert_eq!(&buf[..4], &(Opcode::DropIndex as u32).to_le_bytes());
        assert_eq!(
            &buf[4..8],
            &schema_tag(EntityType::Node).to_le_bytes(),
            "schema type, not a count"
        );
    }

    #[test]
    fn a_foreign_version_is_refused() {
        let mut buf = new_buffer();
        write_delete_node(&mut buf, &IdList::from([1]), &[7]);
        buf[0] = 2;
        assert_eq!(read_buffer(&buf), Err(DecodeError::UnsupportedVersion(2)));
    }

    #[test]
    fn an_unknown_opcode_is_refused() {
        let mut buf = new_buffer();
        write_u32(&mut buf, 99);
        write_u32(&mut buf, 0);
        assert_eq!(read_buffer(&buf), Err(DecodeError::BadOpcode(99)));
    }

    #[test]
    fn a_truncated_buffer_is_an_error_not_a_panic() {
        let mut buf = new_buffer();
        write_create_edge(
            &mut buf,
            &IdList::from([5, 6]),
            1,
            &IdList::from([1, 1]),
            &IdList::from([2, 2]),
            &[0],
            &[Value::Int(1), Value::Int(2)],
        );
        // From 3: a header with no records after it is a valid empty payload,
        // not a truncated one.
        for cut in 3..buf.len() {
            assert!(read_buffer(&buf[..cut]).is_err(), "cut at {cut}");
        }
    }

    // ── the payload header ──

    #[test]
    fn the_flags_byte_is_reserved_even_though_nothing_sets_it() {
        // Reserving it now is the point: adding a byte to the header later
        // would cost another version bump for something compression needs.
        let mut buf = new_buffer();
        write_delete_node(&mut buf, &IdList::from([1]), &[7]);
        assert_eq!(buf[0], EFFECTS_VERSION);
        assert_eq!(buf[1], 0, "no flags by default");
        assert_eq!(read_buffer(&buf).unwrap().len(), 1);
    }

    #[test]
    fn compression_round_trips_and_shrinks() {
        // 10,000 nodes of one shape: highly repetitive, which is the case worth
        // compressing at all.
        let ids: IdList = (0..10_000).collect();
        let rows: Vec<Value> = (0..10_000).map(|i| Value::Int(i as i64)).collect();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &ids, &[7], &[0], &rows);
        let plain = buf.clone();

        assert!(maybe_compress(&mut buf, 1024), "should compress");
        assert_eq!(buf[0], EFFECTS_VERSION, "the header stays plaintext");
        assert_eq!(buf[1], FLAG_COMPRESSED);
        // A third, not a quarter — and the change is not a regression in the
        // format. At `COMPRESSION_LEVEL = 1` zstd's fast match-finder phase-locks
        // onto this payload's 12-byte record period, so the ratio depends on
        // where the records happen to start. Measured on this exact body,
        // shifted one byte at a time: 31,098 / 10,399 / 31,105 / 31,105 / 31,109
        // ... one alignment in nine compresses 3x better than the rest, and the
        // encoding this replaces happened to sit on it. Level 3 is 27,250 at
        // every alignment. See the note in `maybe_compress`.
        assert!(
            buf.len() < plain.len() / 3,
            "{} vs {}",
            buf.len(),
            plain.len()
        );

        // And it decodes to exactly what the uncompressed payload does.
        assert_eq!(read_buffer(&buf).unwrap(), read_buffer(&plain).unwrap());
    }

    #[test]
    fn compression_is_off_at_zero_and_declines_when_it_would_not_help() {
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from([0]), &[7], &[0], &[Value::Int(1)]);
        let before = buf.clone();

        assert!(!maybe_compress(&mut buf, 0), "0 disables it");
        assert_eq!(buf, before);

        // Enabled, but a 33-byte record is below any sane floor — and even
        // forced, zstd inflates a payload this small, so the smaller form wins.
        assert!(!maybe_compress(&mut buf, 1));
        assert_eq!(
            buf, before,
            "left uncompressed because compressing is bigger"
        );
    }

    #[test]
    fn an_unknown_flag_is_refused_rather_than_ignored() {
        // An old node meeting a future payload must fail loudly. Decoding the
        // records anyway would apply a prefix of something whose shape it does
        // not know.
        let mut buf = new_buffer();
        write_delete_node(&mut buf, &IdList::from([1]), &[7]);
        buf[1] = 0x80;
        assert_eq!(read_buffer(&buf), Err(DecodeError::UnknownFlags(0x80)));
    }

    #[test]
    fn no_flipped_bit_is_accepted_as_different_data() {
        // The property that matters is not that every corruption errors — a few
        // bits live in zstd header fields it ignores, and flipping those yields
        // byte-identical output, which is not a failure. It is that corruption
        // is never accepted *as something else*. zstd catches most of it; the
        // checksum is what closes the rest, independently of the compressor.
        let ids: IdList = (0..2_000).collect();
        let rows: Vec<Value> = (0..2_000).map(|i| Value::Int(i as i64)).collect();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &ids, &[7], &[0], &rows);
        assert!(maybe_compress(&mut buf, 16));
        let want = read_buffer(&buf).expect("the clean buffer must decode");

        let mut refused = 0;
        let mut benign = 0;
        for i in 2..buf.len() {
            let mut bad = buf.clone();
            bad[i] ^= 0x01;
            match read_buffer(&bad) {
                Err(_) => refused += 1,
                Ok(got) => {
                    assert_eq!(got, want, "byte {i} decoded to different records");
                    benign += 1;
                }
            }
        }
        assert_eq!(refused + benign, buf.len() - 2);
        assert!(
            benign * 100 < refused,
            "{benign} benign against {refused} refused — the checksum is not working"
        );
    }

    #[test]
    fn a_tampered_checksum_is_refused() {
        let ids: IdList = (0..2_000).collect();
        let rows: Vec<Value> = (0..2_000).map(|i| Value::Int(i as i64)).collect();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &ids, &[7], &[0], &rows);
        assert!(maybe_compress(&mut buf, 16));

        // version, flags, u32 plain_len, then the checksum.
        buf[6] ^= 0xFF;
        assert!(matches!(
            read_buffer(&buf),
            Err(DecodeError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn compressing_twice_is_refused() {
        // A query can commit more than once, and index DDL appends after the
        // last commit — so a caller that compresses per commit hands the second
        // pass a payload that is already a frame. That produced a buffer
        // nothing could read: `BadOpcode(24021)`, because the reader inflates
        // once and then meets the first pass's length prefix where a record
        // should be. Compression now happens once, where the payload becomes a
        // command, and a second attempt is refused rather than silently
        // corrupting.
        let ids: Vec<u64> = (0..2_000).collect();
        let rows: Vec<Value> = (0..2_000).map(|i| Value::Int(i as i64)).collect();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &IdList::from(ids.as_slice()), &[7], &[0], &rows);

        assert!(maybe_compress(&mut buf, 16), "the first pass compresses");
        let once = buf.clone();
        assert!(!maybe_compress(&mut buf, 16), "the second must decline");
        assert_eq!(buf, once, "and must leave the payload untouched");
        assert_eq!(read_buffer(&buf).unwrap().len(), 1);
    }

    #[test]
    fn a_corrupt_frame_is_an_error_not_a_panic() {
        let ids: IdList = (0..2_000).collect();
        let rows: Vec<Value> = (0..2_000).map(|i| Value::Int(i as i64)).collect();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &ids, &[7], &[0], &rows);
        assert!(maybe_compress(&mut buf, 16));

        // Truncating anywhere inside the frame must error, never panic.
        for cut in 7..buf.len() {
            assert!(read_buffer(&buf[..cut]).is_err(), "cut at {cut}");
        }
        // And a frame that expands to the wrong size is caught by the length.
        let mut lied = buf.clone();
        lied[2] = lied[2].wrapping_add(1);
        assert!(matches!(
            read_buffer(&lied),
            Err(DecodeError::CompressedLengthMismatch { .. })
        ));
    }

    #[test]
    fn a_tiny_buffer_cannot_ask_the_replica_for_a_34gb_allocation() {
        // The whole path, not just `read_ids`: a DELETE_NODE whose Range block is
        // three bytes but whose count is u32::MAX. Materializing it would be a
        // ~34 GB `Vec<u64>`, and the Range encoding carries no bytes proportional
        // to the count for a length guard to weigh.
        let mut buf = new_buffer();
        write_delete_node(&mut buf, &IdList::from([0, 1, 2]), &[]);
        // Patch the count that follows the opcode: `header = u32 opcode · u32 count`.
        let count_at = 2 + 4;
        buf[count_at..count_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());

        assert_eq!(
            read_buffer(&buf),
            Err(DecodeError::TooManyIds {
                count: u64::from(u32::MAX),
                max: MAX_RECORD_IDS,
            })
        );
    }

    #[test]
    fn a_compression_bomb_is_capped_by_the_declared_length() {
        // A frame of zeros expands enormously. The declared plaintext length is
        // the allocation ceiling, so a frame that wants more than it says is
        // refused rather than inflated first and measured after.
        let plain = vec![0_u8; 4 << 20];
        let frame = zstd::stream::encode_all(plain.as_slice(), 1).unwrap();
        assert!(frame.len() < 4096, "frame should be tiny: {}", frame.len());

        let mut buf = vec![EFFECTS_VERSION, FLAG_COMPRESSED];
        // Declare a plaintext far smaller than the frame really expands to.
        buf.extend_from_slice(&64_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&frame);

        assert!(matches!(
            open_payload(&buf),
            Err(DecodeError::BadCompression(_))
        ));
    }

    #[test]
    fn only_a_create_constraint_carries_a_status() {
        // C sends no status on either constraint effect — see
        // `create_constraint_effect.c` and `drop_constraint_effect.c`, whose
        // documented formats are identical and list no such field. v3 adds one to
        // the *create*, deliberately: a replica does not validate, so it is the
        // only way it can learn whether the constraint enforces. A drop had no use
        // for it — the apply path calls `drop_constraint` and never reads it — so
        // carrying one was a `u32` of pure divergence.
        let spec = |status| ConstraintSpec {
            constraint_type: ConstraintType::Unique,
            entity_type: EntityType::Node,
            status,
            label_id: 3,
            label: "Person",
            props: &[(0, "email")],
        };

        let mut created = new_buffer();
        write_constraint(&mut created, true, &spec(Some(ConstraintStatus::Failed)));
        let mut dropped = new_buffer();
        write_constraint(&mut dropped, false, &spec(None));

        // Same fields either way apart from the status word.
        assert_eq!(
            created.len() - dropped.len(),
            4,
            "a create carries exactly one u32 more than a drop"
        );

        // And it round-trips as present on one and absent on the other, rather
        // than defaulting to something that reads as meaningful.
        let Record::Constraint { status, .. } = &read_buffer(&created).unwrap()[0] else {
            panic!("expected a constraint record");
        };
        assert_eq!(*status, Some(ConstraintStatus::Failed));
        let Record::Constraint { status, .. } = &read_buffer(&dropped).unwrap()[0] else {
            panic!("expected a constraint record");
        };
        assert_eq!(*status, None);
    }

    #[test]
    fn buffer_header_is_the_version() {
        assert_eq!(
            new_buffer(),
            vec![3_u8, 0],
            "version, then the reserved flags byte"
        );
    }
}
