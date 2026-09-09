//! The records themselves, and reading a whole payload.

use std::borrow::Cow;
use std::ops::Deref;

use crate::runtime::value::Value;

use super::*;

// ── records ──

/// Open a batchable record: its opcode, then how many entities it covers.
///
/// Every block that follows takes its length from this one count — it is never
/// repeated. `count == 1` and `count == 10_000` are the same record, which is
/// why no record type is left un-batchable and the decoder has one shape per
/// opcode.
fn write_header<W: EffectWrite + ?Sized>(
    buf: &mut W,
    opcode: Opcode,
    count: Option<u32>,
) {
    debug_assert_eq!(
        count.is_some(),
        opcode.is_batchable(),
        "{opcode:?} was given the wrong kind of header"
    );
    buf.u32(opcode as u32);
    if let Some(count) = count {
        buf.u32(count);
    }
}

/// `CREATE_INDEX`'s options, in the layout the RDB already uses.
///
/// Not a map. Every layer but the wire already holds these as typed fields —
/// C's `IndexField` has two nested option structs, C's v19 RDB writes them at
/// fixed positions, and this engine parses them into `IndexOptions` the moment
/// they arrive. The wire carried a `T_MAP` of stringly-typed keys only because
/// C's `create_index_effect.c` had an `SIValue` in hand at the call site and
/// passed the parser's representation straight through.
///
/// The layout here mirrors `_RdbLoadIndex`/`_RdbLoadIndexField`, which is a
/// format both engines already implement and cross-check: RDBs have been
/// mutually loadable since #2459, so this is a tested shape rather than a new
/// one. Reusing it means the far side can reuse its own RDB field logic, and
/// there is no new agreement to reach about mask bits or defaults.
///
/// **Each option carries a presence byte**, which is the one place this layout
/// departs from the RDB's. The RDB writes defaults for what the statement
/// omitted and can afford to: it saves a whole index at once, so "weight 1.0"
/// and "no weight given" produce the same index either way. An effect is an
/// instruction against an index that may already exist, and there the two are
/// different — `create_index` refuses an explicit language when one is already
/// set for the label. Materialising defaults here made `test_CRUD_replication`
/// diverge a replica with "Language is already set for label \'L\'", so absence
/// is information and travels as itself.
///
/// A byte per option rather than one mask for all of them, written and read
/// through [`put_opt`]/[`take_opt`] so a value cannot be written without its
/// flag or read without it. The text half is written whatever the field type,
/// with five zero bytes saying "nothing said"; only the vector half is gated,
/// because its `dimension` has no absent form.
///
/// `phonetic` is the algorithm code as a string, not a bool: C stores
/// `char *phonetic` and accepts `dm:fr`/`dm:pt`/`dm:es`, which this engine
/// currently rejects. A bool here would have baked one engine's narrowing into
/// the format.
///
/// The vector block is gated on `field_type & INDEX_FLD_VECTOR` rather than a
/// flag of its own — `field_type` is already on the wire ahead of this, and the
/// RDB gates on exactly the same bit.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct IndexOptions {
    pub language: Option<String>,
    pub stopwords: Option<Vec<String>>,
    pub weight: Option<f64>,
    pub nostem: Option<bool>,
    /// The algorithm code, e.g. `"dm:en"`. `None` when the statement said
    /// nothing; C stores `char *phonetic` and accepts codes this engine
    /// rejects, so a bool here would bake one engine's narrowing into the wire.
    pub phonetic: Option<String>,
    /// Present iff the statement's `field_type` carries `INDEX_FLD_VECTOR`.
    pub vector: Option<VectorOptions>,
}

/// The HNSW half, written only for a vector field.
///
/// `u64` throughout, matching C's `size_t` and its RDB writer. An absent field
/// is one the statement did not state, and the receiver supplies its own
/// default — the same ones the RDB writes: `M` 16, `ef_construction` 200,
/// `ef_runtime` 10, `sim_func` 0 (L2).
///
/// `sim_func` is the `VecSimMetric` discriminant — 0 = L2/euclidean, 1 = IP,
/// 2 = cosine. An enum rather than a name because the RDB already persists it
/// this way on both engines, which makes it shared format vocabulary like
/// `IndexFieldType` and `SIType` rather than an internal representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct VectorOptions {
    /// Not optional: a vector index cannot exist without one.
    pub dimension: u64,
    pub m: Option<u64>,
    pub ef_construction: Option<u64>,
    pub ef_runtime: Option<u64>,
    pub sim_func: Option<u64>,
}

impl IndexOptions {
    /// The block a statement with no `OPTIONS` clause carries: nothing said.
    ///
    /// Every field absent, which is **not** every field at its default. An
    /// effect adds a field to an index that may already exist, so "no language
    /// given" and "language english" are different instructions — the second is
    /// refused when a language is already set for the label. The RDB can
    /// materialise defaults because it writes a whole index at once; this wire
    /// cannot, and a flow test proved it: an explicit "english" diverged a
    /// replica with "Language is already set for label 'L'".
    #[must_use]
    pub fn none_given(vector: Option<VectorOptions>) -> Self {
        Self {
            vector,
            ..Self::default()
        }
    }
}

impl VectorOptions {
    /// A vector field of the given dimension, with nothing else stated.
    #[must_use]
    pub const fn of_dimension(dimension: u64) -> Self {
        Self {
            dimension,
            m: None,
            ef_construction: None,
            ef_runtime: None,
            sim_func: None,
        }
    }
}

/// One optional field: a presence byte, then the value if present.
///
/// A byte per field rather than one mask for all of them. A mask is smaller but
/// has to be kept in step with the field list by hand — set the bit, write the
/// value, in two places, in the same order, forever. These helpers make the
/// presence byte and the value one operation, so a field cannot be written
/// without its flag or read without it.
fn put_opt<T, W: EffectWrite + ?Sized>(
    buf: &mut W,
    v: Option<&T>,
    write: impl FnOnce(&mut W, &T),
) {
    match v {
        None => buf.u8(0),
        Some(x) => {
            buf.u8(1);
            write(buf, x);
        }
    }
}

fn take_opt<T>(
    r: &mut Reader<'_>,
    read: impl FnOnce(&mut Reader<'_>) -> Result<T, DecodeError>,
) -> Result<Option<T>, DecodeError> {
    match r.u8()? {
        0 => Ok(None),
        1 => Ok(Some(read(r)?)),
        other => Err(DecodeError::BadBool {
            value: u64::from(other),
        }),
    }
}

impl IndexOptions {
    /// Write the block, gated on the statement's `field_type`.
    ///
    /// `field_type` and not `self.vector.is_some()`, because that is what the
    /// reader has: the vector half's presence is a property of the statement,
    /// stated once on the wire ahead of this. Gating the two directions on
    /// different things desynchronises the stream — the writer emits `u64`s the
    /// reader never consumes and the next record is parsed from the middle of
    /// them.
    fn encode_with<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
        field_type: u32,
    ) {
        debug_assert_eq!(
            self.vector.is_some(),
            field_type & index_field_type::INDEX_FLD_VECTOR != 0,
            "vector options must be present exactly when the field type says so"
        );
        put_opt(buf, self.language.as_ref(), |b, s| b.string(s));
        put_opt(buf, self.stopwords.as_ref(), |b, sw| {
            b.u64(sw.len() as u64);
            for s in sw {
                b.string(s);
            }
        });
        put_opt(buf, self.weight.as_ref(), |b, w| b.f64(*w));
        put_opt(buf, self.nostem.as_ref(), |b, n| b.u8(u8::from(*n)));
        put_opt(buf, self.phonetic.as_ref(), |b, s| b.string(s));
        if field_type & index_field_type::INDEX_FLD_VECTOR != 0 {
            let v = self
                .vector
                .unwrap_or_else(|| VectorOptions::of_dimension(0));
            buf.u64(v.dimension);
            put_opt(buf, v.m.as_ref(), |b, x| b.u64(*x));
            put_opt(buf, v.ef_construction.as_ref(), |b, x| b.u64(*x));
            put_opt(buf, v.ef_runtime.as_ref(), |b, x| b.u64(*x));
            put_opt(buf, v.sim_func.as_ref(), |b, x| b.u64(*x));
        }
    }
}

impl EffectDecodeSized<3> for IndexOptions {
    /// The statement's `field_type`, which says whether a vector block follows.
    type Size = u32;

    fn decode_sized(
        r: &mut Reader<'_>,
        field_type: Self::Size,
    ) -> Result<Self, DecodeError> {
        let language = take_opt(r, |r| r.string())?;
        let stopwords = take_opt(r, |r| {
            let n = r.u64()?;
            // A stopword is at least an 8-byte length plus its NUL.
            let n = r.guard_count(n, 9)?;
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(r.string()?);
            }
            Ok(out)
        })?;
        let weight = take_opt(r, |r| r.f64())?;
        let nostem = take_opt(r, |r| match r.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(DecodeError::BadBool {
                value: u64::from(other),
            }),
        })?;
        let phonetic = take_opt(r, |r| r.string())?;
        let vector = if field_type & index_field_type::INDEX_FLD_VECTOR == 0 {
            None
        } else {
            Some(VectorOptions {
                dimension: r.u64()?,
                m: take_opt(r, |r| r.u64())?,
                ef_construction: take_opt(r, |r| r.u64())?,
                ef_runtime: take_opt(r, |r| r.u64())?,
                sim_func: take_opt(r, |r| r.u64())?,
            })
        };
        Ok(Self {
            language,
            stopwords,
            weight,
            nostem,
            phonetic,
            vector,
        })
    }
}

/// The smallest an `AttrRef` can encode to: a 2-byte attribute id, an 8-byte
/// name length, and the one byte that length can never go below.
///
/// The NUL is what makes it 11 and not 10. `EffectWrite::string` writes the
/// terminator unconditionally, so the shortest possible name — the empty one —
/// still costs a byte. Ten let `guard_count` accept a count a tenth higher than
/// the buffer could hold.
const MIN_ATTR_REF_BYTES: usize = 11;

/// One attribute, by id and name — the pair every schema-bearing record carries.
///
/// A struct rather than a `(u16, S)` tuple because the id sits next to other
/// small integers on the wire, and transposing two of those is precisely the
/// mistake that does not fail on the far side. It was a struct for index fields
/// and a bare tuple for constraint properties, which is one wire concept with
/// two spellings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttrRef<S> {
    pub id: u16,
    pub name: S,
}

/// Every field of one index statement: `u16 n` then `(id, name) x n`.
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
/// Generic over the field's string, so a `Record`'s owned `AttrRef<String>`
/// and a caller's borrowed `AttrRef<&str>` both go through it. The dispatch
/// used to collect the owned ones into a borrowed `Vec` per index record —
/// an allocation to satisfy a signature.
/// `IndexFields` — `u16 n` · `(u16 id, string name) × n`.
///
/// Generic over the field's string for the same reason [`AttrRef`] is: a
/// `Record` holds owned `AttrRef<String>` and a caller may hold borrowed
/// `AttrRef<&str>`, and neither should have to allocate to be written.
pub struct IndexFields<S>(pub S);

impl<T: AsRef<str>> EffectEncode<3> for IndexFields<&[AttrRef<T>]> {
    fn encode<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
    ) {
        let fields = self.0;
        // Floor: 2 bytes of id and an 8-byte length per field, the same minimum
        // the decode below guards the count against.
        buf.reserve(2 + fields.len() * MIN_ATTR_REF_BYTES);
        buf.u16(fields.len() as u16);
        for field in fields {
            buf.u16(field.id);
            buf.string(field.name.as_ref());
        }
    }
}

impl EffectDecode<3> for IndexFields<Vec<AttrRef<String>>> {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let n = r.u16()?;
        let n = r.guard_count(u64::from(n), MIN_ATTR_REF_BYTES)?;
        let mut fields = Vec::with_capacity(n);
        for _ in 0..n {
            fields.push(AttrRef {
                id: r.u16()?,
                name: r.string()?,
            });
        }
        Ok(Self(fields))
    }
}

/// `13 CREATE_CONSTRAINT` / `14 DROP_CONSTRAINT`.
///
/// The property count is a **`u8`**, not the `u16` used elsewhere — C's
/// `uint8_t n`. Each property travels as `(id, name)` for the same reason the
/// index records do.
/// A constraint as a record carries it.
///
/// Grouped for the same reason as [`AttrRef`]: the writer took eight
/// positional arguments, six of which described one constraint.
pub struct ConstraintSpec<'a> {
    pub constraint_type: ConstraintType,
    pub entity_type: EntityType,
    /// The primary's outcome, and `None` for a drop. See [`write_constraint`].
    pub status: Option<ConstraintStatus>,
    pub label_id: u32,
    pub label: &'a str,
    pub props: &'a [AttrRef<&'a str>],
}

/// One decoded record.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    Update {
        entity: EntityType,
        ids: IdList,
        /// The nodes' derived labels. Always empty for [`EntityType::Relationship`],
        /// which carries a [`Self::Update::relation_id`] in the same slot instead.
        labels: Vec<u32>,
        /// The edges' one relationship type. `None` for [`EntityType::Node`],
        /// whose membership is the label set above.
        relation_id: Option<u32>,
        attr_ids: Vec<u16>,
        rows: Vec<Value>,
    },
    CreateNode {
        ids: IdList,
        labels: Vec<u32>,
        attr_ids: Vec<u16>,
        rows: Vec<Value>,
    },
    CreateEdge {
        ids: IdList,
        relation_id: u32,
        src: IdList,
        dst: IdList,
        attr_ids: Vec<u16>,
        rows: Vec<Value>,
    },
    DeleteNode {
        ids: IdList,
        labels: Vec<u32>,
    },
    DeleteEdge {
        ids: IdList,
        relation_id: u32,
        src: IdList,
        dst: IdList,
    },
    Labels {
        add: bool,
        ids: IdList,
        labels: Vec<u32>,
    },
    AddSchema {
        schema_type: EntityType,
        id: u32,
        name: String,
    },
    AddAttribute {
        id: u16,
        name: String,
    },
    Index {
        create: bool,
        schema_type: EntityType,
        label_id: u32,
        label: String,
        /// C's index-field flags. A property of the statement, not of a field.
        field_type: u32,
        /// Every field of the statement. See [`write_index_fields`].
        fields: Vec<AttrRef<String>>,
        /// `None` on a drop, which carries no options.
        options: Option<IndexOptions>,
    },
    Constraint {
        create: bool,
        constraint_type: ConstraintType,
        entity_type: EntityType,
        /// The primary's outcome; `None` on a drop, which carries none.
        status: Option<ConstraintStatus>,
        label_id: u32,
        label: String,
        props: Vec<AttrRef<String>>,
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
    // A record covering no entities says nothing about any of them, so it is
    // not a legal record. Refused here, before any block is read, because
    // `count` sizes every block that follows — checking it once at the header
    // is what keeps a zero-length parse path out of each of them, rather than
    // requiring both engines to agree on all of those paths separately.
    //
    // The emitter cannot produce one: records come from grouping, and a group
    // only exists once an id has been pushed into it. So this is a statement
    // about what may arrive, not a case anything here emits.
    if opcode.is_batchable() && count == 0 {
        return Err(DecodeError::EmptyRecord {
            opcode: opcode as u32,
        });
    }
    let record = match opcode {
        Opcode::UpdateNode | Opcode::UpdateEdge => {
            let entity = if opcode == Opcode::UpdateNode {
                EntityType::Node
            } else {
                EntityType::Relationship
            };
            // One slot, read as whichever the opcode says it is. A decoder that
            // reads the wrong one here does not fail — it consumes the next
            // block's first bytes as a count and misaligns everything after it,
            // which is the class of bug the fixtures exist to catch.
            let (labels, relation_id) = if entity == EntityType::Node {
                (LabelSet::decode(r)?.0, None)
            } else {
                (Vec::new(), Some(RelType::decode(r)?.0))
            };
            let attr_ids = AttrIds::decode(r)?.0;
            let ids = IdList::decode_sized(r, count)?;
            let rows = AttrValues::decode_sized(r, (count, attr_ids.len()))?.0;
            Record::Update {
                entity,
                ids,
                labels,
                relation_id,
                attr_ids,
                rows,
            }
        }
        Opcode::CreateNode => {
            let labels = LabelSet::decode(r)?.0;
            let attr_ids = AttrIds::decode(r)?.0;
            let ids = IdList::decode_sized(r, count)?;
            let rows = AttrValues::decode_sized(r, (count, attr_ids.len()))?.0;
            Record::CreateNode {
                ids,
                labels,
                attr_ids,
                rows,
            }
        }
        Opcode::CreateEdge => {
            let relation_id = RelType::decode(r)?.0;
            let attr_ids = AttrIds::decode(r)?.0;
            let ids = IdList::decode_sized(r, count)?;
            let src = IdList::decode_sized(r, count)?;
            let dst = IdList::decode_sized(r, count)?;
            let rows = AttrValues::decode_sized(r, (count, attr_ids.len()))?.0;
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
            let labels = LabelSet::decode(r)?.0;
            Record::DeleteNode {
                labels,
                ids: IdList::decode_sized(r, count)?,
            }
        }
        Opcode::DeleteEdge => {
            let relation_id = RelType::decode(r)?.0;
            let ids = IdList::decode_sized(r, count)?;
            let src = IdList::decode_sized(r, count)?;
            let dst = IdList::decode_sized(r, count)?;
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
            let labels = LabelSet::decode(r)?.0;
            Record::Labels {
                add: opcode == Opcode::SetLabels,
                ids: IdList::decode_sized(r, count)?,
                labels,
            }
        }
        Opcode::AddSchema => {
            let schema_type = entity_from_schema_tag(r.u32()?)?;
            let id = r.u32()?;
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
            let label_id = r.u32()?;
            let label = r.string()?;
            let field_type = r.u32()?;
            let fields = IndexFields::decode(r)?.0;
            let options = if create {
                Some(IndexOptions::decode_sized(r, field_type)?)
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
            let label_id = r.u32()?;
            let label = r.string()?;
            let n = r.u8()?;
            let n = r.guard_count(u64::from(n), MIN_ATTR_REF_BYTES)?;
            let mut props = Vec::with_capacity(n);
            for _ in 0..n {
                let attr_id = r.u16()?;
                props.push(AttrRef {
                    id: attr_id,
                    name: r.string()?,
                });
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
///
/// Each arm *is* the record's layout — there is no writer beside it taking the
/// same fields positionally. That split cost more than a hop: adding
/// `UPDATE_EDGE`'s `RelType` meant editing the variant, a seven-argument
/// parameter list and this dispatch, three things that had to agree with
/// nothing checking that they did. Two adjacent `&[u32]` arguments transpose
/// silently; two named fields do not.
impl EffectEncode<3> for Record {
    fn encode<W: EffectWrite + ?Sized>(
        &self,
        buf: &mut W,
    ) {
        match self {
            // `9 ADD_SCHEMA` — `SchemaType · LabelID|RelationID · name`.
            //
            // Carries its id, which v2 did not: the replica used to infer one
            // from append order, so a dictionary of a different length assigned
            // a different id and every later record referenced the wrong entry.
            Record::AddSchema {
                schema_type,
                id,
                name,
            } => {
                write_header(buf, Opcode::AddSchema, None);
                buf.u32(schema_tag(*schema_type));
                buf.schema_id(*id);
                buf.string(name);
            }

            // `10 ADD_ATTRIBUTE` — `AttributeID · name`. Same reason as above.
            Record::AddAttribute { id, name } => {
                write_header(buf, Opcode::AddAttribute, None);
                buf.u16(*id);
                buf.string(name);
            }

            // `3 CREATE_NODE` — `count · LabelSet · AttrIds · IdList · AttrValues`.
            //
            // `rows` is `ids.len() × attr_ids.len()` values in row-major order;
            // row *k* belongs to the k-th id in `ids`, as written.
            Record::CreateNode {
                ids,
                labels,
                attr_ids,
                rows,
            } => {
                write_header(buf, Opcode::CreateNode, Some(ids.count()));
                LabelSet(labels.as_slice()).encode(buf);
                AttrIds(attr_ids.as_slice()).encode(buf);
                ids.encode(buf);
                AttrValues(rows.as_slice()).encode(buf);
            }

            // `4 CREATE_EDGE` — `count · RelType · AttrIds · IdList · IdList(src)
            // · IdList(dst) · AttrValues`.
            //
            // Endpoints repeat — many edges share a source — and each must stay
            // positionally aligned with its edge id, which is why nothing here
            // may be re-sorted into a set.
            Record::CreateEdge {
                ids,
                relation_id,
                src,
                dst,
                attr_ids,
                rows,
            } => {
                debug_assert_eq!(ids.len(), src.len(), "one source per edge");
                debug_assert_eq!(ids.len(), dst.len(), "one destination per edge");
                write_header(buf, Opcode::CreateEdge, Some(ids.count()));
                RelType(*relation_id).encode(buf);
                AttrIds(attr_ids.as_slice()).encode(buf);
                ids.encode(buf);
                src.encode(buf);
                dst.encode(buf);
                AttrValues(rows.as_slice()).encode(buf);
            }

            // `1 UPDATE_NODE` — `count · LabelSet · AttrIds · IdList · AttrValues`.
            // `2 UPDATE_EDGE` — `count · RelType  · AttrIds · IdList · AttrValues`.
            //
            // Each form carries the entity's schema membership in the same slot:
            // a node's `LabelSet`, an edge's `RelType`. Neither is an
            // instruction — the replica indexes under what the record states
            // rather than re-deriving it, which is what makes its index hold
            // what the primary's holds instead of agreeing with its own
            // possibly-drifted matrices.
            //
            // C's `EFFECT_UPDATE_EDGE` has always carried the relation id, and
            // `update_edge_effect.c` refuses a record naming a type it does not
            // have. Not `src`/`dst`, which C also sends: those are per *edge*
            // rather than per record, so they would cost two more `IdList`s
            // where the type costs four bytes.
            //
            // A property being removed is `T_NULL` in its slot — the tag's only
            // meaning here.
            Record::Update {
                entity,
                ids,
                labels,
                relation_id,
                attr_ids,
                rows,
            } => {
                write_header(buf, update_opcode(*entity), Some(ids.count()));
                match entity {
                    EntityType::Node => LabelSet(labels.as_slice()).encode(buf),
                    EntityType::Relationship => {
                        RelType(relation_id.expect("UPDATE_EDGE carries its relationship type"))
                            .encode(buf)
                    }
                }
                AttrIds(attr_ids.as_slice()).encode(buf);
                ids.encode(buf);
                AttrValues(rows.as_slice()).encode(buf);
            }

            // `7 SET_LABELS` / `8 REMOVE_LABELS` — `count · LabelSet · IdList`.
            //
            // The labels, and all their nodes — not one `(node, label)` pair per
            // node. v2 shipped a serialized GraphBLAS vector here, which is an
            // internal representation two engines cannot agree on.
            Record::Labels { add, ids, labels } => {
                let opcode = if *add {
                    Opcode::SetLabels
                } else {
                    Opcode::RemoveLabels
                };
                write_header(buf, opcode, Some(ids.count()));
                LabelSet(labels.as_slice()).encode(buf);
                ids.encode(buf);
            }

            // `5 DELETE_NODE` — `count · LabelSet · IdList`.
            //
            // The labels are the node's **actual** labels, captured when it was
            // deleted — not the ones the query's pattern named. `MATCH (n:A)
            // DELETE n` over an `(:A:B)` node must clear `:B`'s indexes too, and
            // the pattern cannot say so.
            Record::DeleteNode { ids, labels } => {
                write_header(buf, Opcode::DeleteNode, Some(ids.count()));
                LabelSet(labels.as_slice()).encode(buf);
                ids.encode(buf);
            }

            // `6 DELETE_EDGE` — `count · RelType · IdList · IdList(src) · IdList(dst)`.
            //
            // C needs the endpoints to locate the edge in its adjacency
            // matrices, and unlike an update they cannot be recovered
            // afterwards: the edge is gone by the time effects are built.
            Record::DeleteEdge {
                ids,
                relation_id,
                src,
                dst,
            } => {
                debug_assert_eq!(ids.len(), src.len(), "one source per edge");
                debug_assert_eq!(ids.len(), dst.len(), "one destination per edge");
                write_header(buf, Opcode::DeleteEdge, Some(ids.count()));
                RelType(*relation_id).encode(buf);
                ids.encode(buf);
                src.encode(buf);
                dst.encode(buf);
            }

            // `11 CREATE_INDEX` / `12 DROP_INDEX`.
            //
            // `field_type` is a bit set, not a discriminant — range and fulltext
            // at once is one statement — so it travels as the union C stores.
            // Only a create carries `OPTIONS`: v2 could not encode the map at
            // all and forced the whole statement to replicate as a verbatim
            // query.
            Record::Index {
                create,
                schema_type,
                label_id,
                label,
                field_type,
                fields,
                options,
            } => {
                write_header(
                    buf,
                    if *create {
                        Opcode::CreateIndex
                    } else {
                        Opcode::DropIndex
                    },
                    None,
                );
                buf.u32(schema_tag(*schema_type));
                buf.schema_id(*label_id);
                buf.string(label);
                buf.u32(*field_type);
                IndexFields(fields.as_slice()).encode(buf);
                if *create {
                    // A create always carries options; absent ones travel as
                    // the RDB's defaults rather than as a presence mask.
                    options
                        .as_ref()
                        .expect("a CREATE_INDEX record carries options")
                        .encode_with(buf, *field_type);
                }
            }

            // `13 CREATE_CONSTRAINT` / `14 DROP_CONSTRAINT`.
            Record::Constraint {
                create,
                constraint_type,
                entity_type,
                status,
                label_id,
                label,
                props,
            } => {
                write_header(
                    buf,
                    if *create {
                        Opcode::CreateConstraint
                    } else {
                        Opcode::DropConstraint
                    },
                    None,
                );
                debug_assert_eq!(
                    *create,
                    status.is_some(),
                    "status belongs to a create record and only to a create record"
                );
                buf.u32(constraint_tag(*constraint_type));
                buf.u32(entity_tag(*entity_type));
                // Create only, and the one place v3 deliberately carries more
                // than C: C's `EffectsBuffer_AddCreateConstraintEffect` sends no
                // status, so its replica cannot tell an enforcing constraint
                // from one still being built. A replica does not validate, so
                // this is the only thing that can tell it — and it is what makes
                // the second announcement, after validation finishes, converge
                // rather than duplicate.
                //
                // A *drop* has no such need: the apply path calls
                // `drop_constraint` and never reads the field.
                if let Some(status) = status {
                    buf.u32(constraint_status_tag(*status));
                }
                buf.schema_id(*label_id);
                buf.string(label);
                // Floor: 2 bytes of id and an 8-byte length per property.
                buf.reserve(1 + props.len() * 10);
                // Not `as u8`, and not a `debug_assert`. C reads this count as a
                // `uint8`, so a 256th property would write **0** and replicate a
                // constraint over no properties at all — silently, and only in
                // release, where an assert is compiled out. `GRAPH.CONSTRAINT`
                // caps the count at 255 so this cannot fire, which is exactly
                // why it must be loud rather than truncating: if it ever does,
                // the guard three layers up has gone.
                let n = u8::try_from(props.len())
                    .expect("GRAPH.CONSTRAINT caps properties at 255; C reads the count as uint8");
                buf.u8(n);
                for AttrRef { id, name } in props {
                    buf.u16(*id);
                    buf.string(name);
                }
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
/// is borrowed straight from the caller's buffer. Both cases are real and
/// neither can be dropped: a compressed frame has to be inflated whole before
/// any record can be read, so that plaintext must outlive the records
/// borrowing from it, while an uncompressed payload is already the plaintext
/// and copying it would be pure waste on the common path.
///
/// Which is `Cow`, so this is a newtype over one rather than the same two
/// variants written out again. The wrapper keeps the domain name and gives
/// `records` somewhere to live; `Deref` below means nothing else has to match
/// on which case it holds.
pub struct Payload<'a>(Cow<'a, [u8]>);

impl Deref for Payload<'_> {
    type Target = [u8];

    /// The plaintext, whichever case it came from — which is what every reader
    /// of a payload actually wants.
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl Payload<'_> {
    /// The record stream, decoded one at a time.
    #[must_use]
    pub fn records(&self) -> Records<'_> {
        Records {
            failed: false,
            r: Reader::new(&self.0),
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
        return Ok(Payload(Cow::Borrowed(r.rest())));
    }

    // A frame has to be whole before any of it can be read, so this is the one
    // place the payload cannot be streamed.
    let plain_len = r.u32()? as usize;
    let comp_len = r.u32()? as usize;
    let checksum = r.u32()?;
    // Exactly `comp_len`, not "the rest": the frame states its own length, so a
    // payload is self-delimiting and a reader can bound the decompress input
    // before handing it to zstd rather than trusting where the buffer happens
    // to end. `take` refuses a length the buffer cannot satisfy.
    let frame = r.take(comp_len)?;
    // Trailing bytes mean the header and the payload disagree about where the
    // frame ends, and nothing after it can be trusted either.
    if !r.is_empty() {
        return Err(DecodeError::TrailingBytes {
            after_frame: r.remaining(),
        });
    }
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
    Ok(Payload(Cow::Owned(plain)))
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

/// Bytes a compressed payload adds before the frame: `u32 plain_len`,
/// `u32 comp_len`, `u32 checksum`. Compression has to save more than this to be
/// worth doing.
pub const COMPRESSED_PREFIX: usize = 12;

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
    // The two declared lengths and the checksum ride along, so they count.
    if frame.len() + COMPRESSED_PREFIX >= buf.len() - HEADER {
        return false;
    }

    let plain_len = (buf.len() - HEADER) as u32;
    let comp_len = frame.len() as u32;
    let checksum = crc32fast::hash(&buf[HEADER..]);
    buf.truncate(HEADER);
    buf[1] |= FLAG_COMPRESSED;
    buf.bytes(&plain_len.to_le_bytes());
    buf.bytes(&comp_len.to_le_bytes());
    buf.bytes(&checksum.to_le_bytes());
    buf.bytes(&frame);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Decode a whole payload back into its records.
    ///
    /// Here rather than in a shared test helper because it is the only part of
    /// that helper with no graph in it — the rest builds graphs and drives the
    /// emitter, neither of which this module knows about.
    fn read_buffer(buf: &[u8]) -> Result<Vec<Record>, DecodeError> {
        open_payload(buf)?.records().collect()
    }

    // ── records ──

    #[test]
    fn create_node_pins_its_bytes_at_count_one() {
        // One node, label 7, one int property. The smallest possible record —
        // and the order is the invariant: schema first, data last.
        let mut buf = new_buffer();
        Record::CreateNode {
            ids: IdList::from([0]),
            labels: vec![7],
            attr_ids: vec![0],
            rows: vec![Value::Int(1)],
        }
        .encode(&mut buf);
        assert_eq!(
            format!("{buf:02x?}"),
            concat!(
                "[03, 00, ",                                       // version, flags
                "03, 00, 00, 00, 01, 00, 00, 00, ",                // CREATE_NODE, count = 1
                "01, 00, 07, 00, 00, 00, ",                        // LabelSet: n = 1, label 7
                "01, 00, 00, 00, ",                                // AttrIds: n = 1, attr 0
                "01, 00, 00, 00, 00, 00, 01, ", // IdList: 1 segment; range, base 0, len 1
                "00, 20, 00, 00, 01, 00, 00, 00, 00, 00, 00, 00]"  // AttrValues: T_INT64, 1
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
        Record::CreateNode {
            ids: ids.clone(),
            labels: vec![7],
            attr_ids: vec![0],
            rows: rows.to_vec(),
        }
        .encode(&mut buf);

        assert_eq!(&buf[2..6], &(Opcode::CreateNode as u32).to_le_bytes());
        assert_eq!(&buf[6..10], &10_000_u32.to_le_bytes());
        // Schema first: header ends at 10, LabelSet (6 B) then AttrIds (4 B),
        // so the id block starts at 20. Sequentially allocated ids are a
        // consecutive run, so all ten thousand of them are three bytes.
        // One segment: count, header byte carrying both widths, then the base
        // and the length. Flat in the id count — a million ids cost what ten do.
        assert_eq!(&buf[20..28], &[1, 0, 0, 0, 0x10, 0, 0x10, 0x27]);

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
    fn a_descending_endpoint_column_survives_a_record() {
        // The shape the descending segments exist for. `src` and `dst` follow
        // *edge* id order, so an endpoint column written by a downward scan
        // descends while the edge ids ascend — three directions in one record,
        // and every row has to come back on the entity it went out with.
        let ids = IdList::from([100_u64, 101, 102, 103]);
        let src = IdList::from([50_u64, 49, 48, 47]);
        let dst = IdList::from([9_u64, 9, 9, 9]);
        let mut buf = new_buffer();
        Record::CreateEdge {
            ids: ids.clone(),
            relation_id: 2,
            src: src.clone(),
            dst: dst.clone(),
            attr_ids: vec![0],
            rows: vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)],
        }
        .encode(&mut buf);

        let records = read_buffer(&buf).unwrap();
        let Record::CreateEdge {
            ids: gi,
            src: gs,
            dst: gd,
            rows,
            ..
        } = &records[0]
        else {
            panic!("wrong record: {:?}", records[0]);
        };
        assert_eq!(gi.iter().collect::<Vec<_>>(), vec![100, 101, 102, 103]);
        assert_eq!(
            gs.iter().collect::<Vec<_>>(),
            vec![50, 49, 48, 47],
            "the descending column must come back descending, not sorted"
        );
        assert_eq!(gd.iter().collect::<Vec<_>>(), vec![9, 9, 9, 9]);
        assert_eq!(rows[0], Value::Int(1), "row 0 still belongs to edge 100");
        assert_eq!(rows[3], Value::Int(4), "row 3 still belongs to edge 103");
    }

    #[test]
    fn create_edge_keeps_endpoints_aligned_with_their_edges() {
        // Three edges out of one source: an IdSet would have collapsed the
        // sources to one entry and misaligned every row after the first.
        let ids = IdList::from([10_u64, 11, 12]);
        let src = IdList::from([7_u64, 7, 7]);
        let dst = IdList::from([1_u64, 2, 3]);
        let mut buf = new_buffer();
        Record::CreateEdge {
            ids: ids.clone(),
            relation_id: 4,
            src: src.clone(),
            dst: dst.clone(),
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut buf);

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
        Record::Labels {
            add: true,
            ids: ids.clone(),
            labels: vec![5],
        }
        .encode(&mut buf);
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
        Record::AddSchema {
            schema_type: EntityType::Node,
            id: 3,
            name: "Person".to_owned(),
        }
        .encode(&mut buf);
        Record::AddAttribute {
            id: 9,
            name: "name".to_owned(),
        }
        .encode(&mut buf);

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
        Record::AddAttribute {
            id: 9,
            name: "n".to_owned(),
        }
        .encode(&mut buf);
        // opcode, then the id straight away — no 4-byte count in between.
        assert_eq!(&buf[..4], &(Opcode::AddAttribute as u32).to_le_bytes());
        assert_eq!(&buf[4..6], &9_u16.to_le_bytes());
    }

    #[test]
    fn every_record_type_round_trips() {
        let mut buf = new_buffer();
        Record::AddSchema {
            schema_type: EntityType::Relationship,
            id: 1,
            name: "KNOWS".to_owned(),
        }
        .encode(&mut buf);
        Record::AddAttribute {
            id: 0,
            name: "since".to_owned(),
        }
        .encode(&mut buf);
        Record::CreateNode {
            ids: IdList::from([1, 2]),
            labels: vec![7],
            attr_ids: vec![0],
            rows: vec![Value::Int(1), Value::Null],
        }
        .encode(&mut buf);
        Record::CreateEdge {
            ids: IdList::from([5, 6]),
            relation_id: 1,
            src: IdList::from([1, 1]),
            dst: IdList::from([2, 2]),
            attr_ids: vec![0],
            rows: vec![Value::Int(2020), Value::Int(2021)],
        }
        .encode(&mut buf);
        Record::Update {
            entity: EntityType::Node,
            ids: IdList::from([1]),
            labels: vec![7],
            relation_id: None,
            attr_ids: vec![0],
            rows: vec![Value::Int(9)],
        }
        .encode(&mut buf);
        Record::Update {
            entity: EntityType::Relationship,
            ids: IdList::from([5]),
            labels: vec![],
            relation_id: Some(1),
            attr_ids: vec![0],
            rows: vec![Value::Null],
        }
        .encode(&mut buf);
        Record::Labels {
            add: true,
            ids: IdList::from([1, 2]),
            labels: vec![7, 8],
        }
        .encode(&mut buf);
        Record::Labels {
            add: false,
            ids: IdList::from([1]),
            labels: vec![8],
        }
        .encode(&mut buf);
        Record::DeleteEdge {
            ids: IdList::from([5, 6]),
            relation_id: 1,
            src: IdList::from([1, 1]),
            dst: IdList::from([2, 2]),
        }
        .encode(&mut buf);
        Record::DeleteNode {
            ids: IdList::from([1, 2]),
            labels: vec![7],
        }
        .encode(&mut buf);
        Record::Index {
            create: true,
            schema_type: EntityType::Node,
            label_id: 7,
            label: "L".to_owned(),
            field_type: INDEX_FLD_RANGE,
            fields: vec![AttrRef {
                id: 0,
                name: "since".to_owned(),
            }],
            options: Some(IndexOptions::none_given(None)),
        }
        .encode(&mut buf);
        Record::Index {
            create: false,
            schema_type: EntityType::Node,
            label_id: 7,
            label: "L".to_owned(),
            field_type: INDEX_FLD_RANGE,
            fields: vec![AttrRef {
                id: 0,
                name: "since".to_owned(),
            }],
            options: None,
        }
        .encode(&mut buf);
        Record::Constraint {
            create: true,
            constraint_type: ConstraintType::Unique,
            entity_type: EntityType::Node,
            status: Some(ConstraintStatus::Operational),
            label_id: 7,
            label: "L".to_owned(),
            props: vec![AttrRef {
                id: 0,
                name: "since".to_owned(),
            }],
        }
        .encode(&mut buf);
        Record::Constraint {
            create: false,
            constraint_type: ConstraintType::Mandatory,
            entity_type: EntityType::Relationship,
            status: None,
            label_id: 1,
            label: "KNOWS".to_owned(),
            props: vec![],
        }
        .encode(&mut buf);

        let records = read_buffer(&buf).unwrap();
        assert_eq!(records.len(), 14, "one command, fourteen records");

        // Re-encoding what was decoded must reproduce the buffer exactly.
        //
        // One line, now that a record encodes itself. This used to restate the
        // whole match — fourteen arms unpacking a variant only to hand its
        // fields back to a writer positionally — which is exactly the
        // duplication that made adding `UPDATE_EDGE`'s `RelType` a three-place
        // edit.
        let mut again = new_buffer();
        for record in &records {
            record.encode(&mut again);
        }
        assert_eq!(again, buf, "encode(decode(x)) must equal x");
    }

    /// One `CREATE_NODE` record per distinct shape, for 10,000 nodes split
    /// `shapes` ways.
    fn create_10k_in_shapes(shapes: usize) -> Vec<u8> {
        let per = 10_000 / shapes;
        let mut buf = new_buffer();
        for s in 0..shapes {
            let ids: IdList = (0..per).map(|j| (j * shapes + s) as u64).collect();
            let rows: Vec<Value> = (0..per).map(|j| Value::Int(j as i64)).collect();
            Record::CreateNode {
                ids: ids.clone(),
                labels: vec![7],
                attr_ids: vec![s as u16],
                rows: rows.to_vec(),
            }
            .encode(&mut buf);
        }
        buf
    }

    #[test]
    fn grouping_makes_one_record_per_shape() {
        // `UNWIND range(0,9999) AS i CREATE (:L {v:i})` and its neighbours.
        // What this asserts is the *grouping*: how many records 10,000 nodes
        // become. The byte counts it used to print are a measurement and live
        // in the benches, where a number can be compared against a baseline
        // instead of scrolling past in test output.
        for shapes in [1_usize, 20, 500, 2_000, 10_000] {
            let buf = create_10k_in_shapes(shapes);
            assert_eq!(
                read_buffer(&buf).unwrap().len(),
                shapes,
                "one record per shape, at {shapes} shapes"
            );
        }

        // The singleton floor, as a bound rather than a reading. A segment list
        // states both its segment count and every segment's length, so that it
        // is well-formed on its own rather than only inside the record carrying
        // it — five bytes per single-id record, four for the count and one for
        // the length. This catches that floor regressing; it does not report it.
        let worst = create_10k_in_shapes(10_000);
        assert!(
            worst.len() <= 380_001,
            "singleton floor regressed: {}",
            worst.len()
        );

        // Two shapes with interleaved ids — the alternating-label query. Stride
        // 2 breaks the single run, so each id list falls back to a bitmap, and
        // the point is still that it is two records and not 10,000.
        let even: IdList = (0..5_000).map(|i| i * 2).collect();
        let odd: IdList = (0..5_000).map(|i| i * 2 + 1).collect();
        let half: Vec<Value> = (0..5_000).map(|i| Value::Int(i as i64)).collect();
        let mut alt = new_buffer();
        Record::CreateNode {
            ids: even,
            labels: vec![7],
            attr_ids: vec![0],
            rows: half.to_vec(),
        }
        .encode(&mut alt);
        Record::CreateNode {
            ids: odd,
            labels: vec![8],
            attr_ids: vec![0],
            rows: half,
        }
        .encode(&mut alt);
        assert_eq!(read_buffer(&alt).unwrap().len(), 2);

        // 10,000 edges out of one supernode: one record, because the source
        // column is a single `Repeat`.
        let mut edges = new_buffer();
        Record::CreateEdge {
            ids: (0..10_000).collect(),
            relation_id: 1,
            src: std::iter::repeat_n(4_000_000_000_u64, 10_000).collect(),
            dst: (0..10_000).collect(),
            attr_ids: vec![],
            rows: vec![],
        }
        .encode(&mut edges);
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
        Record::CreateNode {
            ids: IdList::from([0]),
            labels: vec![7],
            attr_ids: vec![0],
            rows: vec![Value::Int(1)],
        }
        .encode(&mut small);
        assert_eq!(small.len(), 37, "id 0 narrows to one byte");

        let mut mid = Vec::new();
        Record::CreateNode {
            ids: IdList::from([5_000]),
            labels: vec![7],
            attr_ids: vec![0],
            rows: vec![Value::Int(1)],
        }
        .encode(&mut mid);
        assert_eq!(mid.len(), 38, "a mid-size graph's id takes two");

        let mut large = Vec::new();
        Record::CreateNode {
            ids: IdList::from([5_000_000]),
            labels: vec![7],
            attr_ids: vec![0],
            rows: vec![Value::Int(1)],
        }
        .encode(&mut large);
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
        Record::Index {
            create: true,
            schema_type: EntityType::Node,
            label_id: 3,
            label: "Person".to_owned(),
            field_type: INDEX_FLD_RANGE,
            fields: vec![AttrRef {
                id: 9,
                name: "name".to_owned(),
            }],
            options: Some(IndexOptions::none_given(None)),
        }
        .encode(&mut buf);
        let records = read_buffer(&buf).unwrap();
        assert_eq!(
            records[0],
            Record::Index {
                create: true,
                schema_type: EntityType::Node,
                label_id: 3,
                label: "Person".into(),
                field_type: INDEX_FLD_RANGE,
                fields: vec![AttrRef {
                    id: 9,
                    name: "name".to_owned(),
                }],
                options: Some(IndexOptions::none_given(None)),
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
        let fields: Vec<AttrRef<String>> = [(0_u16, "a"), (1, "b"), (2, "c")]
            .into_iter()
            .map(|(id, name)| AttrRef {
                id,
                name: name.to_owned(),
            })
            .collect();
        Record::Index {
            create: true,
            schema_type: EntityType::Node,
            label_id: 1,
            label: "L".to_owned(),
            field_type: INDEX_FLD_RANGE,
            fields,
            options: Some(IndexOptions::none_given(None)),
        }
        .encode(&mut buf);

        let records = read_buffer(&buf).unwrap();
        assert_eq!(records.len(), 1, "one statement, one record");
        let Record::Index { fields, .. } = &records[0] else {
            panic!("wrong record: {:?}", records[0]);
        };
        assert_eq!(
            fields
                .iter()
                .map(|f| (f.id, f.name.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, "a"), (1, "b"), (2, "c")]
        );
    }

    /// The presence bytes are positional, and a round trip cannot see their
    /// order.
    ///
    /// `put_opt` and `take_opt` walk the same field list in the same order, so
    /// swapping two fields in both leaves every round-trip test green — the
    /// assertion is symmetric in the bug. The corpus pins the all-absent shape
    /// (`rec_create_index`) and the all-stated one (`rec_create_index_vector`);
    /// neither shows a flag that has drifted one field along, because in both
    /// every flag holds the same value. A partial block is where that shows,
    /// and the C engine needs these positions fixed.
    #[test]
    fn a_partly_stated_options_block_pins_where_each_flag_sits() {
        let mut opts = IndexOptions::none_given(None);
        opts.weight = Some(2.0);

        let mut buf = Vec::new();
        opts.encode_with(&mut buf, INDEX_FLD_FULLTEXT);
        assert_eq!(
            format!("{buf:02x?}"),
            concat!(
                "[00, ",                                // language: nothing said
                "00, ",                                 // stopwords
                "01, 00, 00, 00, 00, 00, 00, 00, 40, ", // weight: 2.0, f64 LE
                "00, ",                                 // nostem
                "00]"                                   // phonetic
            ),
            "one byte per option, in this order, and only the stated one \
             carries a value"
        );

        let mut r = Reader::new(&buf);
        let back = IndexOptions::decode_sized(&mut r, INDEX_FLD_FULLTEXT).unwrap();
        assert!(r.is_empty(), "the block is exactly as long as it says");
        assert_eq!(back, opts, "what was stated comes back stated, and only it");
    }

    /// The vector half: `dimension` bare, the rest flagged.
    #[test]
    fn a_vector_dimension_carries_no_flag_because_it_cannot_be_absent() {
        let mut v = VectorOptions::of_dimension(4);
        v.sim_func = Some(2);
        let opts = IndexOptions::none_given(Some(v));

        let mut buf = Vec::new();
        opts.encode_with(&mut buf, INDEX_FLD_VECTOR);
        assert_eq!(
            format!("{buf:02x?}"),
            concat!(
                "[00, 00, 00, 00, 00, ",               // the text half, all absent
                "04, 00, 00, 00, 00, 00, 00, 00, ",    // dimension 4, u64 LE, no flag
                "00, ",                                // M
                "00, ",                                // efConstruction
                "00, ",                                // efRuntime
                "01, 02, 00, 00, 00, 00, 00, 00, 00]"  // simFunc: cosine
            ),
            "the text half is written whatever the field type, so there is one \
             gate rather than two"
        );

        let mut r = Reader::new(&buf);
        let back = IndexOptions::decode_sized(&mut r, INDEX_FLD_VECTOR).unwrap();
        assert!(r.is_empty());
        assert_eq!(back, opts);
    }

    #[test]
    fn drop_index_carries_no_options() {
        let mut buf = new_buffer();
        Record::Index {
            create: false,
            schema_type: EntityType::Relationship,
            label_id: 1,
            label: "KNOWS".to_owned(),
            field_type: INDEX_FLD_RANGE,
            fields: vec![AttrRef {
                id: 0,
                name: "since".to_owned(),
            }],
            options: None,
        }
        .encode(&mut buf);
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
        Record::Constraint {
            create: true,
            constraint_type: ConstraintType::Unique,
            entity_type: EntityType::Node,
            status: Some(ConstraintStatus::Operational),
            label_id: 3,
            label: "Person".to_owned(),
            props: vec![
                AttrRef {
                    id: 0,
                    name: "first".to_owned(),
                },
                AttrRef {
                    id: 1,
                    name: "last".to_owned(),
                },
            ],
        }
        .encode(&mut buf);
        // opcode, ct, et, status, label_id, then the string, then a single
        // count byte.
        let after_label = 4 + 4 + 4 + 4 + 4 + (8 + "Person".len() + 1);
        assert_eq!(buf[after_label], 2, "the count is one byte wide");
    }

    #[test]
    fn constraints_round_trip_with_their_property_names() {
        let mut buf = new_buffer();
        Record::Constraint {
            create: true,
            constraint_type: ConstraintType::Unique,
            entity_type: EntityType::Node,
            status: Some(ConstraintStatus::Operational),
            label_id: 3,
            label: "Person".to_owned(),
            props: vec![
                AttrRef {
                    id: 0,
                    name: "first".to_owned(),
                },
                AttrRef {
                    id: 1,
                    name: "last".to_owned(),
                },
            ],
        }
        .encode(&mut buf);
        Record::Constraint {
            create: false,
            constraint_type: ConstraintType::Mandatory,
            entity_type: EntityType::Relationship,
            status: None,
            label_id: 1,
            label: "KNOWS".to_owned(),
            props: vec![AttrRef {
                id: 2,
                name: "since".to_owned(),
            }],
        }
        .encode(&mut buf);

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
                props: vec![
                    AttrRef {
                        id: 0,
                        name: "first".to_owned()
                    },
                    AttrRef {
                        id: 1,
                        name: "last".to_owned()
                    }
                ],
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
        assert_eq!(
            props,
            &vec![AttrRef {
                id: 2,
                name: "since".to_owned().to_string()
            }]
        );
    }

    #[test]
    fn ddl_records_carry_no_count() {
        // Like the schema records, these are singular.
        let mut buf = Vec::new();
        Record::Index {
            create: false,
            schema_type: EntityType::Node,
            label_id: 1,
            label: "L".to_owned(),
            field_type: INDEX_FLD_RANGE,
            fields: vec![AttrRef {
                id: 0,
                name: "a".to_owned(),
            }],
            options: None,
        }
        .encode(&mut buf);
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
        Record::DeleteNode {
            ids: IdList::from([1]),
            labels: vec![7],
        }
        .encode(&mut buf);
        buf[0] = 2;
        assert_eq!(read_buffer(&buf), Err(DecodeError::UnsupportedVersion(2)));
    }

    #[test]
    fn an_unknown_opcode_is_refused() {
        let mut buf = new_buffer();
        buf.u32(99);
        buf.u32(0);
        assert_eq!(read_buffer(&buf), Err(DecodeError::BadOpcode(99)));
    }

    #[test]
    fn a_truncated_buffer_is_an_error_not_a_panic() {
        let mut buf = new_buffer();
        Record::CreateEdge {
            ids: IdList::from([5, 6]),
            relation_id: 1,
            src: IdList::from([1, 1]),
            dst: IdList::from([2, 2]),
            attr_ids: vec![0],
            rows: vec![Value::Int(1), Value::Int(2)],
        }
        .encode(&mut buf);
        // From 3: a header with no records after it is a valid empty payload,
        // not a truncated one.
        for cut in 3..buf.len() {
            assert!(
                matches!(
                    read_buffer(&buf[..cut]),
                    Err(DecodeError::UnexpectedEof { .. } | DecodeError::ImplausibleCount { .. })
                ),
                "cut at {cut}: a truncated payload must run out of bytes, not \
                 trip an unrelated check that happens to fire first"
            );
        }
    }

    // ── the payload header ──

    #[test]
    fn the_flags_byte_is_reserved_even_though_nothing_sets_it() {
        // Reserving it now is the point: adding a byte to the header later
        // would cost another version bump for something compression needs.
        let mut buf = new_buffer();
        Record::DeleteNode {
            ids: IdList::from([1]),
            labels: vec![7],
        }
        .encode(&mut buf);
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
        Record::CreateNode {
            ids: ids.clone(),
            labels: vec![7],
            attr_ids: vec![0],
            rows: rows.to_vec(),
        }
        .encode(&mut buf);
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
        Record::CreateNode {
            ids: IdList::from([0]),
            labels: vec![7],
            attr_ids: vec![0],
            rows: vec![Value::Int(1)],
        }
        .encode(&mut buf);
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
        Record::DeleteNode {
            ids: IdList::from([1]),
            labels: vec![7],
        }
        .encode(&mut buf);
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
        Record::CreateNode {
            ids: ids.clone(),
            labels: vec![7],
            attr_ids: vec![0],
            rows: rows.to_vec(),
        }
        .encode(&mut buf);
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
        Record::CreateNode {
            ids: ids.clone(),
            labels: vec![7],
            attr_ids: vec![0],
            rows: rows.to_vec(),
        }
        .encode(&mut buf);
        assert!(maybe_compress(&mut buf, 16));

        // version, flags, u32 plain_len, u32 comp_len, then the checksum.
        buf[10] ^= 0xFF;
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
        Record::CreateNode {
            ids: IdList::from(ids.as_slice()),
            labels: vec![7],
            attr_ids: vec![0],
            rows: rows.to_vec(),
        }
        .encode(&mut buf);

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
        Record::CreateNode {
            ids: ids.clone(),
            labels: vec![7],
            attr_ids: vec![0],
            rows: rows.to_vec(),
        }
        .encode(&mut buf);
        assert!(maybe_compress(&mut buf, 16));

        // Truncating anywhere inside the frame must error, never panic.
        for cut in 7..buf.len() {
            assert!(
                matches!(
                    read_buffer(&buf[..cut]),
                    Err(DecodeError::UnexpectedEof { .. } | DecodeError::ImplausibleCount { .. })
                ),
                "cut at {cut}: a truncated payload must run out of bytes, not \
                 trip an unrelated check that happens to fire first"
            );
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
    fn a_tiny_buffer_with_an_absurd_count_costs_what_it_weighs() {
        // The whole path, not just `read_ids`: a DELETE_NODE whose segment list
        // is a few bytes but whose count is `u32::MAX`.
        //
        // Decoding stops at the segments, so no part of this asks for the ~34 GB
        // the count implies — the record is refused because the segments do not
        // total it. `GRAPH.EFFECT` runs inline on the Redis main thread with no
        // timeout, so a decoder whose cost is not bounded by its input is a
        // denial of service and not merely a memory problem.
        let mut buf = new_buffer();
        Record::DeleteNode {
            ids: IdList::from([0, 1, 2]),
            labels: vec![],
        }
        .encode(&mut buf);
        // Patch the count that follows the opcode: `header = u32 opcode · u32 count`.
        let count_at = 2 + 4;
        buf[count_at..count_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());

        assert!(matches!(
            read_buffer(&buf),
            Err(DecodeError::CardinalityMismatch { .. })
        ));
    }

    /// A record covering no entities says nothing about any of them.
    ///
    /// Refused at the header, before any block is read, because `count` sizes
    /// every block that follows — one check there keeps a zero-length parse
    /// path out of each of them rather than requiring both engines to agree on
    /// all of those paths separately.
    ///
    /// Hand-built, because the emitter cannot produce one: records come from
    /// grouping and a group only exists once an id is pushed into it. So this
    /// pins what may *arrive*, which is the only way such a record could occur.
    #[test]
    fn a_record_covering_no_entities_is_refused() {
        // DELETE_NODE, count 0, then the blocks a count of 0 implies: an empty
        // label set and an id list of no segments. Well-formed byte-wise.
        let mut buf = vec![EFFECTS_VERSION, 0];
        buf.bytes(&(Opcode::DeleteNode as u32).to_le_bytes());
        buf.bytes(&0_u32.to_le_bytes()); // count
        buf.bytes(&0_u16.to_le_bytes()); // empty label set
        buf.bytes(&0_u32.to_le_bytes()); // no segments

        let mut r = Reader::new(&buf[2..]);
        assert!(
            matches!(read_record(&mut r), Err(DecodeError::EmptyRecord { .. })),
            "a zero-count record must be refused at the header"
        );

        // A count of 1 with the same shape is the control: the check must be
        // about the count, not about the blocks being empty. An empty label set
        // stays legal — sixteen fixtures depend on it.
        let mut ok = vec![EFFECTS_VERSION, 0];
        ok.bytes(&(Opcode::DeleteNode as u32).to_le_bytes());
        ok.bytes(&1_u32.to_le_bytes());
        ok.bytes(&0_u16.to_le_bytes());
        ok.bytes(&1_u32.to_le_bytes()); // one segment
        ok.bytes(&[0_u8, 7, 1]); // header, base 7, len 1
        let mut r = Reader::new(&ok[2..]);
        assert!(
            read_record(&mut r).is_ok(),
            "count 1 with an empty label set must still decode"
        );
    }

    #[test]
    fn a_compressed_payload_states_where_its_frame_ends() {
        // The frame length is on the wire, so the payload is self-delimiting:
        // a reader bounds the decompress input from the header instead of from
        // wherever the buffer happens to stop. That is what lets a payload be
        // carried inside a larger stream, and what lets C size its input
        // without a bounds check it does not have in release builds.
        let ids: IdList = (0..2_000).collect();
        let rows: Vec<Value> = (0..2_000).map(|i| Value::Int(i as i64)).collect();
        let mut buf = new_buffer();
        Record::CreateNode {
            ids,
            labels: vec![7],
            attr_ids: vec![0],
            rows,
        }
        .encode(&mut buf);
        assert!(maybe_compress(&mut buf, 16));

        // version, flags, u32 plain_len, u32 comp_len, u32 checksum, frame.
        let comp_len = u32::from_le_bytes(buf[6..10].try_into().unwrap()) as usize;
        assert_eq!(
            comp_len,
            buf.len() - 2 - COMPRESSED_PREFIX,
            "the declared frame length must be the frame that is there"
        );

        // A byte appended after the frame is not silently ignored.
        let mut trailing = buf.clone();
        trailing.push(0);
        assert!(
            matches!(
                open_payload(&trailing),
                Err(DecodeError::TrailingBytes { .. })
            ),
            "bytes after the frame mean the header and the payload disagree"
        );

        // And a frame that claims to be longer than the buffer is refused
        // rather than read past.
        let mut overlong = buf.clone();
        overlong[6..10].copy_from_slice(&(comp_len as u32 + 1).to_le_bytes());
        assert!(
            matches!(
                open_payload(&overlong),
                Err(DecodeError::UnexpectedEof { .. })
            ),
            "a frame longer than the buffer must run out of bytes, not be \
             refused by some unrelated check that happens to fire first"
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
        // Declare a plaintext far smaller than the frame really expands to. The
        // compressed length is honest: this is about the expansion ceiling, not
        // about framing.
        buf.bytes(&64_u32.to_le_bytes());
        buf.bytes(&(frame.len() as u32).to_le_bytes());
        buf.bytes(&0_u32.to_le_bytes());
        buf.bytes(&frame);

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
        let announce = |create, status| Record::Constraint {
            create,
            constraint_type: ConstraintType::Unique,
            entity_type: EntityType::Node,
            status,
            label_id: 3,
            label: "Person".to_owned(),
            props: vec![AttrRef {
                id: 0,
                name: "email".to_owned(),
            }],
        };

        let mut created = new_buffer();
        announce(true, Some(ConstraintStatus::Failed)).encode(&mut created);
        let mut dropped = new_buffer();
        announce(false, None).encode(&mut dropped);

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
