//! The cross-engine conformance corpus: committed payloads, and the test that
//! keeps them honest.
//!
//! Two engines have to agree on these bytes. Round-tripping each side against
//! itself proves only self-consistency — C could read every field one width
//! narrow and still pass its own tests, which is precisely the failure that
//! produced the `AttributeSet_Update` segfault when C was fed a Rust buffer.
//! What settles it is a corpus one engine produced and the other has to read.
//!
//! So the files under `tests/fixtures/effects_v3/` are checked in, and this
//! module regenerates them and asserts equality. A change to the encoder that
//! moves a single byte fails here with the case that moved — before it reaches
//! a replica, and before the C side is left chasing a format that shifted under
//! it.
//!
//! ## What the far side does with them
//!
//! The minimal C test needs no expectation format at all: decode a `.hex`,
//! re-encode it, compare. That catches every width, order and truncation error,
//! because a decoder reading one field wrong writes it back wrong. The `.json`
//! beside it says what the bytes mean, for a reviewer and for the targeted
//! assertions worth writing by hand.
//!
//! ## Regenerating
//!
//! ```text
//! UPDATE_EFFECTS_FIXTURES=1 cargo test -p graph effects::v3::fixtures
//! ```
//!
//! Regenerate only when the format was *meant* to change, and say so in the
//! commit — a diff here is a wire break, which for the C engine means a version
//! bump rather than a patch.

use std::{fmt::Write as _, fs, path::PathBuf, sync::Arc};

use crate::runtime::{string_pool, value::Value};

use super::*;

// ── the wire numbers, restated ──

/// Segment kinds, written out rather than imported from [`super::id_list`].
///
/// The corpus is a statement about the wire, so it has to be able to disagree
/// with the encoder. Importing the encoder's own constants would make a
/// renumbering invisible here — the fixtures would regenerate happily and both
/// sides of the comparison would move together. These are the numbers
/// `docs/effects-v3.md` publishes; if the encoder stops matching them, the
/// shape assertions below fail, which is the alarm.
const KIND_RANGE: u8 = 0;
const KIND_ASCENDING: u8 = 1;
const KIND_REPEAT: u8 = 2;

/// Bit 6 of the segment header. Restated here for the same reason the kinds
/// are: the corpus has to be able to disagree with the encoder.
const DESCENDING: u8 = 0b0100_0000;

/// Bits 2-3 and 4-5. The code is the log2 of the byte width: 1, 2, 4, 8 map to
/// 0, 1, 2, 3.
const VALUE_WIDTH_SHIFT: u8 = 2;
const COUNT_WIDTH_SHIFT: u8 = 4;

/// Where the committed corpus lives, from this crate's manifest.
fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/effects_v3")
}

// ── building the cases ──

/// Frame records into a complete payload, exactly as replication would.
///
/// Uncompressed, always. Compression is an encoder choice — level, and zstd's
/// version — and baking one into a conformance vector would pin C to a
/// compressor rather than to a format. The flags byte is still here, and is
/// still zero, so the far side parses the same header it will see in
/// production.
fn payload(records: &[Record]) -> Vec<u8> {
    let mut buf = vec![EFFECTS_VERSION, 0];
    for rec in records {
        rec.encode(&mut buf);
    }
    buf
}

/// A list of `n` ids, `step` apart — ascending and sparse, so it cannot stay one
/// range.
fn sparse(
    n: usize,
    step: u64,
) -> IdList {
    (0..n as u64).map(|i| i * step).collect()
}

/// Whether a list encodes to exactly one `Ascending` segment.
///
/// Read off the encoded block rather than asked of the struct: what a reviewer
/// and the C side see is the bytes, and the shape claim should be made about
/// those.
fn is_one_bitmap(ids: &IdList) -> bool {
    let mut buf = Vec::new();
    ids.encode(&mut buf);
    buf[..4] == [1, 0, 0, 0] && buf[4] & 0b11 == KIND_ASCENDING
}

/// The smallest sparse ascending list that collapses to a bitmap.
///
/// Found rather than hardcoded. The trigger is arithmetic over the run's shape
/// — `range_bytes >= 32 && 5 + bitmap_bytes < range_bytes` — and writing the
/// answer down here would leave the corpus pinned to a boundary the encoder had
/// since moved, straddling nothing.
fn collapse_boundary(step: u64) -> usize {
    (2..512)
        .find(|&n| is_one_bitmap(&sparse(n, step)))
        .expect("a sparse ascending run collapses well before 512 ids")
}

/// A property value of every shape that can reach the wire.
fn every_value_kind() -> Vec<Value> {
    vec![
        Value::Null,
        Value::Bool(true),
        Value::Bool(false),
        Value::Int(-1),
        Value::Int(i64::MAX),
        Value::Float(1.5),
        Value::Float(f64::NEG_INFINITY),
        Value::String(Arc::new("plain".to_owned())),
        // The same contents through the pool, so the corpus carries both string
        // tags. C must accept `T_INTERN | T_STRING` and plain `T_STRING` as the
        // same value — the intern bit is a hint about the primary's pool, not a
        // different type, and a decoder that switches on the whole tag word
        // rather than masking will reject one of these two.
        Value::String(string_pool::global().intern(Arc::new("pooled".to_owned()))),
        Value::List(Arc::new(thin_vec::thin_vec![
            Value::Int(1),
            Value::List(Arc::new(thin_vec::thin_vec![Value::Bool(true)])),
        ])),
        Value::Point(crate::runtime::value::Point {
            latitude: 32.0853,
            longitude: 34.7818,
        }),
        Value::VecF32(Arc::new(thin_vec::thin_vec![1.0_f32, -0.5])),
        Value::Datetime(1_700_000_000),
        Value::Date(19_723),
        Value::Time(3_661),
        Value::Duration(86_400),
    ]
}

/// The corpus: a name, and the records that name frames.
///
/// Ordered so a reviewer reads it as an argument — the segment kinds first,
/// then the boundary they turn on, then one case per record.
fn cases() -> Vec<(&'static str, Vec<Record>)> {
    let step = 1024;
    let n = collapse_boundary(step);

    let mut out: Vec<(&'static str, Vec<Record>)> = Vec::new();

    // ── segment kinds, each isolated ──
    // `DeleteNode` with no labels is the thinnest carrier there is: an opcode, a
    // count, an empty label set, and the list. What differs between these cases
    // is the segment, and nothing else.
    let seg = |ids: IdList| {
        vec![Record::DeleteNode {
            ids,
            labels: Vec::new(),
        }]
    };

    out.push(("seg_range", seg((100..108).collect())));
    out.push(("seg_range_single", seg(IdList::from([7]))));
    // Descending: every id its own range of one, the shuffled case.
    out.push((
        "seg_range_many",
        seg([9_u64, 4, 7, 1].into_iter().collect()),
    ));
    out.push(("seg_repeat", seg(std::iter::repeat_n(42, 6).collect())));
    // Comfortably past the boundary, not sitting on it: the two `collapse_*`
    // cases below are the boundary, and a general example of the kind should not
    // move every time the trigger is tuned.
    out.push(("seg_ascending", seg(sparse(n * 4, step))));

    // ── direction: bit 6 ──
    // The same ids in both orders. A pair rather than one case because that is
    // what makes the bit's *meaning* checkable: on its own, a descending
    // fixture proves only that some bit is set somewhere. Here the two files
    // differ in bit 6 and in the base, and in nothing else — an engine that
    // ignores the bit reads the second as the first and the ids come back
    // reversed rather than malformed, which no round-trip catches.
    let both_ways: Vec<u64> = (200..208).collect();
    out.push(("dir_ascending", seg(both_ways.iter().copied().collect())));
    out.push((
        "dir_descending",
        seg(both_ways.iter().rev().copied().collect()),
    ));

    // A descending range that ends at zero. The base is the *highest* id for a
    // descending segment, so the arithmetic runs toward 0 and an off-by-one
    // underflows rather than overshooting — the case a signed or wider
    // intermediate hides.
    out.push(("dir_descending_to_zero", seg((0..8).rev().collect())));

    // A descending gapped run, past the collapse boundary: the bitmap is a set
    // and has no direction, so the payload here is byte-identical to the
    // ascending form and only bit 6 says which way the ids came. That is the
    // strongest statement the corpus can make about the bit.
    out.push((
        "dir_descending_bitmap",
        seg((0..n as u64).rev().map(|i| i * step).collect()),
    ));

    // ── value width codes 2 and 3 ──
    // Every existing case is width code 0 or 1, so a wrong entry for the two
    // larger widths passes the whole corpus. Past 2^16 takes 4 bytes; past
    // 2^32 takes 8.
    out.push(("value_width_4_bytes", seg((0x1_0000..0x1_0008).collect())));
    out.push((
        "value_width_8_bytes",
        seg((0x1_0000_0000..0x1_0000_0008).collect()),
    ));

    // ── count width codes 1 and 2 ──
    // Code 0 across all 25 existing cases, so a bug confined to the count path
    // is invisible in the corpus today. One range, long enough that its length
    // needs 2 bytes, and one that needs 4.
    out.push(("count_width_2_bytes", seg((0..300).collect())));
    out.push(("count_width_4_bytes", seg((0..70_000).collect())));

    // ── the collapse boundary, from both sides ──
    // One id apart, and the two files differ in kind rather than in size. If the
    // trigger ever moves, these two stop straddling it and `shape` fails.
    out.push(("collapse_below", seg(sparse(n - 1, step))));
    out.push(("collapse_above", seg(sparse(n, step))));

    // ── a list of several segments, of mixed kind ──
    // A range, then a repeat, then a descending straggler: the three kinds in
    // one block, which is where a decoder that reads the segment count as a
    // "kind" or stops after the first segment gives itself away.
    let mut mixed = IdList::new();
    for id in 10..14 {
        mixed.push(id);
    }
    for _ in 0..3 {
        mixed.push(99);
    }
    mixed.push(5);
    out.push(("seg_mixed", seg(mixed)));

    // ── one case per record ──
    let ids: IdList = (1..4).collect();
    let attrs = vec![7_u16, 9];
    let rows = vec![
        Value::Int(1),
        Value::String(Arc::new("a".to_owned())),
        Value::Null,
        Value::Bool(true),
        Value::Float(0.25),
        Value::Int(-9),
    ];

    out.push((
        "rec_create_node",
        vec![Record::CreateNode {
            ids: ids.clone(),
            labels: vec![0, 3],
            attr_ids: attrs.clone(),
            rows: rows.clone(),
        }],
    ));
    out.push((
        "rec_create_edge",
        vec![Record::CreateEdge {
            ids: ids.clone(),
            relation_id: 2,
            // A supernode's fan-out: one `Repeat` for the source, a `Range` for
            // the destinations. This is the shape the `Repeat` kind exists for.
            src: std::iter::repeat_n(100, 3).collect(),
            dst: (200..203).collect(),
            attr_ids: attrs.clone(),
            rows: rows.clone(),
        }],
    ));
    // ── the empty attribute set ──
    // `CREATE (:Person)` and `CREATE (a)-[:R]->(b)`: the commonest writes in
    // the language, and the corpus had no case for either. Every create fixture
    // above carries two attributes, so a decoder that treats `n_attrs == 0` as
    // malformed — reading "entities but no attribute ids" as saying nothing
    // about them, rather than as saying they have no properties — passes the
    // whole corpus while rejecting ordinary traffic.
    //
    // Count is 3, not 1, so a decoder that happens to work for a single entity
    // and mis-strides the empty value block does not pass by accident.
    out.push((
        "create_node_no_attrs",
        vec![Record::CreateNode {
            ids: ids.clone(),
            labels: vec![0],
            attr_ids: Vec::new(),
            rows: Vec::new(),
        }],
    ));
    out.push((
        "create_edge_no_attrs",
        vec![Record::CreateEdge {
            ids: ids.clone(),
            relation_id: 2,
            src: std::iter::repeat_n(100, 3).collect(),
            dst: (200..203).collect(),
            attr_ids: Vec::new(),
            rows: Vec::new(),
        }],
    ));

    out.push((
        "rec_update_node",
        vec![Record::Update {
            entity: EntityType::Node,
            ids: ids.clone(),
            labels: vec![1],
            relation_id: None,
            attr_ids: attrs.clone(),
            rows: rows.clone(),
        }],
    ));
    out.push((
        "rec_update_edge",
        vec![Record::Update {
            entity: EntityType::Relationship,
            ids: ids.clone(),
            // An edge carries no labels: it carries its one relationship type
            // in the same slot. A decoder that reads a label set here consumes
            // the type's four bytes as a two-byte count and a stray label, and
            // misaligns every block after it.
            labels: Vec::new(),
            relation_id: Some(2),
            attr_ids: attrs.clone(),
            rows: rows.clone(),
        }],
    ));
    out.push((
        "rec_delete_node",
        vec![Record::DeleteNode {
            ids: ids.clone(),
            labels: vec![0, 1, 2],
        }],
    ));
    out.push((
        "rec_delete_edge",
        vec![Record::DeleteEdge {
            ids: ids.clone(),
            relation_id: 5,
            src: (10..13).collect(),
            dst: std::iter::repeat_n(77, 3).collect(),
        }],
    ));
    out.push((
        "rec_set_labels",
        vec![Record::Labels {
            add: true,
            ids: ids.clone(),
            labels: vec![4, 6],
        }],
    ));
    out.push((
        "rec_remove_labels",
        vec![Record::Labels {
            add: false,
            ids: ids.clone(),
            labels: vec![4],
        }],
    ));
    out.push((
        "rec_add_schema_node",
        vec![Record::AddSchema {
            schema_type: EntityType::Node,
            id: 3,
            name: "Person".to_owned(),
        }],
    ));
    out.push((
        "rec_add_schema_edge",
        vec![Record::AddSchema {
            schema_type: EntityType::Relationship,
            id: 1,
            name: "KNOWS".to_owned(),
        }],
    ));
    out.push((
        "rec_add_attribute",
        vec![Record::AddAttribute {
            id: 12,
            name: "name".to_owned(),
        }],
    ));
    out.push((
        "rec_create_index",
        vec![Record::Index {
            create: true,
            schema_type: EntityType::Node,
            label_id: 3,
            label: "Person".to_owned(),
            // A bit set, not an enum: range and fulltext at once is one
            // statement, and a decoder that switches on the word rather than
            // testing bits cannot represent it.
            field_type: index_field_type::INDEX_FLD_STR | index_field_type::INDEX_FLD_NUMERIC,
            fields: vec![
                AttrRef {
                    id: 12,
                    name: "name".to_owned(),
                },
                AttrRef {
                    id: 13,
                    name: "age".to_owned(),
                },
            ],
            // What the OPTIONS map used to say, now typed: a vector field of
            // dimension 4, with the RDB's HNSW defaults.
            // A range statement, so no vector half — the invariant is that the
            // options match the field type.
            options: Some(IndexOptions::none_given(None)),
        }],
    )); // The vector half of the options block, which the range case above cannot
    // reach: `field_type` carries INDEX_FLD_VECTOR, so five more u64s follow
    // the phonetic string. Without this the HNSW parameters are unexercised.
    out.push((
        "rec_create_index_vector",
        vec![Record::Index {
            create: true,
            schema_type: EntityType::Node,
            label_id: 0,
            label: "P".to_owned(),
            field_type: index_field_type::INDEX_FLD_VECTOR,
            fields: vec![AttrRef {
                id: 0,
                name: "embedding".to_owned(),
            }],
            // Every vector option stated, so the presence bytes are all 1 and
            // the values are non-default — a fixture where every field absent
            // would leave the payload indistinguishable from `none_given`.
            options: Some(IndexOptions::none_given(Some(VectorOptions {
                dimension: 128,
                m: Some(32),
                ef_construction: Some(400),
                ef_runtime: Some(20),
                // cosine, not the L2 default, so the field is not zero
                sim_func: Some(2),
            }))),
        }],
    ));

    out.push((
        "rec_drop_index",
        vec![Record::Index {
            create: false,
            schema_type: EntityType::Relationship,
            label_id: 1,
            label: "KNOWS".to_owned(),
            field_type: index_field_type::INDEX_FLD_VECTOR,
            fields: vec![AttrRef {
                id: 14,
                name: "vec".to_owned(),
            }],
            // A drop carries no options at all — not an empty map.
            options: None,
        }],
    ));
    out.push((
        "rec_create_constraint",
        vec![Record::Constraint {
            create: true,
            constraint_type: ConstraintType::Unique,
            entity_type: EntityType::Node,
            // The primary's outcome, carried so a replica does not re-derive it.
            status: Some(ConstraintStatus::Operational),
            label_id: 3,
            label: "Person".to_owned(),
            props: vec![AttrRef {
                id: 12,
                name: "name".to_owned(),
            }],
        }],
    ));
    out.push((
        "rec_drop_constraint",
        vec![Record::Constraint {
            create: false,
            constraint_type: ConstraintType::Mandatory,
            entity_type: EntityType::Relationship,
            status: None,
            label_id: 1,
            label: "KNOWS".to_owned(),
            props: vec![
                AttrRef {
                    id: 12,
                    name: "name".to_owned(),
                },
                AttrRef {
                    id: 13,
                    name: "age".to_owned(),
                },
            ],
        }],
    ));

    // ── every value shape, in one record ──
    let vals = every_value_kind();
    out.push((
        "values_all_kinds",
        vec![Record::CreateNode {
            ids: IdList::from([1]),
            labels: vec![0],
            attr_ids: (0..vals.len() as u16).collect(),
            rows: vals,
        }],
    ));

    // ── several records in one payload ──
    // Batching removes per-record framing; it never multiplies commands. A
    // decoder that stops after the first record, or that re-reads the version
    // byte per record, fails here and nowhere else.
    out.push((
        "payload_multi_record",
        vec![
            Record::AddSchema {
                schema_type: EntityType::Node,
                id: 0,
                name: "N".to_owned(),
            },
            Record::AddAttribute {
                id: 0,
                name: "p".to_owned(),
            },
            Record::CreateNode {
                ids: (1..5).collect(),
                labels: vec![0],
                attr_ids: vec![0],
                rows: vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)],
            },
            Record::DeleteNode {
                ids: IdList::from([2]),
                labels: vec![0],
            },
        ],
    ));

    out
}

// ── rendering ──

/// The payload as hex: a comment header, then 32 bytes a line.
///
/// Hex rather than a binary blob so the corpus reviews in a pull request and
/// diffs by line when it changes. A reader strips `#` comments and whitespace,
/// which is three lines in C and one in Python.
fn to_hex(
    name: &str,
    bytes: &[u8],
) -> String {
    let mut s = format!(
        "# effects v3 conformance fixture: {name}\n# {} bytes, uncompressed\n",
        bytes.len()
    );
    for chunk in bytes.chunks(32) {
        for b in chunk {
            let _ = write!(s, "{b:02x}");
        }
        s.push('\n');
    }
    s
}

/// One value, as a flat tagged string.
///
/// Flat on purpose: the point is to pin the *type tag* a decoder must produce,
/// and a nested object would need a real parser on the far side to say what a
/// `"int:1"` says on its own.
fn describe_value(v: &Value) -> String {
    match v {
        Value::Null => "null".to_owned(),
        Value::Bool(b) => format!("bool:{b}"),
        Value::Int(i) => format!("int:{i}"),
        Value::Float(f) => format!("float:{f:?}"),
        Value::String(s) => {
            // The tag the encoder will actually write, not the type alone.
            let tag = if string_pool::global().is_interned(s) {
                "intern"
            } else {
                "string"
            };
            format!("{tag}:{s}")
        }
        Value::List(items) => format!(
            "list[{}]:{}",
            items.len(),
            items
                .iter()
                .map(describe_value)
                .collect::<Vec<_>>()
                .join("|")
        ),
        Value::Map(m) => format!(
            "map[{}]:{}",
            m.len(),
            m.iter()
                .map(|(k, v)| format!("{k}={}", describe_value(v)))
                .collect::<Vec<_>>()
                .join("|")
        ),
        Value::Point(p) => format!("point:{:?},{:?}", p.latitude, p.longitude),
        Value::VecF32(v) => format!(
            "vecf32[{}]:{}",
            v.len(),
            v.iter()
                .map(|f| format!("{f:?}"))
                .collect::<Vec<_>>()
                .join("|")
        ),
        Value::Datetime(t) => format!("datetime:{t}"),
        Value::Date(t) => format!("date:{t}"),
        Value::Time(t) => format!("time:{t}"),
        Value::Duration(d) => format!("duration:{d}"),
        other => unreachable!("not encodable: {other:?}"),
    }
}

/// A JSON array literal of pre-rendered scalars.
fn arr<T: std::fmt::Display>(items: impl IntoIterator<Item = T>) -> String {
    let inner = items
        .into_iter()
        .map(|i| format!("\"{i}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

/// One record's fields, as `"key": value` lines.
///
/// Deliberately one field per line and never nested past an array of scalars:
/// the far side has no JSON parser, and `strstr` for `"opcode"` has to be a
/// workable way to read this.
fn describe(rec: &Record) -> Vec<(String, String)> {
    let mut f: Vec<(String, String)> = Vec::new();
    let mut put = |k: &str, v: String| f.push((k.to_owned(), v));
    let ids = |l: &IdList| arr(l.iter().map(|i| i.to_string()));

    match rec {
        Record::CreateNode {
            ids: i,
            labels,
            attr_ids,
            rows,
        } => {
            put("opcode", "3".to_owned());
            put("record", "\"CREATE_NODE\"".to_owned());
            put("count", i.count().to_string());
            put("ids", ids(i));
            put("labels", arr(labels.iter().map(u32::to_string)));
            put("attr_ids", arr(attr_ids.iter().map(u16::to_string)));
            put("rows", arr(rows.iter().map(describe_value)));
        }
        Record::CreateEdge {
            ids: i,
            relation_id,
            src,
            dst,
            attr_ids,
            rows,
        } => {
            put("opcode", "4".to_owned());
            put("record", "\"CREATE_EDGE\"".to_owned());
            put("count", i.count().to_string());
            put("relation_id", relation_id.to_string());
            put("ids", ids(i));
            put("src", ids(src));
            put("dst", ids(dst));
            put("attr_ids", arr(attr_ids.iter().map(u16::to_string)));
            put("rows", arr(rows.iter().map(describe_value)));
        }
        Record::Update {
            entity,
            ids: i,
            labels,
            relation_id,
            attr_ids,
            rows,
        } => {
            let node = *entity == EntityType::Node;
            put("opcode", if node { "1" } else { "2" }.to_owned());
            put(
                "record",
                if node {
                    "\"UPDATE_NODE\""
                } else {
                    "\"UPDATE_EDGE\""
                }
                .to_owned(),
            );
            put("count", i.count().to_string());
            put("ids", ids(i));
            if node {
                put("labels", arr(labels.iter().map(u32::to_string)));
            } else {
                put(
                    "relation_id",
                    relation_id.map_or_else(|| "null".to_owned(), |r| r.to_string()),
                );
            }
            put("attr_ids", arr(attr_ids.iter().map(u16::to_string)));
            put("rows", arr(rows.iter().map(describe_value)));
        }
        Record::DeleteNode { ids: i, labels } => {
            put("opcode", "5".to_owned());
            put("record", "\"DELETE_NODE\"".to_owned());
            put("count", i.count().to_string());
            put("ids", ids(i));
            put("labels", arr(labels.iter().map(u32::to_string)));
        }
        Record::DeleteEdge {
            ids: i,
            relation_id,
            src,
            dst,
        } => {
            put("opcode", "6".to_owned());
            put("record", "\"DELETE_EDGE\"".to_owned());
            put("count", i.count().to_string());
            put("relation_id", relation_id.to_string());
            put("ids", ids(i));
            put("src", ids(src));
            put("dst", ids(dst));
        }
        Record::Labels {
            add,
            ids: i,
            labels,
        } => {
            put("opcode", if *add { "7" } else { "8" }.to_owned());
            put(
                "record",
                if *add {
                    "\"SET_LABELS\""
                } else {
                    "\"REMOVE_LABELS\""
                }
                .to_owned(),
            );
            put("count", i.count().to_string());
            put("ids", ids(i));
            put("labels", arr(labels.iter().map(u32::to_string)));
        }
        Record::AddSchema {
            schema_type,
            id,
            name,
        } => {
            put("opcode", "9".to_owned());
            put("record", "\"ADD_SCHEMA\"".to_owned());
            // 0-based here, unlike the constraint records' entity tag. The two
            // numberings are the reason this field is spelled out.
            put(
                "schema_type",
                if *schema_type == EntityType::Node {
                    "0"
                } else {
                    "1"
                }
                .to_owned(),
            );
            put("id", id.to_string());
            put("name", format!("\"{name}\""));
        }
        Record::AddAttribute { id, name } => {
            put("opcode", "10".to_owned());
            put("record", "\"ADD_ATTRIBUTE\"".to_owned());
            put("id", id.to_string());
            put("name", format!("\"{name}\""));
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
            put("opcode", if *create { "11" } else { "12" }.to_owned());
            put(
                "record",
                if *create {
                    "\"CREATE_INDEX\""
                } else {
                    "\"DROP_INDEX\""
                }
                .to_owned(),
            );
            put(
                "schema_type",
                if *schema_type == EntityType::Node {
                    "0"
                } else {
                    "1"
                }
                .to_owned(),
            );
            put("label_id", label_id.to_string());
            put("label", format!("\"{label}\""));
            // A bit set. Printed in hex so it reads as flags rather than as an
            // ordinal a reader might try to match against an enum.
            put("field_type", format!("\"{field_type:#06x}\""));
            put(
                "fields",
                arr(fields.iter().map(|f| format!("{}={}", f.id, f.name))),
            );
            put(
                "options",
                options.as_ref().map_or_else(
                    || "null".to_owned(),
                    |o| {
                        let mut parts = Vec::new();
                        let opt =
                            |name: &str, v: Option<String>| v.map(|s| format!("\"{name}\": {s}"));
                        parts.extend(opt(
                            "language",
                            o.language.as_ref().map(|l| format!("\"{l}\"")),
                        ));
                        parts.extend(opt(
                            "stopwords",
                            o.stopwords.as_ref().map(|sw| arr(sw.iter())),
                        ));
                        parts.extend(opt("weight", o.weight.map(|w| w.to_string())));
                        parts.extend(opt("nostem", o.nostem.map(|n| n.to_string())));
                        parts.extend(opt(
                            "phonetic",
                            o.phonetic.as_ref().map(|p| format!("\"{p}\"")),
                        ));
                        if let Some(v) = &o.vector {
                            let mut vp = vec![format!("\"dimension\": {}", v.dimension)];
                            vp.extend(opt("M", v.m.map(|x| x.to_string())));
                            vp.extend(opt(
                                "efConstruction",
                                v.ef_construction.map(|x| x.to_string()),
                            ));
                            vp.extend(opt("efRuntime", v.ef_runtime.map(|x| x.to_string())));
                            vp.extend(opt("simFunc", v.sim_func.map(|x| x.to_string())));
                            parts.push(format!("\"vector\": {{{}}}", vp.join(", ")));
                        }
                        format!("{{{}}}", parts.join(", "))
                    },
                ),
            );
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
            put("opcode", if *create { "13" } else { "14" }.to_owned());
            put(
                "record",
                if *create {
                    "\"CREATE_CONSTRAINT\""
                } else {
                    "\"DROP_CONSTRAINT\""
                }
                .to_owned(),
            );
            put(
                "constraint_type",
                format!("\"{constraint_type}\"").to_lowercase(),
            );
            // 1-based, because C's `GraphEntityType` reserves 0 for UNKNOWN —
            // the other of the two numberings.
            put(
                "entity_type",
                if *entity_type == EntityType::Node {
                    "1"
                } else {
                    "2"
                }
                .to_owned(),
            );
            put(
                "status",
                status.map_or_else(|| "null".to_owned(), |s| format!("\"{s}\"").to_lowercase()),
            );
            put("label_id", label_id.to_string());
            put("label", format!("\"{label}\""));
            put(
                "props",
                arr(props.iter().map(|p| format!("{}={}", p.id, p.name))),
            );
        }
    }
    f
}

/// The whole payload as JSON.
fn to_json(
    name: &str,
    records: &[Record],
    bytes: &[u8],
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "{{");
    let _ = writeln!(s, "  \"case\": \"{name}\",");
    let _ = writeln!(s, "  \"version\": {EFFECTS_VERSION},");
    let _ = writeln!(s, "  \"flags\": 0,");
    let _ = writeln!(s, "  \"bytes\": {},", bytes.len());
    let _ = writeln!(s, "  \"records\": [");
    for (i, rec) in records.iter().enumerate() {
        let _ = writeln!(s, "    {{");
        let fields = describe(rec);
        for (j, (k, v)) in fields.iter().enumerate() {
            let comma = if j + 1 == fields.len() { "" } else { "," };
            let _ = writeln!(s, "      \"{k}\": {v}{comma}");
        }
        let comma = if i + 1 == records.len() { "" } else { "," };
        let _ = writeln!(s, "    }}{comma}");
    }
    let _ = writeln!(s, "  ]");
    let _ = writeln!(s, "}}");
    s
}

// ── the corpus README ──

fn readme(cases: &[(&'static str, Vec<Record>)]) -> String {
    let mut s = String::new();
    s.push_str(
        "# Effects v3 conformance fixtures\n\
         \n\
         Generated. Do not hand-edit — see `graph/src/effects/v3/fixtures.rs`, which\n\
         regenerates these and fails the build if they drift:\n\
         \n\
         ```sh\n\
         UPDATE_EFFECTS_FIXTURES=1 cargo test -p graph effects::v3::fixtures\n\
         ```\n\
         \n\
         Each case is a pair. `<case>.hex` is a complete `GRAPH.EFFECT` payload —\n\
         version byte, flags byte, then records — as hex, 32 bytes a line, with `#`\n\
         comments and whitespace to be stripped by the reader. `<case>.json` says what\n\
         those bytes decode to.\n\
         \n\
         ## What these are for\n\
         \n\
         Two engines have to agree on this format, and each round-tripping against\n\
         itself proves only that it is self-consistent. An engine can read every field\n\
         one width narrow and pass all of its own tests; that is the failure that\n\
         segfaulted C in `AttributeSet_Update` when it was handed a Rust buffer. These\n\
         files are the shared statement neither side can quietly move.\n\
         \n\
         The minimal test on the far side needs no JSON parser: read the `.hex`, decode\n\
         it, re-encode it, compare. A decoder that reads a field wrong writes it back\n\
         wrong, so byte equality after a round trip catches width, order and truncation\n\
         errors on its own. Rust runs that same test, on these same files rather than\n\
         on bytes regenerated beside them.\n\
         \n\
         ## They are not ground truth\n\
         \n\
         They are generated from the Rust encoder. If it is wrong about C\u{2019}s format,\n\
         they enshrine the error faithfully. The ground truth is C\u{2019}s own source \u{2014} the\n\
         widths, the field order and the tag numbering in `effects.c` and `effects.h`,\n\
         which the encoder cites field by field.\n\
         \n\
         So a disagreement between C and a fixture is not settled by the fixture. It is\n\
         settled against C\u{2019}s format definition, and the fixture regenerated if Rust was\n\
         the one that was wrong. What these files are for is narrower: freezing what has\n\
         been agreed, so neither side moves it by accident.\n\
         \n\
         ## What the `.json` is, and is not\n\
         \n\
         Documentation. It is generated from the same records the `.hex` is, so nothing\n\
         cross-checks one against the other and it adds no verification power.\n\
         \n\
         It earns its place at review time. When a change moves the wire, a `.hex` diff\n\
         is unreadable and the `.json` diff says what actually changed \u{2014} which record,\n\
         which field, which value. It also lets someone writing assertions on the far\n\
         side see what a case contains without decoding it by hand.\n\
         \n\
         ## Two things the corpus is deliberately pinning\n\
         \n\
         - **`schema_type` is 0-based; `entity_type` is 1-based.** C's\n\
           `GraphEntityType` reserves 0 for UNKNOWN, so the schema records and the\n\
           constraint records number node-vs-edge differently. Both appear here.\n\
         - **`T_INTERN | T_STRING` and `T_STRING` are the same value.** The intern bit\n\
           is a hint about the primary's string pool. `values_all_kinds` carries one of\n\
           each, so a decoder that switches on the whole tag word instead of masking\n\
           fails on exactly one of them.\n\
         \n\
         Payloads are uncompressed. Compression is an encoder choice — a level, and a\n\
         zstd version — and pinning one here would bind the far side to a compressor\n\
         rather than to a format.\n\
         \n\
         ## Cases\n\
         \n\
         | case | records | bytes |\n\
         | --- | --- | ---: |\n",
    );
    for (name, records) in cases {
        let _ = writeln!(
            s,
            "| `{name}` | {} | {} |",
            records.len(),
            payload(records).len()
        );
    }
    s
}

// ── tests ──

#[cfg(test)]
mod tests {
    use super::*;

    /// Regenerate the corpus and compare, or rewrite it under
    /// `UPDATE_EFFECTS_FIXTURES=1`.
    ///
    /// A failure here is not a flaky test. It means the bytes this engine writes
    /// have changed, which for a format two engines implement is a wire break —
    /// so the fix is either to undo the change or to bump the version and tell
    /// the other side, never to regenerate and move on.
    #[test]
    fn corpus_matches_the_committed_files() {
        let dir = corpus_dir();
        let cases = cases();
        let update = std::env::var_os("UPDATE_EFFECTS_FIXTURES").is_some();

        if update {
            fs::create_dir_all(&dir).expect("create fixture dir");
            fs::write(dir.join("README.md"), readme(&cases)).expect("write README");
        }

        let mut drifted = Vec::new();
        for (name, records) in &cases {
            let bytes = payload(records);
            for (ext, want) in [
                ("hex", to_hex(name, &bytes)),
                ("json", to_json(name, records, &bytes)),
            ] {
                let path = dir.join(format!("{name}.{ext}"));
                if update {
                    fs::write(&path, &want).expect("write fixture");
                    continue;
                }
                match fs::read_to_string(&path) {
                    Ok(have) if have == want => {}
                    Ok(_) => drifted.push(format!("{name}.{ext}: contents differ")),
                    Err(e) => drifted.push(format!("{name}.{ext}: {e}")),
                }
            }
        }

        assert!(
            drifted.is_empty(),
            "the committed corpus no longer matches this encoder:\n  {}\n\n\
             If the format was meant to change, regenerate with \
             UPDATE_EFFECTS_FIXTURES=1 and say so in the commit — the C engine \
             reads these files.",
            drifted.join("\n  ")
        );
    }

    /// The test the far side will run, run here first — **on the file it will
    /// read**, not on bytes regenerated beside it.
    ///
    /// That distinction is the whole point of reading from disk here. Proving
    /// that freshly encoded bytes round-trip proves the codec agrees with
    /// itself; C consumes the committed `.hex`, so this decodes the committed
    /// `.hex`. The parse is the one C will write: strip `#` comments, ignore
    /// whitespace, two characters to a byte.
    ///
    /// If a case cannot survive this on the engine that produced it, shipping
    /// it as a conformance vector would send the other side chasing our bug.
    #[test]
    fn the_committed_payloads_round_trip() {
        for (name, records) in cases() {
            let path = corpus_dir().join(format!("{name}.hex"));
            let text =
                fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let digits: String = text
                .lines()
                .map(|l| l.split('#').next().unwrap_or(""))
                .collect::<String>()
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            assert_eq!(digits.len() % 2, 0, "{name}: odd number of hex digits");
            let bytes: Vec<u8> = (0..digits.len() / 2)
                .map(|i| {
                    u8::from_str_radix(&digits[i * 2..i * 2 + 2], 16)
                        .unwrap_or_else(|e| panic!("{name}: {e}"))
                })
                .collect();

            let payload_in = open_payload(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
            let decoded: Vec<Record> = payload_in
                .records()
                .collect::<Result<_, _>>()
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(decoded, records, "{name}: decoded to a different record");
            assert_eq!(
                payload(&decoded),
                bytes,
                "{name}: re-encoded to different bytes"
            );
        }
    }

    /// The segment cases really do carry the segment they are named for, and the
    /// two boundary cases really do sit on opposite sides.
    ///
    /// Without this the corpus could regenerate into something uniform — every
    /// case a range — and still pass every test above, having quietly stopped
    /// covering anything.
    #[test]
    fn the_named_shapes_are_the_shapes_on_the_wire() {
        // `DeleteNode` with no labels: 4 opcode + 4 count + 2 empty label set,
        // after the 2-byte payload header.
        const ID_LIST_AT: usize = 2 + 4 + 4 + 2;
        let shape = |name: &str| -> (u32, u8) {
            let (_, records) = cases()
                .into_iter()
                .find(|(n, _)| *n == name)
                .unwrap_or_else(|| panic!("no case {name}"));
            let bytes = payload(&records);
            let n = u32::from_le_bytes(
                bytes[ID_LIST_AT..ID_LIST_AT + 4]
                    .try_into()
                    .expect("4 bytes"),
            );
            (n, bytes[ID_LIST_AT + 4] & 0b11)
        };

        assert_eq!(shape("seg_range"), (1, KIND_RANGE));
        assert_eq!(shape("seg_range_single"), (1, KIND_RANGE));
        // Descending: four ids, four one-long ranges, no collapse.
        assert_eq!(shape("seg_range_many"), (4, KIND_RANGE));
        assert_eq!(shape("seg_repeat"), (1, KIND_REPEAT));
        assert_eq!(shape("seg_ascending"), (1, KIND_ASCENDING));

        // The boundary, from both sides. One id apart and a different kind.
        let (below_n, below_kind) = shape("collapse_below");
        assert!(below_n > 1, "collapse_below collapsed after all");
        assert_eq!(below_kind, KIND_RANGE);
        assert_eq!(shape("collapse_above"), (1, KIND_ASCENDING));

        // Mixed: a range, a repeat, and a straggler that is its own range.
        assert_eq!(shape("seg_mixed"), (3, KIND_RANGE));

        // ── the gap cases assert the bits they were added for ──
        // Otherwise a case named `value_width_8_bytes` that quietly encodes as
        // width 0 would sit in the corpus looking like coverage.
        let header = |name: &str| -> u8 {
            let (_, records) = cases()
                .into_iter()
                .find(|(n, _)| *n == name)
                .unwrap_or_else(|| panic!("no case {name}"));
            payload(&records)[ID_LIST_AT + 4]
        };

        // Direction. The ascending half must NOT set the bit — half the value
        // of the pair is that one of them is a negative control.
        assert_eq!(header("dir_ascending") & DESCENDING, 0);
        assert_ne!(header("dir_descending") & DESCENDING, 0);
        assert_ne!(header("dir_descending_to_zero") & DESCENDING, 0);
        assert_ne!(header("dir_descending_bitmap") & DESCENDING, 0);
        assert_eq!(shape("dir_descending_bitmap").1, KIND_ASCENDING);

        // The pair is the same ids, so the two payloads must differ only in the
        // header byte. If they differ anywhere else, the bit is not carrying the
        // direction on its own and the C side cannot rely on it.
        let asc = payload(
            &cases()
                .into_iter()
                .find(|(n, _)| *n == "dir_ascending")
                .unwrap()
                .1,
        );
        let desc = payload(
            &cases()
                .into_iter()
                .find(|(n, _)| *n == "dir_descending")
                .unwrap()
                .1,
        );
        assert_eq!(asc.len(), desc.len(), "the pair must be the same length");
        let differing: Vec<usize> = (0..asc.len()).filter(|&i| asc[i] != desc[i]).collect();
        assert_eq!(
            differing,
            vec![ID_LIST_AT + 4, ID_LIST_AT + 5],
            "only the header byte and the base may differ: {differing:?}"
        );

        // Widths. The code is in bits 2-3 and 4-5.
        let value_code = |name: &str| (header(name) >> VALUE_WIDTH_SHIFT) & 0b11;
        let count_code = |name: &str| (header(name) >> COUNT_WIDTH_SHIFT) & 0b11;
        assert_eq!(value_code("value_width_4_bytes"), 2);
        assert_eq!(value_code("value_width_8_bytes"), 3);
        assert_eq!(count_code("count_width_2_bytes"), 1);
        assert_eq!(count_code("count_width_4_bytes"), 2);
    }

    /// Case names are the file names, so a duplicate would silently overwrite.
    #[test]
    fn case_names_are_unique() {
        let mut names: Vec<&str> = cases().iter().map(|(n, _)| *n).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "duplicate fixture case name");
    }
}
