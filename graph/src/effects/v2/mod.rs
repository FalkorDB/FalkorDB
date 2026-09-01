//! The **v2** effects wire format: Rust's own, predating C compatibility.
//!
//! Separated from `pending.rs` for the same reason as v3 — a mutation
//! accumulator should not know what a replication frame looks like — but also
//! because this format is on its way out. v3 is the one two engines can both
//! read; v2 exists until every peer speaks v3, and then this file is deleted
//! rather than untangled.
//!
//! Do not add to it. In particular the value tags below are **not** C's: they
//! are sequential 0..12 where C uses a bitmask, `Point` is two `f64` where C
//! writes two `f32`, and list counts are `u64` where C reads `u32`. Every one
//! of those is a divergence v3 exists to close, so sharing code between the two
//! codecs would mean sharing the bugs.

use std::sync::Arc;

use atomic_refcell::AtomicRefCell;

use crate::{
    graph::graph::Graph,
    runtime::{pending::Pending, value::Value},
};

// ── Effects buffer constants and helpers ──

pub const EFFECTS_VERSION: u8 = 2;

pub const EFFECT_UPDATE_NODE: u8 = 1;
pub const EFFECT_UPDATE_EDGE: u8 = 2;
pub const EFFECT_CREATE_NODE: u8 = 3;
pub const EFFECT_CREATE_EDGE: u8 = 4;
pub const EFFECT_DELETE_NODE: u8 = 5;
pub const EFFECT_DELETE_EDGE: u8 = 6;
pub const EFFECT_SET_LABELS: u8 = 7;
pub const EFFECT_REMOVE_LABELS: u8 = 8;
pub const EFFECT_ADD_SCHEMA: u8 = 9;
pub const EFFECT_ADD_ATTRIBUTE: u8 = 10;
pub const EFFECT_CREATE_INDEX: u8 = 11;
pub const EFFECT_DROP_INDEX: u8 = 12;

// Schema type tags (used in EFFECT_ADD_SCHEMA)
pub const SCHEMA_NODE_LABEL: u8 = 0;
pub const SCHEMA_REL_TYPE: u8 = 1;

// Attribute type tags (used in EFFECT_ADD_ATTRIBUTE)
pub const ATTR_NODE: u8 = 0;
pub const ATTR_REL: u8 = 1;

// Value type tags for effect serialization
const VALUE_NULL: u8 = 0;
const VALUE_BOOL: u8 = 1;
const VALUE_INT: u8 = 2;
const VALUE_FLOAT: u8 = 3;
const VALUE_STRING: u8 = 4;
const VALUE_LIST: u8 = 5;
const VALUE_POINT: u8 = 6;
const VALUE_VECF32: u8 = 7;
const VALUE_DATETIME: u8 = 8;
const VALUE_DATE: u8 = 9;
const VALUE_TIME: u8 = 10;
const VALUE_DURATION: u8 = 11;
const VALUE_INTERN_STRING: u8 = 12;

pub fn write_u16(
    buf: &mut Vec<u8>,
    v: u16,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_string(
    buf: &mut Vec<u8>,
    s: &str,
) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn write_value(
    buf: &mut Vec<u8>,
    value: &Value,
) {
    match value {
        Value::Null => buf.push(VALUE_NULL),
        Value::Bool(b) => {
            buf.push(VALUE_BOOL);
            buf.push(u8::from(*b));
        }
        Value::Int(i) => {
            buf.push(VALUE_INT);
            buf.extend_from_slice(&i.to_le_bytes());
        }
        Value::Float(f) => {
            buf.push(VALUE_FLOAT);
            buf.extend_from_slice(&f.to_le_bytes());
        }
        Value::String(s) => {
            if crate::runtime::string_pool::global().is_interned(s) {
                buf.push(VALUE_INTERN_STRING);
            } else {
                buf.push(VALUE_STRING);
            }
            write_string(buf, s);
        }
        Value::List(items) => {
            buf.push(VALUE_LIST);
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items.iter() {
                write_value(buf, item);
            }
        }
        Value::Point(p) => {
            buf.push(VALUE_POINT);
            buf.extend_from_slice(&(p.latitude as f64).to_le_bytes());
            buf.extend_from_slice(&(p.longitude as f64).to_le_bytes());
        }
        Value::VecF32(v) => {
            buf.push(VALUE_VECF32);
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for f in v.iter() {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        Value::Datetime(ts) => {
            buf.push(VALUE_DATETIME);
            buf.extend_from_slice(&ts.to_le_bytes());
        }
        Value::Date(ts) => {
            buf.push(VALUE_DATE);
            buf.extend_from_slice(&ts.to_le_bytes());
        }
        Value::Time(ts) => {
            buf.push(VALUE_TIME);
            buf.extend_from_slice(&ts.to_le_bytes());
        }
        Value::Duration(dur) => {
            buf.push(VALUE_DURATION);
            buf.extend_from_slice(&dur.to_le_bytes());
        }
        _ => {
            debug_assert!(false, "Unsupported value type in effects buffer: {value:?}");
            buf.push(VALUE_NULL); // Fallback for unsupported types
        }
    }
}

pub fn read_string(
    buf: &[u8],
    offset: &mut usize,
) -> Result<Arc<String>, String> {
    if *offset + 8 > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let len = u64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap()) as usize;
    *offset += 8;
    if *offset + len > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let s = std::str::from_utf8(&buf[*offset..*offset + len])
        .map_err(|e| format!("invalid utf8 in effects buffer: {e}"))?;
    *offset += len;
    Ok(Arc::new(s.to_string()))
}

pub fn read_u16(
    buf: &[u8],
    offset: &mut usize,
) -> Result<u16, String> {
    if *offset + 2 > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let v = u16::from_le_bytes(buf[*offset..*offset + 2].try_into().unwrap());
    *offset += 2;
    Ok(v)
}

pub fn read_u64(
    buf: &[u8],
    offset: &mut usize,
) -> Result<u64, String> {
    if *offset + 8 > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let v = u64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    Ok(v)
}

pub fn read_value(
    buf: &[u8],
    offset: &mut usize,
) -> Result<Value, String> {
    if *offset >= buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let tag = buf[*offset];
    *offset += 1;
    match tag {
        VALUE_NULL => Ok(Value::Null),
        VALUE_BOOL => {
            if *offset >= buf.len() {
                return Err("effects buffer truncated".to_string());
            }
            let b = buf[*offset] != 0;
            *offset += 1;
            Ok(Value::Bool(b))
        }
        VALUE_INT => {
            let v = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Int(v))
        }
        VALUE_FLOAT => {
            let v = f64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Float(v))
        }
        VALUE_STRING => {
            let s = read_string(buf, offset)?;
            Ok(Value::String(s))
        }
        VALUE_INTERN_STRING => {
            let s = read_string(buf, offset)?;
            Ok(Value::String(
                crate::runtime::string_pool::global().intern(s),
            ))
        }
        VALUE_LIST => {
            let len = read_u64(buf, offset)? as usize;
            let mut items = thin_vec::ThinVec::with_capacity(len);
            for _ in 0..len {
                items.push(read_value(buf, offset)?);
            }
            Ok(Value::List(Arc::new(items)))
        }
        VALUE_POINT => {
            let lat = f64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            let lon = f64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Point(crate::runtime::value::Point {
                latitude: lat as f32,
                longitude: lon as f32,
            }))
        }
        VALUE_VECF32 => {
            let len = read_u64(buf, offset)? as usize;
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                let f = f32::from_le_bytes(
                    buf.get(*offset..*offset + 4)
                        .ok_or("truncated")?
                        .try_into()
                        .unwrap(),
                );
                *offset += 4;
                v.push(f);
            }
            Ok(Value::VecF32(Arc::new(v.into())))
        }
        VALUE_DATETIME => {
            let ts = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Datetime(ts))
        }
        VALUE_DATE => {
            let ts = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Date(ts))
        }
        VALUE_TIME => {
            let ts = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Time(ts))
        }
        VALUE_DURATION => {
            let dur = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Duration(dur))
        }
        _ => Err(format!("unknown value tag in effects buffer: {tag}")),
    }
}

/// Build a **v2** effects payload — Rust's own format, not C's.
///
/// Kept whole and separate so that retiring it is deleting this file. Nothing
/// else should grow a dependency on it: v3 is the format two engines can both
/// read, and this one exists until every peer speaks it.
pub fn build_effects_buffer(
    p: &Pending,
    g: &AtomicRefCell<Graph>,
    buf: &mut Vec<u8>,
) -> u64 {
    let mut n_effects = 0u64;

    // Pre-allocate buffer: entity headers plus ~12 bytes per attribute
    // (2-byte attribute id + tagged value payload).
    let attr_bytes: usize = p
        .new_nodes_attrs
        .values()
        .chain(p.existing_nodes_attrs.values())
        .chain(p.new_relationships_attrs.values())
        .chain(p.existing_relationships_attrs.values())
        .map(|m| m.len() * 12)
        .sum();
    let estimated_bytes = (p.created_nodes.len() as usize) * 15
        + p.created_rel_types.len() * 30
        + (p.deleted_nodes.len() as usize) * 10
        + (p.deleted_relationships.len() as usize) * 25
        + attr_bytes;
    buf.reserve(estimated_bytes);

    // Version header (only write once at the start)
    if buf.is_empty() {
        buf.push(EFFECTS_VERSION);
    }

    // --- Schema additions (new labels, relationship types) ---
    {
        let graph = g.borrow();
        let labels = graph.get_labels();
        for label in labels.iter().skip(p.schema_label_count) {
            buf.push(EFFECT_ADD_SCHEMA);
            buf.push(SCHEMA_NODE_LABEL);
            write_string(buf, label);
            n_effects += 1;
        }
        let types = graph.get_types();
        for rel_type in types.iter().skip(p.schema_rel_type_count) {
            buf.push(EFFECT_ADD_SCHEMA);
            buf.push(SCHEMA_REL_TYPE);
            write_string(buf, rel_type);
            n_effects += 1;
        }

        // --- Attribute additions (new node/rel attribute names) ---
        let node_attrs = graph.get_node_attribute_names();
        for attr in node_attrs.iter().skip(p.schema_node_attr_count) {
            buf.push(EFFECT_ADD_ATTRIBUTE);
            buf.push(ATTR_NODE);
            write_string(buf, attr);
            n_effects += 1;
        }
        let rel_attrs = graph.get_relationship_attribute_names();
        for attr in rel_attrs.iter().skip(p.schema_rel_attr_count) {
            buf.push(EFFECT_ADD_ATTRIBUTE);
            buf.push(ATTR_REL);
            write_string(buf, attr);
            n_effects += 1;
        }
    }

    // Attribute keys, label ids, and relationship type ids are encoded
    // as u16 ids; the id → name mapping is established on the replica by
    // the EFFECT_ADD_SCHEMA / EFFECT_ADD_ATTRIBUTE records above (and by
    // in-order replay of earlier queries), which mirror the master's
    // registration order exactly.

    // --- Created nodes ---
    for node_id in &p.created_nodes {
        buf.push(EFFECT_CREATE_NODE);
        buf.extend_from_slice(&node_id.to_le_bytes());

        // Labels
        if let Some(label_ids) = p.set_labels.get(&node_id) {
            write_u16(buf, label_ids.len() as u16);
            for &label_id in label_ids {
                write_u16(buf, label_id as u16);
            }
        } else {
            write_u16(buf, 0);
        }

        // Attributes
        if let Some(attrs) = p.new_nodes_attrs.get(&node_id) {
            write_u16(buf, attrs.len() as u16);
            for (attr_id, value) in attrs {
                write_u16(buf, *attr_id);
                write_value(buf, value);
            }
        } else {
            write_u16(buf, 0);
        }
        n_effects += 1;
    }

    // --- Created relationships ---
    if !p.created_rels_by_type.is_empty() {
        let graph = g.borrow();
        for (type_name, entries) in &p.created_rels_by_type {
            let type_id = graph
                .get_type_id(type_name)
                .expect("created relationship type must be registered")
                .0 as u16;
            for &(rel_id, from, to) in entries {
                buf.push(EFFECT_CREATE_EDGE);
                buf.extend_from_slice(&u64::from(rel_id).to_le_bytes());
                buf.extend_from_slice(&u64::from(from).to_le_bytes());
                buf.extend_from_slice(&u64::from(to).to_le_bytes());
                write_u16(buf, type_id);

                if let Some(attrs) = p.new_relationships_attrs.get(&u64::from(rel_id)) {
                    write_u16(buf, attrs.len() as u16);
                    for (attr_id, value) in attrs {
                        write_u16(buf, *attr_id);
                        write_value(buf, value);
                    }
                } else {
                    write_u16(buf, 0);
                }
                n_effects += 1;
            }
        }
    }

    // --- Updated node attributes (existing nodes only) ---
    for (node_id, attrs) in &p.existing_nodes_attrs {
        buf.push(EFFECT_UPDATE_NODE);
        buf.extend_from_slice(&node_id.to_le_bytes());
        write_u16(buf, attrs.len() as u16);
        for (attr_id, value) in attrs {
            write_u16(buf, *attr_id);
            write_value(buf, value);
        }
        n_effects += 1;
    }

    // --- Updated relationship attributes (existing rels only) ---
    for (rel_id, attrs) in &p.existing_relationships_attrs {
        buf.push(EFFECT_UPDATE_EDGE);
        buf.extend_from_slice(&rel_id.to_le_bytes());
        write_u16(buf, attrs.len() as u16);
        for (attr_id, value) in attrs {
            write_u16(buf, *attr_id);
            write_value(buf, value);
        }
        n_effects += 1;
    }

    // --- Set labels (non-created nodes only) ---
    for (&node_id, label_ids) in &p.set_labels {
        if !p.created_nodes.contains(node_id) {
            buf.push(EFFECT_SET_LABELS);
            buf.extend_from_slice(&node_id.to_le_bytes());
            write_u16(buf, label_ids.len() as u16);
            for &label_id in label_ids {
                write_u16(buf, label_id as u16);
            }
            n_effects += 1;
        }
    }

    // --- Remove labels ---
    for (&node_id, label_ids) in &p.remove_labels {
        buf.push(EFFECT_REMOVE_LABELS);
        buf.extend_from_slice(&node_id.to_le_bytes());
        write_u16(buf, label_ids.len() as u16);
        for &label_id in label_ids {
            write_u16(buf, label_id as u16);
        }
        n_effects += 1;
    }

    // --- Deleted relationships (before nodes, so replica removes edges first) ---
    for &(rel_id, _type_id, from, to) in &p.deleted_endpoints {
        buf.push(EFFECT_DELETE_EDGE);
        buf.extend_from_slice(&u64::from(rel_id).to_le_bytes());
        buf.extend_from_slice(&u64::from(from).to_le_bytes());
        buf.extend_from_slice(&u64::from(to).to_le_bytes());
        n_effects += 1;
    }

    // --- Deleted nodes ---
    for node_id in &p.deleted_nodes {
        buf.push(EFFECT_DELETE_NODE);
        buf.extend_from_slice(&node_id.to_le_bytes());
        n_effects += 1;
    }

    n_effects
}
