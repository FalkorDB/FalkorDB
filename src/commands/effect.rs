//! `GRAPH.EFFECT` command handler.
//!
//! Applies serialized effects (mutations) received from the primary to
//! maintain replica consistency.  The binary effects buffer is produced
//! by `Pending::build_effects_buffer()` on the primary and contains the
//! exact mutations that occurred during query execution.
//!
//! ## Command syntax
//! ```text
//! GRAPH.EFFECT <key> <effects_buffer>
//! ```

use crate::{config::CONFIGURATION_CACHE_SIZE, graph_core::ThreadedGraph, redis_type::GRAPH_TYPE};
use graph::{
    entity_type::EntityType,
    graph::graph::{Graph, TypeId},
    index::IndexType,
    runtime::{
        pending::{
            ATTR_NODE, ATTR_REL, EFFECT_ADD_ATTRIBUTE, EFFECT_ADD_SCHEMA, EFFECT_CREATE_EDGE,
            EFFECT_CREATE_INDEX, EFFECT_CREATE_NODE, EFFECT_DELETE_EDGE, EFFECT_DELETE_NODE,
            EFFECT_DROP_INDEX, EFFECT_REMOVE_LABELS, EFFECT_SET_LABELS, EFFECT_UPDATE_EDGE,
            EFFECT_UPDATE_NODE, EFFECTS_VERSION, SCHEMA_NODE_LABEL, SCHEMA_REL_TYPE, read_string,
            read_u16, read_u64, read_value,
        },
        value::Value,
    },
};
use parking_lot::RwLock;
use redis_module::{Context, NextArg, RedisResult, RedisString, RedisValue};
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
use std::sync::Arc;

pub fn graph_effect(
    ctx: &Context,
    args: Vec<RedisString>,
) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let key_str = args.next_arg()?;
    let effects_buf = args.next_arg()?;

    let buf = effects_buf.as_slice();
    if buf.is_empty() {
        return Ok(RedisValue::SimpleStringStatic("OK"));
    }

    // Open existing graph or create a new one
    let key = ctx.open_key_writable(&key_str);
    let graph = if let Some(g) = key.get_value::<Arc<RwLock<ThreadedGraph>>>(&GRAPH_TYPE)? {
        g.clone()
    } else {
        let g = Arc::new(RwLock::new(ThreadedGraph::new(
            *CONFIGURATION_CACHE_SIZE.lock(ctx) as usize,
            &key_str.to_string(),
        )));
        key.set_value(&GRAPH_TYPE, g.clone())?;
        crate::graph_core::register_graph(key_str.to_string(), g.clone());
        g
    };

    let mut tg = graph.write();
    let Some(g_arc) = tg.graph.write() else {
        return Err(redis_module::RedisError::String(
            "ERR another write is in progress, retry the query".to_string(),
        ));
    };

    let result = {
        let mut g = g_arc.borrow_mut();
        apply_effects(&mut g, buf)
    };

    match result {
        Ok(()) => {
            tg.graph.commit(g_arc);
            ctx.replicate_verbatim();
            Ok(RedisValue::SimpleStringStatic("OK"))
        }
        Err(e) => {
            tg.graph.rollback();
            Err(redis_module::RedisError::String(format!(
                "ERR effect apply failed: {e}"
            )))
        }
    }
}

#[allow(clippy::too_many_lines)]
fn apply_effects(
    g: &mut Graph,
    buf: &[u8],
) -> Result<(), String> {
    let mut offset = 0;

    if offset >= buf.len() {
        return Err("empty effects buffer".to_string());
    }
    let version = buf[offset];
    offset += 1;
    if version != EFFECTS_VERSION {
        return Err(format!("unsupported effects version: {version}"));
    }

    let mut index_add_docs: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
    let mut index_remove_docs: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
    let mut index_add_edge_docs: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
    let mut index_remove_edge_docs: FxHashMap<u64, FxHashMap<u64, (u64, u64)>> =
        FxHashMap::default();
    let mut has_index_ops = false;

    // Entity effects reference labels, relationship types, and attributes by
    // u16 id. The replica's id space matches the master's because schema
    // registrations are replicated in order (EFFECT_ADD_SCHEMA /
    // EFFECT_ADD_ATTRIBUTE below, or verbatim query replay). Track current
    // counts to validate ids against corrupt or out-of-order buffers.
    let mut label_count_g = g.get_labels().len();
    let mut type_count_g = g.get_types().len();
    let mut node_attr_count = g.get_node_attribute_names().len();
    let mut rel_attr_count = g.get_relationship_attribute_names().len();

    // Deletions are batched: the writer emits them as contiguous runs (edges
    // then nodes, see `Pending::build_effects_buffer`), and both graph-side
    // deletes are bulk operations whose per-call overhead — a full-size
    // GraphBLAS mask per matrix, a scan of `node_labels_matrix` — is paid once
    // per call, not once per entity. Applying a node's edges one record at a
    // time made a replica ~40x slower than the master that produced the writes.
    // Mirrors C's `ApplyDeleteNode` / `ApplyDeleteEdge` stream look-ahead.
    let mut del_nodes = RoaringTreemap::new();
    let mut del_edges = RoaringTreemap::new();

    while offset < buf.len() {
        let effect_type = buf[offset];
        offset += 1;

        // Flush a pending batch as soon as the run of same-type records ends, so
        // effects still take hold in stream order. Only one batch is ever
        // non-empty: starting one flushes the other.
        match effect_type {
            EFFECT_DELETE_EDGE => flush_del_nodes(g, &mut del_nodes, &mut index_remove_docs)?,
            EFFECT_DELETE_NODE => {
                flush_del_edges(g, &mut del_edges, &mut index_remove_edge_docs)?;
            }
            _ => {
                flush_del_edges(g, &mut del_edges, &mut index_remove_edge_docs)?;
                flush_del_nodes(g, &mut del_nodes, &mut index_remove_docs)?;
            }
        }

        match effect_type {
            EFFECT_CREATE_NODE => {
                let node_id_raw = read_u64(buf, &mut offset)?;
                g.inc_reserved_node_count();

                // Labels
                let label_count = read_u16(buf, &mut offset)?;
                let mut label_rows = Vec::with_capacity(label_count as usize);
                let mut label_cols = Vec::with_capacity(label_count as usize);
                for _ in 0..label_count {
                    let label_id = read_u16(buf, &mut offset)?;
                    if label_id as usize >= label_count_g {
                        return Err(format!("label id {label_id} out of range"));
                    }
                    label_rows.push(node_id_raw);
                    label_cols.push(label_id as u64);
                }

                // Create the node
                let mut nodes = RoaringTreemap::new();
                nodes.insert(node_id_raw);
                g.create_nodes(&nodes);

                // Apply labels
                if label_count > 0 {
                    g.set_nodes_labels_bulk(&label_rows, &label_cols, &mut index_add_docs, true);
                }

                // Attributes
                let attr_count = read_u16(buf, &mut offset)?;
                if attr_count > 0 {
                    let attrs = read_attrs(buf, &mut offset, attr_count, node_attr_count)?;
                    let mut attr_map = FxHashMap::default();
                    attr_map.insert(node_id_raw, attrs);
                    g.set_nodes_attributes(&attr_map, &mut index_add_docs)?;
                }
            }

            EFFECT_CREATE_EDGE => {
                let rel_id_raw = read_u64(buf, &mut offset)?;
                let src_id = read_u64(buf, &mut offset)?;
                let dst_id = read_u64(buf, &mut offset)?;
                let type_id = read_u16(buf, &mut offset)?;
                if type_id as usize >= type_count_g {
                    return Err(format!("relationship type id {type_id} out of range"));
                }
                let type_name = g
                    .get_type(TypeId(type_id as usize))
                    .ok_or_else(|| format!("unknown relationship type id {type_id}"))?;

                g.inc_reserved_relationship_count();

                g.create_relationships_bulk(&type_name, &[src_id], &[dst_id], &[rel_id_raw]);

                // Attributes
                let attr_count = read_u16(buf, &mut offset)?;
                if attr_count > 0 {
                    let attrs = read_attrs(buf, &mut offset, attr_count, rel_attr_count)?;
                    let mut attr_map = FxHashMap::default();
                    attr_map.insert(rel_id_raw, attrs);
                    g.set_relationships_attributes(&attr_map, &mut index_add_edge_docs)?;
                }
            }

            EFFECT_UPDATE_NODE => {
                let node_id = read_u64(buf, &mut offset)?;
                let attr_count = read_u16(buf, &mut offset)?;
                let attrs = read_attrs(buf, &mut offset, attr_count, node_attr_count)?;
                let mut attr_map = FxHashMap::default();
                attr_map.insert(node_id, attrs);
                g.set_nodes_attributes(&attr_map, &mut index_add_docs)?;
            }

            EFFECT_UPDATE_EDGE => {
                let rel_id = read_u64(buf, &mut offset)?;
                let attr_count = read_u16(buf, &mut offset)?;
                let attrs = read_attrs(buf, &mut offset, attr_count, rel_attr_count)?;
                let mut attr_map = FxHashMap::default();
                attr_map.insert(rel_id, attrs);
                g.set_relationships_attributes(&attr_map, &mut index_add_edge_docs)?;
            }

            EFFECT_SET_LABELS => {
                let node_id = read_u64(buf, &mut offset)?;
                let label_count = read_u16(buf, &mut offset)?;
                let mut label_rows = Vec::with_capacity(label_count as usize);
                let mut label_cols = Vec::with_capacity(label_count as usize);
                for _ in 0..label_count {
                    let label_id = read_u16(buf, &mut offset)?;
                    if label_id as usize >= label_count_g {
                        return Err(format!("label id {label_id} out of range"));
                    }
                    label_rows.push(node_id);
                    label_cols.push(label_id as u64);
                }
                g.set_nodes_labels_bulk(&label_rows, &label_cols, &mut index_add_docs, false);
            }

            EFFECT_REMOVE_LABELS => {
                let node_id = read_u64(buf, &mut offset)?;
                let label_count = read_u16(buf, &mut offset)?;
                let mut label_rows = Vec::with_capacity(label_count as usize);
                let mut label_cols = Vec::with_capacity(label_count as usize);
                for _ in 0..label_count {
                    let label_id = read_u16(buf, &mut offset)?;
                    if label_id as usize >= label_count_g {
                        return Err(format!("label id {label_id} out of range"));
                    }
                    label_rows.push(node_id);
                    label_cols.push(label_id as u64);
                }
                g.remove_nodes_labels(&label_rows, &label_cols, &mut index_remove_docs);
            }

            EFFECT_DELETE_NODE => {
                del_nodes.insert(read_u64(buf, &mut offset)?);
            }

            EFFECT_DELETE_EDGE => {
                del_edges.insert(read_u64(buf, &mut offset)?);
                let _src_id = read_u64(buf, &mut offset)?;
                let _dst_id = read_u64(buf, &mut offset)?;
            }

            EFFECT_ADD_SCHEMA => {
                if offset >= buf.len() {
                    return Err("truncated EFFECT_ADD_SCHEMA".to_string());
                }
                let schema_type = buf[offset];
                offset += 1;
                let name = read_string(buf, &mut offset)?;
                match schema_type {
                    SCHEMA_NODE_LABEL => {
                        g.get_label_id_mut(&name);
                        label_count_g = g.get_labels().len();
                    }
                    SCHEMA_REL_TYPE => {
                        g.get_type_id_mut(&name);
                        type_count_g = g.get_types().len();
                    }
                    _ => return Err(format!("unknown schema type: {schema_type}")),
                }
            }

            EFFECT_ADD_ATTRIBUTE => {
                if offset >= buf.len() {
                    return Err("truncated EFFECT_ADD_ATTRIBUTE".to_string());
                }
                let attr_type = buf[offset];
                offset += 1;
                let name = read_string(buf, &mut offset)?;
                match attr_type {
                    ATTR_NODE => {
                        g.add_node_attribute_name(&name);
                        node_attr_count = g.get_node_attribute_names().len();
                    }
                    ATTR_REL => {
                        g.add_rel_attribute_name(&name);
                        rel_attr_count = g.get_relationship_attribute_names().len();
                    }
                    _ => return Err(format!("unknown attribute type: {attr_type}")),
                }
            }

            EFFECT_CREATE_INDEX => {
                let index_type = read_index_type(buf, &mut offset)?;
                let entity_type = read_entity_type(buf, &mut offset)?;
                let label = read_string(buf, &mut offset)?;
                let attr_count = read_u16(buf, &mut offset)?;
                let mut attrs = Vec::with_capacity(attr_count as usize);
                for _ in 0..attr_count {
                    attrs.push(read_string(buf, &mut offset)?);
                }
                // Use sync variant to avoid spawning async population threads on the replica
                g.create_index_sync(&index_type, &entity_type, &label, &attrs, None)?;
                has_index_ops = true;
            }

            EFFECT_DROP_INDEX => {
                let index_type = read_index_type(buf, &mut offset)?;
                let entity_type = read_entity_type(buf, &mut offset)?;
                let label = read_string(buf, &mut offset)?;
                let attr_count = read_u16(buf, &mut offset)?;
                let mut attrs = Vec::with_capacity(attr_count as usize);
                for _ in 0..attr_count {
                    attrs.push(read_string(buf, &mut offset)?);
                }
                g.drop_index(&index_type, &entity_type, &label, &attrs)?;
            }

            _ => return Err(format!("unknown effect type: {effect_type}")),
        }
    }

    flush_del_edges(g, &mut del_edges, &mut index_remove_edge_docs)?;
    flush_del_nodes(g, &mut del_nodes, &mut index_remove_docs)?;

    g.commit_index(&mut index_add_docs, &mut index_remove_docs);
    g.commit_edge_index(&mut index_add_edge_docs, &mut index_remove_edge_docs);

    if has_index_ops {
        g.populate_indexes_sync();
    }

    Ok(())
}

/// Apply a batch of `EFFECT_DELETE_NODE` ids, leaving the batch empty.
fn flush_del_nodes(
    g: &mut Graph,
    nodes: &mut RoaringTreemap,
    index_remove_docs: &mut FxHashMap<u64, RoaringTreemap>,
) -> Result<(), String> {
    if nodes.is_empty() {
        return Ok(());
    }
    g.delete_nodes(nodes, index_remove_docs)?;
    nodes.clear();
    Ok(())
}

/// Apply a batch of `EFFECT_DELETE_EDGE` ids, leaving the batch empty.
fn flush_del_edges(
    g: &mut Graph,
    rels: &mut RoaringTreemap,
    index_remove_edge_docs: &mut FxHashMap<u64, FxHashMap<u64, (u64, u64)>>,
) -> Result<(), String> {
    if rels.is_empty() {
        return Ok(());
    }
    g.delete_relationships(rels, index_remove_edge_docs)?;
    rels.clear();
    Ok(())
}

fn read_index_type(
    buf: &[u8],
    offset: &mut usize,
) -> Result<IndexType, String> {
    if *offset >= buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let tag = buf[*offset];
    *offset += 1;
    match tag {
        0 => Ok(IndexType::Range),
        1 => Ok(IndexType::Fulltext),
        2 => Ok(IndexType::Vector),
        _ => Err(format!("unknown index type tag: {tag}")),
    }
}

fn read_entity_type(
    buf: &[u8],
    offset: &mut usize,
) -> Result<EntityType, String> {
    if *offset >= buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let tag = buf[*offset];
    *offset += 1;
    match tag {
        0 => Ok(EntityType::Node),
        1 => Ok(EntityType::Relationship),
        _ => Err(format!("unknown entity type tag: {tag}")),
    }
}

fn read_attrs(
    buf: &[u8],
    offset: &mut usize,
    count: u16,
    attr_id_bound: usize,
) -> Result<Vec<(u16, Value)>, String> {
    // Attribute ids arrive id-sorted from the writer (pending stores are
    // id-sorted), matching what the attribute stores expect.
    let mut pairs: Vec<(u16, Value)> = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let id = read_u16(buf, offset)?;
        if id as usize >= attr_id_bound {
            return Err(format!("attribute id {id} out of range"));
        }
        let value = read_value(buf, offset)?;
        pairs.push((id, value));
    }
    debug_assert!(pairs.is_sorted_by_key(|(k, _)| *k));
    Ok(pairs)
}
