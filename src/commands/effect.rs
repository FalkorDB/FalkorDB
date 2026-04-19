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
    graph::graph::{Graph, NodeId, RelationshipId},
    graph::graphblas::matrix::{Matrix, New, Set},
    graph::graphblas::tensor::GrB_INDEX_MAX,
    index::IndexType,
    runtime::{
        ordermap::OrderMap,
        pending::{
            ATTR_NODE, ATTR_REL, EFFECT_ADD_ATTRIBUTE, EFFECT_ADD_SCHEMA, EFFECT_CREATE_EDGE,
            EFFECT_CREATE_INDEX, EFFECT_CREATE_NODE, EFFECT_DELETE_EDGE, EFFECT_DELETE_NODE,
            EFFECT_DROP_INDEX, EFFECT_REMOVE_LABELS, EFFECT_SET_LABELS, EFFECT_UPDATE_EDGE,
            EFFECT_UPDATE_NODE, EFFECTS_VERSION, PendingRelationship, SCHEMA_NODE_LABEL,
            SCHEMA_REL_TYPE, read_string, read_u16, read_u64, read_value,
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
        g
    };

    let mut tg = graph.write();
    let Some(g_arc) = tg.graph.write() else {
        return Err(redis_module::RedisError::String(
            "ERR write lock unavailable".to_string(),
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
            let value = tg.graph.read().borrow().maybe_flush_caches();
            if let Err(e) = value {
                ctx.log_warning(&format!("FalkorDB: cache flush failed: {e}"));
            }
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

    while offset < buf.len() {
        let effect_type = buf[offset];
        offset += 1;

        match effect_type {
            EFFECT_CREATE_NODE => {
                let node_id_raw = read_u64(buf, &mut offset)?;
                g.inc_reserved_node_count();

                // Labels
                let label_count = read_u16(buf, &mut offset)?;
                let mut set_labels = Matrix::new(GrB_INDEX_MAX, GrB_INDEX_MAX);
                for _ in 0..label_count {
                    let label_name = read_string(buf, &mut offset)?;
                    let label_id = g.get_label_id_mut(&label_name);
                    set_labels.set(node_id_raw, label_id.0 as u64, true);
                }

                // Create the node
                let mut nodes = RoaringTreemap::new();
                nodes.insert(node_id_raw);
                g.create_nodes(&nodes);

                // Apply labels
                if label_count > 0 {
                    g.set_nodes_labels(&mut set_labels, &mut index_add_docs);
                }

                // Attributes
                let attr_count = read_u16(buf, &mut offset)?;
                if attr_count > 0 {
                    let attrs = read_attrs(buf, &mut offset, attr_count)?;
                    let mut attr_map = FxHashMap::default();
                    attr_map.insert(node_id_raw, attrs);
                    g.set_nodes_attributes(&attr_map, &mut index_add_docs)?;
                }
            }

            EFFECT_CREATE_EDGE => {
                let rel_id_raw = read_u64(buf, &mut offset)?;
                let src_id = read_u64(buf, &mut offset)?;
                let dst_id = read_u64(buf, &mut offset)?;
                let type_name = read_string(buf, &mut offset)?;

                g.inc_reserved_relationship_count();

                let pending_rel =
                    PendingRelationship::new(NodeId::from(src_id), NodeId::from(dst_id), type_name);
                let mut rels = FxHashMap::default();
                rels.insert(RelationshipId::from(rel_id_raw), pending_rel);
                g.create_relationships(&rels);

                // Attributes
                let attr_count = read_u16(buf, &mut offset)?;
                if attr_count > 0 {
                    let attrs = read_attrs(buf, &mut offset, attr_count)?;
                    let mut attr_map = FxHashMap::default();
                    attr_map.insert(rel_id_raw, attrs);
                    g.set_relationships_attributes(&attr_map, &mut index_add_edge_docs)?;
                }
            }

            EFFECT_UPDATE_NODE => {
                let node_id = read_u64(buf, &mut offset)?;
                let attr_count = read_u16(buf, &mut offset)?;
                let attrs = read_attrs(buf, &mut offset, attr_count)?;
                let mut attr_map = FxHashMap::default();
                attr_map.insert(node_id, attrs);
                g.set_nodes_attributes(&attr_map, &mut index_add_docs)?;
            }

            EFFECT_UPDATE_EDGE => {
                let rel_id = read_u64(buf, &mut offset)?;
                let attr_count = read_u16(buf, &mut offset)?;
                let attrs = read_attrs(buf, &mut offset, attr_count)?;
                let mut attr_map = FxHashMap::default();
                attr_map.insert(rel_id, attrs);
                g.set_relationships_attributes(&attr_map, &mut index_add_edge_docs)?;
            }

            EFFECT_SET_LABELS => {
                let node_id = read_u64(buf, &mut offset)?;
                let label_count = read_u16(buf, &mut offset)?;
                let mut set_labels = Matrix::new(GrB_INDEX_MAX, GrB_INDEX_MAX);
                for _ in 0..label_count {
                    let label_name = read_string(buf, &mut offset)?;
                    let label_id = g.get_label_id_mut(&label_name);
                    set_labels.set(node_id, label_id.0 as u64, true);
                }
                g.set_nodes_labels(&mut set_labels, &mut index_add_docs);
            }

            EFFECT_REMOVE_LABELS => {
                let node_id = read_u64(buf, &mut offset)?;
                let label_count = read_u16(buf, &mut offset)?;
                let mut remove_labels = Matrix::new(GrB_INDEX_MAX, GrB_INDEX_MAX);
                for _ in 0..label_count {
                    let label_name = read_string(buf, &mut offset)?;
                    if let Some(label_id) = g.get_label_id(&label_name) {
                        remove_labels.set(node_id, label_id.0 as u64, true);
                    }
                }
                g.remove_nodes_labels(&mut remove_labels, &mut index_remove_docs);
            }

            EFFECT_DELETE_NODE => {
                let node_id = read_u64(buf, &mut offset)?;
                let mut nodes = RoaringTreemap::new();
                nodes.insert(node_id);
                g.delete_nodes(&nodes, &mut index_remove_docs)?;
            }

            EFFECT_DELETE_EDGE => {
                let rel_id = read_u64(buf, &mut offset)?;
                let src_id = read_u64(buf, &mut offset)?;
                let dst_id = read_u64(buf, &mut offset)?;
                let rel = RelationshipId::from(rel_id);
                let mut rels = FxHashMap::default();
                rels.insert(rel, (NodeId::from(src_id), NodeId::from(dst_id)));
                g.delete_relationships(&rels, &mut index_remove_edge_docs)?;
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
                    }
                    SCHEMA_REL_TYPE => {
                        g.get_type_id_mut(&name);
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
                    ATTR_NODE => g.add_node_attribute_name(&name),
                    ATTR_REL => g.add_rel_attribute_name(&name),
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

    g.commit_attrs()?;
    g.commit_index(&mut index_add_docs, &mut index_remove_docs);
    g.commit_edge_index(&mut index_add_edge_docs, &mut index_remove_edge_docs);

    if has_index_ops {
        g.populate_indexes_sync();
    }

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
) -> Result<OrderMap<Arc<String>, Value>, String> {
    let pairs: Vec<_> = (0..count)
        .map(|_| {
            let key = read_string(buf, offset)?;
            let value = read_value(buf, offset)?;
            Ok((key, value))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(OrderMap::from_vec(pairs))
}
