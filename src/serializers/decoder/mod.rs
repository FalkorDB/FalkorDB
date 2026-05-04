use std::sync::Arc;

use graph::entity_type::EntityType;
use graph::graph::attribute_store::AttributeStore;
use graph::graph::graph::Graph;
use graph::graph::graphblas::matrix::New;
use graph::graph::graphblas::serialization::{Decode, Reader};
use graph::graph::graphblas::tensor::Tensor;
use graph::graph::graphblas::versioned_matrix::VersionedMatrix;
use graph::index::IndexInfo;
use redis_module::RedisModuleIO;
use roaring::RoaringTreemap;

use super::EncodeState;
use super::Header;
use super::Schema;
use super::buffered_io::BufferedReader;
use super::{DECODE_STATE, PendingGraph};

/// Decode a graph key from the RDB stream (v19 format).
///
/// Returns `Ok(Some(graph))` for single-key graphs (key_count == 1),
/// or `Ok(None)` when the key data has been accumulated into
/// `DECODE_STATE` for multi-key graphs (key_count > 1).
#[allow(clippy::too_many_lines)]
pub fn rdb_load_graph(
    rdb: *mut RedisModuleIO,
    cache_size: usize,
) -> Result<Option<Graph>, String> {
    let mut r = BufferedReader::new(rdb);

    // --- Header ---
    let hdr = Header::decode(&mut r)?;

    // --- Schema ---
    let schema = Schema::decode(&mut r)?;

    // --- Key Schema (payload directory) ---
    let payload_count = r.read_unsigned()?;
    let mut payloads = Vec::with_capacity(payload_count as usize);
    for _ in 0..payload_count {
        let state = r.read_unsigned()?;
        let count = r.read_unsigned()?;
        let state =
            EncodeState::from_u64(state).ok_or_else(|| format!("unknown encode state: {state}"))?;
        payloads.push((state, count));
    }

    // For multi-key graphs, check if we already have a pending graph in DECODE_STATE.
    if hdr.key_count > 1 {
        let mut decode_state = DECODE_STATE.lock();
        let is_first_key = !decode_state.pending.contains_key(&hdr.graph_name);

        if is_first_key {
            // First key: initialize the pending graph.
            let node_attrs = AttributeStore::new(&format!("{}/nodes", hdr.graph_name), 0);
            let mut rel_attrs =
                AttributeStore::new(&format!("{}/relationships", hdr.graph_name), 0);

            // Set attribute names on the stores now -- they are the same across all keys.
            let mut node_attrs_init = node_attrs;
            for name in &schema.attribute_names {
                node_attrs_init.attrs_name.insert(name.clone());
                rel_attrs.attrs_name.insert(name.clone());
            }

            let pg = PendingGraph {
                keys_remaining: hdr.key_count - 1, // this key + remaining
                cache_size,
                header: Header {
                    graph_name: hdr.graph_name.clone(),
                    node_count: hdr.node_count,
                    edge_count: hdr.edge_count,
                    deleted_node_count: hdr.deleted_node_count,
                    deleted_edge_count: hdr.deleted_edge_count,
                    label_count: hdr.label_count,
                    relationship_count: hdr.relationship_count,
                    multi_edge: hdr.multi_edge.clone(),
                    key_count: hdr.key_count,
                },
                schema: Schema {
                    attribute_names: schema.attribute_names.clone(),
                    node_labels: schema.node_labels.clone(),
                    relationship_types: schema.relationship_types.clone(),
                    indexes: schema.indexes,
                    constraints: schema.constraints,
                },
                node_attrs: node_attrs_init,
                rel_attrs,
                deleted_nodes: RoaringTreemap::new(),
                deleted_rels: RoaringTreemap::new(),
                label_matrices: Vec::new(),
                relationship_tensors: Vec::new(),
                adj_matrix: VersionedMatrix::new(0, 0),
                lbls_matrix: VersionedMatrix::new(0, 0),
            };
            decode_state.pending.insert(hdr.graph_name.clone(), pg);
        } else {
            // Subsequent key: just decrement keys_remaining.
            if let Some(pg) = decode_state.pending.get_mut(&hdr.graph_name) {
                pg.keys_remaining -= 1;
            }
        }

        // Decode this key's payloads into the pending graph.
        {
            let pg = decode_state.pending.get_mut(&hdr.graph_name).unwrap();
            decode_payloads_into_pending(&mut r, &payloads, pg, &hdr)?;
        }

        // If all keys have been loaded, finalize immediately.
        // This avoids depending on aux_load ordering between module types.
        let should_finalize = {
            let pg = decode_state.pending.get(&hdr.graph_name).unwrap();
            pg.keys_remaining == 0
        };
        if should_finalize {
            let graph_name = hdr.graph_name.clone();
            let pg = decode_state.pending.remove(&graph_name).unwrap();
            let graph = finalize_pending_graph(pg)?;
            // Store the finalized graph in DECODE_STATE for the caller to retrieve.
            decode_state.finalized.insert(graph_name, graph);
        }

        return Ok(None);
    }

    // Single-key path (key_count == 1): decode everything in one go.
    let mut node_attrs = AttributeStore::new(&format!("{}/nodes", hdr.graph_name), 0);
    let mut rel_attrs = AttributeStore::new(&format!("{}/relationships", hdr.graph_name), 0);

    for name in &schema.attribute_names {
        node_attrs.attrs_name.insert(name.clone());
        rel_attrs.attrs_name.insert(name.clone());
    }

    let mut deleted_nodes = RoaringTreemap::new();
    let mut deleted_rels = RoaringTreemap::new();
    let mut label_matrices: Vec<VersionedMatrix> = Vec::new();
    let mut relationship_tensors: Vec<Tensor> = Vec::new();
    let mut adj_matrix = VersionedMatrix::new(0, 0);
    let mut lbls_matrix = VersionedMatrix::new(0, 0);

    for (state, count) in &payloads {
        match *state {
            EncodeState::Nodes => {
                node_attrs.decode_with_count(&mut r, *count)?;
            }
            EncodeState::DeletedNodes => {
                deleted_nodes.decode_with_count(&mut r, *count)?;
            }
            EncodeState::Edges => {
                rel_attrs.decode_with_count(&mut r, *count)?;
            }
            EncodeState::DeletedEdges => {
                deleted_rels.decode_with_count(&mut r, *count)?;
            }
            EncodeState::LabelsMatrices => {
                let count = r.read_unsigned()?;
                for _ in 0..count {
                    let _label_id = r.read_unsigned()?;
                    label_matrices.push(VersionedMatrix::decode(&mut r)?);
                }
            }
            EncodeState::RelationMatrices => {
                for _ in 0..hdr.relationship_count {
                    let _relation_id = r.read_unsigned()?;
                    relationship_tensors.push(Tensor::decode(&mut r)?);
                }
            }
            EncodeState::AdjMatrix => {
                adj_matrix = VersionedMatrix::decode(&mut r)?;
            }
            EncodeState::LblsMatrix => {
                lbls_matrix = VersionedMatrix::decode(&mut r)?;
            }
            _ => {}
        }
    }

    node_attrs
        .commit()
        .map_err(|e| format!("commit node attrs: {e}"))?;
    rel_attrs
        .commit()
        .map_err(|e| format!("commit rel attrs: {e}"))?;

    let mut graph = Graph::restore(
        &hdr.graph_name,
        cache_size,
        hdr.node_count,
        hdr.edge_count,
        deleted_nodes,
        deleted_rels,
        adj_matrix,
        lbls_matrix,
        VersionedMatrix::new(0, 0),
        VersionedMatrix::new(0, 0),
        label_matrices,
        relationship_tensors,
        schema.node_labels,
        schema.relationship_types,
        node_attrs,
        rel_attrs,
    );

    graph.rebuild_derived_matrices();
    rebuild_indexes(&mut graph, &schema.indexes);
    for c in schema.constraints {
        graph.add_constraint_raw(c);
    }
    graph.populate_indexes_sync();

    Ok(Some(graph))
}

/// Decode payload data from the RDB stream into a pending multi-key graph.
fn decode_payloads_into_pending(
    r: &mut BufferedReader,
    payloads: &[(EncodeState, u64)],
    pg: &mut PendingGraph,
    hdr: &Header,
) -> Result<(), String> {
    for (state, count) in payloads {
        match *state {
            EncodeState::Nodes => {
                pg.node_attrs.decode_with_count(r, *count)?;
            }
            EncodeState::DeletedNodes => {
                pg.deleted_nodes.decode_with_count(r, *count)?;
            }
            EncodeState::Edges => {
                pg.rel_attrs.decode_with_count(r, *count)?;
            }
            EncodeState::DeletedEdges => {
                pg.deleted_rels.decode_with_count(r, *count)?;
            }
            EncodeState::LabelsMatrices => {
                let count = r.read_unsigned()?;
                for _ in 0..count {
                    let _label_id = r.read_unsigned()?;
                    pg.label_matrices.push(VersionedMatrix::decode(r)?);
                }
            }
            EncodeState::RelationMatrices => {
                for _ in 0..hdr.relationship_count {
                    let _relation_id = r.read_unsigned()?;
                    pg.relationship_tensors.push(Tensor::decode(r)?);
                }
            }
            EncodeState::AdjMatrix => {
                pg.adj_matrix = VersionedMatrix::decode(r)?;
            }
            EncodeState::LblsMatrix => {
                pg.lbls_matrix = VersionedMatrix::decode(r)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Finalize a pending multi-key graph: commit attrs, build Graph, rebuild derived matrices.
pub fn finalize_pending_graph(pg: PendingGraph) -> Result<Graph, String> {
    let mut node_attrs = pg.node_attrs;
    let mut rel_attrs = pg.rel_attrs;

    node_attrs
        .commit()
        .map_err(|e| format!("commit node attrs: {e}"))?;
    rel_attrs
        .commit()
        .map_err(|e| format!("commit rel attrs: {e}"))?;

    let mut graph = Graph::restore(
        &pg.header.graph_name,
        pg.cache_size,
        pg.header.node_count,
        pg.header.edge_count,
        pg.deleted_nodes,
        pg.deleted_rels,
        pg.adj_matrix,
        pg.lbls_matrix,
        VersionedMatrix::new(0, 0),
        VersionedMatrix::new(0, 0),
        pg.label_matrices,
        pg.relationship_tensors,
        pg.schema.node_labels,
        pg.schema.relationship_types,
        node_attrs,
        rel_attrs,
    );

    graph.rebuild_derived_matrices();
    rebuild_indexes(&mut graph, &pg.schema.indexes);
    for c in pg.schema.constraints {
        graph.add_constraint_raw(c);
    }
    graph.populate_indexes_sync();

    Ok(graph)
}

/// Rebuild indexes from the decoded schema information.
fn rebuild_indexes(
    graph: &mut Graph,
    indexes: &[IndexInfo],
) {
    for info in indexes {
        // `IndexInfo::entity_type` is set to `"NODE"` or
        // `"RELATIONSHIP"` by `Graph::index_info` when the schema is
        // captured on encode — honor it here so an edge index doesn't
        // get rebuilt as a node index on decode (which would also
        // collide with any node label that shares the name).
        let entity_type = match info.entity_type.as_str() {
            "RELATIONSHIP" => EntityType::Relationship,
            _ => EntityType::Node,
        };

        for (attr_name, fields) in &info.fields {
            for field in fields {
                // The Field.name includes the type prefix (e.g. "range:val").
                // The attr_name key in the HashMap is the raw attribute name.
                let attr = Arc::new(attr_name.to_string());

                let options = field.vector_options().map_or_else(
                    || {
                        field
                            .options()
                            .map(|topts| graph::index::indexer::IndexOptions::Text(topts.clone()))
                    },
                    |vopts| Some(graph::index::indexer::IndexOptions::Vector(vopts.clone())),
                );

                if let Err(e) = graph.create_index_sync(
                    &field.ty,
                    &entity_type,
                    &info.label,
                    &vec![attr],
                    options,
                ) {
                    eprintln!("FalkorDB: failed to rebuild index on {:?}: {e}", info.label);
                }
            }
        }
    }
}

/// Decode a graph from a pipe fd (for GRAPH.COPY fork+pipe).
///
/// Same single-key decode logic as `rdb_load_graph` but reads from a pipe
/// via `PipeReader` and overrides the graph name to `dest_name`.
pub fn pipe_load_graph(
    fd: std::os::unix::io::OwnedFd,
    cache_size: usize,
    dest_name: &str,
) -> Result<Graph, String> {
    let mut r = super::buffered_io::PipeReader::new(fd);
    let graph = load_graph_from_reader(&mut r, cache_size, dest_name)?;
    r.close();
    Ok(graph)
}

/// Decode a graph from a byte buffer (for GRAPH.RESTORE on replicas).
///
/// Uses `BufferedReader::from_slice` to read the v19-encoded data produced by
/// `vec_save_graph`.
pub fn vec_load_graph(
    data: &[u8],
    cache_size: usize,
    dest_name: &str,
) -> Result<Graph, String> {
    let mut r = BufferedReader::from_slice(data);
    load_graph_from_reader(&mut r, cache_size, dest_name)
}

/// Shared single-key decode logic used by both `pipe_load_graph` and `vec_load_graph`.
fn load_graph_from_reader(
    r: &mut dyn Reader,
    cache_size: usize,
    dest_name: &str,
) -> Result<Graph, String> {
    let hdr = Header::decode(r)?;

    if hdr.key_count != 1 {
        return Err(format!(
            "expected single-key payload, found {} keys",
            hdr.key_count
        ));
    }

    let schema = Schema::decode(r)?;

    let payload_count = r.read_unsigned()?;
    let mut payloads = Vec::with_capacity(payload_count as usize);
    for _ in 0..payload_count {
        let state = r.read_unsigned()?;
        let count = r.read_unsigned()?;
        let state =
            EncodeState::from_u64(state).ok_or_else(|| format!("unknown encode state: {state}"))?;
        payloads.push((state, count));
    }

    let mut node_attrs = AttributeStore::new(&format!("{dest_name}/nodes"), 0);
    let mut rel_attrs = AttributeStore::new(&format!("{dest_name}/relationships"), 0);

    for name in &schema.attribute_names {
        node_attrs.attrs_name.insert(name.clone());
        rel_attrs.attrs_name.insert(name.clone());
    }

    let mut deleted_nodes = RoaringTreemap::new();
    let mut deleted_rels = RoaringTreemap::new();
    let mut label_matrices: Vec<VersionedMatrix> = Vec::new();
    let mut relationship_tensors: Vec<Tensor> = Vec::new();
    let mut adj_matrix = VersionedMatrix::new(0, 0);
    let mut lbls_matrix = VersionedMatrix::new(0, 0);

    for (state, count) in &payloads {
        match *state {
            EncodeState::Nodes => {
                node_attrs.decode_with_count(r, *count)?;
            }
            EncodeState::DeletedNodes => {
                deleted_nodes.decode_with_count(r, *count)?;
            }
            EncodeState::Edges => {
                rel_attrs.decode_with_count(r, *count)?;
            }
            EncodeState::DeletedEdges => {
                deleted_rels.decode_with_count(r, *count)?;
            }
            EncodeState::LabelsMatrices => {
                let count = r.read_unsigned()?;
                for _ in 0..count {
                    let _label_id = r.read_unsigned()?;
                    label_matrices.push(VersionedMatrix::decode(r)?);
                }
            }
            EncodeState::RelationMatrices => {
                for _ in 0..hdr.relationship_count {
                    let _relation_id = r.read_unsigned()?;
                    relationship_tensors.push(Tensor::decode(r)?);
                }
            }
            EncodeState::AdjMatrix => {
                adj_matrix = VersionedMatrix::decode(r)?;
            }
            EncodeState::LblsMatrix => {
                lbls_matrix = VersionedMatrix::decode(r)?;
            }
            _ => {}
        }
    }

    node_attrs
        .commit()
        .map_err(|e| format!("commit node attrs: {e}"))?;
    rel_attrs
        .commit()
        .map_err(|e| format!("commit rel attrs: {e}"))?;

    let mut graph = Graph::restore(
        dest_name,
        cache_size,
        hdr.node_count,
        hdr.edge_count,
        deleted_nodes,
        deleted_rels,
        adj_matrix,
        lbls_matrix,
        VersionedMatrix::new(0, 0),
        VersionedMatrix::new(0, 0),
        label_matrices,
        relationship_tensors,
        schema.node_labels,
        schema.relationship_types,
        node_attrs,
        rel_attrs,
    );

    graph.rebuild_derived_matrices();
    rebuild_indexes(&mut graph, &schema.indexes);
    for c in schema.constraints {
        graph.add_constraint_raw(c);
    }
    graph.populate_indexes_sync();

    Ok(graph)
}
