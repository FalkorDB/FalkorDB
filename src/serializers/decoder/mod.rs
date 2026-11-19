use std::{collections::HashSet, sync::Arc};

use graph::entity_type::EntityType;
use graph::graph::graph::Graph;
use graph::graph::graphblas::serialization::{Decode, Reader};
use graph::graph::graphblas::tensor::Tensor;
use graph::graph::graphblas::versioned_matrix::VersionedMatrix;
use graph::graph::{
    attribute_store::{AttrNameMap, AttributeStore},
    schema::NameList,
};
use graph::index::IndexInfo;
use roaring::RoaringTreemap;
use valkey_module::raw::RedisModuleIO;

use super::EncodeState;
use super::Header;
use super::Schema;
use super::buffered_io::BufferedReader;
use super::{DECODE_MIN_VERSION, ENCODING_VERSION};
use super::{DECODE_STATE, PendingGraph};

/// What one RDB key turned out to hold.
pub enum LoadedKey {
    /// The whole graph: the key stood alone (`key_count == 1`).
    Graph(Box<Graph>),
    /// A slice of a multi-key graph, accumulated into `DECODE_STATE`.
    Partial {
        /// True when this key is one of the graph's virtual keys rather than
        /// the key the graph itself lives at. Such a key is bookkeeping: it
        /// must not be registered as a graph, and it leaves the keyspace once
        /// the load ends.
        is_virtual: bool,
    },
}

/// Read a key's payload directory: the `(EncodeState, count)` pairs it holds.
///
/// Encoding version 20 is version 19 plus one payload, `EncodeState::CchIndices`, which C
/// writes only for a graph that has a CCH index; any other v20 key is a v19 key. This module
/// has no CCH index, so it refuses a key that lists that payload: the load fails with an
/// error naming it, before any payload is read. (Before, 10 decoded as `Final`, its payload
/// was never read, and the load went on from the wrong place in the stream.)
///
/// `Init` and `Final` are never written to an RDB, by C or by this module; a directory that
/// lists one is refused too, since no payload arm reads them and the stream would misalign
/// the same way.
///
/// # Errors
/// A truncated stream, an unknown state, `Init`, `Final`, or a CCH index payload.
fn decode_payload_directory(r: &mut dyn Reader) -> Result<Vec<(EncodeState, u64)>, String> {
    let payload_count = r.read_unsigned()?;
    // No capacity from `payload_count`: it is read from the stream, unchecked.
    let mut payloads = Vec::new();
    for _ in 0..payload_count {
        let state = r.read_unsigned()?;
        let count = r.read_unsigned()?;
        let state =
            EncodeState::from_u64(state).ok_or_else(|| format!("unknown encode state: {state}"))?;
        match state {
            EncodeState::CchIndices => {
                return Err(
                    "RDB v20 graph with CCH indexes: this module does not support CCH indexes"
                        .to_string(),
                );
            }
            EncodeState::Init | EncodeState::Final => {
                return Err(format!("encode state {state:?} is never saved to an RDB"));
            }
            _ => {}
        }
        payloads.push((state, count));
    }
    Ok(payloads)
}

/// Refuse a key whose encoding version the decoder cannot read, as C's `rdb_load` does.
///
/// A newer version may lay its payloads out in a way this decoder does not know, and an
/// older one predates the v19 layout. Either would be decoded as v19 and misread, so the
/// load fails before anything is read.
///
/// # Errors
/// `encver` is above [`ENCODING_VERSION`] or below [`DECODE_MIN_VERSION`].
pub fn check_encoding_version(encver: i32) -> Result<(), String> {
    if encver > ENCODING_VERSION {
        return Err(format!(
            "RDB encoding version {encver} is newer than this module's \
             {ENCODING_VERSION}: not forward compatible"
        ));
    }
    if encver < DECODE_MIN_VERSION {
        return Err(format!(
            "RDB encoding version {encver} is older than the oldest this module \
             loads, {DECODE_MIN_VERSION}"
        ));
    }
    Ok(())
}

/// Decode a graph key from the RDB stream (v19 format, or v20 without CCH indexes).
///
/// `encver` is the key's encoding version, which Redis passes to `rdb_load`; see
/// [`check_encoding_version`].
///
/// `key_name` is the Redis key this payload arrived under. A multi-key graph
/// spans several keys but names itself only once, in every payload's header,
/// so the key whose name matches is the graph's own and the rest are its
/// virtual keys — see [`DecodeState::meta_keys`](super::DecodeState::meta_keys).
///
/// Returns [`LoadedKey::Graph`] for single-key graphs (key_count == 1), or
/// [`LoadedKey::Partial`] when the key data has been accumulated into
/// `DECODE_STATE` for multi-key graphs (key_count > 1).
#[allow(clippy::too_many_lines)]
pub fn rdb_load_graph(
    rdb: *mut RedisModuleIO,
    encver: i32,
    key_name: &str,
    cache_size: usize,
) -> Result<LoadedKey, String> {
    check_encoding_version(encver)?;
    let mut r = BufferedReader::new(rdb);

    // --- Header ---
    let hdr = Header::decode(&mut r)?;

    // --- Schema ---
    let schema = Schema::decode(&mut r)?;

    // --- Key Schema (payload directory) ---
    let payloads = decode_payload_directory(&mut r)?;

    // For multi-key graphs, check if we already have a pending graph in DECODE_STATE.
    if hdr.key_count > 1 {
        let mut decode_state = DECODE_STATE.lock();

        // C's `decode_graph.c`: "the virtual key name is not equal the graph
        // name". Everything this key contributes lands in the state keyed by
        // `hdr.graph_name`, and the finished graph is installed under that
        // name, so any other key holding a slice of it is a virtual key and
        // has to be deleted once the load ends.
        if key_name != hdr.graph_name {
            decode_state.meta_keys.push(key_name.to_string());
        }

        let is_first_key = !decode_state.pending.contains_key(&hdr.graph_name);

        if is_first_key {
            // First key: initialize the pending graph.
            let node_attrs_init = AttributeStore::new();
            let rel_attrs = AttributeStore::new();

            // The RDB's flat `attribute_names` list *is* the graph's dictionary: index is
            // the id. Previously this seeded a copy into each store, which is what gave an
            // RDB-loaded replica different numbering from a master that had never been
            // reloaded (#2457).
            let mut attrs_name = AttrNameMap::PropertyKeys(NameList::default());
            for name in &schema.attribute_names {
                attrs_name.insert(name);
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
                attrs_name,
                node_attrs: node_attrs_init,
                rel_attrs,
                deleted_nodes: RoaringTreemap::new(),
                deleted_rels: RoaringTreemap::new(),
                label_matrices: Vec::new(),
                relationship_tensors: Vec::new(),
                adj_matrix: VersionedMatrix::<bool>::new(0, 0),
                lbls_matrix: VersionedMatrix::<bool>::new(0, 0),
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
            let pg = decode_state
                .pending
                .get_mut(&hdr.graph_name)
                .ok_or_else(|| {
                    format!(
                        "pending graph {} not found while loading multi-key RDB payload",
                        hdr.graph_name
                    )
                })?;
            decode_payloads_into_pending(&mut r, &payloads, pg, &hdr)?;
        }

        // If all keys have been loaded, finalize immediately.
        // This avoids depending on aux_load ordering between module types.
        let should_finalize = {
            let pg = decode_state.pending.get(&hdr.graph_name).ok_or_else(|| {
                format!(
                    "pending graph {} disappeared during multi-key RDB load",
                    hdr.graph_name
                )
            })?;
            pg.keys_remaining == 0
        };
        if should_finalize {
            let graph_name = hdr.graph_name.clone();
            let pg = decode_state.pending.remove(&graph_name).ok_or_else(|| {
                format!("pending graph {graph_name} not found at multi-key RDB finalization")
            })?;
            let graph = finalize_pending_graph(pg);
            // Store the finalized graph in DECODE_STATE for the caller to retrieve.
            decode_state.finalized.insert(graph_name, graph);
        }

        return Ok(LoadedKey::Partial {
            is_virtual: key_name != hdr.graph_name,
        });
    }

    // Single-key path (key_count == 1): decode everything in one go.
    let mut node_attrs = AttributeStore::new();
    let mut rel_attrs = AttributeStore::new();

    // One dictionary, shared by both stores — see the multi-key path above.
    let mut attrs_name = AttrNameMap::PropertyKeys(NameList::default());
    for name in &schema.attribute_names {
        attrs_name.insert(name);
    }

    let mut deleted_nodes = RoaringTreemap::new();
    let mut deleted_rels = RoaringTreemap::new();
    let mut label_matrices = Vec::new();
    let mut relationship_tensors: Vec<Tensor> = Vec::new();
    let mut adj_matrix = VersionedMatrix::<bool>::new(0, 0);
    let mut lbls_matrix = VersionedMatrix::<bool>::new(0, 0);

    for (state, count) in &payloads {
        match *state {
            EncodeState::Nodes => {
                node_attrs.decode_with_count(&mut r, *count, attrs_name.len())?;
            }
            EncodeState::DeletedNodes => {
                deleted_nodes.decode_with_count(&mut r, *count, attrs_name.len())?;
            }
            EncodeState::Edges => {
                rel_attrs.decode_with_count(&mut r, *count, attrs_name.len())?;
            }
            EncodeState::DeletedEdges => {
                deleted_rels.decode_with_count(&mut r, *count, attrs_name.len())?;
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

    let mut graph = Graph::restore(
        &hdr.graph_name,
        cache_size,
        hdr.node_count,
        hdr.edge_count,
        deleted_nodes,
        deleted_rels,
        adj_matrix,
        lbls_matrix,
        VersionedMatrix::<bool>::new(0, 0),
        label_matrices,
        relationship_tensors,
        schema.node_labels,
        schema.relationship_types,
        attrs_name,
        node_attrs,
        rel_attrs,
    );

    graph.rebuild_derived_matrices();
    rebuild_indexes(&mut graph, &schema.indexes);
    for c in schema.constraints {
        graph.add_constraint_raw(c);
    }
    graph.populate_indexes_sync();

    Ok(LoadedKey::Graph(Box::new(graph)))
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
                let attr_limit = pg.attrs_name.len();
                pg.node_attrs.decode_with_count(r, *count, attr_limit)?;
            }
            EncodeState::DeletedNodes => {
                pg.deleted_nodes
                    .decode_with_count(r, *count, pg.attrs_name.len())?;
            }
            EncodeState::Edges => {
                let attr_limit = pg.attrs_name.len();
                pg.rel_attrs.decode_with_count(r, *count, attr_limit)?;
            }
            EncodeState::DeletedEdges => {
                pg.deleted_rels
                    .decode_with_count(r, *count, pg.attrs_name.len())?;
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

/// Finalize a pending multi-key graph: build Graph, rebuild derived matrices.
pub fn finalize_pending_graph(pg: PendingGraph) -> Graph {
    let attrs_name = pg.attrs_name;
    let node_attrs = pg.node_attrs;
    let rel_attrs = pg.rel_attrs;

    let mut graph = Graph::restore(
        &pg.header.graph_name,
        pg.cache_size,
        pg.header.node_count,
        pg.header.edge_count,
        pg.deleted_nodes,
        pg.deleted_rels,
        pg.adj_matrix,
        pg.lbls_matrix,
        VersionedMatrix::<bool>::new(0, 0),
        pg.label_matrices,
        pg.relationship_tensors,
        pg.schema.node_labels,
        pg.schema.relationship_types,
        attrs_name,
        node_attrs,
        rel_attrs,
    );

    graph.rebuild_derived_matrices();
    rebuild_indexes(&mut graph, &pg.schema.indexes);
    for c in pg.schema.constraints {
        graph.add_constraint_raw(c);
    }
    graph.populate_indexes_sync();

    graph
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

        // Index-level language/stopwords ride on the first fulltext field
        // create only: `create_index` rejects language/stopwords once the
        // label already has a fulltext field ("Can not override ...").
        let mut text_meta_pending = info.language.is_some() || info.stopwords.is_some();

        // Every field of the label, in order. `create_index_sync` builds the label's spec anew
        // for each call, so consecutive fields of one type with the same options (every range
        // field has none) go in one call: a label with N such fields costs one spec, not N.
        let mut pending: Vec<(Arc<String>, &Arc<graph::index::Field>)> = info
            .field_order
            .iter()
            .filter_map(|attr_name| {
                let attr = Arc::new(attr_name.to_string());
                info.fields
                    .get(attr_name)
                    .map(|fields| fields.iter().map(move |f| (attr.clone(), f)))
            })
            .flatten()
            .collect();
        pending.reverse();

        let create = |graph: &mut Graph,
                      field: &graph::index::Field,
                      attrs: &Vec<Arc<String>>,
                      options: Option<graph::index::indexer::IndexOptions>| {
            graph.create_index_sync(&field.ty, &entity_type, &info.label, attrs, options)
        };

        while let Some((attr, field)) = pending.pop() {
            // The Field.name includes the type prefix (e.g. "range:val").
            // The attr_name key in the HashMap is the raw attribute name.
            let mut carries_meta = false;
            let options = field.vector_options().map_or_else(
                || {
                    let mut topts = field.options().cloned();
                    if text_meta_pending && field.ty == graph::index::IndexType::Fulltext {
                        let topts = topts.get_or_insert_with(Default::default);
                        topts.language.clone_from(&info.language);
                        topts.stopwords.clone_from(&info.stopwords);
                        text_meta_pending = false;
                        carries_meta = true;
                    }
                    topts.map(graph::index::indexer::IndexOptions::Text)
                },
                |vopts| Some(graph::index::indexer::IndexOptions::Vector(vopts.clone())),
            );

            // The fields that follow and are created exactly like this one. Not a vector
            // field (each has its own options) and not the one that carries the label's
            // language and stopwords (the options it is created with are not its own).
            let mut attrs = vec![attr];
            if field.vector_options().is_none() && !carries_meta {
                let mut seen: HashSet<Arc<String>> = attrs.iter().cloned().collect();
                while let Some((next, next_field)) = pending.last()
                    && next_field.ty == field.ty
                    && next_field.vector_options().is_none()
                    && next_field.options() == field.options()
                    && seen.insert(next.clone())
                {
                    attrs.push(next.clone());
                    pending.pop();
                }
            }

            if let Err(e) = create(graph, field, &attrs, options.clone()) {
                // One field that cannot be created must not take the others with it.
                let mut failed = vec![e];
                if attrs.len() > 1 {
                    failed.clear();
                    for attr in &attrs {
                        if let Err(e) = create(graph, field, &vec![attr.clone()], options.clone()) {
                            failed.push(e);
                        }
                    }
                }
                for e in failed {
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

    let payloads = decode_payload_directory(r)?;

    let mut node_attrs = AttributeStore::new();
    let mut rel_attrs = AttributeStore::new();

    // One dictionary, shared by both stores — see the other decode paths above.
    let mut attrs_name = AttrNameMap::PropertyKeys(NameList::default());
    for name in &schema.attribute_names {
        attrs_name.insert(name);
    }

    let mut deleted_nodes = RoaringTreemap::new();
    let mut deleted_rels = RoaringTreemap::new();
    let mut label_matrices = Vec::new();
    let mut relationship_tensors: Vec<Tensor> = Vec::new();
    let mut adj_matrix = VersionedMatrix::<bool>::new(0, 0);
    let mut lbls_matrix = VersionedMatrix::<bool>::new(0, 0);

    for (state, count) in &payloads {
        match *state {
            EncodeState::Nodes => {
                node_attrs.decode_with_count(r, *count, attrs_name.len())?;
            }
            EncodeState::DeletedNodes => {
                deleted_nodes.decode_with_count(r, *count, attrs_name.len())?;
            }
            EncodeState::Edges => {
                rel_attrs.decode_with_count(r, *count, attrs_name.len())?;
            }
            EncodeState::DeletedEdges => {
                deleted_rels.decode_with_count(r, *count, attrs_name.len())?;
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

    let mut graph = Graph::restore(
        dest_name,
        cache_size,
        hdr.node_count,
        hdr.edge_count,
        deleted_nodes,
        deleted_rels,
        adj_matrix,
        lbls_matrix,
        VersionedMatrix::<bool>::new(0, 0),
        label_matrices,
        relationship_tensors,
        schema.node_labels,
        schema.relationship_types,
        attrs_name,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serializers::buffered_io::VecWriter;
    use graph::graph::graphblas::serialization::{Encode, Writer};

    fn directory(entries: &[(u64, u64)]) -> Vec<u8> {
        let mut w = VecWriter::new();
        w.write_unsigned(entries.len() as u64);
        for &(state, count) in entries {
            w.write_unsigned(state);
            w.write_unsigned(count);
        }
        w.into_vec()
    }

    /// v19 and v20 load; a newer version, which may lay its payloads out differently, and an
    /// older one, which predates the v19 layout, are refused before anything is read.
    #[test]
    fn only_versions_19_and_20_load() {
        assert_eq!(ENCODING_VERSION, 20);
        assert!(check_encoding_version(19).is_ok());
        assert!(check_encoding_version(20).is_ok());
        let newer = check_encoding_version(21).unwrap_err();
        assert!(newer.contains("not forward compatible"), "{newer}");
        let older = check_encoding_version(18).unwrap_err();
        assert!(older.contains("older than the oldest"), "{older}");
    }

    /// A v20 key without CCH indexes lists the same payloads a v19 key does.
    #[test]
    fn a_directory_without_cch_decodes() {
        let bytes = directory(&[(1, 3), (5, 1), (9, 1)]);
        let payloads = decode_payload_directory(&mut BufferedReader::from_slice(&bytes)).unwrap();
        assert_eq!(
            payloads,
            [
                (EncodeState::Nodes, 3),
                (EncodeState::GraphSchema, 1),
                (EncodeState::LblsMatrix, 1),
            ]
        );
    }

    /// C's v20 lists `CchIndices` (10) last, only for a graph that has a CCH index. It used
    /// to decode as `Final` and be skipped, leaving its payload to be misread.
    #[test]
    fn a_directory_with_cch_is_refused() {
        let bytes = directory(&[(1, 3), (9, 1), (10, 1)]);
        let err = decode_payload_directory(&mut BufferedReader::from_slice(&bytes)).unwrap_err();
        assert!(err.contains("CCH"), "unexpected error: {err}");
    }

    /// `Final` (11, 10 before v20) and `Init` (0) are never written: no payload arm reads them,
    /// so accepting one would leave the stream at the wrong place.
    #[test]
    fn a_state_never_saved_is_refused() {
        for state in [0, 11] {
            let bytes = directory(&[(1, 3), (state, 1)]);
            let err =
                decode_payload_directory(&mut BufferedReader::from_slice(&bytes)).unwrap_err();
            assert!(err.contains("never saved"), "state {state}: {err}");
        }
    }

    /// A synthetic single-key value, as C's v20 lays one out for a graph with a CCH index, up to
    /// and including the directory: header and schema (written by this module's own encoders),
    /// then a directory ending in `CchIndices`, then only the CCH payload's words. The payloads
    /// the directory lists before it are left out: the decoder refuses the directory before it
    /// reads any payload. `vec_load_graph` is the `GRAPH.RESTORE` path; `pipe_load_graph`
    /// (`GRAPH.COPY`) and `rdb_load_graph` read the directory with the same function.
    #[test]
    fn a_v20_value_with_a_cch_index_is_refused_at_the_directory() {
        let name = |s: &str| Arc::new(s.to_string());
        let mut w = VecWriter::new();
        Encode::<19>::encode(
            &Header {
                graph_name: "g".to_string(),
                node_count: 2,
                edge_count: 1,
                deleted_node_count: 0,
                deleted_edge_count: 0,
                label_count: 1,
                relationship_count: 1,
                multi_edge: vec![false],
                key_count: 1,
            },
            &mut w,
        );
        Encode::<19>::encode(
            &Schema {
                attribute_names: vec![name("time")],
                node_labels: vec![name("N")],
                relationship_types: vec![name("ROAD")],
                indexes: Vec::new(),
                constraints: Vec::new(),
            },
            &mut w,
        );
        let directory = [
            (EncodeState::Nodes, 2),
            (EncodeState::Edges, 1),
            (EncodeState::LabelsMatrices, 1),
            (EncodeState::RelationMatrices, 1),
            (EncodeState::AdjMatrix, 1),
            (EncodeState::LblsMatrix, 1),
            (EncodeState::CchIndices, 1),
        ];
        w.write_unsigned(directory.len() as u64);
        for (state, count) in directory {
            w.write_unsigned(state as u64);
            w.write_unsigned(count);
        }
        // The CCH payload, as C's `RdbSaveCCH_v20` writes it: one index over relationship
        // type 0, weighted by attribute 0. The other payloads would precede it; none is read.
        for word in [1, 1, 0, 0] {
            w.write_unsigned(word);
        }
        let Err(err) = vec_load_graph(&w.into_vec(), 0, "g") else {
            panic!("a v20 value with a CCH index loaded");
        };
        assert!(
            err.contains("this module does not support CCH indexes"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn an_unknown_state_is_refused() {
        let bytes = directory(&[(12, 1)]);
        let err = decode_payload_directory(&mut BufferedReader::from_slice(&bytes)).unwrap_err();
        assert!(
            err.contains("unknown encode state: 12"),
            "unexpected error: {err}"
        );
    }
}
