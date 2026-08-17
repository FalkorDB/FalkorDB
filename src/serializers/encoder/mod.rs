use graph::graph::graph::Graph;
use graph::graph::graphblas::serialization::{Encode, EncodeState, PayloadEntry, Writer};
use redis_module::raw::RedisModuleIO;

use super::buffered_io::BufferedWriter;
use super::{Header, Schema};

/// Encode a full graph into a single RDB key (v19 format, single-key mode).
///
/// This is the backward-compatible entry point used when `key_count == 1`.
pub fn rdb_save_graph(
    rdb: *mut RedisModuleIO,
    graph: &Graph,
) {
    let payloads = build_payloads(graph);
    rdb_save_graph_key(rdb, graph, &payloads, 1);
}

/// Encode a single key's portion of the graph (used for both primary and virtual keys).
pub fn rdb_save_graph_key(
    rdb: *mut RedisModuleIO,
    graph: &Graph,
    payloads: &[PayloadEntry],
    key_count: u64,
) {
    let mut w = BufferedWriter::new(rdb);
    let global_attrs = graph.build_global_attrs();
    encode_graph(&mut w, graph, payloads, key_count, &global_attrs);
    w.finish();
}

/// Encode a full graph into a pipe fd (for GRAPH.COPY fork+pipe).
///
/// Uses the same v19 format as RDB, with `PipeWriter` instead of `BufferedWriter`.
pub fn pipe_save_graph(
    fd: std::os::unix::io::OwnedFd,
    graph: &Graph,
) {
    let payloads = build_payloads(graph);
    let mut w = super::buffered_io::PipeWriter::new(fd);
    let global_attrs = graph.build_global_attrs();
    encode_graph(&mut w, graph, &payloads, 1, &global_attrs);
    w.finish();
}

/// Encode a full graph into a `Vec<u8>` (for GRAPH.RESTORE replication).
///
/// Uses the same v19 format as RDB, with `VecWriter` instead of `BufferedWriter`.
/// The returned bytes can be decoded with `BufferedReader::from_vec()`.
pub fn vec_save_graph(graph: &Graph) -> Vec<u8> {
    let payloads = build_payloads(graph);
    let mut w = super::buffered_io::VecWriter::new();
    let global_attrs = graph.build_global_attrs();
    encode_graph(&mut w, graph, &payloads, 1, &global_attrs);
    w.into_vec()
}

/// Shared encoding logic: header, schema, payload directory, payload data.
fn encode_graph(
    w: &mut dyn Writer,
    graph: &Graph,
    payloads: &[PayloadEntry],
    key_count: u64,
    global_attrs: &[std::sync::Arc<String>],
) {
    Header::from_graph(graph, key_count).encode(w);
    Schema::from_graph(graph, global_attrs.to_vec()).encode(w);

    w.write_unsigned(payloads.len() as u64);
    for p in payloads {
        w.write_unsigned(p.state as u64);
        w.write_unsigned(p.count);
    }

    for p in payloads {
        graph.encode_payload(w, p);
    }
}

/// Build per-key payload distributions for multi-key encoding.
///
/// Returns a Vec of per-key payload lists. Key 0 always gets matrices.
/// Entity payloads (nodes, edges, deleted nodes, deleted edges) are distributed
/// across keys such that each key gets at most `vkey_max` entities.
pub fn build_multi_key_payloads(
    graph: &Graph,
    vkey_max: u64,
) -> Vec<Vec<PayloadEntry>> {
    let nc = graph.node_count();
    let ec = graph.relationship_count();
    let dnc = graph.deleted_nodes_count();
    let dec = graph.deleted_relationships_count();

    let total_entities = nc + ec + dnc + dec;
    let key_count = if total_entities == 0 || vkey_max == 0 {
        1u64
    } else {
        total_entities.div_ceil(vkey_max)
    };

    // Entity types in encoding order with their total counts.
    let entity_types: Vec<(EncodeState, u64)> = [
        (EncodeState::Nodes, nc),
        (EncodeState::DeletedNodes, dnc),
        (EncodeState::Edges, ec),
        (EncodeState::DeletedEdges, dec),
    ]
    .into_iter()
    .filter(|(_, count)| *count > 0)
    .collect();

    // Distribute entities across keys, vkey_max per key.
    let mut keys: Vec<Vec<PayloadEntry>> = Vec::with_capacity(key_count as usize);

    // Track global offset per entity type.
    let mut type_offsets: Vec<u64> = vec![0; entity_types.len()];
    let mut type_idx = 0usize; // current entity type index

    for key_idx in 0..key_count {
        let mut key_payloads = Vec::new();
        // When vkey_max == 0, store everything in first key (unlimited capacity)
        let mut remaining_capacity = if vkey_max == 0 { u64::MAX } else { vkey_max };

        // Fill this key with entities from the current position
        while remaining_capacity > 0 && type_idx < entity_types.len() {
            let (state, total) = entity_types[type_idx];
            let available = total - type_offsets[type_idx];
            let take = remaining_capacity.min(available);

            if take > 0 {
                key_payloads.push(PayloadEntry {
                    state,
                    count: take,
                    offset: type_offsets[type_idx],
                });
                type_offsets[type_idx] += take;
                remaining_capacity -= take;
            }

            if type_offsets[type_idx] >= total {
                type_idx += 1;
            }
        }

        // Key 0 always gets matrices
        if key_idx == 0 {
            let lmc = graph.label_matrices().len() as u64;
            if lmc > 0 {
                key_payloads.push(PayloadEntry {
                    state: EncodeState::LabelsMatrices,
                    count: lmc,
                    offset: 0,
                });
            }
            let rmc = graph.relationship_tensors().len() as u64;
            if rmc > 0 {
                key_payloads.push(PayloadEntry {
                    state: EncodeState::RelationMatrices,
                    count: rmc,
                    offset: 0,
                });
            }
            key_payloads.push(PayloadEntry {
                state: EncodeState::AdjMatrix,
                count: 1,
                offset: 0,
            });
            key_payloads.push(PayloadEntry {
                state: EncodeState::LblsMatrix,
                count: 1,
                offset: 0,
            });
        }

        keys.push(key_payloads);
    }

    keys
}

/// Build the list of (state, entity_count) payloads for a single-key encode.
fn build_payloads(graph: &Graph) -> Vec<PayloadEntry> {
    let mut payloads = Vec::new();

    let nc = graph.node_count();
    if nc > 0 {
        payloads.push(PayloadEntry {
            state: EncodeState::Nodes,
            count: nc,
            offset: 0,
        });
    }
    let dnc = graph.deleted_nodes_count();
    if dnc > 0 {
        payloads.push(PayloadEntry {
            state: EncodeState::DeletedNodes,
            count: dnc,
            offset: 0,
        });
    }
    let ec = graph.relationship_count();
    if ec > 0 {
        payloads.push(PayloadEntry {
            state: EncodeState::Edges,
            count: ec,
            offset: 0,
        });
    }
    let dec = graph.deleted_relationships_count();
    if dec > 0 {
        payloads.push(PayloadEntry {
            state: EncodeState::DeletedEdges,
            count: dec,
            offset: 0,
        });
    }
    let lmc = graph.label_matrices().len();
    if lmc > 0 {
        payloads.push(PayloadEntry {
            state: EncodeState::LabelsMatrices,
            count: lmc as u64,
            offset: 0,
        });
    }
    let rmc = graph.relationship_tensors().len();
    if rmc > 0 {
        payloads.push(PayloadEntry {
            state: EncodeState::RelationMatrices,
            count: rmc as u64,
            offset: 0,
        });
    }
    payloads.push(PayloadEntry {
        state: EncodeState::AdjMatrix,
        count: 1,
        offset: 0,
    });
    payloads.push(PayloadEntry {
        state: EncodeState::LblsMatrix,
        count: 1,
        offset: 0,
    });

    payloads
}
