//! Checks on what a payload says about itself, made before anything it says is
//! allowed to size an allocation or index a table.
//!
//! A `GRAPH.RESTORE` payload (and an RDB, and a `GRAPH.COPY` pipe) is decoded
//! from bytes nobody vouches for: any client allowed to run `GRAPH.RESTORE` can
//! send whatever it likes. The header's counts are the contract everything else
//! in the payload is held to — entity ids stay below them, collections cannot be
//! longer than they are or than the bytes that remain — and the header itself
//! is held to limits no real graph reaches.

use graph::graph::graphblas::serialization::{DecodeLimits, EncodeState, Reader, check_count_fits};
use roaring::RoaringTreemap;

use crate::serializers::{Header, Schema};

/// Most node ids (live plus deleted) a payload may declare, and the same for
/// edge ids. The id space is dense, so the id bound is also how far the
/// attribute directory must stretch: its root takes 8 bytes per 4096 ids, so
/// one attribute on an entity near the bound costs a 32 MiB root at 2^34 (17
/// billion ids, beyond any graph that fits in memory); 2^36 would be 128 MiB.
pub const MAX_ID_BOUND: u64 = 1 << 34;

/// Most labels, relationship types or payload sections a header may declare.
pub const MAX_SCHEMAS: u64 = 1 << 20;

/// What `Header::decode` leaves for the caller to vouch for: the counts are in
/// range, and the id bounds they imply are the [`DecodeLimits`] for the rest of
/// the payload.
pub fn check_header(hdr: &Header) -> Result<DecodeLimits, String> {
    let bound = |what: &str, live: u64, deleted: u64| -> Result<u64, String> {
        live.checked_add(deleted)
            .filter(|&b| b <= MAX_ID_BOUND)
            .ok_or_else(|| {
                format!(
                    "{what} counts {live} and {deleted} are out of range (limit {MAX_ID_BOUND})"
                )
            })
    };
    let node_bound = bound("node", hdr.node_count, hdr.deleted_node_count)?;
    let edge_bound = bound("edge", hdr.edge_count, hdr.deleted_edge_count)?;
    for (what, n) in [
        ("label", hdr.label_count),
        ("relationship type", hdr.relationship_count),
    ] {
        if n > MAX_SCHEMAS {
            return Err(format!("{what} count {n} is out of range"));
        }
    }
    Ok(DecodeLimits {
        node_bound,
        edge_bound,
    })
}

/// Hold the payload directory's counts to the header's, so that a section
/// cannot claim more entities than the graph has.
pub fn check_directory(
    payloads: &[(EncodeState, u64)],
    hdr: &Header,
) -> Result<(), String> {
    for &(state, count) in payloads {
        let limit = match state {
            EncodeState::Nodes => hdr.node_count,
            EncodeState::DeletedNodes => hdr.deleted_node_count,
            EncodeState::Edges => hdr.edge_count,
            EncodeState::DeletedEdges => hdr.deleted_edge_count,
            _ => u64::MAX,
        };
        if count > limit {
            return Err(format!(
                "{state:?} section holds {count} entities, more than the {limit} the header declares"
            ));
        }
    }
    Ok(())
}

/// How many labels a `LabelsMatrices` section holds, read and checked.
pub fn label_matrix_count(
    r: &mut dyn Reader,
    hdr: &Header,
) -> Result<u64, String> {
    let count = r.read_unsigned()?;
    if count > hdr.label_count {
        return Err(format!(
            "label matrix section holds {count} labels, more than the {} the header declares",
            hdr.label_count
        ));
    }
    // A label is an id and a matrix of several buffers.
    check_count_fits(r, "label matrix", count, 18)?;
    Ok(count)
}

/// How many relationship tensors a `RelationMatrices` section holds: the
/// header's count, which must fit in what is left.
pub fn tensor_count(
    r: &dyn Reader,
    hdr: &Header,
) -> Result<u64, String> {
    check_count_fits(r, "relationship tensor", hdr.relationship_count, 18)?;
    Ok(hdr.relationship_count)
}

/// The deleted ids a decode collected are the ones the header counted, and all
/// lie under the id bound.
fn check_deleted(
    what: &str,
    ids: &RoaringTreemap,
    declared: u64,
) -> Result<(), String> {
    if ids.len() != declared {
        return Err(format!(
            "the header declares {declared} deleted {what}s but the payload lists {}",
            ids.len()
        ));
    }
    Ok(())
}

/// What a finished decode has to satisfy before it becomes a graph.
pub fn check_assembled(
    hdr: &Header,
    schema: &Schema,
    deleted_nodes: &RoaringTreemap,
    deleted_rels: &RoaringTreemap,
    label_matrices: usize,
    relationship_tensors: usize,
) -> Result<(), String> {
    check_deleted("node", deleted_nodes, hdr.deleted_node_count)?;
    check_deleted("edge", deleted_rels, hdr.deleted_edge_count)?;
    // Indexed by label id and relationship id throughout the engine.
    if label_matrices != schema.node_labels.len() {
        return Err(format!(
            "the payload has {label_matrices} label matrices for {} labels",
            schema.node_labels.len()
        ));
    }
    if relationship_tensors != schema.relationship_types.len() {
        return Err(format!(
            "the payload has {relationship_tensors} relationship tensors for {} types",
            schema.relationship_types.len()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serializers::buffered_io::{BufferedReader, VecWriter};
    use graph::graph::attribute_store::AttributeStore;
    use graph::graph::graphblas::serialization::{Decode, Writer};

    fn header(
        node_count: u64,
        edge_count: u64,
    ) -> Header {
        Header {
            graph_name: "g".to_string(),
            node_count,
            edge_count,
            deleted_node_count: 0,
            deleted_edge_count: 0,
            label_count: 1,
            relationship_count: 1,
            multi_edge: vec![false],
            key_count: 1,
        }
    }

    /// #3588: a header count in the billions panicked in `GrB_Matrix_new`.
    #[test]
    fn a_header_count_beyond_any_graph_is_refused() {
        let limits = check_header(&header(10, 20)).unwrap();
        assert_eq!((limits.node_bound, limits.edge_bound), (10, 20));
        for hdr in [
            header(u64::MAX, 0),
            header(0, u64::MAX),
            header(MAX_ID_BOUND + 1, 0),
        ] {
            assert!(check_header(&hdr).is_err());
        }
        let mut hdr = header(MAX_ID_BOUND, 0);
        hdr.deleted_node_count = 1;
        assert!(
            check_header(&hdr).is_err(),
            "live + deleted overflows the bound"
        );
        let mut hdr = header(1, 1);
        hdr.relationship_count = u64::MAX;
        assert!(check_header(&hdr).is_err());
    }

    #[test]
    fn a_section_cannot_hold_more_entities_than_the_header_counts() {
        let hdr = header(10, 20);
        assert!(
            check_directory(&[(EncodeState::Nodes, 10), (EncodeState::Edges, 20)], &hdr).is_ok()
        );
        assert!(check_directory(&[(EncodeState::Nodes, 11)], &hdr).is_err());
        assert!(check_directory(&[(EncodeState::Edges, u64::MAX)], &hdr).is_err());
        assert!(check_directory(&[(EncodeState::DeletedNodes, 1)], &hdr).is_err());
    }

    /// A count that the bytes left cannot hold is refused before the loop that
    /// would read it, however the payload came by it.
    #[test]
    fn a_slice_reader_refuses_a_count_it_cannot_back() {
        let mut w = VecWriter::new();
        w.write_unsigned(7); // entity id
        w.write_unsigned(1); // attribute count, with no attribute after it
        let bytes = w.into_vec();
        let mut r = BufferedReader::from_slice(&bytes);
        let mut store = AttributeStore::with_id_bound(8);
        // The loop ends at the end of the payload.
        let err = store.decode_with_count(&mut r, 1, 1).unwrap_err();
        assert!(err.contains("end of data"), "unexpected error: {err}");

        // And a count of entities, when the payload ends first.
        let mut r = BufferedReader::from_slice(&bytes);
        let err = AttributeStore::with_id_bound(8)
            .decode_with_count(&mut r, u64::MAX, 1)
            .unwrap_err();
        assert!(err.contains("exceeds"), "unexpected error: {err}");
    }

    /// The payload ending is an error, not a call into Redis with no handle.
    #[test]
    fn a_slice_reader_reports_the_end_of_the_payload() {
        let mut r = BufferedReader::from_slice(&[]);
        assert!(r.read_unsigned().is_err());
        assert!(r.read_buffer().is_err());
    }
}
