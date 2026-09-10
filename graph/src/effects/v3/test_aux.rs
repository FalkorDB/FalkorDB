//! Fixtures and helpers the v3 tests share.
//!
//! Here rather than in the files they exercise, for the reason
//! [`super::staging`] gives for the same thing: test scaffolding does not
//! belong in a production file's surface. Two of these — [`digest`] and
//! [`read_buffer`] — were `#[cfg(test)]` functions sitting at module level in
//! `emit.rs` and `records.rs`, which is scaffolding in the middle of the
//! encoder and the decoder, and a third was written twice.
//!
//! Only what more than one test module wants. A helper with a single caller
//! stays beside it, where a reader can see what it does without leaving the
//! file — `payload` in `format.rs`, `roundtrip` in `id_list.rs`,
//! `create_10k_in_shapes` in `records.rs`, `write_delete` in `apply.rs`.
//!
//! Split from [`super::staging`] rather than merged into it: that module has
//! one job, building a `Pending` through the runtime's own API, and graph
//! fixtures are a second.

use atomic_refcell::AtomicRefCell;
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::effects::DecodeError;
use crate::effects::v3::{self as v3, EffectEncode, Record, emit::for_each_record, open_payload};
use crate::graph::graph::Graph;
use crate::graph::graphblas::test_init::ensure_init;
use crate::runtime::pending::Pending;

// ── graphs ──

/// An empty graph, with GraphBLAS initialised.
///
/// `ensure_init` because `GrB_init` is process-wide and may only run once,
/// whichever test gets there first.
pub(crate) fn graph() -> Graph {
    ensure_init();
    Graph::new(64, 64, 0, 0, "t")
}

/// The same graph, wrapped as the emit path takes it.
pub(crate) fn graph_cell() -> AtomicRefCell<Graph> {
    AtomicRefCell::new(graph())
}

/// Register node attribute names, so the ids the tests use resolve.
pub(crate) fn with_attrs(
    g: &AtomicRefCell<Graph>,
    names: &[&str],
) {
    let mut graph = g.borrow_mut();
    for name in names {
        graph.add_node_attribute_name(name);
    }
}

/// One live edge of `type_name`, so a delete or an update has something real to
/// name.
pub(crate) fn with_edge(
    g: &AtomicRefCell<Graph>,
    type_name: &str,
    id: u64,
) {
    let mut graph = g.borrow_mut();
    graph
        .create_relationships_bulk(&Arc::new(type_name.to_owned()), &[0], &[1], &[id], None)
        .expect("no batch, nothing to refuse");
}

/// One live node, labelled.
pub(crate) fn live_node(
    g: &AtomicRefCell<Graph>,
    id: u64,
    labels: &[&str],
) {
    let mut graph = g.borrow_mut();
    let ids: RoaringTreemap = std::iter::once(id).collect();
    graph.create_allocated_nodes(&ids);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for name in labels {
        rows.push(id);
        cols.push(graph.get_label_id_mut(name).0 as u64);
    }
    if !rows.is_empty() {
        let mut docs = FxHashMap::default();
        graph.set_nodes_labels_bulk(&rows, &cols, &mut docs, true);
    }
}

// ── records and buffers ──

/// Every record a committed `Pending` implies, as values.
///
/// Not on the emit path, which streams through `for_each_record` and never
/// holds more than one record at a time. This exists because tests and the
/// codec benchmarks need the records as values — to assert on their shape, or
/// to time encoding separately from digesting — and reconstructing them by
/// decoding a buffer would test the decoder rather than the digest.
#[must_use]
pub(crate) fn digest(
    p: &Pending,
    g: &AtomicRefCell<Graph>,
) -> Vec<Record> {
    let mut out = Vec::new();
    for_each_record(p, g, |record| out.push(record));
    out
}

/// The loop `format::build` runs, so a test goes through the bytes rather than
/// stopping at the records.
pub(crate) fn encode_all(
    p: &Pending,
    g: &AtomicRefCell<Graph>,
    buf: &mut Vec<u8>,
) {
    for_each_record(p, g, |record| record.encode(buf).unwrap());
}

/// Encode, then decode: what a replica would see.
pub(crate) fn round_trip(
    p: &Pending,
    g: &AtomicRefCell<Graph>,
) -> Vec<Record> {
    let mut buf = v3::new_buffer();
    encode_all(p, g, &mut buf);
    read_buffer(&buf).expect("v3 buffer must decode")
}

/// Every record in a payload, materialized.
///
/// For tests and for the decode benchmark, which wants a whole payload's worth
/// of work in one call. Nothing on the apply path uses it and nothing should:
/// `apply_effects` streams, so it never holds more than one record at a time,
/// and `EffectsPayload::describe` renders them one at a time for the same
/// reason.
///
/// # Errors
///
/// Returns [`DecodeError`] if the payload or any record in it is malformed.
pub(crate) fn read_buffer(buf: &[u8]) -> Result<Vec<Record>, DecodeError> {
    open_payload(buf)?.records().collect()
}
