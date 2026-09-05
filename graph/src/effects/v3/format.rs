//! v3 as a [`EffectsFormat`]: what a whole payload is, sent and applied.
//!
//! The version-specific half of `effects::EffectsPayload`. It lives here rather
//! than beside the trait so that adding v4 means adding a directory and one arm
//! to the dispatch, and touching nothing else — the trait's module stays free
//! of any single version's rules.

use atomic_refcell::AtomicRefCell;

use super::emit::{build_constraint_buffer, build_effects_buffer, build_index_buffer};
use super::{EFFECTS_VERSION, apply::ApplyError, apply::apply_effects, seal};
use crate::effects::announce::{AnnouncedConstraint, AnnouncedIndex, SchemaBaseline};
use crate::effects::{EffectsFormat, EffectsPayload, ReplicationSink};
use crate::graph::graph::Graph;
use crate::runtime::pending::Pending;

impl EffectsFormat<EFFECTS_VERSION> for EffectsPayload {
    fn is_empty(buf: &[u8]) -> bool {
        // `u8 version` + `u8 flags`, and nothing after them.
        buf.len() <= HEADER_LEN
    }

    fn replicate(
        sink: &dyn ReplicationSink,
        key: &[u8],
        mut buf: Vec<u8>,
    ) {
        // Compression, and the last thing that touches the bytes. It rewrites
        // everything after the header, so it can only run once and only here —
        // which is why it is not reachable on its own. What "worth compressing"
        // means is `seal`'s: it reads the threshold itself, so the
        // configuration stays behind this boundary too.
        seal(&mut buf);
        sink.replicate("GRAPH.EFFECT", &[key, buf.as_slice()]);
    }

    fn apply(
        graph: &mut Graph,
        buf: &[u8],
    ) -> Result<(), ApplyError> {
        apply_effects(graph, buf)
    }

    fn build(
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
        buf: &mut Vec<u8>,
    ) -> u64 {
        build_effects_buffer(pending, graph, buf)
    }

    fn build_index(
        pending: &Pending,
        graph: &AtomicRefCell<Graph>,
        create: bool,
        index: &AnnouncedIndex<'_>,
        buf: &mut Vec<u8>,
    ) -> Result<(), String> {
        build_index_buffer(pending, graph, create, index, buf)
    }

    fn build_constraint(
        graph: &Graph,
        create: bool,
        constraint: &AnnouncedConstraint<'_>,
        baseline: &SchemaBaseline,
        buf: &mut Vec<u8>,
    ) -> Result<(), String> {
        build_constraint_buffer(graph, create, constraint, baseline, buf)
    }
}

/// `u8 version` + `u8 flags`.
const HEADER_LEN: usize = 2;
