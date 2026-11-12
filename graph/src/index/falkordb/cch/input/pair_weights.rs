//! Per-pair weights read from the graph: the cheapest edge of one node pair, by the weight rule
//! the index build scores with ([`edge_weight`]). For the few pairs a scoped recustomization
//! seeds and for the procedure's path edges; each lookup reads one pair's own edges, never a
//! row or a matrix.

use std::sync::Arc;

use super::weight_ops::edge_weight;
use crate::graph::graph::{Graph, NodeId, RelationshipId};
use crate::index::falkordb::cch::{CchError, CchIndexDef};

/// The cheapest edge from `src` to `dst` among `types`, by [`edge_weight`], as
/// `(edge id, weight)`; `None` when there is none.
///
/// The edges are [`Graph::get_src_dest_relationships`]: the pair's own edges in each type,
/// never a row scan. An empty `types` means every type. Ties go to the earlier type in
/// `types`, then to the lowest edge id.
pub(crate) fn cheapest_edge(
    g: &Graph,
    types: &[Arc<String>],
    attribute: u16,
    src: NodeId,
    dst: NodeId,
) -> Option<(RelationshipId, f64)> {
    let attributes = g.relationship_attrs();
    let mut best: Option<(RelationshipId, f64)> = None;
    for id in g.get_src_dest_relationships(src, dst, types) {
        let weight = edge_weight(attributes, attribute, u64::from(id));
        if best.is_none_or(|(_, best_weight)| weight < best_weight) {
            best = Some((id, weight));
        }
    }
    best
}

/// The current weight of one node pair over an index's definition, read from the graph
/// pair by pair: the seeds a scoped recustomization asks for
/// ([`SeedFn`](super::super::partial::SeedFn)), without building the whole weight matrix.
///
/// Agrees with [`CchInput::build`](super::super::CchInput::build) entry for entry: the same edges (every tensor of the
/// definition, parallel edges included, pending writes of this version included, deletions
/// excluded), the same per-edge rule ([`edge_weight`]) and the same merge (the minimum).
/// Weights are returned unchecked; the scoped path checks the seeds of the pairs it was
/// told changed, and the write-time check keeps every other edge valid.
pub(crate) struct PairWeights<'g> {
    graph: &'g Graph,
    types: Vec<Arc<String>>,
    attribute: u16,
}

impl<'g> PairWeights<'g> {
    /// Resolve `def` once against `graph`.
    ///
    /// # Errors
    /// [`CchError::UnknownRelType`] / [`CchError::UnknownAttribute`], as [`CchInput::build`](super::super::CchInput::build).
    pub(crate) fn new(
        graph: &'g Graph,
        def: &CchIndexDef,
    ) -> Result<Self, CchError> {
        let all = graph.get_types();
        let types = def
            .rel_types()
            .iter()
            .map(|type_id| all.get(type_id.0).cloned().ok_or(CchError::UnknownRelType))
            .collect::<Result<_, _>>()?;
        if usize::from(def.weight_attr) >= graph.property_key_count() {
            return Err(CchError::UnknownAttribute);
        }
        Ok(Self {
            graph,
            types,
            attribute: def.weight_attr,
        })
    }

    /// The cheapest weight of the edges `src -> dst`, `None` when there is none.
    pub(crate) fn get(
        &self,
        src: u64,
        dst: u64,
    ) -> Option<f64> {
        cheapest_edge(
            self.graph,
            &self.types,
            self.attribute,
            NodeId::from(src),
            NodeId::from(dst),
        )
        .map(|(_, weight)| weight)
    }
}
