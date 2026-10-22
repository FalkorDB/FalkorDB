//! `Graph` -> [`CchInput`]: the graph's relationship tensors as the two CSR
//! arrays the hierarchy builder consumes, a symmetric self-loop-free skeleton
//! for Phase 1 and a directed weight matrix for Phase 2, computed in
//! GraphBLAS. Per relationship type, one `GrB_Matrix_apply` scores every
//! forward pair and one `GrB_mxv` per `me` block, over a `MIN` semiring,
//! scores every parallel edge; the types merge through `MIN`.
//!
//! The C original is `CCH_BuildWeightMatrix` / `_get_edge_weight` in
//! `src/index/cch_index.c`, plus `_build_undirected_skeleton` in
//! `src/algorithms/cch.c`. The one structural difference is edge-id storage:
//! C's tensor cell is either a scalar id or a tagged `GrB_Vector` of ids, so
//! one operator covers both. This repo stores a single-edge pair's id inline
//! in the forward matrix and promotes a multi-edge pair to the
//! [`MULTI_EDGE`](crate::graph::graphblas::tensor::MULTI_EDGE) sentinel, with
//! its ids living in `me` keyed by
//! [`compound_key`](crate::graph::graphblas::tensor::compound_key), so it
//! takes two passes, merged by `MIN`: the same shape as `algo.MSF`
//! (`runtime/functions/algo_procedures.rs`).

mod pair_weights;
#[cfg(test)]
mod tests;
mod weight_ops;

use crate::graph::graph::Graph;
use crate::graph::graphblas::matrix::{Descriptor, Matrix};
use crate::runtime::value::Value;

use super::{CchError, CchIndexDef, CchInput, id_index};
pub(crate) use pair_weights::{PairWeights, cheapest_edge};
pub(super) use weight_ops::edge_weight;
use weight_ops::{WeightContext, WeightOps};

/// Whether `value`, as an edge's weight attribute, is a weight an index takes: what
/// [`edge_weight`](weight_ops::edge_weight) reads it as is `>= 0`, `+inf` (a closed edge) and `-0.0` included. A
/// missing or non-numeric value reads as [`DEFAULT_WEIGHT`](weight_ops::DEFAULT_WEIGHT), so
/// only a negative number (`-inf` included) or NaN is refused. The rule the weight operators
/// score by ([`cch_edge_weight`](weight_ops::cch_edge_weight)) and the write-time check
/// ([`check_edge_weights`](super::maint::check_edge_weights)) refuses by.
pub(super) fn is_valid_weight(value: &Value) -> bool {
    match value {
        // NaN fails the comparison; `+inf` and `-0.0` pass it.
        Value::Float(weight) => *weight >= 0.0,
        Value::Int(integer) => *integer >= 0,
        _ => true,
    }
}

impl CchInput {
    /// Build the hierarchy builder's input from `graph`, over the relationship
    /// types and weight attribute named by `def`.
    ///
    /// Node ids are used unchanged as CCH ids, so `n` is the graph's matrix
    /// dimension ([`Graph::node_cap`]) rather than its live node count: ids
    /// freed by deletions, and the slack the capacity grows in, appear as
    /// isolated vertices. This mirrors C, which sizes the hierarchy with
    /// `Graph_RequiredMatrixDim`.
    ///
    /// Guarantees the result satisfies, which Phase 1 and Phase 2 may rely on:
    /// - `skeleton` is symmetric and has no `u == v` entry;
    /// - `pattern(weights) ⊆ pattern(skeleton)`, so every weighted arc has a
    ///   skeleton edge — the skeleton is derived from the weight matrix's own
    ///   pattern, so this is structural rather than hoped for;
    /// - `weights` has no `u == v` entry either: a self-loop can never lie on
    ///   a shortest path, and dropping it from both sides is what keeps the
    ///   inclusion above true;
    /// - every weight is `>= 0`, `+inf` (a closed edge) included;
    /// - within a row, column ids ascend and do not repeat.
    ///
    /// # Errors
    /// - [`CchError::UnknownRelType`] / [`CchError::UnknownAttribute`] if
    ///   `def` names something this graph does not have.
    /// - [`CchError::InvalidWeight`] if any edge of an indexed type carries a
    ///   negative or NaN weight — every edge, not only the cheapest of
    ///   its pair, and self-loops included even though they are then dropped.
    /// - [`CchError::Internal`] if a multi-edge pair has no edge in `me`, a
    ///   broken tensor invariant.
    ///
    /// # Panics
    /// If a GraphBLAS call fails, the export included (see `grb!`). Every matrix
    /// and operator here is built by this function, so that is a bug or an
    /// allocation failure, never something in the graph's data: bad weights
    /// and unknown names are the errors above.
    pub(crate) fn build(
        graph: &Graph,
        def: &CchIndexDef,
    ) -> Result<Self, CchError> {
        let (node_count, mut weights) = Self::scored_weights(graph, def)?;

        // Drop self-loops from the weights, which is what lets the skeleton be
        // derived from this pattern and still be diagonal free. After the
        // check, so a bad weight on a self-loop is refused all the same.
        weights.drop_diagonal();

        // Skeleton = pattern(W) ∪ pattern(Wᵀ): a weight of 0.0 is an entry
        // like any other.
        let mut skeleton = Matrix::<bool>::new(node_count, node_count);
        skeleton.pattern_union(&weights, &weights, Some(Descriptor::T1));

        // Copy each matrix out as sorted, 64-bit CSR. Each export consumes its
        // matrix, so the skeleton's GraphBLAS arrays are freed before the
        // weights are copied.
        let skeleton = skeleton.into_csr_pattern().into();
        let weights = weights.into_csr().into();
        let input = Self {
            n: id_index(node_count),
            skeleton,
            weights,
        };
        debug_assert!(input.skeleton.is_symmetric_simple());
        Ok(input)
    }

    /// The weight CSR, for tests that customize a hierarchy over it.
    #[cfg(test)]
    pub(crate) const fn weights(&self) -> &super::WeightCsr {
        &self.weights
    }

    /// What `CREATE CCH INDEX` checks before it returns, while the hierarchy itself is built in
    /// the background: the definition resolves and every edge of an indexed type has a valid
    /// weight. The first half of [`Self::build`], its weight passes and their reductions (about
    /// 5 ms on California), without the skeleton and the export. Once it passes, the write-time
    /// check ([`check_edge_weights`](super::maint::check_edge_weights)) keeps every later
    /// version valid, so the build cannot fail on the graph's data.
    ///
    /// # Errors
    /// As [`Self::build`]: [`CchError::UnknownRelType`], [`CchError::UnknownAttribute`],
    /// [`CchError::InvalidWeight`], and [`CchError::Internal`] for a broken tensor.
    pub(crate) fn validate(
        graph: &Graph,
        def: &CchIndexDef,
    ) -> Result<(), CchError> {
        Self::scored_weights(graph, def).map(drop)
    }

    /// The node count and the weight matrix: one entry per directed pair, the cheapest weight
    /// over the pair's edges of every indexed type, self-loops still in. Refuses a bad weight.
    fn scored_weights(
        graph: &Graph,
        def: &CchIndexDef,
    ) -> Result<(u64, Matrix<f64>), CchError> {
        let tensors = graph.relationship_tensors();
        let selected = def
            .rel_types()
            .iter()
            .map(|rel_type| tensors.get(rel_type.0).ok_or(CchError::UnknownRelType))
            .collect::<Result<Vec<_>, _>>()?;
        if usize::from(def.weight_attr) >= graph.property_key_count() {
            return Err(CchError::UnknownAttribute);
        }

        let node_count = graph.node_cap();
        for tensor in &selected {
            let forward = tensor.fwd_m();
            if (forward.nrows(), forward.ncols()) != (node_count, node_count) {
                return Err(CchError::Internal("a tensor is not node_cap square"));
            }
        }

        let context = WeightContext::new(graph.relationship_attrs(), def.weight_attr);
        let ops = WeightOps::get();
        // Each type is checked before the merge, whose `MIN` would hide an
        // [`UNSCORED`] pair behind another type's edge on the same pair.
        let mut merged: Option<Matrix<f64>> = None;
        for tensor in &selected {
            let type_weights = ops.tensor_weights(tensor, &context);
            check_weights(&type_weights)?;
            merged = Some(match merged {
                None => type_weights,
                Some(mut weights) => {
                    weights.min_union(&type_weights);
                    weights
                }
            });
        }
        let weights = merged.unwrap_or_else(|| Matrix::<f64>::new(node_count, node_count));
        Ok((node_count, weights))
    }
}

/// Refuse the build if any weight in one type's `weights` is bad, as
/// [`INVALID_WEIGHT`](weight_ops::INVALID_WEIGHT) and
/// [`UNSCORED`](weight_ops::UNSCORED) describe.
///
/// One `PLUS` reduction accepts the valid case: the sum is `>= 0` exactly when
/// there is neither `-inf` (which gives `-inf`, or NaN next to `+inf`) nor NaN.
/// Only a failed build pays for a `MIN` reduction, which tells them apart.
fn check_weights(weights: &Matrix<f64>) -> Result<(), CchError> {
    // The sum of an empty matrix is 0, accepted; a NaN sum fails `>=`.
    if weights.sum_value() >= 0.0 {
        Ok(())
    } else if weights.min_value() < 0.0 {
        Err(CchError::InvalidWeight)
    } else {
        Err(CchError::Internal("a multi-edge pair has no edge in `me`"))
    }
}
