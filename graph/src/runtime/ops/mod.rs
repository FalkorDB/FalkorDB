//! Runtime operator implementations for the query execution engine.
//!
//! Operators process data in batches for improved throughput. The primary
//! execution path uses [`BatchOp`](crate::runtime::batch::BatchOp) which
//! processes up to [`BATCH_SIZE`](crate::runtime::batch::BATCH_SIZE) rows
//! per operator invocation.
//!
//! ```text
//!                          BatchOp (enum dispatch)
//!                                  |
//!          +-----------+-----------+-----------+-- ...
//!          |           |           |           |
//!     AggregateOp  FilterOp   SortOp    ProjectOp
//!          |           |
//!       child op    child op
//!
//! Pull model:  parent.next()  -->  child.next()  -->  ...  -->  scan.next()
//! ```
//!
//! ## Operator categories
//!
//! | Category     | Operators |
//! |--------------|-----------|
//! | **Scans**    | `NodeByLabelScan`, `NodeByIndexScan`, `NodeByIdSeek`, `NodeByLabelAndIdScan`, `NodeByFulltextScan` |
//! | **Traversal**| `CondTraverse`, `CondVarLenTraverse`, `ExpandInto` |
//! | **Filter**   | `Filter`, `Distinct`, `Skip`, `Limit`, `Sort` |
//! | **Mutation** | `Create`, `Delete`, `Set`, `Remove`, `Merge`, `Commit` |
//! | **Control**  | `Apply`, `SemiApply`, `Optional`, `OrApplyMultiplexer`, `CartesianProduct`, `Union`, `Argument` |
//! | **Transform**| `Project`, `Aggregate`, `Unwind`, `PathBuilder`, `LoadCsv`, `ProcedureCall` |

pub mod aggregate;
pub mod all_shortest_paths;
pub mod apply;
pub(crate) mod batched_result_emitter;
pub mod cartesian_product;
pub mod commit;
pub mod cond_traverse;
pub mod cond_var_len_traverse;
pub mod create;
pub mod delete;
pub mod distinct;
pub mod edge_by_fulltext_scan;
pub mod edge_by_index_scan;
pub mod edge_by_vector_scan;
pub mod expand_into;
pub mod filter;
pub mod foreach;
pub mod include_pending;
pub mod limit;
pub mod load_csv;
pub mod merge;
pub mod node_by_fulltext_scan;
pub mod node_by_id_seek;
pub mod node_by_index_scan;
pub mod node_by_label_and_id_scan;
pub mod node_by_label_scan;
pub mod node_by_vector_scan;
pub mod optional;
pub mod or_apply_multiplexer;
pub mod path_builder;
pub mod procedure_call;
pub mod project;
pub mod remove;
pub mod semi_apply;
pub mod set;
pub mod skip;
pub mod sort;
pub mod union;
pub mod unwind;
pub mod value_hash_join;

pub use aggregate::AggregateOp;
pub use all_shortest_paths::AllShortestPathsOp;
pub use apply::ApplyOp;
pub use cartesian_product::CartesianProductOp;
pub use commit::CommitOp;
pub use cond_traverse::CondTraverseOp;
pub use cond_var_len_traverse::CondVarLenTraverseOp;
pub use create::CreateOp;
pub use delete::DeleteOp;
pub use distinct::DistinctOp;
pub use edge_by_fulltext_scan::EdgeByFulltextScanOp;
pub use edge_by_index_scan::EdgeByIndexScanOp;
pub use edge_by_vector_scan::EdgeByVectorScanOp;
pub use expand_into::ExpandIntoOp;
pub use filter::FilterOp;
pub use foreach::ForEachOp;
pub use include_pending::IncludePendingOp;
pub use limit::LimitOp;
pub use load_csv::LoadCsvOp;
pub use merge::MergeOp;
pub use node_by_fulltext_scan::NodeByFulltextScanOp;
pub use node_by_id_seek::NodeByIdSeekOp;
pub use node_by_index_scan::NodeByIndexScanOp;
pub use node_by_label_and_id_scan::NodeByLabelAndIdScanOp;
pub use node_by_label_scan::NodeByLabelScanOp;
pub use node_by_vector_scan::NodeByVectorScanOp;
pub use optional::OptionalOp;
pub use or_apply_multiplexer::OrApplyMultiplexerOp;
pub use path_builder::PathBuilderOp;
pub use procedure_call::ProcedureCallOp;
pub use project::ProjectOp;
pub use remove::RemoveOp;
pub use semi_apply::SemiApplyOp;
pub use set::SetOp;
pub use skip::SkipOp;
pub use sort::SortOp;
pub use union::UnionOp;
pub use unwind::UnwindOp;
pub use value_hash_join::ValueHashJoinOp;

use std::collections::VecDeque;

use crate::graph::graph::{Graph, RelationshipId};
use crate::runtime::value::Value;

use super::{
    batch::{BATCH_SIZE, BatchBuilder},
    row::{Row, RowView},
};

/// Drain rows from `pending` into `builder` until `BATCH_SIZE` is reached
/// or all pending rows are exhausted.
///
/// Shared helper used by operators that buffer intermediate results
/// (CondTraverse, ExpandInto, Unwind, ForEach, CondVarLenTraverse, LoadCsv).
pub fn drain_pending(
    pending: &mut VecDeque<Row>,
    builder: &mut BatchBuilder,
) {
    while builder.len() < BATCH_SIZE {
        if let Some(row) = pending.pop_front() {
            builder.push_row(&row);
        } else {
            break;
        }
    }
}

/// Marks an entry of a `sibling_edges` list as a *collapsed* edge: one the
/// planner stopped emitting per edge, so it binds one arbitrary representative
/// of its parallel edges. The flag is the top bit of the alias id. The
/// collapsed edge matches a single relationship type, that of its
/// representative.
pub const COLLAPSED_SIBLING: u32 = 1 << 31;

/// Like [`COLLAPSED_SIBLING`], for a collapsed edge that matches every
/// relationship type (an untyped pattern). A collapsed edge matching several
/// but not all types is not flagged: its types are not known here.
pub const COLLAPSED_SIBLING_ANY_TYPE: u32 = 1 << 30;

/// Every flag bit of a `sibling_edges` entry.
const SIBLING_FLAGS: u32 = COLLAPSED_SIBLING | COLLAPSED_SIBLING_ANY_TYPE;

/// What an operator knows about the other edges it could have bound in place of
/// a collapsed edge's representative; see [`edge_already_used`].
pub struct Parallel<'a> {
    pub graph: &'a Graph,
    /// The tensor indices the expanding relationship matches when it is
    /// collapsed (`emit_relationship == false`), `None` when it emits.
    pub own_types: Option<&'a [usize]>,
    /// Relationships deleted by the running query, which are not alternatives.
    pub deleted: &'a dyn Fn(RelationshipId) -> bool,
}

/// The relationship types an alternative edge may have.
#[derive(Clone, Copy)]
enum Matching<'a> {
    /// These tensors.
    Tensors(&'a [usize]),
    /// The type of the edge in question.
    TypeOfEdge,
    /// Any type.
    AnyType,
}

impl Parallel<'_> {
    /// Does another live edge join the endpoints of `edge_id` with a type in
    /// `types`?
    fn has_alternative(
        &self,
        edge_id: RelationshipId,
        types: Matching,
    ) -> bool {
        let g = self.graph;
        let raw = u64::from(edge_id);
        let Some((src, dst)) = g.endpoints_for_edge(raw) else {
            return false;
        };
        // Live edges of the pair in tensor `tidx`; stops counting at two. Only
        // the stored direction counts: an undirected pattern reaches an edge
        // stored the other way round through its own row (the reversed pair),
        // not as a stand-in for this one.
        let live = |tidx: usize| -> usize {
            g.relationship_tensors().get(tidx).map_or(0, |tensor| {
                tensor
                    .get(src, dst)
                    .filter(|&other| !(self.deleted)(RelationshipId::from(other)))
                    .take(2)
                    .count()
            })
        };
        // No allocation: this runs on every uniqueness collision of a
        // collapsed edge.
        let mut total = 0;
        match types {
            Matching::Tensors(tensors) => {
                for &tidx in tensors {
                    total += live(tidx);
                    if total > 1 {
                        break;
                    }
                }
            }
            Matching::TypeOfEdge => {
                if let Some((_, tidx)) = g.relationship_type_matrix_iter(raw, raw).next() {
                    total = live(tidx as usize);
                }
            }
            Matching::AnyType => {
                for tidx in 0..g.relationship_tensors().len() {
                    total += live(tidx);
                    if total > 1 {
                        break;
                    }
                }
            }
        }
        total > 1
    }
}

/// Check whether a given edge ID is already bound to another relationship
/// variable in the environment.
///
/// Implements Cypher relationship uniqueness:
/// within a single MATCH clause, each relationship pattern must bind to a
/// distinct physical edge.
///
/// `own_alias_id` is the variable slot of the current relationship being
/// expanded — it is skipped during the scan. `parallel` describes the
/// alternatives a collapsed edge has (see below).
///
/// `sibling_edges` contains the alias IDs of other relationship variables
/// in the same MATCH clause component. Only these are checked — relationship
/// variables from other MATCH/OPTIONAL MATCH clauses are ignored per the
/// Cypher spec.
///
/// A collapsed edge stands for all the parallel edges of its (src, dst) pair
/// and binds an arbitrary one of them, so its representative colliding with
/// another edge says nothing when a parallel edge exists: some other choice
/// would not collide. Such a collision is ignored, which keeps the result from
/// depending on which parallel edge was picked. A parallel edge counts when it
/// is live and its type is one the collapsed pattern matches; when the pair
/// holds no such edge the representative is the only choice and the check
/// applies as usual.
#[must_use]
pub fn edge_already_used<R: RowView + ?Sized>(
    env: &R,
    parallel: &Parallel,
    edge_id: RelationshipId,
    own_alias_id: u32,
    sibling_edges: &[u32],
) -> bool {
    for &entry in sibling_edges {
        let sibling_id = entry & !SIBLING_FLAGS;
        if sibling_id == own_alias_id {
            continue;
        }
        if let Some(Value::Relationship(rel)) = env.value_at(sibling_id)
            && rel == edge_id
        {
            // The collision can be avoided when either side could have bound
            // another edge of the types it matches.
            let own_avoids = parallel
                .own_types
                .is_some_and(|types| parallel.has_alternative(edge_id, Matching::Tensors(types)));
            let sibling_avoids = if entry & COLLAPSED_SIBLING_ANY_TYPE != 0 {
                parallel.has_alternative(edge_id, Matching::AnyType)
            } else if entry & COLLAPSED_SIBLING != 0 {
                parallel.has_alternative(edge_id, Matching::TypeOfEdge)
            } else {
                false
            };
            if own_avoids || sibling_avoids {
                continue;
            }
            return true;
        }
    }
    false
}
