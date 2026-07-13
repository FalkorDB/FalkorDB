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

use crate::graph::graph::RelationshipId;
use crate::runtime::value::Value;

use super::{
    batch::{BATCH_SIZE, Batch, BatchBuilder},
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

pub fn drain_pending_batches(
    pending: &mut VecDeque<Batch<'_>>,
    builder: &mut BatchBuilder,
) {
    while builder.len() < BATCH_SIZE {
        if let Some(batch) = pending.pop_front() {
            // Count and index *active* rows only: a batch may carry a selection
            // vector, so `len()` (total rows) would overshoot the fit check and
            // raw `0..` indices could address filtered-out rows.
            if builder.len() + batch.active_len() <= BATCH_SIZE {
                // The whole batch fits: push every active row.
                for row in batch.active_indices() {
                    builder.push_batch_row(&batch, row, batch.origin_row(row));
                }
            } else {
                // Only part of the batch fits: push the first `remaining` active
                // rows and re-queue the rest (gathering by active index keeps the
                // selection honoured).
                let remaining = BATCH_SIZE - builder.len();
                let active: Vec<usize> = batch.active_indices().collect();
                for &row in &active[..remaining] {
                    builder.push_batch_row(&batch, row, batch.origin_row(row));
                }
                pending.push_front(batch.gather(&active[remaining..]));
                break;
            }
        } else {
            break;
        }
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
/// expanded — it is skipped during the scan.
///
/// `sibling_edges` contains the alias IDs of other relationship variables
/// in the same MATCH clause component. Only these are checked — relationship
/// variables from other MATCH/OPTIONAL MATCH clauses are ignored per the
/// Cypher spec.
#[must_use]
pub fn edge_already_used<R: RowView + ?Sized>(
    env: &R,
    edge_id: RelationshipId,
    own_alias_id: u32,
    sibling_edges: &[u32],
) -> bool {
    for &sibling_id in sibling_edges {
        if sibling_id == own_alias_id {
            continue;
        }
        if let Some(Value::Relationship(rel)) = env.value_at(sibling_id)
            && rel == edge_id
        {
            return true;
        }
    }
    false
}
