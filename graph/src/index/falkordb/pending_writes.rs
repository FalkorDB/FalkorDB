//! Native-index changes a unit of work has staged but not yet applied.
//!
//! The graph's write paths record into the [`PendingIndexWrites`] carried by the unit of work's
//! [`IndexDocs`](crate::index::index_docs::IndexDocs) instead of changing the index themselves,
//! and whoever owns the unit of work applies it once, where it publishes RediSearch's documents:
//! a Cypher query at the end of each `Commit` segment, a replicated effects buffer at the end of
//! the buffer, `GRAPH.BULK` after each token. One application per unit instead of one per write
//! call, and the functions in [`graph_writes`](super::graph_writes) need only read the graph.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{entity_type::EntityType, runtime::value::Value};

use super::falkordb_index::{FalkorDbIndex, IndexKey};

/// Per column, its staged changes in the order they were made.
pub(crate) type ColumnWrites = HashMap<IndexKey, Vec<IndexWrite>>;

/// One staged change to one column: an encoded `(key, id)` tuple going in or coming out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IndexWrite {
    pub(crate) key: u64,
    pub(crate) id: u64,
    pub(crate) add: bool,
}

/// Native-index changes staged by graph writes, applied once by their owner.
///
/// **Order is kept, and only the last change to a tuple counts.** One unit of work can add a tuple
/// and later remove it: a write step that sets `n.v` and then deletes `n` stages `add (v, n)` from
/// the SET and `remove (v, n)` from the DELETE, because `Pending::commit` applies its SETs before
/// its DELETEs. Applying all removes and then all adds — what each write call could safely do on
/// its own — would leave the deleted node indexed. [`apply`](Self::apply) therefore collapses each
/// tuple's history to its final change first.
#[derive(Default)]
pub struct PendingIndexWrites {
    nodes: ColumnWrites,
    edges: ColumnWrites,
}

impl PendingIndexWrites {
    /// Whether nothing is staged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }

    /// Apply every staged change to `index` and leave the buffer empty.
    ///
    /// `index` must belong to the private version the changes were staged against: the index is
    /// part of the graph version, so a version that is later refused takes these changes down
    /// with it.
    pub fn apply(
        &mut self,
        index: &mut FalkorDbIndex,
    ) {
        if self.is_empty() {
            return;
        }
        index.apply_writes(EntityType::Node, self.nodes.drain());
        index.apply_writes(EntityType::Relationship, self.edges.drain());
    }

    /// Drop everything staged without applying it — the unit of work failed.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
    }

    /// Stage `(value, id)` going into or out of column `(label, attr)`, if `index` has that
    /// column and the column indexes `value`. Encoded now, so the buffer holds 17 bytes per
    /// change and no `Value`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &mut self,
        index: &FalkorDbIndex,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        value: &Value,
        id: u64,
        add: bool,
    ) {
        let Some(key) = index.encode_value(entity, label, attr, value) else {
            return;
        };
        let columns = match entity {
            EntityType::Node => &mut self.nodes,
            EntityType::Relationship => &mut self.edges,
        };
        columns
            .entry((label.clone(), attr.clone()))
            .or_default()
            .push(IndexWrite { key, id, add });
    }
}

/// Collapse one column's staged changes to the last change per `(key, id)`, writing the tuples
/// that end up removed to `removes` and those that end up added to `adds`, both sorted. Each is
/// cleared first, so the caller can reuse their capacity across columns.
///
/// The sort is stable, so equal tuples keep the order they were staged in and the last of each
/// run is the latest change.
pub(crate) fn last_write_wins(
    log: &mut [IndexWrite],
    removes: &mut Vec<(u64, u64)>,
    adds: &mut Vec<(u64, u64)>,
) {
    removes.clear();
    adds.clear();
    log.sort_by_key(|w| (w.key, w.id));
    for (i, w) in log.iter().enumerate() {
        let superseded = log
            .get(i + 1)
            .is_some_and(|next| (next.key, next.id) == (w.key, w.id));
        if superseded {
            continue;
        }
        if w.add {
            adds.push((w.key, w.id));
        } else {
            removes.push((w.key, w.id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(
        key: u64,
        id: u64,
        add: bool,
    ) -> IndexWrite {
        IndexWrite { key, id, add }
    }

    #[test]
    fn the_last_change_to_a_tuple_wins() {
        let (mut removes, mut adds) = (Vec::new(), Vec::new());
        // SET n.v = 5 then DELETE n: add then remove of (5, n). The remove must win.
        let mut log = vec![w(1, 7, false), w(5, 7, true), w(5, 7, false)];
        last_write_wins(&mut log, &mut removes, &mut adds);
        assert_eq!(removes, vec![(1, 7), (5, 7)]);
        assert!(
            adds.is_empty(),
            "the deleted node's tuple must not be added"
        );

        // v0 -> v1 -> v0: remove then add of (1, 7). The add must win.
        let mut log = vec![w(1, 7, false), w(2, 7, true), w(2, 7, false), w(1, 7, true)];
        last_write_wins(&mut log, &mut removes, &mut adds);
        assert_eq!(removes, vec![(2, 7)]);
        assert_eq!(adds, vec![(1, 7)]);
    }

    #[test]
    fn untouched_tuples_pass_through_sorted() {
        let (mut removes, mut adds) = (vec![(9, 9)], vec![(9, 9)]);
        let mut log = vec![w(3, 1, true), w(1, 2, false), w(2, 1, true)];
        last_write_wins(&mut log, &mut removes, &mut adds);
        assert_eq!(removes, vec![(1, 2)], "cleared first, then filled");
        assert_eq!(adds, vec![(2, 1), (3, 1)]);
    }
}
