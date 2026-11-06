//! Native-index changes a unit of work has staged but not yet applied.
//!
//! Each write function in [`graph_writes`](super::graph_writes) reads the graph and returns the
//! [`IndexChanges`] its write makes; it holds no state. The graph's write method that called it
//! adds them to the [`PendingIndexWrites`] carried by the unit of work's
//! [`IndexDocs`](crate::index::index_docs::IndexDocs), next to the RediSearch documents it fills
//! there, and whoever owns the unit of work applies it once, where it publishes those documents:
//! a Cypher query at the end of each `Commit` segment, a replicated effects buffer at the end of
//! the buffer, `GRAPH.BULK` after each token. One application per unit instead of one per write
//! call.

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

/// The native-index changes one graph write makes, in the order it makes them: what each write
/// function in [`graph_writes`](super::graph_writes) returns. It only describes the changes;
/// [`PendingIndexWrites::extend`] stages them.
pub struct IndexChanges {
    entity: EntityType,
    changes: Vec<(IndexKey, IndexWrite)>,
}

impl IndexChanges {
    /// No changes yet, for one entity kind. Allocates nothing.
    #[must_use]
    pub fn new(entity: EntityType) -> Self {
        Self {
            entity,
            changes: Vec::new(),
        }
    }

    /// Whether the write changes nothing in the index.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// `(value, id)` comes out of column `(label, attr)`, if `index` has that column and the
    /// column indexes `value`.
    pub(crate) fn remove(
        &mut self,
        index: &FalkorDbIndex,
        label: &Arc<String>,
        attr: &Arc<String>,
        value: &Value,
        id: u64,
    ) {
        self.push(index, label, attr, value, id, false);
    }

    /// `(value, id)` goes into column `(label, attr)`, if `index` has that column and the column
    /// indexes `value`.
    pub(crate) fn add(
        &mut self,
        index: &FalkorDbIndex,
        label: &Arc<String>,
        attr: &Arc<String>,
        value: &Value,
        id: u64,
    ) {
        self.push(index, label, attr, value, id, true);
    }

    /// Encoded now, so a change holds 17 bytes of payload and no `Value`.
    fn push(
        &mut self,
        index: &FalkorDbIndex,
        label: &Arc<String>,
        attr: &Arc<String>,
        value: &Value,
        id: u64,
        add: bool,
    ) {
        let Some(key) = index.encode_value(self.entity, label, attr, value) else {
            return;
        };
        self.changes
            .push(((label.clone(), attr.clone()), IndexWrite { key, id, add }));
    }
}

/// Native-index changes staged by graph writes, applied once by their owner.
///
/// **Order is kept, and only the last change to a tuple counts.** One unit of work can add a tuple
/// and later remove it: a write step that sets `n.v` and then deletes `n` stages `add (v, n)` from
/// the SET and `remove (v, n)` from the DELETE, because `Pending::commit` applies its SETs before
/// its DELETEs. Applying all removes and then all adds — what each write call could safely do on
/// its own — would leave the deleted node indexed. [`FalkorDbIndex::apply`] therefore collapses
/// each tuple's history to its final change first.
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

    /// Take every staged change for `entity`, column by column, leaving none behind but keeping
    /// the buffer's capacity for the next unit of work. What [`FalkorDbIndex::apply`] consumes.
    pub(crate) fn drain(
        &mut self,
        entity: EntityType,
    ) -> impl Iterator<Item = (IndexKey, Vec<IndexWrite>)> + '_ {
        match entity {
            EntityType::Node => self.nodes.drain(),
            EntityType::Relationship => self.edges.drain(),
        }
    }

    /// Drop everything staged without applying it — the unit of work failed.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
    }

    /// Stage one write's changes after everything already staged, keeping their order.
    pub fn extend(
        &mut self,
        changes: IndexChanges,
    ) {
        let columns = match changes.entity {
            EntityType::Node => &mut self.nodes,
            EntityType::Relationship => &mut self.edges,
        };
        for (key, write) in changes.changes {
            columns.entry(key).or_default().push(write);
        }
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

    /// Long runs of changes to the same tuples, interleaved with each other and with tuples changed
    /// once: each tuple must end at its last change, which only a sort that keeps the order of
    /// equal elements guarantees. Short logs are sorted by insertion either way, so this one is
    /// long.
    #[test]
    fn the_last_change_wins_in_a_long_log() {
        let (mut removes, mut adds) = (Vec::new(), Vec::new());
        let mut log = Vec::new();
        for round in 0..40 {
            for tuple in 0..10 {
                log.push(w(tuple, 7, (round + tuple) % 2 == 0));
            }
            log.push(w(100 + round, 8, true));
        }
        last_write_wins(&mut log, &mut removes, &mut adds);
        // Round 39 is the last: tuple t ends added when 39 + t is even.
        let ends = |added: bool| -> Vec<(u64, u64)> {
            (0..10)
                .filter(|t| ((39 + t) % 2 == 0) == added)
                .map(|t| (t, 7))
                .collect()
        };
        assert_eq!(removes, ends(false));
        let mut want = ends(true);
        want.extend((0..40).map(|round| (100 + round, 8)));
        assert_eq!(adds, want);
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
