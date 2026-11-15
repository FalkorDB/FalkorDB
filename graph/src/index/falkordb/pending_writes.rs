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

use super::encode::KeyKind;
use super::falkordb_index::{FalkorDbIndex, IndexKey};
use super::numeric::EncodedTuples;

/// Per column, its staged changes in the order they were made.
pub(crate) type ColumnWrites = HashMap<IndexKey, Vec<IndexWrite>>;

/// One staged change to one column: an encoded `(key, id)` tuple going in or coming out of the
/// column's tree for `kind`.
///
/// `kind` is part of the tuple's identity. A scalar `1` and a list element `1` on the same entity
/// are different tuples in different trees, so a value that changes kind within one unit of work
/// stages a remove from one tree and an add to the other, and neither may cancel the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IndexWrite {
    pub(crate) key: u64,
    pub(crate) id: u64,
    pub(crate) kind: KeyKind,
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

    /// Encoded now, so a change holds 18 bytes of payload and no `Value`. A list value stages one
    /// change per numeric element, each into the column's array tree.
    fn push(
        &mut self,
        index: &FalkorDbIndex,
        label: &Arc<String>,
        attr: &Arc<String>,
        value: &Value,
        id: u64,
        add: bool,
    ) {
        let changes = &mut self.changes;
        index.for_each_stored_key(self.entity, label, attr, value, |kind, key| {
            changes.push((
                (label.clone(), attr.clone()),
                IndexWrite { key, id, kind, add },
            ));
        });
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

/// Collapse one column's staged changes to the last change per `(kind, key, id)`, writing the
/// tuples that end up removed to `removes` and those that end up added to `adds`, each with the
/// tuples of its kind, sorted. Both are cleared first, so the caller can reuse their capacity
/// across columns.
///
/// The kind is part of the tuple: collapsing on `(key, id)` alone would let a scalar `1` turning
/// into the list `[1]` cancel itself (the remove from the scalar tree and the add to the array
/// tree are the same `(key, id)`) and leave the tuple in the wrong tree.
///
/// The sort is stable, so equal tuples keep the order they were staged in and the last of each
/// run is the latest change.
pub(crate) fn last_write_wins(
    log: &mut [IndexWrite],
    removes: &mut EncodedTuples,
    adds: &mut EncodedTuples,
) {
    removes.clear();
    adds.clear();
    let tuple = |w: &IndexWrite| (w.kind, w.key, w.id);
    log.sort_by_key(tuple);
    for (i, w) in log.iter().enumerate() {
        let superseded = log.get(i + 1).is_some_and(|next| tuple(next) == tuple(w));
        if superseded {
            continue;
        }
        let side = if w.add { &mut *adds } else { &mut *removes };
        side[w.kind].push((w.key, w.id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A change to the scalar tree.
    fn w(
        key: u64,
        id: u64,
        add: bool,
    ) -> IndexWrite {
        IndexWrite {
            key,
            id,
            kind: KeyKind::Scalar,
            add,
        }
    }

    /// A change to the array tree.
    fn arr(
        key: u64,
        id: u64,
        add: bool,
    ) -> IndexWrite {
        IndexWrite {
            kind: KeyKind::Array,
            ..w(key, id, add)
        }
    }

    fn scalars(tuples: &[(u64, u64)]) -> EncodedTuples {
        EncodedTuples::scalars(tuples.to_vec())
    }

    #[test]
    fn the_last_change_to_a_tuple_wins() {
        let (mut removes, mut adds) = (EncodedTuples::default(), EncodedTuples::default());
        // SET n.v = 5 then DELETE n: add then remove of (5, n). The remove must win.
        let mut log = vec![w(1, 7, false), w(5, 7, true), w(5, 7, false)];
        last_write_wins(&mut log, &mut removes, &mut adds);
        assert_eq!(removes, scalars(&[(1, 7), (5, 7)]));
        assert!(
            adds.is_empty(),
            "the deleted node's tuple must not be added"
        );

        // v0 -> v1 -> v0: remove then add of (1, 7). The add must win.
        let mut log = vec![w(1, 7, false), w(2, 7, true), w(2, 7, false), w(1, 7, true)];
        last_write_wins(&mut log, &mut removes, &mut adds);
        assert_eq!(removes, scalars(&[(2, 7)]));
        assert_eq!(adds, scalars(&[(1, 7)]));
    }

    /// Long runs of changes to the same tuples, interleaved with each other and with tuples changed
    /// once: each tuple must end at its last change, which only a sort that keeps the order of
    /// equal elements guarantees. Short logs are sorted by insertion either way, so this one is
    /// long.
    #[test]
    fn the_last_change_wins_in_a_long_log() {
        let (mut removes, mut adds) = (EncodedTuples::default(), EncodedTuples::default());
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
        assert_eq!(removes, scalars(&ends(false)));
        let mut want = ends(true);
        want.extend((0..40).map(|round| (100 + round, 8)));
        assert_eq!(adds, scalars(&want));
    }

    #[test]
    fn untouched_tuples_pass_through_sorted() {
        let (mut removes, mut adds) = (scalars(&[(9, 9)]), scalars(&[(9, 9)]));
        let mut log = vec![w(3, 1, true), w(1, 2, false), w(2, 1, true)];
        last_write_wins(&mut log, &mut removes, &mut adds);
        assert_eq!(removes, scalars(&[(1, 2)]), "cleared first, then filled");
        assert_eq!(adds, scalars(&[(2, 1), (3, 1)]));
    }

    /// A value changing kind inside one unit of work. `SET n.v = [1, 2]` on a node whose `v` was
    /// the scalar `1` stages `remove scalar (1, n)` and `add array (1, n)`, `add array (2, n)`:
    /// the same `(key, id)` leaving one tree and entering the other. Collapsing on `(key, id)`
    /// alone would keep only the later change, so the scalar tuple would never be removed and
    /// `n.v = 1` would keep matching a node whose `v` is now a list.
    #[test]
    fn a_change_of_kind_keeps_both_trees_changes() {
        let (mut removes, mut adds) = (EncodedTuples::default(), EncodedTuples::default());
        let mut log = vec![w(1, 7, false), arr(1, 7, true), arr(2, 7, true)];
        last_write_wins(&mut log, &mut removes, &mut adds);
        assert_eq!(
            removes[KeyKind::Scalar],
            vec![(1, 7)],
            "the scalar tuple comes out"
        );
        assert!(removes[KeyKind::Array].is_empty());
        assert!(adds[KeyKind::Scalar].is_empty());
        assert_eq!(
            adds[KeyKind::Array],
            vec![(1, 7), (2, 7)],
            "the elements go in"
        );

        // And back: `SET n.v = 1` on `[1, 2]`.
        let mut log = vec![arr(1, 7, false), arr(2, 7, false), w(1, 7, true)];
        last_write_wins(&mut log, &mut removes, &mut adds);
        assert_eq!(removes[KeyKind::Array], vec![(1, 7), (2, 7)]);
        assert_eq!(adds[KeyKind::Scalar], vec![(1, 7)]);
        assert!(removes[KeyKind::Scalar].is_empty() && adds[KeyKind::Array].is_empty());
    }

    /// The kind rides in what used to be padding: a staged change is no bigger than before.
    #[test]
    fn a_staged_change_did_not_grow() {
        assert_eq!(std::mem::size_of::<IndexWrite>(), 24);
    }
}
