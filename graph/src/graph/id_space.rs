//! Validating an id space across a batch of creates and deletes.
//!
//! Written for nodes and used for relationships too: both are counted ids with a
//! recycle bin, and neither the invariant below nor the checks that enforce it
//! mention which. The caller says which entity it is asking about when it renders
//! a refusal.
//!
//! ## The invariant, and why the graph cannot check it alone
//!
//! `Graph` tracks liveness with `node_count` and `deleted_nodes` and no
//! live set, so the only boundary it can offer is
//! `node_count + deleted_nodes.len()`. That is the boundary *if the id space is
//! dense*, and `max_node_id` is the same arithmetic — so an assertion written in
//! terms of them checks one derivation of the state against another derivation
//! of the same state, and passes on a graph that is already wrong.
//!
//! Density holds between batches, not within one. The effects apply path ingests
//! records grouped by shape rather than ordered by id, so a create of 500..600
//! may precede one of 0..500; between them the graph holds 100 live nodes whose
//! highest id is 599 and reports its boundary as 100.
//!
//! ## What this holds
//!
//! The boundary as it stood before the batch, and every id at or above it that
//! the batch has created. That is enough to state the invariant directly:
//!
//! ```text
//! created == [ entry_bound, entry_bound + created.len() )
//! ```
//!
//! Two independent things can break it, so [`IdSpace::verify`] checks both.
//!
//! **The shape of what arrived.** Ids created at or above the boundary must
//! fill the range from it upward with no hole. An allocator hands out the lowest
//! free id, so it cannot reach an id without having handed out everything below
//! — a batch that leaves a hole did not come from one. This is the check that
//! catches a replica which has missed a buffer: told to create 500..600 when its
//! own boundary is 0, it would otherwise accept an id space its master does not
//! have and collide on its next allocation.
//!
//! **That the graph counted it.** `node_count` is an independent counter, and
//! comparing the graph's own boundary against `entry_bound + created.len()` is
//! the only place anything checks it against a value not derived from it.
//!
//! This is a backstop rather than the front line. The two ways an id can be
//! wrong — already live, or claimed twice by this batch — are both refused at
//! the record by [`IdSpace::create`], which names the id. What is left
//! for the count is anything that moves `node_count` without going through the
//! operations that maintain the batch, which is to say a path that does not
//! exist today and would be a mistake tomorrow.
//!
//! Nothing here knows about replication, buffers or a peer engine. It is a
//! statement about one graph and the ids handed to it.

use roaring::RoaringTreemap;
use thiserror::Error;

/// Why a batch of ids does not describe a possible id space.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IdSpaceError {
    /// A delete named an id at or above the entry boundary that this batch never
    /// created, so it was never allocated here at all.
    #[error("{0} was never allocated here")]
    NeverCreated(u64),

    /// Ids are missing between the entry boundary and the highest one created.
    #[error("{entry_bound}..={highest} were allocated but only {created} of them created")]
    Hole {
        entry_bound: u64,
        highest: u64,
        created: u64,
    },

    /// The graph's own boundary is not where the created ids put it — it
    /// counted something twice, or not at all.
    #[error("the graph's boundary is {graph_bound}, but the ids created put it at {expected}")]
    Miscounted { graph_bound: u64, expected: u64 },

    /// Already free: it is in the recycle bin, so there is nothing to delete.
    #[error("{0} is already in the recycle bin")]
    AlreadyRecycled(u64),

    /// An id below the boundary and not free: it was handed out before the
    /// batch and has not been given back, so it is live.
    #[error("{id} is already live below the boundary {entry_bound}")]
    AlreadyLive { id: u64, entry_bound: u64 },

    /// A batch named `u64::MAX`, which has no boundary above it.
    #[error("{0} is past the end of the id space")]
    IdOutOfRange(u64),
}

/// An id space, observed across one batch of creates and deletes.
///
/// Built before the batch, told what it creates, and checked once at the end.
/// It decides nothing while the batch runs — [`Graph::create_nodes`] already
/// refuses an id that is live below the boundary — so this records and then
/// judges, rather than gating each step.
pub struct IdSpace {
    /// The boundary as it stood before the batch: ids below it were handed out,
    /// ids at or above it never were.
    entry_bound: u64,
    /// Every id at or above `entry_bound` this batch has created. Only grows —
    /// a later delete does not un-allocate an id, it frees one that was.
    created: RoaringTreemap,
}

impl IdSpace {
    /// The id space as it stands before a batch.
    ///
    /// `node_count + deleted_nodes_count` is the boundary *because* this is
    /// taken between batches, where the id space is dense. That assumption is
    /// stated once, here, rather than implied by an accessor every caller has to
    /// know not to trust.
    ///
    /// Not `max_node_id() + 1`, though the arithmetic agrees whenever the graph
    /// holds a live node: `max_node_id` returns a 0 *sentinel* when it holds
    /// none, which is indistinguishable from a graph whose highest id is 0 and
    /// reads as "id 0 has been handed out".
    #[must_use]
    pub(crate) fn at(entry_bound: u64) -> Self {
        Self {
            entry_bound,
            created: RoaringTreemap::new(),
        }
    }

    /// Refuse ids that are already free.
    ///
    /// Associated rather than a method, because this half of "is this node live"
    /// needs the recycle bin and nothing else — no boundary, so no batch. That
    /// is what lets the write path, which never opens one, be answered by the
    /// same code as the effects path instead of a second copy of it.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::AlreadyRecycled`] for the lowest id already in the bin.
    pub(crate) fn refuse_recycled(
        nodes: &RoaringTreemap,
        recycled: &RoaringTreemap,
    ) -> Result<(), IdSpaceError> {
        match (nodes & recycled).min() {
            Some(id) => Err(IdSpaceError::AlreadyRecycled(id)),
            None => Ok(()),
        }
    }

    /// Record `nodes` as created by this batch, or refuse them and record
    /// nothing.
    ///
    /// `recycled` is the ids that are free right now — the graph's recycle bin.
    /// An id in it is free whatever its value, so it is neither live nor a
    /// duplicate, and removing it is what lets the two checks below tell a
    /// genuine double claim from a legitimate recreate.
    ///
    /// One operation rather than a check and a record the caller sequences,
    /// because the order between them is load-bearing: recording first would
    /// leave the duplicate check finding the ids it had just inserted. Merged,
    /// there is no order for a caller to get wrong and no way to record without
    /// checking.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::AlreadyLive`] for the lowest id that is already live —
    /// either handed out before the batch, or claimed earlier within it.
    ///
    /// [`IdSpaceError::IdOutOfRange`] for `u64::MAX`. Nothing can be allocated
    /// above it, and letting it through would wrap the arithmetic in
    /// [`Self::verify`].
    pub(crate) fn record_created(
        &mut self,
        nodes: &RoaringTreemap,
        recycled: &RoaringTreemap,
    ) -> Result<(), IdSpaceError> {
        if nodes.contains(u64::MAX) {
            return Err(IdSpaceError::IdOutOfRange(u64::MAX));
        }

        // What is not free: the only ids either check can object to.
        let claimed = nodes - recycled;

        // Live before the batch: below the boundary and not free.
        if let Some(id) = claimed.min().filter(|&id| id < self.entry_bound) {
            return Err(IdSpaceError::AlreadyLive {
                id,
                entry_bound: self.entry_bound,
            });
        }
        // Live because *this batch* created it. `created` alone cannot say that
        // — it still holds ids the batch has since deleted, and recreating one of
        // those is legitimate, which is how a multi-commit query reaches a
        // replica. The recycle bin is what separates them, and it is gone from
        // `claimed`: an id this batch created and then deleted is in the bin, so
        // anything left that `created` also holds was claimed twice.
        //
        // `is_disjoint` first, so the ordinary record answers from container
        // headers and only a genuine duplicate pays for the intersection.
        if !self.created.is_disjoint(&claimed) {
            let id = (&claimed & &self.created)
                .min()
                .expect("the sets are not disjoint");
            return Err(IdSpaceError::AlreadyLive {
                id,
                entry_bound: self.entry_bound,
            });
        }

        // Both arms are the same operation — union in the ids at or above the
        // boundary — and differ only in whether anything has to be trimmed first.
        // A recycled id sits below the boundary and was already counted in it, so
        // it is not the id space growing.
        //
        // The allocator hands out recycled ids before fresh ones, so a single
        // create can carry both and neither arm is unusual. Trimming a copy
        // rather than filtering the iterator keeps the slow arm a container
        // merge too, instead of an insert per id.
        if nodes.min().is_some_and(|min| min >= self.entry_bound) {
            self.created |= nodes;
        } else {
            let mut above = nodes.clone();
            above.remove_range(..self.entry_bound);
            self.created |= above;
        }
        Ok(())
    }

    /// Check that ids the batch is about to delete were allocated here.
    ///
    /// The half that needs a batch. The other half — whether the id is already
    /// free — is [`Self::refuse_recycled`], which needs only the bin, so a caller
    /// with no batch can still ask it. Together they are liveness.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::NeverCreated`] for the highest id at or above the entry
    /// boundary that this batch did not create. Above the boundary it was never
    /// allocated before the batch either, so nothing has ever held it.
    pub(crate) fn refuse_undeletable(
        &self,
        nodes: &RoaringTreemap,
    ) -> Result<(), IdSpaceError> {
        // Nothing at or above the boundary means nothing this type can object
        // to, and that is the ordinary delete — of nodes that predate the batch
        // entirely. One `max` settles it without touching the sets.
        if nodes.max().is_none_or(|max| max < self.entry_bound) {
            return Ok(());
        }
        // Otherwise a set difference, which is a container merge. Walking the
        // ids and probing `created` per id reads as the cheaper option and is
        // not: measured over 100k ids, the walk costs 561us where the difference
        // costs 1.1us when every id was created by this batch, and 224us against
        // 966ns on a mixed delete. It only wins on the shape the guard above has
        // already returned from.
        match (nodes - &self.created)
            .max()
            .filter(|&id| id >= self.entry_bound)
        {
            Some(id) => Err(IdSpaceError::NeverCreated(id)),
            None => Ok(()),
        }
    }

    /// Check that the batch left a possible id space behind.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::Hole`] if the created ids do not fill the range from the
    /// entry boundary upward, and [`IdSpaceError::Miscounted`] if the graph's own
    /// boundary disagrees with where those ids put it. See the module docs for
    /// why both are needed.
    pub(crate) fn verify(
        &self,
        graph_bound: u64,
    ) -> Result<(), IdSpaceError> {
        let created = self.created.len();
        // `created == [entry_bound, entry_bound + created.len())`, spelled as the
        // two things that make it true: the set starts at the boundary, and it has
        // no gap between there and its highest id.
        //
        // Both are checked rather than one derived from the other. That the lowest
        // id is the boundary does follow from the second equality — every recorded
        // id is at or above the boundary, so `len` of them spanning exactly `len`
        // slots can start nowhere else — but that argument leans on an invariant
        // `created` maintains, and a check here should not have to know what
        // another function promises. Stated outright it holds regardless.
        //
        // Subtractions rather than `highest - entry_bound + 1 == len`: the set is
        // non-empty in this branch, so neither side can wrap, where the `+ 1`
        // form can.
        //
        // The order of the two tests is load-bearing for that. `||`
        // short-circuits, so `highest - entry_bound` is only reached once the
        // lowest id is known to *be* the boundary — and the lowest cannot exceed
        // the highest. Swap them and a set sitting entirely below the boundary,
        // which is the case the first test exists to catch, underflows instead.
        if let Some((_, highest)) =
            self.created
                .min()
                .zip(self.created.max())
                .filter(|&(lowest, highest)| {
                    lowest != self.entry_bound || created - 1 != highest - self.entry_bound
                })
        {
            return Err(IdSpaceError::Hole {
                entry_bound: self.entry_bound,
                highest,
                created,
            });
        }

        let expected = self
            .entry_bound
            .checked_add(created)
            .ok_or(IdSpaceError::IdOutOfRange(u64::MAX))?;
        if graph_bound != expected {
            return Err(IdSpaceError::Miscounted {
                graph_bound,
                expected,
            });
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::{IdSpace, IdSpaceError};
    use crate::graph::graph::{Graph, NodeOpError};
    use crate::graph::graphblas::test_init::ensure_init;
    use roaring::RoaringTreemap;
    use rustc_hash::FxHashMap;

    fn graph() -> Graph {
        ensure_init();
        Graph::new(64, 64, 0, 0, "t")
    }

    fn ids(v: &[u64]) -> RoaringTreemap {
        v.iter().copied().collect()
    }

    fn range(r: std::ops::Range<u64>) -> RoaringTreemap {
        r.collect()
    }

    /// Through the graph's own operations, because that is where the batch is
    /// maintained — a test that fed the batch directly would not exercise the
    /// thing that makes a caller unable to forget.
    fn create(
        g: &mut Graph,
        space: &mut IdSpace,
        nodes: &RoaringTreemap,
    ) -> Result<(), NodeOpError> {
        g.create_nodes(nodes, space)
    }

    fn delete(
        g: &mut Graph,
        space: &IdSpace,
        nodes: &RoaringTreemap,
    ) -> Result<(), NodeOpError> {
        g.delete_nodes(nodes, &mut FxHashMap::default(), Some(space))
            .map(|_| ())
    }

    #[test]
    fn ids_arriving_out_of_order_still_fill_the_range() {
        // The case a boundary derived from the counter cannot express: the high
        // half arrives first, so mid-batch the graph reports its boundary as 100
        // while id 599 is allocated.
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &range(500..600)).expect("the high half is legitimate");
        create(&mut g, &mut space, &range(0..500)).expect("and so is the low half");
        space
            .verify(g.node_id_bound())
            .expect("the batch closed the hole");
        assert_eq!(g.node_count(), 600);
    }

    #[test]
    fn a_hole_left_at_the_end_is_reported() {
        // 500..600 and nothing else. An allocator hands out the lowest free id,
        // so it cannot reach 500 without having handed out 0..499 — whoever
        // produced this was not working from the same id space.
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &range(500..600)).expect("nothing is wrong yet");

        let err = space
            .verify(g.node_id_bound())
            .expect_err("must report the hole");
        assert_eq!(
            err,
            IdSpaceError::Hole {
                entry_bound: 0,
                highest: 599,
                created: 100,
            }
        );
    }

    #[test]
    fn a_node_the_batch_never_recorded_is_caught_by_the_counter() {
        // The half of the invariant the shape check cannot see. `created` stays
        // contiguous from the boundary, so `Hole` is satisfied and says nothing —
        // but the graph counted a node the batch never recorded, and only
        // comparing its own boundary against `entry_bound + created.len()`
        // notices. That is why both checks are here rather than one.
        //
        // The extra node arrives by the write path, which records nothing into a
        // batch. On a replica the same shape is an applied record the batch did
        // not account for.
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &range(0..3)).expect("three ids, all recorded");
        space
            .verify(g.node_id_bound())
            .expect("the graph and the batch agree so far");

        g.create_allocated_nodes(&ids(&[3]));

        let err = space
            .verify(g.node_id_bound())
            .expect_err("the counter moved and the recorded ids did not");
        assert_eq!(
            err,
            IdSpaceError::Miscounted {
                graph_bound: 4,
                expected: 3,
            }
        );
    }

    #[test]
    fn a_range_that_starts_above_the_boundary_is_a_hole() {
        // The tightest form of the same failure, and why the lowest id is checked
        // rather than derived: {1,2,3} is internally consecutive, so a test of
        // `highest - lowest` alone would pass it. Measured from the boundary it is
        // one short, and id 0 is the id nobody accounted for.
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &ids(&[1, 2, 3])).expect("the graph takes them");

        let err = space
            .verify(g.node_id_bound())
            .expect_err("id 0 is unaccounted for");
        assert_eq!(
            err,
            IdSpaceError::Hole {
                entry_bound: 0,
                highest: 3,
                created: 3,
            }
        );
    }

    #[test]
    fn an_id_claimed_twice_is_refused_at_the_record() {
        // The set absorbs the duplicate, so the range still looks whole and the
        // count check would only report a discrepancy at the end. Intersecting
        // the incoming ids with what the batch has already created names the id
        // where it happens — and does not fire on a legitimate recreate, because
        // the caller has removed the recycle bin first and a recreated id is in
        // it. See `creating_deleting_and_recreating_one_id_in_a_batch`.
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &range(0..6)).expect("six fresh ids");

        let err = create(&mut g, &mut space, &ids(&[5])).expect_err("5 is already this batch's");
        assert_eq!(
            err,
            NodeOpError::IdSpace(IdSpaceError::AlreadyLive {
                id: 5,
                entry_bound: 0
            })
        );
        assert_eq!(g.node_count(), 6, "and nothing was applied for it");
    }

    #[test]
    fn a_recycled_id_does_not_grow_the_space() {
        // Ids below the entry boundary were counted in the boundary the batch
        // started from, so recreating one leaves the range and the count alone.
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &ids(&[0, 1])).expect("two fresh ids");
        delete(&mut g, &space, &ids(&[0])).expect("deleting what this batch created");
        space.verify(g.node_id_bound()).expect("whole");

        // A fresh batch: id 0 sits in the recycle bin, so it is free to come
        // back and does not extend the range — which is only true if the bin is
        // part of the boundary this batch started from.
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &ids(&[0])).expect("id 0 is free");
        space.verify(g.node_id_bound()).expect("whole again");
    }

    #[test]
    fn creating_deleting_and_recreating_one_id_in_a_batch() {
        // Three commits of one query reach a replica as a single buffer, and the
        // allocator hands the freed id straight back, so `C(0) · D(0) · C(0)` is
        // legitimate. The set does not shrink on the delete, so the recreate adds
        // nothing and the range stays whole.
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &ids(&[0])).expect("create");
        delete(&mut g, &space, &ids(&[0])).expect("delete");
        create(&mut g, &mut space, &ids(&[0])).expect("the recreate is legitimate");
        space.verify(g.node_id_bound()).expect("whole");
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn a_cancelled_reservation_arrives_as_a_pair() {
        // What a create-then-delete in one segment ships: both ids created, then
        // one deleted. The id space grew by two and one of them is free.
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &ids(&[0, 1])).expect("create");
        delete(&mut g, &space, &ids(&[0])).expect("delete");
        space.verify(g.node_id_bound()).expect("whole");
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.deleted_nodes_count(), 1);
    }

    #[test]
    fn deleting_the_same_id_twice_inside_one_batch_is_refused() {
        // Where the two halves of "is it live" have to cooperate. The batch owns
        // the boundary half — id 1 is at or above the boundary, and the batch did
        // create it, so `deletable` is satisfied both times. The graph owns the
        // other half, and the first delete put id 1 in the recycle bin, so the
        // second is refused there.
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &ids(&[0, 1])).expect("create");
        delete(&mut g, &space, &ids(&[1])).expect("the first delete is legitimate");

        let err = delete(&mut g, &space, &ids(&[1])).expect_err("the second is not");
        assert_eq!(err, NodeOpError::IdSpace(IdSpaceError::AlreadyRecycled(1)));
        assert_eq!(g.node_count(), 1, "and it was not deleted twice");
    }

    #[test]
    fn deleting_a_live_id_from_before_the_batch_is_allowed() {
        // The other side of the same split: id 0 is *below* the boundary, so the
        // batch has nothing to say about it — `deletable` only speaks for ids at
        // or above. Whether it is live is the recycle bin's answer, and it is not
        // in the bin, so the delete stands.
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &ids(&[0, 1])).expect("create");
        space.verify(g.node_id_bound()).expect("whole");

        let space = IdSpace::at(g.node_id_bound());
        delete(&mut g, &space, &ids(&[0])).expect("a live id from before the batch");
        assert_eq!(g.node_count(), 1);
        space
            .verify(g.node_id_bound())
            .expect("a delete leaves no hole");
    }

    #[test]
    fn deleting_an_id_never_created_is_refused() {
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &ids(&[0, 1])).expect("create");

        let err = delete(&mut g, &space, &ids(&[7])).expect_err("7 was never allocated");
        assert_eq!(err, NodeOpError::IdSpace(IdSpaceError::NeverCreated(7)));
    }

    #[test]
    fn the_first_id_zero_is_not_read_as_already_handed_out() {
        // `max_node_id()` returns 0 for an empty graph, so a boundary taken from
        // it reads as "id 0 has been handed out" and refuses this.
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        create(&mut g, &mut space, &ids(&[0])).expect("id 0 is fresh");
        space.verify(g.node_id_bound()).expect("whole");
    }

    #[test]
    fn the_last_id_is_refused_rather_than_wrapping_the_arithmetic() {
        let mut g = graph();
        let mut space = IdSpace::at(g.node_id_bound());
        let err = create(&mut g, &mut space, &ids(&[u64::MAX])).expect_err("not creatable");
        assert_eq!(
            err,
            NodeOpError::IdSpace(IdSpaceError::IdOutOfRange(u64::MAX))
        );
    }

    #[test]
    fn the_allocators_own_ids_are_not_validated() {
        // The write path's ids came from `reserve_nodes`, so there is nothing to
        // check and no batch to record into. It goes through the other entry
        // point entirely, which accepts a gap — that is what makes it a different
        // question rather than this one with the batch left out.
        let mut g = graph();
        g.create_allocated_nodes(&ids(&[0, 5]));
        g.delete_nodes(&ids(&[0]), &mut FxHashMap::default(), None)
            .expect("delete");
        assert_eq!(g.node_count(), 1);
    }
}
