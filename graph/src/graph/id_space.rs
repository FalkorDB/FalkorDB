//! The authority over a space of counted ids: which are live, which are free,
//! which may be handed out next, and whether what a batch did to them was
//! possible.
//!
//! Written for nodes and used for relationships too: both are counted ids with a
//! free set, and nothing below mentions which. The caller names the entity when
//! it renders a refusal.
//!
//! ## What it owns
//!
//! `live` and `recycled` — the entity count and the free set. Every id that
//! moves between those states moves through a method here: [`IdSpace::reserve`]
//! hands ids out, [`IdSpace::cancel`] takes an unused one back,
//! [`IdSpace::create`] makes them live, [`IdSpace::release`] frees them. `Graph`
//! keeps no counter and no bitmap of its own; `node_count()` and
//! `deleted_nodes()` read through to these.
//!
//! That is the point rather than a side effect. The boundary is
//! `live + recycled.len()`, so every one of those operations has to move both
//! halves or neither, and a free-set write that nothing counted was #2797 and
//! the node-id-liveness bug before it. Here there is no way to write one half.
//!
//! It is a field of the *versioned* `Graph` because a write query is in reader
//! mode for its whole mutation phase; what makes that safe is that the free set
//! it mutates is a private MVCC clone, discarded by one atomic store on
//! rollback. An id space living elsewhere would have to reproduce that.
//!
//! ## The invariant
//!
//! One equation, true at every moment, returned by every operation that touches
//! any of the four fields:
//!
//! ```text
//! live + recycled.len() == entry_bound + taken.len()
//! ```
//!
//! `taken` is every id at or above `entry_bound` the open batch has taken —
//! created, or reserved and then cancelled. Those are the same thing to the
//! boundary: both mean one more id has been handed out. Counting only creates
//! made the equation false for every cancelled reservation, which is what kept
//! the write path from being held to it.
//!
//! So an operation that moves `live` or `recycled` without recording fails the
//! query, naming all four fields, rather than waiting for a replica to refuse a
//! buffer. What it cannot see is this file getting its arithmetic right and its
//! meaning wrong, which is what the differential test is for.
//!
//! ## The batch, and the shape it is judged by
//!
//! `entry_bound` and `taken` are the *batch*: the span judged as one unit — a
//! whole effects buffer, or one commit segment of a write query.
//! [`IdSpace::open_batch`] begins one, and has a precondition worth reading.
//!
//! A batch exists because density holds between batches and not within one. The
//! effects apply path ingests records grouped by shape rather than ordered by
//! id, so a create of 500..600 may precede one of 0..500; between them the space
//! holds 100 live ids whose highest is 599 and reports its boundary as 100.
//! Judged against a boundary derived moment to moment the second create is
//! refused; judged against the boundary as it stood before any of it, both are
//! fine.
//!
//! So the batch owes one more thing at the end, which [`IdSpace::verify`] asks
//! and the invariant deliberately does not:
//!
//! ```text
//! taken == [ entry_bound, entry_bound + taken.len() )
//! ```
//!
//! Ids taken at or above the boundary must fill the range from it upward with no
//! hole. An allocator hands out the lowest free id, so it cannot reach an id
//! without having handed out everything below — a batch that leaves a hole did
//! not come from one. This is the check that catches a replica which has missed
//! a buffer: told to create 500..600 when its own boundary is 0, it would
//! otherwise accept an id space its master does not have and collide on its next
//! allocation. It is a *postcondition* rather than an invariant precisely
//! because it is legitimately false partway through a batch, which is the whole
//! reason a batch is a thing.
//!
//! Nothing here knows about replication, buffers or a peer engine. It is a
//! statement about one graph and the ids handed to it.

use roaring::RoaringTreemap;
use thiserror::Error;

/// Why a batch of ids does not describe a possible id space.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IdSpaceError {
    /// The boundary and the ledger disagree: an id changed state without both
    /// halves of the change happening.
    ///
    /// Not a claim about the caller — every other refusal here is. This one says
    /// a bug in this module reached production. Returned rather than asserted
    /// because a corrupt id space hands out ids that are already live, and
    /// failing the query discards the private MVCC version holding the damage.
    #[error(
        "the id space contradicts itself: {live} live + {recycled} free puts the boundary at \
         {bound}, but a batch opened at {entry_bound} having taken {taken} puts it at {expected}"
    )]
    Inconsistent {
        live: u64,
        recycled: u64,
        bound: u64,
        entry_bound: u64,
        taken: u64,
        expected: u64,
    },

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

    /// An id this batch has already taken — cancelled twice, or cancelled
    /// after being created. Either way the caller has lost track of it.
    #[error("{0} was already taken by this batch")]
    AlreadyTaken(u64),

    /// A batch named `u64::MAX`, which has no boundary above it.
    #[error("{0} is past the end of the id space")]
    IdOutOfRange(u64),
}

/// An id space, observed across one batch of creates and deletes.
///
/// Built before the batch, then asked for every id that changes state while it
/// runs, and checked once at the end. The per-id refusals are immediate —
/// [`Self::create`] names an id that is already live rather than letting the
/// batch finish and reporting arithmetic — and [`Self::verify`] is the backstop
/// behind them.
pub struct IdSpace {
    /// Ids handed out and still held. The entity count, and the half of the
    /// boundary that is a number rather than a set.
    live: u64,
    /// Ids handed out and given back: free to hand out again.
    ///
    /// A *reserved* id is in here too, deliberately, until the batch that
    /// reserved it commits — [`Self::max_id`] and [`Self::is_free`] are derived
    /// from this, and would go wrong mid-batch if a reservation removed one. So
    /// this alone does not mean "free"; see [`Self::reserve`].
    recycled: RoaringTreemap,
    /// The boundary as it stood when the open batch began: ids below it were
    /// handed out, ids at or above it never were.
    entry_bound: u64,
    /// Every id the open batch has *taken* — created, or reserved and then
    /// cancelled — at any value. Those are the same thing to the boundary:
    /// both move it by one, a create by making an id live and a cancellation by
    /// putting an allocated id in the free set. Counting only creates would make
    /// the invariant below false for every cancelled reservation, which is what
    /// used to keep [`Self::verify`] off the write path.
    ///
    /// Untrimmed, and that is load-bearing in two different directions.
    /// [`Self::reserve`] must not re-issue *any* id this batch has taken, and a
    /// reclaimed one sits below the boundary — trim it away and the batch hands
    /// the same id to two nodes, which reaches a replica as one id created
    /// twice in one buffer. The boundary arithmetic wants the opposite: an id
    /// below the entry boundary was counted into it before the batch began, so
    /// it must not be counted again. So the set keeps everything and
    /// [`Self::checked`] and [`Self::verify`] trim at the point of counting.
    ///
    /// Only grows within a batch — a later delete does not un-allocate an id, it
    /// frees one that was.
    taken: RoaringTreemap,
}

/// Append up to `count` ids from `pool` that `held` does not already hold, to
/// `out`. Returns how many were appended, which is fewer than `count` if the
/// pool runs out of free ids.
///
/// `pool - held` *is* the reclaimable set, so this takes its lowest ids and
/// stops. Stating it as set arithmetic rather than as a walk is not only
/// shorter: the walk had to be told where to start, and the rank it was given —
/// how many of the pool's ids the caller already held — is only the right place
/// to start while those ids are the pool's lowest. Cancelling a reservation
/// that came *from* the pool breaks that, leaving a free id below the rank that
/// the walk stepped over and did not come back for. A difference has nowhere to
/// step over.
///
/// Measured against the walk on the shape that matters — a 1M-id pool drained
/// in batches of 1024, which is what a large `CREATE` does — at 13.8ms against
/// 11.4ms, best of five alternating runs in one binary. Single runs on this
/// machine vary by a third, so the alternation is what makes the 2.4ms real
/// rather than noise. The difference is rebuilt per batch where the walk sought
/// into the pool in place, and that is the whole of the gap; against the create
/// it belongs to it is a quarter of one percent. On the ordinary path, where
/// nothing has been deleted and the pool is empty, the difference is the faster
/// of the two — 7.8us against 9.5us over 977 calls.
/// How many of `ids` sit at or above `bound`.
///
/// By rank rather than by trimming a copy: the answer is a count, and the set
/// it would be counted from can hold a million ids.
fn above(
    ids: &RoaringTreemap,
    bound: u64,
) -> u64 {
    let below = if bound == 0 { 0 } else { ids.rank(bound - 1) };
    ids.len() - below
}

fn reclaim_ids(
    pool: &RoaringTreemap,
    taken: &RoaringTreemap,
    issued: &RoaringTreemap,
    count: u64,
    out: &mut Vec<u64>,
) -> u64 {
    let free = pool - taken - issued;
    let before = out.len();
    out.extend(free.iter().take(count as usize));
    (out.len() - before) as u64
}

impl Default for IdSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl IdSpace {
    /// An empty id space: nothing handed out, nothing free.
    #[must_use]
    pub fn new() -> Self {
        Self {
            live: 0,
            recycled: RoaringTreemap::new(),
            entry_bound: 0,
            taken: RoaringTreemap::new(),
        }
    }

    /// The id space an RDB decoded, with no batch open.
    ///
    /// The one way to build a populated space from outside.
    #[must_use]
    pub fn restored(
        live: u64,
        recycled: RoaringTreemap,
    ) -> Self {
        let entry_bound = live + recycled.len();
        Self {
            live,
            recycled,
            entry_bound,
            taken: RoaringTreemap::new(),
        }
    }

    /// Ids handed out and still held.
    #[must_use]
    pub const fn live(&self) -> u64 {
        self.live
    }

    /// Ids handed out and given back.
    #[must_use]
    pub const fn recycled(&self) -> &RoaringTreemap {
        &self.recycled
    }

    /// How many of them there are.
    #[must_use]
    pub fn recycled_count(&self) -> u64 {
        self.recycled.len()
    }

    /// Whether `id` is free — handed out once and given back.
    #[must_use]
    pub fn is_free(
        &self,
        id: u64,
    ) -> bool {
        self.recycled.contains(id)
    }

    /// Where the id space ends: ids below were handed out, ids at or above
    /// never were.
    ///
    /// True as stated between batches, where the space is dense. Within one it
    /// is the boundary *plus* whatever the batch has created, which is why
    /// [`Self::reserve`] counts off `entry_bound` rather than off this.
    #[must_use]
    pub fn bound(&self) -> u64 {
        self.live + self.recycled.len()
    }

    /// The highest id handed out, or `0` when none has been.
    ///
    /// A sentinel rather than an answer in the empty case, and callers have to
    /// know it: `0` is indistinguishable from a space whose only id is 0. That
    /// is why a batch opens at [`Self::bound`] and not at this plus one.
    #[must_use]
    pub fn max_id(&self) -> u64 {
        if self.live == 0 {
            return 0;
        }
        self.bound() - 1
    }

    /// The same id space in a new MVCC version, with a fresh batch.
    ///
    /// A version is where a batch begins, so `taken` is not carried forward:
    /// that would put per-batch state in a version readers hold, and have a
    /// `verify` there measure the previous version's work.
    #[must_use]
    pub fn new_version(&self) -> Self {
        Self::restored(self.live, self.recycled.clone())
    }

    /// An id space wedged into a state no production path builds, for tests of
    /// the checks that exist to catch exactly that.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn wedged(
        live: u64,
        recycled: RoaringTreemap,
        entry_bound: u64,
        taken: RoaringTreemap,
    ) -> Self {
        Self {
            live,
            recycled,
            entry_bound,
            taken,
        }
    }

    /// The invariant this type maintains at every moment, in one expression:
    ///
    /// ```text
    /// live + recycled.len() == entry_bound + taken.len()
    /// ```
    ///
    /// The count half only. [`Self::verify`] also asks the shape, which is
    /// legitimately false partway through a batch — so the count is an invariant
    /// and the shape a postcondition, checked in different places.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::Inconsistent`], carrying both sides of the arithmetic.
    fn checked(&self) -> Result<(), IdSpaceError> {
        let taken = above(&self.taken, self.entry_bound);
        let bound = self.bound();
        match self.entry_bound.checked_add(taken) {
            Some(expected) if expected == bound => Ok(()),
            expected => Err(IdSpaceError::Inconsistent {
                live: self.live,
                recycled: self.recycled.len(),
                bound,
                entry_bound: self.entry_bound,
                taken,
                expected: expected.unwrap_or(u64::MAX),
            }),
        }
    }

    /// Close whatever batch was open and begin one here.
    ///
    /// Rebuilt rather than re-anchored: carrying `taken` across would measure
    /// two batches against one boundary. An id the last batch took is an
    /// ordinary recycled id to the next one.
    ///
    /// # Every reservation must be settled first
    ///
    /// Created or cancelled — not left outstanding. The boundary counts live and
    /// free ids and deliberately not reserved ones: reservations would grow the
    /// GraphBLAS matrix dimension `algo_procedures.rs` derives from it while
    /// leaving its liveness gate off, so `CALL algo.*` inside a write query would
    /// walk phantom nodes. An id still held when this runs lands *below* the
    /// boundary it opens, and the next [`Self::create`] of it is refused as
    /// [`IdSpaceError::AlreadyLive`].
    ///
    /// `Pending::end_segment` satisfies this by construction — every reserved id
    /// is recorded the moment it is handed out, and commit settles all of them.
    /// Written down because nothing enforces it.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::Inconsistent`] if the space this opens over already
    /// contradicts itself. Checked here and not only on the way out because
    /// this is the one operation that would *hide* it: re-anchoring the boundary
    /// on the corrupt value and clearing the ledger makes the arithmetic agree
    /// again, and the batch after it would look clean.
    pub fn open_batch(&mut self) -> Result<(), IdSpaceError> {
        self.checked()?;
        self.entry_bound = self.bound();
        self.taken.clear();
        Ok(())
    }

    /// Reserve `count` ids, freed ones first and then fresh.
    ///
    /// `issued` is every id this batch has been handed and not yet settled —
    /// reserved, still on its way to `create`. Borrowed rather than kept,
    /// because the caller already holds those ids (`Pending::created_nodes`) and
    /// a copy here could drift from them. Cancelled ids must **not** be passed:
    /// [`Self::cancel`] records them in `taken`, and passing them again counts
    /// one id twice and leaves a gap the batch never fills. A caller with
    /// nothing outstanding passes an empty set.
    ///
    /// A reserved id is left *in* `recycled`: [`Self::max_id`] and
    /// [`Self::is_free`] are derived from it and would go wrong mid-batch if a
    /// reservation removed one. So `recycled` alone does not mean "free", and
    /// taking the difference against what has been issued is what makes it mean
    /// that.
    ///
    /// A cancelled id must not be reissued inside the batch that cancelled it:
    /// the effects buffer emits it as its own create/delete pair, so a replica
    /// told to create it twice in one buffer refuses the whole payload. Between
    /// batches it is genuinely free again, which is why a batch is opened afresh
    /// per commit segment rather than re-anchored.
    ///
    /// # Errors
    ///
    /// A `count` that cannot be allocated. `GRAPH.BULK` sizes this from a
    /// client-declared count, so the size is attacker-influenced:
    /// `Vec::with_capacity` panicked on the capacity overflow and the panic
    /// hook exits the process, which took the server down (#2426).
    pub fn reserve(
        &self,
        count: usize,
        issued: &RoaringTreemap,
    ) -> Result<Vec<u64>, String> {
        let mut ids = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|_| format!("failed to reserve {count} ids"))?;
        let count = count as u64;

        // What this batch has already taken is excluded without being asked for:
        // the space recorded those itself, and a caller that creates as it goes
        // — `GRAPH.BULK`, or anything reserving one id at a time — would
        // otherwise have to hand its own creations back to the thing that
        // recorded them.
        let reclaimed = reclaim_ids(&self.recycled, &self.taken, issued, count, &mut ids);

        // Above every id ever handed out. `entry_bound` accounts for everything
        // issued before this batch; the ids issued *by* it that sit at or above
        // the boundary are the rest. Those below came out of the free set, which
        // the boundary already counted. The two sets are disjoint, which is what
        // lets this add their counts rather than merge them.
        let start = self.entry_bound
            + above(&self.taken, self.entry_bound)
            + above(issued, self.entry_bound);
        ids.extend(start..start + (count - reclaimed));

        Ok(ids)
    }

    /// Hand a reserved id back, unused.
    ///
    /// The id joins the free set so the space stays dense, and joins `taken`
    /// because the boundary moved: an allocated id is now free, which is one
    /// more id handed out. Recording it is also what stops [`Self::reserve`]
    /// offering it again inside this batch, since reserve excludes `taken`.
    ///
    /// A *reclaimed* reservation moves neither half: [`Self::reserve`] leaves it
    /// in the free set, and it sits below the boundary, which already counted
    /// it. Both decline for the same reason, so the invariant needs no special
    /// case.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::Inconsistent`], if returning the id left the boundary and
    /// the ledger disagreeing.
    pub fn cancel(
        &mut self,
        id: u64,
    ) -> Result<(), IdSpaceError> {
        if !self.taken.insert(id) {
            return Err(IdSpaceError::AlreadyTaken(id));
        }
        self.recycled.insert(id);
        self.checked()
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
    fn refuse_recycled(
        &self,
        nodes: &RoaringTreemap,
    ) -> Result<(), IdSpaceError> {
        match (nodes & &self.recycled).min() {
            Some(id) => Err(IdSpaceError::AlreadyRecycled(id)),
            None => Ok(()),
        }
    }

    /// Make `nodes` live, or refuse them and change nothing.
    ///
    /// The whole of the transition: the ids leave the free set, the live count
    /// grows by as many, and the batch records them. One call because any half
    /// of it alone is a boundary that moved for no reason.
    ///
    /// A free id is neither live nor a duplicate whatever its value, so
    /// subtracting the free set first is what lets the checks below tell a
    /// genuine double claim from a legitimate recreate. The order is
    /// load-bearing — freeing first would hide a double claim, recording first
    /// would leave the duplicate check finding what it just inserted — which is
    /// why this is one call and not three for a caller to sequence.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::AlreadyLive`] for the lowest id that is already live —
    /// either handed out before the batch, or claimed earlier within it.
    ///
    /// [`IdSpaceError::IdOutOfRange`] for `u64::MAX`. Nothing can be allocated
    /// above it, and letting it through would wrap the arithmetic in
    /// [`Self::verify`].
    pub fn create(
        &mut self,
        nodes: &RoaringTreemap,
    ) -> Result<(), IdSpaceError> {
        if nodes.contains(u64::MAX) {
            return Err(IdSpaceError::IdOutOfRange(u64::MAX));
        }

        // What is not free: the only ids either check can object to.
        let claimed = nodes - &self.recycled;

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
        if !self.taken.is_disjoint(&claimed) {
            let id = (&claimed & &self.taken)
                .min()
                .expect("the sets are not disjoint");
            return Err(IdSpaceError::AlreadyLive {
                id,
                entry_bound: self.entry_bound,
            });
        }

        // Checks passed, so the transition happens in full.
        self.recycled -= nodes;
        self.live += nodes.len();

        self.taken |= nodes;
        self.checked()
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
    fn refuse_undeletable(
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
        match (nodes - &self.taken)
            .max()
            .filter(|&id| id >= self.entry_bound)
        {
            Some(id) => Err(IdSpaceError::NeverCreated(id)),
            None => Ok(()),
        }
    }

    /// Free `freed`, having first refused the whole of `requested`.
    ///
    /// The mirror of [`Self::create`].
    ///
    /// Two sets because the relationship side genuinely has two:
    /// `delete_relationships` frees only the ids it could resolve to a type and
    /// both endpoints, skipping stale ones deliberately, but the refusals must
    /// judge everything it was *asked* for — an id that is already free is
    /// unresolvable for exactly that reason, and judging only the freed set
    /// would wave it through. The node side passes the same set twice.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::AlreadyRecycled`] for an id that is already free, and
    /// [`IdSpaceError::NeverCreated`] for one at or above the entry boundary
    /// that this batch never allocated.
    pub fn release(
        &mut self,
        requested: &RoaringTreemap,
        freed: &RoaringTreemap,
    ) -> Result<(), IdSpaceError> {
        // Both halves of "is this live": the bin's half needs only the bin, the
        // boundary's half needs the batch.
        debug_assert!(
            freed.is_subset(requested),
            "freeing an id that was not asked for"
        );
        self.refuse_recycled(requested)?;
        self.refuse_undeletable(requested)?;
        self.recycled |= freed;
        self.live -= freed.len();
        self.checked()
    }

    /// Check that the batch left a possible id space behind.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::Hole`] if the created ids do not fill the range from the
    /// entry boundary upward, and [`IdSpaceError::Miscounted`] if the graph's own
    /// boundary disagrees with where those ids put it. See the module docs for
    /// why both are needed.
    pub fn verify(&self) -> Result<(), IdSpaceError> {
        let graph_bound = self.bound();
        // What the batch has *handed out* at or above the boundary, which is
        // what the boundary has to account for. On the effects path that is the
        // created ids and nothing else. On the write path it also holds the
        // reservations the batch cancelled: those were allocated, they sit in
        // the recycle bin, and the graph's own boundary counts them — so
        // judging against `created` alone would read a legitimate cancellation
        // as a hole and refuse it.
        // Only the part at or above the boundary: `taken` is untrimmed, and an
        // id below the boundary was counted into it before this batch began.
        let created = above(&self.taken, self.entry_bound);
        let lowest_above = self.taken.select(self.taken.len() - created);
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
            lowest_above
                .zip(self.taken.max())
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
        nodes: &RoaringTreemap,
    ) -> Result<(), NodeOpError> {
        g.create_nodes(nodes)
    }

    fn delete(
        g: &mut Graph,
        nodes: &RoaringTreemap,
    ) -> Result<(), NodeOpError> {
        g.delete_nodes(nodes, &mut FxHashMap::default()).map(|_| ())
    }

    #[test]
    fn ids_arriving_out_of_order_still_fill_the_range() {
        // The case a boundary derived from the counter cannot express: the high
        // half arrives first, so mid-batch the graph reports its boundary as 100
        // while id 599 is allocated.
        let mut g = graph();
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &range(500..600)).expect("the high half is legitimate");
        create(&mut g, &range(0..500)).expect("and so is the low half");
        g.node_id_space()
            .verify()
            .expect("the batch closed the hole");
        assert_eq!(g.node_count(), 600);
    }

    #[test]
    fn a_hole_left_at_the_end_is_reported() {
        // 500..600 and nothing else. An allocator hands out the lowest free id,
        // so it cannot reach 500 without having handed out 0..499 — whoever
        // produced this was not working from the same id space.
        let mut g = graph();
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &range(500..600)).expect("nothing is wrong yet");

        let err = g
            .node_id_space()
            .verify()
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
        // but the count says four ids were handed out where the batch recorded
        // three, and only comparing the two notices.
        //
        // Built outright because no path reaches this state any more: `create`
        // and `release` are the only things that move the count and the free
        // set, and they move `taken` with them. The arm is kept for arithmetic
        // inside this file — a backstop with no live caller.
        let space = IdSpace::wedged(4, RoaringTreemap::new(), 0, range(0..3));

        let err = space
            .verify()
            .expect_err("four ids counted, three recorded");
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
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &ids(&[1, 2, 3])).expect("the graph takes them");

        let err = g
            .node_id_space()
            .verify()
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
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &range(0..6)).expect("six fresh ids");

        let err = create(&mut g, &ids(&[5])).expect_err("5 is already this batch's");
        assert_eq!(
            err,
            NodeOpError::node(IdSpaceError::AlreadyLive {
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
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &ids(&[0, 1])).expect("two fresh ids");
        delete(&mut g, &ids(&[0])).expect("deleting what this batch created");
        g.node_id_space().verify().expect("whole");

        // A fresh batch: id 0 sits in the recycle bin, so it is free to come
        // back and does not extend the range — which is only true if the bin is
        // part of the boundary this batch started from.
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &ids(&[0])).expect("id 0 is free");
        g.node_id_space().verify().expect("whole again");
    }

    #[test]
    fn creating_deleting_and_recreating_one_id_in_a_batch() {
        // Three commits of one query reach a replica as a single buffer, and the
        // allocator hands the freed id straight back, so `C(0) · D(0) · C(0)` is
        // legitimate. The set does not shrink on the delete, so the recreate adds
        // nothing and the range stays whole.
        let mut g = graph();
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &ids(&[0])).expect("create");
        delete(&mut g, &ids(&[0])).expect("delete");
        create(&mut g, &ids(&[0])).expect("the recreate is legitimate");
        g.node_id_space().verify().expect("whole");
        assert_eq!(g.node_count(), 1);
    }

    /// An id cannot be cancelled twice, and cannot be cancelled after being
    /// created. Either means the caller has lost track of it.
    #[test]
    fn an_id_the_batch_already_took_cannot_be_cancelled() {
        let mut space = IdSpace::new();
        space.reserve(2, &RoaringTreemap::new()).expect("reserved");

        space.cancel(0).expect("the first cancellation is ordinary");
        assert_eq!(
            space.cancel(0).expect_err("the second is not"),
            IdSpaceError::AlreadyTaken(0)
        );

        space.create(&ids(&[1])).expect("id 1 is this batch's");
        assert_eq!(
            space
                .cancel(1)
                .expect_err("created, so no longer cancellable"),
            IdSpaceError::AlreadyTaken(1)
        );
    }

    /// And the guard is against `taken`, not the free set.
    ///
    /// A *reclaimed* reservation is already free before it is cancelled —
    /// [`IdSpace::reserve`] leaves it there — so a guard that asked "is this id
    /// already free" would refuse the one caller behaving correctly. The
    /// question is whether this batch has taken it, not whether it is free.
    #[test]
    fn cancelling_a_reclaimed_id_is_still_allowed() {
        // One id handed out and given back before this batch: free on entry.
        let mut space = IdSpace::restored(0, ids(&[0]));
        space.open_batch().expect("a consistent space");

        let reclaimed = space.reserve(1, &RoaringTreemap::new()).expect("reserved");
        assert_eq!(reclaimed, vec![0], "and it comes back out of the free set");
        assert!(space.is_free(0), "reserve leaves it free until commit");

        space
            .cancel(0)
            .expect("already free, but this batch has not taken it before now");
    }

    #[test]
    fn a_cancelled_reclaimed_id_is_not_reissued_in_the_same_batch() {
        // One id handed out and given back before the batch opens: free, and
        // below the boundary this opens at.
        let mut space = IdSpace::restored(0, ids(&[0]));
        assert_eq!(space.bound(), 1);
        space.open_batch().expect("a consistent space");

        let reclaimed = space.reserve(1, &RoaringTreemap::new()).expect("reserved");
        assert_eq!(reclaimed, vec![0], "the free id comes back first");
        space.cancel(0).expect("a consistent space");

        let next = space.reserve(1, &RoaringTreemap::new()).expect("reserved");
        assert_eq!(next, vec![1], "not 0 again, even though it is free");
    }

    #[test]
    fn a_cancelled_reservation_arrives_as_a_pair() {
        // What a create-then-delete in one segment ships: both ids created, then
        // one deleted. The id space grew by two and one of them is free.
        let mut g = graph();
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &ids(&[0, 1])).expect("create");
        delete(&mut g, &ids(&[0])).expect("delete");
        g.node_id_space().verify().expect("whole");
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
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &ids(&[0, 1])).expect("create");
        delete(&mut g, &ids(&[1])).expect("the first delete is legitimate");

        let err = delete(&mut g, &ids(&[1])).expect_err("the second is not");
        assert_eq!(err, NodeOpError::node(IdSpaceError::AlreadyRecycled(1)));
        assert_eq!(g.node_count(), 1, "and it was not deleted twice");
    }

    #[test]
    fn deleting_a_live_id_from_before_the_batch_is_allowed() {
        // The other side of the same split: id 0 is *below* the boundary, so the
        // batch has nothing to say about it — `deletable` only speaks for ids at
        // or above. Whether it is live is the recycle bin's answer, and it is not
        // in the bin, so the delete stands.
        let mut g = graph();
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &ids(&[0, 1])).expect("create");
        g.node_id_space().verify().expect("whole");

        g.roll_id_batches().expect("a consistent space");
        delete(&mut g, &ids(&[0])).expect("a live id from before the batch");
        assert_eq!(g.node_count(), 1);
        g.node_id_space().verify().expect("a delete leaves no hole");
    }

    #[test]
    fn deleting_an_id_never_created_is_refused() {
        let mut g = graph();
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &ids(&[0, 1])).expect("create");

        let err = delete(&mut g, &ids(&[7])).expect_err("7 was never allocated");
        assert_eq!(err, NodeOpError::node(IdSpaceError::NeverCreated(7)));
    }

    #[test]
    fn the_first_id_zero_is_not_read_as_already_handed_out() {
        // `max_node_id()` returns 0 for an empty graph, so a boundary taken from
        // it reads as "id 0 has been handed out" and refuses this.
        let mut g = graph();
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &ids(&[0])).expect("id 0 is fresh");
        g.node_id_space().verify().expect("whole");
    }

    #[test]
    fn the_last_id_is_refused_rather_than_wrapping_the_arithmetic() {
        let mut g = graph();
        g.roll_id_batches().expect("a consistent space");
        let err = create(&mut g, &ids(&[u64::MAX])).expect_err("not creatable");
        assert_eq!(err, NodeOpError::node(IdSpaceError::IdOutOfRange(u64::MAX)));
    }

    /// [`IdSpace::open_batch`]'s precondition, seen from the failing side: the
    /// boundary does not count reservations, so closing a batch over one puts it
    /// below the next boundary, where creating it reads as recreating something
    /// live. Found by the differential test.
    #[test]
    fn a_reservation_that_outlives_its_batch_is_refused() {
        let mut space = IdSpace::new();
        let reserved = space.reserve(3, &RoaringTreemap::new()).expect("reserved");
        assert_eq!(reserved, vec![0, 1, 2]);

        // Settle two of the three. Out of order is legitimate mid-batch — that
        // is what the entry boundary exists for — so this is refused by nothing
        // yet, and the boundary moves to 2 while id 0 is still outstanding.
        space.create(&ids(&[1, 2])).expect("both are this batch's");
        assert_eq!(space.bound(), 2);
        space.open_batch().expect("a consistent space");

        let err = space
            .create(&ids(&[0]))
            .expect_err("id 0 is below the new boundary and nothing freed it");
        assert_eq!(
            err,
            IdSpaceError::AlreadyLive {
                id: 0,
                entry_bound: 2
            }
        );
    }

    /// `release` frees what the caller resolved, not what it was handed. The
    /// two sets differ on the relationship side, where `delete_relationships`
    /// skips ids it cannot resolve to a type and both endpoints.
    #[test]
    fn release_frees_the_resolved_and_leaves_the_rest() {
        // Four ids handed out and all live, so the batch opens at 4.
        let mut space = IdSpace::restored(4, RoaringTreemap::new());

        space
            .release(&ids(&[0, 1, 2, 3]), &ids(&[1, 2]))
            .expect("all four are live below the boundary");

        assert_eq!(
            space.recycled(),
            &ids(&[1, 2]),
            "only the resolved are freed"
        );
        assert_eq!(space.live(), 2, "and the count drops by as many");
        assert_eq!(space.bound(), 4, "the boundary does not move on a delete");
    }

    /// And it judges everything it was *asked* for. This is the half a
    /// single-set `release` would lose: an id that is already free is
    /// unresolvable for exactly that reason, so judging only the freed set
    /// would wave it through and leave a double delete unreported.
    #[test]
    fn release_refuses_a_free_id_it_would_not_have_freed() {
        // Three live and id 3 already free: four handed out, so the batch
        // opens at 4.
        let mut space = IdSpace::restored(3, ids(&[3]));

        let err = space
            .release(&ids(&[1, 3]), &ids(&[1]))
            .expect_err("3 is already free");

        assert_eq!(err, IdSpaceError::AlreadyRecycled(3));
        assert_eq!(space.recycled(), &ids(&[3]), "refused, so nothing moved");
        assert_eq!(space.live(), 3);
    }

    #[test]
    fn a_gap_is_creatable_and_only_verify_objects_to_it() {
        // Creating 0 and 5 and nothing between is a legitimate sequence of
        // operations — the graph takes both, and the delete that follows takes
        // the one it names. Only `verify` has an opinion, and only when asked.
        // That is the whole difference between the per-id refusals and the
        // end-of-batch check.
        let mut g = graph();
        g.roll_id_batches().expect("a consistent space");
        create(&mut g, &ids(&[0, 5])).expect("neither id is live");
        delete(&mut g, &ids(&[0])).expect("delete");
        assert_eq!(g.node_count(), 1);
        assert!(
            g.node_id_space().verify().is_err(),
            "the gap is real and only the end-of-batch check sees it"
        );
    }
}

#[cfg(test)]
mod reclaim_ids_tests {
    use super::*;

    /// What `reclaim_ids` is: the pool minus what the caller holds, lowest
    /// first. Written out independently so the tests below compare against a
    /// statement of the contract rather than against the implementation.
    fn free_ids(
        pool: &RoaringTreemap,
        held: &RoaringTreemap,
        count: u64,
    ) -> Vec<u64> {
        pool.iter()
            .filter(|id| !held.contains(*id))
            .take(count as usize)
            .collect()
    }

    fn reclaim(
        pool: &RoaringTreemap,
        held: &RoaringTreemap,
        count: u64,
    ) -> (Vec<u64>, u64) {
        let mut out: Vec<u64> = Vec::new();
        let taken = reclaim_ids(pool, &RoaringTreemap::new(), held, count, &mut out);
        assert_eq!(
            taken,
            out.len() as u64,
            "the count must match what it wrote"
        );
        (out, taken)
    }

    /// Nothing held: the pool's lowest `count` ids, in order.
    #[test]
    fn takes_the_lowest_ids_of_the_pool() {
        let pool: RoaringTreemap = [3u64, 9, 10, 40, 900].into_iter().collect();
        let none = RoaringTreemap::new();

        assert_eq!(reclaim(&pool, &none, 3).0, vec![3, 9, 10]);
        assert_eq!(reclaim(&pool, &none, 99).0, vec![3, 9, 10, 40, 900]);
        assert!(reclaim(&pool, &none, 0).0.is_empty());
    }

    /// Held ids are not handed out, wherever in the pool they sit.
    ///
    /// This is the whole contract. The previous form took a rank to start from
    /// and only skipped what it met after it, which was right while the held
    /// ids were the pool's lowest and wrong as soon as one of them was given
    /// back — a free id below the rank was stepped over. Taking the difference
    /// has no such position to be wrong about.
    #[test]
    fn never_hands_out_an_id_the_caller_holds() {
        let pool: RoaringTreemap = (0..6u64).collect();
        let held: RoaringTreemap = [1u64, 2].into_iter().collect();

        assert_eq!(reclaim(&pool, &held, 4).0, vec![0, 3, 4, 5]);

        // The freed-below case: id 0 is free again, and it comes out first
        // rather than being stepped past.
        let held: RoaringTreemap = [1u64, 2, 3].into_iter().collect();
        assert_eq!(reclaim(&pool, &held, 2).0, vec![0, 4]);
    }

    /// It reports a short count rather than making the number up.
    #[test]
    fn reports_how_many_it_could_take() {
        let pool: RoaringTreemap = (0..4u64).collect();
        let held: RoaringTreemap = [2u64].into_iter().collect();

        let (out, taken) = reclaim(&pool, &held, 9);
        assert_eq!(out, vec![0, 1, 3]);
        assert_eq!(taken, 3);
    }

    /// An empty pool, and a pool the caller holds entirely, both yield nothing.
    /// The first is the ordinary case — a graph with no deletions behind it.
    #[test]
    fn yields_nothing_when_there_is_nothing_free() {
        let empty = RoaringTreemap::new();
        let pool: RoaringTreemap = (0..4u64).collect();

        assert!(reclaim(&empty, &empty, 10).0.is_empty());
        assert!(reclaim(&pool, &pool, 10).0.is_empty());
    }

    /// It appends, so a caller can reclaim into a vector that already holds
    /// ids — which `reserve` does when a batch spans the free set and the
    /// fresh range.
    #[test]
    fn appends_rather_than_replaces() {
        let pool: RoaringTreemap = (10..20u64).collect();
        let mut out: Vec<u64> = vec![7, 8];
        reclaim_ids(
            &pool,
            &RoaringTreemap::new(),
            &RoaringTreemap::new(),
            3,
            &mut out,
        );
        assert_eq!(out, vec![7, 8, 10, 11, 12]);
    }

    /// Agrees with the contract across container boundaries — a treemap splits
    /// at 2^32 and each map at 2^16 — and at every batch size, with a held set
    /// scattered through the pool rather than sitting at its front.
    #[test]
    fn agrees_with_the_contract_on_a_scattered_pool() {
        let pool: RoaringTreemap = (0..300u64)
            .map(|i| i * 1_000)
            .chain(70_000..70_400)
            .chain(1_000_000..1_000_050)
            .chain(u64::from(u32::MAX) - 5..u64::from(u32::MAX) + 5)
            .collect();
        let held: RoaringTreemap = pool.iter().step_by(3).collect();

        for count in [1u64, 13, 97, 5_000] {
            assert_eq!(
                reclaim(&pool, &held, count).0,
                free_ids(&pool, &held, count),
                "count {count}"
            );
        }
    }
}

#[cfg(test)]
mod differential {
    use super::IdSpace;
    use roaring::RoaringTreemap;
    use std::collections::BTreeSet;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// The same id space as two plain sets and a boundary, maintained by
    /// definition rather than by arithmetic.
    ///
    /// `IdSpace` keeps a count where this keeps a set, and roaring containers
    /// where this keeps a `BTreeSet`, so the two agree only if the arithmetic
    /// and the container merges are both right.
    #[derive(Default)]
    struct Model {
        live: BTreeSet<u64>,
        free: BTreeSet<u64>,
        entry_bound: u64,
        taken: BTreeSet<u64>,
        /// Reserved and not yet created or cancelled — the caller's half, which
        /// `Pending` holds and lends back as `issued`.
        held: BTreeSet<u64>,
    }

    impl Model {
        fn bound(&self) -> u64 {
            self.live.len() as u64 + self.free.len() as u64
        }

        fn open_batch(&mut self) {
            self.entry_bound = self.bound();
            self.taken.clear();
        }

        /// What `verify` should say, from the model: `taken` must fill the range
        /// from the boundary upward, and the boundary must be where it puts it.
        fn verify_should_pass(&self) -> bool {
            // Only the part at or above the boundary, mirroring the untrimmed
            // set the space keeps.
            let above: Vec<u64> = self.taken.range(self.entry_bound..).copied().collect();
            let contiguous = above.is_empty()
                || (above[0] == self.entry_bound
                    && *above.last().unwrap() == self.entry_bound + above.len() as u64 - 1);
            contiguous && self.bound() == self.entry_bound + above.len() as u64
        }
    }

    fn as_set(m: &RoaringTreemap) -> BTreeSet<u64> {
        m.iter().collect()
    }

    fn agree(
        space: &IdSpace,
        model: &Model,
        step: usize,
        op: &str,
    ) {
        assert_eq!(
            space.live(),
            model.live.len() as u64,
            "step {step} {op}: live"
        );
        assert_eq!(
            as_set(space.recycled()),
            model.free,
            "step {step} {op}: free set"
        );
        assert_eq!(space.bound(), model.bound(), "step {step} {op}: bound");
        assert_eq!(
            space.max_id(),
            if model.live.is_empty() {
                0
            } else {
                model.bound() - 1
            },
            "step {step} {op}: max_id"
        );
        assert_eq!(as_set(&space.taken), model.taken, "step {step} {op}: taken");
        assert!(
            space.checked().is_ok(),
            "step {step} {op}: {:?}",
            space.checked()
        );
        assert_eq!(
            space.verify().is_ok(),
            model.verify_should_pass(),
            "step {step} {op}: verify disagrees with the model ({:?})",
            space.verify()
        );
    }

    /// Drive every operation against a reference model, for long enough that
    /// batches close over reclaimed ids and cancellations several times.
    ///
    /// Encapsulation stops anything outside this file desynchronising the count
    /// from the ledger; this is what covers the file getting its own arithmetic
    /// wrong, which only a second implementation can see.
    #[test]
    fn random_operation_sequences_agree_with_a_reference_model() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut space = IdSpace::new();
        let mut model = Model::default();
        let (mut reserves, mut creates, mut cancels, mut releases, mut batches) = (0, 0, 0, 0, 0);

        for step in 0..20_000 {
            let r = xorshift(&mut state);
            match r % 10 {
                // Reserve a few ids. The model checks the allocator's promise
                // directly: never an id that is live, held, or already taken.
                0..=3 => {
                    let n = ((r >> 8) % 4 + 1) as usize;
                    let held: RoaringTreemap = model.held.iter().copied().collect();
                    let ids = space.reserve(n, &held).expect("small counts fit");
                    assert_eq!(ids.len(), n, "step {step} reserve: short count");
                    for &id in &ids {
                        assert!(
                            !model.live.contains(&id),
                            "step {step}: reissued a live id {id}"
                        );
                        assert!(
                            !model.held.contains(&id),
                            "step {step}: reissued a held id {id}"
                        );
                        assert!(
                            !model.taken.contains(&id),
                            "step {step}: reissued a taken id {id}"
                        );
                        assert!(model.held.insert(id), "step {step}: reserve repeated {id}");
                    }
                    reserves += 1;
                    agree(&space, &model, step, "reserve");
                }
                // Create some of what is held.
                4..=6 => {
                    if model.held.is_empty() {
                        continue;
                    }
                    let take = ((r >> 16) % model.held.len() as u64 + 1) as usize;
                    let ids: Vec<u64> = model.held.iter().copied().take(take).collect();
                    let set: RoaringTreemap = ids.iter().copied().collect();
                    space.create(&set).expect("the allocator's own ids");
                    for id in ids {
                        model.held.remove(&id);
                        model.free.remove(&id);
                        model.live.insert(id);
                        model.taken.insert(id);
                    }
                    creates += 1;
                    agree(&space, &model, step, "create");
                }
                // Cancel one held id.
                7 => {
                    let Some(&id) = model.held.iter().nth(((r >> 24) % 7) as usize) else {
                        continue;
                    };
                    space.cancel(id).expect("a consistent space");
                    model.held.remove(&id);
                    model.free.insert(id);
                    model.taken.insert(id);
                    cancels += 1;
                    agree(&space, &model, step, "cancel");
                }
                // Release one live id.
                8 => {
                    let Some(&id) = model.live.iter().nth(((r >> 32) % 11) as usize) else {
                        continue;
                    };
                    let set: RoaringTreemap = std::iter::once(id).collect();
                    space
                        .release(&set, &set)
                        .expect("a live id below the boundary");
                    model.live.remove(&id);
                    model.free.insert(id);
                    releases += 1;
                    agree(&space, &model, step, "release");
                }
                // Close the batch. Every outstanding reservation is settled
                // first, which is `open_batch`'s stated precondition and what
                // `Pending::commit` does — it creates everything it holds before
                // `clear` and the next `open_id_batches`.
                _ => {
                    if !model.held.is_empty() {
                        let set: RoaringTreemap = model.held.iter().copied().collect();
                        space.create(&set).expect("settling what is held");
                        for id in std::mem::take(&mut model.held) {
                            model.free.remove(&id);
                            model.live.insert(id);
                            model.taken.insert(id);
                        }
                        agree(&space, &model, step, "settle");
                    }
                    space.open_batch().expect("a consistent space");
                    model.open_batch();
                    batches += 1;
                    agree(&space, &model, step, "open_batch");
                }
            }
        }

        // The mix is what makes the run mean anything, so it is asserted rather
        // than hoped for: a filter that silently stopped producing cancels would
        // otherwise read as a pass.
        assert!(reserves > 1_000, "too few reserves: {reserves}");
        assert!(creates > 1_000, "too few creates: {creates}");
        assert!(cancels > 200, "too few cancels: {cancels}");
        assert!(releases > 200, "too few releases: {releases}");
        assert!(batches > 500, "too few batch boundaries: {batches}");
    }
}
