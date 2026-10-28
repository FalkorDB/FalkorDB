//! What every native index carries, whatever its kind: its build state, and the two sides that
//! change it, writers and indexer batches.
//!
//! A kind (today only the tuple column, [`IndexColumn`]) implements [`VersionedIndex`]: how it
//! records a writer's changes, and how it reads and adds an indexer batch. The lifecycle around
//! those operations is written once here, over [`IndexEntry<K>`], and never names a kind.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use roaring::RoaringTreemap;

use crate::entity_type::EntityType;
use crate::graph::graph::Graph;

use super::falkordb_index::{FalkorDbIndex, IndexColumn, IndexKey};

/// Build state of a versioned index, generic over its kind's cursor.
///
/// An index is filled by indexer batches. Each reads the next entities from the cursor at the
/// values they hold *then* ([`VersionedIndex::indexer_read`]), adds them, and moves the cursor past
/// them ([`IndexEntry::indexer_apply`]). The batch that reaches the end flips the index
/// [`Ready`](Self::Ready). While [`Building`](Self::Building), a read the index would serve is
/// answered [`NotReady`](super::falkordb_index::NumericAnswer::NotReady) and the read path scans
/// instead. `CREATE INDEX` leaves its index `Building` for the background builder
/// ([`online_build`](super::online_build)); an RDB load takes one batch over every entity, so the
/// index is `Ready` once the graph is loaded.
///
/// A `Building` index holds exactly the current tuples of the entities below its cursor, and
/// nothing at or above it. Live writes keep that true ([`VersionedIndex::writer_record`]). An
/// addition for an entity below the cursor is applied, and one for an entity at or above it is
/// not, since the batch that reaches the entity reads its value then. Every removal is applied,
/// because an id's place can move past the cursor within one unit of work: an edge behind it is
/// deleted and its id reused by an edge ahead of it, so the place no longer tells whether the
/// index holds the deleted edge's tuples. Removing a tuple the index does not hold changes
/// nothing. Additions are dropped before a unit's changes collapse to the last per tuple, so a
/// tuple the unit removed and added back, as the reused id's new edge can, is still removed. This
/// is how C populates an index (`src/index/index_construct.c`), in batches interleaved with
/// writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BuildState<C> {
    /// Some entities are not indexed yet — reads scan instead. `next` is where the next indexer
    /// batch resumes.
    Building { next: C },
    /// Every entity is indexed — the index serves reads.
    Ready,
}

/// The name of any versioned index: what the builder's chains, a commit's report of what it created
/// and every per-index dispatch key on.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum IndexId {
    /// A tuple column: `(label, attr)` of a node label or a relationship type.
    Column(EntityType, IndexKey),
}

/// One versioned index: the kind's own structure plus its build state.
#[derive(Clone)]
pub(crate) struct IndexEntry<K: VersionedIndex> {
    pub(crate) index: K,
    pub(crate) state: BuildState<K::Cursor>,
    /// Which index this is: the version's count of indexes created when this one was. An index
    /// dropped and created again under the same name has another.
    pub(crate) generation: u64,
    /// Indexer batches applied to this index. Only a batch moves it, so a batch's copy one ahead
    /// of the published copy means no other batch committed on the index since the batch's
    /// version was made.
    pub(crate) indexer_commits: u64,
    /// While `Building`: the entities writers changed in the index since a batch last committed
    /// on it. A batch read before some of them were written catches up on them at its commit
    /// ([`VersionedIndex::indexer_catch_up`]). Empty once `Ready`.
    pub(crate) written_since_last_indexer_step: Arc<RoaringTreemap>,
    /// While `Building`, until this version is published: the entities it changed in the index,
    /// every staged change counted, even one the cursor kept out. A writer whose fork batches have
    /// moved past re-applies them at its claim ([`VersionedIndex::writer_rebase`]); published,
    /// they join `written_since_last_indexer_step`.
    pub(crate) written_by_this_version: RoaringTreemap,
}

/// One indexer batch of index `name`: the work it read, and where the index's next batch resumes
/// (`None`: the batch reached the end).
pub(crate) struct IndexerBatch<K: VersionedIndex> {
    pub(crate) name: K::Name,
    pub(crate) work: K::Batch,
    pub(crate) next: Option<K::Cursor>,
}

/// An indexer batch of any kind, as the indexer's turn keeps it from the version it was applied
/// to ([`Graph::indexer_prepare`]) to its commit
/// ([`MvccGraph::commit_indexer_batch`](crate::graph::mvcc_graph::MvccGraph::commit_indexer_batch)).
pub struct PreparedBatch(Prepared);

/// [`PreparedBatch`] by kind.
enum Prepared {
    Column(IndexerBatch<IndexColumn>),
}

impl PreparedBatch {
    /// The index the batch was applied to.
    pub(crate) fn id(&self) -> IndexId {
        match &self.0 {
            Prepared::Column(batch) => IndexId::Column(batch.name.0, batch.name.1.clone()),
        }
    }
}

/// How an indexer batch went in: what its commit published, given what committed since its
/// version was made ([`Graph::indexer_fit`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexerCommit {
    /// Nothing committed since: the batch's version was published as it is.
    Same,
    /// Only other indexes' batches committed since: the batch's index was put into a new build
    /// version of the published one, which was published.
    Swap,
    /// Writers committed since, and no other batch on the batch's index: the index as the batch
    /// left it, caught up on what the writers changed in it, was put into a new build version of
    /// the published one, which was published.
    CatchUp(CatchUp),
    /// Nothing was published, for the reason given. The index's next batch starts from the
    /// published index's cursor.
    Dropped(Dropped),
}

/// How an indexer batch caught up on writers' changes ([`IndexerCommit::CatchUp`]): whichever
/// costs less, by [`REFRESH_COST`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatchUp {
    /// The entities the writers changed were refreshed in the batch's index.
    Refresh,
    /// The batch, less the entities the writers changed, was inserted into the published index,
    /// and those of them in the batch's range were added as the published version has them.
    Insert,
}

/// How many tuples inserted into an index count as one entity refreshed in it, under the host's
/// locks: a batch that writers committed after refreshes the entities they wrote if there are fewer
/// than its batch's tuples over this, and inserts the batch otherwise ([`CatchUp`]).
///
/// The true ratio depends on how the batch's values spread over the column. Measured on 2M nodes
/// with batches of 10,000 (on the online build's first branch, `0742bcefb`), a refreshed entity
/// cost 0.5–1.8 µs, and an inserted tuple 0.31–0.35 µs when the values were scattered but
/// 0.010–0.015 µs when they were consecutive: a ratio of about 2–5 one way and 40–100 the other.
/// 10 keeps the cost of the wrong choice to about 1.5 ms of held locks either way.
pub(crate) const REFRESH_COST: u64 = 10;

/// How a batch of `batch` tuples catches up on `written` entities: refresh them if that costs
/// less than inserting the batch, an entity costing [`REFRESH_COST`] tuples.
fn catch_up_by(
    written: u64,
    batch: usize,
) -> CatchUp {
    #[cfg(test)]
    if let Some(how) = CATCH_UP_BY.with(std::cell::Cell::get) {
        return how;
    }
    if written.saturating_mul(REFRESH_COST) < batch as u64 {
        CatchUp::Refresh
    } else {
        CatchUp::Insert
    }
}

/// Why an indexer batch published nothing ([`IndexerCommit::Dropped`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dropped {
    /// The index was dropped, or dropped and created again.
    Gone,
    /// Another batch committed on the same index since. The builder runs one chain of batches per
    /// index, so this is a bug in that exclusion, and it is reported ([`report_batch_race`]).
    AnotherBatch,
}

/// What a kind of index does on each side of its lifecycle. The state those operations move is
/// [`IndexEntry<K>`]'s, the same for every kind.
pub(crate) trait VersionedIndex: Clone + Sized {
    /// Where a `Building` index's next batch resumes, and an entity's place in that order.
    type Cursor: Copy + Eq + fmt::Debug;
    /// One unit of work's staged changes to one index.
    type Changes;
    /// What one indexer batch read, ready to add.
    type Batch;
    /// How an index of this kind is found in a version's [`FalkorDbIndex`].
    type Name;
    /// Buffers [`writer_record`](Self::writer_record) reuses from one index to the next.
    type Scratch: Default;

    /// Index `name` in `index`, if it exists.
    fn entry<'a>(
        index: &'a FalkorDbIndex,
        name: &Self::Name,
    ) -> Option<&'a IndexEntry<Self>>;

    /// Index `name` in `index`, if it exists, to change.
    fn entry_mut<'a>(
        index: &'a mut FalkorDbIndex,
        name: &Self::Name,
    ) -> Option<&'a mut IndexEntry<Self>>;

    /// Writer side: apply one unit of work's `changes`. While `state` is `Building`, an addition
    /// is applied only for an entity `place` puts below the cursor, and every removal is applied,
    /// wherever `place` puts its entity or if it has none: removing a tuple the index does not hold
    /// changes nothing. Additions are dropped before the changes collapse to the last per tuple,
    /// so a dropped addition cannot cancel an earlier removal of the same tuple ([`BuildState`]
    /// says why).
    ///
    /// While `Building`, the id of every entity a change names goes into `written`, whether the
    /// change is applied or not.
    fn writer_record(
        &mut self,
        state: BuildState<Self::Cursor>,
        changes: Self::Changes,
        place: impl Fn(u64) -> Option<Self::Cursor>,
        scratch: &mut Self::Scratch,
        written: &mut RoaringTreemap,
    );

    /// Writer side, at its claim: this is `published`'s copy of index `name`, which batches moved
    /// on after the writer's fork was made. Re-apply `written`, the entities the fork changed in
    /// it: their tuples as `published` has them come out, and as `fork` has them go in, while
    /// `state` is `Building` only for an entity `fork` places below the cursor.
    fn writer_rebase(
        &mut self,
        state: BuildState<Self::Cursor>,
        name: &Self::Name,
        written: &RoaringTreemap,
        published: &Graph,
        fork: &Graph,
    );

    /// Indexer side, under the lock: this is the batch's copy of index `name`, read on
    /// `batch_version`, and writers committed since. Catch it up on `written`, the entities they
    /// changed in it: their tuples as `batch_version` has them come out, and as `published` has
    /// them go in, while `state` is `Building` only for an entity `published` places below the
    /// cursor.
    fn indexer_catch_up(
        &mut self,
        state: BuildState<Self::Cursor>,
        name: &Self::Name,
        written: &RoaringTreemap,
        batch_version: &Graph,
        published: &Graph,
    );

    /// Indexer side, under the lock, the other way to catch up: this is `published`'s copy of
    /// index `name`, whose cursor is `from`, and `batch` was read from `from` to `next` on an
    /// older version. Insert `batch` less the entities in `written`, the ones writers changed
    /// since, and add those of them `published` places from `from` up to `next` as it has them.
    fn indexer_insert(
        &mut self,
        name: &Self::Name,
        batch: &Self::Batch,
        written: &RoaringTreemap,
        from: Self::Cursor,
        next: Option<Self::Cursor>,
        published: &Graph,
    );

    /// How many tuples `batch` holds: what inserting it costs.
    fn batch_len(batch: &Self::Batch) -> usize;

    /// Indexer side, off the lock: read index `name`'s batch from `from`, at most `budget`
    /// entities, at the values `version` holds. Also returns where the next batch resumes, or
    /// `None` when this one reaches the end.
    fn indexer_read(
        &self,
        version: &Graph,
        name: &Self::Name,
        from: Self::Cursor,
        budget: usize,
    ) -> (Self::Batch, Option<Self::Cursor>);

    /// Add `batch`, read from this same version.
    fn indexer_add(
        &mut self,
        batch: &mut Self::Batch,
    );

    /// Heap bytes this index's own structure holds in this version: every page it reaches,
    /// counted once, whether or not another version shares it. So the sizes of two versions that
    /// share pages do not add up; each says what that version would cost alone.
    fn memory_usage(&self) -> usize;
}

impl<K: VersionedIndex> IndexEntry<K> {
    pub(crate) fn new(
        index: K,
        state: BuildState<K::Cursor>,
        generation: u64,
    ) -> Self {
        Self {
            index,
            state,
            generation,
            indexer_commits: 0,
            written_since_last_indexer_step: Arc::default(),
            written_by_this_version: RoaringTreemap::new(),
        }
    }

    /// Whether some entities are not indexed yet.
    pub(crate) fn is_building(&self) -> bool {
        matches!(self.state, BuildState::Building { .. })
    }

    /// Where the next indexer batch resumes, or `None` once the index is `Ready`.
    pub(crate) fn cursor(&self) -> Option<K::Cursor> {
        match self.state {
            BuildState::Building { next } => Some(next),
            BuildState::Ready => None,
        }
    }

    /// Apply one unit of work's `changes`, as this index's state allows
    /// ([`VersionedIndex::writer_record`]).
    pub(crate) fn writer_record(
        &mut self,
        changes: K::Changes,
        place: impl Fn(u64) -> Option<K::Cursor>,
        scratch: &mut K::Scratch,
    ) {
        self.index.writer_record(
            self.state,
            changes,
            place,
            scratch,
            &mut self.written_by_this_version,
        );
    }

    /// Heap bytes this index holds in this version: its structure ([`VersionedIndex::memory_usage`])
    /// and, while `Building`, the entities written since a batch last committed on it, by their
    /// serialized size. `written_by_this_version` is not counted: a published version, which is
    /// what `GRAPH.MEMORY USAGE` reads, always has it empty ([`Self::publish`]).
    pub(crate) fn memory_usage(&self) -> usize {
        let written = &self.written_since_last_indexer_step;
        self.index.memory_usage()
            + if written.is_empty() {
                0
            } else {
                written.serialized_size()
            }
    }

    /// Add an indexer batch's `work`, read from this same version, then move the cursor to `next`,
    /// or flip the index `Ready` when it is `None`. `false`, changing nothing, if the index is not
    /// `Building`.
    pub(crate) fn indexer_apply(
        &mut self,
        work: &mut K::Batch,
        next: Option<K::Cursor>,
    ) -> bool {
        if !self.is_building() {
            return false;
        }
        self.index.indexer_add(work);
        self.state = next.map_or(BuildState::Ready, |next| BuildState::Building { next });
        self.indexer_commits += 1;
        // The batch read every entity at this version's values, writers' changes included.
        if !self.written_since_last_indexer_step.is_empty() {
            self.written_since_last_indexer_step = Arc::default();
        }
        true
    }

    /// Ready this copy to be published: while `Building`, the entities this version changed join
    /// `written_since_last_indexer_step`; once `Ready`, neither set is kept.
    pub(crate) fn publish(&mut self) {
        let written = std::mem::take(&mut self.written_by_this_version);
        if self.is_building() {
            if !written.is_empty() {
                *Arc::make_mut(&mut self.written_since_last_indexer_step) |= written;
            }
        } else if !self.written_since_last_indexer_step.is_empty() {
            self.written_since_last_indexer_step = Arc::default();
        }
    }

    /// Why this copy of an index, which an indexer batch moved on, cannot go in over `published`,
    /// the published copy under its name: gone or created again (another generation), or moved
    /// on by another batch since. `None` if it follows it: the same index, one batch ahead.
    fn dropped_against(
        &self,
        published: Option<&Self>,
    ) -> Option<Dropped> {
        match published {
            Some(published) if published.generation == self.generation => {
                (published.indexer_commits + 1 != self.indexer_commits)
                    .then_some(Dropped::AnotherBatch)
            }
            _ => Some(Dropped::Gone),
        }
    }
}

impl FalkorDbIndex {
    /// Apply one indexer batch to this version ([`IndexEntry::indexer_apply`]). `false`, changing
    /// nothing, if the index is gone or no longer `Building`.
    pub(crate) fn apply_indexer_batch<K: VersionedIndex>(
        &mut self,
        batch: &mut IndexerBatch<K>,
    ) -> bool {
        K::entry_mut(self, &batch.name)
            .is_some_and(|entry| entry.indexer_apply(&mut batch.work, batch.next))
    }
}

impl Graph {
    /// One indexer batch of index `id` on this version: read its next entities at the values this
    /// version holds, at most `budget` of them, add them to the index and move its cursor. Returns
    /// the batch, for its commit, or `None`, changing nothing, if the index is not `Building`.
    pub(crate) fn indexer_prepare(
        &mut self,
        id: &IndexId,
        budget: usize,
    ) -> Option<PreparedBatch> {
        match id {
            IndexId::Column(entity, key) => self
                .prepare_batch::<IndexColumn>((*entity, key.clone()), budget)
                .map(|batch| PreparedBatch(Prepared::Column(batch))),
        }
    }

    /// How `batch`, applied to this version, a build version of an older published one
    /// ([`Self::new_build_version`]), fits `published`, the current one:
    /// - its index was dropped or created again since, or another batch committed on it:
    ///   [`IndexerCommit::Dropped`];
    /// - otherwise, a writer committed: [`IndexerCommit::CatchUp`];
    /// - otherwise, nothing committed since: [`IndexerCommit::Same`];
    /// - otherwise only other indexes' batches did: [`IndexerCommit::Swap`].
    pub(crate) fn indexer_fit(
        &self,
        published: &Self,
        batch: &PreparedBatch,
    ) -> IndexerCommit {
        let id = batch.id();
        match &batch.0 {
            Prepared::Column(batch) => self.fit_of::<IndexColumn>(
                published,
                &batch.name,
                &id,
                IndexColumn::batch_len(&batch.work),
            ),
        }
    }

    /// [`Self::indexer_fit`] for one kind.
    fn fit_of<K: VersionedIndex>(
        &self,
        published: &Self,
        name: &K::Name,
        id: &IndexId,
        batch_len: usize,
    ) -> IndexerCommit {
        let Some(ours) = K::entry(&self.falkordb_index, name) else {
            return IndexerCommit::Dropped(Dropped::Gone);
        };
        let theirs = K::entry(&published.falkordb_index, name);
        if let Some(why) = ours.dropped_against(theirs) {
            return IndexerCommit::Dropped(why);
        }
        if published.version != self.version {
            let written = theirs.map_or(0, |theirs| theirs.written_since_last_indexer_step.len());
            return IndexerCommit::CatchUp(catch_up_by(written, batch_len));
        }
        // No writer committed, so only batches can have changed an index since.
        if self
            .falkordb_index
            .same_batches_except(&published.falkordb_index, id)
        {
            IndexerCommit::Same
        } else {
            IndexerCommit::Swap
        }
    }

    /// A new build version of `published` with `batch`'s index as this version has it, in place
    /// of its own: for a batch that only other indexes' batches committed after
    /// ([`IndexerCommit::Swap`]).
    pub(crate) fn with_batch_index(
        &self,
        published: &Self,
        batch: &PreparedBatch,
    ) -> Self {
        let mut version = published.new_build_version();
        match &batch.0 {
            Prepared::Column(batch) => version.take_index::<IndexColumn>(self, &batch.name),
        }
        version
    }

    /// A new build version of `published` with `batch`'s index as this version has it, caught up
    /// on the entities writers changed in it since this version was made
    /// ([`VersionedIndex::indexer_catch_up`]): for a batch that writers committed after
    /// ([`IndexerCommit::CatchUp`]).
    pub(crate) fn caught_up_onto(
        &self,
        published: &Self,
        batch: &PreparedBatch,
        how: CatchUp,
    ) -> Self {
        let mut version = published.new_build_version();
        match (&batch.0, how) {
            (Prepared::Column(batch), CatchUp::Refresh) => {
                version.catch_up_index::<IndexColumn>(self, published, &batch.name);
            }
            (Prepared::Column(batch), CatchUp::Insert) => {
                version.insert_index::<IndexColumn>(self, published, &batch.name, &batch.work);
            }
        }
        version
    }

    /// [`Self::caught_up_onto`] for one kind, by [`CatchUp::Insert`]: `published`'s copy of index
    /// `name` with `work`, the batch read on `batch_version`, inserted less the entities writers
    /// changed since, and those in the batch's range added as `published` has them
    /// ([`VersionedIndex::indexer_insert`]). The index's build moves on as the batch's did.
    fn insert_index<K: VersionedIndex>(
        &mut self,
        batch_version: &Self,
        published: &Self,
        name: &K::Name,
        work: &K::Batch,
    ) {
        let (Some(ours), Some(theirs)) = (
            K::entry(&batch_version.falkordb_index, name),
            K::entry(&published.falkordb_index, name),
        ) else {
            return;
        };
        let Some(from) = theirs.cursor() else {
            return;
        };
        let mut merged = theirs.clone();
        merged.index.indexer_insert(
            name,
            work,
            &theirs.written_since_last_indexer_step,
            from,
            ours.cursor(),
            published,
        );
        merged.state = ours.state;
        merged.indexer_commits = ours.indexer_commits;
        merged.written_since_last_indexer_step = Arc::default();
        if let Some(slot) = K::entry_mut(&mut self.falkordb_index, name) {
            *slot = merged;
        }
    }

    /// [`Self::caught_up_onto`] for one kind: index `name` as `batch_version` has it, caught up on
    /// `published`'s writes, in place of this version's copy.
    fn catch_up_index<K: VersionedIndex>(
        &mut self,
        batch_version: &Self,
        published: &Self,
        name: &K::Name,
    ) {
        let (Some(ours), Some(theirs)) = (
            K::entry(&batch_version.falkordb_index, name),
            K::entry(&published.falkordb_index, name),
        ) else {
            return;
        };
        let mut caught = ours.clone();
        caught.index.indexer_catch_up(
            caught.state,
            name,
            &theirs.written_since_last_indexer_step,
            batch_version,
            published,
        );
        if let Some(slot) = K::entry_mut(&mut self.falkordb_index, name) {
            *slot = caught;
        }
    }

    /// For a writer's fork that no other writer's commit made stale, at its claim: rebase it onto
    /// the indexer batches committed since it was made. Each index still `Building` here that a
    /// batch moved on in `published` becomes `published`'s copy, with the entities the fork changed
    /// in it re-applied ([`VersionedIndex::writer_rebase`]). An index no batch moved on keeps the
    /// fork's copy, and so does one the fork created, or dropped and created again. Batches change
    /// no data, so what the writer read still holds.
    pub(crate) fn rebase_onto_indexer_batches(
        &mut self,
        published: &Self,
    ) {
        for name in self.falkordb_index.column_names() {
            self.rebase_index::<IndexColumn>(published, &name);
        }
    }

    /// Whether [`Self::rebase_onto_indexer_batches`] has anything to do: whether an indexer batch
    /// moved on, in `published`, an index still `Building` here.
    pub(crate) fn overtaken_by_indexer_batches(
        &self,
        published: &Self,
    ) -> bool {
        self.falkordb_index
            .column_names()
            .iter()
            .any(|name| self.overtaken::<IndexColumn>(published, name))
    }

    /// Whether an indexer batch moved on, in `published`, index `name`, still `Building` here: the
    /// same index (the same generation), with more batches committed on it.
    fn overtaken<K: VersionedIndex>(
        &self,
        published: &Self,
        name: &K::Name,
    ) -> bool {
        match (
            K::entry(&self.falkordb_index, name),
            K::entry(&published.falkordb_index, name),
        ) {
            (Some(ours), Some(theirs)) => {
                ours.is_building()
                    && theirs.generation == ours.generation
                    && theirs.indexer_commits != ours.indexer_commits
            }
            _ => false,
        }
    }

    /// [`Self::rebase_onto_indexer_batches`] for one index.
    fn rebase_index<K: VersionedIndex>(
        &mut self,
        published: &Self,
        name: &K::Name,
    ) {
        if !self.overtaken::<K>(published, name) {
            return;
        }
        let (Some(ours), Some(theirs)) = (
            K::entry(&self.falkordb_index, name),
            K::entry(&published.falkordb_index, name),
        ) else {
            return;
        };
        let written = ours.written_by_this_version.clone();
        let mut rebased = theirs.clone();
        rebased
            .index
            .writer_rebase(rebased.state, name, &written, published, self);
        rebased.written_by_this_version = written;
        if let Some(slot) = K::entry_mut(&mut self.falkordb_index, name) {
            *slot = rebased;
        }
    }

    /// Put index `name` as `from` has it in place of this version's copy.
    fn take_index<K: VersionedIndex>(
        &mut self,
        from: &Self,
        name: &K::Name,
    ) {
        if let (Some(theirs), Some(ours)) = (
            K::entry(&from.falkordb_index, name),
            K::entry_mut(&mut self.falkordb_index, name),
        ) {
            *ours = theirs.clone();
        }
    }

    /// [`Self::indexer_prepare`] for one kind: read the batch from the index's cursor, then apply
    /// it to the same version. Returns the batch, or `None` if the index is not `Building`.
    fn prepare_batch<K: VersionedIndex>(
        &mut self,
        name: K::Name,
        budget: usize,
    ) -> Option<IndexerBatch<K>> {
        let entry = K::entry(&self.falkordb_index, &name)?;
        let from = entry.cursor()?;
        let (work, next) = entry.index.indexer_read(self, &name, from, budget);
        let mut batch = IndexerBatch { name, work, next };
        self.falkordb_index
            .apply_indexer_batch(&mut batch)
            .then_some(batch)
    }

    /// Whether index `id` is `Building` in this version.
    pub(crate) fn index_building(
        &self,
        id: &IndexId,
    ) -> bool {
        match id {
            IndexId::Column(entity, key) => {
                IndexColumn::entry(&self.falkordb_index, &(*entity, key.clone()))
                    .is_some_and(IndexEntry::is_building)
            }
        }
    }

    /// Every index this version holds `Building`: what registering a graph with the builder starts.
    pub(crate) fn building_indexes(&self) -> Vec<IndexId> {
        self.falkordb_index
            .column_names()
            .into_iter()
            .filter(|name| {
                IndexColumn::entry(&self.falkordb_index, name).is_some_and(IndexEntry::is_building)
            })
            .map(|(entity, key)| IndexId::Column(entity, key))
            .collect()
    }

    /// The indexes this version holds `Building` that `older` does not hold as the same index:
    /// created since, or dropped and created again. What a commit hands the builder, so each build
    /// starts once, however many commits it spans.
    pub(crate) fn indexes_to_build_since(
        &self,
        older: &Self,
    ) -> Vec<IndexId> {
        self.falkordb_index
            .column_names()
            .into_iter()
            .filter(|name| self.building_since::<IndexColumn>(older, name))
            .map(|(entity, key)| IndexId::Column(entity, key))
            .collect()
    }

    /// [`Self::indexes_to_build_since`] for one index: `Building` here, and in `older` absent or
    /// another generation.
    fn building_since<K: VersionedIndex>(
        &self,
        older: &Self,
        name: &K::Name,
    ) -> bool {
        K::entry(&self.falkordb_index, name).is_some_and(|ours| {
            ours.is_building()
                && K::entry(&older.falkordb_index, name)
                    .is_none_or(|theirs| theirs.generation != ours.generation)
        })
    }
}

/// Report an indexer batch that found another batch committed on its index
/// ([`Dropped::AnotherBatch`]): a bug in the builder's one chain of batches per index. A debug
/// build stops on it. A release build logs it once per process and goes on: the batch publishes
/// nothing, and the index's next batch starts from its published cursor.
pub(crate) fn report_batch_race(id: &IndexId) {
    #[cfg(test)]
    if ALLOW_BATCH_RACE.with(std::cell::Cell::get) {
        return;
    }
    if cfg!(debug_assertions) {
        panic!(
            "two indexer batches raced on {id:?}: the builder runs one chain of batches per index"
        );
    }
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if !REPORTED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "native index: two indexer batches raced on {id:?}, which the builder should rule out; \
             the later one published nothing. Further races are not logged."
        );
    }
}

#[cfg(test)]
thread_local! {
    /// For tests, which take each way on purpose: while set, every batch that writers committed
    /// after catches up this way, whatever the sizes ([`catch_up_by`]).
    pub(crate) static CATCH_UP_BY: std::cell::Cell<Option<CatchUp>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// For tests that race two batches on one index on purpose: while set, [`report_batch_race`]
    /// does not stop the test.
    pub(crate) static ALLOW_BATCH_RACE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
