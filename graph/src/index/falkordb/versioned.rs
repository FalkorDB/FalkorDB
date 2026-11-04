//! What every native index carries, whatever its kind: its build state, and the two sides that
//! change it, writers and indexer batches.
//!
//! A kind (today only the tuple column, [`IndexColumn`]) implements [`VersionedIndex`]: how it
//! records a writer's changes, and how it reads and adds an indexer batch. The lifecycle around
//! those operations is written once here, over [`IndexEntry<K>`], and never names a kind.

use std::fmt;
use std::sync::Arc;

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
/// answered with a [`ScanFilter`](super::falkordb_index::ScanFilter) and the read path scans
/// instead. `CREATE INDEX` takes one batch over every entity, so its index is `Ready` when it
/// returns.
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
///
/// While `Building`, the index also keeps which entities writers changed in it ([`Written`]), so a
/// batch read before them, or a writer's fork made before a batch, can take them again. A `Ready`
/// index keeps none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum BuildState<C> {
    /// Some entities are not indexed yet — reads scan instead. `next` is where the next indexer
    /// batch resumes.
    Building { next: C, written: Written },
    /// Every entity is indexed — the index serves reads.
    Ready,
}

impl<C: Copy> BuildState<C> {
    /// `Building` from `next`, with no entity written yet.
    pub(crate) fn building(next: C) -> Self {
        Self::Building {
            next,
            written: Written::default(),
        }
    }

    /// Where the next indexer batch resumes, or `None` once `Ready`.
    pub(crate) fn next(&self) -> Option<C> {
        match self {
            Self::Building { next, .. } => Some(*next),
            Self::Ready => None,
        }
    }
}

/// The entities writers changed in a `Building` index, every staged change counted, even one the
/// cursor kept out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Written {
    /// Since a batch last committed on the index. A batch read before some of them were written
    /// catches up on them at its commit ([`VersionedIndex::indexer_catch_up`]).
    pub(crate) since_last_indexer_step: Arc<RoaringTreemap>,
    /// By this version, until it is published. A writer whose fork batches have moved past
    /// re-applies them as it escalates ([`VersionedIndex::writer_rebase`]); published, they join
    /// `since_last_indexer_step`.
    pub(crate) by_this_version: RoaringTreemap,
}

/// The name of any versioned index: what the builder's chains, a commit's report of what it created
/// and every per-index dispatch key on.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum IndexId {
    /// A tuple column: `(label, attr)` of a node label or a relationship type.
    Column(EntityType, IndexKey),
}

/// One versioned index: the kind's own structure plus its build state.
///
/// The kind (`K`, today [`IndexColumn`]) is only the data: a column's tuples. The build state is
/// the entry's, as are the generation and the batch count, so the build's rules are written once
/// here for every kind. That is why a kind's operations are handed the state
/// ([`VersionedIndex::writer_record`] and the rest) rather than reading it from `self`.
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
    /// The batch's own work was discarded and nothing was published, for the reason given. The
    /// index's next batch starts from the published index's cursor.
    Discarded(Discarded),
}

/// How an indexer batch caught up with writes committed after it was read
/// ([`IndexerCommit::CatchUp`]). The commit takes the cheaper way, as [`REFRESH_COST`] decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatchUp {
    /// The entities the writers changed were refreshed in the batch's index.
    Refresh,
    /// The batch was inserted into the published index, without the entities the writers
    /// changed. Those of them in the batch's range were then added with their published values.
    Insert,
}

/// The cost of refreshing one entity, counted in inserted tuples; both happen under the host's
/// locks. When writers committed after a batch was read, the batch refreshes the entities they
/// changed if `changed * REFRESH_COST` is less than its tuple count, and inserts itself otherwise
/// ([`CatchUp`]).
///
/// The real ratio depends on how the batch's values are spread over the column. Measured on 2M
/// nodes with batches of 10,000 (`0742bcefb`, the online build's first branch):
///
/// - refreshing an entity cost 0.5–1.8 µs;
/// - inserting a tuple cost 0.31–0.35 µs when the values were scattered, and 0.010–0.015 µs when
///   they were consecutive.
///
/// That is a ratio of about 2–5 for scattered values and 40–100 for consecutive ones. At 10, the
/// wrong choice costs about 1.5 ms of extra lock time either way.
pub(crate) const REFRESH_COST: u64 = 10;

/// Whether a batch of `batch` tuples refreshes the `written` entities or inserts itself: refresh
/// if that is cheaper, counting each entity as [`REFRESH_COST`] tuples.
fn catch_up_by(
    written: u64,
    batch: usize,
) -> CatchUp {
    if written.saturating_mul(REFRESH_COST) < batch as u64 {
        CatchUp::Refresh
    } else {
        CatchUp::Insert
    }
}

/// Why an indexer batch's own work was discarded ([`IndexerCommit::Discarded`]): it published
/// nothing, and the index's next batch starts from the published cursor. Nothing a writer
/// committed is discarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Discarded {
    /// The index was dropped, or dropped and created again.
    Gone,
    /// Another batch committed on the same index since. The builder runs one chain of batches per
    /// index, so this is a bug in that exclusion, for the caller to report.
    AnotherBatch,
}

/// What a kind of index does on each side of its lifecycle. The state those operations move is
/// [`IndexEntry<K>`]'s, the same for every kind: `self` is the kind's data alone, so each operation
/// is handed the state.
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
    /// While `Building`, the id of every entity a change names goes into the state's
    /// [`Written::by_this_version`], whether the change is applied or not.
    fn writer_record(
        &mut self,
        state: &mut BuildState<Self::Cursor>,
        changes: Self::Changes,
        place: impl Fn(u64) -> Option<Self::Cursor>,
        scratch: &mut Self::Scratch,
    );

    /// Writer side, as it escalates: this is `published`'s copy of index `name`, which batches moved
    /// on after the writer's fork was made. Re-apply `written`, the entities the fork changed in
    /// it: their tuples as `published` has them come out, and as `fork` has them go in, while
    /// `state` is `Building` only for an entity `fork` places below the cursor.
    fn writer_rebase(
        &mut self,
        state: &BuildState<Self::Cursor>,
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
        state: &BuildState<Self::Cursor>,
        name: &Self::Name,
        written: &RoaringTreemap,
        batch_version: &Graph,
        published: &Graph,
    );

    /// Indexer side, under the lock: catch up by inserting ([`CatchUp::Insert`]). `self` is
    /// `published`'s copy of index `name`, with its cursor at `from`. `batch` was read from `from`
    /// up to `next` on an older version, and `written` holds the entities writers changed since.
    /// Insert `batch` without the `written` entities, then add the `written` entities that
    /// `published` places from `from` up to `next`, with their values in `published`.
    fn indexer_insert(
        &mut self,
        name: &Self::Name,
        batch: &Self::Batch,
        written: &RoaringTreemap,
        from: Self::Cursor,
        next: Option<Self::Cursor>,
        published: &Graph,
    );

    /// The number of tuples in `batch`: what inserting it costs.
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
        }
    }

    /// Whether some entities are not indexed yet.
    pub(crate) fn is_building(&self) -> bool {
        matches!(self.state, BuildState::Building { .. })
    }

    /// Where the next indexer batch resumes, or `None` once the index is `Ready`.
    pub(crate) fn cursor(&self) -> Option<K::Cursor> {
        self.state.next()
    }

    /// Apply one unit of work's `changes`, as this index's state allows
    /// ([`VersionedIndex::writer_record`]).
    pub(crate) fn writer_record(
        &mut self,
        changes: K::Changes,
        place: impl Fn(u64) -> Option<K::Cursor>,
        scratch: &mut K::Scratch,
    ) {
        self.index
            .writer_record(&mut self.state, changes, place, scratch);
    }

    /// Heap bytes this index holds in this version: its structure ([`VersionedIndex::memory_usage`])
    /// and, while `Building`, the entities written since a batch last committed on it
    /// ([`Written::since_last_indexer_step`]), by their serialized size.
    /// [`Written::by_this_version`] is not counted: a published version, which is what
    /// `GRAPH.MEMORY USAGE` reads, always has it empty ([`Self::publish`]).
    pub(crate) fn memory_usage(&self) -> usize {
        let written = match &self.state {
            BuildState::Building { written, .. } if !written.since_last_indexer_step.is_empty() => {
                written.since_last_indexer_step.serialized_size()
            }
            _ => 0,
        };
        self.index.memory_usage() + written
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
        self.indexer_commits += 1;
        match (next, &mut self.state) {
            (
                Some(next),
                BuildState::Building {
                    next: cursor,
                    written,
                },
            ) => {
                *cursor = next;
                // The batch read every entity at this version's values, writers' changes included.
                if !written.since_last_indexer_step.is_empty() {
                    written.since_last_indexer_step = Arc::default();
                }
            }
            _ => self.state = BuildState::Ready,
        }
        true
    }

    /// Ready this copy to be published: while `Building`, the entities this version changed join
    /// [`Written::since_last_indexer_step`].
    pub(crate) fn publish(&mut self) {
        if let BuildState::Building { written, .. } = &mut self.state {
            let mine = std::mem::take(&mut written.by_this_version);
            if !mine.is_empty() {
                *Arc::make_mut(&mut written.since_last_indexer_step) |= mine;
            }
        }
    }

    /// Why this copy of an index, which an indexer batch moved on, cannot go in over `published`,
    /// the published copy under its name: gone or created again (another generation), or moved
    /// on by another batch since. `None` if it follows it: the same index, one batch ahead.
    fn discarded_against(
        &self,
        published: Option<&Self>,
    ) -> Option<Discarded> {
        match published {
            Some(published) if published.generation == self.generation => {
                (published.indexer_commits + 1 != self.indexer_commits)
                    .then_some(Discarded::AnotherBatch)
            }
            _ => Some(Discarded::Gone),
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
    ///   [`IndexerCommit::Discarded`];
    /// - otherwise, a writer committed: [`IndexerCommit::CatchUp`];
    /// - otherwise, nothing committed since: [`IndexerCommit::Same`];
    /// - otherwise only other indexes' batches did: [`IndexerCommit::Swap`].
    ///
    /// Columns are the only kind of index, so this reads the batch's column entry directly. A
    /// second kind adds a variant to [`Prepared`], and the `let` below becomes a `match` on it,
    /// each arm asking these same questions of its kind's entry ([`VersionedIndex::entry`]). Once
    /// the arms differ only in the kind, move their body into one function generic over
    /// [`VersionedIndex`].
    pub(crate) fn indexer_fit(
        &self,
        published: &Self,
        batch: &PreparedBatch,
    ) -> IndexerCommit {
        let Prepared::Column(column) = &batch.0;
        let Some(ours) = IndexColumn::entry(&self.falkordb_index, &column.name) else {
            return IndexerCommit::Discarded(Discarded::Gone);
        };
        let theirs = IndexColumn::entry(&published.falkordb_index, &column.name);
        if let Some(why) = ours.discarded_against(theirs) {
            return IndexerCommit::Discarded(why);
        }
        if published.version != self.version {
            let written = theirs.map_or(0, |theirs| match &theirs.state {
                BuildState::Building { written, .. } => written.since_last_indexer_step.len(),
                BuildState::Ready => 0,
            });
            return IndexerCommit::CatchUp(catch_up_by(
                written,
                IndexColumn::batch_len(&column.work),
            ));
        }
        // No writer committed, so only batches can have changed an index since, and none did if
        // none was published.
        if published
            .falkordb_index
            .batch_published_since(&self.falkordb_index)
        {
            IndexerCommit::Swap
        } else {
            IndexerCommit::Same
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

    /// Used when writers committed after this version was made ([`IndexerCommit::CatchUp`]).
    /// Returns a new build version of `published` in which `batch`'s index is caught up with what
    /// those writers changed, the way `how` says:
    /// - [`CatchUp::Refresh`]: this version's copy, with the changed entities refreshed
    ///   ([`VersionedIndex::indexer_catch_up`]);
    /// - [`CatchUp::Insert`]: `published`'s copy, with the batch inserted
    ///   ([`VersionedIndex::indexer_insert`]).
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

    /// The [`CatchUp::Insert`] case of [`Self::caught_up_onto`], for one kind of index: take
    /// `published`'s copy of index `name` and insert `work`, the batch read on `batch_version`
    /// ([`VersionedIndex::indexer_insert`]). The index's build then goes on from where the batch
    /// left it.
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
        let BuildState::Building {
            next: from,
            written,
        } = &theirs.state
        else {
            return;
        };
        let mut merged = theirs.clone();
        merged.index.indexer_insert(
            name,
            work,
            &written.since_last_indexer_step,
            *from,
            ours.cursor(),
            published,
        );
        // The batch's state: its cursor, with nothing written since it.
        merged.state = ours.state.clone();
        merged.indexer_commits = ours.indexer_commits;
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
        let none = Arc::default();
        let written = match &theirs.state {
            BuildState::Building { written, .. } => &written.since_last_indexer_step,
            BuildState::Ready => &none,
        };
        let mut caught = ours.clone();
        caught
            .index
            .indexer_catch_up(&caught.state, name, written, batch_version, published);
        if let Some(slot) = K::entry_mut(&mut self.falkordb_index, name) {
            *slot = caught;
        }
    }

    /// For a writer's fork that no other writer's commit made stale, as it escalates: rebase it onto
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
    /// moved on, in `published`, an index still `Building` here. Every escalating write asks, so it
    /// looks no index up unless a batch was published since this version was made.
    pub(crate) fn overtaken_by_indexer_batches(
        &self,
        published: &Self,
    ) -> bool {
        published
            .falkordb_index
            .batch_published_since(&self.falkordb_index)
            && self
                .falkordb_index
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
        let BuildState::Building { written, .. } = &ours.state else {
            return;
        };
        let written = written.by_this_version.clone();
        let mut rebased = theirs.clone();
        rebased
            .index
            .writer_rebase(&rebased.state, name, &written, published, self);
        if let BuildState::Building {
            written: theirs, ..
        } = &mut rebased.state
        {
            theirs.by_this_version = written;
        }
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
}
