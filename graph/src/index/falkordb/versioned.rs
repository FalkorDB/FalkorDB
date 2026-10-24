//! What every native index carries, whatever its kind: its build state, and the two sides that
//! change it, writers and indexer batches.
//!
//! A kind (today only the tuple column, [`IndexColumn`]) implements [`VersionedIndex`]: how it
//! records a writer's changes, and how it reads and adds an indexer batch. The lifecycle around
//! those operations is written once here, over [`IndexEntry<K>`], and never names a kind.

use std::fmt;

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
}

/// One indexer batch of index `name`: the work it read, and where the index's next batch resumes
/// (`None`: the batch reached the end).
pub(crate) struct IndexerBatch<K: VersionedIndex> {
    pub(crate) name: K::Name,
    pub(crate) work: K::Batch,
    pub(crate) next: Option<K::Cursor>,
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
    fn writer_record(
        &mut self,
        state: BuildState<Self::Cursor>,
        changes: Self::Changes,
        place: impl Fn(u64) -> Option<Self::Cursor>,
        scratch: &mut Self::Scratch,
    );

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
}

impl<K: VersionedIndex> IndexEntry<K> {
    pub(crate) fn new(
        index: K,
        state: BuildState<K::Cursor>,
    ) -> Self {
        Self { index, state }
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
        self.index
            .writer_record(self.state, changes, place, scratch);
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
        true
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
    /// version holds, at most `budget` of them, add them to the index and move its cursor. `false`,
    /// changing nothing, if the index is not `Building`.
    pub(crate) fn indexer_prepare(
        &mut self,
        id: &IndexId,
        budget: usize,
    ) -> bool {
        match id {
            IndexId::Column(entity, key) => self
                .prepare_batch::<IndexColumn>((*entity, key.clone()), budget)
                .is_some(),
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
