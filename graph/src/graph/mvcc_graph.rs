//! Multi-Version Concurrency Control (MVCC) for graph access.
//!
//! [`MvccGraph`] holds the published version of a graph. Readers share it; a
//! writer forks a private version, mutates it, and publishes it with
//! [`MvccGraph::commit`].
//!
//! - Readers never block: [`MvccGraph::read`] loads the `Arc` from an
//!   `ArcSwap`, which takes no lock.
//! - A writer builds into a [`Fork`] ([`MvccGraph::fork`]), Copy-on-Write and
//!   invisible to readers until it is committed.
//! - [`MvccGraph::commit`] publishes a fork only if it still extends the
//!   published version, so a fork that another commit overtook is refused
//!   rather than published over it.
//!
//! ## Who serializes publishers
//!
//! The host's global lock does. `commit` and `commit_indexer_batch` take
//! `&self`, so they need no per-graph lock, and every host publishes while
//! holding the global lock. A commit also publishes by compare-and-swap
//! against the version it checked, so a second publisher that slipped past
//! the lock is refused ([`CommitError::Stale`]) rather than published over.
//! Forking takes nothing at all, which is why a fork has to be checked for
//! staleness instead.
//!
//! ```text
//!  MvccGraph
//!  ┌─────────────────────────────────────────────────────┐
//!  │  graph: ArcSwap<Graph>             (published v1)   │
//!  └─────────────────────────────────────────────────────┘
//!
//!  Reader 1 ──read()──▶ loads Arc ──▶ sees Graph v1
//!  Reader 2 ──read()──▶ loads Arc ──▶ sees Graph v1
//!
//!  Writer ──fork()──▶ Graph::new_version() ──▶ Fork v2 (private)
//!    │                 (COW matrices, COW AttributeStore)
//!    │── mutations on v2 ──▶ ...
//!    │── commit(v2) ──▶ v2 still extends v1? valid? swap graph pointer
//!    │
//!    │  Readers now see v2; v1 is dropped when last Arc goes away
//!
//!  Abandoned writer ──▶ drops v2; there is nothing to release
//! ```
//!
//! ## Version Lifecycle
//!
//! ```text
//!  ┌──────┐   fork()     ┌──────────┐  commit()   ┌───────────┐
//!  │  v1  │ ────────────▶│ v2 (wip) │ ──────────▶ │ v2 (live) │
//!  │(live)│              │          │             │           │
//!  └──────┘              └──────────┘             └───────────┘
//!                              │
//!                   dropped, or refused by commit()
//!                              │
//!                              ▼
//!                         (discarded)
//! ```
//!
//! ## Thread Safety
//!
//! Readers never block -- they load the `Arc` to the current graph from the
//! `ArcSwap`, and a commit publishes by swapping it, through `&self`.
//!
//! A published version is an `Arc<Graph>`, so nothing can change it in place:
//! the types allow only shared access to it. Only a [`Fork`] is mutable, and
//! only until it is published. A fork holds its version in an `Arc` it alone
//! owns; [`MvccGraph::commit`] publishes a clone of that `Arc`, so whoever
//! holds the fork goes on reading the version it built, now published, and a
//! [`Fork::borrow_mut`] after that panics instead of changing it.

use std::{ops::Deref, sync::Arc};

use arc_swap::ArcSwap;
use atomic_refcell::{AtomicRef, AtomicRefCell, AtomicRefMut, BorrowError, BorrowMutError};
use thiserror::Error;

use crate::graph::graph::{Graph, NodeOpError};
#[cfg(feature = "index-falkordb")]
use crate::index::falkordb::versioned::{IndexerCommit, PreparedBatch};

/// A writer's private version of the graph, from [`MvccGraph::fork`].
///
/// Mutable through a shared reference until [`MvccGraph::commit`] publishes
/// it. The version stays readable after that — it is the published one — but
/// [`Self::borrow_mut`] panics: a published version is never changed.
pub struct Fork(AtomicRefCell<Arc<Graph>>);

impl Fork {
    #[must_use]
    pub fn new(graph: Graph) -> Self {
        Self(AtomicRefCell::new(Arc::new(graph)))
    }

    /// Shared access, while the fork is built and after it is published.
    #[must_use]
    pub fn borrow(&self) -> AtomicRef<'_, Graph> {
        AtomicRef::map(self.0.borrow(), |g| &**g)
    }

    /// Shared access, unless the fork is borrowed mutably.
    ///
    /// # Errors
    ///
    /// If the fork is borrowed mutably.
    pub fn try_borrow(&self) -> Result<AtomicRef<'_, Graph>, BorrowError> {
        self.0.try_borrow().map(|g| AtomicRef::map(g, |g| &**g))
    }

    /// Exclusive access to a fork that is not published yet.
    ///
    /// # Panics
    ///
    /// If the fork is borrowed, or was published.
    #[must_use]
    pub fn borrow_mut(&self) -> AtomicRefMut<'_, Graph> {
        AtomicRefMut::map(self.0.borrow_mut(), unpublished)
    }

    /// Exclusive access, unless the fork is borrowed.
    ///
    /// # Errors
    ///
    /// If the fork is borrowed.
    ///
    /// # Panics
    ///
    /// If the fork was published.
    pub fn try_borrow_mut(&self) -> Result<AtomicRefMut<'_, Graph>, BorrowMutError> {
        self.0
            .try_borrow_mut()
            .map(|g| AtomicRefMut::map(g, unpublished))
    }

    /// The version this fork holds: the one [`MvccGraph::commit`] publishes.
    #[must_use]
    pub fn version(&self) -> Arc<Graph> {
        Arc::clone(&self.0.borrow())
    }
}

fn unpublished(graph: &mut Arc<Graph>) -> &mut Graph {
    Arc::get_mut(graph).expect("a published version is never changed")
}

/// A version a query runs on: a published one, or a writer's [`Fork`].
#[derive(Clone)]
pub enum GraphVersion {
    Published(Arc<Graph>),
    Fork(Arc<Fork>),
}

impl GraphVersion {
    /// Shared access to the version.
    #[must_use]
    pub fn borrow(&self) -> GraphRef<'_> {
        match self {
            Self::Published(g) => GraphRef::Published(g),
            Self::Fork(f) => GraphRef::Fork(f.borrow()),
        }
    }
}

impl From<Arc<Graph>> for GraphVersion {
    fn from(g: Arc<Graph>) -> Self {
        Self::Published(g)
    }
}

impl From<Arc<Fork>> for GraphVersion {
    fn from(f: Arc<Fork>) -> Self {
        Self::Fork(f)
    }
}

/// Shared access to a [`GraphVersion`].
pub enum GraphRef<'r> {
    /// A published version needs no borrow: nothing can change it.
    Published(&'r Graph),
    /// A fork's borrow, which keeps it from being changed meanwhile.
    Fork(AtomicRef<'r, Graph>),
}

impl Deref for GraphRef<'_> {
    type Target = Graph;

    fn deref(&self) -> &Graph {
        match self {
            Self::Published(g) => g,
            Self::Fork(g) => g,
        }
    }
}

/// MVCC coordinator for concurrent graph access.
///
/// Provides snapshot isolation: readers see a consistent committed state
/// while a writer can make changes that become visible only on commit.
pub struct MvccGraph {
    /// Current committed graph version. Replaced only by a commit's compare-and-swap.
    graph: ArcSwap<Graph>,
}

/// Why [`MvccGraph::commit`] refused a version. Whichever it is, the version
/// was not published and the published graph is as it was.
#[derive(Debug, PartialEq, Eq, Error)]
pub enum CommitError {
    /// Another version was published after this one was forked, so it no
    /// longer extends the published graph.
    #[error("the graph was written after this version was forked")]
    Stale,
    /// The version failed [`Graph::validate`].
    #[error(transparent)]
    Invalid(#[from] NodeOpError),
}

unsafe impl Send for MvccGraph {}
unsafe impl Sync for MvccGraph {}

impl MvccGraph {
    #[must_use]
    pub fn new(
        n: u64,
        e: u64,
        cache_size: usize,
        name: &str,
    ) -> Self {
        Self::from_graph(Graph::new(n, e, cache_size, 0, name))
    }

    /// Create an `MvccGraph` from an already-constructed `Graph`.
    /// Used by the RDB load path.
    #[must_use]
    pub fn from_graph(graph: Graph) -> Self {
        Self {
            graph: ArcSwap::from_pointee(graph),
        }
    }

    /// The published version. Takes no lock.
    ///
    /// Each call loads the pointer again, so two calls can return different
    /// versions if something committed in between.
    #[must_use]
    pub fn read(&self) -> Arc<Graph> {
        self.graph.load_full()
    }

    /// A private version of the published graph to build a write into.
    ///
    /// Readers do not see it until [`Self::commit`] publishes it, and
    /// dropping it discards the write: forking takes nothing that has to be
    /// given back. So a writer can fork while it is still matching as a
    /// reader, and an inline writer that commits meanwhile is never refused
    /// for it (#3093) — the fork goes stale instead ([`Self::is_stale`]).
    #[must_use]
    pub fn fork(&self) -> Arc<Fork> {
        Arc::new(Fork::new(self.graph.load().new_version()))
    }

    /// Whether a version was published after `fork` was made, so it no
    /// longer extends the published graph and [`Self::commit`] would refuse
    /// it.
    ///
    /// For a writer that forked before it held the write lock: it asks once
    /// it holds it, before it touches state that is not versioned — the index
    /// documents a `Commit` publishes, which dropping the fork cannot take
    /// back. Nothing else commits while the lock is held, so the answer stands
    /// until this writer's own commit.
    #[must_use]
    pub fn is_stale(
        &self,
        fork: &Fork,
    ) -> bool {
        self.graph.load().version + 1 != fork.borrow().version
    }

    /// Bring a writer's `fork`, which is not stale ([`Self::is_stale`]), onto the indexer batches
    /// published since it was made. A batch changes no data, so what the writer read still holds:
    /// each index the batches moved on becomes the published copy, with their progress, and what
    /// the fork wrote in it is re-applied (`Graph::rebase_onto_indexer_batches`).
    ///
    /// For a writer that forked before it held the write lock, as it escalates, right after
    /// [`Self::is_stale`]: what it publishes and reads from then on has the batches' progress, and
    /// nothing else publishes before its own commit.
    ///
    /// # Errors
    ///
    /// [`CommitError::Stale`] if a batch overtook `fork` and it is borrowed, which no escalating
    /// writer's is: a debug build stops on it, and in a release build the write runs again.
    #[cfg(feature = "index-falkordb")]
    pub fn rebase_onto_indexer_batches(
        &self,
        fork: &Fork,
    ) -> Result<(), CommitError> {
        let published = self.graph.load_full();
        // Borrowed mutably only when a batch overtook it: until a build runs in the background, no
        // published column is `Building`, and nothing is rebased.
        if fork
            .try_borrow()
            .is_ok_and(|fork| !fork.overtaken_by_indexer_batches(&published))
        {
            return Ok(());
        }
        // An escalating writer's fork is never borrowed: its writer has stopped reading it to
        // escalate. One that is is a bug, which a debug build stops on. A release build refuses
        // it, and the host runs the write again.
        let Ok(mut fork) = fork.try_borrow_mut() else {
            if cfg!(debug_assertions) {
                panic!("a fork is rebased while it is borrowed");
            }
            return Err(CommitError::Stale);
        };
        fork.rebase_onto_indexer_batches(&published);
        Ok(())
    }

    /// Publish `fork` as the current version.
    ///
    /// Checks before touching anything, so a refusal leaves the published
    /// version untouched and `fork` unpublished. This is the only place a
    /// version becomes visible, so every writer gets these checks; a writer
    /// that publishes unversioned state before it commits also asks
    /// [`Self::is_stale`] first.
    ///
    /// A refusal needs no cleanup: nothing was taken, and the refused version
    /// is just dropped.
    ///
    /// Takes `&self`: publishing is a compare-and-swap of the published
    /// pointer. The caller holds the host's global lock, which keeps commits
    /// apart and the swap out of the way of the host forking (#452); it needs no
    /// per-graph lock.
    ///
    /// Once published, `fork` still reads the version, now the published one,
    /// and can no longer be changed ([`Fork::borrow_mut`]).
    ///
    /// # Errors
    ///
    /// [`CommitError::Stale`] if `fork` no longer extends the published
    /// graph, if it was published already, or if another version was
    /// published while this commit ran (only a host publishing without the
    /// global lock lets that happen),
    /// [`CommitError::Invalid`] if [`Graph::validate`] refuses it.
    pub fn commit(
        &self,
        fork: &Fork,
    ) -> Result<(), CommitError> {
        let published = self.graph.load_full();
        let version = {
            let mut slot = fork.0.borrow_mut();
            let graph = Arc::get_mut(&mut slot).ok_or(CommitError::Stale)?;
            Self::prepare(&published, graph, true)?;
            Arc::clone(&slot)
        };
        self.publish(&published, version)
    }

    /// Commit an indexer batch. `prepared` is a build version of a published one
    /// ([`Graph::new_build_version`]) with `batch` already applied
    /// ([`Graph::indexer_prepare`]), made with no lock held. How it goes in depends on what
    /// committed since ([`Graph::indexer_fit`]):
    /// - nothing: `prepared` is published as it is;
    /// - only other indexes' batches: `batch`'s index is put into a new build version of the
    ///   published one, which is published ([`Graph::with_batch_index`]);
    /// - writers, and no other batch on `batch`'s index: the index is caught up with what the
    ///   writers changed, whichever way is cheaper. Either the changed entities are refreshed in
    ///   `prepared`'s copy, or the batch, without them, is inserted into the published copy. The
    ///   result goes into a new build version of the published one, which is published
    ///   ([`Graph::caught_up_onto`]);
    /// - `batch`'s index was dropped or created again, or another batch committed on the same
    ///   index: nothing is published, and the index's next batch starts from its published cursor.
    ///   Another batch on the index is a bug in the builder's one chain per index
    ///   ([`AnotherBatch`](crate::index::falkordb::versioned::Discarded::AnotherBatch)), which the
    ///   caller reports.
    ///
    /// The commit changes no graph data and keeps the data version, so it does not make a writer's
    /// fork stale ([`Self::is_stale`]).
    ///
    /// The caller holds the host's global lock, as a writer does to publish; it needs no
    /// per-graph lock. The fit is made against the published version this commit loads, and the
    /// version is published over that one: a batch that writers overtook is caught up, not
    /// refused.
    ///
    /// # Errors
    ///
    /// [`CommitError::Invalid`] if [`Graph::validate`] refuses the version, [`CommitError::Stale`]
    /// if `prepared` was published already, or if another version was published while this commit
    /// ran (only a host publishing without the global lock lets that happen). Nothing is
    /// published.
    #[cfg(feature = "index-falkordb")]
    pub fn commit_indexer_batch(
        &self,
        prepared: &Fork,
        batch: &PreparedBatch,
    ) -> Result<IndexerCommit, CommitError> {
        let published = self.graph.load_full();
        let fit = prepared.borrow().indexer_fit(&published, batch);
        let version = match fit {
            IndexerCommit::Same => {
                let mut slot = prepared.0.borrow_mut();
                let graph = Arc::get_mut(&mut slot).ok_or(CommitError::Stale)?;
                Self::prepare(&published, graph, false)?;
                Arc::clone(&slot)
            }
            IndexerCommit::Swap => {
                let mut graph = prepared.borrow().with_batch_index(&published, batch);
                Self::prepare(&published, &mut graph, false)?;
                Arc::new(graph)
            }
            IndexerCommit::CatchUp(how) => {
                let mut graph = prepared.borrow().caught_up_onto(&published, batch, how);
                Self::prepare(&published, &mut graph, false)?;
                Arc::new(graph)
            }
            IndexerCommit::Discarded(_) => return Ok(fit),
        };
        self.publish(&published, version)?;
        Ok(fit)
    }

    /// The checks and the last changes before `new_graph` is published over `published`, the
    /// published version as the caller loaded it: `writer` for a writer's version, `false` for an
    /// indexer batch's, which keeps the data version.
    fn prepare(
        published: &Graph,
        new_graph: &mut Graph,
        writer: bool,
    ) -> Result<(), CommitError> {
        // Before any of the work below, so a refusal changes nothing. A batch's version is fitted
        // onto the published one, at its data version ([`Graph::indexer_fit`]): only a writer's
        // can be stale.
        if writer && published.version + 1 != new_graph.version {
            return Err(CommitError::Stale);
        }
        debug_assert!(writer || published.version == new_graph.version);
        new_graph.validate()?;
        #[cfg(feature = "index-falkordb")]
        {
            new_graph
                .falkordb_index
                .set_batches_published(&published.falkordb_index, !writer);
            new_graph.falkordb_index.publish();
        }

        // If the schema changed (new labels, relationship types, or attributes), make sure
        // schema_version moved.
        let schema = |g: &Graph| {
            (
                g.get_labels().len(),
                g.get_types().len(),
                g.get_node_attribute_names().len(),
                g.get_relationship_attribute_names().len(),
            )
        };
        if schema(published) != schema(new_graph)
            && new_graph.schema_version == published.schema_version
        {
            new_graph.schema_version += 1;
        }

        new_graph.trim_attr_stores();

        // Fold away any delta that has grown comparable to its base, then
        // materialize every committed base before publishing. The transaction
        // is done mutating and the write lock is still held, so this is the
        // last point at which a delete-everything's tombstones can be applied
        // — otherwise both the base and a base-sized `dm` stay resident until
        // some later transaction happens to touch the same matrix.
        //
        // The bases must be materialized because readers reach them lock-free
        // (`m()`, raw `m.nvals()`), and any GrB call on a pending matrix
        // finishes that work internally (a mutation) — so two readers racing
        // on a shared pending base corrupt GrB state (GrB_INVALID_OBJECT, heap
        // corruption under stress). dp/dm stay lazy: every read of them goes
        // through the mutex-guarded Matrix::wait first. Per-matrix cost when
        // already synced is one atomic load; waiting the deltas here too was
        // measured at +35% instructions on `create 10k`.
        new_graph.fold_oversized_deltas();
        Ok(())
    }

    /// Publish `version` over `published`, and only over it. Every publisher holds the host's
    /// global lock, so nothing replaces it meanwhile; if a host broke that, the later commit is
    /// refused instead of overwriting the other's version.
    fn publish(
        &self,
        published: &Arc<Graph>,
        version: Arc<Graph>,
    ) -> Result<(), CommitError> {
        let previous = self.graph.compare_and_swap(published, Arc::clone(&version));
        if !Arc::ptr_eq(&previous, published) {
            return Err(CommitError::Stale);
        }
        // After the swap, so a refused version is never handed to the indexers.
        version.set_indexer_graph(Arc::clone(&version));
        Ok(())
    }
}

impl Drop for MvccGraph {
    fn drop(&mut self) {
        self.graph.load().cancel_indexing();
    }
}

#[cfg(test)]
mod tests {
    use super::{CommitError, MvccGraph};
    use crate::graph::graphblas::test_init::ensure_init;
    use roaring::RoaringTreemap;
    use std::sync::Arc;

    /// A refused version publishes nothing, and the graph takes the next one.
    #[test]
    fn a_refused_commit_publishes_nothing() {
        ensure_init();
        let mvcc = MvccGraph::new(64, 64, 0, "t");
        let published = mvcc.read();

        let invalid = mvcc.fork();
        // Ids 0 and 5 with nothing between: a batch that cannot have come from
        // an allocator, which is what `Graph::validate` refuses.
        let holed: RoaringTreemap = [0u64, 5].into_iter().collect();
        invalid
            .borrow_mut()
            .create_nodes(&holed)
            .expect("neither id is live, so the graph takes them");
        assert!(
            matches!(mvcc.commit(&invalid), Err(CommitError::Invalid(_))),
            "a hole in the id space must refuse the version"
        );
        assert!(
            Arc::ptr_eq(&mvcc.read(), &published),
            "a refused version must not be published"
        );

        let next = mvcc.fork();
        mvcc.commit(&next)
            .expect("a refusal leaves nothing behind to block the next write");
    }

    /// A fork takes nothing, so a writer that forks while it is still
    /// matching never blocks one that commits meanwhile (#3093) — its own
    /// fork goes stale instead, and is refused rather than published over the
    /// other writer's version.
    #[test]
    fn a_fork_blocks_no_writer_and_goes_stale_when_one_commits() {
        ensure_init();
        let mvcc = MvccGraph::new(64, 64, 0, "t");

        let matching = mvcc.fork();
        assert!(!mvcc.is_stale(&matching));

        let inline = mvcc.fork();
        mvcc.commit(&inline)
            .expect("an open fork must not refuse another writer");

        assert!(
            mvcc.is_stale(&matching),
            "the fork predates the inline commit"
        );
        assert_eq!(mvcc.commit(&matching), Err(CommitError::Stale));
        assert!(
            Arc::ptr_eq(&mvcc.read(), &inline.version()),
            "the refused fork must not replace the version it was overtaken by"
        );

        let fresh = mvcc.fork();
        assert!(!mvcc.is_stale(&fresh));
        mvcc.commit(&fresh)
            .expect("a fork of the current version commits");
    }

    /// A reader keeps the published version it loaded across a commit, as a
    /// read runtime does for its whole query. The commit publishes by swapping
    /// the pointer, so the reader keeps the version it started on and the next
    /// `read()` returns the new one.
    #[test]
    fn a_commit_leaves_a_held_published_version_alone() {
        ensure_init();
        let mvcc = MvccGraph::new(64, 64, 0, "t");
        let reader = mvcc.read();
        let version = reader.version;

        let next = mvcc.fork();
        next.borrow_mut().get_label_id_mut("L");
        mvcc.commit(&next).expect("a new label commits");

        assert_eq!(reader.version, version);
        assert!(
            reader.get_labels().is_empty(),
            "the reader's version is unchanged"
        );
        assert_eq!(mvcc.read().version, version + 1);
        assert_eq!(mvcc.read().get_labels().len(), 1);
    }

    /// A committed fork reads the version it published, and nothing can change
    /// it through the fork any more: committing it again is refused, and a
    /// mutable borrow panics.
    #[test]
    fn a_committed_fork_is_the_published_version_and_stays_unchanged() {
        ensure_init();
        let mvcc = MvccGraph::new(64, 64, 0, "t");
        let fork = mvcc.fork();
        fork.borrow_mut().get_label_id_mut("L");
        mvcc.commit(&fork).expect("a new label commits");

        assert!(Arc::ptr_eq(&mvcc.read(), &fork.version()));
        assert_eq!(fork.borrow().get_labels().len(), 1);
        assert_eq!(mvcc.commit(&fork), Err(CommitError::Stale));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                drop(fork.borrow_mut());
            }))
            .is_err(),
            "a published version must not be borrowed mutably"
        );
    }

    /// A commit publishes only over the version it was checked against. Every
    /// publisher holds the host's global lock, so nothing replaces that version
    /// meanwhile; if something did, the commit is refused instead of overwriting
    /// the other publisher's version.
    #[test]
    fn a_commit_checked_against_a_replaced_version_is_refused() {
        ensure_init();
        let mvcc = MvccGraph::new(64, 64, 0, "t");
        let checked = mvcc.read();
        let first = mvcc.fork();
        let second = mvcc.fork();
        mvcc.commit(&first).expect("the first version commits");

        assert_eq!(
            mvcc.publish(&checked, second.version()),
            Err(CommitError::Stale),
            "checked against the replaced version, so not published"
        );
        assert!(
            Arc::ptr_eq(&mvcc.read(), &first.version()),
            "the first version stays published"
        );
    }
}
