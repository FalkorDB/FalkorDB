//! Multi-Version Concurrency Control (MVCC) for graph access.
//!
//! [`MvccGraph`] holds the published version of a graph. Readers share it; a
//! writer forks a private version, mutates it, and publishes it with
//! [`MvccGraph::commit`].
//!
//! - Readers never block: [`MvccGraph::read`] loads the `Arc` from an
//!   `ArcSwap`, which takes no lock.
//! - A writer builds into a fork ([`MvccGraph::fork`]): a `Graph` it owns,
//!   Copy-on-Write and invisible to readers until it is committed.
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
//!  Writer ──fork()──▶ Graph::new_version() ──▶ Graph v2 (owned, private)
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
//! the types allow only shared access to it. A fork is a `Graph` its writer
//! owns and changes through `&mut`; [`MvccGraph::commit`] takes it by value
//! and hands back the `Arc` it published.

use std::sync::Arc;

use arc_swap::ArcSwap;
use thiserror::Error;

use crate::graph::graph::{Graph, NodeOpError};
#[cfg(feature = "index-falkordb")]
use crate::index::falkordb::online_build;
#[cfg(feature = "index-falkordb")]
use crate::index::falkordb::versioned::{IndexerCommit, PreparedBatch};

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
    ) -> Arc<Self> {
        Self::from_graph(Graph::new(n, e, cache_size, 0, name))
    }

    /// Create an `MvccGraph` from an already-constructed `Graph`.
    /// Used by the RDB load path.
    #[must_use]
    pub fn from_graph(graph: Graph) -> Arc<Self> {
        Arc::new(Self {
            graph: ArcSwap::from_pointee(graph),
        })
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
    pub fn fork(&self) -> Graph {
        self.graph.load().new_version()
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
        fork: &Graph,
    ) -> bool {
        self.graph.load().version + 1 != fork.version
    }

    /// Bring a writer's `fork`, which is not stale ([`Self::is_stale`]), onto the indexer batches
    /// published since it was made. A batch changes no data, so what the writer read still holds:
    /// each index the batches moved on becomes the published copy, with their progress, and what
    /// the fork wrote in it is re-applied (`Graph::rebase_onto_indexer_batches`).
    ///
    /// For a writer that forked before it held the write lock, as it escalates, right after
    /// [`Self::is_stale`]: what it publishes and reads from then on has the batches' progress, and
    /// nothing else publishes before its own commit.
    #[cfg(feature = "index-falkordb")]
    pub fn rebase_onto_indexer_batches(
        &self,
        fork: &mut Graph,
    ) {
        let published = self.graph.load_full();
        // Until a build runs in the background, no published column is `Building`, and nothing
        // is rebased.
        if fork.overtaken_by_indexer_batches(&published) {
            fork.rebase_onto_indexer_batches(&published);
        }
    }

    /// Publish `fork` as the current version, and return it.
    ///
    /// Checks before touching anything, so a refusal leaves the published
    /// version untouched. This is the only place a version becomes visible, so
    /// every writer gets these checks; a writer that publishes unversioned
    /// state before it commits also asks [`Self::is_stale`] first.
    ///
    /// A refusal needs no cleanup: nothing was taken, and the refused version
    /// is dropped.
    ///
    /// Takes a shared reference: publishing is a compare-and-swap of the published
    /// pointer. The caller holds the host's global lock, which keeps commits
    /// apart and the swap out of the way of the host forking (#452); it needs no
    /// per-graph lock. It takes the `Arc` the graph lives in so that it can hand each background
    /// index build it starts a weak handle to it ([`Arc::downgrade`]).
    ///
    /// # Errors
    ///
    /// [`CommitError::Stale`] if `fork` no longer extends the published
    /// graph, or if another version was published while this commit ran (only
    /// a host publishing without the global lock lets that happen),
    /// [`CommitError::Invalid`] if [`Graph::validate`] refuses it.
    pub fn commit(
        self: &Arc<Self>,
        mut fork: Graph,
    ) -> Result<Arc<Graph>, CommitError> {
        let published = self.graph.load_full();
        Self::prepare(&published, &mut fork, true)?;
        // The indexes to build, asked of the version this one replaces before it is replaced.
        #[cfg(feature = "index-falkordb")]
        let to_build = fork.indexes_to_build_since(&published);
        let version = self.publish(&published, Arc::new(fork))?;
        // Each `Building` index this version brings in, new or created again, starts its build: a
        // job per generation, started only by the commit that brings it in, so a commit made during
        // the build, an indexer batch's included, starts nothing.
        #[cfg(feature = "index-falkordb")]
        for (id, generation) in to_build {
            online_build::enqueue(Arc::downgrade(self), id, generation);
        }
        Ok(version)
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
    /// - writers, and catching up would do more than `batch`'s limit allows
    ///   ([`PreparedBatch::limit_catch_up`]): nothing is published
    ///   ([`CatchUpTooLarge`](crate::index::falkordb::versioned::Discarded::CatchUpTooLarge)), and
    ///   the index's next batch starts from its published cursor;
    /// - `batch`'s index was dropped or created again, or another batch committed on the same
    ///   index: nothing is published, and the index's next batch starts from its published cursor.
    ///   Another batch on the index is a bug in the builder's one job per generation
    ///   ([`AnotherBatch`](crate::index::falkordb::versioned::Discarded::AnotherBatch)), which
    ///   leaves nothing wrong.
    ///
    /// The commit changes no graph data and keeps the data version, so it does not make a writer's
    /// fork stale ([`Self::is_stale`]).
    ///
    /// The caller holds the host's global lock, as a writer does to publish; it needs no
    /// per-graph lock. The fit is made against the published version this commit loads, and the
    /// version is published over that one: a batch that writers overtook is caught up, or, if that
    /// would do more than its limit allows, discarded; it is never refused as stale
    /// ([`CommitError::Stale`]) for being overtaken.
    ///
    /// # Errors
    ///
    /// [`CommitError::Invalid`] if [`Graph::validate`] refuses the version, [`CommitError::Stale`]
    /// if another version was published while this commit ran (only a host publishing without the
    /// global lock lets that happen). Nothing is published.
    #[cfg(feature = "index-falkordb")]
    pub(crate) fn commit_indexer_batch(
        &self,
        mut prepared: Graph,
        batch: &PreparedBatch,
    ) -> Result<IndexerCommit, CommitError> {
        let published = self.graph.load_full();
        let fit = prepared.indexer_fit(&published, batch);
        let version = match fit {
            IndexerCommit::Same => {
                Self::prepare(&published, &mut prepared, false)?;
                prepared
            }
            IndexerCommit::Swap => {
                let mut graph = prepared.with_batch_index(&published, batch);
                Self::prepare(&published, &mut graph, false)?;
                graph
            }
            IndexerCommit::CatchUp(how) => {
                let mut graph = prepared.caught_up_onto(&published, batch, how);
                Self::prepare(&published, &mut graph, false)?;
                graph
            }
            IndexerCommit::Discarded(_) => return Ok(fit),
        };
        self.publish(&published, Arc::new(version))?;
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
    ) -> Result<Arc<Graph>, CommitError> {
        let previous = self.graph.compare_and_swap(published, Arc::clone(&version));
        if !Arc::ptr_eq(&previous, published) {
            return Err(CommitError::Stale);
        }
        // After the swap, so a refused version is never handed to the indexers.
        version.set_indexer_graph(Arc::clone(&version));
        Ok(version)
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

        let mut invalid = mvcc.fork();
        // Ids 0 and 5 with nothing between: a batch that cannot have come from
        // an allocator, which is what `Graph::validate` refuses.
        let holed: RoaringTreemap = [0u64, 5].into_iter().collect();
        invalid
            .create_nodes(&holed)
            .expect("neither id is live, so the graph takes them");
        assert!(
            matches!(mvcc.commit(invalid), Err(CommitError::Invalid(_))),
            "a hole in the id space must refuse the version"
        );
        assert!(
            Arc::ptr_eq(&mvcc.read(), &published),
            "a refused version must not be published"
        );

        mvcc.commit(mvcc.fork())
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

        let inline = mvcc
            .commit(mvcc.fork())
            .expect("an open fork must not refuse another writer");

        assert!(
            mvcc.is_stale(&matching),
            "the fork predates the inline commit"
        );
        assert_eq!(mvcc.commit(matching).err(), Some(CommitError::Stale));
        assert!(
            Arc::ptr_eq(&mvcc.read(), &inline),
            "the refused fork must not replace the version it was overtaken by"
        );

        let fresh = mvcc.fork();
        assert!(!mvcc.is_stale(&fresh));
        mvcc.commit(fresh)
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

        let mut next = mvcc.fork();
        next.get_label_id_mut("L");
        let published = mvcc.commit(next).expect("a new label commits");

        assert_eq!(reader.version, version);
        assert!(
            reader.get_labels().is_empty(),
            "the reader's version is unchanged"
        );
        assert!(Arc::ptr_eq(&mvcc.read(), &published));
        assert_eq!(published.version, version + 1);
        assert_eq!(published.get_labels().len(), 1);
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
        let second = mvcc.fork();
        let first = mvcc.commit(mvcc.fork()).expect("the first version commits");

        assert_eq!(
            mvcc.publish(&checked, Arc::new(second)).err(),
            Some(CommitError::Stale),
            "checked against the replaced version, so not published"
        );
        assert!(
            Arc::ptr_eq(&mvcc.read(), &first),
            "the first version stays published"
        );
    }
}
