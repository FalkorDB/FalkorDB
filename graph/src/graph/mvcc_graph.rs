//! Multi-Version Concurrency Control (MVCC) for graph access.
//!
//! [`MvccGraph`] holds the published version of a graph. Readers share it; a
//! writer forks a private version, mutates it, and publishes it with
//! [`MvccGraph::commit`].
//!
//! - Readers never block: [`MvccGraph::read`] clones the `Arc`.
//! - A writer builds into a fork ([`MvccGraph::fork`]), Copy-on-Write and
//!   invisible to readers until it is committed.
//! - [`MvccGraph::commit`] publishes a fork only if it still extends the
//!   published version, so a fork that another commit overtook is refused
//!   rather than published over it.
//!
//! ## Who serializes writers
//!
//! The host does. `commit` takes `&mut self`, and the host only hands that out
//! under its per-graph write lock, so two commits cannot overlap and this
//! module needs no lock of its own. Forking takes nothing at all, which is
//! why a fork has to be checked for staleness instead.
//!
//! ```text
//!  MvccGraph
//!  ┌─────────────────────────────────────────────────────┐
//!  │  graph: Arc<AtomicRefCell<Graph>>  (published v1)   │
//!  └─────────────────────────────────────────────────────┘
//!
//!  Reader 1 ──read()──▶ clones Arc ──▶ sees Graph v1
//!  Reader 2 ──read()──▶ clones Arc ──▶ sees Graph v1
//!
//!  Writer ──fork()──▶ Graph::new_version() ──▶ Graph v2 (private)
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
//! Readers never block -- they simply clone the `Arc` to the current graph.
//! `AtomicRefCell` provides runtime borrow checking for the rare cases
//! where mutable access to the committed graph is needed (e.g., indexer
//! graph reference updates on commit).

use std::sync::Arc;

use atomic_refcell::AtomicRefCell;
use thiserror::Error;

use crate::graph::graph::{Graph, NodeOpError};
#[cfg(feature = "index-falkordb")]
use crate::index::falkordb::versioned::{IndexerCommit, PreparedBatch};

/// MVCC coordinator for concurrent graph access.
///
/// Provides snapshot isolation: readers see a consistent committed state
/// while a writer can make changes that become visible only on commit.
pub struct MvccGraph {
    /// Current committed graph version
    graph: Arc<AtomicRefCell<Graph>>,
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
        Self {
            graph: Arc::new(AtomicRefCell::new(Graph::new(n, e, cache_size, 0, name))),
        }
    }

    /// Create an `MvccGraph` from an already-constructed `Graph`.
    /// Used by the RDB load path.
    #[must_use]
    pub fn from_graph(graph: Graph) -> Self {
        Self {
            graph: Arc::new(AtomicRefCell::new(graph)),
        }
    }

    #[must_use]
    pub fn read(&self) -> Arc<AtomicRefCell<Graph>> {
        self.graph.clone()
    }

    /// A private version of the published graph to build a write into.
    ///
    /// Readers do not see it until [`Self::commit`] publishes it, and
    /// dropping it discards the write: forking takes nothing that has to be
    /// given back. So a writer can fork while it is still matching as a
    /// reader, and an inline writer that commits meanwhile is never refused
    /// for it (#3093) — the fork goes stale instead ([`Self::is_stale`]).
    #[must_use]
    pub fn fork(&self) -> Arc<AtomicRefCell<Graph>> {
        Arc::new(AtomicRefCell::new(self.graph.borrow().new_version()))
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
        fork: &AtomicRefCell<Graph>,
    ) -> bool {
        self.graph.borrow().version + 1 != fork.borrow().version
    }

    /// Publish `new_graph` as the current version.
    ///
    /// Checks before touching anything, so a refusal leaves the published
    /// version untouched and `new_graph` unpublished. This is the only place a
    /// version becomes visible, so every writer gets these checks; a writer
    /// that publishes unversioned state before it commits also asks
    /// [`Self::is_stale`] first.
    ///
    /// A refusal needs no cleanup: nothing was taken, and the refused version
    /// is just dropped.
    ///
    /// # Errors
    ///
    /// [`CommitError::Stale`] if `new_graph` no longer extends the published
    /// graph, [`CommitError::Invalid`] if [`Graph::validate`] refuses it.
    pub fn commit(
        &mut self,
        new_graph: Arc<AtomicRefCell<Graph>>,
    ) -> Result<(), CommitError> {
        self.commit_version(new_graph, true)
    }

    /// Commit an indexer batch. `prepared` is a build version of a published one
    /// ([`Graph::new_build_version`]) with `batch` already applied
    /// ([`Graph::indexer_prepare`]), made with no lock held. How it goes in depends on what
    /// committed since ([`Graph::indexer_fit`]):
    /// - nothing: `prepared` is published as it is;
    /// - only other indexes' batches: `batch`'s index is put into a new build version of the
    ///   published one, which is published ([`Graph::with_batch_index`]);
    /// - `batch`'s index was dropped or created again, or a writer committed, or another batch
    ///   committed on the same index: nothing is published, and the index's next batch starts from
    ///   its published cursor. Another batch on the index is a bug in the builder's one chain per
    ///   index ([`AnotherBatch`](crate::index::falkordb::versioned::Discarded::AnotherBatch)),
    ///   which the caller reports.
    ///
    /// The commit changes no graph data and keeps the data version, so it does not make a writer's
    /// fork stale ([`Self::is_stale`]).
    ///
    /// # Errors
    ///
    /// [`CommitError::Invalid`] if [`Graph::validate`] refuses the version. Nothing is published.
    #[cfg(feature = "index-falkordb")]
    pub fn commit_indexer_batch(
        &mut self,
        prepared: &Arc<AtomicRefCell<Graph>>,
        batch: &PreparedBatch,
    ) -> Result<IndexerCommit, CommitError> {
        let fit = prepared.borrow().indexer_fit(&self.graph.borrow(), batch);
        let version = match fit {
            IndexerCommit::Same => Arc::clone(prepared),
            IndexerCommit::Swap => Arc::new(AtomicRefCell::new(
                prepared
                    .borrow()
                    .with_batch_index(&self.graph.borrow(), batch),
            )),
            IndexerCommit::Discarded(_) => return Ok(fit),
        };
        self.commit_version(version, false)?;
        Ok(fit)
    }

    /// [`Self::commit`]: `writer` for a writer's version, `false` for an indexer batch's, which
    /// keeps the data version.
    fn commit_version(
        &mut self,
        new_graph: Arc<AtomicRefCell<Graph>>,
        writer: bool,
    ) -> Result<(), CommitError> {
        // Before any of the work below, so a refusal changes nothing. A batch's version is fitted
        // onto the published one, at its data version ([`Graph::indexer_fit`]): only a writer's
        // can be stale.
        if writer && self.is_stale(&new_graph) {
            return Err(CommitError::Stale);
        }
        debug_assert!(writer || self.graph.borrow().version == new_graph.borrow().version);
        new_graph.borrow().validate()?;
        #[cfg(feature = "index-falkordb")]
        new_graph
            .borrow_mut()
            .falkordb_index
            .set_batches_published(&self.graph.borrow().falkordb_index, !writer);

        // Check if schema changed (new labels, relationship types, or attributes)
        // Single borrow for old graph to collect all schema counts
        let (old_labels, old_types, old_node_attrs, old_rel_attrs, old_schema_version) = {
            let g = self.graph.borrow();
            (
                g.get_labels().len(),
                g.get_types().len(),
                g.get_node_attribute_names().len(),
                g.get_relationship_attribute_names().len(),
                g.schema_version,
            )
        };

        // Single borrow for new graph to collect all schema counts
        let (new_labels, new_types, new_node_attrs, new_rel_attrs, new_schema_version) = {
            let g = new_graph.borrow();
            (
                g.get_labels().len(),
                g.get_types().len(),
                g.get_node_attribute_names().len(),
                g.get_relationship_attribute_names().len(),
                g.schema_version,
            )
        };

        // If schema changed, ensure schema_version is incremented
        if (old_labels != new_labels
            || old_types != new_types
            || old_node_attrs != new_node_attrs
            || old_rel_attrs != new_rel_attrs)
            && new_schema_version == old_schema_version
        {
            new_graph.borrow_mut().schema_version += 1;
        }

        new_graph.borrow_mut().trim_attr_stores();
        new_graph.borrow_mut().end_id_quarantine();

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
        new_graph.borrow_mut().fold_oversized_deltas();

        // Use an immutable borrow here: `set_indexer_graph` only publishes
        // `new_graph` into the indexers' own `Mutex`-guarded fields. Holding
        // a mutable borrow across this call previously created a race with
        // the background index population thread, which fetches this same
        // graph reference from the indexer and immediately calls `.borrow()`
        // on it -- if that happened before this statement's `borrow_mut()`
        // guard was dropped, it panicked with "already mutably borrowed".
        new_graph.borrow().set_indexer_graph(new_graph.clone());

        self.graph = new_graph;
        Ok(())
    }
}

impl Drop for MvccGraph {
    fn drop(&mut self) {
        self.graph.borrow().cancel_indexing();
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
        let mut mvcc = MvccGraph::new(64, 64, 0, "t");
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
            matches!(mvcc.commit(invalid), Err(CommitError::Invalid(_))),
            "a hole in the id space must refuse the version"
        );
        assert!(
            Arc::ptr_eq(&mvcc.read(), &published),
            "a refused version must not be published"
        );

        let next = mvcc.fork();
        mvcc.commit(next)
            .expect("a refusal leaves nothing behind to block the next write");
    }

    /// A fork takes nothing, so a writer that forks while it is still
    /// matching never blocks one that commits meanwhile (#3093) — its own
    /// fork goes stale instead, and is refused rather than published over the
    /// other writer's version.
    #[test]
    fn a_fork_blocks_no_writer_and_goes_stale_when_one_commits() {
        ensure_init();
        let mut mvcc = MvccGraph::new(64, 64, 0, "t");

        let matching = mvcc.fork();
        assert!(!mvcc.is_stale(&matching));

        let inline = mvcc.fork();
        mvcc.commit(Arc::clone(&inline))
            .expect("an open fork must not refuse another writer");

        assert!(
            mvcc.is_stale(&matching),
            "the fork predates the inline commit"
        );
        assert_eq!(mvcc.commit(matching), Err(CommitError::Stale));
        assert!(
            Arc::ptr_eq(&mvcc.read(), &inline),
            "the refused fork must not replace the version it was overtaken by"
        );

        let fresh = mvcc.fork();
        assert!(!mvcc.is_stale(&fresh));
        mvcc.commit(fresh)
            .expect("a fork of the current version commits");
    }
}
