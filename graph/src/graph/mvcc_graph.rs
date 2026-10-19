//! Multi-Version Concurrency Control (MVCC) for graph access.
//!
//! This module provides [`MvccGraph`], the top-level coordinator for concurrent
//! graph access. It ensures:
//!
//! - Multiple readers can access the graph simultaneously (lock-free)
//! - Only one writer at a time (serialized via `AtomicBool`)
//! - Writers work on a Copy-on-Write versioned copy, committing atomically
//! - Readers always see a consistent, committed snapshot
//!
//! ## Concurrency Model
//!
//! ```text
//!  MvccGraph
//!  ┌─────────────────────────────────────────────────────┐
//!  │  graph: Arc<AtomicRefCell<Graph>>  (committed v1)   │
//!  │  write: AtomicBool (false = no write in progress)   │
//!  └─────────────────────────────────────────────────────┘
//!
//!  Reader 1 ──read()──▶ clones Arc ──▶ sees Graph v1
//!  Reader 2 ──read()──▶ clones Arc ──▶ sees Graph v1
//!
//!  Writer ──write()──▶ CAS(false→true) ──▶ Graph::new_version() ──▶ Graph v2
//!    │                                       (COW matrices, COW AttributeStore)
//!    │── mutations on v2 ──▶ ...
//!    │── commit(v2) ──▶ swap graph pointer, store(false)
//!    │
//!    │  Readers now see v2; v1 is dropped when last Arc goes away
//!
//!  Failed writer ──write()──▶ CAS fails ──▶ returns None
//! ```
//!
//! With the `index-falkordb` feature, a background index-build step commits too
//! (`MvccGraph::commit_index_build`), and changes only the index. A writer whose fork is stale only
//! because build steps committed since it was made does not run again: [`MvccGraph::claim`] gives
//! its fork their progress, and it commits.
//!
//! ## Version Lifecycle
//!
//! ```text
//!  ┌──────┐   write()    ┌──────────┐  commit()   ┌───────────┐
//!  │  v1  │ ────────────▶│ v2 (wip) │ ──────────▶ │ v2 (live) │
//!  │(live)│              │          │             │           │
//!  └──────┘              └──────────┘             └───────────┘
//!                              │
//!                         rollback()
//!                              │
//!                              ▼
//!                         (discarded)
//! ```
//!
//! ## Thread Safety
//!
//! Readers never block -- they simply clone the `Arc` to the current graph.
//! The `AtomicBool` only serializes write acquisition, not read access.
//! `AtomicRefCell` provides runtime borrow checking for the rare cases
//! where mutable access to the committed graph is needed (e.g., indexer
//! graph reference updates on commit).

#[cfg(feature = "index-falkordb")]
use std::sync::OnceLock;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use atomic_refcell::AtomicRefCell;

use crate::graph::graph::{Graph, NodeOpError};
#[cfg(feature = "index-falkordb")]
use crate::index::falkordb::{
    falkordb_index::BuildBatch,
    online_build::{self, BuildHost},
};

/// MVCC coordinator for concurrent graph access.
///
/// Provides snapshot isolation: readers see a consistent committed state
/// while a writer can make changes that become visible only on commit.
pub struct MvccGraph {
    /// Current committed graph version
    graph: Arc<AtomicRefCell<Graph>>,
    /// Write lock (true = write in progress)
    write: AtomicBool,
    /// How the online index builder reaches this graph; set by its host.
    #[cfg(feature = "index-falkordb")]
    build_host: OnceLock<Arc<dyn BuildHost>>,
}

/// What [`MvccGraph::commit_index_build`] did with a batch.
#[cfg(feature = "index-falkordb")]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BuildCommit {
    /// The version it was prepared in was still current, and was published.
    Prepared,
    /// Something committed since: it was merged into the current version, which was published.
    Merged,
    /// Its column was dropped or re-created since it was read: nothing was published.
    Dropped,
}

/// Why [`MvccGraph::claim`] refused a fork.
#[derive(Debug, PartialEq, Eq)]
pub enum Claim {
    /// Another writer holds the write slot.
    Busy,
    /// A commit that changed graph data was made after the fork was.
    Stale,
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
            graph: Arc::new(AtomicRefCell::new(graph)),
            write: AtomicBool::new(false),
            #[cfg(feature = "index-falkordb")]
            build_host: OnceLock::new(),
        }
    }

    #[must_use]
    pub fn read(&self) -> Arc<AtomicRefCell<Graph>> {
        self.graph.clone()
    }

    /// Register how the online index builder reaches this graph. Until a host is set, a commit
    /// that starts a build queues nothing. The first host set is kept.
    #[cfg(feature = "index-falkordb")]
    pub fn set_build_host(
        &self,
        host: Arc<dyn BuildHost>,
    ) {
        let _ = self.build_host.set(host);
    }

    /// Take over `old`'s online-build host, for a coordinator that replaces `old` in place.
    /// Without it, a `CREATE INDEX` on the replacement would never be built.
    #[cfg(feature = "index-falkordb")]
    pub fn inherit_build(
        &self,
        old: &Self,
    ) {
        if let Some(host) = old.build_host.get() {
            let _ = self.build_host.set(Arc::clone(host));
        }
    }

    /// Claim the write slot and fork a private version to build into, in one
    /// step. For writers that already hold the per-graph write lock.
    #[must_use]
    pub fn write(&self) -> Option<Arc<AtomicRefCell<Graph>>> {
        if self
            .write
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(self.fork())
        } else {
            None
        }
    }

    /// A private version of the committed graph, *without* claiming the write
    /// slot.
    ///
    /// For a writer that matches as a reader before it mutates anything shared:
    /// it builds into the fork and claims the slot only when it escalates
    /// ([`Self::claim`]), so the slot is only ever held by a thread holding the
    /// per-graph write lock and an inline writer never finds it taken.
    #[must_use]
    pub fn fork(&self) -> Arc<AtomicRefCell<Graph>> {
        Arc::new(AtomicRefCell::new(self.graph.borrow().new_version()))
    }

    /// Claim the write slot for a version made by [`Self::fork`].
    ///
    /// With the `index-falkordb` feature, a fork that is stale only because index-build commits
    /// were made since it was is not refused: they changed no graph data, so what its writer
    /// matched still holds. The fork takes their progress and becomes the next version
    /// (`Graph::merge_build_progress`).
    ///
    /// # Errors
    ///
    /// [`Claim::Busy`] if another writer holds the slot. [`Claim::Stale`] if
    /// a commit that changed graph data was made since the fork, so it no longer
    /// extends the committed version and must be discarded; the slot is left
    /// free.
    pub fn claim(
        &self,
        fork: &AtomicRefCell<Graph>,
    ) -> Result<(), Claim> {
        if self
            .write
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(Claim::Busy);
        }
        let published = self.graph.borrow();
        if published.version + 1 != fork.borrow().version {
            // A fork that is borrowed, which no claimant's is, is refused rather than panicked on.
            #[cfg(feature = "index-falkordb")]
            if fork
                .try_borrow_mut()
                .is_ok_and(|mut fork| fork.merge_build_progress(&published))
            {
                return Ok(());
            }
            self.write.store(false, Ordering::Release);
            return Err(Claim::Stale);
        }
        Ok(())
    }

    /// Publish `new_graph` as the current version.
    ///
    /// Validates before touching anything, so a refusal leaves the published
    /// version untouched and the private one unpublished. This is the only
    /// place a version becomes visible, which makes it the only place the check
    /// has to be.
    ///
    /// A refusal *is* a [`Self::rollback`], performed here rather than left to
    /// the caller. That is not a convenience: the write slot is a single
    /// `AtomicBool` for the whole graph, so returning without clearing it makes
    /// every later `write()` fail until the process restarts. Five of the six
    /// callers would have had to remember, and the sixth is `unreachable!()`.
    ///
    /// # Errors
    ///
    /// Whatever [`Graph::validate`] refuses. The version is not published and
    /// the write slot is released.
    pub fn commit(
        &mut self,
        new_graph: Arc<AtomicRefCell<Graph>>,
    ) -> Result<(), NodeOpError> {
        self.commit_version(new_graph, true)
    }

    /// Commit an index-build batch. `prepared` is a new version of a published one with `batch`
    /// already merged in, made with no lock held. If that published version is still the current
    /// one, `prepared` holds every write and is published as it is. If anything has committed
    /// since, `prepared` holds older data and is dropped: the batch is merged into a new version
    /// of the current one ([`Graph::merge_index_build`]) and that is published instead. Either
    /// way the commit changes no graph data, so a writer whose fork it makes stale may still
    /// [`claim`](Self::claim) it.
    ///
    /// The caller holds what a writer holds to publish, so the write slot is free. `None` if it
    /// was taken all the same, or the version was refused.
    #[cfg(feature = "index-falkordb")]
    pub(crate) fn commit_index_build(
        &mut self,
        prepared: Arc<AtomicRefCell<Graph>>,
        batch: &mut BuildBatch,
    ) -> Option<BuildCommit> {
        if self
            .write
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let (version, how) = if self.graph.borrow().version + 1 == prepared.borrow().version {
            (prepared, BuildCommit::Prepared)
        } else {
            let current = Arc::new(AtomicRefCell::new(self.graph.borrow().new_version()));
            if !current.borrow_mut().merge_index_build(batch) {
                self.rollback();
                return Some(BuildCommit::Dropped);
            }
            (current, BuildCommit::Merged)
        };
        self.commit_version(version, false).ok().map(|()| how)
    }

    /// [`Self::commit`]; `writer` for a writer's version, `false` for an index build's.
    fn commit_version(
        &mut self,
        new_graph: Arc<AtomicRefCell<Graph>>,
        writer: bool,
    ) -> Result<(), NodeOpError> {
        debug_assert_eq!(self.graph.borrow().version + 1, new_graph.borrow().version);

        // Before any of the work below, so a refusal changes nothing — and
        // releasing the slot on the way out, so a refusal is a rollback rather
        // than a graph nobody can write to again.
        if let Err(e) = new_graph.borrow().validate() {
            self.rollback();
            return Err(e);
        }

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
        new_graph.borrow_mut().prepare_publish(writer);

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

        // A version that starts a build — it has a `Building` column, and the one it replaces has
        // none — queues this graph for the online index builder. The builder keeps the graph until
        // no column is `Building`, so a commit made during the build, the builder's own included,
        // queues nothing.
        #[cfg(feature = "index-falkordb")]
        let starts_build = new_graph.borrow().starts_index_build(&self.graph.borrow());

        self.graph = new_graph;
        self.write.store(false, Ordering::Release);
        #[cfg(feature = "index-falkordb")]
        if starts_build && let Some(host) = self.build_host.get() {
            online_build::enqueue(Arc::clone(host));
        }
        Ok(())
    }

    pub fn rollback(&self) {
        self.write.store(false, Ordering::Release);
    }
}

impl Drop for MvccGraph {
    fn drop(&mut self) {
        self.graph.borrow().cancel_indexing();
    }
}

#[cfg(test)]
mod tests {
    use super::{Claim, MvccGraph};
    use crate::graph::graphblas::test_init::ensure_init;
    use roaring::RoaringTreemap;

    /// A refused version must leave the write slot free.
    ///
    /// The slot is one `AtomicBool` for the whole graph, so a `commit` that
    /// returns without clearing it does not fail one write — it fails every
    /// write for the life of the process. Found in review, by two reviewers
    /// independently, after the validation was added here.
    #[test]
    fn a_refused_commit_releases_the_write_slot() {
        ensure_init();
        let mut mvcc = MvccGraph::new(64, 64, 0, "t");

        let version = mvcc.write().expect("the slot starts free");
        // Ids 0 and 5 with nothing between: a batch that cannot have come from
        // an allocator, which is what `Graph::validate` refuses.
        let holed: RoaringTreemap = [0u64, 5].into_iter().collect();
        version
            .borrow_mut()
            .create_nodes(&holed)
            .expect("neither id is live, so the graph takes them");

        assert!(
            mvcc.commit(version).is_err(),
            "a hole in the id space must refuse the version"
        );
        assert!(
            mvcc.write().is_some(),
            "the refusal must release the write slot, or no write ever succeeds again"
        );
    }

    /// A fork made while another writer is still free to commit: the slot is
    /// claimed only at escalation, and a commit in between makes the fork stale
    /// instead of making the other writer fail (#3093).
    #[test]
    fn a_fork_is_claimed_late_and_refused_once_stale() {
        ensure_init();
        let mut mvcc = MvccGraph::new(64, 64, 0, "t");

        let fork = mvcc.fork();
        // The fork holds no slot, so an inline writer still gets it.
        let inline = mvcc.write().expect("forking does not take the slot");
        assert_eq!(mvcc.claim(&fork), Err(Claim::Busy));
        mvcc.commit(inline).expect("an empty version commits");

        assert_eq!(
            mvcc.claim(&fork),
            Err(Claim::Stale),
            "the fork predates the inline commit"
        );
        // A stale claim leaves the slot free; a fresh fork claims it.
        let fresh = mvcc.fork();
        assert_eq!(mvcc.claim(&fresh), Ok(()));
        assert!(mvcc.write().is_none(), "the claim holds the slot");
        mvcc.commit(fresh).expect("the claimed fork commits");
        assert!(mvcc.write().is_some());
    }
}
