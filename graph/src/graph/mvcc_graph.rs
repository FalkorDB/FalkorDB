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
//!  ┌───────────────────────────────────────────────────────┐
//!  │  graph: ArcSwap<AtomicRefCell<Graph>>  (committed v1) │
//!  │  write: AtomicBool (false = no write in progress)     │
//!  └───────────────────────────────────────────────────────┘
//!
//!  Reader 1 ──read()──▶ loads Arc ──▶ sees Graph v1
//!  Reader 2 ──read()──▶ loads Arc ──▶ sees Graph v1
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
//! Readers never block -- they load the `Arc` to the current graph from an
//! `ArcSwap`, which takes no lock. The `AtomicBool` only serializes write
//! acquisition, not read access, and a commit publishes by storing into the
//! `ArcSwap`, so it needs `&self` and the write slot, not exclusive access.
//!
//! A published version is never changed in place: only a version nobody else
//! can see yet (a writer's fork, an indexer batch's version, before
//! [`MvccGraph::commit`] or [`MvccGraph::commit_indexer_batch`] stores it) is
//! borrowed mutably. A reader may hold its borrow of a published version for as
//! long as it likes, and `AtomicRefCell` turns any `borrow_mut` that met it into
//! a panic rather than a race.
//!
//! The write slot is only ever claimed by a thread holding the host's global
//! lock ([`crate::locks::holds_global_lock`]), which therefore serializes every
//! slot holder, and with it every store.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use arc_swap::ArcSwap;
use atomic_refcell::AtomicRefCell;

use crate::graph::graph::{Graph, NodeOpError};
#[cfg(feature = "index-falkordb")]
use crate::index::falkordb::versioned::{Dropped, IndexerCommit, PreparedBatch, report_batch_race};
use crate::locks::holds_global_lock;

/// MVCC coordinator for concurrent graph access.
///
/// Provides snapshot isolation: readers see a consistent committed state
/// while a writer can make changes that become visible only on commit.
pub struct MvccGraph {
    /// Current committed graph version. Stored only by the write slot's holder.
    graph: ArcSwap<AtomicRefCell<Graph>>,
    /// Write lock (true = write in progress)
    write: AtomicBool,
}

/// Why [`MvccGraph::claim`] refused a fork.
#[derive(Debug, PartialEq, Eq)]
pub enum Claim {
    /// Another writer holds the write slot.
    Busy,
    /// Another writer committed after the fork was made.
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
            graph: ArcSwap::from_pointee(AtomicRefCell::new(graph)),
            write: AtomicBool::new(false),
        }
    }

    /// The published version. Takes no lock; borrow it only immutably.
    ///
    /// Each call loads the pointer again, so two calls can return different
    /// versions if something committed in between.
    #[must_use]
    pub fn read(&self) -> Arc<AtomicRefCell<Graph>> {
        self.graph.load_full()
    }

    /// Claim the write slot and fork a private version to build into, in one
    /// step. For writers that already hold the per-graph write lock.
    #[must_use]
    pub fn write(&self) -> Option<Arc<AtomicRefCell<Graph>>> {
        debug_assert!(
            holds_global_lock(),
            "the write slot is claimed without the global lock"
        );
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
        Arc::new(AtomicRefCell::new(self.graph.load().borrow().new_version()))
    }

    /// Claim the write slot for a version made by [`Self::fork`].
    ///
    /// # Errors
    ///
    /// [`Claim::Busy`] if another writer holds the slot. [`Claim::Stale`] if
    /// another writer committed since the fork was made, so it no longer
    /// extends the committed version and must be discarded; the slot is left
    /// free.
    pub fn claim(
        &self,
        fork: &AtomicRefCell<Graph>,
    ) -> Result<(), Claim> {
        debug_assert!(
            holds_global_lock(),
            "the write slot is claimed without the global lock"
        );
        if self
            .write
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(Claim::Busy);
        }
        // Loaded once the slot is ours: only its holder stores, so it stays the published version.
        let published = self.graph.load_full();
        if published.borrow().version + 1 != fork.borrow().version {
            self.write.store(false, Ordering::Release);
            return Err(Claim::Stale);
        }
        // Only indexer batches committed since the fork was made, if anything did: it takes their
        // progress, with what it wrote re-applied. Borrowed mutably only then: until a build runs
        // in the background, no published column is `Building`, and a claim is what it was.
        #[cfg(feature = "index-falkordb")]
        if fork
            .try_borrow()
            .is_ok_and(|fork| !fork.overtaken_by_indexer_batches(&published.borrow()))
        {
            return Ok(());
        }
        #[cfg(feature = "index-falkordb")]
        {
            // A claimant's fork is never borrowed: its writer has stopped reading it to claim it.
            // One that is is a bug, which a debug build stops on. A release build refuses the
            // claim, and the host runs the write again.
            let Ok(mut fork) = fork.try_borrow_mut() else {
                self.write.store(false, Ordering::Release);
                if cfg!(debug_assertions) {
                    panic!("a fork is claimed while it is borrowed");
                }
                return Err(Claim::Stale);
            };
            fork.rebase_onto_indexer_batches(&published.borrow());
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
    /// Takes `&self`: publishing is a store into the `ArcSwap`, and the caller's
    /// write slot is what keeps two commits apart. The caller must hold the slot
    /// (from [`Self::write`] or [`Self::claim`]) and the host's global lock,
    /// which keeps the store out of the way of the host forking (#452).
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
        &self,
        new_graph: Arc<AtomicRefCell<Graph>>,
    ) -> Result<(), NodeOpError> {
        self.commit_version(self.graph.load_full(), new_graph, true)
    }

    /// Commit an indexer batch. `prepared` is a build version of a published one
    /// ([`Graph::new_build_version`]) with `batch` already applied
    /// ([`Graph::indexer_prepare`]), made with no lock held. How it goes in depends on what
    /// committed since ([`Graph::indexer_fit`]):
    /// - nothing: `prepared` is published as it is;
    /// - only other indexes' batches: `batch`'s index is put into a new build version of the
    ///   published one, which is published ([`Graph::with_batch_index`]);
    /// - writers, and no other batch on `batch`'s index: the index caught up on the entities the
    ///   writers changed in it, by refreshing them in `prepared`'s copy or by inserting the batch
    ///   less them into the published copy, whichever costs less, is put into a new build version
    ///   of the published one, which is published ([`Graph::caught_up_onto`]);
    /// - `batch`'s index was dropped or created again, or another batch committed on the same
    ///   index: nothing is published, and the index's next batch starts from its published cursor.
    ///   Another batch on the index is a bug in the builder's one chain per index, and is reported
    ///   ([`report_batch_race`]).
    ///
    /// The commit changes no graph data and keeps the data version, so a writer's fork made before
    /// it can still be claimed ([`Self::claim`]).
    ///
    /// The caller holds the host's global lock, as a writer does to publish, and the write slot is
    /// free; it needs no per-graph lock. Returns `None` if the slot was taken all the same, or the
    /// version was refused. The slot is released however the commit ends, an unwind included.
    #[cfg(feature = "index-falkordb")]
    pub fn commit_indexer_batch(
        &self,
        prepared: &Arc<AtomicRefCell<Graph>>,
        batch: &PreparedBatch,
    ) -> Option<IndexerCommit> {
        debug_assert!(
            holds_global_lock(),
            "the write slot is claimed without the global lock"
        );
        if self
            .write
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let slot = ReleaseOnDrop(&self.write);
        // Loaded once the slot is ours: only its holder stores, so the fit and the publish see the
        // same version.
        let published = self.graph.load_full();
        let fit = prepared.borrow().indexer_fit(&published.borrow(), batch);
        let version = match fit {
            IndexerCommit::Same => Arc::clone(prepared),
            IndexerCommit::Swap => Arc::new(AtomicRefCell::new(
                prepared
                    .borrow()
                    .with_batch_index(&published.borrow(), batch),
            )),
            IndexerCommit::CatchUp(how) => Arc::new(AtomicRefCell::new(
                prepared
                    .borrow()
                    .caught_up_onto(&published.borrow(), batch, how),
            )),
            IndexerCommit::Dropped(why) => {
                if why == Dropped::AnotherBatch {
                    report_batch_race(&batch.id());
                }
                return Some(fit);
            }
        };
        let committed = self.commit_version(published, version, false);
        // `commit_version` released the slot, whichever way it returned.
        std::mem::forget(slot);
        committed.ok().map(|()| fit)
    }

    /// [`Self::commit`]: `writer` for a writer's version, `false` for an indexer batch's, which
    /// keeps the data version. `published` is the published version, loaded by the caller after it
    /// took the slot: only the slot's holder stores, so it stays the published one until the
    /// store below.
    fn commit_version(
        &self,
        published: Arc<AtomicRefCell<Graph>>,
        new_graph: Arc<AtomicRefCell<Graph>>,
        writer: bool,
    ) -> Result<(), NodeOpError> {
        debug_assert!(
            self.write.load(Ordering::Relaxed),
            "commit without the write slot"
        );
        debug_assert!(holds_global_lock(), "commit without the global lock");
        debug_assert_eq!(
            published.borrow().version + u64::from(writer),
            new_graph.borrow().version
        );

        // Before any of the work below, so a refusal changes nothing — and
        // releasing the slot on the way out, so a refusal is a rollback rather
        // than a graph nobody can write to again.
        if let Err(e) = new_graph.borrow().validate() {
            self.rollback();
            return Err(e);
        }
        #[cfg(test)]
        if !writer && PANIC_IN_INDEXER_COMMIT.with(std::cell::Cell::get) {
            panic!("an indexer batch's commit unwinds");
        }
        #[cfg(feature = "index-falkordb")]
        new_graph.borrow_mut().falkordb_index.publish();

        // Check if schema changed (new labels, relationship types, or attributes)
        // Single borrow for old graph to collect all schema counts
        let (old_labels, old_types, old_node_attrs, old_rel_attrs, old_schema_version) = {
            let g = published.borrow();
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

        // Let go of the old version before the store, so that, as before, the
        // store frees it unless a reader still holds it.
        drop(published);
        self.graph.store(new_graph);
        self.write.store(false, Ordering::Release);
        Ok(())
    }

    pub fn rollback(&self) {
        self.write.store(false, Ordering::Release);
    }
}

/// Releases the write slot when dropped, unless forgotten: so a commit that unwinds does not leave
/// the slot taken, which would refuse every later write until the process restarts.
#[cfg(feature = "index-falkordb")]
struct ReleaseOnDrop<'a>(&'a AtomicBool);

#[cfg(feature = "index-falkordb")]
impl Drop for ReleaseOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl Drop for MvccGraph {
    fn drop(&mut self) {
        self.graph.load().borrow().cancel_indexing();
    }
}

#[cfg(test)]
thread_local! {
    /// For the test that an indexer batch's commit that unwinds releases the write slot: while
    /// set, the commit of an indexer batch's version panics after validating it.
    pub(crate) static PANIC_IN_INDEXER_COMMIT: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
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
        let mvcc = MvccGraph::new(64, 64, 0, "t");

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
        let mvcc = MvccGraph::new(64, 64, 0, "t");

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

    /// A reader keeps its borrow of the published version across a commit, as a
    /// read runtime does for its whole query. The commit borrows the old version
    /// only immutably and publishes by swapping the pointer, so the reader keeps
    /// the version it started on and the next `read()` returns the new one.
    #[test]
    fn a_commit_leaves_a_borrowed_published_version_alone() {
        ensure_init();
        let mvcc = MvccGraph::new(64, 64, 0, "t");
        let old = mvcc.read();
        let reader = old.borrow();
        let version = reader.version;

        let next = mvcc.write().expect("the slot starts free");
        next.borrow_mut().get_label_id_mut("L");
        mvcc.commit(next).expect("a new label commits");

        assert_eq!(reader.version, version);
        assert!(
            reader.get_labels().is_empty(),
            "the reader's version is unchanged"
        );
        assert_eq!(mvcc.read().borrow().version, version + 1);
        assert_eq!(mvcc.read().borrow().get_labels().len(), 1);
    }
}
