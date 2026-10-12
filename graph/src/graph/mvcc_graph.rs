//! Multi-Version Concurrency Control (MVCC) for graph access.
//!
//! This module provides [`MvccGraph`], the top-level coordinator for concurrent
//! graph access. It ensures:
//!
//! - Multiple readers can access the graph simultaneously (lock-free)
//! - Only one writer at a time (serialized via the write slot, an atomic [`Slot`])
//! - Writers work on a Copy-on-Write versioned copy, committing atomically
//! - Readers always see a consistent, committed snapshot
//!
//! ## Concurrency Model
//!
//! ```text
//!  MvccGraph
//!  ┌─────────────────────────────────────────────────────┐
//!  │  graph: Arc<AtomicRefCell<Graph>>  (committed v1)   │
//!  │  write: AtomicSlot (free / writer / build step)      │
//!  └─────────────────────────────────────────────────────┘
//!
//!  Reader 1 ──read()──▶ clones Arc ──▶ sees Graph v1
//!  Reader 2 ──read()──▶ clones Arc ──▶ sees Graph v1
//!
//!  Writer ──write()──▶ CAS(free→writer) ──▶ Graph::new_version() ──▶ Graph v2
//!    │                                       (COW matrices, COW AttributeStore)
//!    │── mutations on v2 ──▶ ...
//!    │── commit(v2) ──▶ swap graph pointer, store(free)
//!    │
//!    │  Readers now see v2; v1 is dropped when last Arc goes away
//!
//!  Failed writer ──write()──▶ slot held by a writer ──▶ returns None
//! ```
//!
//! A background index-build step takes the slot with `write_for_build()` and gives it up to the
//! first writer that asks: `write()` takes a slot held by a build step over instead of failing,
//! and the step, finding it lost the slot (`claim_build()` returns `false`), drops its version.
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
//! The write slot only serializes write acquisition, not read access.
//! `AtomicRefCell` provides runtime borrow checking for the rare cases
//! where mutable access to the committed graph is needed (e.g., indexer
//! graph reference updates on commit).

use std::sync::{Arc, atomic::Ordering};

use atomic_enum::atomic_enum;
use atomic_refcell::AtomicRefCell;

use crate::graph::graph::{Graph, NodeOpError};

/// Who holds the write slot.
#[atomic_enum]
#[derive(PartialEq, Eq)]
enum Slot {
    /// Nobody.
    Free,
    /// A writer; only its `commit` or `rollback` frees the slot.
    Writer,
    /// A background index-build step, which gives the slot up to any writer that asks (see
    /// [`MvccGraph::write_for_build`]).
    Build,
}

/// MVCC coordinator for concurrent graph access.
///
/// Provides snapshot isolation: readers see a consistent committed state
/// while a writer can make changes that become visible only on commit.
pub struct MvccGraph {
    /// Current committed graph version
    graph: Arc<AtomicRefCell<Graph>>,
    /// The write slot: who holds it.
    write: AtomicSlot,
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
            write: AtomicSlot::new(Slot::Free),
        }
    }

    /// Create an `MvccGraph` from an already-constructed `Graph`.
    /// Used by the RDB load path.
    #[must_use]
    pub fn from_graph(graph: Graph) -> Self {
        Self {
            graph: Arc::new(AtomicRefCell::new(graph)),
            write: AtomicSlot::new(Slot::Free),
        }
    }

    #[must_use]
    pub fn read(&self) -> Arc<AtomicRefCell<Graph>> {
        self.graph.clone()
    }

    /// Take the write slot and a private new version to write into. `None` if another writer holds
    /// it. A slot a build step holds is taken over: the step gives way to every writer (see
    /// [`Self::write_for_build`]).
    #[must_use]
    pub fn write(&self) -> Option<Arc<AtomicRefCell<Graph>>> {
        let mut seen = self.write.load(Ordering::Relaxed);
        loop {
            if seen == Slot::Writer {
                return None;
            }
            match self.write.compare_exchange_weak(
                seen,
                Slot::Writer,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(self.new_version()),
                Err(now) => seen = now,
            }
        }
    }

    /// Take the write slot for a background index-build step. `None` if anyone holds it.
    ///
    /// The step writes into the version this returns without holding any lock, and gives the slot
    /// up to the first writer that asks: [`Self::write`] takes it over instead of failing. Before
    /// committing, the step calls [`Self::claim_build`] to become an ordinary writer; if a writer
    /// took the slot meanwhile, that returns `false` and the step drops its version, which was
    /// forked from a version the writer is about to replace.
    ///
    /// Assumes one build step per graph at a time — the host's, which runs them one after another.
    #[must_use]
    pub fn write_for_build(&self) -> Option<Arc<AtomicRefCell<Graph>>> {
        self.write
            .compare_exchange(
                Slot::Free,
                Slot::Build,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .ok()
            .map(|_| self.new_version())
    }

    /// Turn a build step's hold on the write slot into an ordinary writer's, so it can
    /// [`commit`](Self::commit). `false` if a writer took the slot over: nothing was published
    /// in between, but the step's version is no longer the next one, so the step must drop it, and
    /// the slot is no longer the step's to release.
    #[must_use]
    pub fn claim_build(&self) -> bool {
        self.write
            .compare_exchange(
                Slot::Build,
                Slot::Writer,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    /// Give up a build step's hold on the write slot without committing. A no-op if a writer took
    /// the slot over already.
    pub fn abandon_build(&self) {
        let _ = self.write.compare_exchange(
            Slot::Build,
            Slot::Free,
            Ordering::Release,
            Ordering::Relaxed,
        );
    }

    fn new_version(&self) -> Arc<AtomicRefCell<Graph>> {
        Arc::new(AtomicRefCell::new(self.graph.borrow().new_version()))
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
    /// atomic for the whole graph, so returning without clearing it makes
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

        // Wake the online index builder when this version brings in a `Building` column — a
        // `CREATE INDEX` however it arrived. Not for a version that only carries on a build:
        // the builder is already on it, and its own commits would wake it every turn.
        #[cfg(feature = "index-falkordb")]
        let wake_builder = {
            use crate::index::falkordb::graph_writes::has_building_columns;
            has_building_columns(&new_graph.borrow()) && !has_building_columns(&self.graph.borrow())
        };

        self.graph = new_graph;
        self.write.store(Slot::Free, Ordering::Release);
        #[cfg(feature = "index-falkordb")]
        if wake_builder {
            crate::index::falkordb::online_build::wake();
        }
        Ok(())
    }

    /// Release the write slot a writer holds, dropping nothing: the caller drops its version.
    pub fn rollback(&self) {
        self.write.store(Slot::Free, Ordering::Release);
    }
}

impl Drop for MvccGraph {
    fn drop(&mut self) {
        self.graph.borrow().cancel_indexing();
    }
}

#[cfg(test)]
mod tests {
    use super::MvccGraph;
    use crate::graph::graphblas::test_init::ensure_init;
    use roaring::RoaringTreemap;

    /// A refused version must leave the write slot free.
    ///
    /// The slot is one atomic for the whole graph, so a `commit` that
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

    /// A writer takes the slot over from a build step instead of failing, and the step then finds
    /// it lost the slot: it cannot claim it, and giving it up does not release the writer's hold.
    #[test]
    fn a_writer_takes_the_slot_over_from_a_build_step() {
        ensure_init();
        let mut mvcc = MvccGraph::new(64, 64, 0, "t");
        let step = mvcc.write_for_build().expect("the slot starts free");

        let writer = mvcc
            .write()
            .expect("a writer never fails on a build step's hold");
        assert!(!mvcc.claim_build(), "the step lost the slot");
        mvcc.abandon_build();
        assert!(
            mvcc.write().is_none(),
            "the step giving up must not release the writer's hold"
        );
        assert!(mvcc.write_for_build().is_none());

        drop(step);
        let version = writer.borrow().version;
        mvcc.commit(writer).expect("the writer commits as usual");
        assert_eq!(mvcc.read().borrow().version, version);
        assert!(mvcc.write_for_build().is_some(), "and frees the slot");
    }

    /// A build step nobody interrupted claims the slot and commits like a writer; once it has
    /// claimed it, writers fail as they do against any writer.
    #[test]
    fn a_build_step_that_keeps_the_slot_commits() {
        ensure_init();
        let mut mvcc = MvccGraph::new(64, 64, 0, "t");
        let step = mvcc.write_for_build().expect("the slot starts free");
        assert!(mvcc.claim_build());
        assert!(
            mvcc.write().is_none(),
            "a claimed slot is a writer's, not a build step's"
        );
        let version = step.borrow().version;
        mvcc.commit(step).expect("commits");
        assert_eq!(mvcc.read().borrow().version, version);
        assert!(mvcc.write().is_some(), "commit frees the slot");
    }

    /// A build step never takes a slot anyone holds, and giving up a slot it still holds frees it.
    #[test]
    fn a_build_step_takes_only_a_free_slot() {
        ensure_init();
        let mvcc = MvccGraph::new(64, 64, 0, "t");
        let writer = mvcc.write().expect("the slot starts free");
        assert!(mvcc.write_for_build().is_none(), "a writer holds it");
        drop(writer);
        mvcc.rollback();

        let step = mvcc.write_for_build().expect("free again");
        assert!(mvcc.write_for_build().is_none(), "a build step holds it");
        drop(step);
        mvcc.abandon_build();
        assert!(mvcc.write().is_some(), "giving up frees the slot");
    }
}
