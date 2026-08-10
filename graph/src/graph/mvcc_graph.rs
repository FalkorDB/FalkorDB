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

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use atomic_refcell::AtomicRefCell;

use crate::graph::graph::Graph;

/// MVCC coordinator for concurrent graph access.
///
/// Provides snapshot isolation: readers see a consistent committed state
/// while a writer can make changes that become visible only on commit.
pub struct MvccGraph {
    /// Current committed graph version
    graph: Arc<AtomicRefCell<Graph>>,
    /// Write lock (true = write in progress)
    write: AtomicBool,
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
            write: AtomicBool::new(false),
        }
    }

    /// Create an `MvccGraph` from an already-constructed `Graph`.
    /// Used by the RDB load path.
    #[must_use]
    pub fn from_graph(graph: Graph) -> Self {
        Self {
            graph: Arc::new(AtomicRefCell::new(graph)),
            write: AtomicBool::new(false),
        }
    }

    #[must_use]
    pub fn read(&self) -> Arc<AtomicRefCell<Graph>> {
        self.graph.clone()
    }

    #[must_use]
    pub fn write(&self) -> Option<Arc<AtomicRefCell<Graph>>> {
        if self
            .write
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(Arc::new(AtomicRefCell::new(
                self.graph.borrow().new_version(),
            )))
        } else {
            None
        }
    }

    pub fn commit(
        &mut self,
        new_graph: Arc<AtomicRefCell<Graph>>,
    ) {
        debug_assert_eq!(self.graph.borrow().version + 1, new_graph.borrow().version);

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

        self.graph = new_graph;
        self.write.store(false, Ordering::Release);
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
