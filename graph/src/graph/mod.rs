//! Graph data structures and storage.
//!
//! This module contains the core graph representation using sparse matrices
//! backed by GraphBLAS for efficient graph operations.
//!
//! ## Architecture
//!
//! ```text
//!                    ┌──────────────────────────┐
//!                    │       MvccGraph           │  Concurrent access coordinator
//!                    │  (mvcc_graph.rs)          │  Snapshot isolation for readers,
//!                    │                           │  serialized writes via AtomicBool
//!                    └────────────┬──────────────┘
//!                                 │ Arc<AtomicRefCell<Graph>>
//!                                 ▼
//!                    ┌──────────────────────────┐
//!                    │         Graph             │  Core storage (graph.rs)
//!                    │                           │  Sparse matrices + attribute stores
//!                    └──┬──────────┬──────────┬──┘
//!                       │          │          │
//!          ┌────────────┘          │          └────────────┐
//!          ▼                       ▼                       ▼
//!  ┌───────────────┐   ┌────────────────────┐   ┌─────────────────┐
//!  │ GraphBLAS     │   │ AttributeStore     │   │ COW versioning  │
//!  │ (graphblas/)  │   │ (attribute_store)  │   │ (cow.rs)        │
//!  │               │   │                    │   │                 │
//!  │ Matrix/Tensor │   │ Cache + fjall      │   │ Lazy-duplicate  │
//!  │ FFI bindings  │   │ 2-tier storage     │   │ for matrices    │
//!  └───────────────┘   └────────┬───────────┘   └─────────────────┘
//!                               │
//!                               ▼
//!                    ┌────────────────────┐
//!                    │ AttributeCache     │
//!                    │ (attribute_cache)  │
//!                    │                    │
//!                    │ quick_cache LRU    │
//!                    │ Shared via Arc     │
//!                    └────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`graph::Graph`]: The main graph structure holding nodes, edges, labels, and properties
//! - [`mvcc_graph::MvccGraph`]: MVCC wrapper providing snapshot isolation for concurrent access
//! - [`cow::Cow`]: Copy-on-Write wrapper that defers matrix duplication until mutation
//! - [`attribute_store::AttributeStore`]: Two-tier (cache + fjall) property storage for entities
//! - [`attribute_cache::AttributeCache`]: Shared in-memory LRU cache for hot attribute data
//! - [`graphblas`]: FFI bindings to the GraphBLAS C library (auto-generated, do not edit)
//!
//! ## Storage Model
//!
//! The graph uses a sparse matrix representation where:
//! - Nodes are identified by 64-bit IDs (recycled via roaring bitmaps on deletion)
//! - Labels are stored as diagonal sparse matrices (node ID x node ID -> bool)
//! - The adjacency matrix tracks all edges (src x dst -> bool)
//! - Relationship types are stored as 3D tensors (src x dst x edge_id)
//! - Properties are stored in a columnar attribute store backed by cache + fjall
//!
//! ## Concurrency Model
//!
//! - **Readers** clone an `Arc` to the current committed `Graph` (lock-free)
//! - **Writers** create a new version via `Graph::new_version()`, which uses
//!   Copy-on-Write for matrices (only duplicated on first mutation)
//! - On commit, the `MvccGraph` atomically swaps the graph pointer
//! - On rollback, the versioned copy is simply discarded
//!
//! ## GraphBLAS Integration
//!
//! The underlying sparse matrix operations use GraphBLAS, a high-performance
//! library for graph algorithms using linear algebra. The [`graphblas`]
//! submodule contains the FFI bindings (auto-generated).

pub mod attribute_cache;
pub mod attribute_store;
pub mod constraint;
pub mod cow;
pub mod graph;
pub mod graphblas;
pub mod mvcc_graph;
