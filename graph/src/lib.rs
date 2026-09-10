#![allow(clippy::arc_with_non_send_sync)]
#![allow(clippy::type_complexity)]
#![allow(clippy::module_inception)]
// Dependency version duplicates are from transitive dependencies.
#![allow(clippy::multiple_crate_versions)]
// Cast warnings — inherent to architecture: graph IDs are u64, Cypher uses i64,
// matrices use u32/u16, and many cross-type conversions are unavoidable.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
// Functions in this codebase are often long due to match arms over IR/AST variants.
#![allow(clippy::too_many_lines)]
// Significant-drop false positives on moved/consumed values in nursery lint.
#![allow(clippy::significant_drop_tightening)]
// Raw pointer casts used in GraphBLAS FFI.
#![allow(clippy::ref_as_ptr)]
// Indexer must be Send despite raw pointers from C FFI.
#![allow(clippy::non_send_fields_in_send_ty)]
// Inherent to floating-point comparison in filter/index code.
#![allow(clippy::float_cmp)]
// Unused self in trait-like method signatures.
#![allow(clippy::unused_self)]
// collect() sometimes needed for lifetime reasons.
#![allow(clippy::needless_collect)]
// Pedantic style lints with low signal in this codebase.
#![allow(clippy::similar_names)]

//! # FalkorDB Graph Engine
//!
//! This crate contains the core graph database engine for FalkorDB.
//! It provides Cypher query parsing, planning, optimization, and execution
//! over a sparse matrix-based graph representation using GraphBLAS.
//!
//! ## Query Processing Pipeline
//!
//! ```text
//! Cypher Query String
//!        │
//!        ▼
//! ┌─────────────┐
//! │   parser    │  Parse query into AST (hand-written recursive descent)
//! └─────────────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │  planner    │  Bind, plan, and optimize the execution plan (IR)
//! └─────────────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │  runtime    │  Execute plan against the graph
//! └─────────────┘
//! ```
//!
//! ## Module Overview
//!
//! - [`parser`]: Cypher parser, AST definitions, and string escape utilities
//! - [`planner`]: Semantic binding, logical plan generation, and optimization
//! - [`graph`]: Graph data structures (sparse matrices, vectors, MVCC)
//! - [`graph::graphblas`]: GraphBLAS FFI bindings (auto-generated)
//! - [`index`]: Index types, management, and RediSearch FFI bindings
//! - [`locks`]: The lock seam the host provides (escalate a query to writer mode)
//! - [`runtime`]: Query execution engine and built-in functions
//! - [`storage`]: Shared storage seam (per-store backend registry; an in-memory
//!   backend by default, an alternate backend swapped in by a statically-linked add-on)
//! - [`entity_type`]: Node and relationship entity type discriminator
//! - [`identifier_limits`]: Maximum accepted length of an identifier
//! - [`threadpool`]: Thread pool for parallel query execution
//! - [`udf`]: User-defined function registration and dispatch

pub mod effects;
pub mod entity_type;
pub mod graph;
pub mod identifier_limits;
pub mod index;
pub mod locks;
pub mod narrow_int;
pub mod parser;
pub mod planner;
pub mod runtime;
pub mod storage;
pub mod thread_id;
pub mod threadpool;
pub mod udf;
