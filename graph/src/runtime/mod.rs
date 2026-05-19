//! Query execution runtime.
//!
//! This module contains the query execution engine that evaluates execution
//! plans against the graph.
//!
//! ```text
//!  Cypher query pipeline (final stages)
//!
//!    IR Plan Tree (from planner/optimizer)
//!          |
//!          v
//!    +-----------+     +-----------+     +----------+
//!    | runtime   |---->| eval      |---->| value    |
//!    | (execute) |     | (exprs)   |     | (types)  |
//!    +-----------+     +-----------+     +----------+
//!          |
//!    +-----+------+----------+
//!    |     |      |          |
//!    v     v      v          v
//!  batch  env  pending   pool
//! ```
//!
//! ## Key Components
//!
//! - [`runtime::Runtime`]: The main execution engine that processes plan operators
//! - [`value::Value`]: Runtime representation of all Cypher values
//! - [`eval::ExprEval`]: Expression evaluator (used by runtime and optimizer)
//! - [`functions`]: Built-in Cypher function implementations
//! - [`pending`]: Deferred write operations for transactional semantics
//!
//! ## Execution Model
//!
//! The runtime uses a pull-based iterator model where each operator pulls
//! batches of rows from its children. This enables lazy evaluation and early
//! termination for LIMIT clauses. Rows flow through the operator tree in
//! [`batch::Batch`] units of up to 1024 rows.
//!
//! ## Supporting Infrastructure
//!
//! - [`batch::Batch`]: Columnar row batches with selection-vector filtering
//! - [`env::Env`]: Variable-binding tuple flowing through the pipeline
//! - [`pool::Pool`]: Per-query object pool to amortize allocation cost
//! - [`bitset::BitSet`]: Compact bit set for tracking bound variables
//! - [`vectorized`]: SIMD-friendly comparison kernels for typed columns
//!
//! ## Data Structures
//!
//! - [`ordermap::OrderMap`]: Insertion-ordered map for consistent iteration
//! - [`orderset::OrderSet`]: Insertion-ordered set for label/type collections

pub mod batch;
pub mod bitset;
pub mod env;
pub mod eval;
pub mod functions;
pub mod ops;
pub mod ordermap;
pub mod orderset;
pub mod pending;
pub mod pool;
pub mod runtime;
pub mod string_pool;
pub mod value;
pub mod vec_distance;
pub mod vectorized;
