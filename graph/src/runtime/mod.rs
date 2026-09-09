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
//!    +-----+------+
//!    |     |      |
//!    v     v      v
//!  batch  row  pending
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
//! - [`bitset::BitSet`]: Compact bit set for tracking bound variables
//! - [`vector_expr`]: Columnar evaluation of whole expression trees
//! - [`vectorized`]: SIMD-friendly comparison kernels for typed columns
//!
//! ## Data Structures
//!
//! - [`ordermap::OrderMap`]: Insertion-ordered map for consistent iteration
//! - [`orderset::OrderSet`]: Insertion-ordered set for label/type collections
//!
//! ## Output
//!
//! - [`double_format`]: `%.*g` double rendering matching the C implementation

pub mod batch;
pub mod bitset;
pub mod double_format;
pub mod eval;
pub mod functions;
pub mod ops;
pub mod ordermap;
pub mod orderset;
pub mod pending;
pub mod row;
pub mod runtime;
pub mod string_pool;
pub mod value;
pub mod vec_distance;
pub mod vector_expr;
pub mod vectorized;
