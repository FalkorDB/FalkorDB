//! # UDF (User-Defined Functions) Module
//!
//! This module enables users to define custom functions in JavaScript that can
//! be invoked from Cypher queries. It uses [QuickJS](https://bellard.org/quickjs/)
//! (via the `rquickjs` crate) as the embedded JavaScript runtime.
//!
//! ## Architecture Overview
//!
//! ```text
//!   Cypher Query: RETURN mylib.myfunc(n)
//!          |
//!          v
//!   +-------------+     +--------------+     +----------------+
//!   | repository   |<--->| js_context   |---->| js_globals     |
//!   | (UdfRepo)    |     | (QuickJS     |     | (falkor.register,|
//!   | stores libs  |     |  runtime +   |     |  falkor.log,   |
//!   | & metadata   |     |  call bridge)|     |  graph.traverse)|
//!   +-------------+     +--------------+     +----------------+
//!                              |
//!                    +---------+---------+
//!                    |                   |
//!                    v                   v
//!            +-------------+     +----------------+
//!            | js_classes   |     | type_convert   |
//!            | (Node, Edge, |     | (Rust Value <->|
//!            |  Path in JS) |     |  JS Value)     |
//!            +-------------+     +----------------+
//! ```
//!
//! ## Submodules
//!
//! - [`js_classes`] -- JS object representations of graph types (Node, Edge, Path)
//!   with methods like `getNeighbors()` and `graph.traverse()`.
//! - [`js_context`] -- Thread-local QuickJS runtime management, script validation,
//!   and the `call_udf_bridge` entry point used by the query evaluator.
//! - [`js_globals`] -- Setup of the `falkor` global object in JS, including
//!   `falkor.register()` and `falkor.log()`, in both validation and runtime modes.
//! - [`repository`] -- Thread-safe, versioned storage of UDF library definitions
//!   (name, source code, registered function names).
//! - [`type_convert`] -- Bidirectional conversion between Rust [`Value`](crate::runtime::value::Value)
//!   and QuickJS values, handling all supported types including graph entities.
//!
//! ## Global Singleton
//!
//! A process-wide [`UdfRepo`] is stored in a `OnceLock` and initialized once
//! via [`init_udf_repo`]. All subsequent access goes through [`get_udf_repo`].

pub mod js_classes;
pub mod js_context;
pub mod js_globals;
pub mod repository;
pub mod type_convert;

use repository::UdfRepo;
use std::sync::OnceLock;

static UDF_REPO: OnceLock<UdfRepo> = OnceLock::new();

pub fn init_udf_repo() {
    let _ = UDF_REPO.set(UdfRepo::new());
}

pub fn get_udf_repo() -> &'static UdfRepo {
    UDF_REPO.get().expect("UDF repository not initialized")
}
