//! A thin, typed, safe layer over LAGraph, on top of the `graphblas` crate.
//!
//! This crate is the only code allowed to call `lagraph-sys`.
//!
//! # Rules
//!
//! * **Every call is a `Result`.** LAGraph reports a status and writes its
//!   reason into a `msg` buffer; both come back in [`Error`], never dropped.
//!   A positive status is a warning and counts as success.
//! * **A [`Graph`] owns its adjacency matrix and its cached properties.**
//!   LAGraph treats the graph object as mutable (see
//!   [`connected_components`]), so a matrix is moved in with
//!   [`Graph::new`] and moved back out with [`Graph::into_matrix`]; it is
//!   never shared with a reader on another thread while LAGraph has it.
//! * **The graph is held by row, and this crate makes it so.**
//!   `LAGraph_CheckGraph` rejects a matrix held by column, and GraphBLAS
//!   holds every `n x 1` matrix by column, the `1 x 1` matrix of a one-node
//!   graph included. [`Graph::new`] pins the adjacency matrix and
//!   [`Graph::cache_transpose`] pins the transpose LAGraph allocates.
//! * **A graph is checked when it is built**, not when the first algorithm
//!   runs on it.
//! * **An algorithm does not compute what it needs behind the caller's
//!   back.** These are LAGraph's "advanced" entry points: a missing cached
//!   property is [`Status::NotCached`], and the caller asks for it with a
//!   `cache_*` method, once, where the cost is visible.
//! * **Results are typed, and the type is checked.** LAGraph allocates its
//!   outputs; each is adopted as the `Vector<T>` its documentation states
//!   only after its element type has been compared with `T`'s.
//!
//! # Order of initialisation
//!
//! [`init`] alone, in place of [`graphblas::init`]: `LAGr_Init` starts
//! GraphBLAS itself, and shuts it down again when it fails. Neither library
//! is ever shut down; both stay up until the process exits.
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![allow(
    clippy::redundant_pub_crate,
    reason = "`pub(crate)` marks what is internal in a crate whose `pub` items are its API"
)]

/// The generated bindings, reachable from inside this crate only.
pub(crate) use lagraph_sys as sys;

mod algorithm;
mod error;
mod graph;
mod init;
mod output;

pub use algorithm::{betweenness, connected_components, page_rank};
pub use error::{Error, Result, Status};
pub use graph::{Graph, Kind};
pub use init::{init, is_initialized};
