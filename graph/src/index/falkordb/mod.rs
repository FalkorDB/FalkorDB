//! The FalkorDB index (the in-repo replacement for RediSearch).
//!
//! [`data_structures`] is the always-compiled substrate (the CoW B⁺-tree). The
//! index proper — the numeric key [`encode`]r, the tree-backed index, and the
//! hooks that turn graph writes into index changes — lives behind the
//! `index-falkordb` feature, off by default.

pub mod data_structures;

#[cfg(feature = "index-falkordb")]
pub mod encode;

#[cfg(feature = "index-falkordb")]
pub mod falkordb_index;

#[cfg(feature = "index-falkordb")]
pub mod graph_writes;

#[cfg(feature = "index-falkordb")]
pub mod numeric;
