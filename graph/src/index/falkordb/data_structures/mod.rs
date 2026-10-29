//! Standalone data structures backing the FalkorDB (non-RediSearch) index.
//!
//! These are dependency-free building blocks (depend only on `std`); the index layer is wired on top
//! in later changes.

pub mod cow_btree;
