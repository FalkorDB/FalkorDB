//! The FalkorDB index (the in-repo replacement for RediSearch).
//!
//! The only Range index: [`data_structures`] is its substrate (the CoW B⁺-tree), and the index
//! proper is the [`range`] column with its numeric ([`encode`]), [`tag`] and [`geo`] kinds, and the
//! hooks that turn graph writes into index changes. Fulltext and Vector indexes are RediSearch's.
//! The [`cch`] shortest-path index's state is a field of the same
//! [`FalkorDbIndex`](falkordb_index::FalkorDbIndex).

pub mod data_structures;

pub mod cch;

pub mod encode;

pub mod falkordb_index;

pub mod geo;

pub mod graph_writes;

pub mod numeric;

pub mod online_build;

pub mod pending_writes;

pub mod range;

pub mod tag;

pub mod unique;

pub mod versioned;
