//! The native FalkorDB index (the in-repo replacement for RediSearch).
//!
//! Currently this holds only the standalone [`data_structures`]; the index trait, encoders, and
//! runtime wiring land in subsequent changes.

pub mod data_structures;
