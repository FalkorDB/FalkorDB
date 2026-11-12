//! A thin, typed, safe layer over the GraphBLAS C API.
//!
//! This crate is the only code allowed to call `graphblas-sys`, and the only
//! code allowed to contain `unsafe` on the GraphBLAS side of the engine. The
//! engine depends on this crate and never sees the sys crate.
//!
//! # Rules
//!
//! * **No implicit casts.** GraphBLAS casts among built-in types silently;
//!   this layer refuses, exactly as GraphBLAS itself refuses for user-defined
//!   types. The one exception so far is explicit: operators that never read a
//!   value are typed with [`Pattern`] and accept any input type.
//! * **Failures are `Result`s.** [`Error`] carries an [`ErrorKind`], matched
//!   from the status the C call returned, and GraphBLAS's message for it.
//!   Queries that cannot fail on a live handle return the value.
//! * **User-defined objects are statics.** A type is declared with
//!   [`user_type!`]; it is created on first use and lives as long as the
//!   process.
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

/// The generated bindings, reachable from inside this crate only.
pub(crate) use graphblas_sys as sys;

/// For the exported `user_type!` macro only; not part of the API.
#[doc(hidden)]
pub use graphblas_sys as __sys;
/// The index type of the C API, as it appears in this crate's signatures.
pub use sys::GrB_Index;
/// For the exported macros only; not part of the API.
#[doc(hidden)]
pub mod __sealed {
    pub use crate::sealed::Sealed;
}
#[doc(hidden)]
pub use types::user_defined::UserTypeCell as __UserTypeCell;

mod error;
mod init;
mod types;

/// The supertrait that keeps [`GrbType`] closed to this crate: every
/// implementation hands a raw pointer to C.
mod sealed {
    /// Closes the element-type and freezable traits. Re-exported, hidden,
    /// because `user_type!` implements it for the caller's type.
    pub trait Sealed {}
}

#[cfg(test)]
mod tests;

pub use error::{Error, ErrorKind, Result};
pub use init::{Allocators, JitControl, init, is_initialized, set_burble, set_jit, set_nthreads};
pub use types::{Accepts, BuiltinPattern, BuiltinType, GrbType, Pattern};
