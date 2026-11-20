//! A thin, typed, safe layer over the GraphBLAS C API.
//!
//! This crate is the only code allowed to call `graphblas-sys`, and the only
//! code allowed to contain `unsafe` on the GraphBLAS side of the engine. The
//! engine depends on this crate and never sees the sys crate.
//!
//! # Rules
//!
//! * **C argument order, verbatim.** Every operation is a free function whose
//!   parameters are in the order the C API lists them:
//!   `(C, Mask, accum, op, A, B, desc)`. Nothing is defaulted or chosen on the
//!   caller's behalf: no hidden accumulator, operator or descriptor.
//! * **The mask names its mode.** [`Mask`] is built as `Mask::NONE`,
//!   `Mask::structural(m)`, `Mask::structural_complement(m)`,
//!   `Mask::valued(m)` or `Mask::valued_complement(m)`; the descriptor's mask
//!   bits are derived from it. Valued masks are `bool` only (see the casting
//!   rule), or [`AsBool`] of a built-in type.
//! * **The descriptor is data.** [`Desc`] holds `replace`, `transpose_a`,
//!   `transpose_b`; the built-in `GrB_DESC_*` handle is looked up inside.
//! * **Operators carry their domains.** [`UnaryOp<X, Z>`],
//!   [`BinaryOp<X, Y, Z>`], [`IndexUnaryOp<X, Y, Z>`], [`Monoid<T>`] and
//!   [`Semiring<X, Y, Z>`] are typed handles, checked against the matrices
//!   they are applied to through [`Accepts`].
//! * **No implicit casts.** GraphBLAS casts among built-in types silently;
//!   this layer refuses, exactly as GraphBLAS itself refuses for user-defined
//!   types. The one exception so far is explicit: operators that never read a
//!   value are typed with [`Pattern`] and accept any input type.
//! * **Aliasing is spelled.** `&mut c` with `&a` cannot alias; the marker
//!   [`Output`] in an input or mask slot says "the output matrix here".
//! * **Built-ins need no `unsafe`.** Each is a [`Lazy`] static that reads
//!   the C symbol once after [`init`] and panics if read before it.
//! * **Failures are `Result`s.** [`Error`] carries an [`ErrorKind`], matched
//!   from the status the C call returned, and GraphBLAS's message for it.
//!   Queries that cannot fail on a live handle return the value.
//! * **User-defined objects are statics.** A type is declared with
//!   [`user_type!`], operators with [`user_ops!`]; each is created on first
//!   use and lives as long as the process.
//!
//! # What does not compile
//!
//! A valued mask of a non-`bool` matrix (spell the cast: `Mask::valued(AsBool(&m))`):
//!
//! ```compile_fail,E0277
//! use graphblas::*;
//! let m = Matrix::<u64>::new(2, 2).unwrap();
//! let _ = Mask::<'_, graphblas::__sys::GrB_Matrix>::valued(&m);
//! ```
//!
//! An operand of another crate's making, which could hand C any pointer:
//!
//! ```compile_fail,E0050
//! use graphblas::{__sys::GrB_Matrix, *};
//! struct Local;
//! impl<'a> Operand<'a, Local, GrB_Matrix> for AsBool<Local> {
//!     type Elem = bool;
//!     fn raw(self, output: GrB_Matrix) -> GrB_Matrix {
//!         output
//!     }
//! }
//! ```
//!
//! # Thread safety
//!
//! Handles are `Send`, not `Sync`: GraphBLAS allows concurrent use of one
//! object only when every use is read-only and no work is pending, an
//! invariant the layer above this one owns. Operators and built-ins are
//! immutable after creation and are `Send + Sync`.
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

mod builtin;
mod error;
mod handle;
mod init;
mod lazy;
mod mask;
mod operand;
mod ops;
mod types;

/// The supertrait that keeps [`GrbType`], [`Operand`] and [`AsMask`] closed to this crate: every
/// implementation hands a raw pointer to C.
mod sealed {
    /// Closes the element-type and freezable traits. Re-exported, hidden,
    /// because `user_type!` implements it for the caller's type.
    pub trait Sealed {}

    /// Marks the types that may implement the traits whose safe methods
    /// hand a raw handle to C (`Operand`, `AsMask`). Never re-exported.
    pub trait Handle {}

    /// An argument of those methods. Sealing the type alone is not enough:
    /// another crate could still implement the trait with its own type as
    /// a parameter (`Operand<'_, Local, _> for AsBool<Local>`) and return
    /// any pointer. No other crate can name `Token`, so none can write the
    /// method, and none can implement the trait.
    #[derive(Clone, Copy)]
    pub struct Token;
}

pub mod matrix;

#[cfg(test)]
mod tests;

pub use error::{Error, ErrorKind, Result};
pub use init::{
    Allocators, JitControl, WaitMode, init, is_initialized, set_burble, set_jit, set_nthreads,
};
pub use lazy::Lazy;
pub use mask::{AsBool, AsMask, Desc, Mask, MaskRef, ValuedMask};
pub use matrix::Matrix;
pub use operand::{Operand, Output};
pub use ops::*;
pub use types::{Accepts, BuiltinPattern, BuiltinType, GrbType, Pattern};
