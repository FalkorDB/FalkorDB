//! Raw bindings to LAGraph (`LAGraph.h` and the experimental `LAGraphX.h`).
//!
//! Everything here is bindgen output (`bindings.rs`) and carries the C API's
//! contract unchanged: every function is `unsafe`, returns an `int` status
//! and writes its error text into a caller-supplied `msg` buffer of
//! [`LAGRAPH_MSG_LEN`] bytes. The `lagraph` crate is the one consumer.
//!
//! GraphBLAS types are not redeclared: the bindings name `GrB_Matrix`,
//! `GrB_Vector` and the rest from `graphblas-sys`, re-exported here. The
//! exception is `GrB_Info`: a LAGraph function declared to return it
//! returns a plain `c_int` here.
//!
//! The library is linked by `build.rs` from the archives `native-deps`
//! builds for the submodule pin, so the bindings and the binary come from
//! the same LAGraph revision.

pub use graphblas_sys::*;

/// C's `FILE`, which the print and Matrix Market functions take by pointer.
/// Opaque: its layout is the C library's and differs per platform.
#[repr(C)]
pub struct FILE {
    _opaque: [u8; 0],
}

/// The bindgen output, in a module of its own so that the naming and style
/// lints are off for it and for nothing else.
#[allow(
    unused,
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    clippy::use_self,
    clippy::unreadable_literal,
    clippy::pub_underscore_fields,
    clippy::derive_partial_eq_without_eq,
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
    clippy::wildcard_imports,
    reason = "bindgen output: it mirrors the C header and is not edited"
)]
mod generated {
    use graphblas_sys::*;

    use super::FILE;

    /// LAGraph declares some functions as returning `GrB_Info`, but its
    /// code is not covered by the check that GraphBLAS returns only listed
    /// statuses, and may return one of its own (`LAGRAPH_INVALID_GRAPH`).
    /// Here the status is the `int` it is in the C ABI, shadowing the Rust
    /// enum of `graphblas-sys`.
    type GrB_Info = ::std::os::raw::c_int;

    include!("bindings.rs");
}

pub use generated::*;

#[cfg(test)]
mod tests {
    use std::os::raw::c_int;

    use super::{GrB_Index, GrB_Matrix, GrB_Vector, LAGraph_BF_basic};

    #[test]
    fn a_lagraph_function_declared_to_return_grb_info_returns_an_int() {
        let _: unsafe extern "C" fn(*mut GrB_Vector, GrB_Matrix, GrB_Index) -> c_int =
            LAGraph_BF_basic;
    }
}
