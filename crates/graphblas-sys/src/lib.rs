//! Raw bindings to SuiteSparse:GraphBLAS.
//!
//! Everything here is bindgen output (`bindings.rs`) and carries the C API's
//! contract unchanged: every function is `unsafe`, every built-in object is an
//! extern static the library defines, and nothing knows about ownership or
//! element types.
//!
//! The library is linked by `build.rs` from the archive that `native-deps`
//! builds for the submodule pin, so the bindings and the binary always come
//! from the same GraphBLAS release.

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
    reason = "bindgen output: it mirrors the C header and is not edited"
)]
mod generated {
    include!("bindings.rs");
}

pub use generated::*;
