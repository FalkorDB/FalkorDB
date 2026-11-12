//! Raw bindings to SuiteSparse:GraphBLAS.
//!
//! Everything here is bindgen output (`bindings.rs`) and carries the C API's
//! contract unchanged: every function is `unsafe`, every built-in object is an
//! extern static the library defines, and nothing knows about ownership or
//! element types. The `graphblas` crate is the one consumer and the one place
//! these symbols are allowed to be called from.
//!
//! The library is linked by `build.rs` from the archive that `native-deps`
//! builds for the submodule pin, so the bindings and the binary always come
//! from the same GraphBLAS release.

use std::fmt;

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

/// Prints the constant's name, as the Rust enum this type used to be did:
/// `GrB_OUT_OF_MEMORY`, not `GrB_Info(-102)`. A status with no constant
/// prints as its number.
impl fmt::Debug for GrB_Info {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let name = match *self {
            Self::GrB_SUCCESS => "GrB_SUCCESS",
            Self::GrB_NO_VALUE => "GrB_NO_VALUE",
            Self::GxB_EXHAUSTED => "GxB_EXHAUSTED",
            Self::GrB_UNINITIALIZED_OBJECT => "GrB_UNINITIALIZED_OBJECT",
            Self::GrB_NULL_POINTER => "GrB_NULL_POINTER",
            Self::GrB_INVALID_VALUE => "GrB_INVALID_VALUE",
            Self::GrB_INVALID_INDEX => "GrB_INVALID_INDEX",
            Self::GrB_DOMAIN_MISMATCH => "GrB_DOMAIN_MISMATCH",
            Self::GrB_DIMENSION_MISMATCH => "GrB_DIMENSION_MISMATCH",
            Self::GrB_OUTPUT_NOT_EMPTY => "GrB_OUTPUT_NOT_EMPTY",
            Self::GrB_NOT_IMPLEMENTED => "GrB_NOT_IMPLEMENTED",
            Self::GrB_ALREADY_SET => "GrB_ALREADY_SET",
            Self::GrB_PANIC => "GrB_PANIC",
            Self::GrB_OUT_OF_MEMORY => "GrB_OUT_OF_MEMORY",
            Self::GrB_INSUFFICIENT_SPACE => "GrB_INSUFFICIENT_SPACE",
            Self::GrB_INVALID_OBJECT => "GrB_INVALID_OBJECT",
            Self::GrB_INDEX_OUT_OF_BOUNDS => "GrB_INDEX_OUT_OF_BOUNDS",
            Self::GrB_EMPTY_OBJECT => "GrB_EMPTY_OBJECT",
            Self::GxB_JIT_ERROR => "GxB_JIT_ERROR",
            Self::GxB_GPU_ERROR => "GxB_GPU_ERROR",
            Self::GxB_OUTPUT_IS_READONLY => "GxB_OUTPUT_IS_READONLY",
            Self(other) => return write!(f, "GrB_Info({other})"),
        };
        f.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::GrB_Info;

    #[test]
    fn a_status_prints_as_its_constant() {
        assert_eq!(format!("{:?}", GrB_Info::GrB_SUCCESS), "GrB_SUCCESS");
        assert_eq!(
            format!("{:?}", GrB_Info::GrB_OUT_OF_MEMORY),
            "GrB_OUT_OF_MEMORY"
        );
        assert_eq!(
            format!("{:?}", GrB_Info::GxB_OUTPUT_IS_READONLY),
            "GxB_OUTPUT_IS_READONLY"
        );
        // One the header does not list is still a value, and says so.
        assert_eq!(format!("{:?}", GrB_Info(42)), "GrB_Info(42)");
    }
}
