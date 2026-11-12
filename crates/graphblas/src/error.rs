//! The error a GraphBLAS call reports, and the `check` helpers that turn a
//! `GrB_Info` into a [`Result`].

use std::fmt;

use crate::sys::GrB_Info;

/// Why a GraphBLAS call failed: the C API's `GrB_Info`, as a Rust enum.
///
/// [`Other`](Self::Other) keeps a status that is not an error code of
/// GraphBLAS: `GrB_NO_VALUE` where a value was required, or one of
/// LAGraph's own codes, which the `lagraph` crate passes through here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// `GrB_UNINITIALIZED_OBJECT`: an object was used before it was created.
    UninitializedObject,
    /// `GrB_NULL_POINTER`: a required pointer was null.
    NullPointer,
    /// `GrB_INVALID_VALUE`: an argument has a value the call does not accept.
    InvalidValue,
    /// `GrB_INVALID_INDEX`: a row or column index is out of range.
    InvalidIndex,
    /// `GrB_DOMAIN_MISMATCH`: the element types of the arguments do not fit together.
    DomainMismatch,
    /// `GrB_DIMENSION_MISMATCH`: the dimensions of the arguments do not fit together.
    DimensionMismatch,
    /// `GrB_OUTPUT_NOT_EMPTY`: the output must be empty and is not.
    OutputNotEmpty,
    /// `GrB_NOT_IMPLEMENTED`: the library does not implement this case.
    NotImplemented,
    /// `GrB_ALREADY_SET`: a write-once field was already set.
    AlreadySet,
    /// `GrB_PANIC`: the library is not initialised, or failed beyond recovery.
    Panic,
    /// `GrB_OUT_OF_MEMORY`: an allocation failed.
    OutOfMemory,
    /// `GrB_INSUFFICIENT_SPACE`: a caller-supplied buffer is too small.
    InsufficientSpace,
    /// `GrB_INVALID_OBJECT`: an object is corrupt, or was cleared by a failed call.
    InvalidObject,
    /// `GrB_INDEX_OUT_OF_BOUNDS`: an index found while running is out of bounds.
    IndexOutOfBounds,
    /// `GrB_EMPTY_OBJECT`: a scalar that must hold a value is empty.
    EmptyObject,
    /// `GxB_JIT_ERROR`: the JIT failed to compile or load a kernel.
    JitError,
    /// `GxB_GPU_ERROR`: a GPU kernel failed.
    GpuError,
    /// `GxB_OUTPUT_IS_READONLY`: the output object is read-only.
    OutputIsReadonly,
    /// A status this crate has no name for, as the library returned it.
    Other(i32),
}

impl ErrorKind {
    /// The kind for a status code of the C API.
    #[must_use]
    pub const fn from_code(code: i32) -> Self {
        match code {
            c if c == GrB_Info::GrB_UNINITIALIZED_OBJECT as i32 => Self::UninitializedObject,
            c if c == GrB_Info::GrB_NULL_POINTER as i32 => Self::NullPointer,
            c if c == GrB_Info::GrB_INVALID_VALUE as i32 => Self::InvalidValue,
            c if c == GrB_Info::GrB_INVALID_INDEX as i32 => Self::InvalidIndex,
            c if c == GrB_Info::GrB_DOMAIN_MISMATCH as i32 => Self::DomainMismatch,
            c if c == GrB_Info::GrB_DIMENSION_MISMATCH as i32 => Self::DimensionMismatch,
            c if c == GrB_Info::GrB_OUTPUT_NOT_EMPTY as i32 => Self::OutputNotEmpty,
            c if c == GrB_Info::GrB_NOT_IMPLEMENTED as i32 => Self::NotImplemented,
            c if c == GrB_Info::GrB_ALREADY_SET as i32 => Self::AlreadySet,
            c if c == GrB_Info::GrB_PANIC as i32 => Self::Panic,
            c if c == GrB_Info::GrB_OUT_OF_MEMORY as i32 => Self::OutOfMemory,
            c if c == GrB_Info::GrB_INSUFFICIENT_SPACE as i32 => Self::InsufficientSpace,
            c if c == GrB_Info::GrB_INVALID_OBJECT as i32 => Self::InvalidObject,
            c if c == GrB_Info::GrB_INDEX_OUT_OF_BOUNDS as i32 => Self::IndexOutOfBounds,
            c if c == GrB_Info::GrB_EMPTY_OBJECT as i32 => Self::EmptyObject,
            c if c == GrB_Info::GxB_JIT_ERROR as i32 => Self::JitError,
            c if c == GrB_Info::GxB_GPU_ERROR as i32 => Self::GpuError,
            c if c == GrB_Info::GxB_OUTPUT_IS_READONLY as i32 => Self::OutputIsReadonly,
            other => Self::Other(other),
        }
    }

    /// The C API's name for it, which is what the GraphBLAS documentation
    /// and logs use.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::UninitializedObject => "GrB_UNINITIALIZED_OBJECT",
            Self::NullPointer => "GrB_NULL_POINTER",
            Self::InvalidValue => "GrB_INVALID_VALUE",
            Self::InvalidIndex => "GrB_INVALID_INDEX",
            Self::DomainMismatch => "GrB_DOMAIN_MISMATCH",
            Self::DimensionMismatch => "GrB_DIMENSION_MISMATCH",
            Self::OutputNotEmpty => "GrB_OUTPUT_NOT_EMPTY",
            Self::NotImplemented => "GrB_NOT_IMPLEMENTED",
            Self::AlreadySet => "GrB_ALREADY_SET",
            Self::Panic => "GrB_PANIC",
            Self::OutOfMemory => "GrB_OUT_OF_MEMORY",
            Self::InsufficientSpace => "GrB_INSUFFICIENT_SPACE",
            Self::InvalidObject => "GrB_INVALID_OBJECT",
            Self::IndexOutOfBounds => "GrB_INDEX_OUT_OF_BOUNDS",
            Self::EmptyObject => "GrB_EMPTY_OBJECT",
            Self::JitError => "GxB_JIT_ERROR",
            Self::GpuError => "GxB_GPU_ERROR",
            Self::OutputIsReadonly => "GxB_OUTPUT_IS_READONLY",
            Self::Other(_) => "GrB_Info",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Other(code) => write!(f, "GrB_Info {code}"),
            known => f.write_str(known.name()),
        }
    }
}

/// A failed GraphBLAS call.
///
/// `kind` is the status the call returned. `message` is the text GraphBLAS
/// attached to the output object (`GrB_Matrix_error` / `GrB_Vector_error`),
/// copied out immediately because GraphBLAS keeps it only until the next call
/// on that object; it is empty when the call had no output object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
}

impl Error {
    /// The error for a status the C API returned, with a message.
    pub(crate) const fn from_info(
        info: GrB_Info,
        message: String,
    ) -> Self {
        Self {
            kind: ErrorKind::from_code(info as i32),
            message,
        }
    }
}

/// `Result` of a GraphBLAS call.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if self.message.is_empty() {
            write!(f, "GraphBLAS: {}", self.kind)
        } else {
            write!(f, "GraphBLAS: {}: {}", self.kind, self.message)
        }
    }
}

impl std::error::Error for Error {}

/// `Ok` on `GrB_SUCCESS`, else an [`Error`] with no message.
pub fn check(info: GrB_Info) -> Result<()> {
    if info == GrB_Info::GrB_SUCCESS {
        Ok(())
    } else {
        Err(Error::from_info(info, String::new()))
    }
}
