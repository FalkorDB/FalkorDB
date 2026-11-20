//! Typed values of GraphBLAS storage settings, read and written through the
//! handles' getters and setters (`GrB_get` / `GrB_set` with the `GxB_*`
//! fields). Never cast from an integer: the setters pass the discriminant,
//! the getters match the value GraphBLAS reports and panic on one they do
//! not know.

use crate::sys;

/// `GxB_SPARSITY_CONTROL` / `GxB_SPARSITY_STATUS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sparsity {
    Hypersparse,
    Sparse,
    Bitmap,
    Full,
    /// Let GraphBLAS choose among all four (control only).
    Auto,
}

/// `GxB_FORMAT`: `GxB_BY_ROW` (CSR) or `GxB_BY_COL` (CSC).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    ByRow,
    ByCol,
}

/// `GxB_ROWINDEX_INTEGER_HINT` and friends: 32- or 64-bit internal indices.
/// A hint on write; the getter reports what GraphBLAS actually uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerWidth {
    Bits32,
    Bits64,
}

impl Sparsity {
    /// The `GxB_SPARSITY_CONTROL` value.
    #[allow(
        clippy::cast_possible_wrap,
        reason = "the constants are below 16, and `try_from` is not usable in a `const fn`"
    )]
    pub(crate) const fn raw(self) -> i32 {
        match self {
            Self::Hypersparse => sys::GxB_HYPERSPARSE as i32,
            Self::Sparse => sys::GxB_SPARSE as i32,
            Self::Bitmap => sys::GxB_BITMAP as i32,
            Self::Full => sys::GxB_FULL as i32,
            Self::Auto => sys::GxB_AUTO_SPARSITY as i32,
        }
    }

    /// A `GxB_SPARSITY_STATUS` value.
    pub(crate) fn from_status(status: i32) -> Self {
        [Self::Hypersparse, Self::Sparse, Self::Bitmap, Self::Full]
            .into_iter()
            .find(|sparsity| sparsity.raw() == status)
            .unwrap_or_else(|| panic!("GraphBLAS reported an unknown sparsity status {status}"))
    }
}

impl Orientation {
    /// The `GxB_FORMAT` value.
    pub(crate) const fn raw(self) -> i32 {
        match self {
            Self::ByRow => sys::GxB_Format_Value::GxB_BY_ROW as i32,
            Self::ByCol => sys::GxB_Format_Value::GxB_BY_COL as i32,
        }
    }

    pub(crate) fn from_raw(format: i32) -> Self {
        match format {
            x if x == sys::GxB_Format_Value::GxB_BY_ROW as i32 => Self::ByRow,
            x if x == sys::GxB_Format_Value::GxB_BY_COL as i32 => Self::ByCol,
            other => panic!("GraphBLAS reported an unknown format {other}"),
        }
    }
}

impl IntegerWidth {
    /// The hint value: the number of bits.
    pub(crate) const fn raw(self) -> i32 {
        match self {
            Self::Bits32 => 32,
            Self::Bits64 => 64,
        }
    }

    pub(crate) fn from_bits(bits: i32) -> Self {
        match bits {
            32 => Self::Bits32,
            64 => Self::Bits64,
            other => panic!("GraphBLAS reported an unknown integer width {other}"),
        }
    }
}
