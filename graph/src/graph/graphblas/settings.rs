//! Typed values of GraphBLAS storage settings, as read and written by
//! [`Matrix`](super::matrix::Matrix)'s storage setters and getters.
//!
//! Each enum is declared once by [`graphblas_setting!`] from a list of
//! `Variant = CONSTANT` entries: the discriminants are the bindings' own
//! constants, so `setting as i32` is the value GraphBLAS takes, and
//! `TryFrom<i32>` is the checked way back from the value GraphBLAS reports.

use thiserror::Error;

use super::{GrB_Orientation, GxB_BITMAP, GxB_FULL, GxB_HYPERSPARSE, GxB_SPARSE};

/// Declare a `#[repr(i32)]` enum whose discriminants are GraphBLAS constants,
/// with `impl TryFrom<i32>` refusing any other value with `$error`, a
/// generated error type whose message is `$message` (`{0}` is the value).
///
/// The values need not be consecutive (the sparsity formats are bit flags),
/// so `TryFrom` is a checked match on the discriminants, never a transmute.
macro_rules! graphblas_setting {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $( $(#[$vmeta:meta])* $variant:ident = $value:expr ),+ $(,)?
        }
        $(#[$emeta:meta])*
        error $error:ident = $message:literal;
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(i32)]
        pub enum $name {
            $( $(#[$vmeta])* $variant = ($value) as i32, )+
        }

        $(#[$emeta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
        #[error($message)]
        pub struct $error(pub i32);

        impl TryFrom<i32> for $name {
            type Error = $error;

            fn try_from(value: i32) -> Result<Self, $error> {
                $(
                    if value == Self::$variant as i32 {
                        return Ok(Self::$variant);
                    }
                )+
                Err($error(value))
            }
        }

        #[cfg(test)]
        impl $name {
            /// Every variant, in declaration order.
            const ALL: &[Self] = &[$(Self::$variant),+];
        }
    };
}

graphblas_setting! {
    /// A matrix's storage format: the value of `GxB_SPARSITY_STATUS`, and what
    /// [`Matrix::set_sparsity`](super::matrix::Matrix::set_sparsity) pins
    /// `GxB_SPARSITY_CONTROL` to.
    pub enum Sparsity {
        Hypersparse = GxB_HYPERSPARSE,
        Sparse = GxB_SPARSE,
        Bitmap = GxB_BITMAP,
        Full = GxB_FULL,
    }
    /// `GxB_SPARSITY_STATUS` reported a value that is not one format.
    error UnknownSparsity = "GxB_SPARSITY_STATUS returned {0}";
}

graphblas_setting! {
    /// Whether a matrix is stored by row or by column (`GrB_STORAGE_ORIENTATION_HINT`).
    pub enum Orientation {
        RowMajor = GrB_Orientation::GrB_ROWMAJOR,
        ColMajor = GrB_Orientation::GrB_COLMAJOR,
    }
    /// `GrB_STORAGE_ORIENTATION_HINT` reported neither orientation.
    error UnknownOrientation = "GrB_STORAGE_ORIENTATION_HINT returned {0}";
}

graphblas_setting! {
    /// Width of one of a matrix's integer arrays: its offsets (`p`), row indices
    /// or column indices. The value is the width in bits, as GraphBLAS's
    /// `*_INTEGER_HINT` and `*_INTEGER_BITS` fields take and report it.
    pub enum IntegerWidth {
        Bits32 = 32,
        Bits64 = 64,
    }
    /// GraphBLAS reported an integer array width other than 32 or 64 bits.
    error UnknownIntegerWidth = "GraphBLAS reported a {0}-bit integer array";
}

impl IntegerWidth {
    /// The width of the integer type `T` (`u32` or `u64`, the types the CSR
    /// export hands arrays out as).
    pub(super) const fn of<T: Into<u64>>() -> Self {
        match size_of::<T>() {
            4 => Self::Bits32,
            8 => Self::Bits64,
            _ => panic!("a CSR array is either 32 or 64 bits wide"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GrB_Orientation, GxB_BITMAP, GxB_FULL, GxB_HYPERSPARSE, GxB_SPARSE, IntegerWidth,
        Orientation, Sparsity, UnknownIntegerWidth, UnknownOrientation, UnknownSparsity,
    };
    use crate::graph::graphblas::matrix::Matrix;
    use crate::graph::graphblas::test_init;

    #[test]
    fn every_variant_round_trips_through_its_value() {
        for &s in Sparsity::ALL {
            assert_eq!(Sparsity::try_from(s as i32), Ok(s));
        }
        for &o in Orientation::ALL {
            assert_eq!(Orientation::try_from(o as i32), Ok(o));
        }
        for &w in IntegerWidth::ALL {
            assert_eq!(IntegerWidth::try_from(w as i32), Ok(w));
        }
        assert_eq!(Sparsity::ALL.len(), 4);
        assert_eq!(Orientation::ALL.len(), 2);
        assert_eq!(IntegerWidth::ALL.len(), 2);
    }

    #[test]
    fn the_discriminants_are_the_binding_constants() {
        assert_eq!(Sparsity::Hypersparse as i32, GxB_HYPERSPARSE as i32);
        assert_eq!(Sparsity::Sparse as i32, GxB_SPARSE as i32);
        assert_eq!(Sparsity::Bitmap as i32, GxB_BITMAP as i32);
        assert_eq!(Sparsity::Full as i32, GxB_FULL as i32);
        assert_eq!(
            Orientation::RowMajor as i32,
            GrB_Orientation::GrB_ROWMAJOR as i32
        );
        assert_eq!(
            Orientation::ColMajor as i32,
            GrB_Orientation::GrB_COLMAJOR as i32
        );
        assert_eq!(IntegerWidth::Bits32 as i32, 32);
        assert_eq!(IntegerWidth::Bits64 as i32, 64);
    }

    #[test]
    fn an_unknown_value_is_refused() {
        // The sparsity values are bit flags: neither 0, a combination (3) nor
        // the auto-select mask (15) is one format.
        for v in [0, 3, 15, -1] {
            assert_eq!(Sparsity::try_from(v), Err(UnknownSparsity(v)));
        }
        assert_eq!(
            UnknownSparsity(3).to_string(),
            "GxB_SPARSITY_STATUS returned 3"
        );
        assert_eq!(Orientation::try_from(7), Err(UnknownOrientation(7)));
        assert_eq!(
            UnknownOrientation(7).to_string(),
            "GrB_STORAGE_ORIENTATION_HINT returned 7"
        );
        for v in [0, 16, 63] {
            assert_eq!(IntegerWidth::try_from(v), Err(UnknownIntegerWidth(v)));
        }
        assert_eq!(
            UnknownIntegerWidth(16).to_string(),
            "GraphBLAS reported a 16-bit integer array"
        );
    }

    #[test]
    fn of_is_the_width_of_the_type() {
        assert_eq!(IntegerWidth::of::<u32>(), IntegerWidth::Bits32);
        assert_eq!(IntegerWidth::of::<u64>(), IntegerWidth::Bits64);
    }

    /// Every storage setter shows through its getter, and the entries survive
    /// the conversions. A 32-bit index hint that cannot fit (more than `2^31`
    /// columns) is accepted and ignored, which is why the widths have getters.
    #[test]
    fn storage_settings_round_trip() {
        test_init::ensure_init();
        let mut m = Matrix::<f64>::new(4, 4);
        m.set(0, 1, 2.0);
        m.set(2, 3, 2.0);
        m.wait();

        m.set_sparsity(Sparsity::Sparse);
        assert_eq!(m.sparsity(), Sparsity::Sparse);
        m.set_orientation(Orientation::ColMajor);
        assert_eq!(m.orientation(), Orientation::ColMajor);
        m.set_orientation(Orientation::RowMajor);
        assert_eq!(m.orientation(), Orientation::RowMajor);
        for width in [IntegerWidth::Bits64, IntegerWidth::Bits32] {
            m.set_offset_width(width);
            assert_eq!(m.offset_width(), width);
            m.set_col_index_width(width);
            assert_eq!(m.col_index_width(), width);
            m.set_row_index_width(width);
            assert_eq!(m.row_index_width(), width);
        }
        assert!(m.is_iso(), "two equal values are stored iso");
        m.set_iso(false);
        assert!(!m.is_iso());
        assert_eq!(m.get(0, 1), Some(2.0));
        assert_eq!(m.get(2, 3), Some(2.0));
        assert_eq!(m.nvals(), 2);

        let mut wide = Matrix::<bool>::new(2, (1 << 31) + 1);
        wide.set_col_index_width(IntegerWidth::Bits32);
        assert_eq!(wide.col_index_width(), IntegerWidth::Bits64);
    }
}
