//! Input operands: a borrowed handle, or the marker [`Output`] that says
//! "the output object here".
//!
//! `&mut c` with `&a` cannot alias, so `C = C * B` is not expressible with
//! references. The C API allows the alias and handles it per operation
//! (`apply` rewrites values in place; `mxm`, `transpose`, `select` and the
//! element-wise operations compute a temporary and transplant it). `Output`
//! names the alias in the input slot, and [`Operand::Elem`] ties the aliased
//! input's element type to the output's at compile time.

use crate::{
    matrix::Matrix,
    sealed::{Handle, Token},
    sys::{GrB_Matrix, GrB_Vector},
    types::GrbType,
    vector::Vector,
};

/// "Use the output object in this slot." Accepted wherever an input matrix
/// or vector, or a mask, is expected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Output;

/// An input operand of an operation whose output has element type `C` and
/// raw handle kind `R` (`GrB_Matrix` or `GrB_Vector`).
///
/// Implemented by `&Matrix<T>` / `&Vector<T>` (`Elem = T`) and by
/// [`Output`] (`Elem = C`). Sealed: an implementation hands a raw handle to
/// C.
pub trait Operand<'a, C, R>: Handle {
    /// The operand's element type.
    type Elem: GrbType;

    /// The raw handle to pass, given the output's.
    #[doc(hidden)]
    fn raw(
        self,
        output: R,
        _: Token,
    ) -> R;
}

impl<T> Handle for &Matrix<T> {}
impl<T> Handle for &Vector<T> {}
impl Handle for Output {}

impl<'a, T: GrbType, C> Operand<'a, C, GrB_Matrix> for &'a Matrix<T> {
    type Elem = T;

    fn raw(
        self,
        _output: GrB_Matrix,
        _: Token,
    ) -> GrB_Matrix {
        self.as_raw()
    }
}

impl<'a, T: GrbType, C> Operand<'a, C, GrB_Vector> for &'a Vector<T> {
    type Elem = T;

    fn raw(
        self,
        _output: GrB_Vector,
        _: Token,
    ) -> GrB_Vector {
        self.as_raw()
    }
}

impl<C: GrbType, R> Operand<'_, C, R> for Output {
    type Elem = C;

    fn raw(
        self,
        output: R,
        _: Token,
    ) -> R {
        output
    }
}
