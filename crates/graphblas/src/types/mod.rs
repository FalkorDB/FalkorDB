//! Element types: the Rust types a `Matrix<T>`,
//! `Vector<T>` or `Scalar<T>` can carry,
//! and how each reaches the type-suffixed C entry points
//! (`GrB_Matrix_setElement_BOOL`, `..._UDT`, ...).
//!
//! # The casting rule
//!
//! GraphBLAS casts silently among its built-in types and never for
//! user-defined ones. This crate behaves like the latter everywhere: an
//! operator is applied only to the exact types it was declared for, checked
//! by [`Accepts`]. The two named exceptions are [`Pattern`] (the operator
//! never reads the value) and `matrix::cast` /
//! `vector::cast` (change the type on purpose).

use std::convert::Infallible;

use crate::sealed;
use crate::sys::{
    GrB_BinaryOp, GrB_Descriptor, GrB_Index, GrB_IndexUnaryOp, GrB_Info, GrB_Matrix, GrB_Monoid,
    GrB_Scalar, GrB_Type, GrB_Vector,
};

/// Marker for an operator input that is never read. A label only: it has
/// no values.
///
/// `PAIR`, `ONEB` and the positional operators look only at the sparsity
/// pattern. A built-in typed with `Pattern` in an input position accepts a
/// matrix of any element type there; see [`Accepts`].
#[derive(Debug, Clone, Copy)]
pub struct Pattern(Infallible);

/// Marker for an operator input that is never read but must still be of a
/// built-in type.
///
/// The unary `ONE` ignores its input, yet GraphBLAS checks that the
/// operand's type can be cast to the operator's nominal input type, which
/// no user-defined type can. So it accepts every [`BuiltinType`] and
/// nothing else.
#[derive(Debug, Clone, Copy)]
pub struct BuiltinPattern(Infallible);

/// `X: Accepts<A>` says an operator whose input domain is `X` may be applied
/// to a matrix of element type `A`.
///
/// Exactly three cases hold: `A: Accepts<A>` (the domains match),
/// `Pattern: Accepts<A>` for every `A`, and `BuiltinPattern: Accepts<A>` for
/// every built-in `A`. There is deliberately no case for GraphBLAS's
/// implicit typecasting.
pub trait Accepts<T> {}

/// A GraphBLAS built-in type, as opposed to a user-defined one: the only
/// types `cast` converts between.
pub trait BuiltinType: GrbType {}

/// A Rust type with a GraphBLAS type behind it.
///
/// Implemented for the built-ins by this module and for user-defined
/// `#[repr(C)]` types by [`user_type!`](crate::user_type). The
/// `#[doc(hidden)]` methods forward to the type-suffixed C functions; the
/// safe API is their only caller.
///
/// # Safety
///
/// The safe API trusts an implementation for the size and layout of what C
/// reads and writes: [`grb_type`](Self::grb_type) must be a type whose size
/// is `size_of::<Self>()`, and every forwarding method must call the C
/// function for that type. Implement it through [`user_type!`](crate::user_type),
/// never by hand.
#[allow(clippy::too_many_arguments)] // mirrors C signatures of up to nine arguments
pub unsafe trait GrbType: Copy + Send + Sync + 'static + sealed::Sealed {
    /// The GraphBLAS name (`"GrB_BOOL"`, or the Rust type name for a
    /// user-defined type).
    const NAME: &'static str;

    /// The type handle. For a user-defined type, created on first use.
    /// Panics before [`init`](crate::init).
    fn grb_type() -> GrB_Type;

    #[doc(hidden)]
    unsafe fn matrix_set_element(
        c: GrB_Matrix,
        x: Self,
        i: GrB_Index,
        j: GrB_Index,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn matrix_extract_element(
        x: *mut Self,
        a: GrB_Matrix,
        i: GrB_Index,
        j: GrB_Index,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn matrix_build(
        c: GrB_Matrix,
        rows: *const GrB_Index,
        cols: *const GrB_Index,
        x: *const Self,
        n: GrB_Index,
        dup: GrB_BinaryOp,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn matrix_extract_tuples(
        rows: *mut GrB_Index,
        cols: *mut GrB_Index,
        x: *mut Self,
        n: *mut GrB_Index,
        a: GrB_Matrix,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn matrix_select(
        c: GrB_Matrix,
        mask: GrB_Matrix,
        accum: GrB_BinaryOp,
        op: GrB_IndexUnaryOp,
        a: GrB_Matrix,
        y: Self,
        desc: GrB_Descriptor,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn matrix_apply_index(
        c: GrB_Matrix,
        mask: GrB_Matrix,
        accum: GrB_BinaryOp,
        op: GrB_IndexUnaryOp,
        a: GrB_Matrix,
        y: Self,
        desc: GrB_Descriptor,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn matrix_assign_scalar(
        c: GrB_Matrix,
        mask: GrB_Matrix,
        accum: GrB_BinaryOp,
        x: Self,
        rows: *const GrB_Index,
        nrows: GrB_Index,
        cols: *const GrB_Index,
        ncols: GrB_Index,
        desc: GrB_Descriptor,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn matrix_reduce(
        x: *mut Self,
        accum: GrB_BinaryOp,
        monoid: GrB_Monoid,
        a: GrB_Matrix,
        desc: GrB_Descriptor,
    ) -> GrB_Info;

    #[doc(hidden)]
    unsafe fn vector_set_element(
        w: GrB_Vector,
        x: Self,
        i: GrB_Index,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn vector_extract_element(
        x: *mut Self,
        u: GrB_Vector,
        i: GrB_Index,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn vector_build(
        w: GrB_Vector,
        idx: *const GrB_Index,
        x: *const Self,
        n: GrB_Index,
        dup: GrB_BinaryOp,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn vector_extract_tuples(
        idx: *mut GrB_Index,
        x: *mut Self,
        n: *mut GrB_Index,
        u: GrB_Vector,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn vector_select(
        w: GrB_Vector,
        mask: GrB_Vector,
        accum: GrB_BinaryOp,
        op: GrB_IndexUnaryOp,
        u: GrB_Vector,
        y: Self,
        desc: GrB_Descriptor,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn vector_apply_index(
        w: GrB_Vector,
        mask: GrB_Vector,
        accum: GrB_BinaryOp,
        op: GrB_IndexUnaryOp,
        u: GrB_Vector,
        y: Self,
        desc: GrB_Descriptor,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn vector_assign_scalar(
        w: GrB_Vector,
        mask: GrB_Vector,
        accum: GrB_BinaryOp,
        x: Self,
        idx: *const GrB_Index,
        n: GrB_Index,
        desc: GrB_Descriptor,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn vector_reduce(
        x: *mut Self,
        accum: GrB_BinaryOp,
        monoid: GrB_Monoid,
        u: GrB_Vector,
        desc: GrB_Descriptor,
    ) -> GrB_Info;

    #[doc(hidden)]
    unsafe fn scalar_set_element(
        s: GrB_Scalar,
        x: Self,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn scalar_extract_element(
        x: *mut Self,
        s: GrB_Scalar,
    ) -> GrB_Info;
    #[doc(hidden)]
    unsafe fn monoid_new(
        m: *mut GrB_Monoid,
        op: GrB_BinaryOp,
        identity: Self,
    ) -> GrB_Info;
}

mod builtins;
pub mod user_defined;
