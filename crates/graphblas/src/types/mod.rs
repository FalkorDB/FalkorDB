//! Element types: the Rust types a [`Matrix<T>`](crate::Matrix),
//! [`Vector<T>`](crate::Vector) or [`Scalar<T>`](crate::Scalar) can carry,
//! and how each reaches the type-suffixed C entry points
//! (`GrB_Matrix_setElement_BOOL`, `..._UDT`, ...).
//!
//! # The casting rule
//!
//! GraphBLAS casts silently among its built-in types and never for
//! user-defined ones. This crate never casts silently: an operator is
//! applied only to the exact types it was declared for, checked by
//! [`Accepts`] on its inputs and [`Produces`] on its output. Three things
//! are spelled instead:
//!
//! * [`Pattern`]: the operator never reads the value, so any type will do.
//! * `op.typecast()`: let GraphBLAS cast at the edges of *this* operator, in
//!   this call. No work is done and the handle is the same; the operator's
//!   type records the casts with [`Typecast`] labels.
//! * [`matrix::cast`](crate::matrix::cast) /
//!   [`vector::cast`](crate::vector::cast): store a converted copy.

use std::{convert::Infallible, marker::PhantomData};

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
/// For an operator as declared, three cases hold: `A: Accepts<A>` (the
/// domains match), `Pattern: Accepts<A>` for every `A`, and
/// `BuiltinPattern: Accepts<A>` for every built-in `A`. A fourth is reached
/// only through `op.typecast()`: `Typecast<X, A>: Accepts<A>` (and
/// [`MonoidTypecast`] likewise) when GraphBLAS can cast an `A` to `X`.
/// There is deliberately no case for a cast nobody asked for.
pub trait Accepts<T> {}

/// `Z: Produces<C>` says an operator whose output domain is `Z` may write
/// into a matrix, vector or scalar of element type `C`.
///
/// For an operator as declared, `C: Produces<C>`: the domains match.
/// Through `op.typecast()`, `Typecast<Z, C>: Produces<C>` (and
/// [`MonoidTypecast`] likewise) when GraphBLAS can cast a `Z` to `C`.
///
/// Closed to this crate. Reducing into a user-defined scalar goes through
/// an entry point that cannot see the scalar's type and writes the
/// monoid's, so these cases are what keeps that write in bounds.
#[diagnostic::on_unimplemented(
    message = "an operator whose output is `{Self}` does not write `{C}` elements",
    label = "the output's elements are `{C}`, the operator produces `{Self}`",
    note = "GraphBLAS would cast one built-in type to another silently; write `.typecast()` on the operator to allow it in this call"
)]
pub trait Produces<C>: sealed::ProducesRule<C> {}
impl<Z: sealed::ProducesRule<C>, C> Produces<C> for Z {}

impl<C: GrbType> sealed::ProducesRule<C> for C {}

/// `X: PassesThrough<E, Z>` says a set union (`ewise_add`) under an
/// operator with input domain `X` and output domain `Z` may take an input
/// of element type `E`.
///
/// A union copies the entries present on one side only, and GraphBLAS
/// gives them the operator's output type on the way, without the operator
/// being called. So for an operator as declared the operand's type is the
/// output's: `E: PassesThrough<E, E>`, and `Pattern` likewise.
///
/// Through `op.typecast()` the case is
/// `Typecast<D, E>: PassesThrough<E, Typecast<D, C>>`: the operator reads
/// and writes one declared type `D`, and GraphBLAS can cast an `E` to it.
/// That is narrower than GraphBLAS, on purpose. When exactly one operand
/// is transposed, by the descriptor or because it is stored in the other
/// orientation than the output, SuiteSparse first casts it to the
/// operator's *input* type (`GB_ewise.c`), so its lone entries go
/// `E -> X -> Z` instead of `E -> Z`. With a comparison (`u64` in, `bool`
/// out) a lone `0.5` would land as `true` or as `false` depending on how
/// the matrix is stored. With `X = Z` the two paths are one.
#[diagnostic::on_unimplemented(
    message = "a union under this operator cannot copy `{E}` entries through",
    label = "the entries only this operand has would pass from `{E}` through `{Self}` to `{Z}`",
    note = "as declared, a union's operands and output share one element type; `op.typecast()` lifts that only for an operator that reads and writes one type"
)]
pub trait PassesThrough<E, Z>: sealed::PassesThroughRule<E, Z> {}
impl<X: sealed::PassesThroughRule<E, Z>, E, Z> PassesThrough<E, Z> for X {}

impl<T: GrbType> sealed::PassesThroughRule<T, T> for T {}
impl<T: GrbType> sealed::PassesThroughRule<T, T> for Pattern {}
impl<T: BuiltinType> sealed::PassesThroughRule<T, T> for BuiltinPattern {}

/// `D: TypecastFrom<S>` says GraphBLAS takes an `S` where a `D` is
/// declared: its typecast rule, as a bound.
///
/// The rule (`GB_Type_compatible`) is short. Any built-in type casts to any
/// other. A user-defined type casts to nothing but itself. An input the
/// operator never reads takes anything ([`Pattern`]), or anything built-in
/// ([`BuiltinPattern`]). Since it depends on the types alone,
/// `op.typecast()` is checked when the program is compiled and cannot fail
/// when it runs. Closed to this crate.
#[diagnostic::on_unimplemented(
    message = "GraphBLAS cannot cast `{Source}` to `{Self}`",
    note = "only built-in types cast to one another; a user-defined type is used as itself"
)]
pub trait TypecastFrom<Source>: sealed::Casts<Source> {}
impl<D: sealed::Casts<S>, S> TypecastFrom<S> for D {}

// Two types cast when they are of one kind: every built-in is of the kind
// `BuiltinKind`, and a user-defined type is its own.
impl<D: GrbType, S: GrbType<Kind = D::Kind>> sealed::Casts<S> for D {}
impl<S: GrbType> sealed::Casts<S> for Pattern {}
impl<S: BuiltinType> sealed::Casts<S> for BuiltinPattern {}

/// [`GrbType::Kind`] of every built-in type. Not part of the API.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct BuiltinKind(Infallible);

/// A label in an operator's type: "declared for `Declared`, used here on
/// `Seen`, and GraphBLAS casts between the two".
///
/// `op.typecast()` returns the same handle with every domain relabelled
/// this way, which is what lets an operation take operands the operator was
/// not declared for:
///
/// ```ignore
/// // A semiring over f64 on u64 matrices, into a u64 matrix. The three
/// // domains are inferred from the operands.
/// mxm(&mut c, Mask::NONE, None, &PLUS_TIMES_SEMIRING_FP64.typecast(), &a, &b, Desc::NONE)?;
/// // The same, with the domains written out for the reader.
/// let semiring = PLUS_TIMES_SEMIRING_FP64.typecast::<u64, u64, u64>();
/// // Row indices (`i64` in GraphBLAS) into a `u64` vector.
/// vxm(&mut ids, Mask::NONE, None, &PLUS_FIRSTJ_INT64.typecast(), &x, &a, Desc::NONE)?;
/// ```
///
/// # What the cast is
///
/// The operator itself does not change. GraphBLAS converts each input
/// entry to the declared input type, runs the operator, and converts the
/// result once to the output's type. For a semiring that is not a semiring
/// over the new types: the monoid still adds in the declared type and the
/// sum is cast once at the end, so `PLUS_TIMES` over `f64` into a `bool`
/// matrix gives `false` where the products sum to `0.0`.
///
/// The conversions are C's, with two exceptions that make them total:
/// a float becomes an integer by saturating at the integer's range, and
/// NaN becomes `0`. So number to number is Rust's `as`. Anything becomes a
/// `bool` as `x != 0`, and a `bool` becomes `0` or `1`.
///
/// # What it costs
///
/// No copy is made, unlike [`matrix::cast`](crate::matrix::cast). An *input*
/// cast does take a binary operator or a semiring off SuiteSparse's
/// built-in kernels, which exist only for operands of the operator's own
/// types (`GB_binop_builtin`): the call then needs a `PreJIT` or JIT
/// kernel, or runs the generic one. So a `.typecast()` in the engine is a
/// kernel to harvest. Two cases lose nothing: an input the operator does
/// not read, and the `f32` / `f64` operators, whose built-in kernels the
/// engine's build of GraphBLAS leaves out (`GB_control.patch`).
///
/// # What is refused
///
/// A relabelled handle is for passing to an operation. It is not an
/// operator over the new types, so the constructors do not take it:
///
/// ```compile_fail,E0277
/// use graphblas::*;
/// let plus = PLUS_UINT64.typecast::<bool, bool, bool>();
/// let _ = Monoid::new(&plus, false);
/// ```
///
/// User-defined types never cast, so this is refused as GraphBLAS would
/// refuse it, but when the program is compiled:
///
/// ```compile_fail,E0277
/// use graphblas::*;
/// #[repr(C)]
/// #[derive(Clone, Copy)]
/// struct Weight { value: f64 }
/// user_type!(Weight);
/// let _ = PLUS_FP64.typecast::<Weight, f64, f64>();
/// ```
///
/// And domains written out must be the operands':
///
/// ```compile_fail,E0271
/// use graphblas::*;
/// let a = Matrix::<u64>::new(2, 2).unwrap();
/// let mut c = Matrix::<u64>::new(2, 2).unwrap();
/// let semiring = PLUS_TIMES_SEMIRING_FP64.typecast::<bool, bool, u64>();
/// mxm(&mut c, Mask::NONE, None, &semiring, &a, &a, Desc::NONE).unwrap();
/// ```
///
/// A union takes a relabelled operator only if it reads and writes one
/// declared type, because GraphBLAS's answer otherwise depends on how the
/// operands are stored (see [`PassesThrough`]):
///
/// ```compile_fail,E0277
/// use graphblas::*;
/// let a = Matrix::<f64>::new(2, 2).unwrap();
/// let b = Matrix::<bool>::new(2, 2).unwrap();
/// let mut c = Matrix::<u64>::new(2, 2).unwrap();
/// let above = GT_UINT64.typecast();
/// matrix::ewise_add(&mut c, Mask::NONE, None, &above, &a, &b, Desc::NONE).unwrap();
/// ```
///
/// The accumulator is not relabelled: it stays `BinaryOp<C, C, C>` on the
/// output's type, so with one, the operator's result is cast to `C` and
/// accumulated there. The scalar of an [`IndexUnaryOp`](crate::IndexUnaryOp)
/// is a Rust value of the declared type, converted by the caller.
#[derive(Debug, Clone, Copy)]
pub struct Typecast<Declared, Seen> {
    _never: Infallible,
    _domains: PhantomData<(Declared, Seen)>,
}

impl<X: TypecastFrom<A>, A> Accepts<A> for Typecast<X, A> {}
impl<Z, C: GrbType + TypecastFrom<Z>> sealed::ProducesRule<C> for Typecast<Z, C> {}
impl<D: TypecastFrom<E>, E, C> sealed::PassesThroughRule<E, Typecast<D, C>> for Typecast<D, E> {}

/// [`Typecast`] for a monoid, whose one domain is both what it reads and
/// what it produces: "declared for `Declared`, folding `Input` entries
/// into an `Output`".
///
/// ```ignore
/// // How many `true` entries: the bools are summed as u64.
/// let mut count = 0u64;
/// matrix::reduce(&mut count, None, &PLUS_MONOID_UINT64.typecast(), &flags, Desc::NONE)?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MonoidTypecast<Declared, Input, Output> {
    _never: Infallible,
    _domains: PhantomData<(Declared, Input, Output)>,
}

impl<T: TypecastFrom<A>, A, C> Accepts<A> for MonoidTypecast<T, A, C> {}
impl<T, A, C: GrbType + TypecastFrom<T>> sealed::ProducesRule<C> for MonoidTypecast<T, A, C> {}

/// A GraphBLAS built-in type, as opposed to a user-defined one: the only
/// types GraphBLAS converts between, in `op.typecast()` and in
/// [`cast`](crate::matrix::cast).
pub trait BuiltinType: GrbType<Kind = BuiltinKind> {}

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

    /// What GraphBLAS casts this type to and from: [`BuiltinKind`] for
    /// every built-in type, and the type itself for a user-defined one.
    #[doc(hidden)]
    type Kind;

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
