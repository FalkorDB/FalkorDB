//! The eleven built-in element types: one `builtin_type!` invocation per
//! type, naming the type-suffixed C entry point for each operation.

use super::{Accepts, BuiltinKind, BuiltinPattern, BuiltinType, GrbType, Pattern};
use crate::{
    sealed,
    sys::{
        GrB_BinaryOp, GrB_Descriptor, GrB_Index, GrB_IndexUnaryOp, GrB_Info, GrB_Matrix,
        GrB_Monoid, GrB_Scalar, GrB_Type, GrB_Vector,
    },
};

/// Implement [`GrbType`] for a Rust type backed by a GraphBLAS built-in
/// type, naming the type-suffixed C function for each operation.
macro_rules! builtin_type {
    (
        $t:ty, $grb:ident,
        matrix: $mset:ident, $mget:ident, $mbuild:ident, $mtuples:ident, $msel:ident, $mapply:ident, $massign:ident, $mred:ident;
        vector: $vset:ident, $vget:ident, $vbuild:ident, $vtuples:ident, $vsel:ident, $vapply:ident, $vassign:ident, $vred:ident;
        scalar: $sset:ident, $sget:ident; monoid: $monoid:ident;
    ) => {
        impl sealed::Sealed for $t {}
        impl BuiltinType for $t {}
        impl Accepts<$t> for $t {}
        impl Accepts<$t> for Pattern {}
        impl Accepts<$t> for BuiltinPattern {}

        // SAFETY: `$grb` is the built-in type of the same size and layout,
        // and every method forwards to its own type-suffixed entry point.
        unsafe impl GrbType for $t {
            type Kind = BuiltinKind;

            const NAME: &'static str = stringify!($grb);

            fn grb_type() -> GrB_Type {
                crate::init::assert_initialized(stringify!($grb));
                // SAFETY: a by-value read of a handle `GrB_init` set and
                // nothing writes afterwards.
                unsafe { crate::sys::$grb }
            }

            // Each body forwards its arguments unchanged to the C function;
            // the caller is responsible for the C contract.
            unsafe fn matrix_set_element(
                c: GrB_Matrix,
                x: Self,
                i: GrB_Index,
                j: GrB_Index,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$mset(c, x, i, j) }
            }
            unsafe fn matrix_extract_element(
                x: *mut Self,
                a: GrB_Matrix,
                i: GrB_Index,
                j: GrB_Index,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$mget(x, a, i, j) }
            }
            unsafe fn matrix_build(
                c: GrB_Matrix,
                rows: *const GrB_Index,
                cols: *const GrB_Index,
                x: *const Self,
                n: GrB_Index,
                dup: GrB_BinaryOp,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$mbuild(c, rows, cols, x, n, dup) }
            }
            unsafe fn matrix_extract_tuples(
                rows: *mut GrB_Index,
                cols: *mut GrB_Index,
                x: *mut Self,
                n: *mut GrB_Index,
                a: GrB_Matrix,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$mtuples(rows, cols, x, n, a) }
            }
            unsafe fn matrix_select(
                c: GrB_Matrix,
                mask: GrB_Matrix,
                accum: GrB_BinaryOp,
                op: GrB_IndexUnaryOp,
                a: GrB_Matrix,
                y: Self,
                desc: GrB_Descriptor,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$msel(c, mask, accum, op, a, y, desc) }
            }
            unsafe fn matrix_apply_index(
                c: GrB_Matrix,
                mask: GrB_Matrix,
                accum: GrB_BinaryOp,
                op: GrB_IndexUnaryOp,
                a: GrB_Matrix,
                y: Self,
                desc: GrB_Descriptor,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$mapply(c, mask, accum, op, a, y, desc) }
            }
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
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$massign(c, mask, accum, x, rows, nrows, cols, ncols, desc) }
            }
            unsafe fn matrix_reduce(
                x: *mut Self,
                accum: GrB_BinaryOp,
                monoid: GrB_Monoid,
                a: GrB_Matrix,
                desc: GrB_Descriptor,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$mred(x, accum, monoid, a, desc) }
            }

            unsafe fn vector_set_element(
                w: GrB_Vector,
                x: Self,
                i: GrB_Index,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$vset(w, x, i) }
            }
            unsafe fn vector_extract_element(
                x: *mut Self,
                u: GrB_Vector,
                i: GrB_Index,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$vget(x, u, i) }
            }
            unsafe fn vector_build(
                w: GrB_Vector,
                idx: *const GrB_Index,
                x: *const Self,
                n: GrB_Index,
                dup: GrB_BinaryOp,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$vbuild(w, idx, x, n, dup) }
            }
            unsafe fn vector_extract_tuples(
                idx: *mut GrB_Index,
                x: *mut Self,
                n: *mut GrB_Index,
                u: GrB_Vector,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$vtuples(idx, x, n, u) }
            }
            unsafe fn vector_select(
                w: GrB_Vector,
                mask: GrB_Vector,
                accum: GrB_BinaryOp,
                op: GrB_IndexUnaryOp,
                u: GrB_Vector,
                y: Self,
                desc: GrB_Descriptor,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$vsel(w, mask, accum, op, u, y, desc) }
            }
            unsafe fn vector_apply_index(
                w: GrB_Vector,
                mask: GrB_Vector,
                accum: GrB_BinaryOp,
                op: GrB_IndexUnaryOp,
                u: GrB_Vector,
                y: Self,
                desc: GrB_Descriptor,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$vapply(w, mask, accum, op, u, y, desc) }
            }
            unsafe fn vector_assign_scalar(
                w: GrB_Vector,
                mask: GrB_Vector,
                accum: GrB_BinaryOp,
                x: Self,
                idx: *const GrB_Index,
                n: GrB_Index,
                desc: GrB_Descriptor,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$vassign(w, mask, accum, x, idx, n, desc) }
            }
            unsafe fn vector_reduce(
                x: *mut Self,
                accum: GrB_BinaryOp,
                monoid: GrB_Monoid,
                u: GrB_Vector,
                desc: GrB_Descriptor,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$vred(x, accum, monoid, u, desc) }
            }

            unsafe fn scalar_set_element(
                s: GrB_Scalar,
                x: Self,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$sset(s, x) }
            }
            unsafe fn scalar_extract_element(
                x: *mut Self,
                s: GrB_Scalar,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$sget(x, s) }
            }
            unsafe fn monoid_new(
                m: *mut GrB_Monoid,
                op: GrB_BinaryOp,
                identity: Self,
            ) -> GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { crate::sys::$monoid(m, op, identity) }
            }
        }
    };
}

builtin_type! {
    bool, GrB_BOOL,
    matrix: GrB_Matrix_setElement_BOOL, GrB_Matrix_extractElement_BOOL, GrB_Matrix_build_BOOL, GrB_Matrix_extractTuples_BOOL, GrB_Matrix_select_BOOL, GrB_Matrix_apply_IndexOp_BOOL, GrB_Matrix_assign_BOOL, GrB_Matrix_reduce_BOOL;
    vector: GrB_Vector_setElement_BOOL, GrB_Vector_extractElement_BOOL, GrB_Vector_build_BOOL, GrB_Vector_extractTuples_BOOL, GrB_Vector_select_BOOL, GrB_Vector_apply_IndexOp_BOOL, GrB_Vector_assign_BOOL, GrB_Vector_reduce_BOOL;
    scalar: GrB_Scalar_setElement_BOOL, GrB_Scalar_extractElement_BOOL; monoid: GrB_Monoid_new_BOOL;
}
builtin_type! {
    i8, GrB_INT8,
    matrix: GrB_Matrix_setElement_INT8, GrB_Matrix_extractElement_INT8, GrB_Matrix_build_INT8, GrB_Matrix_extractTuples_INT8, GrB_Matrix_select_INT8, GrB_Matrix_apply_IndexOp_INT8, GrB_Matrix_assign_INT8, GrB_Matrix_reduce_INT8;
    vector: GrB_Vector_setElement_INT8, GrB_Vector_extractElement_INT8, GrB_Vector_build_INT8, GrB_Vector_extractTuples_INT8, GrB_Vector_select_INT8, GrB_Vector_apply_IndexOp_INT8, GrB_Vector_assign_INT8, GrB_Vector_reduce_INT8;
    scalar: GrB_Scalar_setElement_INT8, GrB_Scalar_extractElement_INT8; monoid: GrB_Monoid_new_INT8;
}
builtin_type! {
    i16, GrB_INT16,
    matrix: GrB_Matrix_setElement_INT16, GrB_Matrix_extractElement_INT16, GrB_Matrix_build_INT16, GrB_Matrix_extractTuples_INT16, GrB_Matrix_select_INT16, GrB_Matrix_apply_IndexOp_INT16, GrB_Matrix_assign_INT16, GrB_Matrix_reduce_INT16;
    vector: GrB_Vector_setElement_INT16, GrB_Vector_extractElement_INT16, GrB_Vector_build_INT16, GrB_Vector_extractTuples_INT16, GrB_Vector_select_INT16, GrB_Vector_apply_IndexOp_INT16, GrB_Vector_assign_INT16, GrB_Vector_reduce_INT16;
    scalar: GrB_Scalar_setElement_INT16, GrB_Scalar_extractElement_INT16; monoid: GrB_Monoid_new_INT16;
}
builtin_type! {
    i32, GrB_INT32,
    matrix: GrB_Matrix_setElement_INT32, GrB_Matrix_extractElement_INT32, GrB_Matrix_build_INT32, GrB_Matrix_extractTuples_INT32, GrB_Matrix_select_INT32, GrB_Matrix_apply_IndexOp_INT32, GrB_Matrix_assign_INT32, GrB_Matrix_reduce_INT32;
    vector: GrB_Vector_setElement_INT32, GrB_Vector_extractElement_INT32, GrB_Vector_build_INT32, GrB_Vector_extractTuples_INT32, GrB_Vector_select_INT32, GrB_Vector_apply_IndexOp_INT32, GrB_Vector_assign_INT32, GrB_Vector_reduce_INT32;
    scalar: GrB_Scalar_setElement_INT32, GrB_Scalar_extractElement_INT32; monoid: GrB_Monoid_new_INT32;
}
builtin_type! {
    i64, GrB_INT64,
    matrix: GrB_Matrix_setElement_INT64, GrB_Matrix_extractElement_INT64, GrB_Matrix_build_INT64, GrB_Matrix_extractTuples_INT64, GrB_Matrix_select_INT64, GrB_Matrix_apply_IndexOp_INT64, GrB_Matrix_assign_INT64, GrB_Matrix_reduce_INT64;
    vector: GrB_Vector_setElement_INT64, GrB_Vector_extractElement_INT64, GrB_Vector_build_INT64, GrB_Vector_extractTuples_INT64, GrB_Vector_select_INT64, GrB_Vector_apply_IndexOp_INT64, GrB_Vector_assign_INT64, GrB_Vector_reduce_INT64;
    scalar: GrB_Scalar_setElement_INT64, GrB_Scalar_extractElement_INT64; monoid: GrB_Monoid_new_INT64;
}
builtin_type! {
    u8, GrB_UINT8,
    matrix: GrB_Matrix_setElement_UINT8, GrB_Matrix_extractElement_UINT8, GrB_Matrix_build_UINT8, GrB_Matrix_extractTuples_UINT8, GrB_Matrix_select_UINT8, GrB_Matrix_apply_IndexOp_UINT8, GrB_Matrix_assign_UINT8, GrB_Matrix_reduce_UINT8;
    vector: GrB_Vector_setElement_UINT8, GrB_Vector_extractElement_UINT8, GrB_Vector_build_UINT8, GrB_Vector_extractTuples_UINT8, GrB_Vector_select_UINT8, GrB_Vector_apply_IndexOp_UINT8, GrB_Vector_assign_UINT8, GrB_Vector_reduce_UINT8;
    scalar: GrB_Scalar_setElement_UINT8, GrB_Scalar_extractElement_UINT8; monoid: GrB_Monoid_new_UINT8;
}
builtin_type! {
    u16, GrB_UINT16,
    matrix: GrB_Matrix_setElement_UINT16, GrB_Matrix_extractElement_UINT16, GrB_Matrix_build_UINT16, GrB_Matrix_extractTuples_UINT16, GrB_Matrix_select_UINT16, GrB_Matrix_apply_IndexOp_UINT16, GrB_Matrix_assign_UINT16, GrB_Matrix_reduce_UINT16;
    vector: GrB_Vector_setElement_UINT16, GrB_Vector_extractElement_UINT16, GrB_Vector_build_UINT16, GrB_Vector_extractTuples_UINT16, GrB_Vector_select_UINT16, GrB_Vector_apply_IndexOp_UINT16, GrB_Vector_assign_UINT16, GrB_Vector_reduce_UINT16;
    scalar: GrB_Scalar_setElement_UINT16, GrB_Scalar_extractElement_UINT16; monoid: GrB_Monoid_new_UINT16;
}
builtin_type! {
    u32, GrB_UINT32,
    matrix: GrB_Matrix_setElement_UINT32, GrB_Matrix_extractElement_UINT32, GrB_Matrix_build_UINT32, GrB_Matrix_extractTuples_UINT32, GrB_Matrix_select_UINT32, GrB_Matrix_apply_IndexOp_UINT32, GrB_Matrix_assign_UINT32, GrB_Matrix_reduce_UINT32;
    vector: GrB_Vector_setElement_UINT32, GrB_Vector_extractElement_UINT32, GrB_Vector_build_UINT32, GrB_Vector_extractTuples_UINT32, GrB_Vector_select_UINT32, GrB_Vector_apply_IndexOp_UINT32, GrB_Vector_assign_UINT32, GrB_Vector_reduce_UINT32;
    scalar: GrB_Scalar_setElement_UINT32, GrB_Scalar_extractElement_UINT32; monoid: GrB_Monoid_new_UINT32;
}
builtin_type! {
    u64, GrB_UINT64,
    matrix: GrB_Matrix_setElement_UINT64, GrB_Matrix_extractElement_UINT64, GrB_Matrix_build_UINT64, GrB_Matrix_extractTuples_UINT64, GrB_Matrix_select_UINT64, GrB_Matrix_apply_IndexOp_UINT64, GrB_Matrix_assign_UINT64, GrB_Matrix_reduce_UINT64;
    vector: GrB_Vector_setElement_UINT64, GrB_Vector_extractElement_UINT64, GrB_Vector_build_UINT64, GrB_Vector_extractTuples_UINT64, GrB_Vector_select_UINT64, GrB_Vector_apply_IndexOp_UINT64, GrB_Vector_assign_UINT64, GrB_Vector_reduce_UINT64;
    scalar: GrB_Scalar_setElement_UINT64, GrB_Scalar_extractElement_UINT64; monoid: GrB_Monoid_new_UINT64;
}
builtin_type! {
    f32, GrB_FP32,
    matrix: GrB_Matrix_setElement_FP32, GrB_Matrix_extractElement_FP32, GrB_Matrix_build_FP32, GrB_Matrix_extractTuples_FP32, GrB_Matrix_select_FP32, GrB_Matrix_apply_IndexOp_FP32, GrB_Matrix_assign_FP32, GrB_Matrix_reduce_FP32;
    vector: GrB_Vector_setElement_FP32, GrB_Vector_extractElement_FP32, GrB_Vector_build_FP32, GrB_Vector_extractTuples_FP32, GrB_Vector_select_FP32, GrB_Vector_apply_IndexOp_FP32, GrB_Vector_assign_FP32, GrB_Vector_reduce_FP32;
    scalar: GrB_Scalar_setElement_FP32, GrB_Scalar_extractElement_FP32; monoid: GrB_Monoid_new_FP32;
}
builtin_type! {
    f64, GrB_FP64,
    matrix: GrB_Matrix_setElement_FP64, GrB_Matrix_extractElement_FP64, GrB_Matrix_build_FP64, GrB_Matrix_extractTuples_FP64, GrB_Matrix_select_FP64, GrB_Matrix_apply_IndexOp_FP64, GrB_Matrix_assign_FP64, GrB_Matrix_reduce_FP64;
    vector: GrB_Vector_setElement_FP64, GrB_Vector_extractElement_FP64, GrB_Vector_build_FP64, GrB_Vector_extractTuples_FP64, GrB_Vector_select_FP64, GrB_Vector_apply_IndexOp_FP64, GrB_Vector_assign_FP64, GrB_Vector_reduce_FP64;
    scalar: GrB_Scalar_setElement_FP64, GrB_Scalar_extractElement_FP64; monoid: GrB_Monoid_new_FP64;
}

// The complex types `GxB_FC32` / `GxB_FC64` are not mapped: the engine has
// no use for them and Rust has no built-in complex type.
