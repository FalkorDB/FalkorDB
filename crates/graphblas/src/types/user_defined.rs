//! The `user_type!` macro: [`GrbType`](super::GrbType) for a caller's
//! `#[repr(C)]` struct, through the `_UDT` entry points.
//!
//! Both macros are `#[macro_export]`, so they live at the crate root
//! regardless of this file; everything they name goes through `$crate`.

use std::{
    ffi::CString,
    ptr::{null, null_mut},
    sync::OnceLock,
};

use crate::{
    init::assert_initialized,
    sys::{self, GrB_Info, GrB_Type},
};

/// The lazily created `GrB_Type` of one user-defined type. One static per
/// `user_type!` invocation; not part of the API.
#[doc(hidden)]
pub struct UserTypeCell(OnceLock<usize>);

impl UserTypeCell {
    #[must_use]
    pub const fn new() -> Self {
        Self(OnceLock::new())
    }

    /// The handle, created on the first call with `GxB_Type_new(size)` and
    /// never freed: it lives as long as the library. `name` is given to GraphBLAS when it is a
    /// C identifier (it shows in the burble and in `GrB_get`).
    ///
    /// # Panics
    ///
    /// Before `init`, or when GraphBLAS cannot create the type.
    pub fn get_or_create(
        &self,
        size: usize,
        name: &str,
    ) -> GrB_Type {
        let handle = *self.0.get_or_init(|| {
            assert_initialized(name);
            let is_identifier = name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
            let c_name =
                is_identifier.then(|| CString::new(name).expect("no NUL in an identifier"));
            let mut raw: GrB_Type = null_mut();
            // SAFETY: a valid out-pointer; the name, when given, is a
            // NUL-terminated string that outlives the call (GraphBLAS copies
            // it); no C definition is given, so the JIT treats the type as
            // opaque.
            let info = unsafe {
                sys::GxB_Type_new(
                    &raw mut raw,
                    size,
                    c_name.as_ref().map_or(null(), |n| n.as_ptr()),
                    null(),
                )
            };
            assert!(
                info == GrB_Info::GrB_SUCCESS && !raw.is_null(),
                "GxB_Type_new for `{name}` ({size} bytes) failed: {info:?}"
            );
            raw as usize
        });
        handle as GrB_Type
    }
}

impl Default for UserTypeCell {
    fn default() -> Self {
        Self::new()
    }
}

/// Make a `#[repr(C)]`, `Copy` struct usable as a GraphBLAS element type.
///
/// ```ignore
/// #[repr(C)]
/// #[derive(Clone, Copy)]
/// struct ScoredEdge { score: f64, rel: u64 }
/// graphblas::user_type!(ScoredEdge);
/// ```
///
/// The `GrB_Type` is created with `GrB_Type_new(size_of::<T>())` on the
/// first use of `ScoredEdge::grb_type()`, which every constructor of a
/// `Matrix<ScoredEdge>` or an operator over it goes through, so it cannot
/// run before [`init`](crate::init) (the first use asserts it). The type
/// lives until the process exits. Values reach the
/// `_UDT` entry points by pointer. GraphBLAS never casts user-defined types,
/// and neither does this crate: `Accepts` holds only for `ScoredEdge` itself
/// and for `Pattern`.
#[macro_export]
macro_rules! user_type {
    ($t:ty) => {
        $crate::__user_type_impl!($t);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __user_type_impl {
    ($t:ty) => {
        // GraphBLAS refuses a zero-sized type, and it hands operators
        // pointers into arrays from the allocator, aligned for nothing
        // wider than 16 bytes.
        const _: () = assert!(
            ::core::mem::size_of::<$t>() > 0 && ::core::mem::align_of::<$t>() <= 16,
            "a user-defined GraphBLAS type needs a size and an alignment of at most 16",
        );
        impl $crate::__sealed::Sealed for $t {}
        impl $crate::Accepts<$t> for $t {}
        impl $crate::Accepts<$t> for $crate::Pattern {}

        // SAFETY: the type handle is created with `size_of::<$t>()`, and
        // every method forwards to the `_UDT` entry point.
        unsafe impl $crate::GrbType for $t {
            const NAME: &'static str = stringify!($t);

            fn grb_type() -> $crate::__sys::GrB_Type {
                static CELL: $crate::__UserTypeCell = $crate::__UserTypeCell::new();
                CELL.get_or_create(::core::mem::size_of::<Self>(), Self::NAME)
            }

            // Each body forwards to the `_UDT` entry point, passing the value
            // by pointer as the C API takes it. Those entry points take a
            // `void *` even for an input they only read, hence `cast_mut`.
            unsafe fn matrix_set_element(
                c: $crate::__sys::GrB_Matrix,
                x: Self,
                i: u64,
                j: u64,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; `&x` outlives the call.
                unsafe {
                    $crate::__sys::GrB_Matrix_setElement_UDT(
                        c,
                        (&raw const x).cast_mut().cast(),
                        i,
                        j,
                    )
                }
            }
            unsafe fn matrix_extract_element(
                x: *mut Self,
                a: $crate::__sys::GrB_Matrix,
                i: u64,
                j: u64,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { $crate::__sys::GrB_Matrix_extractElement_UDT(x.cast(), a, i, j) }
            }
            unsafe fn matrix_build(
                c: $crate::__sys::GrB_Matrix,
                rows: *const u64,
                cols: *const u64,
                x: *const Self,
                n: u64,
                dup: $crate::__sys::GrB_BinaryOp,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { $crate::__sys::GrB_Matrix_build_UDT(c, rows, cols, x.cast(), n, dup) }
            }
            unsafe fn matrix_extract_tuples(
                rows: *mut u64,
                cols: *mut u64,
                x: *mut Self,
                n: *mut u64,
                a: $crate::__sys::GrB_Matrix,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { $crate::__sys::GrB_Matrix_extractTuples_UDT(rows, cols, x.cast(), n, a) }
            }
            unsafe fn matrix_select(
                c: $crate::__sys::GrB_Matrix,
                mask: $crate::__sys::GrB_Matrix,
                accum: $crate::__sys::GrB_BinaryOp,
                op: $crate::__sys::GrB_IndexUnaryOp,
                a: $crate::__sys::GrB_Matrix,
                y: Self,
                desc: $crate::__sys::GrB_Descriptor,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; `&y` outlives the call.
                unsafe {
                    $crate::__sys::GrB_Matrix_select_UDT(
                        c,
                        mask,
                        accum,
                        op,
                        a,
                        (&raw const y).cast(),
                        desc,
                    )
                }
            }
            unsafe fn matrix_apply_index(
                c: $crate::__sys::GrB_Matrix,
                mask: $crate::__sys::GrB_Matrix,
                accum: $crate::__sys::GrB_BinaryOp,
                op: $crate::__sys::GrB_IndexUnaryOp,
                a: $crate::__sys::GrB_Matrix,
                y: Self,
                desc: $crate::__sys::GrB_Descriptor,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; `&y` outlives the call.
                unsafe {
                    $crate::__sys::GrB_Matrix_apply_IndexOp_UDT(
                        c,
                        mask,
                        accum,
                        op,
                        a,
                        (&raw const y).cast(),
                        desc,
                    )
                }
            }
            unsafe fn matrix_assign_scalar(
                c: $crate::__sys::GrB_Matrix,
                mask: $crate::__sys::GrB_Matrix,
                accum: $crate::__sys::GrB_BinaryOp,
                x: Self,
                rows: *const u64,
                nrows: u64,
                cols: *const u64,
                ncols: u64,
                desc: $crate::__sys::GrB_Descriptor,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; `&x` outlives the call.
                unsafe {
                    $crate::__sys::GrB_Matrix_assign_UDT(
                        c,
                        mask,
                        accum,
                        (&raw const x).cast_mut().cast(),
                        rows,
                        nrows,
                        cols,
                        ncols,
                        desc,
                    )
                }
            }
            unsafe fn matrix_reduce(
                x: *mut Self,
                accum: $crate::__sys::GrB_BinaryOp,
                monoid: $crate::__sys::GrB_Monoid,
                a: $crate::__sys::GrB_Matrix,
                desc: $crate::__sys::GrB_Descriptor,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { $crate::__sys::GrB_Matrix_reduce_UDT(x.cast(), accum, monoid, a, desc) }
            }

            unsafe fn vector_set_element(
                w: $crate::__sys::GrB_Vector,
                x: Self,
                i: u64,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; `&x` outlives the call.
                unsafe {
                    $crate::__sys::GrB_Vector_setElement_UDT(w, (&raw const x).cast_mut().cast(), i)
                }
            }
            unsafe fn vector_extract_element(
                x: *mut Self,
                u: $crate::__sys::GrB_Vector,
                i: u64,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { $crate::__sys::GrB_Vector_extractElement_UDT(x.cast(), u, i) }
            }
            unsafe fn vector_build(
                w: $crate::__sys::GrB_Vector,
                idx: *const u64,
                x: *const Self,
                n: u64,
                dup: $crate::__sys::GrB_BinaryOp,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { $crate::__sys::GrB_Vector_build_UDT(w, idx, x.cast(), n, dup) }
            }
            unsafe fn vector_extract_tuples(
                idx: *mut u64,
                x: *mut Self,
                n: *mut u64,
                u: $crate::__sys::GrB_Vector,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { $crate::__sys::GrB_Vector_extractTuples_UDT(idx, x.cast(), n, u) }
            }
            unsafe fn vector_select(
                w: $crate::__sys::GrB_Vector,
                mask: $crate::__sys::GrB_Vector,
                accum: $crate::__sys::GrB_BinaryOp,
                op: $crate::__sys::GrB_IndexUnaryOp,
                u: $crate::__sys::GrB_Vector,
                y: Self,
                desc: $crate::__sys::GrB_Descriptor,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; `&y` outlives the call.
                unsafe {
                    $crate::__sys::GrB_Vector_select_UDT(
                        w,
                        mask,
                        accum,
                        op,
                        u,
                        (&raw const y).cast(),
                        desc,
                    )
                }
            }
            unsafe fn vector_apply_index(
                w: $crate::__sys::GrB_Vector,
                mask: $crate::__sys::GrB_Vector,
                accum: $crate::__sys::GrB_BinaryOp,
                op: $crate::__sys::GrB_IndexUnaryOp,
                u: $crate::__sys::GrB_Vector,
                y: Self,
                desc: $crate::__sys::GrB_Descriptor,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; `&y` outlives the call.
                unsafe {
                    $crate::__sys::GrB_Vector_apply_IndexOp_UDT(
                        w,
                        mask,
                        accum,
                        op,
                        u,
                        (&raw const y).cast(),
                        desc,
                    )
                }
            }
            unsafe fn vector_assign_scalar(
                w: $crate::__sys::GrB_Vector,
                mask: $crate::__sys::GrB_Vector,
                accum: $crate::__sys::GrB_BinaryOp,
                x: Self,
                idx: *const u64,
                n: u64,
                desc: $crate::__sys::GrB_Descriptor,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; `&x` outlives the call.
                unsafe {
                    $crate::__sys::GrB_Vector_assign_UDT(
                        w,
                        mask,
                        accum,
                        (&raw const x).cast_mut().cast(),
                        idx,
                        n,
                        desc,
                    )
                }
            }
            unsafe fn vector_reduce(
                x: *mut Self,
                accum: $crate::__sys::GrB_BinaryOp,
                monoid: $crate::__sys::GrB_Monoid,
                u: $crate::__sys::GrB_Vector,
                desc: $crate::__sys::GrB_Descriptor,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { $crate::__sys::GrB_Vector_reduce_UDT(x.cast(), accum, monoid, u, desc) }
            }

            unsafe fn scalar_set_element(
                s: $crate::__sys::GrB_Scalar,
                x: Self,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; `&x` outlives the call.
                unsafe {
                    $crate::__sys::GrB_Scalar_setElement_UDT(s, (&raw const x).cast_mut().cast())
                }
            }
            unsafe fn scalar_extract_element(
                x: *mut Self,
                s: $crate::__sys::GrB_Scalar,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; the caller upholds the C contract.
                unsafe { $crate::__sys::GrB_Scalar_extractElement_UDT(x.cast(), s) }
            }
            unsafe fn monoid_new(
                m: *mut $crate::__sys::GrB_Monoid,
                op: $crate::__sys::GrB_BinaryOp,
                identity: Self,
            ) -> $crate::__sys::GrB_Info {
                // SAFETY: forwarded verbatim; `&identity` outlives the call.
                unsafe {
                    $crate::__sys::GrB_Monoid_new_UDT(
                        m,
                        op,
                        (&raw const identity).cast_mut().cast(),
                    )
                }
            }
        }
    };
}
