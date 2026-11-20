//! Typed operator handles and the built-ins the engine uses.
//!
//! Every handle is a `Copy` wrapper over the raw pointer plus the element
//! types it was declared with. GraphBLAS treats operators as immutable after
//! creation and lets any thread use them, hence `Send + Sync`.
//!
//! User-defined operators are declared once, as statics, with
//! [`user_ops!`](crate::user_ops): created on first use and never freed,
//! exactly like a [`user_type!`](crate::user_type). The `new` constructors
//! underneath are the same: an operator built ad hoc also lives until the
//! process exits, so build each one once.

use std::{fmt, marker::PhantomData, mem::MaybeUninit, os::raw::c_void, ptr::null_mut};

use crate::{
    Result,
    builtin::builtin,
    error::{check, expect_success},
    init::ensure_initialized,
    sys::{self, GrB_BinaryOp, GrB_Index, GrB_IndexUnaryOp, GrB_Monoid, GrB_Semiring, GrB_UnaryOp},
    types::{BuiltinPattern, GrbType, Pattern},
};

/// Declare user-defined operators as statics, created on first use.
///
/// Each line is `static NAME: Kind<..> = <expression returning Result<Kind<..>>>;`.
/// The expression runs once, on the first use of `NAME`, and may name other
/// statics declared the same way, which creates them first. An operator
/// made from a C function is an `unsafe` expression, because the function
/// has to meet the contract of [`UnaryOp::new`]; a monoid or semiring made
/// from typed operators is not. Like a
/// [`user_type!`](crate::user_type), the object lives until the process
/// exits, and a failure to create it panics with the static's name.
///
/// ```ignore
/// graphblas::user_ops! {
///     /// `z = x.score`.
///     // SAFETY: `score` reads `x` and then writes a whole `f64`.
///     pub static SCORE: UnaryOp<ScoredEdge, f64> = unsafe { UnaryOp::new(score) };
///     // SAFETY: `keep_min` copies both inputs out before it writes `z`.
///     pub static KEEP_MIN: BinaryOp<ScoredEdge, ScoredEdge, ScoredEdge> =
///         unsafe { BinaryOp::new(keep_min) };
///     pub static MIN_SCORE: Monoid<ScoredEdge> = Monoid::new(&KEEP_MIN, ScoredEdge::MAX);
///     pub static MIN_SCORE_PATH: Semiring<ScoredEdge, ScoredEdge, ScoredEdge> =
///         Semiring::new(&MIN_SCORE, &KEEP_MIN);
/// }
/// ```
#[macro_export]
macro_rules! user_ops {
    () => {};
    (
        $(#[$meta:meta])* $vis:vis static $name:ident : $ty:ty = unsafe { $create:expr } ;
        $($rest:tt)*
    ) => {
        $(#[$meta])*
        $vis static $name: $crate::Lazy<$ty> = $crate::Lazy::new(|| {
            #[allow(
                clippy::macro_metavars_in_unsafe,
                reason = "this arm matches only when the caller wrote the `unsafe` block"
            )]
            // SAFETY: the caller wrote this `unsafe` block, and with it
            // vouches for the contract of the constructor inside.
            let created: $crate::Result<$ty> = unsafe { $create };
            created.unwrap_or_else(|e| panic!(concat!("creating `", stringify!($name), "`: {}"), e))
        });
        $crate::user_ops! { $($rest)* }
    };
    (
        $(#[$meta:meta])* $vis:vis static $name:ident : $ty:ty = $create:expr ;
        $($rest:tt)*
    ) => {
        $(#[$meta])*
        $vis static $name: $crate::Lazy<$ty> = $crate::Lazy::new(|| {
            let created: $crate::Result<$ty> = $create;
            created.unwrap_or_else(|e| panic!(concat!("creating `", stringify!($name), "`: {}"), e))
        });
        $crate::user_ops! { $($rest)* }
    };
}

macro_rules! op_handle {
    ($(#[$meta:meta])* $name:ident < $($p:ident),+ > : $raw:ty) => {
        $(#[$meta])*
        #[repr(transparent)]
        pub struct $name<$($p),+> {
            raw: $raw,
            // `fn() -> (..)` keeps the struct `Send + Sync` and covariant
            // regardless of the parameters, which are only labels.
            _types: PhantomData<fn() -> ($($p,)+)>,
        }

        impl<$($p),+> $name<$($p),+> {
            /// Wrap a raw handle.
            ///
            /// # Safety
            ///
            /// `raw` is a live, non-null handle of this kind whose GraphBLAS
            /// domains are exactly the type parameters, and it outlives every
            /// use of the wrapper (built-ins and registered user operators do).
            pub(crate) const unsafe fn from_raw(raw: $raw) -> Self {
                Self { raw, _types: PhantomData }
            }

            /// The raw handle, for the LAGraph wrapper.
            #[must_use]
            pub const fn as_raw(&self) -> $raw {
                self.raw
            }
        }

        impl<$($p),+> Clone for $name<$($p),+> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<$($p),+> Copy for $name<$($p),+> {}

        impl<$($p),+> fmt::Debug for $name<$($p),+> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({:p})"), self.raw)
            }
        }

        // SAFETY: GraphBLAS operators are immutable once created and may be
        // used concurrently from any thread; the wrapper adds no state.
        unsafe impl<$($p),+> Send for $name<$($p),+> {}
        // SAFETY: as above.
        unsafe impl<$($p),+> Sync for $name<$($p),+> {}
    };
}

op_handle! {
    /// `z = f(x)`: a `GrB_UnaryOp` from domain `X` to `Z`.
    UnaryOp<X, Z>: GrB_UnaryOp
}
op_handle! {
    /// `z = f(x, y)`: a `GrB_BinaryOp`. As an accumulator it is
    /// `BinaryOp<C, C, C>` on the output type; as a `dup` for `build`
    /// likewise.
    BinaryOp<X, Y, Z>: GrB_BinaryOp
}
op_handle! {
    /// `z = f(x, i, j, y)`: a `GrB_IndexUnaryOp`. `Y` is the scalar `select`
    /// and `apply_index` pass alongside the matrix; for `select`, `Z = bool`.
    IndexUnaryOp<X, Y, Z>: GrB_IndexUnaryOp
}
op_handle! {
    /// A commutative, associative `BinaryOp<T, T, T>` with an identity.
    Monoid<T>: GrB_Monoid
}
op_handle! {
    /// `(add: Monoid<Z>, multiply: BinaryOp<X, Y, Z>)`.
    Semiring<X, Y, Z>: GrB_Semiring
}

impl<X: GrbType, Z: GrbType> UnaryOp<X, Z> {
    /// A user-defined unary operator from a typed C function. Never
    /// freed.
    ///
    /// # Safety
    ///
    /// GraphBLAS calls `f` itself, for as long as the library is up, so `f`
    /// must be sound however GraphBLAS calls it:
    ///
    /// * **From several threads at once.** It must not touch shared state
    ///   without synchronising.
    /// * **With `z` equal to an input pointer.** GraphBLAS applies and
    ///   reduces in place. Read the inputs before writing `*z`, and never
    ///   hold `&mut *z` together with a reference to an input.
    /// * **With `*z` uninitialised.** Write a whole `Z`; do not read it
    ///   first.
    /// * **With pointers into GraphBLAS's own buffers.** Its arrays come
    ///   from the allocator. Its stack buffers are byte arrays, which the C
    ///   ABI aligns to 16 in practice but does not promise to; a function
    ///   that wants to be strict uses `read_unaligned` and `write_unaligned`.
    ///
    /// A panic in `f` aborts the process, as for any `extern "C"` function.
    ///
    /// # Errors
    ///
    /// When GraphBLAS refuses the operator (out of memory, not initialised).
    pub unsafe fn new(f: unsafe extern "C" fn(z: *mut Z, x: *const X)) -> Result<Self> {
        ensure_initialized()?;
        // SAFETY: GraphBLAS calls the function with pointers to exactly the
        // types registered as its domains below; the `void *` signature it
        // is declared through is layout-identical.
        let function: sys::GxB_unary_function = Some(unsafe {
            std::mem::transmute::<
                unsafe extern "C" fn(*mut Z, *const X),
                unsafe extern "C" fn(*mut c_void, *const c_void),
            >(f)
        });
        let mut raw = null_mut();
        // SAFETY: a valid out-pointer and live type handles.
        check(unsafe {
            sys::GrB_UnaryOp_new(&raw mut raw, function, Z::grb_type(), X::grb_type())
        })?;
        // SAFETY: a live handle with domains `X -> Z`, never freed.
        Ok(unsafe { Self::from_raw(raw) })
    }
}

impl<X: GrbType, Y: GrbType, Z: GrbType> BinaryOp<X, Y, Z> {
    /// A user-defined binary operator from a typed C function. Never
    /// freed.
    ///
    /// # Safety
    ///
    /// As for [`UnaryOp::new`]. `z` may equal `x` or `y`: a reduction calls
    /// `f (z, z, y)`.
    ///
    /// # Errors
    ///
    /// When GraphBLAS refuses the operator (out of memory, not initialised).
    pub unsafe fn new(
        f: unsafe extern "C" fn(z: *mut Z, x: *const X, y: *const Y)
    ) -> Result<Self> {
        ensure_initialized()?;
        // SAFETY: as in `UnaryOp::new`.
        let function: sys::GxB_binary_function = Some(unsafe {
            std::mem::transmute::<
                unsafe extern "C" fn(*mut Z, *const X, *const Y),
                unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void),
            >(f)
        });
        let mut raw = null_mut();
        // SAFETY: a valid out-pointer and live type handles.
        check(unsafe {
            sys::GrB_BinaryOp_new(
                &raw mut raw,
                function,
                Z::grb_type(),
                X::grb_type(),
                Y::grb_type(),
            )
        })?;
        // SAFETY: a live handle with domains `X x Y -> Z`, never freed.
        Ok(unsafe { Self::from_raw(raw) })
    }
}

impl<X: GrbType, Y: GrbType, Z: GrbType> IndexUnaryOp<X, Y, Z> {
    /// A user-defined index-unary operator from a typed C function.
    /// Never freed.
    ///
    /// # Safety
    ///
    /// As for [`UnaryOp::new`].
    ///
    /// # Errors
    ///
    /// When GraphBLAS refuses the operator (out of memory, not initialised).
    pub unsafe fn new(
        f: unsafe extern "C" fn(z: *mut Z, x: *const X, i: GrB_Index, j: GrB_Index, y: *const Y)
    ) -> Result<Self> {
        ensure_initialized()?;
        // SAFETY: as in `UnaryOp::new`.
        let function: sys::GxB_index_unary_function = Some(unsafe {
            std::mem::transmute::<
                unsafe extern "C" fn(*mut Z, *const X, GrB_Index, GrB_Index, *const Y),
                unsafe extern "C" fn(
                    *mut c_void,
                    *const c_void,
                    GrB_Index,
                    GrB_Index,
                    *const c_void,
                ),
            >(f)
        });
        let mut raw = null_mut();
        // SAFETY: a valid out-pointer and live type handles.
        check(unsafe {
            sys::GrB_IndexUnaryOp_new(
                &raw mut raw,
                function,
                Z::grb_type(),
                X::grb_type(),
                Y::grb_type(),
            )
        })?;
        // SAFETY: a live handle with domains `X, Y -> Z`, never freed.
        Ok(unsafe { Self::from_raw(raw) })
    }
}

impl<T: GrbType> Monoid<T> {
    /// A monoid from a binary operator on `T` and its identity. Never
    /// freed.
    ///
    /// # Errors
    ///
    /// When GraphBLAS refuses the monoid (out of memory, not initialised).
    pub fn new(
        op: &BinaryOp<T, T, T>,
        identity: T,
    ) -> Result<Self> {
        ensure_initialized()?;
        let mut raw = null_mut();
        // SAFETY: a valid out-pointer and a live operator on `T`.
        check(unsafe { T::monoid_new(&raw mut raw, op.as_raw(), identity) })?;
        // SAFETY: a live monoid on `T`, never freed.
        Ok(unsafe { Self::from_raw(raw) })
    }

    /// The identity (`GxB_Monoid_identity`).
    #[must_use]
    pub fn identity(&self) -> T {
        let mut x = MaybeUninit::<T>::uninit();
        // SAFETY: a live monoid on `T` and room for one `T`.
        let info = unsafe { sys::GxB_Monoid_identity(x.as_mut_ptr().cast(), self.raw) };
        expect_success(info, "GxB_Monoid_identity");
        // SAFETY: GraphBLAS wrote the identity on success.
        unsafe { x.assume_init() }
    }
}

impl<T> Monoid<T> {
    /// The monoid's binary operator (`GxB_Monoid_operator`).
    #[must_use]
    pub fn op(&self) -> BinaryOp<T, T, T> {
        let mut op = null_mut();
        // SAFETY: a live monoid and a valid out-pointer.
        let info = unsafe { sys::GxB_Monoid_operator(&raw mut op, self.raw) };
        expect_success(info, "GxB_Monoid_operator");
        // SAFETY: a monoid's operator is `T x T -> T` and lives as long as
        // the monoid.
        unsafe { BinaryOp::from_raw(op) }
    }
}

impl<X, Y, Z: GrbType> Semiring<X, Y, Z> {
    /// A semiring from an additive monoid on `Z` and a multiplicative
    /// operator `X x Y -> Z`. Never freed.
    ///
    /// # Errors
    ///
    /// When GraphBLAS refuses the semiring (out of memory, not initialised).
    pub fn new(
        add: &Monoid<Z>,
        multiply: &BinaryOp<X, Y, Z>,
    ) -> Result<Self> {
        ensure_initialized()?;
        let mut raw = null_mut();
        // SAFETY: a valid out-pointer and live operator handles.
        check(unsafe { sys::GrB_Semiring_new(&raw mut raw, add.as_raw(), multiply.as_raw()) })?;
        // SAFETY: a live semiring with domains `X x Y -> Z`, never freed.
        Ok(unsafe { Self::from_raw(raw) })
    }
}

impl<X, Y, Z> Semiring<X, Y, Z> {
    /// The additive monoid (`GxB_Semiring_add`).
    #[must_use]
    pub fn add(&self) -> Monoid<Z> {
        let mut add = null_mut();
        // SAFETY: a live semiring and a valid out-pointer.
        let info = unsafe { sys::GxB_Semiring_add(&raw mut add, self.raw) };
        expect_success(info, "GxB_Semiring_add");
        // SAFETY: the additive monoid is on `Z` and lives as long as the
        // semiring.
        unsafe { Monoid::from_raw(add) }
    }

    /// The multiplicative operator (`GxB_Semiring_multiply`).
    #[must_use]
    pub fn multiply(&self) -> BinaryOp<X, Y, Z> {
        let mut mul = null_mut();
        // SAFETY: a live semiring and a valid out-pointer.
        let info = unsafe { sys::GxB_Semiring_multiply(&raw mut mul, self.raw) };
        expect_success(info, "GxB_Semiring_multiply");
        // SAFETY: the multiplicative operator is `X x Y -> Z` and lives as
        // long as the semiring.
        unsafe { BinaryOp::from_raw(mul) }
    }

    /// The additive operator: the accumulator that lets `C += A*B` run in
    /// place (`GB_AxB_meta` requires `accum == semiring->add->op` by
    /// identity).
    #[must_use]
    pub fn add_op(&self) -> BinaryOp<Z, Z, Z> {
        self.add().op()
    }
}

// The built-ins the engine uses. The type on the left is the one place each
// symbol's domains are written down; an operator that never reads an input
// is declared with `Pattern` there. A test checks every line against what
// GraphBLAS reports for the object.
builtin! {
    /// `GxB_ONE_BOOL`: `z = true`, never reads `x`, which must still be of
    /// a built-in type.
    pub static ONE_BOOL: UnaryOp<BuiltinPattern, bool> = GxB_ONE_BOOL;
    /// `GrB_IDENTITY_BOOL`.
    pub static IDENTITY_BOOL: UnaryOp<bool, bool> = GrB_IDENTITY_BOOL;
    /// `GrB_IDENTITY_UINT64`.
    pub static IDENTITY_UINT64: UnaryOp<u64, u64> = GrB_IDENTITY_UINT64;
    /// `GrB_IDENTITY_FP64`.
    pub static IDENTITY_FP64: UnaryOp<f64, f64> = GrB_IDENTITY_FP64;
    /// `GrB_AINV_FP64`: `z = -x`.
    pub static AINV_FP64: UnaryOp<f64, f64> = GrB_AINV_FP64;

    /// `GxB_ANY_BOOL`: either operand; as an accumulator, keeps the pattern.
    pub static ANY_BOOL: BinaryOp<bool, bool, bool> = GxB_ANY_BOOL;
    /// `GxB_ANY_UINT64`: either operand.
    pub static ANY_UINT64: BinaryOp<u64, u64, u64> = GxB_ANY_UINT64;
    /// `GrB_FIRST_UINT64`: `z = x`.
    pub static FIRST_UINT64: BinaryOp<u64, u64, u64> = GrB_FIRST_UINT64;
    /// `GrB_SECOND_UINT64`: `z = y` (the delta's value shadows the base's).
    pub static SECOND_UINT64: BinaryOp<u64, u64, u64> = GrB_SECOND_UINT64;
    /// `GrB_PLUS_UINT64`.
    pub static PLUS_UINT64: BinaryOp<u64, u64, u64> = GrB_PLUS_UINT64;
    /// `GrB_PLUS_INT64`.
    pub static PLUS_INT64: BinaryOp<i64, i64, i64> = GrB_PLUS_INT64;
    /// `GrB_PLUS_FP64`.
    pub static PLUS_FP64: BinaryOp<f64, f64, f64> = GrB_PLUS_FP64;
    /// `GrB_MIN_FP64`.
    pub static MIN_FP64: BinaryOp<f64, f64, f64> = GrB_MIN_FP64;
    /// `GrB_MAX_FP64`.
    pub static MAX_FP64: BinaryOp<f64, f64, f64> = GrB_MAX_FP64;
    /// `GrB_GT_UINT64`: `z = x > y`.
    pub static GT_UINT64: BinaryOp<u64, u64, bool> = GrB_GT_UINT64;
    /// `GrB_ONEB_BOOL`: `z = true`, never reads `x` or `y`.
    pub static ONEB_BOOL: BinaryOp<Pattern, Pattern, bool> = GrB_ONEB_BOOL;
    /// `GxB_PAIR_BOOL`: `z = true`, never reads `x` or `y` (the multiply
    /// of `ANY_PAIR_BOOL`).
    pub static PAIR_BOOL: BinaryOp<Pattern, Pattern, bool> = GxB_PAIR_BOOL;

    /// `GrB_VALUEEQ_UINT64`: keep entries equal to `y`.
    pub static VALUEEQ_UINT64: IndexUnaryOp<u64, u64, bool> = GrB_VALUEEQ_UINT64;
    /// `GrB_VALUENE_UINT64`: keep entries not equal to `y`.
    pub static VALUENE_UINT64: IndexUnaryOp<u64, u64, bool> = GrB_VALUENE_UINT64;
    /// `GrB_VALUEEQ_BOOL`: keep entries equal to `y`.
    pub static VALUEEQ_BOOL: IndexUnaryOp<bool, bool, bool> = GrB_VALUEEQ_BOOL;
    /// `GrB_VALUEGT_FP64`: keep entries greater than `y`.
    pub static VALUEGT_FP64: IndexUnaryOp<f64, f64, bool> = GrB_VALUEGT_FP64;
    /// `GrB_VALUELT_FP64`: keep entries less than `y`.
    pub static VALUELT_FP64: IndexUnaryOp<f64, f64, bool> = GrB_VALUELT_FP64;
    /// `GrB_DIAG`: keep the diagonal `j - i == y`; never reads the value.
    pub static DIAG: IndexUnaryOp<Pattern, i64, bool> = GrB_DIAG;
    /// `GrB_OFFDIAG`: drop the diagonal `j - i == y`; never reads the value.
    pub static OFFDIAG: IndexUnaryOp<Pattern, i64, bool> = GrB_OFFDIAG;
    /// `GrB_ROWINDEX_INT64`: `z = i + y`; never reads the value.
    pub static ROWINDEX_INT64: IndexUnaryOp<Pattern, i64, i64> = GrB_ROWINDEX_INT64;
    /// `GrB_COLINDEX_INT64`: `z = j + y`; never reads the value.
    pub static COLINDEX_INT64: IndexUnaryOp<Pattern, i64, i64> = GrB_COLINDEX_INT64;

    /// `GrB_PLUS_MONOID_UINT64`.
    pub static PLUS_MONOID_UINT64: Monoid<u64> = GrB_PLUS_MONOID_UINT64;
    /// `GrB_PLUS_MONOID_INT64`.
    pub static PLUS_MONOID_INT64: Monoid<i64> = GrB_PLUS_MONOID_INT64;
    /// `GrB_PLUS_MONOID_FP64`.
    pub static PLUS_MONOID_FP64: Monoid<f64> = GrB_PLUS_MONOID_FP64;
    /// `GrB_MIN_MONOID_FP64`.
    pub static MIN_MONOID_FP64: Monoid<f64> = GrB_MIN_MONOID_FP64;
    /// `GrB_MAX_MONOID_FP64`.
    pub static MAX_MONOID_FP64: Monoid<f64> = GrB_MAX_MONOID_FP64;
    /// `GrB_LOR_MONOID_BOOL`.
    pub static LOR_MONOID_BOOL: Monoid<bool> = GrB_LOR_MONOID_BOOL;
    /// `GrB_LAND_MONOID_BOOL`.
    pub static LAND_MONOID_BOOL: Monoid<bool> = GrB_LAND_MONOID_BOOL;
    /// `GxB_ANY_BOOL_MONOID`.
    pub static ANY_MONOID_BOOL: Monoid<bool> = GxB_ANY_BOOL_MONOID;

    /// `GxB_ANY_PAIR_BOOL`: pattern-only product; reads no values.
    pub static ANY_PAIR_BOOL: Semiring<Pattern, Pattern, bool> = GxB_ANY_PAIR_BOOL;
    /// `GxB_PLUS_PAIR_UINT64`: counts paths; reads no values.
    pub static PLUS_PAIR_UINT64: Semiring<Pattern, Pattern, u64> = GxB_PLUS_PAIR_UINT64;
    /// `GxB_PLUS_PAIR_INT64`: counts paths; reads no values.
    pub static PLUS_PAIR_INT64: Semiring<Pattern, Pattern, i64> = GxB_PLUS_PAIR_INT64;
    /// `GxB_PLUS_FIRSTJ_INT64`: `z = j`, the index of the left operand
    /// (see the positional idiom in the workspace `CLAUDE.md`).
    pub static PLUS_FIRSTJ_INT64: Semiring<Pattern, Pattern, i64> = GxB_PLUS_FIRSTJ_INT64;
    /// `GrB_PLUS_TIMES_SEMIRING_FP64`.
    pub static PLUS_TIMES_SEMIRING_FP64: Semiring<f64, f64, f64> = GrB_PLUS_TIMES_SEMIRING_FP64;
    /// `GrB_MIN_PLUS_SEMIRING_FP64`.
    pub static MIN_PLUS_SEMIRING_FP64: Semiring<f64, f64, f64> = GrB_MIN_PLUS_SEMIRING_FP64;
    /// `GrB_LOR_LAND_SEMIRING_BOOL`.
    pub static LOR_LAND_SEMIRING_BOOL: Semiring<bool, bool, bool> = GrB_LOR_LAND_SEMIRING_BOOL;
}
