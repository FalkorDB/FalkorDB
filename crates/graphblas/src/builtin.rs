//! Built-in GraphBLAS objects readable from safe code.
//!
//! The bindings expose every built-in as a `static mut` handle that
//! `GrB_init` fills in, so a plain read is `unsafe` and a read before
//! initialisation would cache a null pointer. The `builtin!` macro declares
//! each one as a [`Lazy`](crate::Lazy) static that reads the symbol once,
//! on first use, after checking that [`init`](crate::init) ran, and hands
//! out a typed handle from then on. User-defined operators
//! follow the same shape through [`user_ops!`](crate::user_ops).

/// Declare typed statics for GraphBLAS built-ins.
///
/// The type on the left is what the symbol *means*: the one place a
/// built-in's domains are written down, checked by every call that takes the
/// handle and, at test time, against what GraphBLAS reports for the object.
/// A static `ANY_PAIR_BOOL: Lazy<Semiring<..>>` is passed as
/// `&ANY_PAIR_BOOL` wherever a `&Semiring<..>` is expected.
macro_rules! builtin {
    ($( $(#[$meta:meta])* $vis:vis static $name:ident : $ty:ty = $sym:ident ; )*) => { $(
        $(#[$meta])*
        $vis static $name: $crate::Lazy<$ty> = {
            fn read() -> $ty {
                $crate::init::assert_initialized(stringify!($sym));
                // SAFETY: a by-value read of a handle `GrB_init` set and
                // nothing writes afterwards; no reference to the static is
                // taken.
                let raw = unsafe { $crate::sys::$sym };
                assert!(!raw.is_null(), concat!("GraphBLAS built-in `", stringify!($sym), "` is null"));
                // SAFETY: `raw` is the non-null handle of exactly the object
                // kind and element types the declaration names.
                unsafe { <$ty>::from_raw(raw) }
            }
            $crate::Lazy::new(read)
        };
    )*

        /// Every declaration above, for the declaration check in the tests.
        #[cfg(test)]
        #[allow(
            clippy::redundant_pub_crate,
            reason = "`ops` is re-exported whole, and this is not part of the API"
        )]
        pub(crate) fn all_declared() -> Vec<(&'static str, &'static dyn $crate::tests::Declared)> {
            vec![ $( (stringify!($sym), &*$name as &dyn $crate::tests::Declared), )* ]
        }
    };
}

pub(crate) use builtin;
