//! An owned `GrB_Scalar`.

use std::{marker::PhantomData, mem::MaybeUninit, ptr::null_mut};

use crate::{
    Result,
    error::{check, expect_success},
    init,
    sys::{self, GrB_Info, GrB_Scalar},
    types::GrbType,
};

/// An owned `GrB_Scalar` holding zero or one value of type `T`.
pub struct Scalar<T> {
    raw: GrB_Scalar,
    _elem: PhantomData<T>,
}

// SAFETY: a GraphBLAS handle is not tied to the thread that created it; the
// wrapper holds nothing else. Deliberately not `Sync`.
unsafe impl<T: Send> Send for Scalar<T> {}

impl<T: GrbType> Scalar<T> {
    /// An empty scalar.
    ///
    /// # Errors
    ///
    /// Out of memory, not initialised.
    pub fn new() -> Result<Self> {
        init::ensure_initialized()?;
        let mut raw = null_mut();
        // SAFETY: a valid out-pointer and a live type handle.
        check(unsafe { sys::GrB_Scalar_new(&raw mut raw, T::grb_type()) })?;
        // SAFETY: just created with `T`'s type; owned here from now on.
        Ok(unsafe { Self::from_raw(raw) })
    }

    /// A scalar holding `value`.
    ///
    /// # Errors
    ///
    /// Out of memory, not initialised.
    pub fn from_value(value: T) -> Result<Self> {
        let mut s = Self::new()?;
        s.set(value)?;
        Ok(s)
    }

    /// Take ownership of a raw handle.
    ///
    /// # Safety
    ///
    /// `raw` is a live, non-null scalar whose GraphBLAS type is `T`'s, and
    /// nothing else uses or frees it afterwards: the result owns it alone.
    pub unsafe fn from_raw(raw: GrB_Scalar) -> Self {
        debug_assert!(!raw.is_null());
        Self {
            raw,
            _elem: PhantomData,
        }
    }

    /// The raw handle, still owned here.
    #[must_use]
    pub const fn as_raw(&self) -> GrB_Scalar {
        self.raw
    }

    /// Store `value`.
    ///
    /// # Errors
    ///
    /// Out of memory.
    pub fn set(
        &mut self,
        value: T,
    ) -> Result<()> {
        // SAFETY: a live scalar of `T`'s type.
        check(unsafe { T::scalar_set_element(self.raw, value) })
    }

    /// The stored value, or `None` when empty.
    ///
    /// # Errors
    ///
    /// Never in practice; kept as a `Result` because the C call has one.
    pub fn get(&self) -> Result<Option<T>> {
        let mut x = MaybeUninit::<T>::uninit();
        // SAFETY: a live scalar of `T`'s type and a valid out-pointer.
        let info = unsafe { T::scalar_extract_element(x.as_mut_ptr(), self.raw) };
        match info {
            // SAFETY: GraphBLAS wrote the value on success.
            GrB_Info::GrB_SUCCESS => Ok(Some(unsafe { x.assume_init() })),
            GrB_Info::GrB_NO_VALUE => Ok(None),
            other => check(other).map(|()| None),
        }
    }

    /// Remove the value.
    ///
    /// # Errors
    ///
    /// Never in practice.
    pub fn clear(&mut self) -> Result<()> {
        // SAFETY: a live scalar.
        check(unsafe { sys::GrB_Scalar_clear(self.raw) })
    }

    /// Number of stored values: 0 or 1.
    #[must_use]
    pub fn nvals(&self) -> u64 {
        let mut n = 0;
        // SAFETY: a live scalar and a valid out-pointer.
        let info = unsafe { sys::GrB_Scalar_nvals(&raw mut n, self.raw) };
        expect_success(info, "GrB_Scalar_nvals");
        n
    }
}

impl<T> Drop for Scalar<T> {
    fn drop(&mut self) {
        // SAFETY: `raw` is owned by this value and freed exactly once.
        let _ = unsafe { sys::GrB_Scalar_free(&raw mut self.raw) };
    }
}
