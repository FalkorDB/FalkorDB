//! [`Scalar`]: a typed `GrB_Scalar`, empty or holding one value, freed on drop.

use std::marker::PhantomData;
use std::ptr::null_mut;

use super::matrix::grb;
use super::{
    GrB_BOOL, GrB_Info, GrB_Scalar, GrB_Scalar_free, GrB_Scalar_new, GrB_Scalar_setElement_BOOL,
    GrB_Type,
};

/// A Rust type a [`Scalar`] holds: its GraphBLAS type and setter.
pub(crate) trait ScalarElement: Copy {
    /// The GraphBLAS type of a scalar holding `Self`.
    fn grb_type() -> GrB_Type;

    /// `GrB_Scalar_setElement_<T>`.
    ///
    /// # Safety
    /// `scalar` is a live scalar of type [`Self::grb_type`].
    unsafe fn set_element(
        scalar: GrB_Scalar,
        value: Self,
    ) -> GrB_Info;
}

impl ScalarElement for bool {
    fn grb_type() -> GrB_Type {
        // SAFETY: a read of the type global `GrB_init` set.
        unsafe { GrB_BOOL }
    }

    unsafe fn set_element(
        scalar: GrB_Scalar,
        value: Self,
    ) -> GrB_Info {
        unsafe { GrB_Scalar_setElement_BOOL(scalar, value) }
    }
}

/// A `GrB_Scalar` of `T`, which this value owns and frees on drop.
pub(crate) struct Scalar<T> {
    scalar: GrB_Scalar,
    element: PhantomData<T>,
}

impl<T: ScalarElement> Scalar<T> {
    /// An empty scalar.
    pub(crate) fn new() -> Self {
        let mut scalar: GrB_Scalar = null_mut();
        // SAFETY: `GrB_Scalar_new` writes a new scalar, which `Self` then frees.
        grb!(unsafe { GrB_Scalar_new(&raw mut scalar, T::grb_type()) });
        Self {
            scalar,
            element: PhantomData,
        }
    }

    /// Hold `value`, replacing what the scalar held.
    pub(crate) fn set(
        &mut self,
        value: T,
    ) {
        // SAFETY: our live scalar, of `T`'s type.
        grb!(unsafe { T::set_element(self.scalar, value) });
    }

    /// The handle, for a GraphBLAS call. It stays this scalar's to free.
    pub(crate) const fn ptr(&self) -> GrB_Scalar {
        self.scalar
    }
}

impl<T: ScalarElement> From<T> for Scalar<T> {
    fn from(value: T) -> Self {
        let mut scalar = Self::new();
        scalar.set(value);
        scalar
    }
}

impl<T> Drop for Scalar<T> {
    fn drop(&mut self) {
        // SAFETY: `self.scalar` is live and ours alone.
        let info = unsafe { GrB_Scalar_free(&raw mut self.scalar) };
        debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
    }
}

#[cfg(test)]
mod tests {
    use super::Scalar;
    use crate::graph::graphblas::{GrB_Scalar_extractElement_BOOL, GrB_Scalar_nvals, test_init};

    fn entries(scalar: &Scalar<bool>) -> u64 {
        let mut count = 0;
        // SAFETY: a live scalar.
        unsafe { GrB_Scalar_nvals(&raw mut count, scalar.ptr()) };
        count
    }

    #[test]
    fn new_is_empty_and_set_holds_the_value() {
        test_init::ensure_init();
        let mut scalar = Scalar::<bool>::new();
        assert_eq!(entries(&scalar), 0);
        scalar.set(false);
        assert_eq!(entries(&scalar), 1);
        let mut value = true;
        // SAFETY: a live scalar holding a `bool`.
        unsafe { GrB_Scalar_extractElement_BOOL(&raw mut value, scalar.ptr()) };
        assert!(!value);
        assert_eq!(entries(&Scalar::from(true)), 1);
    }
}
