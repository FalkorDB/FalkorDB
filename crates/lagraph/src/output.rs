//! The objects LAGraph allocates for its outputs, from the C out-pointer to
//! a typed, owned `Vector<T>`.

use std::{mem, ptr::null_mut};

use graphblas::{BuiltinType, Desc, ErrorKind, GrbType, Mask, Vector, vector};

use crate::{
    Error, Result,
    sys::{self, GrB_Vector},
};

/// A raw handle LAGraph may write through an out-pointer.
pub trait Handle: Copy {
    fn null() -> Self;
    fn is_null(self) -> bool;
    /// `GrB_*_free`.
    ///
    /// # Safety
    ///
    /// A live handle nothing else owns.
    unsafe fn free(self);
}

impl Handle for GrB_Vector {
    fn null() -> Self {
        null_mut()
    }

    fn is_null(self) -> bool {
        self.is_null()
    }

    unsafe fn free(mut self) {
        // SAFETY: the caller's contract.
        let _ = unsafe { sys::GrB_Vector_free(&raw mut self) };
    }
}

/// The variable behind an out-pointer. Whatever LAGraph leaves in it is
/// freed on drop unless it was adopted, so a call that fails after
/// allocating an output, or an early return between two outputs, leaks
/// nothing.
pub struct Out<R: Handle>(R);

impl<R: Handle> Out<R> {
    pub(crate) fn new() -> Self {
        Self(R::null())
    }

    /// The out-pointer to pass.
    pub(crate) const fn pointer(&mut self) -> *mut R {
        &raw mut self.0
    }

    /// The handle, now the caller's to free; `None` if LAGraph wrote none.
    fn take(&mut self) -> Option<R> {
        let raw = mem::replace(&mut self.0, R::null());
        (!raw.is_null()).then_some(raw)
    }
}

impl<R: Handle> Drop for Out<R> {
    fn drop(&mut self) {
        if let Some(raw) = self.take() {
            // SAFETY: allocated by LAGraph for this call and not adopted.
            unsafe { raw.free() };
        }
    }
}

fn missing() -> Error {
    Error::graphblas(
        ErrorKind::NullPointer,
        "LAGraph reported success and returned no object",
    )
}

/// Free an output whose element type is not the documented one.
fn unexpected_type<T: GrbType, R: Handle>(raw: R) -> Error {
    // SAFETY: a live handle nobody adopted; freed once, here.
    unsafe { raw.free() };
    Error::graphblas(
        ErrorKind::DomainMismatch,
        &format!("LAGraph returned an object that is not {}", T::NAME),
    )
}

impl Out<GrB_Vector> {
    /// Own the vector as the `Vector<T>` LAGraph's documentation says it
    /// is. The element type is checked, since nothing else ties the two
    /// together.
    pub(crate) fn adopt<T: GrbType>(mut self) -> Result<Vector<T>> {
        let raw = self.take().ok_or_else(missing)?;
        // SAFETY: a live vector LAGraph allocated for this call, whose only
        // owner is now this function.
        unsafe { Vector::try_from_raw(raw) }.map_err(unexpected_type::<T, _>)
    }

    /// Own a vector of node ids or of levels. LAGraph sizes these to the
    /// graph: `INT32` below 2^31 nodes and `INT64` above, or `UINT64`. All
    /// of them widen to `u64` without loss; a signed one is copied to do it.
    pub(crate) fn adopt_node_ids(mut self) -> Result<Vector<u64>> {
        let raw = self.take().ok_or_else(missing)?;
        // SAFETY: as in `adopt`.
        let raw = match unsafe { Vector::<u64>::try_from_raw(raw) } {
            Ok(ids) => return Ok(ids),
            Err(raw) => raw,
        };
        widen::<i32>(raw)
            .or_else(widen::<i64>)
            .unwrap_or_else(|raw| Err(unexpected_type::<u64, _>(raw)))
    }
}

/// `raw` as a `Vector<u64>` if its element type is `S`; `Err(raw)` if not.
fn widen<S: BuiltinType>(raw: GrB_Vector) -> std::result::Result<Result<Vector<u64>>, GrB_Vector> {
    // SAFETY: a live vector whose only owner is the caller.
    let narrow = unsafe { Vector::<S>::try_from_raw(raw) }?;
    let widened = || {
        let mut wide = Vector::new(narrow.size())?;
        vector::cast(&mut wide, Mask::NONE, None, &narrow, Desc::NONE)?;
        Ok(wide)
    };
    Ok(widened())
}
