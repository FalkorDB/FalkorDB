//! The raw handle kinds behind [`Matrix`](crate::Matrix) and
//! [`Vector`](crate::Vector), and the calls both answer the same way, written
//! once over [`RawHandle`].

use std::os::raw::{c_char, c_int};

use crate::{
    Result,
    error::{Error, copy_message, expect_success},
    init::WaitMode,
    mask::{Desc, Mask, descriptor},
    settings::Sparsity,
    sys::{self, GrB_Descriptor, GrB_Info, GrB_Matrix, GrB_Vector, GxB_Option_Field},
};

/// A `GrB_Matrix` or `GrB_Vector`, with the entry points that differ only by
/// that suffix.
///
/// # Safety
///
/// Every method forwards to the C function of the same name for the handle
/// kind, with the C contract: a live handle and valid out-pointers.
pub unsafe trait RawHandle: Copy {
    fn null() -> Self;

    /// `GrB_*_error`.
    unsafe fn error(
        msg: *mut *const c_char,
        h: Self,
    ) -> GrB_Info;
    /// `GrB_*_wait`.
    unsafe fn wait(
        h: Self,
        mode: c_int,
    ) -> GrB_Info;
    /// `GrB_*_get_INT32`.
    unsafe fn get_i32(
        h: Self,
        value: *mut i32,
        field: c_int,
    ) -> GrB_Info;
    /// `GrB_*_set_INT32`.
    unsafe fn set_i32(
        h: Self,
        value: i32,
        field: c_int,
    ) -> GrB_Info;
    /// `GxB_*_memoryUsage`.
    unsafe fn memory_usage(
        size: *mut usize,
        h: Self,
    ) -> GrB_Info;
    /// `GxB_*_iso`.
    unsafe fn iso(
        iso: *mut bool,
        h: Self,
    ) -> GrB_Info;
}

macro_rules! raw_handle {
    ($raw:ty, $error:ident, $wait:ident, $get:ident, $set:ident, $mem:ident, $iso:ident) => {
        // SAFETY: each method is the C function of that name for `$raw`.
        unsafe impl RawHandle for $raw {
            fn null() -> Self {
                std::ptr::null_mut()
            }

            unsafe fn error(
                msg: *mut *const c_char,
                h: Self,
            ) -> GrB_Info {
                // SAFETY: the caller upholds the C contract.
                unsafe { sys::$error(msg, h) }
            }

            unsafe fn wait(
                h: Self,
                mode: c_int,
            ) -> GrB_Info {
                // SAFETY: the caller upholds the C contract.
                unsafe { sys::$wait(h, mode) }
            }

            unsafe fn get_i32(
                h: Self,
                value: *mut i32,
                field: c_int,
            ) -> GrB_Info {
                // SAFETY: the caller upholds the C contract.
                unsafe { sys::$get(h, value, field) }
            }

            unsafe fn set_i32(
                h: Self,
                value: i32,
                field: c_int,
            ) -> GrB_Info {
                // SAFETY: the caller upholds the C contract.
                unsafe { sys::$set(h, value, field) }
            }

            unsafe fn memory_usage(
                size: *mut usize,
                h: Self,
            ) -> GrB_Info {
                // SAFETY: the caller upholds the C contract.
                unsafe { sys::$mem(size, h) }
            }

            unsafe fn iso(
                iso: *mut bool,
                h: Self,
            ) -> GrB_Info {
                // SAFETY: the caller upholds the C contract.
                unsafe { sys::$iso(iso, h) }
            }
        }
    };
}

raw_handle!(
    GrB_Matrix,
    GrB_Matrix_error,
    GrB_Matrix_wait,
    GrB_Matrix_get_INT32,
    GrB_Matrix_set_INT32,
    GxB_Matrix_memoryUsage,
    GxB_Matrix_iso
);
raw_handle!(
    GrB_Vector,
    GrB_Vector_error,
    GrB_Vector_wait,
    GrB_Vector_get_INT32,
    GrB_Vector_set_INT32,
    GxB_Vector_memoryUsage,
    GxB_Vector_iso
);

/// `Ok` on `GrB_SUCCESS`, else the bare status, for a call that only reads
/// its object. GraphBLAS logs a message into an object only when the call
/// names it as the output, so whatever text a read object carries belongs
/// to an earlier call.
pub fn check_read(info: GrB_Info) -> Result<()> {
    if info == GrB_Info::GrB_SUCCESS {
        Ok(())
    } else {
        Err(Error::from_info(info, String::new()))
    }
}

/// `Ok` on `GrB_SUCCESS`, else an [`Error`] carrying the message GraphBLAS
/// attached to `h`, which must be the call's live output object.
pub fn check<R: RawHandle>(
    info: GrB_Info,
    h: R,
) -> Result<()> {
    if info == GrB_Info::GrB_SUCCESS {
        return Ok(());
    }
    let mut msg = std::ptr::null();
    // SAFETY: `h` is a live handle (the caller's output object) and `msg` a
    // valid out-pointer; GraphBLAS writes a pointer to a NUL-terminated
    // string it owns, which `copy_message` reads before any further call on
    // `h`.
    let _ = unsafe { R::error(&raw mut msg, h) };
    Err(Error::from_info(info, copy_message(msg)))
}

/// Run one C operation on the output `out`: resolve the mask handle and the
/// descriptor, call, and turn the result into a `Result` carrying `out`'s
/// message.
pub fn run<R: RawHandle>(
    out: R,
    mask: &Mask<'_, R>,
    desc: Desc,
    call: impl FnOnce(R, R, GrB_Descriptor) -> GrB_Info,
) -> Result<()> {
    let m = mask.raw(out);
    let d = descriptor(mask.mode(), desc);
    check(call(out, m, d), out)
}

/// Finish pending work.
pub fn wait<R: RawHandle>(
    h: R,
    mode: WaitMode,
) -> Result<()> {
    // SAFETY: a live handle.
    check(unsafe { R::wait(h, mode.raw()) }, h)
}

/// `GrB_get` of an integer field that cannot fail on a live handle.
pub fn get_i32<R: RawHandle>(
    h: R,
    field: GxB_Option_Field,
) -> i32 {
    let mut value = 0;
    // SAFETY: a live handle, a valid out-pointer and an integer field.
    let info = unsafe { R::get_i32(h, &raw mut value, field as c_int) };
    expect_success(info, "GrB_get_INT32");
    value
}

/// `GrB_set` of an integer field.
pub fn set_i32<R: RawHandle>(
    h: R,
    field: GxB_Option_Field,
    value: i32,
) -> Result<()> {
    // SAFETY: a live handle and an integer field.
    check(unsafe { R::set_i32(h, value, field as c_int) }, h)
}

/// Whether the object has pending work (`GxB_WILL_WAIT`).
pub fn is_pending<R: RawHandle>(h: R) -> bool {
    get_i32(h, GxB_Option_Field::GxB_WILL_WAIT) != 0
}

/// Bytes GraphBLAS holds for the object.
pub fn memory_usage<R: RawHandle>(h: R) -> usize {
    let mut size = 0;
    // SAFETY: a live handle and a valid out-pointer.
    let info = unsafe { R::memory_usage(&raw mut size, h) };
    expect_success(info, "GxB_memoryUsage");
    size
}

/// An entry count as a `usize`. The entries are in memory, so their count
/// fits wherever GraphBLAS runs.
pub fn entry_count(n: sys::GrB_Index) -> usize {
    usize::try_from(n).expect("an entry count fits in a usize")
}

/// Whether the object is iso (one shared value).
pub fn is_iso<R: RawHandle>(h: R) -> bool {
    let mut iso = false;
    // SAFETY: a live handle and a valid out-pointer.
    let info = unsafe { R::iso(&raw mut iso, h) };
    expect_success(info, "GxB_iso");
    iso
}

/// The current storage format (`GxB_SPARSITY_STATUS`).
pub fn sparsity<R: RawHandle>(h: R) -> Sparsity {
    Sparsity::from_status(get_i32(h, GxB_Option_Field::GxB_SPARSITY_STATUS))
}

/// Pin or free the storage format (`GxB_SPARSITY_CONTROL`).
pub fn set_sparsity<R: RawHandle>(
    h: R,
    control: Sparsity,
) -> Result<()> {
    set_i32(h, GxB_Option_Field::GxB_SPARSITY_CONTROL, control.raw())
}
