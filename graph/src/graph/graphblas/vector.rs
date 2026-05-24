//! Safe Rust wrapper around GraphBLAS boolean sparse vectors.
//!
//! This module provides [`Vector<T>`], which wraps `GrB_Vector` from the
//! SuiteSparse:GraphBLAS C library. Currently only `Vector<bool>` is
//! implemented, used to represent sets of node IDs efficiently.
//!
//! ## Use Cases
//!
//! - **Label membership**: each label maps to a boolean vector where index `i`
//!   is `true` if node `i` has that label.
//! - **Intermediate results**: graph algorithms use vectors to track visited
//!   nodes, frontier sets, etc.
//! - **Filtering**: a vector can mask matrix operations to restrict results
//!   to a subset of nodes.
//!
//! ## Sparse Storage
//!
//! Like matrices, only non-zero entries are stored. A vector representing
//! 3 nodes out of a million only uses memory for those 3 entries.
//!
//! ```text
//!   GrB_Vector (boolean, sparse)
//!
//!   Logical view:        Stored entries:
//!   Index: 0 1 2 3 4       1 -> true
//!          . T . T .        3 -> true
//!
//!   size = 5, nvals = 2
//! ```
//!
//! ## Iterator
//!
//! [`Iter<bool>`] traverses all set entries, yielding their indices.
//! It uses `GxB_Vector_Iterator` internally and is consumed once.

use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    os::raw::c_void,
    ptr::{addr_of_mut, null_mut},
};

use crate::graph::graphblas::{GrB_UINT64, GrB_Vector_clear, GrB_Vector_setElement_UINT64};

use super::serialization::{Decode, Encode, Reader, Writer};
use super::{
    GrB_BOOL, GrB_Info, GrB_Type, GrB_Type_get_String, GrB_Vector, GrB_Vector_free, GrB_Vector_new,
    GrB_Vector_removeElement, GrB_Vector_resize, GrB_Vector_setElement_BOOL, GrB_Vector_size,
    GrB_Vector_wait, GrB_WaitMode, GxB_Iterator, GxB_Iterator_free, GxB_Iterator_get_UINT64,
    GxB_Iterator_new, GxB_MAX_NAME_LEN, GxB_Option_Field, GxB_Type_from_name,
    GxB_Vector_Iterator_attach, GxB_Vector_Iterator_getIndex, GxB_Vector_Iterator_next,
    GxB_Vector_Iterator_seek, GxB_Vector_deserialize, GxB_Vector_load, GxB_Vector_serialize,
    GxB_Vector_unload,
};

/// A sparse vector backed by GraphBLAS.
///
/// Generic over element type T, though currently only bool is implemented.
/// The vector automatically frees its GraphBLAS resources on drop.
pub struct Vector<T> {
    v: GrB_Vector,
    phantom: PhantomData<T>,
}

impl<T> Drop for Vector<T> {
    fn drop(&mut self) {
        unsafe {
            let info = GrB_Vector_free(addr_of_mut!(self.v));
            // debug_assert in Drop: panicking while unwinding aborts the
            // process. A GrB_*_free failure is logically a leak, not state
            // corruption — surface it in debug, swallow in release.
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

impl From<GrB_Vector> for Vector<bool> {
    fn from(v: GrB_Vector) -> Self {
        Self {
            v,
            phantom: PhantomData,
        }
    }
}

impl From<GrB_Vector> for Vector<u64> {
    fn from(v: GrB_Vector) -> Self {
        Self {
            v,
            phantom: PhantomData,
        }
    }
}

impl<T> Vector<T> {
    pub fn clear(&mut self) {
        unsafe {
            let info = GrB_Vector_clear(self.v);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

impl Vector<bool> {
    pub fn new(nrows: u64) -> Self {
        unsafe {
            let mut v: MaybeUninit<GrB_Vector> = MaybeUninit::uninit();
            let info = GrB_Vector_new(v.as_mut_ptr(), GrB_BOOL, nrows);
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GrB_Vector_new failed: {info:?}"
            );
            Self {
                v: v.assume_init(),
                phantom: PhantomData,
            }
        }
    }

    pub fn set(
        &mut self,
        i: u64,
        value: bool,
    ) {
        unsafe {
            let info = GrB_Vector_setElement_BOOL(self.v, value, i);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }

    pub fn wait(&mut self) {
        unsafe {
            let info = GrB_Vector_wait(self.v, GrB_WaitMode::GrB_MATERIALIZE as _);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }

    #[must_use]
    pub const fn ptr(&self) -> GrB_Vector {
        self.v
    }

    #[must_use]
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(&self) -> Iter<bool> {
        Iter::new(self)
    }
}

impl Encode<19> for Vector<u64> {
    fn encode(
        &self,
        w: &mut dyn Writer,
    ) {
        unsafe {
            let mut blob: *mut c_void = null_mut();
            let mut blob_size: u64 = 0;

            let info = GxB_Vector_serialize(&raw mut blob, &raw mut blob_size, self.v, null_mut());
            assert_eq!(info, GrB_Info::GrB_SUCCESS);

            let blob_slice = std::slice::from_raw_parts(blob.cast::<u8>(), blob_size as usize);
            w.write_buffer(blob_slice);

            let layout = std::alloc::Layout::from_size_align(blob_size as usize, 8).unwrap();
            std::alloc::dealloc(blob.cast::<u8>(), layout);
        }
    }
}

impl Decode<19> for Vector<u64> {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        let blob = r.read_buffer()?;
        unsafe {
            let mut v: MaybeUninit<GrB_Vector> = MaybeUninit::uninit();
            let info = GxB_Vector_deserialize(
                v.as_mut_ptr(),
                null_mut(),
                blob.as_ptr().cast(),
                blob.len() as u64,
                null_mut(),
            );
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GxB_Vector_deserialize failed: {info:?}"
            );
            Ok(Self::from(v.assume_init()))
        }
    }
}

impl Encode<19> for Vector<bool> {
    fn encode(
        &self,
        w: &mut dyn Writer,
    ) {
        unsafe {
            let mut arr: *mut c_void = null_mut();
            let mut type_: MaybeUninit<GrB_Type> = MaybeUninit::uninit();
            let mut n_entries: u64 = 0;
            let mut n_bytes: u64 = 0;
            let mut handling: i32 = 0;

            let info = GxB_Vector_unload(
                self.v,
                &raw mut arr,
                type_.as_mut_ptr(),
                &raw mut n_entries,
                &raw mut n_bytes,
                &raw mut handling,
                null_mut(),
            );
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GxB_Vector_unload failed: {info:?}"
            );

            let type_ = type_.assume_init();

            let mut t_name = [0u8; GxB_MAX_NAME_LEN as usize];
            let info = GrB_Type_get_String(
                type_,
                t_name.as_mut_ptr().cast(),
                GxB_Option_Field::GrB_NAME as _,
            );
            assert_eq!(info, GrB_Info::GrB_SUCCESS);

            let t_name_len = t_name
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(GxB_MAX_NAME_LEN as usize)
                + 1;

            let arr_slice = if n_bytes > 0 {
                std::slice::from_raw_parts(arr.cast::<u8>(), n_bytes as usize)
            } else {
                &[]
            };

            w.write_buffer(arr_slice);
            w.write_buffer(&t_name[..t_name_len]);
            w.write_unsigned(n_entries);
            w.write_unsigned(n_bytes);
            w.write_signed(handling as i64);

            // Reload the vector so it remains usable
            let info = GxB_Vector_load(
                self.v,
                &raw mut arr,
                type_,
                n_entries,
                n_bytes,
                handling,
                null_mut(),
            );
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

impl Decode<19> for Vector<bool> {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        let arr_data = r.read_buffer()?;
        let type_name = r.read_buffer()?;
        let n_entries = r.read_unsigned()?;
        let n_bytes = r.read_unsigned()?;
        let handling = r.read_signed()? as i32;

        unsafe {
            let mut type_: MaybeUninit<GrB_Type> = MaybeUninit::uninit();
            let info = GxB_Type_from_name(type_.as_mut_ptr(), type_name.as_ptr().cast());
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GxB_Type_from_name failed: {info:?}"
            );
            let type_ = type_.assume_init();

            let mut v: MaybeUninit<GrB_Vector> = MaybeUninit::uninit();
            let info = GrB_Vector_new(v.as_mut_ptr(), type_, 0);
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GrB_Vector_new failed: {info:?}"
            );
            let v = v.assume_init();

            let mut arr_ptr: *mut c_void = if n_bytes > 0 {
                let layout = std::alloc::Layout::from_size_align(n_bytes as usize, 8).unwrap();
                let ptr = std::alloc::alloc(layout);
                std::ptr::copy_nonoverlapping(arr_data.as_ptr(), ptr, n_bytes as usize);
                ptr.cast()
            } else {
                null_mut()
            };

            let info = GxB_Vector_load(
                v,
                &raw mut arr_ptr,
                type_,
                n_entries,
                n_bytes,
                handling,
                null_mut(),
            );
            assert_eq!(info, GrB_Info::GrB_SUCCESS);

            Ok(Self::from(v))
        }
    }
}

impl Vector<u64> {
    pub fn new(nrows: u64) -> Self {
        unsafe {
            let mut v: MaybeUninit<GrB_Vector> = MaybeUninit::uninit();
            let info = GrB_Vector_new(v.as_mut_ptr(), GrB_UINT64, nrows);
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GrB_Vector_new failed: {info:?}"
            );
            Self {
                v: v.assume_init(),
                phantom: PhantomData,
            }
        }
    }

    pub fn set(
        &mut self,
        i: u64,
        value: u64,
    ) {
        unsafe {
            let info = GrB_Vector_setElement_UINT64(self.v, value, i);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }

    #[must_use]
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(&self) -> Iter<u64> {
        Iter::new(self)
    }
}

pub trait Size<T> {
    fn size(&self) -> u64;
    fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    );
}

pub trait Set<T> {
    fn set(
        &mut self,
        i: u64,
        value: T,
    );
}

pub trait Remove<T> {
    fn remove(
        &mut self,
        i: u64,
    );
}

impl Size<bool> for Vector<bool> {
    fn size(&self) -> u64 {
        unsafe {
            let mut size: u64 = 0;
            let info = GrB_Vector_size(&raw mut size, self.v);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
            size
        }
    }

    fn resize(
        &mut self,
        nrows: u64,
        _ncols: u64,
    ) {
        unsafe {
            let info = GrB_Vector_resize(self.v, nrows);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

impl Set<bool> for Vector<bool> {
    fn set(
        &mut self,
        i: u64,
        value: bool,
    ) {
        unsafe {
            let info = GrB_Vector_setElement_BOOL(self.v, value, i);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

impl Remove<bool> for Vector<bool> {
    fn remove(
        &mut self,
        i: u64,
    ) {
        unsafe {
            let info = GrB_Vector_removeElement(self.v, i);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

pub struct Iter<T> {
    inner: GxB_Iterator,
    depleted: bool,
    phantom: PhantomData<T>,
}

impl<T> Drop for Iter<T> {
    fn drop(&mut self) {
        unsafe {
            let info = GxB_Iterator_free(addr_of_mut!(self.inner));
            // debug_assert: don't panic in Drop (see Vector::drop above).
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

impl<T> Iter<T> {
    #[must_use]
    pub fn new(v: &Vector<T>) -> Self {
        unsafe {
            let mut iter = MaybeUninit::uninit();
            let info = GxB_Iterator_new(iter.as_mut_ptr());
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GxB_Iterator_new failed: {info:?}"
            );
            let iter = iter.assume_init();
            let info = GxB_Vector_Iterator_attach(iter, v.v, null_mut());
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GxB_Vector_Iterator_attach failed: {info:?}"
            );
            let info = GxB_Vector_Iterator_seek(iter, 0);
            Self {
                inner: iter,
                depleted: info == GrB_Info::GxB_EXHAUSTED,
                phantom: PhantomData,
            }
        }
    }
}

impl Iterator for Iter<bool> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.depleted {
            return None;
        }
        unsafe {
            let row = GxB_Vector_Iterator_getIndex(self.inner);
            self.depleted = GxB_Vector_Iterator_next(self.inner) == GrB_Info::GxB_EXHAUSTED;
            Some(row)
        }
    }
}

impl Iterator for Iter<u64> {
    type Item = (u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.depleted {
            return None;
        }
        unsafe {
            let idx = GxB_Vector_Iterator_getIndex(self.inner);
            let val = GxB_Iterator_get_UINT64(self.inner);
            self.depleted = GxB_Vector_Iterator_next(self.inner) == GrB_Info::GxB_EXHAUSTED;
            Some((idx, val))
        }
    }
}
