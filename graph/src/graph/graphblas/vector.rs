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

use crate::graph::graphblas::{
    GrB_PLUS_MONOID_UINT64, GrB_Scalar, GrB_Scalar_free, GrB_Scalar_new,
    GrB_Scalar_setElement_BOOL, GrB_UINT64, GrB_Vector_clear, GrB_Vector_reduce_UINT64,
    GrB_Vector_setElement_UINT64, GxB_Vector_build_Scalar,
};

use super::serialization::{Decode, DecodeLimits, Encode, Reader, Writer};
use super::{
    GrB_BOOL, GrB_Info, GrB_Type, GrB_Type_get_String, GrB_Vector, GrB_Vector_free, GrB_Vector_new,
    GrB_Vector_nvals, GrB_Vector_removeElement, GrB_Vector_resize, GrB_Vector_setElement_BOOL,
    GrB_Vector_size, GrB_Vector_wait, GrB_WaitMode, GxB_Iterator, GxB_Iterator_free,
    GxB_Iterator_get_UINT64, GxB_Iterator_new, GxB_MAX_NAME_LEN, GxB_Option_Field,
    GxB_Type_from_name, GxB_Type_size, GxB_Vector_Iterator_attach, GxB_Vector_Iterator_getIndex,
    GxB_Vector_Iterator_next, GxB_Vector_Iterator_seek, GxB_Vector_deserialize, GxB_Vector_load,
    GxB_Vector_serialize, GxB_Vector_unload,
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
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
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
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }

    pub fn wait(&mut self) {
        unsafe {
            let info = GrB_Vector_wait(self.v, GrB_WaitMode::GrB_MATERIALIZE as _);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }

    /// An indicator vector of size `nrows` with an entry at each of
    /// `indices`, which must be ascending, distinct and below `nrows`.
    ///
    /// One `GxB_Vector_build_Scalar`: no values array, an iso vector, and no
    /// pending tuples to sort the way a `set` per index leaves them.
    #[must_use]
    pub fn from_sorted_indices(
        nrows: u64,
        indices: &[u64],
    ) -> Self {
        debug_assert!(indices.windows(2).all(|w| w[0] < w[1]));
        debug_assert!(indices.last().is_none_or(|&i| i < nrows));
        let v = Self::new(nrows);
        if indices.is_empty() {
            return v;
        }
        unsafe {
            let mut scalar: GrB_Scalar = null_mut();
            let mut info = GrB_Scalar_new(&raw mut scalar, GrB_BOOL);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            info = GrB_Scalar_setElement_BOOL(scalar, true);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            info = GxB_Vector_build_Scalar(v.v, indices.as_ptr(), scalar, indices.len() as u64);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            info = GrB_Scalar_free(&raw mut scalar);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        v
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

    /// Serialize as a `GxB_Vector_serialize` blob — the form C's RDB tensor
    /// section uses for a multi-edge pair's id list (`GrB_Vector_new(&V,
    /// GrB_BOOL, GrB_INDEX_MAX)` with `setElement_BOOL(V, true, edge_id)`, so
    /// the edge ids are the vector's *indices*).
    ///
    /// Distinct from `<Vector<bool> as Encode<19>>::encode`, which writes the
    /// unload-to-array form that `Matrix::encode` uses for container payloads.
    pub fn encode_blob(
        &self,
        w: &mut dyn Writer,
    ) {
        unsafe {
            let mut blob: *mut c_void = null_mut();
            let mut blob_size: u64 = 0;

            let info = GxB_Vector_serialize(&raw mut blob, &raw mut blob_size, self.v, null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);

            let blob_slice = std::slice::from_raw_parts(blob.cast::<u8>(), blob_size as usize);
            w.write_buffer(blob_slice);

            let layout = std::alloc::Layout::from_size_align(blob_size as usize, 8).unwrap();
            std::alloc::dealloc(blob.cast::<u8>(), layout);
        }
    }

    /// Inverse of [`Self::encode_blob`].
    pub fn decode_blob(r: &mut dyn Reader) -> Result<Self, String> {
        let blob = r.read_buffer()?;
        deserialize_untrusted(&blob, unsafe { GrB_BOOL }).map(Self::from)
    }
}

/// Bytes in the header of a GraphBLAS serialized blob (`GB_BLOB_HEADER_SIZE`).
const BLOB_HEADER_SIZE: usize = 160;

/// Offsets of the header fields [`check_blob_header`] reads (`GB_deserialize`
/// reads them in this order).
mod blob_at {
    pub const ENCODING: usize = 8;
    pub const VLEN: usize = 16;
    pub const VDIM: usize = 24;
    pub const NVEC: usize = 32;
    pub const NVEC_NONEMPTY: usize = 40;
    pub const NVALS: usize = 48;
    pub const TYPESIZE: usize = 56;
    /// `Cp_len`, `Ch_len`, `Cb_len`, `Ci_len`, `Cx_len`: five `i64`s.
    pub const LENS: usize = 64;
    pub const SPARSITY_ISO_CSC: usize = 116;
    /// `Cp_nblocks`, `Cp_method`, then the same pair for h, b, i, x.
    pub const NBLOCKS: usize = 120;
}

/// Uncompressed bytes a blob may declare beyond a multiple of its own size.
/// Compression makes a declared length cheap to claim and expensive to honour,
/// so it is bounded in terms of what the payload actually carries.
const BLOB_SLACK: u64 = 64 << 20;
/// Most a legitimate array of edge ids compresses by.
const BLOB_MAX_RATIO: u64 = 64;

fn blob_i64(
    blob: &[u8],
    at: usize,
) -> i64 {
    i64::from_le_bytes(blob[at..at + 8].try_into().expect("8 bytes"))
}

/// Refuse a serialized GraphBLAS blob whose header would make `GB_deserialize`
/// allocate far beyond the blob or read past its end.
///
/// `GB_deserialize` bounds-checks the compressed *data* but takes the
/// *declared* sizes at face value: it allocates every array at its declared
/// uncompressed length before decompressing, and indexes the block-size table
/// by a declared block count. Both come straight from an untrusted payload, so
/// they are checked here, against the blob and against the graph's own limits.
fn check_blob_header(blob: &[u8]) -> Result<u64, String> {
    if blob.len() < BLOB_HEADER_SIZE {
        return Err(format!("blob of {} bytes has no header", blob.len()));
    }
    let blob_len = blob.len() as u64;
    if u64::from_le_bytes(blob[..8].try_into().expect("8 bytes")) != blob_len {
        return Err("blob size does not match its header".to_string());
    }
    let field = |at: usize| blob_i64(blob, at);
    let limits = DecodeLimits::current();
    let max_dim = limits.map_or(1 << 60, |l| l.max_dim().min(1 << 60));
    let max_entries = limits.map_or(1 << 60, |l| l.edge_bound.min(1 << 60));
    let (vlen, vdim, nvec, nvals, typesize) = (
        field(blob_at::VLEN),
        field(blob_at::VDIM),
        field(blob_at::NVEC),
        field(blob_at::NVALS),
        field(blob_at::TYPESIZE),
    );
    // An edge-id vector is indexed by edge id and sized to the largest index
    // GraphBLAS allows; it is sparse, so its length costs nothing and is only
    // held to that limit. What it *holds* is bounded by the graph.
    let in_range = |v: i64, max: u64| u64::try_from(v).is_ok_and(|v| v <= max);
    if !in_range(vlen, 1 << 60)
        || !in_range(vdim, 1 << 60)
        || !in_range(nvec, max_dim)
        || !in_range(nvals, max_dim)
        // What a vector holds is edge ids, each at most once: no more than the
        // graph has. Otherwise a small graph's blob could claim millions.
        || !in_range(nvals, max_entries)
    {
        return Err("blob dimensions are out of range".to_string());
    }
    if !(1..=64).contains(&typesize) {
        return Err(format!("blob element size {typesize} is out of range"));
    }

    let budget = BLOB_SLACK.saturating_add(blob_len.saturating_mul(BLOB_MAX_RATIO));
    let mut lens = [0u64; 5];
    let mut declared = 0u64;
    for (k, len) in lens.iter_mut().enumerate() {
        *len = u64::try_from(field(blob_at::LENS + 8 * k))
            .map_err(|_| "blob array length is negative".to_string())?;
        declared = declared.saturating_add(*len);
    }
    if declared > budget {
        return Err(format!(
            "blob declares {declared} bytes of arrays, more than its {blob_len} bytes can hold"
        ));
    }

    // The arrays have to be as long as the counts say. GraphBLAS takes `nvec`
    // and `nvals` from the header and its own consistency check walks the
    // arrays by them, so a blob whose arrays are shorter is read past its end
    // — by the check as much as by anything after it.
    let encoding = u32::from_le_bytes(
        blob[blob_at::ENCODING..blob_at::ENCODING + 4]
            .try_into()
            .expect("4 bytes"),
    );
    let width = |shift: u32| {
        if (encoding >> shift) & 0xF == 1 {
            4u64
        } else {
            8
        }
    };
    let (pw, jw, iw) = (width(12), width(8), width(4));
    let sparsity_iso = i32::from_le_bytes(
        blob[blob_at::SPARSITY_ISO_CSC..blob_at::SPARSITY_ISO_CSC + 4]
            .try_into()
            .expect("4 bytes"),
    );
    let (sparsity, iso) = (sparsity_iso / 4, sparsity_iso & 2 == 2);
    let [cp, ch, cb, ci, cx] = lens;
    let nonempty = field(blob_at::NVEC_NONEMPTY);
    if nonempty < 0 || nonempty > nvec {
        return Err("blob non-empty count is out of range".to_string());
    }
    let (nvec, nvals, vlen, vdim, typesize) = (
        nvec as u64,
        nvals as u64,
        vlen as u64,
        vdim as u64,
        typesize as u64,
    );
    let cx_need = if iso {
        typesize
    } else {
        nvals.saturating_mul(typesize)
    };
    let consistent = match sparsity {
        // hypersparse: pointers, hyperlist, indices
        1 => {
            nvec <= vdim
                && cp >= (nvec + 1).saturating_mul(pw)
                && ch >= nvec.saturating_mul(jw)
                && ci >= nvals.saturating_mul(iw)
                && cx >= cx_need
        }
        // sparse: one pointer per vector, plus one
        2 => {
            nvec == vdim
                && cp >= (nvec + 1).saturating_mul(pw)
                && ci >= nvals.saturating_mul(iw)
                && cx >= cx_need
        }
        // bitmap: a byte per cell
        4 => {
            cb >= vlen.saturating_mul(vdim)
                && cx
                    >= if iso {
                        typesize
                    } else {
                        vlen.saturating_mul(vdim).saturating_mul(typesize)
                    }
        }
        // full: refused outright. The only vectors decoded here are edge-id
        // vectors, whose length is the largest index GraphBLAS allows: no
        // writer can have a full one, and an iso full blob would claim any
        // number of cells for one stored value.
        8 => {
            return Err("a full vector is not a valid edge-id vector".to_string());
        }
        _ => false,
    };
    if !consistent {
        return Err("blob arrays are shorter than its counts say".to_string());
    }

    // The block-size table follows the header: one u64 per block, per array.
    let mut table = 0u64;
    for k in 0..5 {
        let at = blob_at::NBLOCKS + 8 * k;
        let nblocks = i32::from_le_bytes(blob[at..at + 4].try_into().expect("4 bytes"));
        let nblocks = u64::try_from(nblocks).map_err(|_| "blob block count is negative")?;
        table = table.saturating_add(nblocks);
    }
    if (BLOB_HEADER_SIZE as u64).saturating_add(table.saturating_mul(8)) > blob_len {
        return Err("blob block table runs past its end".to_string());
    }
    // Entries the index array can hold: no more may be read from it.
    Ok((if sparsity == 4 { vlen } else { ci / iw }).min(max_entries))
}

/// `GxB_Vector_deserialize` of an untrusted blob, as the type `ty`.
///
/// The header is checked first (see [`check_blob_header`]), and the vector that
/// comes out is held to the index array it was given, so that iterating it
/// cannot leave the arrays. The indices themselves are untrusted.
fn deserialize_untrusted(
    blob: &[u8],
    ty: GrB_Type,
) -> Result<GrB_Vector, String> {
    let capacity =
        check_blob_header(blob).map_err(|e| format!("GxB_Vector_deserialize refused: {e}"))?;
    unsafe {
        let mut v: MaybeUninit<GrB_Vector> = MaybeUninit::uninit();
        let info = GxB_Vector_deserialize(
            v.as_mut_ptr(),
            ty,
            blob.as_ptr().cast(),
            blob.len() as u64,
            null_mut(),
        );
        // The blob comes from an RDB / `GRAPH.RESTORE` payload, so a
        // failure here is bad input, not a broken invariant: refuse the
        // load instead of taking the server down. Deserialize frees
        // whatever it allocated before returning an error.
        if info != GrB_Info::GrB_SUCCESS {
            return Err(format!("GxB_Vector_deserialize failed: {info:?}"));
        }
        let mut v = v.assume_init();
        // The entry count is the pointer array's last word, which the blob sets,
        // and iterating reads that many indices. Held to the index array that was
        // allocated, an iteration cannot leave it; what the indices say is the
        // caller's to check, as it must for any index it reads.
        //
        // GraphBLAS has a full consistency check (`GxB_Vector_fprint` at
        // `GxB_SILENT`), but it costs 10 to 35 ns an entry, and added some 8% to
        // loading a graph of two million edges (measured, see the PR); this
        // check is constant-time.
        let mut nvals = 0u64;
        let info = GrB_Vector_nvals(&raw mut nvals, v);
        if info != GrB_Info::GrB_SUCCESS || nvals > capacity {
            GrB_Vector_free(addr_of_mut!(v));
            return Err(format!(
                "GxB_Vector_deserialize: corrupt vector: {nvals} entries, room for {capacity}"
            ));
        }
        Ok(v)
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
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);

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
        deserialize_untrusted(&blob, unsafe { GrB_UINT64 }).map(Self::from)
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
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);

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
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

/// One component of a serialized container — an array, its type name and its
/// counts — read and checked for consistency with itself, but not yet loaded
/// into a GraphBLAS vector. Split out so a matrix decoder can examine the
/// arrays of all its components before any of them is handed to GraphBLAS.
pub(super) struct Component {
    pub(super) arr_data: Vec<u8>,
    pub(super) type_name: Vec<u8>,
    /// Entries in use. The array may be allocated larger than they need
    /// (`arr_data.len()` is the allocation's size), so this, not the length
    /// of `arr_data`, is how many there are.
    pub(super) n_entries: u64,
    n_bytes: u64,
}

impl Component {
    pub(super) fn read(r: &mut dyn Reader) -> Result<Self, String> {
        let arr_data = r.read_buffer()?;
        let type_name = r.read_buffer()?;
        let n_entries = r.read_unsigned()?;
        let n_bytes = r.read_unsigned()?;
        // `handling` says who owns `arr_data` and how it is to be freed; it
        // is meaningless to a reader, which copies the bytes into a buffer of
        // its own below. Taking the payload's word could hand GraphBLAS a
        // read-only or foreign-arena buffer it would later free or write.
        let _ = r.read_signed()?;

        // `GRAPH.RESTORE` feeds an entirely attacker-controlled payload here, so
        // the independently decoded `n_bytes` and `type_name` must be validated
        // before they drive an allocation, a raw copy, or an FFI C-string read.
        //
        // The encoder always writes `arr_data` with exactly `n_bytes` bytes
        // (see `encode` above). If `n_bytes` exceeds the buffer length the
        // `copy_nonoverlapping` below would read past the end of `arr_data`
        // (heap over-read / crash), so reject any mismatch.
        if n_bytes as usize != arr_data.len() {
            return Err(format!(
                "Vector decode: declared byte length {n_bytes} does not match buffer length {}",
                arr_data.len()
            ));
        }
        // `GxB_Type_from_name` reads `type_name` as a NUL-terminated C string.
        // The encoder always writes the type name followed by exactly one
        // trailing NUL (see `encode`), so the only NUL must be the final byte.
        // Requiring the terminator at the end (and rejecting interior NULs)
        // both matches the encoder and guarantees the C-string read stays in
        // bounds.
        if type_name.last() != Some(&0) || type_name[..type_name.len() - 1].contains(&0) {
            return Err(String::from(
                "Vector decode: type name is not NUL-terminated",
            ));
        }
        Ok(Self {
            arr_data,
            type_name,
            n_entries,
            n_bytes,
        })
    }

    /// The element type's name, without its NUL.
    pub(super) fn type_name(&self) -> &[u8] {
        &self.type_name[..self.type_name.len() - 1]
    }

    pub(super) fn load(self) -> Result<Vector<bool>, String> {
        let Self {
            arr_data,
            type_name,
            n_entries,
            n_bytes,
        } = self;
        let handling = 0;
        unsafe {
            let mut type_: MaybeUninit<GrB_Type> = MaybeUninit::uninit();
            let info = GxB_Type_from_name(type_.as_mut_ptr(), type_name.as_ptr().cast());
            if info != GrB_Info::GrB_SUCCESS {
                return Err(format!(
                    "Vector decode: GxB_Type_from_name failed: {info:?}"
                ));
            }
            let type_ = type_.assume_init();

            // `GxB_Vector_load` checks `n_bytes >= n_entries * type size` in
            // 64-bit arithmetic, which a huge `n_entries` wraps past.
            let mut type_size = 0usize;
            if GxB_Type_size(&raw mut type_size, type_) != GrB_Info::GrB_SUCCESS
                || type_size == 0
                || n_entries
                    .checked_mul(type_size as u64)
                    .is_none_or(|need| need > n_bytes)
            {
                return Err(format!(
                    "Vector decode: {n_entries} entries do not fit in {n_bytes} bytes"
                ));
            }

            let mut v: MaybeUninit<GrB_Vector> = MaybeUninit::uninit();
            let info = GrB_Vector_new(v.as_mut_ptr(), type_, 0);
            if info != GrB_Info::GrB_SUCCESS {
                return Err(format!("Vector decode: GrB_Vector_new failed: {info:?}"));
            }
            let mut v = v.assume_init();

            // Allocate the backing buffer that `GxB_Vector_load` adopts on
            // success (`n_bytes == arr_data.len()` was validated above). The
            // payload is attacker-controlled, so handle layout and allocation
            // failures explicitly instead of unwrapping or copying into a null
            // pointer, and free the freshly created vector on every error path.
            let layout = if n_bytes > 0 {
                match std::alloc::Layout::from_size_align(n_bytes as usize, 8) {
                    Ok(layout) => Some(layout),
                    Err(e) => {
                        GrB_Vector_free(addr_of_mut!(v));
                        return Err(format!("Vector decode: invalid buffer layout: {e}"));
                    }
                }
            } else {
                None
            };

            let mut arr_ptr: *mut c_void = if let Some(layout) = layout {
                let ptr = std::alloc::alloc(layout);
                if ptr.is_null() {
                    GrB_Vector_free(addr_of_mut!(v));
                    return Err(String::from("Vector decode: buffer allocation failed"));
                }
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
            // A `debug_assert` here would be compiled out in release, letting a
            // malformed payload return a half-initialized vector and leak the
            // buffer. On failure `GxB_Vector_load` does not adopt `arr_ptr`, so
            // we still own it: free the buffer and the vector, then return Err.
            if info != GrB_Info::GrB_SUCCESS {
                if let Some(layout) = layout
                    && !arr_ptr.is_null()
                {
                    std::alloc::dealloc(arr_ptr.cast(), layout);
                }
                GrB_Vector_free(addr_of_mut!(v));
                return Err(format!("Vector decode: GxB_Vector_load failed: {info:?}"));
            }

            Ok(Vector::from(v))
        }
    }
}

impl Decode<19> for Vector<bool> {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        Component::read(r)?.load()
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
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }

    #[must_use]
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(&self) -> Iter<u64> {
        Iter::new(self)
    }

    #[must_use]
    pub const fn ptr(&self) -> GrB_Vector {
        self.v
    }

    /// The sum of the stored values (`PLUS` monoid; 0 when empty).
    #[must_use]
    pub fn sum(&self) -> u64 {
        let mut sum = 0u64;
        unsafe {
            let info = GrB_Vector_reduce_UINT64(
                &raw mut sum,
                null_mut(),
                GrB_PLUS_MONOID_UINT64,
                self.v,
                null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        sum
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
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
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
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
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
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
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
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
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

#[cfg(test)]
mod decode_tests {
    use super::*;
    use std::collections::VecDeque;

    /// Minimal in-memory [`Reader`] that replays scripted values in call order.
    /// It drives `Vector::<bool>::decode`'s input-validation paths without a live
    /// GraphBLAS context: every rejection path returns before any FFI call.
    #[derive(Default)]
    struct MockReader {
        buffers: VecDeque<Vec<u8>>,
        unsigned: VecDeque<u64>,
        signed: VecDeque<i64>,
    }

    impl Reader for MockReader {
        fn read_unsigned(&mut self) -> Result<u64, String> {
            self.unsigned
                .pop_front()
                .ok_or_else(|| "mock: no more unsigned values".to_string())
        }
        fn read_signed(&mut self) -> Result<i64, String> {
            self.signed
                .pop_front()
                .ok_or_else(|| "mock: no more signed values".to_string())
        }
        fn read_double(&mut self) -> Result<f64, String> {
            Err("mock: read_double unused".to_string())
        }
        fn read_buffer(&mut self) -> Result<Vec<u8>, String> {
            self.buffers
                .pop_front()
                .ok_or_else(|| "mock: no more buffers".to_string())
        }
    }

    // `decode` reads, in order: arr_data, type_name (buffers); n_entries,
    // n_bytes (unsigned); handling (signed).
    fn reader(
        arr_data: Vec<u8>,
        type_name: Vec<u8>,
        n_entries: u64,
        n_bytes: u64,
        handling: i64,
    ) -> MockReader {
        MockReader {
            buffers: VecDeque::from(vec![arr_data, type_name]),
            unsigned: VecDeque::from(vec![n_entries, n_bytes]),
            signed: VecDeque::from(vec![handling]),
        }
    }

    /// The blob of a multi-edge pair comes from an RDB / `GRAPH.RESTORE`
    /// payload: a corrupt one must fail the decode, not panic.
    #[test]
    fn decode_blob_rejects_a_corrupt_blob() {
        crate::graph::graphblas::test_init::ensure_init();
        let mut v = Vector::<bool>::new(64);
        v.set(3, true);
        v.set(7, true);
        let mut blob = Vec::new();
        v.encode_blob(&mut BlobWriter(&mut blob));
        blob.truncate(8);
        let mut r = MockReader {
            buffers: VecDeque::from(vec![blob]),
            ..MockReader::default()
        };
        let err = Vector::<bool>::decode_blob(&mut r)
            .err()
            .expect("truncated blob accepted");
        assert!(
            err.contains("GxB_Vector_deserialize"),
            "unexpected error: {err}"
        );
    }

    /// A well-formed blob of the wrong type is refused too: `decode_blob` is
    /// read as a BOOL vector and `Vector<u64>::decode` as a UINT64 one.
    #[test]
    fn decode_blob_rejects_a_blob_of_another_type() {
        crate::graph::graphblas::test_init::ensure_init();
        let mut v = Vector::<u64>::new(64);
        v.set(3, 30);
        let mut blob = Vec::new();
        v.encode(&mut BlobWriter(&mut blob));
        let mut r = MockReader {
            buffers: VecDeque::from(vec![blob.clone()]),
            ..MockReader::default()
        };
        assert!(Vector::<bool>::decode_blob(&mut r).is_err());
        let mut r = MockReader {
            buffers: VecDeque::from(vec![blob]),
            ..MockReader::default()
        };
        assert!(<Vector<u64> as Decode<19>>::decode(&mut r).is_ok());
    }

    /// Captures the single buffer an encoder writes.
    struct BlobWriter<'a>(&'a mut Vec<u8>);

    impl Writer for BlobWriter<'_> {
        fn write_unsigned(
            &mut self,
            _: u64,
        ) {
        }
        fn write_signed(
            &mut self,
            _: i64,
        ) {
        }
        fn write_double(
            &mut self,
            _: f64,
        ) {
        }
        fn write_buffer(
            &mut self,
            data: &[u8],
        ) {
            self.0.extend_from_slice(data);
        }
    }

    #[test]
    fn decode_rejects_n_bytes_buffer_length_mismatch() {
        // arr_data is 3 bytes but the payload declares n_bytes = 999: copying
        // n_bytes would over-read the buffer, so decode must reject it.
        let mut r = reader(vec![1, 2, 3], b"bool\0".to_vec(), 0, 999, 0);
        match <Vector<bool> as Decode<19>>::decode(&mut r) {
            Err(err) => assert!(
                err.contains("does not match buffer length"),
                "unexpected error: {err}"
            ),
            Ok(_) => panic!("expected decode to reject n_bytes/buffer mismatch"),
        }
    }

    #[test]
    fn decode_rejects_type_name_without_trailing_nul() {
        // No NUL anywhere: the C-string read would run off the end of the buffer.
        let mut r = reader(Vec::new(), b"Int".to_vec(), 0, 0, 0);
        match <Vector<bool> as Decode<19>>::decode(&mut r) {
            Err(err) => assert!(err.contains("NUL-terminated"), "unexpected error: {err}"),
            Ok(_) => panic!("expected decode to reject a type name without a trailing NUL"),
        }
    }

    #[test]
    fn decode_rejects_type_name_with_only_interior_nul() {
        // NUL present but not the final byte: rejected so the buffer matches what
        // the encoder writes (name followed by a single trailing NUL).
        let mut r = reader(Vec::new(), vec![b'I', 0, b'X'], 0, 0, 0);
        match <Vector<bool> as Decode<19>>::decode(&mut r) {
            Err(err) => assert!(err.contains("NUL-terminated"), "unexpected error: {err}"),
            Ok(_) => panic!("expected decode to reject a type name with an interior-only NUL"),
        }
    }

    /// `GxB_Vector_load` checks `n_bytes >= n_entries * size` in 64-bit
    /// arithmetic: 2^61 eight-byte entries wrap to zero bytes and pass.
    #[test]
    fn decode_rejects_an_entry_count_that_overflows_its_size() {
        crate::graph::graphblas::test_init::ensure_init();
        let mut r = reader(vec![0; 8], b"uint64_t\0".to_vec(), 1 << 61, 8, 0);
        match <Vector<bool> as Decode<19>>::decode(&mut r) {
            Err(err) => assert!(err.contains("do not fit"), "unexpected error: {err}"),
            Ok(_) => panic!("an overflowing entry count was accepted"),
        }
    }

    fn blob_of_a_vector() -> Vec<u8> {
        crate::graph::graphblas::test_init::ensure_init();
        let mut v = Vector::<bool>::new(64);
        v.set(3, true);
        v.set(7, true);
        let mut blob = Vec::new();
        v.encode_blob(&mut BlobWriter(&mut blob));
        blob
    }

    fn decode_blob_of(blob: Vec<u8>) -> Result<Vector<bool>, String> {
        let mut r = MockReader {
            buffers: VecDeque::from(vec![blob]),
            ..MockReader::default()
        };
        Vector::<bool>::decode_blob(&mut r)
    }

    #[test]
    fn decode_blob_accepts_what_encode_blob_wrote() {
        assert!(decode_blob_of(blob_of_a_vector()).is_ok());
    }

    /// `GB_deserialize` allocates each array at its declared length before it
    /// decompresses, and indexes the block-size table by a declared block
    /// count. Both are the payload's to set, so the header is checked first.
    #[test]
    fn decode_blob_rejects_a_header_that_lies_about_sizes() {
        let patch = |at: usize, bytes: &[u8]| {
            let mut blob = blob_of_a_vector();
            blob[at..at + bytes.len()].copy_from_slice(bytes);
            decode_blob_of(blob)
                .err()
                .expect("a lying header was accepted")
        };
        // Cx_len: terabytes of values from a few hundred bytes.
        let err = patch(super::blob_at::LENS + 32, &(1i64 << 40).to_le_bytes());
        assert!(err.contains("can hold"), "{err}");
        // Cx_len negative.
        let err = patch(super::blob_at::LENS + 32, &(-1i64).to_le_bytes());
        assert!(err.contains("negative"), "{err}");
        // nvals out of range.
        let err = patch(super::blob_at::NVALS, &i64::MAX.to_le_bytes());
        assert!(err.contains("out of range"), "{err}");
        // A block table that runs past the end of the blob.
        let err = patch(super::blob_at::NBLOCKS, &(1i32 << 30).to_le_bytes());
        assert!(err.contains("block table"), "{err}");
        // The size in the header disagreeing with the blob's.
        let err = patch(0, &1u64.to_le_bytes());
        assert!(err.contains("size"), "{err}");
    }

    /// A vector holds edge ids, each once, so a blob for a graph with two edge ids
    /// cannot carry more than two entries (the entry count was only held to the
    /// matrix dimension allowance, which is in the billions).
    #[test]
    fn decode_blob_rejects_more_entries_than_the_graph_has_edges() {
        let _limits = DecodeLimits {
            node_bound: 2,
            edge_bound: 2,
        }
        .scope();
        // The vector of the fixture holds two entries: it fits.
        assert!(decode_blob_of(blob_of_a_vector()).is_ok());
        let mut blob = blob_of_a_vector();
        blob[super::blob_at::NVALS..super::blob_at::NVALS + 8].copy_from_slice(&3i64.to_le_bytes());
        let err = decode_blob_of(blob)
            .err()
            .expect("three entries in a graph of two edges were accepted");
        assert!(err.contains("out of range"), "{err}");
    }

    /// A full iso blob stores one value however many cells it claims, so its
    /// own arrays cannot vouch for its dimensions; and no writer produces a full
    /// edge-id vector, so the format is refused at any size, with or without
    /// decode limits in force.
    #[test]
    fn decode_blob_rejects_a_full_iso_vector_claiming_huge_dimensions() {
        let mut blob = blob_of_a_vector();
        let mut put = |at: usize, bytes: &[u8]| blob[at..at + bytes.len()].copy_from_slice(bytes);
        // Within the matrix dimension allowance, but far past the graph's ids.
        put(super::blob_at::VLEN, &(1i64 << 31).to_le_bytes());
        put(super::blob_at::VDIM, &1i64.to_le_bytes());
        put(super::blob_at::NVEC, &1i64.to_le_bytes());
        put(super::blob_at::NVEC_NONEMPTY, &1i64.to_le_bytes());
        put(super::blob_at::NVALS, &1i64.to_le_bytes());
        // full (8) << 2, iso
        put(super::blob_at::SPARSITY_ISO_CSC, &34i32.to_le_bytes());
        let _limits = DecodeLimits {
            node_bound: 2,
            edge_bound: 2,
        }
        .scope();
        let err = decode_blob_of(blob)
            .err()
            .expect("a full vector of 2^31 cells was accepted");
        assert!(err.contains("full vector"), "{err}");
    }

    /// Every byte of a blob, overwritten in turn, must decode or be refused:
    /// never crash. A header that claimed more entries than its arrays held
    /// sent GraphBLAS's own consistency check off the end of them.
    #[test]
    fn decode_blob_survives_any_single_byte_corruption() {
        crate::graph::graphblas::test_init::ensure_init();
        let mut blobs = Vec::new();
        // sparse, hypersparse (huge dimension) and dense vectors
        for (size, set) in [
            (64u64, vec![3u64, 7]),
            (1 << 40, vec![5, 1 << 30, 1 << 39]),
            (64, (0..64).collect()),
        ] {
            let mut v = Vector::<bool>::new(size);
            for i in set {
                v.set(i, true);
            }
            let mut blob = Vec::new();
            v.encode_blob(&mut BlobWriter(&mut blob));
            blobs.push(blob);
        }
        let mut refused = 0;
        for blob in &blobs {
            for at in 0..blob.len() {
                for value in [0x00, 0x01, 0x02, 0x7F, 0x80, 0x94, 0xFE, 0xFF] {
                    let mut bad = blob.clone();
                    bad[at] = value;
                    match decode_blob_of(bad) {
                        Err(_) => refused += 1,
                        // What was accepted is iterated, as `Tensor::decode`
                        // does: it must stay inside the arrays whatever the
                        // indices say (the values read are the caller's to
                        // range-check).
                        Ok(v) => {
                            let _ = v.iter().take(1 << 20).count();
                        }
                    }
                }
            }
        }
        assert!(refused > 0, "no corruption was detected");
    }
}
