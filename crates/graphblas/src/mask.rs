//! The mask and the descriptor, as data.
//!
//! GraphBLAS spreads one decision over two C arguments: the mask handle, and
//! the `GrB_MASK_STRUCTURE` / `GrB_MASK_COMP` bits of the descriptor. Leaving
//! the bits out gives a *valued* mask, which reads the mask's stored values
//! and treats a stored `false` as "not in the mask". Here the two travel
//! together: [`Mask`] names its mode, and the descriptor bits are derived
//! from it inside the crate.

use std::{marker::PhantomData, ptr::null_mut};

use crate::{
    handle::RawHandle,
    matrix::Matrix,
    operand::Output,
    sealed::{Handle, Token},
    sys::{self, GrB_Descriptor, GrB_Matrix},
    types::{BuiltinType, GrbType},
};

/// A mask source: a borrowed handle, or the output itself (`C<C> = ...`).
/// Produced only by [`AsMask::mask_ref`]; not part of the API.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct MaskRef<'a, R> {
    source: Source<R>,
    _life: PhantomData<&'a ()>,
}

#[derive(Clone, Copy)]
enum Source<R> {
    Handle(R),
    Output,
}

impl<R> MaskRef<'_, R> {
    const fn handle(handle: R) -> Self {
        Self {
            source: Source::Handle(handle),
            _life: PhantomData,
        }
    }
}

/// What a *structural* mask can be made from: a borrowed matrix of
/// any element type, [`Output`], or [`AsBool`] of either. Sealed.
pub trait AsMask<'a, R>: Handle {
    #[doc(hidden)]
    fn mask_ref(
        self,
        _: Token,
    ) -> MaskRef<'a, R>;
}

impl<'a, T: GrbType> AsMask<'a, GrB_Matrix> for &'a Matrix<T> {
    fn mask_ref(
        self,
        _: Token,
    ) -> MaskRef<'a, GrB_Matrix> {
        MaskRef::handle(self.as_raw())
    }
}

impl<R> AsMask<'_, R> for Output {
    fn mask_ref(
        self,
        _: Token,
    ) -> MaskRef<'static, R> {
        MaskRef {
            source: Source::Output,
            _life: PhantomData,
        }
    }
}

/// Use a non-`bool` matrix as a *valued* mask, letting GraphBLAS
/// read each stored value as `value != 0`.
///
/// Does no work: it only records, at the call site, that the cast was asked
/// for. Built-in element types only; GraphBLAS cannot cast a user-defined
/// type to `bool`.
///
/// ```ignore
/// mxm(&mut c, Mask::valued(AsBool(&counts)), None, &PLUS_TIMES_UINT64, &a, &b, Desc::NONE)?;
/// ```
#[derive(Clone, Copy, Debug)]
pub struct AsBool<M>(pub M);

impl<M> Handle for AsBool<M> {}

impl<'a, T: BuiltinType> AsMask<'a, GrB_Matrix> for AsBool<&'a Matrix<T>> {
    fn mask_ref(
        self,
        _: Token,
    ) -> MaskRef<'a, GrB_Matrix> {
        self.0.mask_ref(Token)
    }
}

/// What a *valued* mask can be made from: a `bool` matrix, or
/// [`AsBool`] of a built-in-typed one. Sealed.
///
/// GraphBLAS would cast any built-in type to `bool`; that is an implicit
/// cast, which this crate refuses everywhere, so a non-`bool` valued mask is
/// spelled `AsBool(&m)`. A structural mask takes any element type and needs
/// no such bound. [`Output`] is not a valued mask either, since its element
/// type is not known here.
pub trait ValuedMask<'a, R>: AsMask<'a, R> {}
impl<'a> ValuedMask<'a, GrB_Matrix> for &'a Matrix<bool> {}
impl<'a, T: BuiltinType> ValuedMask<'a, GrB_Matrix> for AsBool<&'a Matrix<T>> {}

/// The mask of an operation, with its mode spelled out.
///
/// Built only through [`Mask::NONE`] and the four constructors, so that a
/// valued mask is always one [`ValuedMask`] accepts.
#[derive(Clone, Copy)]
pub struct Mask<'a, R = GrB_Matrix> {
    mode: MaskMode,
    source: Option<MaskRef<'a, R>>,
}

impl<'a, R> Mask<'a, R> {
    /// No mask: every position of the output is written.
    pub const NONE: Self = Self {
        mode: MaskMode::NONE,
        source: None,
    };

    /// `C<M>` structural: positions where `m` has an entry; `m` may hold
    /// any type, or be [`Output`].
    pub fn structural<M: AsMask<'a, R>>(m: M) -> Self {
        Self {
            mode: MaskMode {
                structural: true,
                complement: false,
            },
            source: Some(m.mask_ref(Token)),
        }
    }

    /// `C<!M>` structural: positions where `m` has no entry.
    pub fn structural_complement<M: AsMask<'a, R>>(m: M) -> Self {
        Self {
            mode: MaskMode {
                structural: true,
                complement: true,
            },
            source: Some(m.mask_ref(Token)),
        }
    }

    /// `C<M>` valued: positions where the `bool` mask stores `true`.
    pub fn valued<M: ValuedMask<'a, R>>(m: M) -> Self {
        Self {
            mode: MaskMode::NONE,
            source: Some(m.mask_ref(Token)),
        }
    }

    /// `C<!M>` valued: positions where the `bool` mask stores nothing or
    /// `false`.
    pub fn valued_complement<M: ValuedMask<'a, R>>(m: M) -> Self {
        Self {
            mode: MaskMode {
                structural: false,
                complement: true,
            },
            source: Some(m.mask_ref(Token)),
        }
    }

    /// The raw mask handle to pass, given the output's; null for `NONE`.
    pub(crate) fn raw(
        &self,
        output: R,
    ) -> R
    where
        R: RawHandle,
    {
        match self.source {
            None => R::null(),
            Some(MaskRef {
                source: Source::Handle(handle),
                ..
            }) => handle,
            Some(MaskRef {
                source: Source::Output,
                ..
            }) => output,
        }
    }

    /// The descriptor bits this mask implies.
    pub(crate) const fn mode(&self) -> MaskMode {
        self.mode
    }
}

/// The `S` and `C` descriptor bits, derived from a [`Mask`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskMode {
    pub(crate) structural: bool,
    pub(crate) complement: bool,
}

impl MaskMode {
    /// No mask bits: what an operation without a mask argument passes.
    pub(crate) const NONE: Self = Self {
        structural: false,
        complement: false,
    };
}

/// The descriptor fields that are not about the mask.
///
/// `transpose_first` is `GrB_INP0` = `GrB_TRAN` (`T0`), `transpose_second` is
/// `GrB_INP1` (`T1`), `replace` is `GrB_OUTP` = `GrB_REPLACE` (`R`). Build one
/// from the constants or the builder methods:
/// `Desc::REPLACE.transpose_first()`.
///
/// The fields name an input by its position in the call, as GraphBLAS does,
/// not by what it is. An operation reads only the fields for the matrix
/// inputs it has and ignores the rest: in `vxm (w, .., u, A, ..)` the matrix
/// is the second input, so `transpose_second` is the one that applies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Desc {
    /// Clear the output before the masked write (`GrB_REPLACE`).
    pub replace: bool,
    /// Use the transpose of the first input (`T0`).
    pub transpose_first: bool,
    /// Use the transpose of the second input (`T1`).
    pub transpose_second: bool,
}

impl Desc {
    /// The default descriptor: no replace, no transposes.
    pub const NONE: Self = Self {
        replace: false,
        transpose_first: false,
        transpose_second: false,
    };
    /// `GrB_REPLACE`.
    pub const REPLACE: Self = Self {
        replace: true,
        ..Self::NONE
    };
    /// Transpose the first input.
    pub const T0: Self = Self {
        transpose_first: true,
        ..Self::NONE
    };
    /// Transpose the second input.
    pub const T1: Self = Self {
        transpose_second: true,
        ..Self::NONE
    };

    #[must_use]
    pub const fn replace(self) -> Self {
        Self {
            replace: true,
            ..self
        }
    }

    #[must_use]
    pub const fn transpose_first(self) -> Self {
        Self {
            transpose_first: true,
            ..self
        }
    }

    #[must_use]
    pub const fn transpose_second(self) -> Self {
        Self {
            transpose_second: true,
            ..self
        }
    }
}

/// The built-in `GrB_DESC_*` for a (mask mode, desc) pair; null for the
/// all-default combination, which is what the C API takes for "no
/// descriptor". Never allocates.
pub fn descriptor(
    mode: MaskMode,
    desc: Desc,
) -> GrB_Descriptor {
    // The built-in names spell the bits in the order R, S, C, T0, T1.
    // SAFETY: by-value reads of handles `GrB_init` set and nothing writes
    // afterwards.
    unsafe {
        match (
            desc.replace,
            mode.structural,
            mode.complement,
            desc.transpose_first,
            desc.transpose_second,
        ) {
            (false, false, false, false, false) => null_mut(),
            (false, false, false, false, true) => sys::GrB_DESC_T1,
            (false, false, false, true, false) => sys::GrB_DESC_T0,
            (false, false, false, true, true) => sys::GrB_DESC_T0T1,
            (false, false, true, false, false) => sys::GrB_DESC_C,
            (false, false, true, false, true) => sys::GrB_DESC_CT1,
            (false, false, true, true, false) => sys::GrB_DESC_CT0,
            (false, false, true, true, true) => sys::GrB_DESC_CT0T1,
            (false, true, false, false, false) => sys::GrB_DESC_S,
            (false, true, false, false, true) => sys::GrB_DESC_ST1,
            (false, true, false, true, false) => sys::GrB_DESC_ST0,
            (false, true, false, true, true) => sys::GrB_DESC_ST0T1,
            (false, true, true, false, false) => sys::GrB_DESC_SC,
            (false, true, true, false, true) => sys::GrB_DESC_SCT1,
            (false, true, true, true, false) => sys::GrB_DESC_SCT0,
            (false, true, true, true, true) => sys::GrB_DESC_SCT0T1,
            (true, false, false, false, false) => sys::GrB_DESC_R,
            (true, false, false, false, true) => sys::GrB_DESC_RT1,
            (true, false, false, true, false) => sys::GrB_DESC_RT0,
            (true, false, false, true, true) => sys::GrB_DESC_RT0T1,
            (true, false, true, false, false) => sys::GrB_DESC_RC,
            (true, false, true, false, true) => sys::GrB_DESC_RCT1,
            (true, false, true, true, false) => sys::GrB_DESC_RCT0,
            (true, false, true, true, true) => sys::GrB_DESC_RCT0T1,
            (true, true, false, false, false) => sys::GrB_DESC_RS,
            (true, true, false, false, true) => sys::GrB_DESC_RST1,
            (true, true, false, true, false) => sys::GrB_DESC_RST0,
            (true, true, false, true, true) => sys::GrB_DESC_RST0T1,
            (true, true, true, false, false) => sys::GrB_DESC_RSC,
            (true, true, true, false, true) => sys::GrB_DESC_RSCT1,
            (true, true, true, true, false) => sys::GrB_DESC_RSCT0,
            (true, true, true, true, true) => sys::GrB_DESC_RSCT0T1,
        }
    }
}
