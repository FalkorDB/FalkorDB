//! Index selection for `assign` and `extract`.

use crate::sys::{self, GrB_Index};

/// Which rows (or columns, or vector positions) an `assign` / `extract`
/// touches: every index, or an explicit list.
///
/// Lists may be unsorted and may repeat. Stride ranges (`GxB_RANGE`,
/// `GxB_STRIDE`) are deliberately not offered yet.
#[derive(Clone, Copy, Debug)]
pub enum Indices<'a> {
    /// `GrB_ALL`.
    All,
    /// An explicit index list.
    List(&'a [GrB_Index]),
}

impl Indices<'_> {
    /// The `(I, ni)` pair the C API takes; `dim` gives the dimension `All`
    /// spans, queried only for `All`.
    pub(crate) fn raw(
        self,
        dim: impl FnOnce() -> GrB_Index,
    ) -> (*const GrB_Index, GrB_Index) {
        match self {
            // SAFETY: a by-value read of a pointer `GrB_init` set and nothing
            // writes afterwards.
            Self::All => (unsafe { sys::GrB_ALL }, dim()),
            Self::List(list) => (list.as_ptr(), list.len() as GrB_Index),
        }
    }
}
