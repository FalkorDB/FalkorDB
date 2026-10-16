//! `Graph` -> [`CchInput`] (#3211).
//!
//! The whole job is to turn the graph's relationship tensors into the two CSR
//! arrays the hierarchy builder consumes: a symmetric self-loop-free skeleton
//! for Phase 1, and a directed weight matrix for Phase 2. It is done in
//! GraphBLAS end to end — per relationship type, one `GrB_Matrix_apply` scores
//! every single-edge pair in parallel and one `GrB_mxv` over a `MIN` semiring
//! scores every parallel edge and keeps each pair's cheapest, both merging
//! into one weight matrix through `MIN`; two reductions then check every
//! weight, and `GxB_unload_Matrix_into_Container` hands back GraphBLAS's own
//! CSR buffers. Nothing walks the graph entry by entry.
//!
//! The C original is `CCH_BuildWeightMatrix` / `_get_edge_weight` in
//! `src/index/cch_index.c`, plus `_build_undirected_skeleton` in
//! `src/algorithms/cch.c`. The one structural difference is edge-id storage:
//! C's tensor cell is either a scalar id or a tagged `GrB_Vector` of ids, so
//! one operator covers both. This repo stores a single-edge pair's id inline
//! in the forward matrix and promotes a multi-edge pair to the
//! [`MULTI_EDGE`](crate::graph::graphblas::tensor::MULTI_EDGE) sentinel, with
//! its ids living in `me` keyed by
//! [`compound_key`](crate::graph::graphblas::tensor::compound_key) — so it
//! takes two passes, merged by `MIN`. The same shape as `algo.MSF`
//! (`runtime/functions/algo_procedures.rs`), which solved this first.

#[cfg(test)]
mod tests;

use std::mem::transmute;
use std::os::raw::c_void;
use std::ptr::{self, null_mut};
use std::sync::OnceLock;

use crate::graph::attribute_store::AttributeStore;
use crate::graph::graph::Graph;
use crate::graph::graphblas::matrix::{
    Descriptor, IntegerWidth, Matrix, Orientation, Sparsity, grb,
};
use crate::graph::graphblas::tensor::{MULTI_EDGE, MeBlock, Tensor, compound_key_inverse};
use crate::graph::graphblas::vector::Vector;
use crate::graph::graphblas::{
    GrB_ALL, GrB_BOOL, GrB_BinaryOp, GrB_FP64, GrB_Index, GrB_IndexUnaryOp, GrB_IndexUnaryOp_new,
    GrB_MIN_FP64, GrB_MIN_MONOID_FP64, GrB_Matrix_apply_IndexOp_UDT, GrB_Scalar, GrB_Scalar_free,
    GrB_Scalar_new, GrB_Scalar_setElement_BOOL, GrB_Semiring, GrB_Semiring_new, GrB_Type,
    GrB_Type_new, GrB_UINT64, GrB_Vector, GrB_Vector_assign_UDT, GrB_Vector_free, GrB_Vector_new,
    GrB_mxv, GxB_BinaryOp_new_IndexOp, GxB_IndexBinaryOp, GxB_IndexBinaryOp_new,
    GxB_index_binary_function, GxB_index_unary_function,
};
use crate::runtime::value::Value;

use super::{CchError, CchIndexDef, CchInput, Csr, WeightCsr, check_size};

/// `AttributeStore`'s read path is `Arc` traversal with no interior
/// mutability (`attribute_store.rs`: `DataBlock::block`), which is what makes
/// [`cch_edge_weight`] safe to call from GraphBLAS worker threads. If that
/// ever stops being true this stops compiling, rather than becoming a data
/// race nobody notices.
const _: () = {
    const fn assert_sync<T: Sync>() {}
    assert_sync::<AttributeStore>();
};

/// Weight of an edge whose weight attribute is missing or non-numeric.
///
/// Matches C's `_edge_weight`, which reads the property with a default of
/// `SI_LongVal(1)`, and matches what Dijkstra and A* already do per edge in
/// this repo.
const DEFAULT_WEIGHT: f64 = 1.0;

/// What the weight operators write for an edge whose weight is negative, NaN
/// or infinite.
///
/// One value for every bad weight, and the smallest there is, so a single
/// `MIN` reduction finds any of them: it beats every valid weight in each
/// `MIN` merge on the way (a pair, a parallel edge, a second type), so it
/// cannot be hidden behind a cheaper edge, and a NaN is never left for `fmin`
/// to skip.
const INVALID_WEIGHT: f64 = f64::NEG_INFINITY;

/// What the inline operator writes for a [`MULTI_EDGE`] pair, whose real ids
/// it cannot see: the identity of the `MIN` merge, so the pair's weight is
/// whatever the multi-edge pass brings, and a pair that pass missed is left
/// at `+inf`, which [`CchInput::build`] reports.
const UNSCORED: f64 = f64::INFINITY;

/// Context handed to the GraphBLAS weight operators.
///
/// Carries a borrowed pointer to the relationship [`AttributeStore`] — the
/// only graph state the operators read — rather than the whole graph. It stays
/// valid for every GraphBLAS call that reads it: [`CchInput::build`] holds
/// `&Graph` across them, so no writer can be mutating the store, and nothing
/// GraphBLAS keeps after `build` returns holds a copy (the multi-edge pass's
/// context vector is freed with the pass).
///
/// `#[repr(C)]` plain-old-data so GraphBLAS can memcpy it as a value of the
/// user-defined type [`WeightOps::ctx_type`].
#[repr(C)]
#[derive(Clone, Copy)]
struct CchWeightCtx {
    attrs: *const AttributeStore,
    attr_idx: u16,
}

/// Weight of one edge id: the numeric weight attribute, or
/// [`DEFAULT_WEIGHT`] when it is missing or not a number, or
/// [`INVALID_WEIGHT`] when it is negative, NaN or infinite.
///
/// # Panic safety
/// Called from GraphBLAS OpenMP worker threads, where a panic ends the process
/// (the panic hook exits, and unwinding out of the `extern "C"` operator would
/// abort anyway), so this must not unwind. It does one read-only
/// attribute lookup — `Arc` traversal plus a `Value` clone — and nothing else.
#[inline]
unsafe fn cch_edge_weight(
    ctx: &CchWeightCtx,
    edge_id: u64,
) -> f64 {
    // No null check on `attrs`: the only caller sets it from a live
    // `&AttributeStore` held for the whole apply, so it cannot be null, and a
    // null check would not catch the one real risk (a dangling pointer).
    let w = edge_weight(unsafe { &*ctx.attrs }, ctx.attr_idx, edge_id);
    // NaN fails both comparisons.
    if (0.0..f64::INFINITY).contains(&w) {
        w
    } else {
        INVALID_WEIGHT
    }
}

/// The weight attribute of one edge as a number, [`DEFAULT_WEIGHT`] when it
/// is missing or not a number. Not validated: the one weight rule shared by
/// the index build, which checks the result, and by [`cheapest_edge`].
pub(crate) fn edge_weight(
    attrs: &AttributeStore,
    attr_idx: u16,
    edge_id: u64,
) -> f64 {
    match attrs.get_attr_by_idx(edge_id, attr_idx) {
        Some(Value::Float(f)) => f,
        #[allow(clippy::cast_precision_loss)]
        Some(Value::Int(k)) => k as f64,
        _ => DEFAULT_WEIGHT,
    }
}

/// The cheapest edge from `src` to `dst` over `tensors`, by [`edge_weight`],
/// as `(edge id, weight)`; `None` when no tensor has one.
///
/// Per tensor, one [`Tensor::get`]: the pair's effective inline value
/// (pending `dp` first, committed `m` unless `dm` deletes it) and, for a
/// [`MULTI_EDGE`] pair, that pair's row of its `me` block, so the cost is the
/// pair's own edges, never a row scan. Ties go to the first tensor in order,
/// then to the lowest edge id. A pair outside a tensor's dimensions has no
/// edge there.
pub(crate) fn cheapest_edge<'t>(
    tensors: impl IntoIterator<Item = &'t Tensor>,
    attrs: &AttributeStore,
    attr_idx: u16,
    src: u64,
    dst: u64,
) -> Option<(u64, f64)> {
    let mut best: Option<(u64, f64)> = None;
    for tensor in tensors {
        let m = tensor.fwd_m();
        if src >= m.nrows() || dst >= m.ncols() {
            continue;
        }
        for id in tensor.get(src, dst) {
            let w = edge_weight(attrs, attr_idx, id);
            if best.is_none_or(|(_, best_w)| w < best_w) {
                best = Some((id, w));
            }
        }
    }
    best
}

// The operators below take the types they mean, and are cast to GraphBLAS's
// `void *` function types where they are registered (as LAGraph does in C).
// The cast is sound: raw pointers to sized types are ABI-compatible, so
// GraphBLAS calls each one exactly as declared here. GraphBLAS also never
// passes a null pointer to an operator, so none is checked for.

/// Index-unary operator over the **inline** forward matrix, where the matrix
/// *value* is the edge id: `z = weight(x)`, or [`UNSCORED`] for a
/// [`MULTI_EDGE`] pair, whose ids the multi-edge pass scores. Every entry
/// gets a value, so the result has the forward matrix's structure.
///
/// # Panic safety
/// Runs on GraphBLAS worker threads; see [`cch_edge_weight`].
unsafe extern "C" fn cch_inline_weight_op(
    z: *mut f64,
    x: *const u64,
    _i: GrB_Index,
    _j: GrB_Index,
    ctx: *const CchWeightCtx,
) {
    unsafe {
        *z = match *x {
            MULTI_EDGE => UNSCORED,
            id => cch_edge_weight(&*ctx, id),
        };
    }
}

type InlineWeightFn =
    unsafe extern "C" fn(*mut f64, *const u64, GrB_Index, GrB_Index, *const CchWeightCtx);

/// Index-binary operator for the multi-edge `mxv`, `v = me ⊕.⊗ c` with `⊕` =
/// `MIN`: `x` is an entry of `me`, whose *column index* `jx` is the edge id
/// (its `bool` value carries nothing); `y` is the entry of the context vector
/// `c` the product pairs it with (every entry of `c` is the context), so
/// `z = weight(jx)`. Theta is unused.
///
/// # Panic safety
/// Runs on GraphBLAS worker threads; see [`cch_edge_weight`].
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn cch_me_weight_op(
    z: *mut f64,
    _x: *const bool,
    _ix: GrB_Index,
    jx: GrB_Index,
    ctx: *const CchWeightCtx,
    _iy: GrB_Index,
    _jy: GrB_Index,
    _theta: *const bool,
) {
    unsafe {
        *z = cch_edge_weight(&*ctx, jx);
    }
}

type MeWeightFn = unsafe extern "C" fn(
    *mut f64,
    *const bool,
    GrB_Index,
    GrB_Index,
    *const CchWeightCtx,
    GrB_Index,
    GrB_Index,
    *const bool,
);

impl CchInput {
    /// Build the hierarchy builder's input from `graph`, over the relationship
    /// types and weight attribute named by `def`.
    ///
    /// Node ids are used unchanged as CCH ids, so `n` is the graph's matrix
    /// dimension ([`Graph::node_cap`]) rather than its live node count: ids
    /// freed by deletions, and the slack the capacity grows in, appear as
    /// isolated vertices. This mirrors C, which sizes the hierarchy with
    /// `Graph_RequiredMatrixDim`.
    ///
    /// Guarantees the result satisfies, which Phase 1 and Phase 2 may rely on:
    /// - `skeleton` is symmetric and has no `u == v` entry;
    /// - `pattern(weights) ⊆ pattern(skeleton)`, so every weighted arc has a
    ///   skeleton edge — the skeleton is derived from the weight matrix's own
    ///   pattern, so this is structural rather than hoped for;
    /// - `weights` has no `u == v` entry either: a self-loop can never lie on
    ///   a shortest path, and dropping it from both sides is what keeps the
    ///   inclusion above true;
    /// - every weight is finite and `>= 0`;
    /// - within a row, column ids ascend and do not repeat.
    ///
    /// # Errors
    /// - [`CchError::UnknownRelType`] / [`CchError::UnknownAttribute`] if
    ///   `def` names something this graph does not have.
    /// - [`CchError::GraphTooLarge`] if the matrix dimension exceeds
    ///   [`MAX_NODES`](super::MAX_NODES) (`2^31 - 1`), or the skeleton holds
    ///   more than [`MAX_SKELETON_ENTRIES`](super::MAX_SKELETON_ENTRIES)
    ///   entries (`2^31 - 1`, i.e. at most 1,073,741,823 connected node
    ///   pairs). Both are checked here, the entry count on the skeleton matrix
    ///   before either CSR export is materialised.
    /// - [`CchError::InvalidWeight`] if any edge of an indexed type carries a
    ///   negative, NaN or infinite weight — every edge, not only the cheapest of
    ///   its pair, and self-loops included even though they are then dropped.
    ///
    /// # Panics
    /// If a GraphBLAS call fails (see `grb!`). Every matrix and operator here is
    /// built by this function, so that is a bug or out of memory, never
    /// something in the graph's data: bad weights, unknown names and size are
    /// the errors above.
    pub fn build(
        graph: &Graph,
        def: &CchIndexDef,
    ) -> Result<Self, CchError> {
        let tensors = graph.relationship_tensors();
        let mut selected = Vec::with_capacity(def.rel_types().len());
        for t in def.rel_types() {
            selected.push(tensors.get(t.0).ok_or(CchError::UnknownRelType)?);
        }
        let attrs = graph.relationship_attrs();
        if usize::from(def.weight_attr) >= graph.property_key_count() {
            return Err(CchError::UnknownAttribute);
        }

        // The hierarchy is sized by the matrix dimension, not the live node
        // count. Take the largest in play, rows and columns, so a tensor that
        // has outgrown `node_cap` cannot silently lose its last rows or
        // columns to the resize below.
        let n = selected
            .iter()
            .map(|t| t.fwd_m().nrows().max(t.fwd_m().ncols()))
            .fold(graph.node_cap(), u64::max);
        // Node count only; the skeleton's entry count is checked once it is built.
        check_size(usize::try_from(n).map_err(|_| CchError::GraphTooLarge)?, 0)?;

        let ctx = CchWeightCtx {
            attrs: ptr::from_ref(attrs),
            attr_idx: def.weight_attr,
        };
        let ops = WeightOps::get();
        let mut weights = Matrix::<f64>::new(n, n);
        csr_ready(&mut weights);
        for tensor in &selected {
            ops.accumulate(&mut weights, tensor, &ctx, n);
        }
        check_weights(&weights)?;

        // Drop self-loops from the weights, which is what lets the skeleton be
        // derived from this pattern and still be diagonal free. After the
        // check, so a bad weight on a self-loop is refused all the same.
        weights.drop_diagonal();
        weights.wait();

        // Skeleton = pattern(W) ∪ pattern(Wᵀ): a weight of 0.0 is an entry
        // like any other.
        let mut skeleton = Matrix::<bool>::new(n, n);
        csr_ready(&mut skeleton);
        skeleton.pattern_union(&weights, &weights, Some(Descriptor::T1));
        skeleton.wait();

        // The contract wants one weight per entry, and the export refuses an iso
        // matrix (one stored value for all entries). The weight passes have not
        // been seen to leave `W` iso, even with every weight equal
        // (`equal_weights_give_one_weight_per_entry`), but GraphBLAS may store
        // any all-equal result that way, so expand it if it is: a no-op if not.
        weights.set_iso(false);

        // The skeleton's entry count is known now, before either export: past
        // `MAX_SKELETON_ENTRIES` the graph is too large. That ceiling (below
        // `2^31`) is also what keeps both entry counts below `2^32`, so the
        // 32-bit offset hint holds on both matrices (the weights have at most
        // as many entries as the skeleton).
        check_size(n as usize, skeleton.nvals())?;

        // Plain sparse storage has an offset per row, `4 * n` bytes at 32 bits,
        // where hypersparse stores only the non-empty rows. So each matrix is
        // converted only just before its own export, and the skeleton is dropped
        // before the weights convert: one sparse matrix at a time.
        //
        // Both matrices were set up by `csr_ready` and are owned by this
        // function alone, so the export cannot refuse them: the 32-bit column
        // hint holds because `n <= MAX_NODES < 2^31`, and the offset hint by the
        // check above. A refusal is a bug.
        skeleton.set_sparsity(Sparsity::Sparse);
        let skel = skeleton
            .to_csr_pattern::<u32, u32>()
            .map_err(|e| CchError::Internal(e.reason()))?;
        drop(skeleton);
        weights.set_sparsity(Sparsity::Sparse);
        let w = weights
            .to_csr_f64::<u32, u32>()
            .map_err(|e| CchError::Internal(e.reason()))?;
        drop(weights);
        Ok(Self {
            n: n as usize,
            skeleton: Csr {
                n: skel.nrows as usize,
                p: skel.row_ptr,
                i: skel.col_idx,
            },
            weights: WeightCsr {
                csr: Csr {
                    n: w.nrows as usize,
                    p: w.row_ptr,
                    i: w.col_idx,
                },
                w: w.vals,
            },
        })
    }
}

/// Set up a matrix this module builds for the CSR export, at creation so no
/// later step converts it: row-major, with the input's widths (32-bit
/// offsets, 32-bit column ids). GraphBLAS keeps these through every operation
/// that writes the matrix, and the export checks them. The sparse format the
/// export also needs is set later, just before each export: see `build`.
fn csr_ready<T>(m: &mut Matrix<T>) {
    m.set_orientation(Orientation::RowMajor);
    m.set_offset_width(IntegerWidth::Bits32);
    m.set_col_index_width(IntegerWidth::Bits32);
}

/// Refuse the build if any edge's weight was bad, with two reductions over
/// the merged weights and no scratch matrix.
///
/// The operators score a bad weight as [`INVALID_WEIGHT`] (`-inf`), which wins
/// every `MIN` merge, so one is still in `w` after the merges, wherever it
/// sat: the minimum is below 0 exactly when some edge was bad. A NaN never
/// reaches `w` (the operators map it too), so `fmin` skipping NaN does not
/// matter. The maximum catches the other direction: [`UNSCORED`] left on a
/// pair means the multi-edge pass did not score a pair the forward matrix
/// marks [`MULTI_EDGE`], a broken tensor invariant rather than bad data.
///
/// No sums, so no overflow caveat. Runs before the self-loops are dropped,
/// so their weights are checked too.
fn check_weights(w: &Matrix<f64>) -> Result<(), CchError> {
    // The minimum of an empty matrix is `+inf` and its maximum `-inf`, both
    // accepted.
    if w.min_value() < 0.0 {
        return Err(CchError::InvalidWeight);
    }
    if w.max_value() == UNSCORED {
        return Err(CchError::Internal("a multi-edge pair has no edge in `me`"));
    }
    Ok(())
}

/// The GraphBLAS objects the weight passes need, created once per process,
/// the first time an input is built, and never freed: as in C
/// (`cch_index.c`, `pthread_once`), so nothing GraphBLAS still holds can
/// outlive them. They are read-only after creation, and GraphBLAS takes
/// them as `const` inputs, so concurrent builds share them.
struct WeightOps {
    /// The user-defined type of [`CchWeightCtx`]: the apply's thunk and the
    /// multi-edge pass's context vector.
    ctx_type: GrB_Type,
    /// [`cch_inline_weight_op`].
    inline_op: GrB_IndexUnaryOp,
    /// `MIN` over [`cch_me_weight_op`].
    me_semiring: GrB_Semiring,
}

// Raw GraphBLAS handles, written once in `new` and only read after.
unsafe impl Send for WeightOps {}
unsafe impl Sync for WeightOps {}

impl WeightOps {
    fn get() -> &'static Self {
        static OPS: OnceLock<WeightOps> = OnceLock::new();
        OPS.get_or_init(Self::new)
    }

    fn new() -> Self {
        unsafe {
            let mut ctx_type: GrB_Type = null_mut();
            grb!(GrB_Type_new(&raw mut ctx_type, size_of::<CchWeightCtx>()));
            // The operators keep their typed signatures, which match what
            // GraphBLAS passes argument by argument (only pointee types
            // differ). The bindings' function types are `Option<extern fn>`,
            // which has the layout of a bare fn pointer with `None` as null,
            // so the transmute yields `Some` of the (non-null) operator.
            let inline_fn =
                transmute::<InlineWeightFn, GxB_index_unary_function>(cch_inline_weight_op);
            let mut inline_op: GrB_IndexUnaryOp = null_mut();
            grb!(GrB_IndexUnaryOp_new(
                &raw mut inline_op,
                inline_fn,
                GrB_FP64,
                GrB_UINT64,
                ctx_type
            ));

            let me_fn = transmute::<MeWeightFn, GxB_index_binary_function>(cch_me_weight_op);
            let mut me_idx_op: GxB_IndexBinaryOp = null_mut();
            grb!(GxB_IndexBinaryOp_new(
                &raw mut me_idx_op,
                me_fn,
                GrB_FP64,
                GrB_BOOL,
                ctx_type,
                GrB_BOOL,
                null_mut(),
                null_mut()
            ));
            // An index-binary operator enters a semiring as a binary operator
            // with its theta bound. Theta is unused (the context comes in `y`,
            // so nothing per build is bound here), but it must be a value.
            let mut theta: GrB_Scalar = null_mut();
            grb!(GrB_Scalar_new(&raw mut theta, GrB_BOOL));
            grb!(GrB_Scalar_setElement_BOOL(theta, false));
            let mut me_op: GrB_BinaryOp = null_mut();
            grb!(GxB_BinaryOp_new_IndexOp(&raw mut me_op, me_idx_op, theta));
            // The binary operator holds its own copy of theta.
            grb!(GrB_Scalar_free(&raw mut theta));
            let mut me_semiring: GrB_Semiring = null_mut();
            grb!(GrB_Semiring_new(
                &raw mut me_semiring,
                GrB_MIN_MONOID_FP64,
                me_op
            ));
            Self {
                ctx_type,
                inline_op,
                me_semiring,
            }
        }
    }

    /// Merge one relationship type's cheapest weight per node pair into `w`:
    /// a bad weight arrives as [`INVALID_WEIGHT`], for [`check_weights`].
    fn accumulate(
        &self,
        w: &mut Matrix<f64>,
        tensor: &Tensor,
        ctx: &CchWeightCtx,
        n: u64,
    ) {
        if tensor.edge_count() == 0 {
            return;
        }
        self.inline_pass(w, tensor, ctx, n);
        if tensor.has_multi_edge() {
            self.multi_pass(w, tensor, ctx, n);
        }
    }

    /// Every forward pair: the edge id is the `UINT64` value of the
    /// effective forward matrix, so one `apply` scores the lot, straight into
    /// `w` through the `MIN` accumulator (no scored copy). A multi-edge pair
    /// is scored [`UNSCORED`], keeping the forward structure whole, and gets
    /// its weight from [`Self::multi_pass`].
    fn inline_pass(
        &self,
        w: &mut Matrix<f64>,
        tensor: &Tensor,
        ctx: &CchWeightCtx,
        n: u64,
    ) {
        let mut eff = tensor.effective_fwd();
        // The forward matrix is `node_cap` square, as `w` is, unless a tensor
        // outgrew it (see `build`): then resize it to `w`'s shape. `to_mut`
        // copies only when `eff` is the tensor's base; a merged matrix is
        // already ours and is resized as is.
        if eff.nrows() != n || eff.ncols() != n {
            eff.to_mut().resize(n, n);
        }
        unsafe {
            grb!(GrB_Matrix_apply_IndexOp_UDT(
                w.inner(),
                null_mut(),
                GrB_MIN_FP64,
                self.inline_op,
                eff.inner(),
                ptr::from_ref(ctx).cast_mut().cast::<c_void>(),
                null_mut()
            ));
        }
    }

    /// Multi-edge pairs: every id lives in its block's `me[row][id]`, where
    /// `(block, row) = compound_key(src, dst)` and the *column index* is the
    /// edge id. One `mxv` per block, `v = me ⊕.⊗ c` over `MIN` and
    /// [`cch_me_weight_op`], scores every id and takes each pair's minimum in
    /// the same pass: `c` is a full vector, one entry per column, every entry
    /// the context. It is iso (one stored value), so it costs nothing however
    /// wide `me` is, and the product walks only `me`'s entries.
    ///
    /// A pair's ids live in exactly one block (the block is the high bits of
    /// its two endpoints), so each block's minima go straight into `w` with
    /// nothing to merge between blocks — the same per-block loop `algo.MSF`
    /// runs. Every graph within 2^30 nodes per axis has the one block; one
    /// between 2^30 and [`MAX_NODES`](super::MAX_NODES) can have up to four.
    fn multi_pass(
        &self,
        w: &mut Matrix<f64>,
        tensor: &Tensor,
        ctx: &CchWeightCtx,
        n: u64,
    ) {
        for (block, me) in tensor.edge_versioned_all() {
            let me = me.effective();
            if me.nvals() == 0 {
                continue;
            }
            let ncols = me.ncols();
            let v = Vector::<f64>::new(me.nrows());
            unsafe {
                let mut c: GrB_Vector = null_mut();
                grb!(GrB_Vector_new(&raw mut c, self.ctx_type, ncols));
                grb!(GrB_Vector_assign_UDT(
                    c,
                    null_mut(),
                    null_mut(),
                    ptr::from_ref(ctx).cast_mut().cast::<c_void>(),
                    GrB_ALL,
                    ncols,
                    null_mut()
                ));
                grb!(GrB_mxv(
                    v.ptr(),
                    null_mut(),
                    null_mut(),
                    self.me_semiring,
                    me.inner(),
                    c,
                    null_mut()
                ));
                grb!(GrB_Vector_free(&raw mut c));
            }
            if let Some(pairs) = pair_weights_from_keys(&v, block, n) {
                w.element_wise_add(None, None, Some(&pairs), None);
            }
        }
    }
}

/// Turn one block's `v[row] = weight`, with `(block, row) = compound_key(src,
/// dst)`, into an `n x n` matrix.
///
/// **The one step of this module that is not pure GraphBLAS**, by choice.
/// The row key packs the low 30 bits of `src` and `dst` side by side, and
/// [`compound_key_inverse`] turns it back into the pair. What is below is the
/// host split `algo.MSF` already ships (`runtime/functions/algo_procedures.rs`),
/// in [`block_pair_coords`]: one `extractTuples`, one iteration per multi-edge
/// **pair** (not per edge: the per-edge minimum was already taken by the
/// `MIN` in the `mxv` upstream), then one `GrB_Matrix_build`.
///
/// The build has no `dup` operator: the keys are a vector's indices, so each
/// appears once, and [`compound_key_inverse`] is injective within a block, so
/// no pair repeats. A repeat would be a bug, and the build refuses it.
///
/// A pure-GraphBLAS split does exist: lift each key into a value with a
/// row-index apply (`GrB_ROWINDEX_INT64`), compute `src` and `dst` from it
/// with shift/and/or applies, and hand the three vectors to
/// `GxB_Matrix_build_Vector`, which takes its indices from vector *values*.
/// It is kept as a host loop over the already reduced vector, so the two
/// procedures stay alike.
///
/// TODO: projecting the graph into one `GrB_Matrix` (pair weights keyed by
/// `(src, dst)` directly, for this index and `algo.MSF` alike) would remove
/// the compound-key split, and with it this loop.
fn pair_weights_from_keys(
    v: &Vector<f64>,
    block: MeBlock,
    n: u64,
) -> Option<Matrix<f64>> {
    let (keys, vals) = v.extract_tuples();
    if keys.is_empty() {
        return None;
    }
    let (rows, cols) = block_pair_coords(block, &keys);
    let mut m = Matrix::<f64>::new(n, n);
    m.build(&rows, &cols, &vals);
    Some(m)
}

/// The host half of [`pair_weights_from_keys`]: the `(src, dst)` of every row
/// key of one `me` block, as parallel row and column arrays in key order.
///
/// Weights are not an input because the split never touches them: the pair at
/// position `k` of the output owns the weight at position `k` of whatever the
/// keys came with. Pure, so the blocks past [`ME_BLOCK_0`] — reachable only
/// above 2^30 nodes on one axis — are testable without a graph that size.
///
/// [`ME_BLOCK_0`]: crate::graph::graphblas::tensor::ME_BLOCK_0
fn block_pair_coords(
    block: MeBlock,
    keys: &[GrB_Index],
) -> (Vec<GrB_Index>, Vec<GrB_Index>) {
    keys.iter()
        .map(|&key| compound_key_inverse(block, key))
        .unzip()
}
