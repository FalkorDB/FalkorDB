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
//! CSR buffers. Nothing walks the graph edge by edge; the one host loop runs once per
//! multi-edge pair, after the per-edge minimum is taken in GraphBLAS
//! ([`pair_weights_from_keys`]).
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
use crate::graph::graphblas::matrix::{CsrExport, Descriptor, Matrix, grb};
use crate::graph::graphblas::scalar::Scalar;
use crate::graph::graphblas::tensor::{MULTI_EDGE, MeBlock, Tensor, compound_key_inverse};
use crate::graph::graphblas::vector::Vector;
use crate::graph::graphblas::{
    GrB_ALL, GrB_BOOL, GrB_BinaryOp, GrB_FP64, GrB_Index, GrB_IndexUnaryOp, GrB_IndexUnaryOp_new,
    GrB_Info, GrB_MIN_FP64, GrB_MIN_MONOID_FP64, GrB_Matrix_apply_IndexOp_UDT, GrB_Semiring,
    GrB_Semiring_new, GrB_Type, GrB_Type_new, GrB_UINT64, GrB_Vector, GrB_Vector_assign_UDT,
    GrB_Vector_free, GrB_Vector_new, GrB_mxv, GxB_BinaryOp_new_IndexOp, GxB_IndexBinaryOp,
    GxB_IndexBinaryOp_new, GxB_index_binary_function, GxB_index_unary_function,
};
use crate::runtime::value::Value;

use super::{CchError, CchIndexDef, CchInput, Csr, WeightCsr, id_index};

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

/// What the weight operators write for an edge whose weight is negative
/// (`-inf` included) or NaN. `+inf` is a weight: a closed edge.
///
/// One value for every bad weight, and the smallest there is, so a single
/// `MIN` reduction finds any of them: it beats every valid weight in each
/// `MIN` merge on the way (a pair, a parallel edge, a second type), so it
/// cannot be hidden behind a cheaper edge, and a NaN is never left for `fmin`
/// to skip.
const INVALID_WEIGHT: f64 = f64::NEG_INFINITY;

/// What the inline operator writes for a [`MULTI_EDGE`] pair, whose real ids
/// it cannot see: NaN, which every `MIN` merge drops (`GrB_MIN_FP64` is
/// `fmin`), so the pair's weight is whatever the multi-edge pass brings, and a
/// pair that pass missed is left at NaN, which [`CchInput::build`] reports.
/// Not `+inf`, the identity of `MIN`: that is a legal weight (a closed edge),
/// so a pair left at it could not be told from a pair of closed edges. No
/// edge's weight is NaN when it reaches `weights`: the operators map it to
/// [`INVALID_WEIGHT`].
const UNSCORED: f64 = f64::NAN;

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
/// user-defined type [`WeightOps::context_type`].
#[repr(C)]
#[derive(Clone, Copy)]
struct WeightContext {
    attributes: *const AttributeStore,
    attribute: u16,
}

impl WeightContext {
    /// The context as GraphBLAS takes a user-defined value: a pointer it
    /// copies the value from, and never writes through.
    fn as_value(&self) -> *mut c_void {
        ptr::from_ref(self).cast_mut().cast()
    }
}

/// Weight of one edge id: the numeric weight attribute, or
/// [`DEFAULT_WEIGHT`] when it is missing or not a number, or
/// [`INVALID_WEIGHT`] when it is negative or NaN. `+inf` is kept: a closed
/// edge.
///
/// # Safety
/// `context.attributes` points to a live `AttributeStore` that nobody writes
/// for the duration of the call, as [`WeightContext`] describes.
///
/// # Panic safety
/// Called from GraphBLAS OpenMP worker threads, where a panic ends the process
/// (the panic hook exits, and unwinding out of the `extern "C"` operator would
/// abort anyway), so this must not unwind. It does one read-only
/// attribute lookup — `Arc` traversal plus a `Value` clone — and nothing else.
#[inline]
unsafe fn cch_edge_weight(
    context: &WeightContext,
    edge_id: u64,
) -> f64 {
    // No null check on `attributes`: the only caller sets it from a live
    // `&AttributeStore` held for the whole apply, so it cannot be null, and a
    // null check would not catch the one real risk (a dangling pointer).
    let weight = match unsafe { (*context.attributes).get_attr_by_idx(edge_id, context.attribute) }
    {
        Some(Value::Float(weight)) => weight,
        #[allow(clippy::cast_precision_loss)]
        Some(Value::Int(integer)) => integer as f64,
        _ => DEFAULT_WEIGHT,
    };
    // NaN fails the comparison; `+inf` and `-0.0` pass it.
    if weight >= 0.0 {
        weight
    } else {
        INVALID_WEIGHT
    }
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
    context: *const WeightContext,
) {
    unsafe {
        *z = match *x {
            MULTI_EDGE => UNSCORED,
            id => cch_edge_weight(&*context, id),
        };
    }
}

type InlineWeightFn =
    unsafe extern "C" fn(*mut f64, *const u64, GrB_Index, GrB_Index, *const WeightContext);

/// Index-binary operator for the multi-edge `mxv`,
/// `pair_minimums = me ⊕.⊗ contexts` with `⊕` = `MIN`: `x` is an entry of
/// `me`, whose *column index* `jx` is the edge id (its `bool` value carries
/// nothing); `y` is the entry of the context vector `contexts` the product
/// pairs it with (every entry is the context), so `z = weight(jx)`. Theta is
/// unused. The operator's own parameters keep GraphBLAS's names.
///
/// # Panic safety
/// Runs on GraphBLAS worker threads; see [`cch_edge_weight`].
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn cch_multi_edge_weight_op(
    z: *mut f64,
    _x: *const bool,
    _ix: GrB_Index,
    jx: GrB_Index,
    context: *const WeightContext,
    _iy: GrB_Index,
    _jy: GrB_Index,
    _theta: *const bool,
) {
    unsafe {
        *z = cch_edge_weight(&*context, jx);
    }
}

type MultiEdgeWeightFn = unsafe extern "C" fn(
    *mut f64,
    *const bool,
    GrB_Index,
    GrB_Index,
    *const WeightContext,
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
    /// - every weight is `>= 0`, `+inf` (a closed edge) included;
    /// - within a row, column ids ascend and do not repeat.
    ///
    /// # Errors
    /// - [`CchError::UnknownRelType`] / [`CchError::UnknownAttribute`] if
    ///   `def` names something this graph does not have.
    /// - [`CchError::InvalidWeight`] if any edge of an indexed type carries a
    ///   negative or NaN weight — every edge, not only the cheapest of
    ///   its pair, and self-loops included even though they are then dropped.
    ///
    /// # Panics
    /// If a GraphBLAS call fails (see `grb!`). Every matrix and operator here is
    /// built by this function, so that is a bug or out of memory, never
    /// something in the graph's data: bad weights and unknown names are the
    /// errors above.
    pub fn build(
        graph: &Graph,
        def: &CchIndexDef,
    ) -> Result<Self, CchError> {
        let tensors = graph.relationship_tensors();
        let selected = def
            .rel_types()
            .iter()
            .map(|rel_type| tensors.get(rel_type.0).ok_or(CchError::UnknownRelType))
            .collect::<Result<Vec<_>, _>>()?;
        let attributes = graph.relationship_attrs();
        if usize::from(def.weight_attr) >= graph.property_key_count() {
            return Err(CchError::UnknownAttribute);
        }

        // The hierarchy is sized by the matrix dimension, not the live node
        // count. Take the largest in play, rows and columns, so a tensor that
        // has outgrown `node_cap` cannot silently lose its last rows or
        // columns to the resize below.
        let node_count = selected
            .iter()
            .map(|tensor| tensor.fwd_m().nrows().max(tensor.fwd_m().ncols()))
            .fold(graph.node_cap(), u64::max);

        let context = WeightContext {
            attributes: ptr::from_ref(attributes),
            attribute: def.weight_attr,
        };
        let ops = WeightOps::get();
        let mut weights = Matrix::<f64>::new(node_count, node_count);
        for tensor in &selected {
            ops.accumulate(&mut weights, tensor, &context);
        }
        check_weights(&weights)?;

        // Drop self-loops from the weights, which is what lets the skeleton be
        // derived from this pattern and still be diagonal free. After the
        // check, so a bad weight on a self-loop is refused all the same.
        weights.drop_diagonal();
        weights.wait();

        // Skeleton = pattern(W) ∪ pattern(Wᵀ): a weight of 0.0 is an entry
        // like any other.
        let mut skeleton = Matrix::<bool>::new(node_count, node_count);
        skeleton.pattern_union(&weights, &weights, Some(Descriptor::T1));
        skeleton.wait();

        // Plain sparse storage has an offset per row, `8 * node_count` bytes,
        // where hypersparse stores only the non-empty rows. So each matrix is
        // set up for the export (`prepare_csr`: row-major, 64-bit offsets and
        // column ids, the input's widths, then sparse) only just before its
        // own export, which takes its arrays and leaves it empty: one sparse
        // matrix at a time, and never a matrix next to a copy of itself.
        //
        // The contract wants one weight per entry, and the export refuses an
        // iso matrix (one stored value for all entries). The weight passes
        // have not been seen to leave `weights` iso, even with every weight equal,
        // but GraphBLAS may store any all-equal result that way, so
        // `prepare_csr` for values expands it if it is: a no-op if not.
        //
        // Both matrices are owned by this function alone, so neither call can
        // find a shared handle, and a set-up matrix cannot be refused by the
        // export: a refusal is a bug, [`CchError::CsrExport`]. A setting
        // GraphBLAS refuses (out of memory while converting) is reported the
        // same way.
        skeleton.prepare_csr::<u64, u64, ()>()?;
        let skeleton = skeleton.take_csr::<u64, u64, ()>()?.into();
        weights.prepare_csr::<u64, u64, f64>()?;
        let weights = weights.take_csr::<u64, u64, f64>()?.into();
        let input = Self {
            n: id_index(node_count),
            skeleton,
            weights,
        };
        debug_assert!(input.skeleton.is_symmetric_simple());
        Ok(input)
    }
}

/// The skeleton as GraphBLAS exported it: its arrays become the `Csr`'s.
impl From<CsrExport<u64, u64, ()>> for Csr {
    fn from(export: CsrExport<u64, u64, ()>) -> Self {
        Self {
            node_count: export.row_count(),
            row_offsets: export.row_offsets,
            neighbors: export.column_indices,
        }
    }
}

/// The weights as GraphBLAS exported them: pattern and values, unchanged.
impl From<CsrExport<u64, u64, f64>> for WeightCsr {
    fn from(export: CsrExport<u64, u64, f64>) -> Self {
        Self {
            csr: Csr {
                node_count: export.row_count(),
                row_offsets: export.row_offsets,
                neighbors: export.column_indices,
            },
            values: export.values,
        }
    }
}

/// Refuse the build if any edge's weight was bad, with two reductions over
/// the merged weights and no scratch matrix.
///
/// The operators score a bad weight as [`INVALID_WEIGHT`] (`-inf`), which wins
/// every `MIN` merge, so one is still in `weights` after the merges, wherever it
/// sat: the minimum is below 0 exactly when some edge was bad. An edge's NaN
/// never reaches `weights` (the operators map it too), and `fmin` skips the
/// [`UNSCORED`] NaN. The sum catches that one: once the minimum passed there
/// is no `-inf` left, so the sum is NaN exactly when an [`UNSCORED`] NaN is
/// left on a pair, i.e. the multi-edge pass did not score a pair the forward
/// matrix marks [`MULTI_EDGE`], a broken tensor invariant rather than bad
/// data. `+inf` (a closed edge) and an overflowing sum both give `+inf`, never
/// NaN.
///
/// Runs before the self-loops are dropped, so their weights are checked too.
fn check_weights(weights: &Matrix<f64>) -> Result<(), CchError> {
    // The minimum of an empty matrix is `+inf` and its sum `0`, both accepted.
    if weights.min_value() < 0.0 {
        return Err(CchError::InvalidWeight);
    }
    if weights.sum_value().is_nan() {
        return Err(CchError::Internal("a multi-edge pair has no edge in `me`"));
    }
    Ok(())
}

/// The multi-edge pass's `contexts`: a full vector of `length` entries of the
/// user-defined type [`WeightOps::context_type`], every one the context (iso,
/// one stored value). Freed on drop.
struct ContextVector(GrB_Vector);

impl ContextVector {
    fn new(
        context_type: GrB_Type,
        length: u64,
        context: &WeightContext,
    ) -> Self {
        let mut vector: GrB_Vector = null_mut();
        // SAFETY: `GrB_Vector_new` writes a new vector, which `Self` then
        // frees.
        grb!(unsafe { GrB_Vector_new(&raw mut vector, context_type, length) });
        let contexts = Self(vector);
        // SAFETY: our live vector, of the context's type; the assign copies
        // the context value into every entry.
        grb!(unsafe {
            GrB_Vector_assign_UDT(
                contexts.0,
                null_mut(),
                null_mut(),
                context.as_value(),
                GrB_ALL,
                length,
                null_mut(),
            )
        });
        contexts
    }
}

impl Drop for ContextVector {
    fn drop(&mut self) {
        // SAFETY: `self.0` is live and ours alone.
        let info = unsafe { GrB_Vector_free(&raw mut self.0) };
        debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
    }
}

/// The GraphBLAS objects the weight passes need, created once per process,
/// the first time an input is built, and never freed: as in C
/// (`cch_index.c`, `pthread_once`), so nothing GraphBLAS still holds can
/// outlive them. They are read-only after creation, and GraphBLAS takes
/// them as `const` inputs, so concurrent builds share them.
struct WeightOps {
    /// The user-defined type of [`WeightContext`]: the apply's thunk and the
    /// multi-edge pass's context vector.
    context_type: GrB_Type,
    /// [`cch_inline_weight_op`].
    inline_op: GrB_IndexUnaryOp,
    /// `MIN` over [`cch_multi_edge_weight_op`].
    multi_edge_semiring: GrB_Semiring,
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
        // The operators keep their typed signatures, which match what
        // GraphBLAS passes argument by argument (only pointee types differ).
        // The bindings' function types are `Option<extern fn>`, which has the
        // layout of a bare fn pointer with `None` as null.
        // SAFETY: both transmutes are between fn pointers of the same ABI and
        // arity, so each yields `Some` of the (non-null) operator.
        let (inline_fn, multi_edge_fn) = unsafe {
            (
                transmute::<InlineWeightFn, GxB_index_unary_function>(cch_inline_weight_op),
                transmute::<MultiEdgeWeightFn, GxB_index_binary_function>(cch_multi_edge_weight_op),
            )
        };
        let mut context_type: GrB_Type = null_mut();
        let mut inline_op: GrB_IndexUnaryOp = null_mut();
        let mut multi_edge_index_op: GxB_IndexBinaryOp = null_mut();
        let mut multi_edge_op: GrB_BinaryOp = null_mut();
        let mut multi_edge_semiring: GrB_Semiring = null_mut();
        // An index-binary operator enters a semiring as a binary operator
        // with its theta bound. Theta is unused (the context comes in `y`,
        // so nothing per build is bound here), but it must be a value; the
        // binary operator keeps its own copy.
        let theta = Scalar::from(false);
        // SAFETY: each call writes a new object to its out-pointer, and takes
        // only objects created before it (or built-in ones). None is freed:
        // they live for the process, as `WeightOps` says.
        unsafe {
            grb!(GrB_Type_new(
                &raw mut context_type,
                size_of::<WeightContext>()
            ));
            grb!(GrB_IndexUnaryOp_new(
                &raw mut inline_op,
                inline_fn,
                GrB_FP64,
                GrB_UINT64,
                context_type
            ));
            grb!(GxB_IndexBinaryOp_new(
                &raw mut multi_edge_index_op,
                multi_edge_fn,
                GrB_FP64,
                GrB_BOOL,
                context_type,
                GrB_BOOL,
                null_mut(),
                null_mut()
            ));
            grb!(GxB_BinaryOp_new_IndexOp(
                &raw mut multi_edge_op,
                multi_edge_index_op,
                theta.ptr()
            ));
            grb!(GrB_Semiring_new(
                &raw mut multi_edge_semiring,
                GrB_MIN_MONOID_FP64,
                multi_edge_op
            ));
        }
        Self {
            context_type,
            inline_op,
            multi_edge_semiring,
        }
    }

    /// Merge one relationship type's cheapest weight per node pair into `weights`:
    /// a bad weight arrives as [`INVALID_WEIGHT`], for [`check_weights`].
    fn accumulate(
        &self,
        weights: &mut Matrix<f64>,
        tensor: &Tensor,
        context: &WeightContext,
    ) {
        // No emptiness guard: both passes are GraphBLAS calls, which do nothing
        // on an empty matrix, where `Tensor::edge_count` can walk every
        // parallel edge on the host when multi-edge deltas are pending.
        self.inline_pass(weights, tensor, context);
        if tensor.has_multi_edge() {
            self.multi_pass(weights, tensor, context);
        }
    }

    /// Every forward pair: the edge id is the `UINT64` value of the
    /// effective forward matrix, so one `apply` scores the lot, straight into
    /// `weights` through the `MIN` accumulator (no scored copy). A multi-edge pair
    /// is scored [`UNSCORED`], keeping the forward structure whole, and gets
    /// its weight from [`Self::multi_pass`].
    fn inline_pass(
        &self,
        weights: &mut Matrix<f64>,
        tensor: &Tensor,
        context: &WeightContext,
    ) {
        let node_count = weights.nrows();
        let mut effective = tensor.effective_fwd();
        // The forward matrix is `node_cap` square, as `weights` is, unless a tensor
        // outgrew it (see `build`): then resize it to `weights`'s shape. `to_mut`
        // copies only when `effective` is the tensor's base; a merged matrix is
        // already ours and is resized as is.
        if effective.nrows() != node_count || effective.ncols() != node_count {
            effective.to_mut().resize(node_count, node_count);
        }
        // SAFETY: live matrices and operators; GraphBLAS copies the context
        // value and calls `cch_inline_weight_op` with it, under `context`'s
        // own guarantee.
        grb!(unsafe {
            GrB_Matrix_apply_IndexOp_UDT(
                weights.inner(),
                null_mut(),
                GrB_MIN_FP64,
                self.inline_op,
                effective.inner(),
                context.as_value(),
                null_mut(),
            )
        });
    }

    /// Multi-edge pairs: every id lives in its block's `multi_edges[row][id]`
    /// (the tensor's `me`), where `(block, row) = compound_key(src, dst)` and
    /// the *column index* is the edge id. One `mxv` per block,
    /// `pair_minimums = multi_edges ⊕.⊗ contexts` over `MIN` and
    /// [`cch_multi_edge_weight_op`], scores every id and takes each pair's
    /// minimum in the same pass: `contexts` is a full vector, one entry per
    /// column, every entry the context. It is iso (one stored value), so it
    /// costs nothing however wide `multi_edges` is, and the product walks only
    /// its entries.
    ///
    /// A pair's ids live in exactly one block (the block is the high bits of
    /// its two endpoints), so each block's minima go straight into `weights` with
    /// nothing to merge between blocks — the same per-block loop `algo.MSF`
    /// runs. Every graph within 2^30 nodes per axis has the one block; a larger
    /// one has one block per occupied pair of 2^30-wide id ranges.
    fn multi_pass(
        &self,
        weights: &mut Matrix<f64>,
        tensor: &Tensor,
        context: &WeightContext,
    ) {
        for (block, multi_edges) in tensor.edge_versioned_all() {
            let multi_edges = multi_edges.effective();
            if multi_edges.nvals() == 0 {
                continue;
            }
            let column_count = multi_edges.ncols();
            let pair_minimums = Vector::<f64>::new(multi_edges.nrows());
            let contexts = ContextVector::new(self.context_type, column_count, context);
            // SAFETY: live matrix, vectors and semiring.
            grb!(unsafe {
                GrB_mxv(
                    pair_minimums.ptr(),
                    null_mut(),
                    null_mut(),
                    self.multi_edge_semiring,
                    multi_edges.inner(),
                    contexts.0,
                    null_mut(),
                )
            });
            if let Some(pairs) = pair_weights_from_keys(&pair_minimums, block, weights.nrows()) {
                weights.element_wise_add(None, None, Some(&pairs), None);
            }
        }
    }
}

/// Turn one block's `pair_minimums[row] = weight`, with `(block, row) = compound_key(src,
/// dst)`, into a `node_count x node_count` matrix.
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
    pair_minimums: &Vector<f64>,
    block: MeBlock,
    node_count: u64,
) -> Option<Matrix<f64>> {
    let (keys, minimums) = pair_minimums.extract_tuples();
    if keys.is_empty() {
        return None;
    }
    let (rows, columns) = block_pair_coords(block, &keys);
    let mut pair_weights = Matrix::<f64>::new(node_count, node_count);
    pair_weights.build(&rows, &columns, &minimums);
    Some(pair_weights)
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
