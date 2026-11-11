//! The GraphBLAS side of [`CchInput::build`](super::CchInput::build): the
//! user-defined type and operators that score edges by their weight attribute,
//! created once per process ([`WeightOps`]), and the two passes that run them
//! over a relationship tensor. Everything here is `unsafe` glue between
//! GraphBLAS's C calling convention and the graph's attribute store.

use std::mem::transmute;
use std::os::raw::c_void;
use std::ptr::{self, null_mut};
use std::sync::OnceLock;

use crate::graph::attribute_store::AttributeStore;
use crate::graph::graphblas::matrix::{Matrix, grb};
use crate::graph::graphblas::scalar::Scalar;
use crate::graph::graphblas::tensor::{MULTI_EDGE, MeBlock, Tensor, compound_key_inverse};
use crate::graph::graphblas::vector::Vector;
use crate::graph::graphblas::{
    GrB_ALL, GrB_BOOL, GrB_BinaryOp, GrB_FP64, GrB_Index, GrB_IndexUnaryOp, GrB_IndexUnaryOp_new,
    GrB_Info, GrB_MIN_MONOID_FP64, GrB_Semiring, GrB_Semiring_new, GrB_Type, GrB_Type_new,
    GrB_UINT64, GrB_Vector, GrB_Vector_assign_UDT, GrB_Vector_free, GrB_Vector_new, GrB_mxv,
    GxB_BinaryOp_new_IndexOp, GxB_IndexBinaryOp, GxB_IndexBinaryOp_new, GxB_index_binary_function,
    GxB_index_unary_function,
};
use crate::runtime::value::Value;

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
pub(super) const DEFAULT_WEIGHT: f64 = 1.0;

/// What the weight operators write for an edge whose weight is negative
/// (`-inf` included) or NaN. `+inf` is a weight: a closed edge.
///
/// One value for every bad weight, and the smallest there is: it wins every
/// `MIN` merge on the way (a parallel edge, a second type), so it cannot be
/// hidden behind a cheaper parallel edge, and is still in its type's weights
/// for [`check_weights`](super::check_weights) to find. No edge's weight reaches the merges as NaN,
/// which `fmin` would skip.
pub(super) const INVALID_WEIGHT: f64 = f64::NEG_INFINITY;

/// What the inline operator writes for a [`MULTI_EDGE`] pair, whose real ids
/// it cannot see: NaN, which the multi-edge merge drops (`GrB_MIN_FP64` is
/// `fmin`), so the pair's weight is whatever the multi-edge pass brings, and a
/// pair that pass missed is left at NaN in its type's weights, which
/// [`check_weights`](super::check_weights) reports before the types are merged.
/// Not `+inf`, the identity of `MIN`: that is a legal weight (a closed edge),
/// so a pair left at it could not be told from a pair of closed edges.
pub(super) const UNSCORED: f64 = f64::NAN;

/// Context handed to the GraphBLAS weight operators.
///
/// Carries a borrowed pointer to the relationship [`AttributeStore`] — the
/// only graph state the operators read — rather than the whole graph. It stays
/// valid for every GraphBLAS call that reads it: [`CchInput::build`](super::CchInput::build) holds
/// `&Graph` across them, so no writer can be mutating the store, and nothing
/// GraphBLAS keeps after `build` returns holds a copy (the multi-edge pass's
/// context vector is freed with the pass).
///
/// `#[repr(C)]` plain-old-data so GraphBLAS can memcpy it as a value of the
/// user-defined type [`WeightOps::context_type`]. Every byte is a field, the
/// padding a zeroed one, so the bytes GraphBLAS copies (and may compare) are
/// all initialized.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct WeightContext {
    store: *const AttributeStore,
    weight_attribute: u16,
    padding: [u8; 6],
}

/// No implicit padding: the fields fill the struct.
const _: () = assert!(size_of::<WeightContext>() == size_of::<usize>() + 8);

impl WeightContext {
    pub(super) const fn new(
        store: &AttributeStore,
        weight_attribute: u16,
    ) -> Self {
        Self {
            store: ptr::from_ref(store),
            weight_attribute,
            padding: [0; 6],
        }
    }

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
/// `context.store` points to a live `AttributeStore` that nobody writes
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
    let weight =
        match unsafe { (*context.store).get_attr_by_idx(edge_id, context.weight_attribute) } {
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
// GraphBLAS calls each one exactly as declared here. GraphBLAS may hand them
// operands staged in byte buffers, so every pointer is read and written
// unaligned.

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
        let weight = match x.read_unaligned() {
            MULTI_EDGE => UNSCORED,
            id => cch_edge_weight(&context.read_unaligned(), id),
        };
        z.write_unaligned(weight);
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
        z.write_unaligned(cch_edge_weight(&context.read_unaligned(), jx));
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
pub(super) struct WeightOps {
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
    pub(super) fn get() -> &'static Self {
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

    /// One relationship type's cheapest weight per node pair, a matrix of the
    /// tensor's shape: a bad weight arrives as [`INVALID_WEIGHT`], for
    /// [`check_weights`](super::check_weights).
    pub(super) fn tensor_weights(
        &self,
        tensor: &Tensor,
        context: &WeightContext,
    ) -> Matrix<f64> {
        let mut weights = self.inline_pass(tensor, context);
        self.multi_edge_pass(&mut weights, tensor, context);
        weights
    }

    /// Every forward pair: the edge id is the `UINT64` value of the
    /// effective forward matrix, so one `apply` scores the lot into a matrix
    /// of the same structure. A multi-edge pair is scored [`UNSCORED`] and
    /// gets its weight from [`Self::multi_edge_pass`].
    fn inline_pass(
        &self,
        tensor: &Tensor,
        context: &WeightContext,
    ) -> Matrix<f64> {
        let effective = tensor.effective_fwd();
        let mut weights = Matrix::<f64>::new(effective.nrows(), effective.ncols());
        // SAFETY: `inline_op` maps `UINT64` to `FP64` with a `WeightContext`
        // thunk, which GraphBLAS copies and hands to `cch_inline_weight_op`,
        // under `context`'s own guarantee.
        unsafe { weights.apply_index_op(self.inline_op, &effective, context.as_value()) };
        weights
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
    /// its two endpoints), so each block's minima go straight into `weights`,
    /// in place over the [`UNSCORED`] entries the inline pass left, with
    /// nothing to merge between blocks. Every graph within 2^30 nodes per
    /// axis has the one block; a larger one has one block per occupied pair of
    /// 2^30-wide id ranges.
    fn multi_edge_pass(
        &self,
        weights: &mut Matrix<f64>,
        tensor: &Tensor,
        context: &WeightContext,
    ) {
        for (block, multi_edges) in tensor.edge_versioned_all() {
            if multi_edges.nvals() == 0 {
                continue;
            }
            let multi_edges = multi_edges.effective();
            let pair_minimums = Vector::<f64>::new(multi_edges.nrows());
            let contexts = ContextVector::new(self.context_type, multi_edges.ncols(), context);
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
                weights.min_assign(&pairs);
            }
        }
    }
}

/// Turn one block's `pair_minimums[row] = weight`, with `(block, row) =
/// compound_key(src, dst)`, into a `node_count x node_count` matrix: one
/// `extractTuples`, a host split of each key into its pair with
/// [`compound_key_inverse`], once per multi-edge **pair** (the per-edge
/// minimum is already taken), and one `GrB_Matrix_build`. The one host loop
/// of this module, by choice: see #3211.
///
/// The build has no `dup` operator: the keys are a vector's indices, so each
/// appears once, and [`compound_key_inverse`] is injective within a block, so
/// no pair repeats. A repeat would be a bug, and the build refuses it.
fn pair_weights_from_keys(
    pair_minimums: &Vector<f64>,
    block: MeBlock,
    node_count: u64,
) -> Option<Matrix<f64>> {
    let (keys, minimums) = pair_minimums.extract_tuples();
    if keys.is_empty() {
        return None;
    }
    let (rows, columns): (Vec<GrB_Index>, Vec<GrB_Index>) = keys
        .iter()
        .map(|&key| compound_key_inverse(block, key))
        .unzip();
    let mut pair_weights = Matrix::<f64>::new(node_count, node_count);
    pair_weights.build(&rows, &columns, &minimums);
    Some(pair_weights)
}
