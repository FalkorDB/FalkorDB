/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "./internal.h"
#include "GraphBLAS.h"
#include "../../graph/tensor/tensor.h"
#include <pthread.h>

//------------------------------------------------------------------------------
// Matrix_EdgeID
//
// Given a matrix A whose entries are edge weights (e.g. the output of
// project_graph_to_matrix), recover a matrix of the same shape whose entries
// are the EdgeID of the edge that produced each weight. This is the inverse
// of project_graph_to_matrix's weight-projection step.
//
// Two strategies are supported:
//   MEID_EQUAL - for A(i,j) = v, search the (possibly tensor-valued) edge
//                entry connecting i->j for the specific edge whose `weight`
//                attribute equals v (falling back to default_value when an
//                edge lacks the attribute). Disambiguates parallel edges by
//                value.
//   MEID_ANY   - for A(i,j) = v, return any EdgeID connecting i->j,
//                regardless of its weight (or A's type - a boolean
//                "presence" matrix works fine here too).
//
// rows, if non-NULL, indicates A has been compacted (the `rows` vector
// project_graph_to_matrix returns when conf.compact == true): A is indexed
// by compacted row/col ids, and rows' k-th structural index (in ascending
// order) is the original NodeID for compacted index k. We expand a copy of
// A up to the full node domain once, do all matching there against the
// relation tensors (which are always indexed by original NodeID), then
// compact the resolved EdgeID matrix back down once at the end.
//------------------------------------------------------------------------------

// context passed into the resolver ops below
typedef struct {
	const Graph   *g;              // graph
	AttributeID    attr_id;        // weight attribute to match against
	double         default_value;  // value assumed when an edge has no
	                                // weight attribute (MEID_EQUAL only)
} meid_ctx;

//------------------------------------------------------------------------------
// resolver ops
//
// Both are GxB_index_binary_function callbacks: x is the entry from A (the
// target weight, cast to FP64 regardless of A's actual type), y is the
// entry from a relation tensor's M or DP plane (a scalar EdgeID, or a
// tensor-vector-encoded set of parallel EdgeIDs - see SCALAR_ENTRY/AS_VECTOR
// in tensor.h), theta is the meid_ctx. z is the resolved EdgeID, or
// MSB_MASK ("no match here") if none of y's candidates qualify.
//
// MSB_MASK doubles as tensor.c's own "deleted, not yet compacted" marker
// (see Tensor_RemoveElements_Flat), so a raw y == MSB_MASK is already "no
// candidate" and both ops treat it that way - no separate DM-plane check is
// needed here (unlike project_graph.c, which combines whole matrices rather
// than resolving individual tensor entries).
//------------------------------------------------------------------------------

// returns the edge's `attr_id` attribute as a double, or `default_value` if
// the edge has no such attribute (or it isn't numeric)
static inline double _edge_weight
(
	const Graph *g,
	EdgeID id,
	AttributeID attr_id,
	double default_value
) {
	Edge e;
	bool found = Graph_GetEdge(g, id, &e);
	ASSERT(found == true);

	SIValue v;
	GraphEntity_GetProperty((GraphEntity *)&e, attr_id, &v);

	double w = default_value;
	if(SI_TYPE(v) & SI_NUMERIC) {
		SIValue_ToDouble(&v, &w);
	}

	return w;
}

static void _resolve_edge_id_equal
(
	uint64_t *z,           // [output] EdgeID whose weight matches x, or
	                        // MSB_MASK if none do
	const double *x,       // [input]  target weight (from A)
	GrB_Index ix,           // [unused]
	GrB_Index jx,           // [unused]
	const uint64_t *y,     // [input]  candidate EdgeID / tensor entry
	GrB_Index iy,           // [unused]
	GrB_Index jy,           // [unused]
	const meid_ctx *theta  // context: graph, attribute, default
) {
	double target = *x;
	uint64_t entry = *y;

	if(entry == MSB_MASK) {
		// deleted-entry marker - no candidate at this position
		*z = MSB_MASK;
		return;
	}

	if(SCALAR_ENTRY(entry)) {
		EdgeID id = (EdgeID) entry;
		double w = _edge_weight(theta->g, id, theta->attr_id,
				theta->default_value);
		*z = (w == target) ? id : MSB_MASK;
		return;
	}

	// tensor entry: walk the parallel edges looking for a weight match
	GrB_Vector V = AS_VECTOR(entry);
	ASSERT(V != NULL);

	struct GB_Iterator_opaque _it;
	GxB_Iterator it = &_it;
	GrB_Info info = GxB_Vector_Iterator_attach(it, V, NULL);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_Vector_Iterator_seek(it, 0);

	uint64_t match = MSB_MASK;
	while(info != GxB_EXHAUSTED) {
		EdgeID id = (EdgeID) GxB_Vector_Iterator_getIndex(it);
		double w = _edge_weight(theta->g, id, theta->attr_id,
				theta->default_value);

		if(w == target) {
			match = id;
			break;
		}

		info = GxB_Vector_Iterator_next(it);
	}

	*z = match;
}

static void _resolve_edge_id_any
(
	uint64_t *z,           // [output] any EdgeID connecting i -> j
	const double *x,       // [unused] target weight
	GrB_Index ix,           // [unused]
	GrB_Index jx,           // [unused]
	const uint64_t *y,     // [input]  candidate EdgeID / tensor entry
	GrB_Index iy,           // [unused]
	GrB_Index jy,           // [unused]
	const meid_ctx *theta  // [unused]
) {
	uint64_t entry = *y;

	if(entry == MSB_MASK) {
		*z = MSB_MASK;
		return;
	}

	if(SCALAR_ENTRY(entry)) {
		*z = entry;
		return;
	}

	GrB_Vector V = AS_VECTOR(entry);
	ASSERT(V != NULL);

	struct GB_Iterator_opaque _it;
	GxB_Iterator it = &_it;
	GrB_Info info = GxB_Vector_Iterator_attach(it, V, NULL);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_Vector_Iterator_seek(it, 0);
	ASSERT(info == GrB_SUCCESS);

	*z = (uint64_t) GxB_Vector_Iterator_getIndex(it);
}

//------------------------------------------------------------------------------
// lazy op / type creation - process lifetime, guarded the same way
// project_graph.c's _ensure_pgtm_ops is (a mutex, not pthread_once: unit
// tests cycle GrB_init/GrB_finalize within the same process, which requires
// re-running init more than once)
//------------------------------------------------------------------------------

static GrB_Type          meid_ctx_type        = NULL;
static GxB_IndexBinaryOp meid_equal_indexop    = NULL;
static GxB_IndexBinaryOp meid_any_indexop      = NULL;
static pthread_mutex_t   meid_ops_mutex        = PTHREAD_MUTEX_INITIALIZER;

static void _init_meid_ops
(
	void
) {
	GrB_OK (GrB_Type_new(&meid_ctx_type, sizeof(meid_ctx)));

	GrB_OK (GxB_IndexBinaryOp_new(&meid_equal_indexop,
			(GxB_index_binary_function) _resolve_edge_id_equal,
			GrB_UINT64, GrB_FP64, GrB_UINT64, meid_ctx_type, NULL, NULL));

	GrB_OK (GxB_IndexBinaryOp_new(&meid_any_indexop,
			(GxB_index_binary_function) _resolve_edge_id_any,
			GrB_UINT64, GrB_FP64, GrB_UINT64, meid_ctx_type, NULL, NULL));
}

static inline void _ensure_meid_ops
(
	void
) {
	pthread_mutex_lock(&meid_ops_mutex);

	if(meid_ctx_type != NULL &&
	   meid_equal_indexop != NULL &&
	   meid_any_indexop != NULL) {
		size_t t_size = 0;
		if(GxB_Type_size(&t_size, meid_ctx_type) == GrB_SUCCESS) {
			pthread_mutex_unlock(&meid_ops_mutex);
			return;
		}

		// GraphBLAS can be finalized/reinitialized in unit tests; stale
		// handles are invalid after finalize and must be recreated.
		meid_ctx_type = NULL;
		meid_equal_indexop = NULL;
		meid_any_indexop = NULL;
	}

	_init_meid_ops();
	ASSERT(meid_ctx_type != NULL);
	ASSERT(meid_equal_indexop != NULL);
	ASSERT(meid_any_indexop != NULL);

	pthread_mutex_unlock(&meid_ops_mutex);
}

// the ith relation id is i if no relation is given, and rels[i] if it is -
// matches project_graph.c/build_weighted_matrix.c's convention for
// "unfiltered means every relation type"
#define GET_RELATION_ID(i) ((rels) ? rels[i] : i)

//------------------------------------------------------------------------------
// Matrix_EdgeID
//------------------------------------------------------------------------------

GrB_Info Matrix_EdgeID
(
	GrB_Matrix *A_eid,             // [output] matrix of EdgeIDs
	const GrB_Matrix A,            // [input]  matrix of edge weights
	const Graph *g,                // graph
	const RelationID *rels,        // [optional] relationship types
	unsigned short n_rels,         // number of relationship types
	const GrB_Vector rows,         // [optional] compacted-index -> NodeID
	GRAPH_EDGE_DIR direction,      // direction A was projected with
	const AttributeID weight,      // weight attribute (MEID_EQUAL only)
	double default_value,          // default weight for attribute-less edges
	MEID_strategy strategy         // MEID_EQUAL / MEID_ANY
) {
	ASSERT(A_eid != NULL);
	ASSERT(A != NULL);
	ASSERT(g != NULL);
	ASSERT((rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0));

	_ensure_meid_ops();

	GrB_Info     info      = GrB_SUCCESS;
	GrB_Matrix   _A        = NULL;  // working copy of A's weights, full domain
	GrB_Matrix   _A_eid    = NULL;  // resolved EdgeIDs (output)
	GrB_Matrix   _A_t      = NULL;  // BOTH direction only: transpose of _A
	GrB_Matrix   _A_eid_t  = NULL;  // BOTH direction only: reverse-resolved
	GrB_Scalar   theta     = NULL;  // meid_ctx, bound into `op`
	GrB_BinaryOp op        = NULL;  // per-call bound resolver op
	GrB_Type     a_type    = NULL;

	GrB_OK (GxB_Matrix_type(&a_type, A));

	GrB_Index dim = Graph_RequiredMatrixDim(g);

	if(rows != NULL) {
		// build the expanded matrix directly from A - A is read-only here,
		// so there's no need to dup it first
		GrB_OK (GrB_Matrix_new(&_A, a_type, dim, dim));

		GrB_Descriptor desc = NULL;
		GrB_OK (GrB_Descriptor_new(&desc));
		GrB_OK (GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST));
		GrB_OK (GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST));
		GrB_OK (GxB_Matrix_assign_Vector(_A, NULL, NULL, A, rows, rows, desc));
		GrB_OK (GrB_free(&desc));

		if(direction == GRAPH_EDGE_DIR_INCOMING) {
			GrB_OK (GrB_transpose(_A, NULL, NULL, _A, NULL));
		}
	} else {
		// resize (and transpose, if needed) in one shot via assign, instead
		// of duplicating A and mutating the copy in place
		GrB_OK (GrB_Matrix_new(&_A, a_type, dim, dim));
		GrB_OK (GrB_assign(_A, NULL, NULL, A, GrB_ALL, 0, GrB_ALL, 0,
				direction == GRAPH_EDGE_DIR_INCOMING ? GrB_DESC_T0 : NULL));
	}

	meid_ctx ctx = {
		.g = g,
		.attr_id = weight,
		.default_value = default_value
	};
	GrB_OK (GrB_Scalar_new(&theta, meid_ctx_type));
	GrB_OK (GrB_Scalar_setElement_UDT(theta, (void *) &ctx));

	GxB_IndexBinaryOp resolver =
		(strategy == MEID_EQUAL) ? meid_equal_indexop : meid_any_indexop;
	GrB_OK (GxB_BinaryOp_new_IndexOp(&op, resolver, theta));

	GrB_OK (GrB_Matrix_new(&_A_eid, GrB_UINT64, dim, dim));

	// seed every position A has a value at with the "not yet resolved"
	// sentinel; every relation tensor pass below can only *decrease* this
	// (via the GrB_MIN_UINT64 accumulator, since real EdgeIDs are always
	// below MSB_MASK), so a position that ends up still holding MSB_MASK
	// after every tensor has been checked genuinely has no match
	GrB_OK (GrB_Matrix_assign_UINT64(_A_eid, _A, NULL, MSB_MASK,
			GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));

	unsigned short rel_count = (rels == NULL) ? Graph_RelationTypeCount(g) : n_rels;

	for(unsigned short i = 0; i < rel_count; i++) {
		Delta_Matrix rm = Graph_GetRelationMatrix(g, GET_RELATION_ID(i), false);
		ASSERT(rm != NULL);

		GrB_Matrix dp = Delta_Matrix_DP(rm);
		GrB_Matrix m  = Delta_Matrix_M(rm);

		// candidates for edge i -> j live at T(i,j)
		GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(_A_eid, _A_eid, GrB_MIN_UINT64,
				op, _A, dp, GrB_DESC_S));
		GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(_A_eid, _A_eid, GrB_MIN_UINT64,
				op, _A, m, GrB_DESC_S));
	}

	if(direction == GRAPH_EDGE_DIR_BOTH) {
		// A is symmetric (A(i,j) may hold the weight of the reverse edge
		// j -> i too); candidates for that edge live at T(j,i). Rather than
		// transpose every (potentially large) tensor plane on the fly once
		// per relation type, transpose the (small) working matrices once:
		// resolve against a transposed copy of _A, then transpose that
		// result back and merge it into _A_eid.
		GrB_OK (GrB_Matrix_new(&_A_t, a_type, dim, dim));
		GrB_OK (GrB_transpose(_A_t, NULL, NULL, _A, NULL));

		GrB_OK (GrB_Matrix_new(&_A_eid_t, GrB_UINT64, dim, dim));
		GrB_OK (GrB_Matrix_assign_UINT64(_A_eid_t, _A_t, NULL, MSB_MASK,
				GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));

		for(unsigned short i = 0; i < rel_count; i++) {
			Delta_Matrix rm = Graph_GetRelationMatrix(g, GET_RELATION_ID(i), false);
			ASSERT(rm != NULL);

			GrB_Matrix dp = Delta_Matrix_DP(rm);
			GrB_Matrix m  = Delta_Matrix_M(rm);

			GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(_A_eid_t, _A_eid_t,
					GrB_MIN_UINT64, op, _A_t, dp, GrB_DESC_S));
			GrB_OK (GrB_Matrix_eWiseMult_BinaryOp(_A_eid_t, _A_eid_t,
					GrB_MIN_UINT64, op, _A_t, m, GrB_DESC_S));
		}

		// transpose the reverse-resolved result back into _A_eid's
		// orientation and merge it in (MIN: a real EdgeID always beats the
		// MSB_MASK sentinel)
		GrB_OK (GrB_transpose(_A_eid, NULL, GrB_MIN_UINT64, _A_eid_t, NULL));
	}

	if(direction == GRAPH_EDGE_DIR_INCOMING) {
		GrB_OK (GrB_transpose(_A_eid, NULL, NULL, _A_eid, NULL));
	}

	if(rows != NULL) {
		GrB_Index k = 0;
		GrB_OK (GrB_Vector_nvals(&k, rows));

		GrB_Matrix compacted = NULL;
		GrB_OK (GrB_Matrix_new(&compacted, GrB_UINT64, k, k));

		GrB_Descriptor desc = NULL;
		GrB_OK (GrB_Descriptor_new(&desc));
		GrB_OK (GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST));
		GrB_OK (GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST));
		GrB_OK (GxB_Matrix_extract_Vector(compacted, NULL, NULL, _A_eid, rows, rows, desc));
		GrB_OK (GrB_free(&desc));

		GrB_OK (GrB_free(&_A_eid));
		_A_eid = compacted;
	}

	// drop positions that never resolved to a real EdgeID rather than
	// exposing the internal MSB_MASK sentinel to the caller
	GrB_OK (GrB_Matrix_select_UINT64(_A_eid, NULL, NULL, GrB_VALUENE_UINT64,
			_A_eid, MSB_MASK, NULL));

	*A_eid = _A_eid;
	_A_eid = NULL;

	GrB_free(&_A);
	GrB_free(&_A_eid);
	GrB_free(&_A_t);
	GrB_free(&_A_eid_t);
	GrB_free(&theta);
	GrB_free(&op);
	return info;
}
