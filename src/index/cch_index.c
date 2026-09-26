/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "cch_index.h"
#include "../value.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../graph/tensor/tensor.h"
#include "../serializers/serializer_io.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

// qsort comparator over RelationID (ascending)
static int _RelationID_cmp
(
	const void *a,
	const void *b
) {
	RelationID ra = *(const RelationID *)a ;
	RelationID rb = *(const RelationID *)b ;
	return (ra > rb) - (ra < rb) ;
}

// write a sorted, de-duplicated copy of 'src' (n entries) into 'dst' (capacity
// >= n) and return the number of unique entries. the relationship-type set is
// the index's identity, so 'ROAD|ROAD' must canonicalize to 'ROAD' -- otherwise
// {ROAD} and {ROAD,ROAD} would masquerade as two different indices.
static uint _sort_unique
(
	RelationID       *dst,   // [out] sorted unique relationship types
	const RelationID *src,   // requested relationship types
	uint              n      // number of requested relationship types
) {
	memcpy (dst, src, n * sizeof (RelationID)) ;
	qsort (dst, n, sizeof (RelationID), _RelationID_cmp) ;

	uint k = 0 ;
	for (uint i = 0 ; i < n ; i++) {
		if (k == 0 || dst [i] != dst [k - 1]) {
			dst [k++] = dst [i] ;
		}
	}
	return k ;
}

CCHIndex *CCHIndex_New
(
	const RelationID *rel_types,   // relationship types the hierarchy spans
	uint              n,           // number of relationship types
	AttributeID       weight_attr  // edge weight attribute
) {
	ASSERT (rel_types != NULL) ;
	ASSERT (n > 0) ;

	CCHIndex *idx = rm_calloc (1, sizeof (CCHIndex)) ;

	// store the relationship types sorted AND de-duplicated -- the set is the
	// index's identity, so a repeated type (ROAD|ROAD) canonicalizes to one
	RelationID *sorted = arr_newlen (RelationID, n) ;
	uint k = _sort_unique (sorted, rel_types, n) ;
	arr_hdr_t *hdr = arr_hdr (sorted) ;
	hdr->len = k ;                    // shrink to the unique count
	idx->rel_types = sorted ;

	idx->weight_attr = weight_attr ;
	idx->cch         = NULL ;   // built later
	idx->dirty           = CCH_DIRTY_CLEAN ;
	idx->dirty_u         = arr_new (NodeID, 0) ;
	idx->dirty_v         = arr_new (NodeID, 0) ;
	idx->pending_rebuild = false ;
	idx->stale           = 0 ;

	return idx ;
}

bool CCHIndex_Matches
(
	const CCHIndex   *idx,         // index to test
	const RelationID *rel_types,   // requested relationship types
	uint              n,           // number of requested relationship types
	AttributeID       weight_attr  // requested weight attribute
) {
	ASSERT (idx != NULL) ;

	if (idx->weight_attr != weight_attr) return false ;

	// canonicalize the request (sort + dedup) and compare against the stored
	// canonical set, so ROAD|ROAD matches an index over {ROAD}
	RelationID *tmp = arr_newlen (RelationID, n) ;
	uint m = _sort_unique (tmp, rel_types, n) ;

	bool eq = (arr_len (idx->rel_types) == m) &&
	          (memcmp (idx->rel_types, tmp, m * sizeof (RelationID)) == 0) ;

	arr_free (tmp) ;
	return eq ;
}

bool CCHIndex_CoversRelation
(
	const CCHIndex *idx,  // index to test
	RelationID      r     // relationship type
) {
	ASSERT (idx != NULL) ;

	uint n = arr_len (idx->rel_types) ;
	for (uint i = 0; i < n; i++) {
		if (idx->rel_types [i] == r) {
			return true ;
		}
	}

	return false ;
}

bool CCHIndex_Built
(
	const CCHIndex *idx  // index to test
) {
	ASSERT (idx != NULL) ;
	return idx->cch != NULL ;
}

const RelationID *CCHIndex_RelTypes
(
	const CCHIndex *idx  // index
) {
	ASSERT (idx != NULL) ;
	return idx->rel_types ;
}

uint CCHIndex_RelTypeCount
(
	const CCHIndex *idx  // index
) {
	ASSERT (idx != NULL) ;
	return arr_len (idx->rel_types) ;
}

AttributeID CCHIndex_WeightAttr
(
	const CCHIndex *idx  // index
) {
	ASSERT (idx != NULL) ;
	return idx->weight_attr ;
}

void CCHIndex_RdbSave
(
	const CCHIndex *idx,
	SerializerIO    io
) {
	ASSERT (idx      != NULL) ;
	ASSERT (idx->cch != NULL) ;   // only built indices are serialized

	// identity: relationship types + weight attribute
	uint n = arr_len (idx->rel_types) ;
	SerializerIO_WriteUnsigned (io, n) ;
	for (uint i = 0 ; i < n ; i++) {
		SerializerIO_WriteUnsigned (io, idx->rel_types [i]) ;
	}
	SerializerIO_WriteUnsigned (io, idx->weight_attr) ;

	// the resident hierarchy
	CCH_RdbSave (idx->cch, io) ;
}

CCHIndex *CCHIndex_RdbLoad
(
	SerializerIO io
) {
	// identity
	uint n = SerializerIO_ReadUnsigned (io) ;
	RelationID *rels = arr_new (RelationID, n) ;
	for (uint i = 0 ; i < n ; i++) {
		arr_append (rels, (RelationID) SerializerIO_ReadUnsigned (io)) ;
	}
	AttributeID weight_attr = (AttributeID) SerializerIO_ReadUnsigned (io) ;

	// the resident hierarchy -- attached directly, no rebuild
	CCH *cch = CCH_RdbLoad (io) ;

	CCHIndex *idx = CCHIndex_New (rels, arr_len (rels), weight_attr) ;
	idx->cch = cch ;

	arr_free (rels) ;
	return idx ;
}

size_t CCHIndex_MemoryUsage
(
	const CCHIndex *idx
) {
	ASSERT (idx != NULL) ;

	size_t sz = RedisModule_MallocSize ((void *) idx) ;

	if (idx->rel_types != NULL) sz += arr_bytesize (idx->rel_types) ;
	if (idx->dirty_u   != NULL) sz += arr_bytesize (idx->dirty_u)   ;
	if (idx->dirty_v   != NULL) sz += arr_bytesize (idx->dirty_v)   ;

	sz += CCH_MemoryUsage (idx->cch) ;   // resident hierarchy (NULL-safe); the bulk

	return sz ;
}

void CCHIndex_Free
(
	CCHIndex *idx
) {
	if (idx == NULL) {
		return ;
	}

	if (idx->rel_types != NULL) {
		arr_free (idx->rel_types) ;
	}

	if (idx->dirty_u != NULL) {
		arr_free (idx->dirty_u) ;
	}

	if (idx->dirty_v != NULL) {
		arr_free (idx->dirty_v) ;
	}

	if (idx->cch != NULL) {
		CCH_Free (idx->cch) ;
	}

	rm_free (idx) ;
}

//------------------------------------------------------------------------------
// hierarchy construction
//------------------------------------------------------------------------------

// resolves a single edge's weight, defaulting to 1 if 'attr_id' is
// missing/non-numeric on this particular edge -- matches Dijkstra/AStar's
// per-edge fallback convention
static double _edge_weight
(
	const Graph *g,       // graph owning the edge
	AttributeID attr_id,  // weight attribute to read
	EdgeID id             // edge whose weight is resolved
) {
	Edge e ;
	bool found = Graph_GetEdge (g, id, &e) ;
	ASSERT (found == true) ;

	SIValue w = GraphEntity_GetNumericPropertyOrDefault ((GraphEntity *)&e, attr_id,
			SI_LongVal (1)) ;
	return SI_GET_NUMERIC (w) ;
}

// context for the GraphBLAS IndexUnaryOp that resolves each matrix entry to a
// weight
typedef struct {
	const Graph *g;       // graph being queried
	AttributeID attr_id;  // attribute id that holds the weight
} EdgeWeightContext;

// GraphBLAS IndexUnaryOp callback: reads the weight attribute off the edge(s)
// at (i,j) and writes it to *z. a tensor cell is either a scalar EdgeID
// (SCALAR_ENTRY) or, for parallel edges, a GrB_Vector of EdgeIDs (AS_VECTOR) --
// in the latter case the cheapest parallel edge wins.
static void _get_edge_weight
(
	double *z,                    // [output] weight value
	const void *x,                // entry value (EdgeID, scalar or tagged vector)
	GrB_Index i,                  // row index -- unused
	GrB_Index j,                  // col index -- unused
	const EdgeWeightContext *ctx  // user-supplied context (theta)
) {
	uint64_t entry = *(const uint64_t *)x ;

	if (SCALAR_ENTRY (entry)) {
		*z = _edge_weight (ctx->g, ctx->attr_id, (EdgeID)entry) ;
		return ;
	}

	// multi-edge cell: the vector's stored indices are the parallel edges'
	// EdgeIDs -- take the cheapest
	GrB_Vector ids = AS_VECTOR (entry) ;

	struct GB_Iterator_opaque _it ;
	GxB_Iterator it = &_it ;
	GrB_OK (GxB_Vector_Iterator_attach (it, ids, NULL)) ;

	double min_w = INFINITY ;
	GrB_Info info = GxB_Vector_Iterator_seek (it, 0) ;
	while (info != GxB_EXHAUSTED) {
		EdgeID id = (EdgeID) GxB_Vector_Iterator_getIndex (it) ;
		double w = _edge_weight (ctx->g, ctx->attr_id, id) ;
		if (w < min_w) {
			min_w = w ;
		}
		info = GxB_Vector_Iterator_next (it) ;
	}

	*z = min_w ;
}

static GrB_Type         ctx_type    = NULL              ;
static GrB_IndexUnaryOp get_weight  = NULL              ;
static pthread_once_t index_op_once = PTHREAD_ONCE_INIT ;

static void _init_tensor_ops
(
	void
) {
	GrB_OK (GrB_Type_new (&ctx_type, sizeof (EdgeWeightContext))) ;

	GrB_OK (GrB_IndexUnaryOp_new (&get_weight,
			(GxB_index_unary_function)_get_edge_weight, GrB_FP64, GrB_UINT64,
			ctx_type)) ;
}

GrB_Matrix CCH_BuildWeightMatrix
(
	Graph *g,                      // graph providing the relation matrices
	const RelationID *relTypeIDs,  // relation types forming the sub-graph
	uint relCount,                 // number of relation types
	AttributeID weightAtt          // edge attribute holding the weight
) {
	GrB_Index dim = Graph_RequiredMatrixDim (g) ;

	GrB_Matrix A_w = NULL ;
	GrB_OK (GrB_Matrix_new (&A_w, GrB_FP64, dim, dim)) ;

	EdgeWeightContext w_ctx = { .g = g, .attr_id = weightAtt } ;

	GrB_Scalar ctx_scalar = NULL ;

	pthread_once (&index_op_once, _init_tensor_ops) ;

	GrB_OK (GrB_Scalar_new (&ctx_scalar, ctx_type)) ;
	GrB_OK (GrB_Scalar_setElement_UDT (ctx_scalar, (void *)&w_ctx)) ;

	for (uint r = 0; r < relCount; r++) {
		Delta_Matrix R = Graph_GetRelationMatrix (g, relTypeIDs [r], false) ;

		GrB_Matrix U = NULL ;
		GrB_OK (Delta_Matrix_export (&U, R, GrB_UINT64, NULL)) ;

		GrB_Matrix Wr = NULL ;
		GrB_OK (GrB_Matrix_new (&Wr, GrB_FP64, dim, dim)) ;
		GrB_OK (GrB_Matrix_apply_IndexOp_Scalar (Wr, NULL, NULL, get_weight, U,
					ctx_scalar, NULL)) ;
		GrB_OK (GrB_free (&U)) ;

		// combine relation types by taking the cheapest parallel edge
		GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp (A_w, NULL, NULL, GrB_MIN_FP64,
					A_w, Wr, NULL)) ;
		GrB_OK (GrB_free (&Wr)) ;
	}

	GrB_OK (GrB_free (&ctx_scalar)) ;

	return A_w ;
}

// drop the accumulated dirty-arc scope (called whenever the index goes clean)
static void _clear_dirty_arcs
(
	CCHIndex *idx
) {
	arr_clear (idx->dirty_u) ;
	arr_clear (idx->dirty_v) ;
}

void CCHIndex_Build
(
	CCHIndex *idx,  // index whose hierarchy to (re)build
	Graph    *g     // graph to build from
) {
	ASSERT (idx != NULL) ;
	ASSERT (g   != NULL) ;

	// discard any previously-built hierarchy
	if (idx->cch != NULL) {
		CCH_Free (idx->cch) ;
		idx->cch = NULL ;
	}

	GrB_Matrix W = CCH_BuildWeightMatrix (g, idx->rel_types,
			arr_len (idx->rel_types), idx->weight_attr) ;

	int64_t dim = (int64_t) Graph_RequiredMatrixDim (g) ;

	CCH *cch = CCH_New (dim) ;
	CCH_EliminationOrder     (cch, W) ;   // Phase 1a (metric-independent)
	CCH_ChordalTriangulation (cch) ;      // Phase 1b (metric-independent)
	CCH_Customize            (cch, W) ;   // Phase 2  (metric)

	GrB_OK (GrB_free (&W)) ;

	idx->cch             = cch ;
	idx->dirty           = CCH_DIRTY_CLEAN ;
	idx->pending_rebuild = false ;   // structure is fresh for the current topology
	idx->stale           = 0 ;       // stale fill-in reclaimed
	_clear_dirty_arcs (idx) ;        // a full rebuild supersedes any scoped work
}

// cheapest metric weight of a directed edge from -> to across the index's
// relationship types (parallel edges collapse to their minimum, +INFINITY when
// none) -- reproduces exactly what CCH_BuildWeightMatrix seeds into W, so a
// scoped recustomization seeds identically
static double _min_edge_weight
(
	const Graph      *g,
	const RelationID *rels,
	uint              relCount,
	AttributeID       weightAtt,
	NodeID            from,
	NodeID            to
) {
	Edge *tmp = arr_new (Edge, 4) ;

	for (uint i = 0 ; i < relCount ; i++) {
		Graph_GetEdgesConnectingNodes (g, from, to, rels [i], &tmp) ;
	}

	double best = INFINITY ;
	for (uint32_t i = 0 ; i < arr_len (tmp) ; i++) {
		double w = _edge_weight (g, weightAtt, tmp [i].id) ;
		if (w < best) best = w ;
	}

	arr_free (tmp) ;
	return best ;
}

// context + callback handed to CCH_RecustomizeScoped for per-arc seed lookups
typedef struct {
	const Graph      *g;          // graph providing edges
	const RelationID *rels;       // relationship types the index spans
	uint              relCount;   // number of relationship types
	AttributeID       weightAtt;  // weight attribute
} ScopedSeedCtx;

static void _scoped_seed
(
	void   *vctx,
	int64_t u,
	int64_t v,
	double *w_uv,
	double *w_vu
) {
	const ScopedSeedCtx *c = vctx ;
	*w_uv = _min_edge_weight (c->g, c->rels, c->relCount, c->weightAtt,
			(NodeID)u, (NodeID)v) ;
	*w_vu = _min_edge_weight (c->g, c->rels, c->relCount, c->weightAtt,
			(NodeID)v, (NodeID)u) ;
}

// deletions leave stale chordal arcs behind; rebuild to reclaim them once this
// many have accumulated. amortizes the rebuild cost over many deletions, with a
// floor so small graphs don't rebuild on every handful of deletes
static uint64_t _stale_threshold
(
	int64_t n
) {
	uint64_t t = (uint64_t) n / 8 ;
	return t < 1024 ? 1024 : t ;
}

void CCHIndex_Recustomize
(
	CCHIndex *idx,  // index to re-weight (topology + order unchanged)
	Graph    *g     // graph to read weights from
) {
	ASSERT (idx != NULL) ;
	ASSERT (g   != NULL) ;
	ASSERT (idx->cch != NULL) ;   // recustomize requires an existing hierarchy

	CCH *cch = idx->cch ;

	// escalate to a full rebuild when a new adjacency was added (the chordal
	// structure must grow) or when enough deletions have piled up stale fill-in
	// (the staleness valve) -- the rebuild also restores elimination-order quality
	if (idx->pending_rebuild || idx->stale >= _stale_threshold (cch->n)) {
		CCHIndex_Build (idx, g) ;   // resets dirty / pending_rebuild / stale
		return ;
	}

	uint64_t k = arr_len (idx->dirty_u) ;   // number of changed arcs

	// the scoped incremental path pays off while the changed set is small; once it
	// spans a large fraction of the hierarchy the affected cones overlap so heavily
	// that a single full customization is simpler and no slower (topology is
	// unchanged here, so no METIS). 'n/8' is a coarse "bulk update" line
	if (k <= (uint64_t) cch->n / 8) {
		// incrementally re-customize just the affected cone of the changed arcs
		ScopedSeedCtx sctx = {
			.g         = g,
			.rels      = idx->rel_types,
			.relCount  = arr_len (idx->rel_types),
			.weightAtt = idx->weight_attr
		} ;

		CCH_RecustomizeScoped (cch, _scoped_seed, &sctx,
				(const int64_t *) idx->dirty_u, (const int64_t *) idx->dirty_v, k) ;
	} else {
		// bulk metric change: rebuild the weight matrix and re-run the full Phase 2
		GrB_Matrix W = CCH_BuildWeightMatrix (g, idx->rel_types,
				arr_len (idx->rel_types), idx->weight_attr) ;
		CCH_Customize (cch, W) ;
		GrB_OK (GrB_free (&W)) ;
	}

	idx->dirty = CCH_DIRTY_CLEAN ;
	_clear_dirty_arcs (idx) ;
}

void CCHIndex_MarkDirty
(
	CCHIndex     *idx,   // index to mark
	CCHDirtyLevel level  // maintenance level (a higher level subsumes a lower)
) {
	ASSERT (idx != NULL) ;
	if (level > idx->dirty) {
		idx->dirty = level ;
	}
}

void CCHIndex_MarkArcDirty
(
	CCHIndex *idx,  // index to mark
	NodeID    u,    // arc source node id
	NodeID    v     // arc destination node id
) {
	ASSERT (idx != NULL) ;

	arr_append (idx->dirty_u, u) ;
	arr_append (idx->dirty_v, v) ;

	if (CCH_DIRTY_RECUSTOMIZE > idx->dirty) {
		idx->dirty = CCH_DIRTY_RECUSTOMIZE ;
	}
}

void CCHIndex_MarkEdgeAdded
(
	CCHIndex *idx,  // index to mark
	NodeID    u,    // edge source node id
	NodeID    v     // edge destination node id
) {
	ASSERT (idx != NULL) ;

	// once a rebuild is already pending, every further addition is subsumed by it:
	// just keep the index flagged dirty so the flush runs, without growing the
	// dirty-arc set (bounds memory for a bulk create of new adjacencies)
	if (idx->pending_rebuild) {
		CCHIndex_MarkDirty (idx, CCH_DIRTY_RECUSTOMIZE) ;
		return ;
	}

	CCHIndex_MarkArcDirty (idx, u, v) ;   // scoped path when the arc already exists

	// a pair with no chordal arc yet is a brand-new adjacency: the structure must
	// grow, which only a full rebuild does. also force one if an endpoint falls
	// outside the built rank space (a node was added alongside this edge)
	CCH *cch = idx->cch ;
	if (cch == NULL || (int64_t) u >= cch->n || (int64_t) v >= cch->n ||
			!CCH_HasArc (cch, cch->iperm [u], cch->iperm [v])) {
		idx->pending_rebuild = true ;
	}
}

void CCHIndex_MarkEdgeDeleted
(
	CCHIndex *idx,  // index to mark
	NodeID    u,    // edge source node id
	NodeID    v     // edge destination node id
) {
	ASSERT (idx != NULL) ;

	// keep the chordal arc; it is re-seeded from the (post-deletion) graph on the
	// next flush. count the deletion toward the staleness valve
	CCHIndex_MarkArcDirty (idx, u, v) ;
	idx->stale++ ;
}

void CCHIndex_MarkClean
(
	CCHIndex *idx  // index to clear
) {
	ASSERT (idx != NULL) ;
	idx->dirty           = CCH_DIRTY_CLEAN ;
	idx->pending_rebuild = false ;   // discard a rolled-back new-adjacency's flag
	_clear_dirty_arcs (idx) ;
}
