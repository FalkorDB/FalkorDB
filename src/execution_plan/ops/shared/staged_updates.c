/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "../../../util/dict.h"
#include "../../../query_ctx.h"
#include "./update_functions.h"

// used by both op_update and op_merge to maintain staged updates
// property updates: SET n.v = n.v+1
// label updates:    SET n:X / REMOVE n:X
typedef struct StagedUpdatesCtx {
	dict *node_updates;          // dict mapping node id to PendingUpdateCtx
	dict *edge_updates;          // dict mapping edge id to PendingUpdateCtx
	GrB_Vector *added_labels;    // array of GrB_Vector denoting labels to add
	uint8_t n_added_labels;      // number of added_labels vectors
	GrB_Vector *removed_labels;  // array of GrB_Vector denoting labels to remove
	uint8_t n_removed_labels;    // number of removed_labels vectors
} StagedUpdatesCtx;

// hash of key is simply key
static uint64_t _id_hash
(
	const void *key
) {
	return ((uint64_t)key);
}

// hashtable value destructor callback
static void dictValDest
(
	dict *d,
	void *val
) {
	PendingUpdateCtx_Free ((PendingUpdateCtx*)val) ;
}

// hashtable callbacks
static dictType _dt = {_id_hash, NULL, NULL, NULL, NULL, dictValDest, NULL,
	NULL, NULL, NULL} ;

StagedUpdatesCtx *StagedUpdatesCtx_New (void) {
	StagedUpdatesCtx *ctx = rm_calloc (1, sizeof (StagedUpdatesCtx)) ;

	ctx->node_updates     = HashTableCreate (&_dt) ;
	ctx->edge_updates     = HashTableCreate (&_dt) ;
	ctx->added_labels     = NULL ;
	ctx->n_added_labels   = 0 ;
	ctx->removed_labels   = NULL ;
	ctx->n_removed_labels = 0 ;

	return ctx ;
}

// get staged node updates
dict *StagedUpdatesCtx_NodeUpdates
(
    StagedUpdatesCtx *ctx  // staged updates context
) {
    ASSERT (ctx               != NULL) ;
    ASSERT (ctx->node_updates != NULL) ;

    return ctx->node_updates ;
}

// get staged edge updates
dict *StagedUpdatesCtx_EdgeUpdates
(
    StagedUpdatesCtx *ctx  // staged updates context
) {
    ASSERT (ctx               != NULL) ;
    ASSERT (ctx->edge_updates != NULL) ;

    return ctx->edge_updates ;
}

// returns true if there are pending node updates
bool StagedUpdatesCtx_HasNodeUpdates
(
	const StagedUpdatesCtx *ctx  // staged updates context
) {
	ASSERT (ctx != NULL) ;

	return (ctx->n_added_labels   > 0 ||
			ctx->n_removed_labels > 0 ||
			HashTableElemCount (ctx->node_updates) > 0) ;
}

// returns true if there are pending edge updates
bool StagedUpdatesCtx_HasEdgeUpdates
(
	const StagedUpdatesCtx *ctx  // staged updates context
) {
	ASSERT (ctx != NULL) ;

	return (HashTableElemCount (ctx->edge_updates) > 0) ;
}

// retrieve an entity update context
PendingUpdateCtx *StagedUpdatesCtx_GetEntityUpdateCtx
(
	StagedUpdatesCtx *ctx,  // staged updates
    GraphEntity *e,         // entity
	GraphEntityType t       // entity type Node/Edge
) {
	ASSERT (e   != NULL) ;
	ASSERT (ctx != NULL) ;
	ASSERT (t == GETYPE_NODE || t == GETYPE_EDGE) ;

	dict *updates ;

	updates = (t == GETYPE_NODE) ?
		StagedUpdatesCtx_NodeUpdates (ctx) :
		StagedUpdatesCtx_EdgeUpdates (ctx) ;

	// does entity already has an entry ?
	PendingUpdateCtx *ret ;
	dictEntry *entry = HashTableFind (updates,
			(void *)ENTITY_GET_ID (e)) ;

	if (entry == NULL) {
		// create a new update context
		ret = rm_malloc (sizeof (PendingUpdateCtx)) ;
		ret->ge = e ;
		ret->attributes = AttributeSet_ShallowClone (*e->attributes) ;
		// add update context to updates dictionary
		HashTableAdd (updates, (void *)ENTITY_GET_ID (e), ret) ;
	} else {
		// update context already exists
		ret = (PendingUpdateCtx *) HashTableGetVal (entry) ;
	}

	ASSERT (ret != NULL) ;
	return ret ;
}

#define REDUNDANCY_ITER_THRESHOLD 512

// clears entries from 'v' that would be redundant label operations:
// - if addition=true:  clears v[i] when node i already carries 'label'
// - if addition=false: clears v[i] when node i does not carry 'label'
static void _RemoveRedundancies
(
	GrB_Vector    v,         // node vector to filter in-place
	bool          addition,  // true = SET n:L, false = REMOVE n:L
	GraphContext *gc,        // graph context (for schema lookup)
	Graph        *g,         // graph (for label matrix access)
	const char   *label      // label name to check against
) {
	Schema *schema = GraphContext_GetSchema (gc, label, SCHEMA_NODE) ;
	ASSERT (schema != NULL) ;

	LabelID lbl_id = Schema_GetID (schema) ;
	Delta_Matrix L = Graph_GetLabelMatrix (g, lbl_id) ;

	GrB_Index nvals ;
	GrB_OK (GrB_Vector_nvals (&nvals, v)) ;

	if (nvals < REDUNDANCY_ITER_THRESHOLD) {
		GxB_Iterator it ;
		GxB_Iterator_new (&it) ;

		GrB_Info info = GxB_Vector_Iterator_attach (it, v, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;

		// seek to the first entry
		info = GxB_Vector_Iterator_seek (it, 0) ;
		bool redundancies = false ;

		while (info != GxB_EXHAUSTED) {
			// get the entry v(i)
			GrB_Index node_id = GxB_Vector_Iterator_getIndex (it) ;
			bool is_set = (Delta_Matrix_isStoredElement (L, node_id, node_id)
				 == GrB_SUCCESS) ;

			if (is_set && addition) {
				// node already carries the label — adding it again is redundant
				GrB_OK (GrB_Vector_setElement (v, false, node_id)) ;
				redundancies = true ;
			}

			else if (!is_set && !addition) {
				// node does not carry the label — removing it is redundant
				GrB_OK (GrB_Vector_setElement (v, false, node_id)) ;
				redundancies = true ;
			}

			// move to the next entry in v
			info = GxB_Vector_Iterator_next (it) ;
		}

		GrB_free (&it) ;

		// clear redundancies
		if (redundancies) {
			GxB_Scalar s ;
			GrB_OK (GxB_Scalar_new (&s, GrB_BOOL)) ;
			GrB_OK (GxB_Scalar_setElement_BOOL (s, true)) ;
			GrB_OK (GrB_select (v, NULL, NULL, GrB_VALUEEQ_BOOL, v, s, NULL)) ;
			GrB_OK (GrB_free (&s)) ;
		}

		return ;
	}

	//--------------------------------------------------------------------------
    // bulk path: diagonal extraction + descriptor-based masking
    //--------------------------------------------------------------------------

    GrB_Matrix M  = Delta_Matrix_M  (L) ;
    GrB_Matrix DP = Delta_Matrix_DP (L) ;
    GrB_Matrix DM = Delta_Matrix_DM (L) ;

    GrB_Index nrows ;
    GrB_OK (Delta_Matrix_nrows (&nrows, L)) ;

    GrB_Vector M_V  = NULL ;
    GrB_Vector DM_V = NULL ;
    GrB_Vector DP_V = NULL ;

	//--------------------------------------------------------------------------
	// extract M, DM and DP main diagonals
	//--------------------------------------------------------------------------

    GrB_OK (GrB_Vector_new (&M_V,  GrB_BOOL, nrows)) ;
    GrB_OK (GrB_Vector_new (&DP_V, GrB_BOOL, nrows)) ;
    GrB_OK (GrB_Vector_new (&DM_V, GrB_BOOL, nrows)) ;

    GrB_OK (GxB_Vector_diag (M_V,  M,  0, NULL)) ;
    GrB_OK (GxB_Vector_diag (DP_V, DP, 0, NULL)) ;
    GrB_OK (GxB_Vector_diag (DM_V, DM, 0, NULL)) ;

	// computing (M ∪ DP) \ DM
	GrB_OK (GrB_eWiseAdd (M_V, DM_V, NULL, GrB_ONEB_BOOL, M_V, DP_V,
				GrB_DESC_RSC)) ;

	// for addition: clear v[i] where L[i,i] is already set (GrB_DESC_RSCT0)
	// for removal:  clear v[i] where L[i,i] is not set     (GrB_DESC_RST0)
    GrB_Descriptor desc = (addition) ? GrB_DESC_RSCT0 : GrB_DESC_RST0 ;

    GrB_OK (GrB_transpose ((GrB_Matrix) v, (const GrB_Matrix) M_V, NULL,
                (GrB_Matrix) v, desc)) ;

    GrB_OK (GrB_free (&M_V))  ;
    GrB_OK (GrB_free (&DP_V)) ;
    GrB_OK (GrB_free (&DM_V)) ;
}

// populate either added_labels or removed_labels depending on `add`
// creates the label vector if it does not already exist
static void _PopulateVector
(
	StagedUpdatesCtx *ctx,  // staged update context
	bool add,               // true for label addition, false for removal
	NodeID *node_ids,       // node IDs
	uint32_t node_count,    // number of nodes
	char *label             // label to add or remove
) {
	ASSERT (ctx        != NULL) ;
	ASSERT (label      != NULL) ;
	ASSERT (node_ids   != NULL) ;
	ASSERT (node_count > 0) ;

	GrB_Vector   v         = NULL ;
	char         name[512] = {0}  ;
	GrB_Vector **vecs      = NULL ;
	uint8_t     *vec_count = NULL ;

	GraphContext *gc = QueryCtx_GetGraphCtx  () ;
	Graph        *g  = GraphContext_GetGraph (gc) ;

	if (add) {
		vecs      = &ctx->added_labels     ;
		vec_count = &ctx->n_added_labels   ;
	} else {
		vecs      = &ctx->removed_labels   ;
		vec_count = &ctx->n_removed_labels ;
	}

	// search for existing label vector
	for (uint8_t i = 0 ; i < *vec_count ; i++) {
		GrB_OK (GrB_get ((*vecs) [i], name, GrB_NAME)) ;

		if (strcmp (name, label) == 0) {
			v = (*vecs) [i] ;
			break ;
		}
	}

	// create and name vector if not found
	if (v == NULL) {
		(*vec_count)++ ;
		*vecs = rm_realloc (*vecs, sizeof (GrB_Vector) * (*vec_count)) ;

		GrB_OK (GrB_Vector_new (&(*vecs) [*vec_count - 1], GrB_BOOL,
					Graph_NodeCap (g))) ;

		GrB_OK (GrB_set ((*vecs) [*vec_count - 1], label, GrB_NAME)) ;
		v = (*vecs) [*vec_count - 1] ;
	}

	ASSERT (v != NULL) ;

    //--------------------------------------------------------------------------
	// populate vector with node IDs
    //--------------------------------------------------------------------------

	GxB_Scalar s ;
	GrB_OK (GxB_Scalar_new (&s, GrB_BOOL)) ;
	GrB_OK (GxB_Scalar_setElement_BOOL (s, true)) ;

	GrB_OK (GrB_assign (v, NULL, NULL, s, (GrB_Index*) node_ids, node_count,
				NULL)) ;

	GrB_OK (GrB_free (&s)) ;

	// make sure vector `v` does not contains redundant information
	_RemoveRedundancies (v, add, gc, g, label) ;
}

void StagedUpdatesCtx_LabelNodes
(
	StagedUpdatesCtx *ctx,  // staged update context
	NodeID *node_ids,       // node IDs
	uint32_t node_count,    // number of nodes
	char *label             // label to add
) {
	ASSERT (ctx        != NULL) ;
	ASSERT (label      != NULL) ;
	ASSERT (node_ids   != NULL) ;
	ASSERT (node_count > 0) ;

	_PopulateVector (ctx, true, node_ids, node_count, label) ;
}

void StagedUpdatesCtx_UnLabelNodes
(
	StagedUpdatesCtx *ctx,  // staged update context
	NodeID *node_ids,       // node IDs
	uint32_t node_count,    // number of nodes
	char *label             // label to remove
) {
	ASSERT (ctx        != NULL) ;
	ASSERT (label      != NULL) ;
	ASSERT (node_ids   != NULL) ;
	ASSERT (node_count > 0) ;

	_PopulateVector (ctx, false, node_ids, node_count, label) ;
}

// returns number of different added labels
uint8_t StagedUpdatesCtx_AddLabelCount
(
    const StagedUpdatesCtx *ctx  // staged update context
) {
    ASSERT (ctx != NULL) ;

    return ctx->n_added_labels ;
}

// returns number of different removed labels
uint8_t StagedUpdatesCtx_RmvLabelCount
(
    const StagedUpdatesCtx *ctx  // staged update context
) {
    ASSERT (ctx != NULL) ;

    return ctx->n_removed_labels ;
}

// returns an array of added label vectors
GrB_Vector *StagedUpdatesCtx_AddLabels
(
    StagedUpdatesCtx *ctx  // staged update context
) {
    ASSERT (ctx != NULL) ;

    return ctx->added_labels ;
}

// returns an array of removed label vectors
GrB_Vector *StagedUpdatesCtx_RmvLabels
(
    StagedUpdatesCtx *ctx  // staged update context
) {
    ASSERT (ctx != NULL) ;

    return ctx->removed_labels ;
}

// free staged update context
void StagedUpdatesCtx_Free
(
	StagedUpdatesCtx **ctx  // staged updates context to free
) {
	ASSERT (ctx != NULL) ;

	if (*ctx == NULL) {
		return ;
	}

	StagedUpdatesCtx *_ctx = *ctx ;

	HashTableRelease (_ctx->node_updates) ;
	HashTableRelease (_ctx->edge_updates) ;

	if (_ctx->added_labels != NULL) {
		ASSERT (_ctx->n_added_labels > 0) ;
		for (uint i = 0 ; i < _ctx->n_added_labels ; i++) {
			GrB_Vector v = _ctx->added_labels [i] ;
			if (v != NULL) {
				GrB_free (&v) ;
			}
		}
		rm_free (_ctx->added_labels) ;
	}

	if (_ctx->removed_labels != NULL) {
		ASSERT (_ctx->n_removed_labels > 0) ;
		for (uint i = 0 ; i < _ctx->n_removed_labels ; i++) {
			GrB_Vector v = _ctx->removed_labels [i] ;
			if (v != NULL) {
				GrB_free (&v) ;
			}
		}
		rm_free (_ctx->removed_labels) ;
	}

	rm_free (_ctx) ;
	*ctx = NULL ;
}

