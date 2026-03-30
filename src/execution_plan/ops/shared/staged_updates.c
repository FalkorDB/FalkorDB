/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// used by both op_update and op_merge to maintain staged updates
// property updates: SET n.v = n.v+1
// label updates:    SET n:X / REMOVE n:X
typedef struct {
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

// populate either added_labels or removed_labels depending on `add`
// creates the label vector if it does not already exist
static void _PopulateVector
(
	StagedUpdatesCtx *ctx,  // staged update context
	bool add,               // true for label addition, false for removal
	NodeID *node_ids,       // node IDs
	uint32_t node_count,    // number of nodes
	const char *label       // label to add or remove
) {
	ASSERT (ctx        != NULL) ;
	ASSERT (label      != NULL) ;
	ASSERT (node_ids   != NULL) ;
	ASSERT (node_count > 0) ;

	GrB_Vector   v         = NULL ;
	char         name[512] = {0}  ;
	GrB_Vector **vecs      = NULL ;
	uint8_t     *vec_count = NULL ;

	if (add) {
		vecs      = &ctx->added_labels     ;
		vec_count = &ctx->n_added_labels   ;	
	} else {
		vecs      = &ctx->removed_labels   ;
		vec_count = &ctx->n_removed_labels ;	
	}

	// search for existing label vector
	for (uint8_t i = 0 ; i < *vec_count ; i++) {
		GrB_OK (GrB_get ((*vecs)[i], name, GrB_NAME)) ;

		if (strcmp (name, label) == 0) {
			v = (*vecs)[i] ;
			break ;
		}
	}

	// create and name vector if not found
	if (v == NULL) {
		(*vec_count)++ ;
		*vecs = rm_realloc (*vecs, sizeof (GrB_Vector) * (*vec_count)) ;

		GrB_OK (GrB_Vector_new (&(*vecs) [*vec_count - 1], GrB_BOOL,
					Graph_UncompactedNodeCount (g))) ;

		GrB_OK (GrB_set ((*vecs)[*vec_count - 1], label, GrB_NAME)) ;
		v = (*vecs)[*vec_count - 1] ;
	}

	ASSERT (v != NULL) ;

	// populate vector with node IDs
	GxB_Scalar s ;
	GrB_OK (GxB_Scalar_new (&s, GrB_BOOL)) ;
	GrB_OK (GxB_Scalar_setElement_BOOL (s, true)) ;
	GrB_OK (GxB_Vector_build_Scalar (v, (GrB_Index*) node_ids, s,
				(GrB_Index) node_count)) ;

	GrB_OK (GrB_free (&s)) ;
}

void StagedUpdatesCtx_LabelNodes
(
	StagedUpdatesCtx *ctx,  // staged update context
	NodeID *node_ids,       // node IDs
	uint32_t node_count,    // number of nodes
	const char *label       // label to add
) {
	ASSERT (ctx        != NULL) ;
	ASSERT (label      != NULL) ;
	ASSERT (node_id    != INVALID_ENTITY_ID) ;
	ASSERT (node_count > 0) ;

	_PopulateVector (ctx, true, node_ids, node_count, label) ;
}

void StagedUpdatesCtx_UnLabelNodes
(
	StagedUpdatesCtx *ctx,  // staged update context
	NodeID *node_ids,       // node IDs
	uint32_t node_count,    // number of nodes
	const char *label       // label to remove
) {
	ASSERT (ctx        != NULL) ;
	ASSERT (label      != NULL) ;
	ASSERT (node_id    != INVALID_ENTITY_ID) ;
	ASSERT (node_count > 0) ;

	_PopulateVector (ctx, false, node_ids, node_count, label) ;
}

// free staged update context
void StagedUpdatesCtx_Free
(
	StagedUpdatesCtx **ctx  // staged updates context to free
) {
	ASSERT (ctx != NULL && *ctx != NULL) ;

	StagedUpdatesCtx *_ctx = *ctx ;

	HashTableEmpty (ctx->node_updates, NULL) ;
	HashTableEmpty (ctx->edge_updates, NULL) ;

	if (ctx->added_labels != NULL) {
		ASSERT (ctx->n_added_labels > 0) ;
		for (uint i = 0 ; i < ctx->n_added_labels ; i++) {
			GrB_Vector v = ctx->added_labels [i] ;
			GrB_free (&v) ;
		}
		rm_free (ctx->added_labels) ;
	}

	if (ctx->removed_labels != NULL) {
		ASSERT (ctx->n_removed_labels > 0) ;
		for (uint i = 0 ; i < ctx->n_removed_labels ; i++) {
			GrB_Vector v = ctx->removed_labels [i] ;
			GrB_free (&v) ;
		}
		rm_free (ctx->removed_labels) ;
	}

	rm_free (_ctx) ;
	*ctx = NULL ;
}

