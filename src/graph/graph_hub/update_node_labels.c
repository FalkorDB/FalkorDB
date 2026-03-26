/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../graph_hub.h"
#include "../graphcontext.h"
#include "../../query_ctx.h"
#include "../../effects/effects.h"
#include "../../undo_log/undo_log.h"

// below this threshold, per-element delta updates outperform bulk matrix ops
#define LABEL_BATCH_THRESHOLD 10

// adds a label to a large set of nodes (>=LABEL_BATCH_THRESHOLD) via
// bulk GraphBLAS matrix operations
//
// builds a diagonal matrix from V and merges it into the label-specific
// Delta_Matrix (DP), cancelling any pending removals (DM) in the process
// also updates the node-labels matrix directly via column/row subassign
// after a Delta_Matrix_wait to ensure no pending state conflicts
// updates indices and records undo/effects log entries if required
//
// prefer _LabelNodes_Single for smaller node sets
static void _LabelNodes_Bulk
(
	GraphContext  *gc,  // graph context
	GrB_Vector     V,   // boolean vector mapping node IDs to label assignment;
                        // must be named (GrB_NAME) with the target label name
	bool           log  // if true, record in undo-log and effects buffer
) {
	ASSERT (gc != NULL);

	UndoLog undo_log  = NULL ;
	EffectsBuffer *eb = NULL ;
	Graph *g = GraphContext_GetGraph (gc) ;

	if (log) {
		eb = QueryCtx_GetEffectsBuffer () ;
		undo_log = QueryCtx_GetUndoLog () ;
	}

	char lbl_name[512] = {0} ;

	//--------------------------------------------------------------------------
	// add labels
	//--------------------------------------------------------------------------

	GrB_OK (GrB_get (V, lbl_name, GrB_NAME)) ;

	GrB_Index v_size, v_nvals ;
	GrB_OK (GrB_Vector_size  (&v_size,  V)) ;
	GrB_OK (GrB_Vector_nvals (&v_nvals, V)) ;

	// create missing schema
	Schema *s = GraphContext_GetSchema (gc, lbl_name, SCHEMA_NODE) ;
	if (s == NULL) {
		s = GraphHub_AddSchema (gc, lbl_name, SCHEMA_NODE, log) ;
	}
	ASSERT (s != NULL) ;
	LabelID lbl_id = Schema_GetID (s) ;

	//--------------------------------------------------------------------------
	// index nodes
	//--------------------------------------------------------------------------

	if (log || Schema_HasIndices (s)) {
		// create an iterator
		GxB_Iterator it ;
		GxB_Iterator_new (&it) ;

		// attach it to the vector v
		GrB_Info info = GxB_Vector_Iterator_attach (it, V, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;

		// seek to the first entry
		info = GxB_Vector_Iterator_seek (it, 0) ;
		while (info != GxB_EXHAUSTED) {
			// get the entry v(i)
			Node node ;
			GrB_Index node_id = GxB_Vector_Iterator_getIndex (it) ;
			bool found = Graph_GetNode (g, node_id, &node) ;
			ASSERT (found == true) ;

			// index node
			if (Schema_HasIndices (s)) {
				Schema_AddNodeToIndex (s, &node) ;
			}

			if (log) {
				UndoLog_AddLabels (undo_log, &node, &lbl_id, 1) ;
				EffectsBuffer_AddLabelsEffect (eb, &node, &lbl_id, 1) ;
			}

			// move to the next entry in v
			info = GxB_Vector_Iterator_next (it) ;
		}
		GrB_free (&it) ;
	}

	//--------------------------------------------------------------------------
	// update the node labels matrix
	//--------------------------------------------------------------------------

	Delta_Matrix lbls = Graph_GetNodeLabelMatrix (g) ;
	GrB_OK (Delta_Matrix_wait (lbls, true)) ;

	GrB_Matrix lbls_M  = Delta_Matrix_M (lbls) ;
	GrB_Matrix lbls_TM = Delta_Matrix_M (Delta_Matrix_getTranspose (lbls)) ;

	// add entries to the lbl_id column of the labels matrix
	GrB_OK (GxB_Col_subassign (lbls_M, V, NULL, V, GrB_ALL, v_size, lbl_id,
				NULL)) ;

	// add entries to the lbl_id row of the transposed labels matrix
	GrB_OK (GxB_Row_subassign (lbls_TM, V, NULL, V, lbl_id, GrB_ALL, v_size,
				NULL)) ;

	//--------------------------------------------------------------------------
	// build a diagonal matrix from V
	//--------------------------------------------------------------------------

	GrB_Matrix diag ;
	GrB_OK (GrB_Matrix_diag (&diag, V, 0)) ;

	Delta_Matrix L    = Graph_GetLabelMatrix (g, lbl_id) ;
	GrB_Matrix L_DP   = Delta_Matrix_DP (L) ; 
	GrB_Matrix L_DM   = Delta_Matrix_DM (L) ;

	// clear both diag and L_DM's main diaginal at the intersection
	// diag ∩ L_DM

	//--------------------------------------------------------------------------
	// step 1: find entries present in both diag and L_DM
	//--------------------------------------------------------------------------

	GrB_Index L_nrows, L_ncols ;
	GrB_OK (Delta_Matrix_nrows (&L_nrows, L)) ;
	GrB_OK (Delta_Matrix_ncols (&L_ncols, L)) ;

	ASSERT (v_size == L_nrows)  ;
	ASSERT (L_nrows == L_ncols) ;

	GrB_Matrix isect ;
	GrB_OK (GrB_Matrix_new (&isect, GrB_BOOL, v_size, v_size)) ;
	GrB_OK (GrB_eWiseMult (isect, NULL, NULL, GrB_ONEB_BOOL, diag, L_DM, NULL)) ;

	//--------------------------------------------------------------------------
	// step 2: remove intersecting entries from both matrices
	//--------------------------------------------------------------------------

	// GrB_DESC_RSC = Replace + Structure + Complement:
	//   - complement: mask is "true" where isect has NO entry
	//   - replace: clears any output entry not covered by the mask
	// net effect: entries where isect IS set get dropped
	GrB_OK (GrB_Matrix_assign (L_DM, isect, NULL, L_DM, GrB_ALL, L_nrows,
				GrB_ALL, L_ncols, GrB_DESC_RSC)) ;

	GrB_OK (GrB_Matrix_assign (diag, isect, NULL, diag, GrB_ALL, L_nrows,
				GrB_ALL, L_ncols, GrB_DESC_RSC)) ;

	GrB_OK (GrB_free (&isect)) ;

	//--------------------------------------------------------------------------
	// assign diag into L_DP
	//--------------------------------------------------------------------------

	// - mask D (structural): only positions where D has entries are affected
	// - no replace: off-diagonal entries in A are preserved
	// - no accum: diagonal entries in A are overwritten with D's values
	GrB_OK (GrB_Matrix_assign (L_DP, diag, NULL, diag, GrB_ALL, v_size,
				GrB_ALL, v_size, GrB_DESC_S)) ;

	GrB_OK (GrB_free (&diag)) ;

	GraphStatistics_IncNodeCount (&g->stats, lbl_id, v_nvals) ;
}

// adds a label to a small set of nodes (<LABEL_BATCH_THRESHOLD) via
// per-element delta updates
//
// for each node in V:
//   - sets the corresponding entry in the label-specific Delta_Matrix L
//   - sets the corresponding entry in the node-labels Delta_Matrix
//   - updates indices if the schema has any
//   - records the operation in the undo-log and effects buffer if log is set
//
// prefer _LabelNodes_Bulk for larger node sets
static void _LabelNodes_Single
(
	GraphContext  *gc,  // graph context
	GrB_Vector     V,   // boolean vector mapping node IDs to label assignment;
                        // must be named (GrB_NAME) with the target label name
	bool           log  // if true, record in undo-log and effects buffer
) {
	ASSERT (gc != NULL);

	UndoLog undo_log  = NULL ;
	EffectsBuffer *eb = NULL ;
	Graph *g = GraphContext_GetGraph (gc) ;

	if (log) {
		eb = QueryCtx_GetEffectsBuffer () ;
		undo_log = QueryCtx_GetUndoLog () ;
	}

	char lbl_name[512] = {0} ;

	//--------------------------------------------------------------------------
	// add labels
	//--------------------------------------------------------------------------

	GrB_OK (GrB_get (V, lbl_name, GrB_NAME)) ;

	GrB_Index v_size, v_nvals ;
	GrB_OK (GrB_Vector_size  (&v_size,  V)) ;
	GrB_OK (GrB_Vector_nvals (&v_nvals, V)) ;

	// create missing schema
	Schema *s = GraphContext_GetSchema (gc, lbl_name, SCHEMA_NODE) ;
	if (s == NULL) {
		s = GraphHub_AddSchema (gc, lbl_name, SCHEMA_NODE, log) ;
	}
	ASSERT (s != NULL) ;
	LabelID lbl_id = Schema_GetID (s) ;

	Delta_Matrix L    = Graph_GetLabelMatrix (g, lbl_id) ;
	Delta_Matrix lbls = Graph_GetNodeLabelMatrix (g) ;

	// create an iterator
	GxB_Iterator it ;
	GxB_Iterator_new (&it) ;

	// attach it to the vector v
	GrB_Info info = GxB_Vector_Iterator_attach (it, V, NULL) ;
	ASSERT (info == GrB_SUCCESS) ;

	// seek to the first entry
	info = GxB_Vector_Iterator_seek (it, 0) ;
	while (info != GxB_EXHAUSTED) {
		// get the entry v(i)
		Node node ;
		GrB_Index node_id = GxB_Vector_Iterator_getIndex (it) ;

		GrB_OK (Delta_Matrix_setElement_BOOL (L,    node_id, node_id)) ;
		GrB_OK (Delta_Matrix_setElement_BOOL (lbls, node_id, lbl_id))  ;

		bool found = Graph_GetNode (g, node_id, &node) ;
		ASSERT (found == true) ;

		// index node
		if (Schema_HasIndices (s)) {
			Schema_AddNodeToIndex (s, &node) ;
		}

		if (log) {
			UndoLog_AddLabels (undo_log, &node, &lbl_id, 1) ;
			EffectsBuffer_AddLabelsEffect (eb, &node, &lbl_id, 1) ;
		}

		// move to the next entry in v
		info = GxB_Vector_Iterator_next (it) ;
	}

	GrB_free (&it) ;
	GraphStatistics_IncNodeCount (&g->stats, lbl_id, v_nvals) ;
}

// removes a label from a large set of nodes (>=LABEL_BATCH_THRESHOLD) via
// bulk GraphBLAS matrix operations
//
// builds a diagonal matrix from V and merges it into the label-specific
// delta_Matrix (DM), cancelling any pending additions (DP) in the process
// updates indices and records undo/effects log entries if required
//
// if the label schema does not exist, returns silently — nothing to remove
// prefer _UnLabelNodes_Single for smaller node sets
static void _UnLabelNodes_Bulk
(
	GraphContext  *gc,  // graph context
	GrB_Vector     V,   // boolean vector mapping node IDs to label removal;
                        // must be named (GrB_NAME) with the target label name
	bool           log  // if true, record in undo-log and effects buffer
) {
	ASSERT (gc != NULL);

	UndoLog undo_log  = NULL ;
	EffectsBuffer *eb = NULL ;
	Graph *g = GraphContext_GetGraph (gc) ;

	if (log) {
		eb = QueryCtx_GetEffectsBuffer () ;
		undo_log = QueryCtx_GetUndoLog () ;
	}

	char lbl_name[512] = {0} ;

	//--------------------------------------------------------------------------
	// remove labels
	//--------------------------------------------------------------------------

	GrB_OK (GrB_get (V, lbl_name, GrB_NAME)) ;

	GrB_Index v_size, v_nvals ;
	GrB_OK (GrB_Vector_size  (&v_size,  V)) ;
	GrB_OK (GrB_Vector_nvals (&v_nvals, V)) ;

	Schema *s = GraphContext_GetSchema (gc, lbl_name, SCHEMA_NODE) ;
	if (s == NULL) {
		return ;
	}

	LabelID lbl_id = Schema_GetID (s) ;

	//--------------------------------------------------------------------------
	// remove nodes from index
	//--------------------------------------------------------------------------

	if (log || Schema_HasIndices (s)) {
		// create an iterator
		GxB_Iterator it ;
		GxB_Iterator_new (&it) ;

		// attach it to the vector v
		GrB_Info info = GxB_Vector_Iterator_attach (it, V, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;

		// seek to the first entry
		info = GxB_Vector_Iterator_seek (it, 0) ;
		while (info != GxB_EXHAUSTED) {
			// get the entry v(i)
			Node node ;
			GrB_Index node_id = GxB_Vector_Iterator_getIndex (it) ;
			bool found = Graph_GetNode (g, node_id, &node) ;
			ASSERT (found == true) ;

			// index node
			if (Schema_HasIndices (s)) {
				Schema_RemoveNodeFromIndex (s, &node) ;
			}

			if (log) {
				UndoLog_RemoveLabels (undo_log, &node, &lbl_id, 1) ;
				EffectsBuffer_AddRemoveLabelsEffect (eb, &node, &lbl_id, 1) ;
			}

			// move to the next entry in v
			info = GxB_Vector_Iterator_next (it) ;
		}
		GrB_free (&it) ;
	}

	//--------------------------------------------------------------------------
	// build a diagonal matrix from V
	//--------------------------------------------------------------------------

	GrB_Matrix diag ;
	GrB_OK (GrB_Matrix_diag (&diag, V, 0)) ;

	Delta_Matrix L  = Graph_GetLabelMatrix (g, lbl_id) ;
	GrB_Matrix L_DP = Delta_Matrix_DP (L) ; 
	GrB_Matrix L_DM = Delta_Matrix_DM (L) ;

	// clear both diag and L_DP's main diaginal at the intersection
	// diag ∩ L_DP

	//--------------------------------------------------------------------------
	// step 1: find entries present in both diag and L_DP
	//--------------------------------------------------------------------------

	GrB_Index L_nrows, L_ncols ;
	GrB_OK (Delta_Matrix_nrows (&L_nrows, L)) ;
	GrB_OK (Delta_Matrix_ncols (&L_ncols, L)) ;

	ASSERT (v_size == L_nrows)  ;
	ASSERT (L_nrows == L_ncols) ;

	GrB_Matrix isect ;
	GrB_OK (GrB_Matrix_new (&isect, GrB_BOOL, v_size, v_size)) ;
	GrB_OK (GrB_eWiseMult (isect, NULL, NULL, GrB_ONEB_BOOL, diag, L_DP, NULL)) ;

	//--------------------------------------------------------------------------
	// step 2: remove intersecting entries from both matrices
	//--------------------------------------------------------------------------

	// GrB_DESC_RSC = Replace + Structure + Complement:
	//   - complement: mask is "true" where isect has NO entry
	//   - replace: clears any output entry not covered by the mask
	// net effect: entries where isect IS set get dropped
	GrB_OK (GrB_Matrix_assign (L_DP, isect, NULL, L_DP, GrB_ALL, L_nrows,
				GrB_ALL, L_ncols, GrB_DESC_RSC)) ;

	GrB_OK (GrB_Matrix_assign (diag, isect, NULL, diag, GrB_ALL, L_nrows,
				GrB_ALL, L_ncols, GrB_DESC_RSC)) ;

	GrB_OK (GrB_free (&isect)) ;

	//--------------------------------------------------------------------------
	// assign diag into L_DM
	//--------------------------------------------------------------------------

	// - mask D (structural): only positions where D has entries are affected
	// - no replace: off-diagonal entries in A are preserved
	// - no accum: diagonal entries in A are overwritten with D's values
	GrB_OK (GrB_Matrix_assign (L_DM, diag, NULL, diag, GrB_ALL, v_size,
				GrB_ALL, v_size, GrB_DESC_S)) ;

	GrB_OK (GrB_free (&diag)) ;

	GraphStatistics_DecNodeCount (&g->stats, lbl_id, v_nvals) ;
}

// removes a label from a small set of nodes (<LABEL_BATCH_THRESHOLD) via
// per-element delta updates
//
// for each node in V:
//   - removes the corresponding entry from the label-specific Delta_Matrix L
//   - removes the corresponding entry from the node-labels Delta_Matrix
//   - updates indices if the schema has any
//   - records the operation in the undo-log and effects buffer if log is set
//
// if the label schema does not exist, returns silently — nothing to remove
// prefer _UnLabelNodes_Bulk for larger node sets
static void _UnLabelNodes_Single
(
	GraphContext  *gc,  // graph context
	GrB_Vector     V,   // boolean vector mapping node IDs to label removal;
                        // must be named (GrB_NAME) with the target label name
	bool           log  // if true, record in undo-log and effects buffer
) {
	ASSERT (gc != NULL);

	UndoLog undo_log  = NULL ;
	EffectsBuffer *eb = NULL ;
	Graph *g = GraphContext_GetGraph (gc) ;

	if (log) {
		eb = QueryCtx_GetEffectsBuffer () ;
		undo_log = QueryCtx_GetUndoLog () ;
	}

	char lbl_name[512] = {0} ;
	Delta_Matrix lbls = Graph_GetNodeLabelMatrix (g) ;

	//--------------------------------------------------------------------------
	// remove labels
	//--------------------------------------------------------------------------

	GrB_OK (GrB_get (V, lbl_name, GrB_NAME)) ;

	GrB_Index v_size, v_nvals ;
	GrB_OK (GrB_Vector_size  (&v_size,  V)) ;
	GrB_OK (GrB_Vector_nvals (&v_nvals, V)) ;

	Schema *s = GraphContext_GetSchema (gc, lbl_name, SCHEMA_NODE) ;
	if (s == NULL) {
		return ;
	}

	LabelID lbl_id = Schema_GetID (s) ;
	Delta_Matrix L = Graph_GetLabelMatrix (g, lbl_id) ;

	//--------------------------------------------------------------------------
	// remove nodes from index
	//--------------------------------------------------------------------------

	// create an iterator
	GxB_Iterator it ;
	GxB_Iterator_new (&it) ;

	// attach it to the vector v
	GrB_Info info = GxB_Vector_Iterator_attach (it, V, NULL) ;
	ASSERT (info == GrB_SUCCESS) ;

	// seek to the first entry
	info = GxB_Vector_Iterator_seek (it, 0) ;
	while (info != GxB_EXHAUSTED) {
		// get the entry v(i)
		Node node ;
		GrB_Index node_id = GxB_Vector_Iterator_getIndex (it) ;
		bool found = Graph_GetNode (g, node_id, &node) ;
		ASSERT (found == true) ;

		// index node
		if (Schema_HasIndices (s)) {
			Schema_RemoveNodeFromIndex (s, &node) ;
		}

		if (log) {
			UndoLog_RemoveLabels (undo_log, &node, &lbl_id, 1) ;
			EffectsBuffer_AddRemoveLabelsEffect (eb, &node, &lbl_id, 1) ;
		}

		GrB_OK (Delta_Matrix_removeElement (L,    node_id, node_id)) ;
		GrB_OK (Delta_Matrix_removeElement (lbls, node_id, lbl_id))  ;

		// move to the next entry in v
		info = GxB_Vector_Iterator_next (it) ;
	}

	GrB_free (&it) ;
	GraphStatistics_DecNodeCount (&g->stats, lbl_id, v_nvals) ;
}

// applies pending label additions and removals to the graph, dispatching to
// single-element or bulk GraphBLAS operations based on the number of affected
// nodes per label
//
// for each label vector in add_labels:
//   - if nvals < LABEL_BATCH_THRESHOLD: _LabelNodes_Single
//   - otherwise:                        _LabelNodes_Bulk
//
// for each label vector in rmv_labels:
//   - if nvals < LABEL_BATCH_THRESHOLD: _UnLabelNodes_Single
//   - otherwise:                        _UnLabelNodes_Bulk
//
// both add and remove passes are applied when both arrays are provided
// each vector in add_labels and rmv_labels must be named (GrB_NAME) with
// the target label name — unnamed vectors are a programming error
//
// add_labels and rmv_labels are assumed to be disjoint; a label appearing
// in both arrays in the same call produces undefined behavior
void _GraphHub_UpdateNodeLabels
(
    GraphContext  *gc,           // graph context
    GrB_Vector    *add_labels,   // per-label vectors for nodes to label;
                                 // NULL if no labels to add
    uint           n_add_labels, // number of vectors in add_labels (0 if NULL)
    GrB_Vector    *rmv_labels,   // per-label vectors for nodes to unlabel;
                                 // NULL if no labels to remove
    uint           n_rmv_labels, // number of vectors in rmv_labels (0 if NULL)
    bool           log           // if true, record in undo-log and effects buffer
) {
	ASSERT (gc != NULL);

	// if add_labels is specified its count must be > 0
	ASSERT (add_labels != NULL && n_add_labels > 0 ||
			add_labels == NULL && n_add_labels == 0) ;

	// if rmv_labels is specified its count must be > 0
	ASSERT (rmv_labels != NULL && n_rmv_labels > 0 ||
			rmv_labels == NULL && n_rmv_labels == 0) ;

	GrB_Vector V ;
	GrB_Index nvals ;

	for (uint i = 0 ; i < n_add_labels; i++) {
		V = add_labels [i] ;
		GrB_OK (GrB_Vector_nvals (&nvals, V)) ;

		if (nvals < LABEL_BATCH_THRESHOLD) {
			_LabelNodes_Single (gc, V, log) ;
		} else {
			_LabelNodes_Bulk (gc, V, log) ;
		}
	}

	for (uint i = 0 ; i < n_rmv_labels; i++) {
		V = rmv_labels [i] ;
		GrB_OK (GrB_Vector_nvals (&nvals, V)) ;

		if (nvals < LABEL_BATCH_THRESHOLD) {
			_UnLabelNodes_Single (gc, V, log) ;
		} else {
			_UnLabelNodes_Bulk (gc, V, log) ;
		}
	}
}

