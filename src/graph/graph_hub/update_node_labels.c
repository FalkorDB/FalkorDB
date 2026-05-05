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

// adds a label to a small set of nodes (<LABEL_BATCH_THRESHOLD) via
// per-element delta updates
//
// for each node in V:
//   - sets the corresponding entry in the label-specific Delta_Matrix L
//   - sets the corresponding entry in the node-labels Delta_Matrix
//   - updates indices if the schema has any
//   - records the operation in the undo-log and effects buffer if log is set
static void _LabelNodes_Single
(
	GraphContext  *gc,  // graph context
	GrB_Vector     *V,  // boolean vector mapping node IDs to label assignment;
                        // must be named (GrB_NAME) with the target label name
	bool           log  // if true, record in undo-log and effects buffer
) {
	ASSERT (gc != NULL);
	ASSERT (V  != NULL && *V != NULL);

	GrB_Vector _V     = *V ;
	UndoLog undo_log  = NULL ;
	EffectsBuffer *eb = NULL ;
	Graph *g = GraphContext_GetGraph (gc) ;

	if (log) {
		eb = QueryCtx_GetEffectsBuffer () ;
		undo_log = QueryCtx_GetUndoLog () ;
	}

	char lbl_name[512] = {0} ;

	//--------------------------------------------------------------------------
	// set label
	//--------------------------------------------------------------------------

	GrB_OK (GrB_get (_V, lbl_name, GrB_NAME)) ;
	Schema *s = GraphContext_GetSchema (gc, lbl_name, SCHEMA_NODE) ;
	ASSERT (s != NULL) ;

	bool index = Schema_HasIndices (s) ;

	// enforce constraints only when:
	// 1. schema has constraints
	// 2. we're required to log i.e. change is due to a GRAPH.QUERY execution
	//    on a master node
	bool enforce_constraints = Schema_HasConstraints (s) && log ;

	LabelID lbl_id = Schema_GetID (s) ;

	// create an iterator
	GxB_Iterator it ;
	GxB_Iterator_new (&it) ;

	// attach it to vector v
	GrB_Info info = GxB_Vector_Iterator_attach (it, _V, NULL) ;
	ASSERT (info == GrB_SUCCESS) ;

	// seek to the first entry
	info = GxB_Vector_Iterator_seek (it, 0) ;
	while (info != GxB_EXHAUSTED) {
		// get the entry v(i)
		Node node ;
		GrB_Index node_id = GxB_Vector_Iterator_getIndex (it) ;
		Graph_LabelNode (g, node_id, &lbl_id, 1) ;

		if (index) {
			bool found = Graph_GetNode (g, node_id, &node) ;
			ASSERT (found == true) ;
			Schema_AddNodeToIndex (s, &node) ;
		}

		if (enforce_constraints) {
			// enforce constraint
			char *err_msg = NULL ;
			if (!Schema_EnforceConstraints (s, (GraphEntity*)(&node), &err_msg)) {
				// constraint violation
				ASSERT (err_msg != NULL) ;
				ErrorCtx_SetError ("%s", err_msg) ;
				free (err_msg) ;
				enforce_constraints = false ; // stop enforcing
				// do not break, let the while loop above finish its work
				// as the undo operation removes all labels associated vector V
				// although it is probably OK to break, we prefer not to
			}
		}

		// move to the next entry in v
		info = GxB_Vector_Iterator_next (it) ;
	}

	if (log) {
		EffectsBuffer_AddLabelsEffect (eb, _V) ;
		UndoLog_AddLabels (undo_log, V) ;
	}

	GrB_free (&it) ;
}

// removes a label from a small set of nodes (<LABEL_BATCH_THRESHOLD) via
// per-element delta updates
//
// for each node in V:
//   - removes the corresponding entry from the label-specific Delta_Matrix L
//   - removes the corresponding entry from the node-labels Delta_Matrix
//   - updates indices if the schema has any
//   - records the operation in the undo-log and effects buffer if log is set
static void _UnLabelNodes_Single
(
	GraphContext  *gc,  // graph context
	GrB_Vector     *V,  // boolean vector mapping node IDs to label removal;
                        // must be named (GrB_NAME) with the target label name
	bool           log  // if true, record in undo-log and effects buffer
) {
	ASSERT (gc != NULL);
	ASSERT (V  != NULL && *V != NULL);

	GrB_Vector _V     = *V ;
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

	GrB_OK (GrB_get (_V, lbl_name, GrB_NAME)) ;
	Schema *s = GraphContext_GetSchema (gc, lbl_name, SCHEMA_NODE) ;
	ASSERT (s != NULL) ;
	LabelID lbl_id = Schema_GetID (s) ;

	bool index = Schema_HasIndices (s) ;

	//--------------------------------------------------------------------------
	// remove nodes from index
	//--------------------------------------------------------------------------

	// create an iterator
	GxB_Iterator it ;
	GxB_Iterator_new (&it) ;

	// attach it to the vector v
	GrB_Info info = GxB_Vector_Iterator_attach (it, _V, NULL) ;
	ASSERT (info == GrB_SUCCESS) ;

	// seek to the first entry
	info = GxB_Vector_Iterator_seek (it, 0) ;
	while (info != GxB_EXHAUSTED) {
		// get the entry v(i)
		Node node ;
		GrB_Index node_id = GxB_Vector_Iterator_getIndex (it) ;

		if (index) {
			bool found = Graph_GetNode (g, node_id, &node) ;
			ASSERT (found == true) ;
			Schema_RemoveNodeFromIndex (s, &node) ;
		}

		Graph_RemoveNodeLabels (g, node_id, &lbl_id, 1) ;

		// move to the next entry in v
		info = GxB_Vector_Iterator_next (it) ;
	}

	if (log) {
		EffectsBuffer_AddRemoveLabelsEffect (eb, _V) ;
		UndoLog_RemoveLabels (undo_log, V) ;
	}

	GrB_free (&it) ;
}

// applies pending label additions and removals to the graph
//
// both add and remove passes are applied when both arrays are provided
// each vector in add_labels and rmv_labels must be named (GrB_NAME) with
// the target label name — unnamed vectors are a programming error
//
// add_labels and rmv_labels are assumed to be disjoint; a label appearing
// in both arrays in the same call produces undefined behavior
void GraphHub_UpdateNodeLabels
(
    GraphContext  *gc,            // graph context
    GrB_Vector    *add_labels,    // per-label vectors for nodes to label;
                                  // NULL if no labels to add
    uint           n_add_labels,  // number of vectors in add_labels (0 if NULL)
    GrB_Vector    *rmv_labels,    // per-label vectors for nodes to unlabel;
                                  // NULL if no labels to remove
    uint           n_rmv_labels,  // number of vectors in rmv_labels (0 if NULL)
    bool           log            // if true, record in undo-log and effects buffer
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

	// associate label(s) with nodes
	for (uint i = 0 ; i < n_add_labels; i++) {
		_LabelNodes_Single (gc, add_labels + i, log) ;
	}

	// remove label(s) from nodes
	for (uint i = 0 ; i < n_rmv_labels; i++) {
		_UnLabelNodes_Single (gc, rmv_labels + i, log) ;
	}
}

