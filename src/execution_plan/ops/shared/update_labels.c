/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../../graph/graphcontext.h"

#define MAX_LABEL_NAME_LEN 512

// applies label additions / removals to an entity within a set of records
// corresponds to Cypher update semantics, e.g.:
// MATCH (n:A) SET n:B REMOVE n:C
void StageLabelAssignments
(
	GraphContext *gc,       // graph context
	const Record *records,  // array of records containing the target entity
	uint n_recs,            // number of records
	uint record_idx,        // index of the target entity within its record
	const char **lbls,      // labels to add to the entity
	uint n_lbls,            // number of labels to add
	GrB_Vector **vecs,      // [input/output] tracks label to node assignment
	uint *n_vecs,           // [input/output] number of vectors
	bool add                // true for label addition, false for removal
) {
	ASSERT (gc      != NULL) ;
	ASSERT (lbls    != NULL) ;
	ASSERT (vecs    != NULL) ;
	ASSERT (n_vecs  != NULL) ;
	ASSERT (records != NULL) ;
	ASSERT (n_lbls  > 0) ;

	Graph *g = GraphContext_GetGraph (gc) ;
	GrB_Index dim = Graph_RequiredMatrixDim (g) ;

	//--------------------------------------------------------------------------
	// pass 1: resolve or create a vector for each label
	// cache resolved vectors in label_vecs[] to avoid re-scanning in pass 2
	//--------------------------------------------------------------------------

	char name [MAX_LABEL_NAME_LEN] = {0} ;  // name of vector
	LabelID label_ids     [n_lbls] ;
	GrB_Vector label_vecs [n_lbls] ;
	GrB_Vector *_vecs = *vecs ;

	for (uint i = 0 ; i < n_lbls ; i++) {
		GrB_Vector v  = NULL  ;
		bool resolved = false ;

		// determine label ID
		Schema *s = GraphContext_GetSchema (gc, lbls[i], SCHEMA_NODE) ;
		label_ids [i] = (s != NULL) ? Schema_GetID (s) : GRAPH_UNKNOWN_LABEL ;

		for (uint j = 0 ; j < *n_vecs ; j++) {
			v = _vecs [j] ;
			GrB_OK (GrB_get (v, name, GrB_NAME)) ;
			if (strcmp (lbls[i], name) == 0) {
				rm_free (name) ;
				resolved = true ;
				break ;
			}
		}

		if (!resolved) {
			// no vector yet for this label — create and register one
			GrB_OK (GrB_Vector_new (&v, GrB_BOOL, dim)) ;
			GrB_OK (GrB_set (v, (char*)lbls[i], GrB_NAME)) ;

			_vecs = rm_realloc (_vecs, (*n_vecs + 1) * sizeof (GrB_Vector)) ;
			_vecs[*n_vecs] = v ;
			(*n_vecs)++ ;
		}

		ASSERT (v != NULL) ;
		label_vecs [i] = v ;
	}

	// set output
	*vecs = _vecs ;

	//--------------------------------------------------------------------------
	// pass 2: for each record, validate the node once, then mark it in
	// every label vector — records-outer so validation runs once per entity
	//--------------------------------------------------------------------------

	for (uint i = 0 ; i < n_recs; i++) {
		Record r = records [i] ;

		//----------------------------------------------------------------------
		// validate entity type
		//----------------------------------------------------------------------

		// entity slot not populated — skip silently
		RecordEntryType t = Record_GetType (r, record_idx) ;
		if (unlikely (t == REC_TYPE_UNKNOWN)) {
			continue ;
		}

		// a populated, non-node slot is a query error
		if (unlikely (t != REC_TYPE_NODE)) {
			ErrorCtx_RaiseRuntimeException (
				"Label update error: entity did not resolve to a node") ;
			return ;
		}

		// get the updated entity
		Node *node = Record_GetNode (r, record_idx) ;

		// deleted nodes are skipped silently
		if (unlikely (Graph_EntityIsDeleted ((const GraphEntity *) node))) {
			continue ;
		}

		EntityID id = ENTITY_GET_ID (node) ;

		// foreach label
		for (uint j = 0 ; j < n_lbls ; j++) {
			GrB_Vector v = label_vecs [j] ;

			bool node_has_label = (label_ids[j] != GRAPH_UNKNOWN_LABEL) &&
				Graph_IsNodeLabeled (g, id, label_ids [j]) ;

			// mark if the current state differs from the desired state:
			// adding a label the node lacks, or removing one it has
			if (add != node_has_label) {
				GrB_OK (GrB_Vector_setElement_BOOL (v, true, id)) ;
			}
		}
	}
}

