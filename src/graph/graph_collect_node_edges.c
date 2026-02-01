/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graph.h"

void Graph_CollectInOutEdges
(
	Edge **outgoing,     // [output] outgoing edges
	Edge **incoming,     // [output] incoming edges
	Graph *g,            // graph
	Node *nodes,         // nodes to collect edges for
	uint64_t node_count  // number of nodes
) {
	ASSERT (g        != NULL) ;
	ASSERT (nodes    != NULL) ;
	ASSERT (outgoing != NULL) ;
	ASSERT (incoming != NULL) ;

	if (node_count == 0) {
		return ;
	}

	GrB_Index info ;

	GrB_Matrix IN;   // M of D_IN
	GrB_Matrix OUT;  // M of D_OUT

	Delta_Matrix DS ;     // selection matrix
	Delta_Matrix D_IN ;   // S * RT
	Delta_Matrix D_OUT ;  // S * R

	GrB_Index m = Graph_RequiredMatrixDim (g) ;

	// create IN & OUT delta matrices
	GrB_OK (Delta_Matrix_new (&D_IN,  GrB_UINT64, m, m, false)) ;
	GrB_OK (Delta_Matrix_new (&D_OUT, GrB_UINT64, m, m, false)) ;

	//--------------------------------------------------------------------------
	// build selection matrix
	//--------------------------------------------------------------------------

	info = Delta_Matrix_new (&DS, GrB_BOOL, m, m, false) ;
	ASSERT (info == GrB_SUCCESS) ;

	GrB_Matrix S = Delta_Matrix_M (DS) ;
	ASSERT (S != NULL) ;

	for (uint64_t i = 0 ; i < node_count ; i++) {
		GrB_Index id = ENTITY_GET_ID (nodes + i) ;
		info = GrB_Matrix_setElement_BOOL (S, true, id, id) ;
		ASSERT (info == GrB_SUCCESS) ;
	}

	//--------------------------------------------------------------------------
	// initialize iterators
	//--------------------------------------------------------------------------

	GxB_Iterator vec_it ;
	GxB_Iterator iterator ;
	GxB_Iterator_new (&vec_it) ;
	GxB_Iterator_new (&iterator) ;

	// scan through relation matrices
	int relationCount = Graph_RelationTypeCount (g) ;
	for (int r = 0; r < relationCount; r++) {
		Tensor R  = Graph_GetRelationMatrix (g, r, false) ;
		Tensor RT = Graph_GetRelationMatrix (g, r, true) ;

		//----------------------------------------------------------------------
		// extract outgoing edges
		//----------------------------------------------------------------------

		// D_OUT = S * R
		info = Delta_mxm (D_OUT, GxB_ANY_SECOND_UINT64, DS, R) ;
		ASSERT (info == GrB_SUCCESS) ;
		OUT = DELTA_MATRIX_M (D_OUT) ;

		// collect edges
		info = GxB_Matrix_Iterator_attach (iterator, OUT, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;

		info = GxB_Matrix_Iterator_seek (iterator, 0) ;
		while (info != GxB_EXHAUSTED) {
			// get the entry OUT(i,j)
			GrB_Index src, dest;
			GxB_Matrix_Iterator_getIndex (iterator, &src, &dest) ;
			uint64_t id = GxB_Iterator_get_UINT64 (iterator) ;

			if (SCALAR_ENTRY (id)) {
				Edge e = {.id         = id,
						  .src_id     = src,
						  .dest_id    = dest,
						  .relationID = r,
						  .attributes = DataBlock_GetItem (g->edges, id)
				} ;

				ASSERT (e.attributes) ;
				array_append (*outgoing, e) ;
			}

			else {
				GrB_Vector V = AS_VECTOR (id) ;
				info = GxB_Vector_Iterator_attach (vec_it, V, NULL) ;

				// seek to the first entry
				info = GxB_Vector_Iterator_seek (vec_it, 0) ;
				while (info != GxB_EXHAUSTED) {
					// get the entry v(i)
					id = GxB_Vector_Iterator_getIndex (vec_it) ;

					Edge e = {.id         = id,
							  .src_id     = src,
							  .dest_id    = dest,
							  .relationID = r,
							  .attributes = DataBlock_GetItem (g->edges, id)
					} ;

					ASSERT (e.attributes) ;
					array_append (*outgoing, e) ;

					// move to the next entry in v
					info = GxB_Vector_Iterator_next (vec_it) ;
				}
			}

			// move to the next entry in OUT
			info = GxB_Matrix_Iterator_next (iterator) ;
		}

		//----------------------------------------------------------------------
		// extract incoming edges
		//----------------------------------------------------------------------

		// D_IN = S * RT
		info = Delta_mxm (D_IN, GxB_ANY_SECOND_BOOL, DS, RT) ;
		ASSERT (info == GrB_SUCCESS) ;
		IN = DELTA_MATRIX_M (D_IN) ;

		// clear outgoing from incoming
		GrB_OK (GrB_transpose (OUT, NULL, NULL, OUT, NULL)) ;
		GrB_OK (GrB_transpose (IN, OUT, NULL, IN, GrB_DESC_RSCT0)) ;

		info = GxB_Matrix_Iterator_attach (iterator, IN, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;

		info = GxB_Matrix_Iterator_seek (iterator, 0) ;
		while (info != GxB_EXHAUSTED) {
			// get the entry x(i,j)
			GrB_Index src, dest;
			GxB_Matrix_Iterator_getIndex (iterator, &src, &dest) ;

			uint64_t id ;
			info = Delta_Matrix_extractElement_UINT64 (&id, R, dest, src) ;
			ASSERT (info == GrB_SUCCESS) ;

			if (SCALAR_ENTRY (id)) {
				Edge e = {.id         = id,
						  .src_id     = dest,
						  .dest_id    = src,
						  .relationID = r,
						  .attributes = DataBlock_GetItem (g->edges, id)
				} ;

				ASSERT (e.attributes) ;
				array_append (*incoming, e) ;
			}
			
			else {
				GrB_Vector V = AS_VECTOR (id) ;
				info = GxB_Vector_Iterator_attach (vec_it, V, NULL) ;

				// seek to the first entry
				info = GxB_Vector_Iterator_seek (vec_it, 0) ;
				while (info != GxB_EXHAUSTED) {
					// get the entry v(i)
					id = GxB_Vector_Iterator_getIndex (vec_it) ;

					Edge e = {.id         = id,
							  .src_id     = dest,
							  .dest_id    = src,
							  .relationID = r,
							  .attributes = DataBlock_GetItem (g->edges, id)
					} ;

					ASSERT (e.attributes) ;
					array_append (*incoming, e) ;

					// move to the next entry in v
					info = GxB_Vector_Iterator_next (vec_it) ;
				}
			}

			// move to the next entry in X
			info = GxB_Matrix_Iterator_next (iterator) ;
		}

		// TODO: do we really need to clear ?
		GrB_OK (GrB_Matrix_clear (IN)) ;
		GrB_OK (GrB_Matrix_clear (OUT)) ;
	}

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	GrB_free          (&vec_it) ;
	GrB_free          (&iterator) ;
	Delta_Matrix_free (&DS) ;
	Delta_Matrix_free (&D_IN) ;
	Delta_Matrix_free (&D_OUT) ;
}

void Graph_CollectOutgoingEdges
(
	Edge **edges,
	Graph *g,
	Node *nodes,
	uint64_t node_count
) {
	ASSERT (g     != NULL) ;
	ASSERT (nodes != NULL) ;
	ASSERT (edges != NULL) ;

	if (node_count == 0) {
		return ;
	}

	GrB_Index info ;
	GrB_Matrix S ;           // selection matrix bool diagonal
	Delta_Matrix DS ;        // delta version of S
	Delta_Matrix outgoing ;

	GrB_Index m = Graph_RequiredMatrixDim (g) ;

	info = Delta_Matrix_new (&DS, GrB_BOOL, m, m, false) ;
	ASSERT (info == GrB_SUCCESS) ;

	info = Delta_Matrix_new (&outgoing, GrB_UINT64, m, m, false) ;
	ASSERT (info == GrB_SUCCESS) ;

	S = Delta_Matrix_M (DS) ;
	ASSERT (S != NULL) ;

	//--------------------------------------------------------------------------
	// build selection matrix
	//--------------------------------------------------------------------------

	for (uint64_t i = 0 ; i < node_count ; i++) {
		GrB_Index id = ENTITY_GET_ID (nodes + i) ;
		info = GrB_Matrix_setElement_BOOL (S, true, id, id) ;
		ASSERT (info == GrB_SUCCESS) ;
	}

	GxB_Iterator vec_it ;
	GxB_Iterator iterator ;
	GxB_Iterator_new (&vec_it) ;
	GxB_Iterator_new (&iterator) ;

	int relationCount = Graph_RelationTypeCount (g) ;
	for (int r = 0; r < relationCount; r++) {
		Tensor R = Graph_GetRelationMatrix (g, r, false) ;
		info = Delta_mxm (outgoing, GxB_ANY_SECOND_UINT64, DS, R) ;
		ASSERT (info == GrB_SUCCESS) ;
		GrB_Matrix x = DELTA_MATRIX_M (outgoing) ;

		info = GxB_Matrix_Iterator_attach (iterator, x, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;

		info = GxB_Matrix_Iterator_seek (iterator, 0) ;
		while (info != GxB_EXHAUSTED) {
			// get the entry x(i,j)
			GrB_Index src, dest;
			GxB_Matrix_Iterator_getIndex (iterator, &src, &dest) ;
			uint64_t id = GxB_Iterator_get_UINT64 (iterator) ;

			if (SCALAR_ENTRY (id)) {
				Edge e = {.id         = id,
						  .src_id     = src,
						  .dest_id    = dest,
						  .relationID = r,
						  .attributes = DataBlock_GetItem (g->edges, id)
				} ;

				ASSERT (e.attributes) ;
				array_append (*edges, e) ;
			}

			else {
				GrB_Vector V = AS_VECTOR (id) ;
				info = GxB_Vector_Iterator_attach (vec_it, V, NULL) ;

				// seek to the first entry
				info = GxB_Vector_Iterator_seek (vec_it, 0) ;
				while (info != GxB_EXHAUSTED) {
					// get the entry v(i)
					id = GxB_Vector_Iterator_getIndex (vec_it) ;

					Edge e = {.id         = id,
							  .src_id     = src,
							  .dest_id    = dest,
							  .relationID = r,
							  .attributes = DataBlock_GetItem (g->edges, id)
					} ;

					ASSERT (e.attributes) ;
					array_append (*edges, e) ;

					// move to the next entry in v
					info = GxB_Vector_Iterator_next (vec_it) ;
				}
			}

			// move to the next entry in X
			info = GxB_Matrix_Iterator_next (iterator) ;
		}

		info = GrB_Matrix_clear (x) ;
		ASSERT (info == GrB_SUCCESS) ;
	}

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	GrB_free          (&vec_it) ;
	GrB_free          (&iterator) ;
	Delta_Matrix_free (&DS) ;
	Delta_Matrix_free (&outgoing) ;
}

void Graph_CollectIncomingEdges
(
	Edge **edges,
	Graph *g,
	Node *nodes,
	uint64_t node_count
) {
	ASSERT (g          != NULL) ;
	ASSERT (nodes      != NULL) ;
	ASSERT (edges      != NULL) ;
	ASSERT (node_count > 0) ;

	GrB_Index    info ;
	GrB_Matrix   S ;
	Delta_Matrix DS ;
	Delta_Matrix outgoing ;

	GrB_Index m = Graph_RequiredMatrixDim (g) ;

	info = Delta_Matrix_new (&DS, GrB_BOOL, m, m, false) ;
	ASSERT (info == GrB_SUCCESS) ;

	info = Delta_Matrix_new (&outgoing, GrB_BOOL, m, m, false) ;
	ASSERT (info == GrB_SUCCESS) ;

	S = Delta_Matrix_M (DS) ;
	ASSERT (S != NULL) ;

	//--------------------------------------------------------------------------
	// build selection matrix
	//--------------------------------------------------------------------------

	for (uint64_t i = 0 ; i < node_count ; i++) {
		GrB_Index id = ENTITY_GET_ID (nodes + i) ;
		info = GrB_Matrix_setElement_BOOL (S, true, id, id) ;
		ASSERT (info == GrB_SUCCESS) ;
	}

	GxB_Iterator vec_it;
	GxB_Iterator iterator ;
	GxB_Iterator_new (&vec_it) ;
	GxB_Iterator_new (&iterator) ;

	int relationCount = Graph_RelationTypeCount (g) ;
	for (int r = 0; r < relationCount; r++) {
		Tensor R  = Graph_GetRelationMatrix (g, r, false) ;
		Tensor RT = Graph_GetRelationMatrix (g, r, true) ;

		info = Delta_mxm (outgoing, GxB_ANY_SECOND_BOOL, DS, RT) ;
		ASSERT (info == GrB_SUCCESS) ;
		GrB_Matrix x = DELTA_MATRIX_M (outgoing) ;

		info = GxB_Matrix_Iterator_attach (iterator, x, NULL) ;
		ASSERT (info == GrB_SUCCESS) ;

		info = GxB_Matrix_Iterator_seek (iterator, 0) ;
		while (info != GxB_EXHAUSTED) {
			// get the entry x(i,j)
			GrB_Index src, dest;
			GxB_Matrix_Iterator_getIndex (iterator, &src, &dest) ;

			uint64_t id ;
			info = Delta_Matrix_extractElement_UINT64 (&id, R, dest, src) ;
			ASSERT (info == GrB_SUCCESS) ;

			if (SCALAR_ENTRY (id)) {
				Edge e = {.id         = id,
						  .src_id     = dest,
						  .dest_id    = src,
						  .relationID = r,
						  .attributes = DataBlock_GetItem (g->edges, id)
				} ;

				ASSERT (e.attributes) ;
				array_append (*edges, e) ;
			}
			
			else {
				GrB_Vector V = AS_VECTOR (id) ;
				info = GxB_Vector_Iterator_attach (vec_it, V, NULL) ;

				// seek to the first entry
				info = GxB_Vector_Iterator_seek (vec_it, 0) ;
				while (info != GxB_EXHAUSTED) {
					// get the entry v(i)
					id = GxB_Vector_Iterator_getIndex (vec_it) ;

					Edge e = {.id         = id,
							  .src_id     = dest,
							  .dest_id    = src,
							  .relationID = r,
							  .attributes = DataBlock_GetItem (g->edges, id)
					} ;

					ASSERT (e.attributes) ;
					array_append (*edges, e) ;

					// move to the next entry in v
					info = GxB_Vector_Iterator_next (vec_it) ;
				}
			}

			// move to the next entry in X
			info = GxB_Matrix_Iterator_next (iterator) ;
		}

		info = GrB_Matrix_clear (x) ;
		ASSERT (info == GrB_SUCCESS) ;
	}

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	GrB_free           (&vec_it) ;
	GrB_free           (&iterator) ;
	Delta_Matrix_free  (&DS) ;
	Delta_Matrix_free  (&outgoing) ;
}

