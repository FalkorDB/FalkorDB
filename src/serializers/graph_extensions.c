/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../RG.h"
#include "../util/arr.h"
#include "graph_extensions.h"
#include "../util/datablock/oo_datablock.h"

// functions declerations - implemented in graph.c
void Graph_FormConnection(Graph *g, NodeID src, NodeID dest, EdgeID edge_id, int r);

void Graph_EnsureNodeCap
(
	Graph *g,
	uint64_t cap
) {
	DataBlock_Ensure(g->nodes, cap);

	uint       n;
	GrB_Index  dim = Graph_RequiredMatrixDim(g);
	Delta_Matrix  M   =  NULL;

	M = Graph_GetAdjacencyMatrix(g, false);
	Delta_Matrix_resize(M, dim, dim);

	M = Graph_GetNodeLabelMatrix(g);
	Delta_Matrix_resize(M, dim, dim);

	n = array_len(g->labels);
	for(int i = 0; i < n; i ++) {
		M = Graph_GetLabelMatrix(g, i);
		Delta_Matrix_resize(M, dim, dim);
	}

	n = array_len(g->relations);
	for(int i = 0; i < n; i ++) {
		M = Graph_GetRelationMatrix(g, i, false);
		Delta_Matrix_resize(M, dim, dim);
	}
}

inline void Serializer_Graph_MarkEdgeDeleted
(
	Graph *g,
	EdgeID id
) {
	DataBlock_MarkAsDeletedOutOfOrder(g->edges, id);
}

inline void Serializer_Graph_MarkNodeDeleted
(
	Graph *g,
	NodeID id
) {
	DataBlock_MarkAsDeletedOutOfOrder(g->nodes, id);
}

void Serializer_Graph_SetNode
(
	Graph *g,
	NodeID id,
	LabelID *labels,
	uint label_count,
	Node *n
) {
	ASSERT(g);

	AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->nodes, id);
	*set = NULL;

	n->id =  id;
	n->attributes =  set;
	GrB_Info info;
	UNUSED(info);

	for(uint i = 0; i < label_count; i ++) {
		LabelID label = labels[i];
		// set label matrix at position [id, id]
		Delta_Matrix  M = Graph_GetLabelMatrix(g, label);
		GrB_Matrix m    = Delta_Matrix_M(M);
		info = GrB_Matrix_setElement_BOOL(m, true, id, id);
		if(info == GrB_INVALID_INDEX) {
			RedisModule_Log(NULL, "notice", "RESIZE LABEL MATRIX");
			Graph_EnsureNodeCap(g, id);
			info = GrB_Matrix_setElement_BOOL(m, true, id, id);
		}
		ASSERT(info == GrB_SUCCESS);
	}
}

// computes NodeLabelMarix out of label matrices
// NodeLabelMatrix[:i] = diag(LabelMatrix[i])
// must be called once after all virtual keys loaded for perf
void Serializer_Graph_SetNodeLabels
(
	Graph *g
) {
	ASSERT(g);

	GrB_Vector v;
	int node_count           = Graph_RequiredMatrixDim(g);
	int label_count          = Graph_LabelTypeCount(g);
	Delta_Matrix node_labels = Graph_GetNodeLabelMatrix(g);
	GrB_Matrix node_labels_m = Delta_Matrix_M(node_labels);

#if RG_DEBUG
	GrB_Index nvals;
	Delta_Matrix_nvals(&nvals, node_labels);
	ASSERT(nvals == 0);
#endif

	GrB_Vector_new(&v, GrB_BOOL, node_count);

	for(int i = 0; i < label_count; i++) {
		Delta_Matrix  M  =  Graph_GetLabelMatrix(g, i);
		GrB_Matrix m     =  Delta_Matrix_M(M);

		GxB_Vector_diag(v, m, 0, NULL);

		GxB_Row_subassign(node_labels_m, NULL, NULL, v, i, GrB_ALL, 0, NULL);
	}

	GrB_transpose(node_labels_m, NULL, NULL, node_labels_m, NULL);

	GrB_Vector_free(&v);
}

// optimized version of Graph_FormConnection
void Serializer_OptimizedFormConnections
(
	Graph *g,
	RelationID r,                     // relation id
	const NodeID *restrict srcs,      // src node id
	const NodeID *restrict dests,     // dest node id
	const EdgeID *restrict ids,       // edge id
	uint64_t n,                       // number of entries
	NodeID max_id,
	bool multi_edge                   // multi edge batch
) {
	// validations
	ASSERT(n      >  0);
	ASSERT(g      != NULL);
	ASSERT(ids    != NULL);
	ASSERT(srcs   != NULL);
	ASSERT(dests  != NULL);
	ASSERT(max_id != INVALID_ENTITY_ID);

	GrB_Info   info;   // GraphBLAS operation result
	GrB_Index  nrows;  // #rows

	Tensor       M   = Graph_GetRelationMatrix(g, r, false);  // relation matrix
	Delta_Matrix adj = Graph_GetAdjacencyMatrix(g, false);    // adj matrix

	GrB_Matrix m      = Delta_Matrix_M(M);
	GrB_Matrix tm     = Delta_Matrix_M(Delta_Matrix_getTranspose(M));
	GrB_Matrix adj_m  = Delta_Matrix_M(adj);
	GrB_Matrix adj_tm = Delta_Matrix_M(Delta_Matrix_getTranspose(adj));

	UNUSED(info);

	// make sure matrix dimension includes max_id
	info = GrB_Matrix_nrows(&nrows, m);
	ASSERT(info == GrB_SUCCESS);

	if(nrows < max_id) {
		// resize the matrices
		RedisModule_Log(NULL, "notice", "RESIZE MATRIX");
		Graph_EnsureNodeCap(g, max_id);
	}

	for(uint64_t i = 0; i < n; i++) {
		uint64_t  x   = ids[i];
		GrB_Index row = srcs[i];
		GrB_Index col = dests[i];

		//----------------------------------------------------------------------
		// update adjacency matrix
		//----------------------------------------------------------------------

		info = GrB_Matrix_setElement_BOOL(adj_m, true, row, col);
		ASSERT(info == GrB_SUCCESS);

		// TODO: might be better to compute transposes at the very end of the load
		// process
		info = GrB_Matrix_setElement_BOOL(adj_tm, true, col, row);
		ASSERT(info == GrB_SUCCESS);

		//----------------------------------------------------------------------
		// update relationship matrix
		//----------------------------------------------------------------------

		if(!multi_edge) {
			//uint64_t _x;
			//info = GrB_Matrix_extractElement_UINT64(&_x, m, row, col);
			//ASSERT(info == GrB_NO_VALUE);

			info = GrB_Matrix_setElement_UINT64(m, x, row, col);
			ASSERT(info == GrB_SUCCESS);

			info = GrB_Matrix_setElement_UINT64(tm, x, col, row);
			ASSERT(info == GrB_SUCCESS);
		}
	}

	if(multi_edge) {
		Tensor_SetElements(M, srcs, dests, ids, n);
	}

	// update graph statistics
	GraphStatistics_IncEdgeCount(&g->stats, r, n);
}

// allocate edge attribute-set
void Serializer_Graph_AllocEdgeAttributes
(
	Graph *g,
	EdgeID edge_id,
	Edge *e
) {
	AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->edges, edge_id);
	*set = NULL;
	e->attributes = set;
}

// returns the graph deleted nodes list
uint64_t *Serializer_Graph_GetDeletedNodesList
(
	Graph *g
) {
	return g->nodes->deletedIdx;
}

// returns the graph deleted edge list
uint64_t *Serializer_Graph_GetDeletedEdgesList
(
	Graph *g
) {
	return g->edges->deletedIdx;
}

