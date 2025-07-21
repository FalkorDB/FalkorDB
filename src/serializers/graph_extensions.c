/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "graph_extensions.h"
#include "../RG.h"
#include "../util/datablock/oo_datablock.h"

// functions declerations - implemented in graph.c
bool Graph_FormConnection(Graph *g, NodeID src, NodeID dest, EdgeID edge_id, int r);

void Graph_EnsureNodeCap
(
	Graph *g,
	uint64_t cap
) {
	DataBlock_Ensure(g->nodes, cap);

	uint       n;
	GrB_Index  dim = Graph_RequiredMatrixDim(g);
	RG_Matrix  M   =  NULL;

	M = Graph_GetAdjacencyMatrix(g, false);
	RG_Matrix_resize(M, dim, dim);

	M = Graph_GetNodeLabelMatrix(g);
	RG_Matrix_resize(M, dim, dim);

	n = array_len(g->labels);
	for(int i = 0; i < n; i ++) {
		M = Graph_GetLabelMatrix(g, i);
		RG_Matrix_resize(M, dim, dim);
	}

	n = array_len(g->relations);
	for(int i = 0; i < n; i ++) {
		M = Graph_GetRelationMatrix(g, i, false);
		RG_Matrix_resize(M, dim, dim);
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
		RG_Matrix  M = Graph_GetLabelMatrix(g, label);
		GrB_Matrix m = RG_MATRIX_M(M);
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
	RG_Matrix node_labels    = Graph_GetNodeLabelMatrix(g);
	GrB_Matrix node_labels_m = RG_MATRIX_M(node_labels);

#if RG_DEBUG
	GrB_Index nvals;
	RG_Matrix_nvals(&nvals, node_labels);
	ASSERT(nvals == 0);
#endif

	GrB_Vector_new(&v, GrB_BOOL, node_count);

	for(int i = 0; i < label_count; i++) {
		RG_Matrix  M  =  Graph_GetLabelMatrix(g, i);
		GrB_Matrix m  =  RG_MATRIX_M(M);

		GxB_Vector_diag(v, m, 0, NULL);

		GxB_Row_subassign(node_labels_m, NULL, NULL, v, i, GrB_ALL, 0, NULL);
	}

	GrB_transpose(node_labels_m, NULL, NULL, node_labels_m, NULL);

	GrB_Vector_free(&v);
}

// optimized version of Graph_FormConnection
// used only when matrix doesn't contains multi edge values
static void _OptimizedSingleEdgeFormConnection
(
	Graph *g,
	NodeID src,
	NodeID dest,
	EdgeID edge_id,
	int r
) {
	GrB_Info info;
	RG_Matrix  M      =  Graph_GetRelationMatrix(g, r, false);
	RG_Matrix  adj    =  Graph_GetAdjacencyMatrix(g, false);
	GrB_Matrix m      =  RG_MATRIX_M(M);
	GrB_Matrix tm     =  RG_MATRIX_TM(M);
	GrB_Matrix adj_m  =  RG_MATRIX_M(adj);
	GrB_Matrix adj_tm =  RG_MATRIX_TM(adj);

	UNUSED(info);

	// rows represent source nodes, columns represent destination nodes

	//--------------------------------------------------------------------------
	// update adjacency matrix
	//--------------------------------------------------------------------------

	info = GrB_Matrix_setElement_BOOL(adj_m, true, src, dest);
	if(info == GrB_INVALID_INDEX) {
		// in case of writing out of matrix bounds resize the matrices
		RedisModule_Log(NULL, "notice", "RESIZE MATRIX SINGLE EDGE");

		uint64_t max_id = MAX(src, dest);
		Graph_EnsureNodeCap(g, max_id);
		info = GrB_Matrix_setElement_BOOL(adj_m, true, src, dest);
	}
	ASSERT(info == GrB_SUCCESS);
	info = GrB_Matrix_setElement_BOOL(adj_tm, true, dest, src);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// update relationship matrix
	//--------------------------------------------------------------------------

	info = GrB_Matrix_setElement_UINT64(m, edge_id, src, dest);
	ASSERT(info == GrB_SUCCESS);
	info = GrB_Matrix_setElement_BOOL(tm, true, dest, src);
	ASSERT(info == GrB_SUCCESS);

	// an edge of type r has just been created, update statistics
	// TODO: stats->edge_count[relation_idx] += nvals;
	GraphStatistics_IncEdgeCount(&g->stats, r, 1);
}

// set a given edge in the graph - Used for deserialization of graph
void Serializer_Graph_SetEdge
(
	Graph *g,
	bool multi_edge,
	EdgeID edge_id,
	NodeID src,
	NodeID dest,
	int r,
	Edge *e
) {
	GrB_Info info;

	AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->edges, edge_id);
	*set = NULL;

	e->id         =  edge_id;
	e->src_id     =  src;
	e->dest_id    =  dest;
	e->attributes =  set;
	e->relationID =  r;

	if(multi_edge) {
		if(!Graph_FormConnection(g, src, dest, edge_id, r)) {
			// resize matrices
			RedisModule_Log(NULL, "notice", "RESIZE MATRIX MULTI EDGE");

			uint64_t max_id = MAX(src, dest);
			Graph_EnsureNodeCap(g, max_id);
			bool res = Graph_FormConnection(g, src, dest, edge_id, r);
			ASSERT(res == true);
		}
	} else {
		_OptimizedSingleEdgeFormConnection(g, src, dest, edge_id, r);
	}
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

