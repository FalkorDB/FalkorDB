/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graphcontext.h"
#include "graph_memoryUsage.h"
#include "delta_matrix/delta_matrix_iter.h"
#include "entities/attribute_set.h"
#include "../schema/schema.h"
#include "../index/index.h"
#include "../util/arr.h"

#include <sys/param.h>

#define MB (1 << 20)

//------------------------------------------------------------------------------
// attribute and index estimation helpers
//------------------------------------------------------------------------------

// checks whether any node in the graph is associated with more than one label
static bool _Overlapping
(
	const GrB_Matrix lbls,  // [input] Node-label adjacency matrix
	GrB_Vector *V           // [output] Boolean vector: V[i]=true if node i has a label
) {
	ASSERT (lbls != NULL) ;
	ASSERT (V != NULL && *V == NULL) ;

	GrB_Index nrows ;

	GrB_OK (GrB_Matrix_nrows (&nrows, lbls)) ;
	GrB_OK (GrB_Vector_new (V, GrB_BOOL, nrows)) ;

	GrB_OK (GrB_Matrix_reduce_Monoid (*V, NULL, NULL, GxB_ANY_BOOL_MONOID, lbls,
			NULL)) ;

	GrB_Index lbls_nvals ;
	GrB_OK (GrB_Matrix_nvals (&lbls_nvals, lbls)) ;

	GrB_Index v_nvals ;
	GrB_OK (GrB_Vector_nvals (&v_nvals, *V)) ;

	return (lbls_nvals > v_nvals) ;
}

// estimates memory usage for a vector of nodes by sampling
static size_t _SampleVector
(
	const Graph *g,      // graph
	const GrB_Vector V,  // vector of node IDs (non-zero entries)
	GxB_Iterator it,     // [input/output] reusable vector iterator
	int64_t samples      // max samples to collect per label
) {
	GrB_Index nvals ;

	GrB_OK (GrB_Vector_nvals (&nvals, V)) ;

	if (nvals == 0) {
		return 0 ;
	}

	size_t  memory_usage      = 0 ;
	int64_t remaining_samples = MIN (nvals, samples) ;
	int64_t attempted_samples = remaining_samples ;

	GrB_OK (GxB_Vector_Iterator_attach (it, V, NULL)) ;

	GrB_Info info = GxB_Vector_Iterator_seek (it, 0) ;
	while (info != GxB_EXHAUSTED && remaining_samples > 0) {
		GrB_Index i = GxB_Vector_Iterator_getIndex (it) ;

		Node n ;
		bool node_found = Graph_GetNode (g, i, &n) ;
		ASSERT (node_found == true) ;

		AttributeSet set = GraphEntity_GetAttributes ((GraphEntity*)&n) ;
		memory_usage += AttributeSet_memoryUsage (set) ;

		remaining_samples-- ;

		info = GxB_Vector_Iterator_next (it) ;
	}

	int64_t n_sampled = attempted_samples - remaining_samples ;
	if (n_sampled == 0) {
		return 0 ;
	}

	double avg = (double)memory_usage / n_sampled ;

	return (size_t)(avg * nvals) ;
}

// estimates memory consumption of unlabeled nodes
static size_t _UnlabeledNodesMemory
(
	const Graph *g,  // graph
	GrB_Vector V,    // vector where V[i]=1 marks labeled nodes; inversed in place
	int64_t samples  // number of nodes to sample
) {
	ASSERT (g       != NULL) ;
	ASSERT (V       != NULL) ;
	ASSERT (samples > 0) ;

	GrB_Scalar x ;

	GrB_OK (GrB_Scalar_new (&x, GrB_BOOL)) ;
	GrB_OK (GrB_Scalar_setElement (x, true)) ;

	GrB_Index len ;
	GrB_OK (GrB_Vector_size (&len, V)) ;

	// V<!V> = true  (mark unlabeled nodes)
	GrB_OK (GrB_Vector_assign_Scalar (V, V, NULL, x, GrB_ALL, len, GrB_DESC_RC)) ;
	GrB_OK (GrB_free (&x)) ;

	GxB_Iterator it ;
	GrB_OK (GxB_Iterator_new (&it)) ;

	NodeID *nodes ;
	uint64_t n_nodes ;
	Graph_DeletedNodes (g, &nodes, &n_nodes) ;
	ASSERT (nodes != NULL) ;

	for (uint64_t i = 0 ; i < n_nodes ; i++) {
		GrB_OK (GrB_Vector_removeElement (V, nodes[i])) ;
	}

	size_t memory_usage = _SampleVector (g, V, it, samples) ;

	GrB_free (&it) ;
	rm_free (nodes) ;

	return memory_usage ;
}

// estimates amortized memory usage for nodes with overlapping labels
static void _EstimateOverlapingNodeAttributeMemory
(
	const Graph *g,            // graph
	GrB_Matrix lbls,           // labels matrix
	int64_t samples,           // max samples per label
	MemoryUsageResult *result  // [output] memory usage
) {
	ASSERT (g       != NULL) ;
	ASSERT (samples > 0) ;

	size_t node_memory_usage = 0 ;
	int n_lbls = Graph_LabelTypeCount (g) ;

	GrB_Vector     V     = NULL ;
	GrB_Vector     P     = NULL ;
	GxB_Iterator   it    = NULL ;
	GrB_Index      nrows = 0 ;
	GrB_Descriptor desc  = GrB_DESC_RSC ;

	GrB_OK (GrB_Matrix_nrows (&nrows, lbls)) ;
	GrB_OK (GrB_Vector_new (&P, GrB_BOOL, nrows)) ;
	GrB_OK (GrB_Vector_new (&V, GrB_BOOL, nrows)) ;
	GrB_OK (GxB_Iterator_new (&it)) ;

	for (int i = 0 ; i < n_lbls ; i++) {
		// V<!P> = lbls[:, i]  (extract unprocessed nodes for label i)
		GrB_OK (GrB_Col_extract (V, P, NULL, lbls, GrB_ALL, nrows, i, desc)) ;

		node_memory_usage = _SampleVector (g, V, it, samples) ;
		arr_append (result->node_attr_by_label_sz, node_memory_usage) ;

		// mark processed: P = P | V
		GrB_OK (GrB_Vector_eWiseAdd_Semiring (P, NULL, NULL, GxB_ANY_PAIR_BOOL,
				P, V, GrB_DESC_S)) ;
	}

	GrB_free (&V) ;
	GrB_free (&P) ;
	GrB_free (&it) ;
}

// estimate total memory usage for all labeled nodes assuming no label overlap
static void _EstimateNonOverlapingNodeAttributeMemory
(
	const Graph *g,            // graph
	int64_t sample_size,       // number of nodes to sample per label
	MemoryUsageResult *result  // [output] memory usage
) {
	ASSERT (g           != NULL) ;
	ASSERT (sample_size >= 0) ;

	int n_lbls = Graph_LabelTypeCount (g) ;

	for (LabelID l = 0 ; l < n_lbls ; l++) {
		Node node ;
		GrB_Index id ;
		GrB_Info info ;
		Delta_MatrixTupleIter it ;
		Delta_Matrix L = Graph_GetLabelMatrix (g, l) ;

		size_t label_memory_usage = 0 ;
		int64_t nodes_remaining = sample_size ;

		info = Delta_MatrixTupleIter_attach (&it, L) ;
		ASSERT (info == GrB_SUCCESS) ;

		info = Delta_MatrixTupleIter_iterate_range (&it, 0, UINT64_MAX) ;
		ASSERT (info == GrB_SUCCESS) ;

		while (Delta_MatrixTupleIter_next_BOOL (&it, &id, NULL, NULL)
				== GrB_SUCCESS && nodes_remaining > 0) {
			bool found = Graph_GetNode (g, id, &node) ;
			ASSERT (found == true) ;

			AttributeSet set = GraphEntity_GetAttributes ((GraphEntity*)&node) ;
			label_memory_usage += AttributeSet_memoryUsage (set) ;
			nodes_remaining-- ;
		}

		Delta_MatrixTupleIter_detach (&it) ;

		int64_t sampled = MAX (1, sample_size - nodes_remaining) ;
		double avg_label_mem = (double)label_memory_usage / sampled ;
		int64_t total_labeled_nodes = Graph_LabeledNodeCount (g, l) ;

		label_memory_usage = (size_t)(avg_label_mem * total_labeled_nodes) ;
		arr_append (result->node_attr_by_label_sz, label_memory_usage) ;
	}
}

// estimate amortized memory consumption of node attribute sets
static void _EstimateNodeAttributeMemory
(
	const GraphContext *gc,    // graph context
	const Graph *g,            // graph
	int64_t samples,           // number of nodes to sample
	MemoryUsageResult *result  // [output] memory usage
) {
	ASSERT (g       != NULL) ;
	ASSERT (gc      != NULL) ;
	ASSERT (samples > 0) ;

	GrB_Vector V    = NULL ;
	GrB_Matrix lbls = NULL ;

	size_t  node_memory_usage = 0 ;
	int64_t node_count        = Graph_NodeCount (g) ;
	int64_t sample_size       = MIN (node_count, samples) ;

	GrB_Orientation fmt = GrB_COLMAJOR ;
	Delta_Matrix D = Graph_GetNodeLabelMatrix (g) ;
	GrB_OK (Delta_Matrix_export (&lbls, D, GrB_BOOL, &fmt)) ;

	bool overlapping = _Overlapping (lbls, &V) ;

	GrB_Index nvals ;
	GrB_OK (GrB_Vector_nvals (&nvals, V)) ;

	bool has_unlabeled_nodes = Graph_NodeCount (g) > nvals ;
	if (has_unlabeled_nodes) {
		GrB_OK (GrB_Vector_resize (V, Graph_UncompactedNodeCount (g))) ;
		node_memory_usage = _UnlabeledNodesMemory (g, V, samples) ;
		result->unlabeled_node_attr_sz = node_memory_usage ;
	}

	GrB_OK (GrB_free (&V)) ;

	if (overlapping) {
		_EstimateOverlapingNodeAttributeMemory (g, lbls, sample_size, result) ;
	} else {
		_EstimateNonOverlapingNodeAttributeMemory (g, sample_size, result) ;
	}

	GrB_free (&lbls) ;
}

// estimate edges attribute-set memory consumption
static void _EstimateEdgeAttributeMemory
(
	GraphContext *gc,          // graph context
	const Graph *g,            // graph
	uint samples,              // samples per relationship type
	MemoryUsageResult *result  // [output] memory usage
) {
	int64_t n_edges           = Graph_EdgeCount (g) ;
	int64_t sample_size       = MIN (n_edges, samples) ;
	int64_t edges_sample_size = sample_size ;
	size_t  edge_memory_usage = 0 ;

	unsigned short n = GraphContext_SchemaCount (gc, SCHEMA_EDGE) ;
	for (RelationID r = 0 ; r < n ; r++) {
		Edge edge ;
		GrB_Index id ;
		Tensor R ;
		TensorIterator it ;
		size_t relation_memory_usage = 0 ;

		R = Graph_GetRelationMatrix (g, r, false) ;
		TensorIterator_ScanRange (&it, R, 0, UINT64_MAX, false) ;

		while (TensorIterator_next (&it, NULL, NULL, &id, NULL) &&
				edges_sample_size > 0) {
			bool res = Graph_GetEdge (g, id, &edge) ;
			ASSERT (res == true) ;

			AttributeSet set = GraphEntity_GetAttributes ((GraphEntity*)&edge) ;
			relation_memory_usage += AttributeSet_memoryUsage (set) ;
			edges_sample_size-- ;
		}

		int64_t n_sampled_edges = MAX (1, sample_size - edges_sample_size) ;
		edge_memory_usage = (relation_memory_usage / (double)n_sampled_edges)
			* Graph_RelationEdgeCount (g, r) ;

		arr_append (result->edge_attr_by_type_sz, edge_memory_usage) ;

		edges_sample_size = sample_size ;
	}
}

// returns the amortized memory consumption of a graph
// populates all MemoryUsageResult fields and converts all sizes to MB on return
// caller must hold at least the graph read lock
void GraphContext_EstimateMemoryUsage
(
	GraphContext      *gc,
	double             samples,
	MemoryUsageResult *result
) {
	ASSERT (gc      != NULL) ;
	ASSERT (samples >= 0) ;
	ASSERT (result  != NULL) ;

	const Graph *g = GraphContext_GetGraph (gc) ;

	Graph_memoryUsage (g, result) ;

	_EstimateNodeAttributeMemory (gc, g, samples, result) ;
	_EstimateEdgeAttributeMemory (gc, g, samples, result) ;

	//--------------------------------------------------------------------------
	// collect indices memory usage
	//--------------------------------------------------------------------------

	int n_node_schema = GraphContext_SchemaCount (gc, SCHEMA_NODE) ;
	for (int i = 0 ; i < n_node_schema ; i++) {
		Schema *s = GraphContext_GetSchemaByID (gc, i, SCHEMA_NODE) ;

		if (!Schema_HasIndices (s)) continue ;

		Index active_idx  = ACTIVE_IDX (s) ;
		Index pending_idx = PENDING_IDX (s) ;

		if (active_idx  != NULL) result->indices_sz += Index_MemoryUsage (active_idx) ;
		if (pending_idx != NULL) result->indices_sz += Index_MemoryUsage (pending_idx) ;
	}

	int n_edge_schema = GraphContext_SchemaCount (gc, SCHEMA_EDGE) ;
	for (int i = 0 ; i < n_edge_schema ; i++) {
		Schema *s = GraphContext_GetSchemaByID (gc, i, SCHEMA_EDGE) ;

		if (!Schema_HasIndices (s)) continue ;

		Index active_idx  = ACTIVE_IDX (s) ;
		Index pending_idx = PENDING_IDX (s) ;

		if (active_idx  != NULL) result->indices_sz += Index_MemoryUsage (active_idx) ;
		if (pending_idx != NULL) result->indices_sz += Index_MemoryUsage (pending_idx) ;
	}

	//--------------------------------------------------------------------------
	// sum and convert all fields from bytes to MB
	//--------------------------------------------------------------------------

	for (int i = 0 ; i < arr_len (result->node_attr_by_label_sz) ; i++) {
		result->total_graph_sz_mb += result->node_attr_by_label_sz[i] ;
		result->node_attr_by_label_sz[i] /= MB ;
	}

	for (int i = 0 ; i < arr_len (result->edge_attr_by_type_sz) ; i++) {
		result->total_graph_sz_mb += result->edge_attr_by_type_sz[i] ;
		result->edge_attr_by_type_sz[i] /= MB ;
	}

	result->total_graph_sz_mb +=
		result->indices_sz             +
		result->lbl_matrices_sz        +
		result->rel_matrices_sz        +
		result->node_block_storage_sz  +
		result->edge_block_storage_sz  +
		result->unlabeled_node_attr_sz ;

	result->indices_sz             /= MB ;
	result->lbl_matrices_sz        /= MB ;
	result->rel_matrices_sz        /= MB ;
	result->node_block_storage_sz  /= MB ;
	result->edge_block_storage_sz  /= MB ;
	result->unlabeled_node_attr_sz /= MB ;
	result->total_graph_sz_mb      /= MB ;
}
