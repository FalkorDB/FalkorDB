/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "cmd_memory.h"
#include "../errors/error_msgs.h"
#include "../util/thpool/pool.h"
#include "../graph/graphcontext.h"
#include "../graph/graph_memoryUsage.h"

#define MB (1 <<20)

// GRAPH.MEMORY command context
typedef struct {
	GraphContext *gc;              // graph context
	int64_t samples;               // number of samples to inspect
	RedisModuleBlockedClient *bc;  // blocked client
} GraphMemoryCtx;

// checks whether any node in the graph is associated with more than one label
// returns true if there exists at least one node with multiple labels
static bool _Overlapping
(
	const GrB_Matrix lbls,  // [input] Node-label adjacency matrix
	GrB_Vector *V           // [output] Boolean vector: V[i] = true
							// if node i has at least one label
) {
	ASSERT (lbls != NULL) ;
	ASSERT (V != NULL && *V == NULL) ;

	GrB_Index nrows ;

	// create V
	GrB_OK (GrB_Matrix_nrows (&nrows, lbls)) ;
	GrB_OK (GrB_Vector_new (V, GrB_BOOL, nrows)) ;

	//----------------------------------------------------------------------
    // reduce each row of the labels matrix to indicate if a node is labeled
    // V[i] = any(lbls(i,:))
    //----------------------------------------------------------------------

	GrB_OK (GrB_Matrix_reduce_Monoid (*V, NULL, NULL, GxB_ANY_BOOL_MONOID, lbls,
			NULL)) ;

	//----------------------------------------------------------------------
    // if total label assignments > number of non-zero entries in V,
    // at least one node has more than one label
    //----------------------------------------------------------------------

	GrB_Index lbls_nvals ;
	GrB_OK (GrB_Matrix_nvals (&lbls_nvals, lbls)) ;

	GrB_Index v_nvals ;
	GrB_OK (GrB_Vector_nvals (&v_nvals, *V)) ;

	return (lbls_nvals > v_nvals) ;
}

// estimates the memory consumption for a vector of nodes by sampling
// returns estimated total memory usage for all nodes in vector V
static size_t _SampleVector
(
    const Graph *g,      // graph
    const GrB_Vector V,  // vector of node IDs (non-zero entries)
    GxB_Iterator it,     // [input/output] reusable vector iterator
    int64_t samples      // max samples to collect per label
) {
	GrB_Index nvals;

	GrB_OK (GrB_Vector_nvals (&nvals, V)) ;

	// if the vector is empty, nothing to sample
	if (nvals == 0) {
		return 0 ;
	}

	size_t  memory_usage      = 0 ;
	int64_t remaining_samples = MIN (nvals, samples) ;
	int64_t attempted_samples = remaining_samples ;

	// attach iterator to vector V
	GrB_OK (GxB_Vector_Iterator_attach (it, V, NULL)) ;

	// seek to the first entry
	GrB_Info info = GxB_Vector_Iterator_seek (it, 0) ;
	while (info != GxB_EXHAUSTED && remaining_samples > 0) {
		// get the entry V(i)
		GrB_Index i = GxB_Vector_Iterator_getIndex (it) ;

		Node n ;
		bool node_found = Graph_GetNode (g, i, &n) ;
		ASSERT (node_found == true) ;

		AttributeSet set = GraphEntity_GetAttributes ((GraphEntity*)&n) ;
		memory_usage += AttributeSet_memoryUsage (set) ;

		remaining_samples-- ;

		// advance iterator
		info = GxB_Vector_Iterator_next (it) ;
	}

	int64_t n_sampled = attempted_samples - remaining_samples ;
	if (n_sampled == 0) {
		return 0 ;
	}

	// estimate total memory usage by scaling the average sample
	double avg = (double)memory_usage / n_sampled ;

	return (size_t)(avg * nvals) ;
}

// estimates memory consumption of unlabeled nodes in the graph
// this function identifies nodes not assigned any label and samples them
static size_t _UnlabeledNodesMemory
(
	const Graph *g,      // graph
    GrB_Vector V,        // vector where V[i] = 1 marks labeled nodes
                         // will be inversed
    int64_t samples      // number of nodes to sample
) {
	ASSERT(g != NULL);
	ASSERT(V != NULL);
	ASSERT(samples > 0);

    GrB_Scalar x;

	// Create a scalar 'true' to assign to unlabeled entries
	GrB_OK (GrB_Scalar_new (&x, GrB_BOOL)) ;
	GrB_OK (GrB_Scalar_setElement(x, true)) ;

	// fill in complement of V (i.e., unlabeled nodes)
	GrB_Index len ;
	GrB_OK (GrB_Vector_size(&len, V)) ;

	// V<!V> = true --> (mark unlabeled nodes)
	GrB_OK (GrB_Vector_assign_Scalar (V, V, NULL, x, GrB_ALL, len, GrB_DESC_RC)) ;
	GrB_OK (GrB_free (&x)) ;

	// sample memory usage for unlabeled nodes
	GxB_Iterator it ;
	GrB_OK (GxB_Iterator_new (&it)) ;

	//--------------------------------------------------------------------------
	// remove deleted nodes from V
	//--------------------------------------------------------------------------

	NodeID *nodes ;     // array of deleted node IDs
	uint64_t n_nodes ;  // number of deleted nodes

	// populate nodes with deleted node IDs
	Graph_DeletedNodes (g, &nodes, &n_nodes) ;
	ASSERT(nodes != NULL);

	// remove deleted node IDs from V
	for (uint64_t i = 0 ; i < n_nodes;  i++) {
		GrB_OK (GrB_Vector_removeElement( V, nodes [i])) ; 
	}

	size_t memory_usage = _SampleVector (g, V, it, samples) ;

	// cleanup
	GrB_free (&it) ;
	rm_free (nodes) ;

	return memory_usage ;
}

// estimates amortized memory usage for nodes with overlapping labels
// by sampling
// this method is slower and should only be used when nodes may share labels
// for faster estimation
// use _EstimateNodeAttributeMemory when labels don't overlap
static void _EstimateOverlapingNodeAttributeMemory
(
	const Graph *g,            // graph
	GrB_Matrix lbls,           // labels matrix
	int64_t samples,           // max samples per label
	MemoryUsageResult *result  // [output] memory usage
) {
	ASSERT (g != NULL) ;
	ASSERT (samples > 0) ;

	size_t node_memory_usage = 0 ;
	int n_lbls = Graph_LabelTypeCount (g) ;

	GrB_Vector     V     = NULL ;          // vector for current label column
	GrB_Vector     P     = NULL ;          // tracks processed nodes
	GxB_Iterator   it    = NULL ;          // iterator for sampling
	GrB_Index      nrows = 0 ;             // number of nodes
	GrB_Descriptor desc  = GrB_DESC_RSC ;  // descriptor for masked extraction

	// get the number of rows (nodes)
	GrB_OK (GrB_Matrix_nrows (&nrows, lbls)) ;

	// create a vector to mark processed nodes
	GrB_OK (GrB_Vector_new (&P, GrB_BOOL, nrows)) ;

	// create a reusable vector for label columns
	GrB_OK (GrB_Vector_new (&V, GrB_BOOL, nrows)) ;

	// create a reusable iterator
	GrB_OK (GxB_Iterator_new (&it)) ;

	// iterate over each label
	for (int i = 0 ; i < n_lbls ; i++) {
		// extract column i (label i), skipping already processed entries
        // V<!P> = lbls[:, i]
		GrB_OK (GrB_Col_extract (V, P, NULL, lbls, GrB_ALL, nrows, i, desc)) ;

		// sample attribute memory usage from unprocessed nodes within label
		node_memory_usage = _SampleVector (g, V, it, samples) ;
		arr_append (result->node_attr_by_label_sz, node_memory_usage) ;

		// mark these nodes as processed: P = P + V
		GrB_OK (GrB_Vector_eWiseAdd_Semiring (P, NULL, NULL, GxB_ANY_PAIR_BOOL,
				P, V, GrB_DESC_S)) ;
	}

	// clean up
	GrB_free (&V) ;
	GrB_free (&P) ;
	GrB_free (&it) ;
}

// estimate total memory usage for all labeled nodes,
// assuming there is no label overlap between nodes
// for overlapping labels,
// use the more expensive _EstimateOverlapingNodeAttributeMemory
static void _EstimateNonOverlapingNodeAttributeMemory
(
    const Graph *g,            // graph
    int64_t sample_size,       // number of nodes to sample per label
	MemoryUsageResult *result  // [output] memory usage
) {
	ASSERT(g != NULL);
	ASSERT(sample_size >= 0);

	size_t total_memory_usage = 0;
	int n_lbls = Graph_LabelTypeCount(g);

	for(LabelID l = 0; l < n_lbls; l++) {
		Node node;
		GrB_Index id;
		GrB_Info info;
		Delta_MatrixTupleIter it;
		Delta_Matrix L = Graph_GetLabelMatrix(g, l);

		size_t label_memory_usage = 0;
		int64_t nodes_remaining = sample_size;

		// attach iterator to label matrix
		info = Delta_MatrixTupleIter_attach (&it, L) ;
		ASSERT (info == GrB_SUCCESS) ;

		info = Delta_MatrixTupleIter_iterate_range (&it, 0, UINT64_MAX) ;
		ASSERT (info == GrB_SUCCESS) ;

		// sample up to `sample_size` nodes with this label
		while (Delta_MatrixTupleIter_next_BOOL (&it, &id, NULL, NULL)
				== GrB_SUCCESS && nodes_remaining > 0) {
			// compute the memory consumption of the current node
			bool found = Graph_GetNode (g, id, &node) ;
			ASSERT (found == true) ;

			AttributeSet set = GraphEntity_GetAttributes ((GraphEntity*)&node) ;

			label_memory_usage += AttributeSet_memoryUsage (set) ;
			nodes_remaining-- ;
		}

		Delta_MatrixTupleIter_detach (&it) ;

		// set number of sampled nodes
		int64_t sampled = MAX (1, sample_size - nodes_remaining) ;

		// compute average and scale by number of labeled nodes
		double avg_label_mem = (double)label_memory_usage / sampled ;
		int64_t total_labeled_nodes = Graph_LabeledNodeCount (g, l) ;

		label_memory_usage = (size_t)(avg_label_mem * total_labeled_nodes) ;

		arr_append (result->node_attr_by_label_sz, label_memory_usage) ;
	}
}

// estimate amortized memory consumption of node attribute sets
// the method adapts based on node label characteristics:
// - if there are overlapping labels, a more expensive estimation is used
// - if there are unlabeled nodes, they are sampled separately
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

	GrB_Info info ;
	GrB_Vector V    = NULL ;
	GrB_Matrix lbls = NULL ;

	size_t  node_memory_usage = 0 ;                   // node memory usage
	int64_t node_count = Graph_NodeCount (g) ;        // number of nodes
	int64_t sample_size = MIN (node_count, samples) ; // sample size

	//--------------------------------------------------------------------------
	// determine if the graph has overlapping labels
	//--------------------------------------------------------------------------

	// export in CSC (column-major) so that per-label column extraction in
	// _EstimateOverlapingNodeAttributeMemory is a direct vector lookup rather
	// than an O(nnz) row scan; setting orientation on the empty matrix here
	// avoids triggering GB_transpose_in_place under concurrent query load
	GrB_Orientation fmt = GrB_COLMAJOR ;
	Delta_Matrix D = Graph_GetNodeLabelMatrix (g) ;
	GrB_OK (Delta_Matrix_export (&lbls, D, GrB_BOOL, &fmt)) ;

	bool overlapping = _Overlapping (lbls, &V) ;

	//--------------------------------------------------------------------------
	// check for unlabeled nodes
	//--------------------------------------------------------------------------

	GrB_Index nvals ;
	GrB_OK (GrB_Vector_nvals (&nvals, V)) ;

	bool has_unlabeled_nodes = Graph_NodeCount (g) > nvals ;  // unlabeled nodes
	if (has_unlabeled_nodes) {
		// resize vector to match actual number of nodes in the graph
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
	uint samples,              // #samples per relationship type to collect
	MemoryUsageResult *result  // [output] memory usage
) {
	int64_t n_edges           = Graph_EdgeCount(g);     // number of edges
	int64_t sample_size       = MIN(n_edges, samples);  // sample size
	int64_t edges_sample_size = sample_size;            // edges sample size
	size_t  edge_memory_usage = 0;                      // sum memory

	// number of relationship-types
	unsigned short n = GraphContext_SchemaCount(gc, SCHEMA_EDGE);
	for(RelationID r = 0; r < n; r++) {
		Edge edge;
		GrB_Index id;
		GrB_Info info;
		Tensor R;
		TensorIterator it;
		size_t relation_memory_usage = 0;

		// attach iterator to the current relation matrix
		R = Graph_GetRelationMatrix(g, r, false);

		TensorIterator_ScanRange(&it, R, 0, UINT64_MAX, false);

		// iterate over relation matrix, limit #iterations to simple_size
		while(TensorIterator_next(&it, NULL, NULL, &id, NULL) &&
				edges_sample_size > 0) {
			// compute the memory consumption of the current edge
			bool res = Graph_GetEdge(g, id, &edge);
			ASSERT(res == true);

			AttributeSet set = GraphEntity_GetAttributes((GraphEntity*)&edge);

			relation_memory_usage += AttributeSet_memoryUsage(set);
			edges_sample_size--;
		}

		// set number of sampled edges
		int64_t n_sampled_edges = MAX (1, sample_size - edges_sample_size) ;

		// compute weighted average
		edge_memory_usage = (relation_memory_usage / (double) n_sampled_edges)
			* Graph_RelationEdgeCount(g, r);

		arr_append(result->edge_attr_by_type_sz, edge_memory_usage);

		// reset sample size
		edges_sample_size = sample_size;
	}
}

// returns the amortized amount of memory consumed by a graph
static void _estimate_memory_consumption
(
	GraphContext *gc,          // graph context
	double samples,            // random set size
	MemoryUsageResult *result  // [output] memory usage
) {
	ASSERT (gc      != NULL) ;
	ASSERT (samples >= 0) ;
	ASSERT (result  != NULL) ;

	// a graph memory consumption is spread across multiple components:
	// 1. matrices
	// 2. datablocks
	// 3. attributes
	// 4. indices

	const Graph *g = GraphContext_GetGraph (gc) ;

	// collect graph's memory consumption
	Graph_memoryUsage (g, result) ;

	//--------------------------------------------------------------------------
	// estimate nodes & edges attribute-set memory consumption
	//--------------------------------------------------------------------------

	// add estimated nodes attribute set size
	_EstimateNodeAttributeMemory (gc, g, samples, result) ;

	// add estimated edges attribute set size
	_EstimateEdgeAttributeMemory (gc, g, samples, result) ;

	//--------------------------------------------------------------------------
	// collect indices memory usage
	//--------------------------------------------------------------------------

	int n_node_schema = GraphContext_SchemaCount (gc, SCHEMA_NODE) ;
	for (int i = 0 ; i < n_node_schema ; i++) {
		Schema *s = GraphContext_GetSchemaByID (gc, i, SCHEMA_NODE) ;

		if (!Schema_HasIndices (s)) {
			continue ;
		}

		Index active_idx  = ACTIVE_IDX (s) ;
		Index pending_idx = PENDING_IDX (s) ;

		if (active_idx  != NULL) result->indices_sz += Index_MemoryUsage (active_idx) ;
		if (pending_idx != NULL) result->indices_sz += Index_MemoryUsage (pending_idx) ;
	}

	int n_edge_schema = GraphContext_SchemaCount (gc, SCHEMA_EDGE) ;
	for (int i = 0 ; i < n_edge_schema ; i++) {
		Schema *s = GraphContext_GetSchemaByID (gc, i, SCHEMA_EDGE) ;

		if (!Schema_HasIndices (s)) {
			continue ;
		}

		Index active_idx  = ACTIVE_IDX (s) ;
		Index pending_idx = PENDING_IDX (s) ;

		if (active_idx  != NULL) result->indices_sz += Index_MemoryUsage (active_idx) ;
		if (pending_idx != NULL) result->indices_sz += Index_MemoryUsage (pending_idx) ;
	}

	//--------------------------------------------------------------------------
	// compute the total graph memory usage
	//--------------------------------------------------------------------------

	// sum up node attributes
	for (int i = 0 ; i < arr_len (result->node_attr_by_label_sz) ; i++) {
		result->total_graph_sz_mb += result->node_attr_by_label_sz [i] ;
		result->node_attr_by_label_sz [i] /= MB ;
	}

	// sum up edge attributes
	for (int i = 0 ; i < arr_len (result->edge_attr_by_type_sz) ; i++) {
		result->total_graph_sz_mb += result->edge_attr_by_type_sz [i] ;
		result->edge_attr_by_type_sz [i] /= MB ;
	}

	// add up the rest of the components
	result->total_graph_sz_mb +=
			result->indices_sz             +
			result->lbl_matrices_sz        +
			result->rel_matrices_sz        +
			result->node_block_storage_sz  +
			result->edge_block_storage_sz  +
			result->unlabeled_node_attr_sz ;

	// convert from bytes to mb
	result->indices_sz             /= MB ;
	result->lbl_matrices_sz        /= MB ;
	result->rel_matrices_sz        /= MB ;
	result->node_block_storage_sz  /= MB ;
	result->edge_block_storage_sz  /= MB ;
	result->unlabeled_node_attr_sz /= MB ;
	result->total_graph_sz_mb      /= MB ;
}

// GRAPH.MEMORY USAGE internal command handler
// the function is executed on a reader thread to avoid blocking the main thread
static void _Graph_Memory
(
	void *_ctx  // command context
) {
	ASSERT (_ctx != NULL) ;

	GraphMemoryCtx *ctx = (GraphMemoryCtx*)_ctx ;

	GraphContext             *gc     = ctx->gc ;
	int64_t                  samples = ctx->samples ;
	RedisModuleBlockedClient *bc     = ctx->bc ;

	//--------------------------------------------------------------------------
	// compute graph memory usage
	//--------------------------------------------------------------------------

	MemoryUsageResult result = {0} ;
	result.edge_attr_by_type_sz  = arr_new (size_t, 0) ;
	result.node_attr_by_label_sz = arr_new (size_t, 0) ;

	// acquire read lock
	GraphContext_AcquireReadLock (gc) ;

	_estimate_memory_consumption (gc, samples, &result) ;

	// release read lock
	GraphContext_ReleaseReadLock (gc) ;

	// counter to GraphContext_Retrieve
	GraphContext_DecreaseRefCount (gc) ;

	//--------------------------------------------------------------------------
	// reply to caller
	//--------------------------------------------------------------------------

	// reply structure:
	// {
	//    total_graph_sz_mb: <total_graph_sz_mb>
	//
	//    label_matrices_sz_mb: <label_matrices_sz_mb>
	//
	//    relation_matrices_sz_mb: <relation_matrices_sz_mb>
	//
	//    amortized_node_sz_mb: <node_sz_mb>
	//
	//    amortized_node_attributes_by_label_sz_mb: {
	//        <label_name>: <node_sz_mb>
	//        ...
	//    }
	//
	//    amortized_unlabeled_nodes_sz_mb: <unlabeled_nodes_sz_mb>
	//
	//    amortized_edge_sz_mb: <edge_sz_mb>
	//
	//    amortized_edge_attributes_by_type_sz_mb: {
	//        <relation_name>: <edge_sz_mb>
	//        ...
	//    }
	//
	//    indices_sz_mb: <indices_sz_mb>
	// }

	RedisModuleCtx *rm_ctx = RedisModule_GetThreadSafeContext (bc) ;

	RedisModule_ReplyWithMap (rm_ctx, 9) ;

	// total_graph_sz_mb
	RedisModule_ReplyWithCString  (rm_ctx, "total_graph_sz_mb") ;
	RedisModule_ReplyWithLongLong (rm_ctx, result.total_graph_sz_mb) ;

	// label_matrices_sz_mb
	RedisModule_ReplyWithCString  (rm_ctx, "label_matrices_sz_mb") ;
	RedisModule_ReplyWithLongLong (rm_ctx, result.lbl_matrices_sz) ;

	// relation_matrices_sz_mb
	RedisModule_ReplyWithCString  (rm_ctx, "relation_matrices_sz_mb") ;
	RedisModule_ReplyWithLongLong (rm_ctx, result.rel_matrices_sz) ;

	// amortized_node_sz_mb
	RedisModule_ReplyWithCString  (rm_ctx, "amortized_node_block_sz_mb") ;
	RedisModule_ReplyWithLongLong (rm_ctx, result.node_block_storage_sz) ;

	// amortized_node_by_label_sz_mb
	RedisModule_ReplyWithCString (rm_ctx, "amortized_node_attributes_by_label_sz_mb") ;
	RedisModule_ReplyWithMap     (rm_ctx, arr_len(result.node_attr_by_label_sz)) ;

	for (size_t i = 0 ; i < arr_len (result.node_attr_by_label_sz) ; i++) {
		Schema *s = GraphContext_GetSchemaByID (gc, i, SCHEMA_NODE) ;
		ASSERT (s != NULL) ;
	
		RedisModule_ReplyWithCString  (rm_ctx, Schema_GetName (s)) ;
		RedisModule_ReplyWithLongLong (rm_ctx, result.node_attr_by_label_sz [i]) ;
	}

	// amortized_unlabeled_nodes_sz_mb
	RedisModule_ReplyWithCString  (rm_ctx, "amortized_unlabeled_nodes_attributes_sz_mb") ;
	RedisModule_ReplyWithLongLong (rm_ctx, result.unlabeled_node_attr_sz) ;

	// amortized_edge_sz_mb
	RedisModule_ReplyWithCString  (rm_ctx, "amortized_edge_block_sz_mb") ;
	RedisModule_ReplyWithLongLong (rm_ctx, result.edge_block_storage_sz) ;

	// amortized_edge_attributes_by_type_sz_mb
	RedisModule_ReplyWithCString (rm_ctx, "amortized_edge_attributes_by_type_sz_mb") ;
	RedisModule_ReplyWithMap (rm_ctx, arr_len (result.edge_attr_by_type_sz)) ;
	for (size_t i = 0 ; i < arr_len (result.edge_attr_by_type_sz) ; i++) {
		Schema *s = GraphContext_GetSchemaByID (gc, i, SCHEMA_EDGE) ;
		ASSERT (s != NULL) ;

		RedisModule_ReplyWithCString (rm_ctx, Schema_GetName (s)) ;
		RedisModule_ReplyWithLongLong (rm_ctx, result.edge_attr_by_type_sz [i]) ;
	}

	// indices_sz_mb
	RedisModule_ReplyWithCString  (rm_ctx, "indices_sz_mb") ;
	RedisModule_ReplyWithLongLong (rm_ctx, result.indices_sz) ;

	// unblock client
    RedisModule_UnblockClient (bc, NULL) ;

	// free redis module context
	RedisModule_FreeThreadSafeContext (rm_ctx) ;

	// free command context
	rm_free (ctx) ;
	arr_free (result.edge_attr_by_type_sz) ;
	arr_free (result.node_attr_by_label_sz) ;
}

// GRAPH.MEMORY USAGE <key> command reports the number of bytes that a graph
// require to be stored in RAM
// e.g. GRAPH.MEMORY USAGE g
// e.g. GRAPH.MEMORY USAGE g [SAMPLES count]
int Graph_Memory
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // arguments
	int argc                   // number of arguments
) {
	// expecting either 3 arguments:
	// GRAPH.MEMORY USAGE <key>
	// GRAPH.MEMORY USAGE <key> SAMPLES <count>
	if (argc != 3 && argc != 5) {
		return RedisModule_WrongArity (ctx) ;
	}

	//--------------------------------------------------------------------------
	// argv[1] should be USAGE
	//--------------------------------------------------------------------------

	RedisModuleString *_arg = argv [1] ;
	const char *arg = RedisModule_StringPtrLen (_arg, NULL) ;
	if (strcasecmp(arg, "USAGE") != 0) {
		RedisModule_ReplyWithErrorFormat (ctx,
			"ERR unknown subcommand '%s'. expecting GRAPH.MEMORY USAGE <key>",
			arg) ;
		return REDISMODULE_OK ;
	}

	//--------------------------------------------------------------------------
	// set number of samples
	//--------------------------------------------------------------------------

	unsigned long long samples = 100 ;  // default number of samples
	if (argc == 5) {
		_arg = argv [3] ;
		arg = RedisModule_StringPtrLen (_arg, NULL) ;
		if (strcasecmp (arg, "SAMPLES") != 0) {
			RedisModule_ReplyWithErrorFormat (ctx,
				"ERR unknown subcommand '%s'. expecting GRAPH.MEMORY USAGE <key> SAMPLES <x>",
				arg) ;
			return REDISMODULE_OK ;
		}

		// convert last argument to numeric
		_arg = argv [4] ;
		if (RedisModule_StringToULongLong (_arg, &samples) == REDISMODULE_ERR) {
			RedisModule_ReplyWithErrorFormat (ctx, EMSG_MUST_BE_NON_NEGATIVE,
					"SAMPLES") ;
			return REDISMODULE_OK ;
		}

		// restrict number of samples to max 10,000
		samples = MAX (1, MIN (samples, 10000)) ;
	}

	//--------------------------------------------------------------------------
	// get graph key
	//--------------------------------------------------------------------------

	GraphContext *gc = GraphContext_Retrieve (ctx, argv [2], true, false) ;
	if (gc == NULL) {
		return REDISMODULE_OK ;
	}

	// GRAPH.MEMORY might be an expensive operation to compute
	// to avoid blocking the main thread
	// delegate the computation to a dedicated thread

	// block the client
	RedisModuleBlockedClient *bc = RedisModule_BlockClient (ctx, NULL, NULL,
			NULL, 0) ;

	// create command context to pass to worker thread
	GraphMemoryCtx *cmd_ctx = rm_calloc (1, sizeof (GraphMemoryCtx)) ;
	ASSERT (cmd_ctx != NULL) ;

	cmd_ctx->gc      = gc ;
	cmd_ctx->bc      = bc ;
	cmd_ctx->samples = samples ;

	ThreadPool_AddWork (_Graph_Memory, cmd_ctx, true) ;

	return REDISMODULE_OK ;
}
