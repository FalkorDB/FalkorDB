/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "cmd_memory.h"
#include "../errors/error_msgs.h"
#include "../util/thpool/pools.h"
#include "../graph/graphcontext.h"

#define MB (1 <<20)

extern RedisModuleType *GraphContextRedisModuleType;

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
	ASSERT(lbls != NULL);
	ASSERT(V != NULL && *V == NULL);

	GrB_Info info;
	GrB_Index ncols;

	// create V
	info = GrB_Matrix_ncols(&ncols, lbls);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Vector_new(V, GrB_UINT8, ncols);
	ASSERT(info == GrB_SUCCESS);

	//----------------------------------------------------------------------
    // Reduce each row of the label matrix to get label count per node
    // V[i] = sum(lbls(i,:))
    //----------------------------------------------------------------------

	info = GrB_Matrix_reduce_Monoid(*V, NULL, NULL, GrB_PLUS_MONOID_UINT8, lbls,
			NULL);
	ASSERT(info == GrB_SUCCESS);

	//----------------------------------------------------------------------
    // Reduce vector V to scalar sum: total number of label assignments
    //----------------------------------------------------------------------

	GrB_Scalar s;
	info = GrB_Scalar_new(&s, GrB_UINT64);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Vector_reduce_Monoid_Scalar(s, NULL, GrB_PLUS_MONOID_UINT64, *V,
			NULL);
	ASSERT(info == GrB_SUCCESS);

	uint64_t total_labels = 0;
	info = GrB_Scalar_extractElement(&total_labels, s);
	ASSERT(info == GrB_SUCCESS || info == GrB_NO_VALUE);

	info = GrB_free(&s);
	ASSERT(info == GrB_SUCCESS);

	//----------------------------------------------------------------------
    // If total label assignments > number of non-zero entries in V,
    // at least one node has more than one label
    //----------------------------------------------------------------------

	GrB_Index nvals;
	info = GrB_Vector_nvals(&nvals, *V);
	ASSERT(info == GrB_SUCCESS);

	return (total_labels > nvals);
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
	GrB_Info  info;
	GrB_Index nvals;

	info = GrB_Vector_nvals(&nvals, V);
	ASSERT(info == GrB_SUCCESS);

	// if the vector is empty, nothing to sample
	if(nvals == 0) return 0;

	size_t  memory_usage      = 0;
	int64_t remaining_samples = MIN(nvals, samples);
	int64_t attempted_samples = remaining_samples;

	// attach iterator to vector V
	info = GxB_Vector_Iterator_attach(it, V, NULL);
	ASSERT(info == GrB_SUCCESS);

	// seek to the first entry
	info = GxB_Vector_Iterator_seek(it, 0);
	while(info != GxB_EXHAUSTED && remaining_samples > 0) {
		// get the entry V(i)
		GrB_Index i = GxB_Vector_Iterator_getIndex(it);

		Node n;
		bool node_found = Graph_GetNode(g, i, &n);
		if(likely(node_found == true)) {
			AttributeSet set = GraphEntity_GetAttributes((GraphEntity*)&n);
			memory_usage += AttributeSet_memoryUsage(set);
		}

		remaining_samples--;

		// advance iterator
		info = GxB_Vector_Iterator_next(it);
	}

	// ensure at least one sample was successfully collected
	ASSERT((attempted_samples - remaining_samples) > 0);

	// estimate total memory usage by scaling the average sample
	float avg = memory_usage / (float)(attempted_samples - remaining_samples);

	return (size_t)(avg * nvals);
}

// estimates memory consumption of unlabeled nodes in the graph
// this function identifies nodes not assigned any label and samples them
static size_t _UnlabeledNodesMemory
(
	const Graph *g,      // graph
    GrB_Vector V,        // vector where V[i] != 0 marks labeled nodes
                         // will be updated to contain unlabeled nodes
    int64_t samples      // number of nodes to sample
) {
	ASSERT(g != NULL);
	ASSERT(V != NULL);
	ASSERT(samples > 0);

    GrB_Info info;
    GrB_Scalar x;

	// Create a scalar 'true' to assign to unlabeled entries
	info = GrB_Scalar_new(&x, GrB_BOOL);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Scalar_setElement(x, true);
	ASSERT(info == GrB_SUCCESS);

	// fill in complement of V (i.e., unlabeled nodes)
	GrB_Index len;
	info = GrB_Vector_size(&len, V);
	ASSERT(info == GrB_SUCCESS);

	// V<!V> = true  --> Mark unlabeled nodes
	info = GrB_Vector_assign_Scalar(V, V, NULL, x, GrB_ALL, len, GrB_DESC_RC);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_free(&x);
	ASSERT(info == GrB_SUCCESS);

	// sample memory usage for unlabeled nodes
	GxB_Iterator it;
	info = GxB_Iterator_new(&it);
	ASSERT(info == GrB_SUCCESS);

	size_t memory_usage = _SampleVector(g, V, it, samples);

	// cleanup
	GrB_free(&it);

	return memory_usage;
}

// estimates total memory usage for nodes with overlapping labels by sampling
// this method is slower and should only be used when nodes may share labels
// for faster estimation
// use _EstimateNodeAttributeMemory when labels don't overlap.
static size_t _EstimateOverlapingNodeAttributeMemory
(
	const Graph *g,   // graph
	GrB_Matrix lbls,  // labels matrix
	int64_t samples   // max samples per label
) {
	ASSERT(g != NULL);
	ASSERT(samples > 0);

	size_t node_memory_usage = 0;
	int n_lbls = Graph_LabelTypeCount(g);

	GrB_Info info;
	GrB_Scalar     x     = NULL;
	GrB_Vector     V     = NULL;          // vector for current label column
	GrB_Vector     P     = NULL;          // tracks processed nodes
	GxB_Iterator   it    = NULL;          // iterator for sampling
	GrB_Index      nrows = 0;             // number of nodes
	GrB_Descriptor desc  = GrB_DESC_RSC;  // descriptor for masked extraction

	// set column-major layout for efficient column extraction
	info = GxB_Matrix_Option_set(lbls, GrB_STORAGE_ORIENTATION_HINT,
			GrB_COLMAJOR);
	ASSERT(info == GrB_SUCCESS);

	// get the number of rows (nodes)
	info = GrB_Matrix_nrows(&nrows, lbls);
	ASSERT(info == GrB_SUCCESS);

	// create a vector to mark processed nodes
	info = GrB_Vector_new(&P, GrB_BOOL, nrows);
	ASSERT(info == GrB_SUCCESS);

	// create a reusable vector for label columns
	info = GrB_Vector_new(&V, GrB_BOOL, nrows);
	ASSERT(info == GrB_SUCCESS);

	// create a reusable iterator
	info = GxB_Iterator_new(&it);
	ASSERT(info == GrB_SUCCESS);

	// iterate over each label
	for(int j = 0; j < n_lbls; j++) {
		// extract column j (label j), skipping already processed entries
        // V<!P> = lbls[:, j]
		info = GrB_Col_extract(V, P, NULL, lbls, GrB_ALL, nrows, j, desc);
		ASSERT(info == GrB_SUCCESS);

		// Sample attribute memory usage from unprocessed nodes with this label
		node_memory_usage += _SampleVector(g, V, it, samples);

		// mark these nodes as processed: P = P + V
		info = GrB_Vector_eWiseAdd_Semiring(P, NULL, NULL, GxB_ANY_PAIR_BOOL,
				P, V, GrB_DESC_S);
		ASSERT(info == GrB_SUCCESS);
	}

	// clean up
	GrB_free(&x);
	GrB_free(&V);
	GrB_free(&P);
	GrB_free(&it);

	return node_memory_usage;
}

// estimate total memory usage for all labeled nodes,
// assuming there is no label overlap between nodes
// for overlapping labels,
// use the more expensive _EstimateOverlapingNodeAttributeMemory
static size_t _EstimateNonOverlapingNodeAttributeMemory
(
    const Graph *g,     // graph
    int64_t sample_size // number of nodes to sample per label
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
		info = Delta_MatrixTupleIter_attach(&it, L);
		ASSERT(info == GrB_SUCCESS);

		info = Delta_MatrixTupleIter_iterate_range(&it, 0, UINT64_MAX);
		ASSERT(info == GrB_SUCCESS);

		// sample up to `sample_size` nodes with this label
		while(Delta_MatrixTupleIter_next_BOOL(&it, &id, NULL, NULL)
				== GrB_SUCCESS && nodes_remaining > 0) {
			// compute the memory consumption of the current node
			bool found = Graph_GetNode(g, id, &node);
			ASSERT(found == true);

			AttributeSet set = GraphEntity_GetAttributes((GraphEntity*)&node);

			label_memory_usage += AttributeSet_memoryUsage(set);
			nodes_remaining--;
		}

		Delta_MatrixTupleIter_detach(&it);

		// set number of sampled nodes
		int64_t sampled = sample_size - nodes_remaining;

		if(sampled > 0) {
			// Compute average and scale by number of labeled nodes
            float avg_label_mem = (float)label_memory_usage / sampled;
            int64_t total_labeled_nodes = Graph_LabeledNodeCount(g, l);
            total_memory_usage += avg_label_mem * total_labeled_nodes;
        }
	}

	return total_memory_usage;
}

// estimate total memory consumption of node attribute sets
// the method adapts based on node label characteristics:
// - if there are overlapping labels, a more expensive estimation is used
// - if there are unlabeled nodes, they are sampled separately
static size_t _EstimateNodeAttributeMemory
(
    const GraphContext *gc,  // graph context
    const Graph *g,          // graph
    int64_t samples          // number of nodes to sample
) {
	ASSERT(g       != NULL);
	ASSERT(gc      != NULL);
    ASSERT(samples > 0);

	GrB_Info info;
	GrB_Vector V    = NULL;
	GrB_Matrix lbls = NULL;

	size_t node_memory_usage = 0;                        // node memory usage
	int64_t node_count       = Graph_NodeCount(g);       // number of nodes
	int64_t sample_size      = MIN(node_count, samples); // sample size

	//--------------------------------------------------------------------------
	// determine if the graph has overlapping labels
	//--------------------------------------------------------------------------

	Delta_Matrix D = Graph_GetNodeLabelMatrix(g);
	info = Delta_Matrix_export(&lbls, D);
	ASSERT(info == GrB_SUCCESS);

	bool overlapping = _Overlapping(lbls, &V);

	//--------------------------------------------------------------------------
	// check for unlabeled nodes
	//--------------------------------------------------------------------------

	GrB_Index nvals;
	info = GrB_Vector_nvals(&nvals, V);
	ASSERT(info == GrB_SUCCESS);

	bool has_unlabeled_nodes = Graph_NodeCount(g) > nvals;  // unlabeled nodes
	if(has_unlabeled_nodes) {
		// resize vector to match actual number of nodes in the graph
		info = GrB_Vector_resize(V, Graph_UncompactedNodeCount(g));
		ASSERT(info == GrB_SUCCESS);
		node_memory_usage += _UnlabeledNodesMemory(g, V, samples);
	}

	info = GrB_free(&V);
	ASSERT(info == GrB_SUCCESS);

	if(overlapping) {
		node_memory_usage +=
			_EstimateOverlapingNodeAttributeMemory(g, lbls, sample_size);
	} else {
		node_memory_usage +=
			_EstimateNonOverlapingNodeAttributeMemory(g, sample_size);
	}

	GrB_free(&lbls);

	// return estimated nodes attribute set size
	return node_memory_usage;
}

// estimate edges attribute-set memory consumption
static size_t _EstimateEdgeAttributeMemory
(
	const GraphContext *gc,  // graph context
	const Graph *g,          // graph
	uint samples             // #samples per relationship type to collect
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
		Delta_Matrix R;
		Delta_MatrixTupleIter it;
		size_t relation_memory_usage = 0;

		// attach iterator to the current relation matrix
		R = Graph_GetRelationMatrix(g, r, false);

		info = Delta_MatrixTupleIter_attach(&it, R);
		ASSERT(info == GrB_SUCCESS);

		info = Delta_MatrixTupleIter_iterate_range(&it, 0, UINT64_MAX);
		ASSERT(info == GrB_SUCCESS);

		// iterate over relation matrix, limit #iterations to simple_size
		while(Delta_MatrixTupleIter_next_BOOL(&it, &id, NULL, NULL)
				== GrB_SUCCESS && edges_sample_size > 0) {
			// compute the memory consumption of the current edge
			Graph_GetEdge(g, id, &edge);
			AttributeSet set = GraphEntity_GetAttributes((GraphEntity*)&edge);

			relation_memory_usage += AttributeSet_memoryUsage(set);
			edges_sample_size--;
		}
		Delta_MatrixTupleIter_detach(&it);

		// set number of sampled edges
		int64_t n_sampled_edges = sample_size - edges_sample_size;

		// compute weighted average
		edge_memory_usage += (relation_memory_usage / n_sampled_edges)
			* Graph_RelationEdgeCount(g, r);

		// reset sample size
		edges_sample_size = sample_size;
	}

	// return estimated edge attribute set size
	return edge_memory_usage;
}

// returns the total amount of memory consumed by a graph
static size_t _estimate_memory_consumption
(
	const GraphContext *gc,      // graph context
	double samples,              // random set size
	size_t *lbl_matrices_sz_mb,  // [output] label matrices memory usage
	size_t *rel_matrices_sz_mb,  // [output] relation matrices memory usage
	size_t *node_storage_sz_mb,  // [output] node storage memory usage
	size_t *edge_storage_sz_mb,  // [output] edge storage memory usage
	size_t *indices_sz_mb        // [output] indices memory usage
) {
	ASSERT(gc                 != NULL);
	ASSERT(samples            >= 0);
	ASSERT(indices_sz_mb      != NULL);
	ASSERT(lbl_matrices_sz_mb != NULL);
	ASSERT(rel_matrices_sz_mb != NULL);
	ASSERT(node_storage_sz_mb != NULL);
	ASSERT(edge_storage_sz_mb != NULL);

	// zero outputs
	*indices_sz_mb      = 0;
	*lbl_matrices_sz_mb = 0;
	*rel_matrices_sz_mb = 0;
	*node_storage_sz_mb = 0;
	*edge_storage_sz_mb = 0;

	// a graph memory consumption is spread across multiple components:
	// 1. matrices
	// 2. datablocks
	// 3. attributes
	// 4. indices

	const Graph *g = GraphContext_GetGraph(gc);

	// collect graph's memory consumption
	Graph_memoryUsage(g, lbl_matrices_sz_mb, rel_matrices_sz_mb,
			node_storage_sz_mb, edge_storage_sz_mb);

	//--------------------------------------------------------------------------
	// estimate nodes & edges attribute-set memory consumption
	//--------------------------------------------------------------------------

	// add estimated nodes attribute set size
	*node_storage_sz_mb += _EstimateNodeAttributeMemory(gc, g, samples);

	// add estimated edges attribute set size
	*edge_storage_sz_mb += _EstimateEdgeAttributeMemory(gc, g, samples);

	//--------------------------------------------------------------------------
	// collect indices memory usage
	//--------------------------------------------------------------------------

	int n_node_schema = GraphContext_SchemaCount(gc, SCHEMA_NODE);
	for(int i = 0; i < n_node_schema; i++) {
		Schema *s = GraphContext_GetSchemaByID(gc, i, SCHEMA_NODE);

		if(!Schema_HasIndices(s)) {
			continue;
		}

		Index   idx = ACTIVE_IDX(s) ? ACTIVE_IDX(s) : PENDING_IDX(s);
		RSIndex *sp = Index_RSIndex(idx);
		*indices_sz_mb += RediSearch_MemUsage(sp);
	}

	int n_edge_schema = GraphContext_SchemaCount(gc, SCHEMA_EDGE);
	for(int i = 0; i < n_edge_schema; i++) {
		Schema *s = GraphContext_GetSchemaByID(gc, i, SCHEMA_EDGE);

		if(!Schema_HasIndices(s)) {
			continue;
		}

		Index   idx = ACTIVE_IDX(s) ? ACTIVE_IDX(s) : PENDING_IDX(s);
		RSIndex *sp = Index_RSIndex(idx);
		*indices_sz_mb += RediSearch_MemUsage(sp);
	}

	// convert from bytes to mb
	*indices_sz_mb      /= MB;
	*lbl_matrices_sz_mb /= MB;
	*rel_matrices_sz_mb /= MB;
	*node_storage_sz_mb /= MB;
	*edge_storage_sz_mb /= MB;

	// return total memory consumption
	return (*lbl_matrices_sz_mb +
			*rel_matrices_sz_mb +
			*node_storage_sz_mb +
			*edge_storage_sz_mb +
			*indices_sz_mb);
}

// GRAPH.MEMORY USAGE internal command handler
// the function is executed on a reader thread to avoid blocking the main thread
static void _Graph_Memory
(
	void *_ctx  // command context
) {
	ASSERT(_ctx != NULL);

	GraphMemoryCtx *ctx = (GraphMemoryCtx*)_ctx;

	GraphContext             *gc     = ctx->gc;
	int64_t                  samples = ctx->samples;
	RedisModuleBlockedClient *bc     = ctx->bc;

	//--------------------------------------------------------------------------
	// compute graph memory usage
	//--------------------------------------------------------------------------

	size_t indices_sz_mb;       // indices memory usage
	size_t lbl_matrices_sz_mb;  // label matrices memory usage
	size_t rel_matrices_sz_mb;  // relation matrices memory usage
	size_t node_storage_sz_mb;  // node storage memory usage
	size_t edge_storage_sz_mb;  // edge storage memory usage

	// acquire read lock
	Graph_AcquireReadLock(gc->g);

	size_t total_graph_sz_mb = _estimate_memory_consumption(gc, samples,
			&lbl_matrices_sz_mb, &rel_matrices_sz_mb, &node_storage_sz_mb,
			&edge_storage_sz_mb, &indices_sz_mb);

	// release read lock
	Graph_ReleaseLock(gc->g);

	// counter to GraphContext_Retrieve
	GraphContext_Release(gc);

	//--------------------------------------------------------------------------
	// reply to caller
	//--------------------------------------------------------------------------

	// reply structure:
	// {
	//    total_graph_sz_mb: <total_graph_sz_mb>
	//    label_matrices_sz_mb: <label_matrices_sz_mb>
	//    relation_matrices_sz_mb: <relation_matrices_sz_mb>
	//    amortized_node_storage_sz_mb: <node_storage_sz_mb>
	//    amortized_edge_storage_sz_mb: <edge_storage_sz_mb>
	//    indices_sz_mb: <indices_sz_mb>
	// }

	RedisModuleCtx *rm_ctx = RedisModule_GetThreadSafeContext(bc);

	// six key value pairs
	RedisModule_ReplyWithMap(rm_ctx, 6);

	// total_graph_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "total_graph_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, total_graph_sz_mb);

	// label_matrices_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "label_matrices_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, lbl_matrices_sz_mb);

	// relation_matrices_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "relation_matrices_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, rel_matrices_sz_mb);

	// amortized_node_storage_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "amortized_node_storage_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, node_storage_sz_mb);

	// amortized_edge_storage_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "amortized_edge_storage_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, edge_storage_sz_mb);

	// indices_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "indices_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, indices_sz_mb);

	// unblock client
    RedisModule_UnblockClient(bc, NULL);

	// free redis module context
	RedisModule_FreeThreadSafeContext(rm_ctx);

	// free command context
	rm_free(ctx);
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
	// GRAPH.MEMORY USAGE <key> SAMPLE <count>
	if(argc != 3 && argc != 5) {
		return RedisModule_WrongArity(ctx);
	}

	//--------------------------------------------------------------------------
	// argv[1] should be USAGE
	//--------------------------------------------------------------------------

	RedisModuleString *_arg = argv[1];
	const char *arg = RedisModule_StringPtrLen(_arg, NULL);
	if(strcasecmp(arg, "USAGE") != 0) {
		RedisModule_ReplyWithErrorFormat(ctx,
			"ERR unknown subcommand '%s'. expecting GRAPH.MEMORY USAGE <key>",
			arg);
		return REDISMODULE_OK;
	}

	//--------------------------------------------------------------------------
	// set number of samples
	//--------------------------------------------------------------------------

	double samples = 100;  // default number of samples
	if(argc == 5) {
		_arg = argv[3];
		arg = RedisModule_StringPtrLen(_arg, NULL);
		if(strcasecmp(arg, "SAMPLES") != 0) {
			RedisModule_ReplyWithErrorFormat(ctx,
				"ERR unknown subcommand '%s'. expecting GRAPH.MEMORY USAGE <key> SAMPLES <x>",
				arg);
			return REDISMODULE_OK;
		}

		// convert last argument to numeric
		_arg = argv[4];
		if(RedisModule_StringToDouble(_arg, &samples) == REDISMODULE_ERR) {
			RedisModule_ReplyWithErrorFormat(ctx, EMSG_MUST_BE_NON_NEGATIVE,
					"SAMPLES");
			return REDISMODULE_OK;
		}

		if(samples < 0) {
			RedisModule_ReplyWithErrorFormat(ctx, EMSG_MUST_BE_NON_NEGATIVE,
					"SAMPLES");
			return REDISMODULE_OK;
		}

		// restrict number of samples to max 10,000
		samples = MIN(samples, 10000);
	}

	//--------------------------------------------------------------------------
	// get graph key
	//--------------------------------------------------------------------------

	GraphContext *gc = GraphContext_Retrieve(ctx, argv[2], true, false);
	if(gc == NULL) {
		return REDISMODULE_OK;
	}

	// GRAPH.MEMORY might be an expensive operation to compute
	// to avoid blocking the main thread
	// delegate the computation to a dedicated thread

	// block the client
	RedisModuleBlockedClient *bc = RedisModule_BlockClient(ctx, NULL, NULL,
			NULL, 0);

	// create command context to pass to worker thread
	GraphMemoryCtx *cmd_ctx = rm_calloc(1, sizeof(GraphMemoryCtx));
	ASSERT(cmd_ctx != NULL);

	cmd_ctx->gc      = gc;
	cmd_ctx->bc      = bc;
	cmd_ctx->samples = samples;

	ThreadPools_AddWorkReader(_Graph_Memory, cmd_ctx, true);

	return REDISMODULE_OK;
}

