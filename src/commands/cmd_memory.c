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

static size_t _SampleVector
(
	const Graph *g,
	const GrB_Vector V,
	GxB_Iterator it,
	int64_t samples  // #samples per label to collect
) {
	GrB_Info  info;
	GrB_Index nvals;

	info = GrB_Vector_nvals(&nvals, V);
	ASSERT(info == GrB_SUCCESS);

	// in case current vector is empty, continue to next label
	if(nvals == 0) return 0;

	size_t  memory_usage  = 0;
	int64_t v_sample_size = MIN(nvals, samples);  // sample size
	int64_t sample_size   = v_sample_size;        // copy v_sample_size

	// iterate over V
	info = GxB_Vector_Iterator_attach(it, V, NULL);
	ASSERT(info == GrB_SUCCESS);

	// seek to the first entry
	info = GxB_Vector_Iterator_seek(it, 0);
	while(info != GxB_EXHAUSTED && v_sample_size > 0) {
		// get the entry V(i)
		GrB_Index i = GxB_Vector_Iterator_getIndex(it);

		Node n;
		bool node_found = Graph_GetNode(g, i, &n);
		if(likely(node_found == true)) {
			AttributeSet set = GraphEntity_GetAttributes((GraphEntity*)&n);
			memory_usage += AttributeSet_memoryUsage(set);
		}

		v_sample_size--;

		// move to the next entry in V
		info = GxB_Vector_Iterator_next(it);
	}

	// average labeled entity memory consumption
	ASSERT((sample_size - v_sample_size) > 0);
	float avg = memory_usage / (sample_size - v_sample_size);

	return avg * nvals;
}

// estimate nodes attribute-set memory consumption
// use this method only when there's some label overlapping between nodes
// as this method is much slower than its counter-part
// _EstimateNodeAttributeMemory
static size_t _EstimateOverlapingNodeAttributeMemory
(
	const GraphContext *gc,  // graph context
	const Graph *g,          // graph
	int64_t samples          // #samples per label to collect
) {
	ASSERT(g       != NULL);
	ASSERT(gc      != NULL);
	ASSERT(samples > 0);

	size_t node_memory_usage = 0;
	int n_lbls = Graph_LabelTypeCount(g);
	Delta_Matrix D = Graph_GetNodeLabelMatrix(g);

	GrB_Info info;
	GrB_Scalar     x     = NULL;
	GrB_Vector     V     = NULL;          // current column
	GrB_Vector     P     = NULL;          // processed entries
	GxB_Iterator   it    = NULL;          // vector iterator
	GrB_Index      nrows = 0;             // number of rows in vector
	GrB_Matrix     lbls  = NULL;          // labels matrix
	GrB_Descriptor desc  = GrB_DESC_RSC;  // GraphBLAS descriptor

	info = Delta_Matrix_export(&lbls, D);
	ASSERT(info == GrB_SUCCESS);

	// switch to CSC as we'll be performing column operations
	info = GxB_Matrix_Option_set(lbls, GrB_STORAGE_ORIENTATION_HINT,
			GrB_COLMAJOR);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_nrows(&nrows, lbls);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Vector_new(&P, GrB_BOOL, nrows);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Vector_new(&V, GrB_BOOL, nrows);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_Iterator_new(&it);
	ASSERT(info == GrB_SUCCESS);

	for(int j = 0; j < n_lbls; j++) {
		// extract current column
		// V<!P> = lbls[:j]
		info = GrB_Col_extract(V, P, NULL, lbls, GrB_ALL, nrows, j, desc);
		ASSERT(info == GrB_SUCCESS);

		// sample
		node_memory_usage += _SampleVector(g, V, it, samples);

		// add V to processed entries
		// P = P + V
		info = GrB_Vector_eWiseAdd_Semiring(P, NULL, NULL, GxB_ANY_PAIR_BOOL,
				P, V, GrB_DESC_S);
		ASSERT(info == GrB_SUCCESS);
	}

	// in case there are unlabeled nodes
	GrB_Index nvals;
	info = GrB_Vector_nvals(&nvals, P);  // total number of labeled nodes
	ASSERT(info == GrB_SUCCESS);

	if(nvals < Graph_NodeCount(g)) {
		// compute memory consumption of unlabeled nodes
		// P = U (lbls[0] .. lbls[n_lbls])
		// V<!P> = 1

		info = GrB_Scalar_new(&x, GrB_BOOL);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Scalar_setElement(x, true);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Vector_assign_Scalar(V, P, NULL, x, GrB_ALL, nrows, GrB_DESC_C);
		ASSERT(info == GrB_SUCCESS);

		node_memory_usage += _SampleVector(g, V, it, samples);
	}

	// clean up
	GrB_free(&x);
	GrB_free(&V);
	GrB_free(&P);
	GrB_free(&it);
	GrB_free(&lbls);

	return node_memory_usage;
}

// estimate nodes attribute-set memory consumption
static size_t _EstimateNodeAttributeMemory
(
	const GraphContext *gc,  // graph context
	const Graph *g,          // graph
	int64_t samples          // #samples per relationship type to collect
) {
	// compute average node attribute memory consumption
	int     n_lbls            = Graph_LabelTypeCount(g);
	int64_t n_nodes           = Graph_NodeCount(g);     // number of nodes
	int64_t sample_size       = MIN(n_nodes, samples);  // sample size
	int64_t nodes_sample_size = sample_size;            // nodes sample size
	int64_t n_labeled_nodes   = 0;                      // #labeled nodes
	size_t  node_memory_usage = 0;                      // node memory usage

	// determine if graph contains any overlaping nodes
	// e.g. CREATE (n:A:B)
	// in which case SUM(#lbl_nodes) > #graph nodes

	uint64_t total_labeled_node_count = 0;
	for(LabelID i = 0; i < n_lbls; i++) {
		total_labeled_node_count += Graph_LabeledNodeCount(g, i);
	}

	bool overlapping = total_labeled_node_count > n_nodes;
	if(overlapping) {
		return _EstimateOverlapingNodeAttributeMemory(gc, g, samples);
	}

	for(LabelID l = 0; l < n_lbls; l++) {
		Node node;
		GrB_Index id;
		GrB_Info info;
		Delta_Matrix L;
		Delta_MatrixTupleIter it;
		size_t label_memory_usage = 0;

		// attach iterator to the current label matrix
		L = Graph_GetLabelMatrix(g, l);
		info = Delta_MatrixTupleIter_attach(&it, L);
		ASSERT(info == GrB_SUCCESS);

		info = Delta_MatrixTupleIter_iterate_range(&it, 0, UINT64_MAX);
		ASSERT(info == GrB_SUCCESS);

		// iterate over label matrix, limit #iterations to simple_size
		while(Delta_MatrixTupleIter_next_BOOL(&it, &id, NULL, NULL)
				== GrB_SUCCESS && nodes_sample_size > 0) {
			// compute the memory consumption of the current node
			Graph_GetNode(g, id, &node);
			AttributeSet set  = GraphEntity_GetAttributes((GraphEntity*)&node);

			label_memory_usage += AttributeSet_memoryUsage(set);
			nodes_sample_size--;
		}
		Delta_MatrixTupleIter_detach(&it);

		// set number of sampled nodes
		int64_t n_sampled_nodes = sample_size - nodes_sample_size;

		// compute weighted average
		node_memory_usage += (label_memory_usage / n_sampled_nodes)
			* Graph_LabeledNodeCount(g, l);

		// sum number of labeled nodes
		n_labeled_nodes += Graph_LabeledNodeCount(g, l);

		// reset sample size
		nodes_sample_size = sample_size;
	}

	if(n_nodes > n_labeled_nodes) {
		// number of unlabeled nodes in the graph
		int64_t n_unlabeled_nodes    = n_nodes - n_labeled_nodes;
		size_t  unlabel_memory_usage = 0;

		// in case there are unlabeled nodes
		// compute memory consumption of a random set of nodes
		for(int i = 0; i < nodes_sample_size; i++) {
			// pick a random node
			Node node;
			NodeID id = rand() % n_nodes;
			Graph_GetNode(g, id, &node);

			// compute the memory consumption of the current node
			AttributeSet set  = GraphEntity_GetAttributes((GraphEntity*)&node);
			unlabel_memory_usage += AttributeSet_memoryUsage(set);
		}

		// compute weighted average
		node_memory_usage += (unlabel_memory_usage / nodes_sample_size)
			* n_unlabeled_nodes;
	}

	// compute average node attribute-set memory consumption
	n_nodes = (n_nodes == 0) ? 1 : n_nodes;

	// return estimated nodes attribute set size
	return node_memory_usage;
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
	ASSERT(ctx != NULL);

	cmd_ctx->gc      = gc;
	cmd_ctx->bc      = bc;
	cmd_ctx->samples = samples;

	ThreadPools_AddWorkReader(_Graph_Memory, cmd_ctx, true);

	return REDISMODULE_OK;
}

