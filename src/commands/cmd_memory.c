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

// collects label assignment statistics by streaming label matrices
static void _CollectLabeledNodes
(
	const Graph *g,                  // graph
	bool *labeled_nodes,             // [output] node ID -> labeled
	size_t node_cap,                 // number of addressable node IDs
	uint64_t *labeled_node_count,    // [output] distinct labeled nodes
	uint64_t *label_assignment_count // [output] node-label assignments
) {
	ASSERT(g                      != NULL);
	ASSERT(labeled_nodes          != NULL);
	ASSERT(labeled_node_count     != NULL);
	ASSERT(label_assignment_count != NULL);

	int n_lbls = Graph_LabelTypeCount(g);
	for(LabelID l = 0; l < n_lbls; l++) {
		GrB_Info info;
		GrB_Index id;
		Delta_MatrixTupleIter it;
		Delta_Matrix L = Graph_GetLabelMatrix(g, l);

		// attach iterator to label matrix
		info = Delta_MatrixTupleIter_attach(&it, L);
		ASSERT(info == GrB_SUCCESS);

		info = Delta_MatrixTupleIter_iterate_range(&it, 0, UINT64_MAX);
		ASSERT(info == GrB_SUCCESS);

		while((info = Delta_MatrixTupleIter_next_BOOL(&it, &id, NULL, NULL))
				== GrB_SUCCESS) {
			ASSERT(id < node_cap);
			(*label_assignment_count)++;

			if(!labeled_nodes[id]) {
				labeled_nodes[id] = true;
				(*labeled_node_count)++;
			}
		}
		ASSERT(info == GxB_EXHAUSTED);

		Delta_MatrixTupleIter_detach(&it);
	}
}

// estimates memory consumption of unlabeled nodes in the graph
// this function identifies nodes not assigned any label and samples them
static size_t _UnlabeledNodesMemory
(
	const Graph *g,                 // graph
	const bool *labeled_nodes,      // node ID -> labeled
	uint64_t unlabeled_node_count,  // number of unlabeled nodes
	int64_t samples                 // number of nodes to sample
) {
	ASSERT(g != NULL);
	ASSERT(samples > 0);

	// if there are no unlabeled nodes, nothing to sample
	if(unlabeled_node_count == 0) return 0;

	size_t memory_usage = 0;
	uint64_t remaining_samples = MIN(unlabeled_node_count, (uint64_t)samples);
	uint64_t attempted_samples = remaining_samples;

	DataBlockIterator *it = Graph_ScanNodes(g);
	ASSERT(it != NULL);

	uint64_t id;
	AttributeSet *set;
	while((set = DataBlockIterator_Next(it, &id)) != NULL &&
			remaining_samples > 0) {
		if(labeled_nodes != NULL && labeled_nodes[id]) continue;

		memory_usage += AttributeSet_memoryUsage(*set);
		remaining_samples--;
	}

	DataBlockIterator_Free(it);

	// ensure at least one sample was successfully collected
	ASSERT((attempted_samples - remaining_samples) > 0);

	// estimate total memory usage by scaling the average sample
	float avg = memory_usage / (float)(attempted_samples - remaining_samples);

	return (size_t)(avg * unlabeled_node_count);
}

// estimates amortized memory usage for nodes with overlapping labels
// by sampling
// this method is slower and should only be used when nodes may share labels
// for faster estimation
// use _EstimateNodeAttributeMemory when labels don't overlap
static void _EstimateOverlapingNodeAttributeMemory
(
	const Graph *g,            // graph
	int64_t samples,           // max samples per label
	MemoryUsageResult *result  // [output] memory usage
) {
	ASSERT(g != NULL);
	ASSERT(samples > 0);

	int n_lbls = Graph_LabelTypeCount(g);
	size_t node_cap = Graph_RequiredMatrixDim(g);
	bool *processed_nodes = rm_calloc(node_cap, sizeof(bool));
	ASSERT(processed_nodes != NULL);

	for(LabelID l = 0; l < n_lbls; l++) {
		GrB_Info info;
		GrB_Index id;
		Delta_MatrixTupleIter it;
		Delta_Matrix L = Graph_GetLabelMatrix(g, l);

		size_t label_memory_usage = 0;
		uint64_t label_node_count = 0;
		uint64_t sampled = 0;

		// attach iterator to label matrix
		info = Delta_MatrixTupleIter_attach(&it, L);
		ASSERT(info == GrB_SUCCESS);

		info = Delta_MatrixTupleIter_iterate_range(&it, 0, UINT64_MAX);
		ASSERT(info == GrB_SUCCESS);

		while((info = Delta_MatrixTupleIter_next_BOOL(&it, &id, NULL, NULL))
				== GrB_SUCCESS) {
			ASSERT(id < node_cap);
			if(processed_nodes[id]) continue;

			// assign each multi-label node to the first label encountered
			processed_nodes[id] = true;
			label_node_count++;

			if(sampled < (uint64_t)samples) {
				Node n;
				bool node_found = Graph_GetNode(g, id, &n);
				ASSERT(node_found == true);

				AttributeSet set = GraphEntity_GetAttributes((GraphEntity*)&n);
				label_memory_usage += AttributeSet_memoryUsage(set);
				sampled++;
			}
		}
		ASSERT(info == GxB_EXHAUSTED);

		Delta_MatrixTupleIter_detach(&it);

		if(label_node_count > 0) {
			ASSERT(sampled > 0);
			float avg_label_mem = (float)label_memory_usage / sampled;
			label_memory_usage = avg_label_mem * label_node_count;
		}

		arr_append(result->node_attr_by_label_sz, label_memory_usage);
	}

	rm_free(processed_nodes);
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
		int64_t sampled = MAX(1, sample_size - nodes_remaining);

		// compute average and scale by number of labeled nodes
		float avg_label_mem = (float)label_memory_usage / sampled;
		int64_t total_labeled_nodes = Graph_LabeledNodeCount(g, l);

		label_memory_usage = avg_label_mem * total_labeled_nodes;

		arr_append(result->node_attr_by_label_sz, label_memory_usage);
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
	ASSERT(g       != NULL);
	ASSERT(gc      != NULL);
    ASSERT(samples > 0);

	size_t  node_memory_usage = 0;                        // node memory usage
	int64_t node_count        = Graph_NodeCount(g);       // number of nodes
	int64_t sample_size       = MIN(node_count, samples); // sample size
	size_t  node_cap          = Graph_RequiredMatrixDim(g);
	int     n_lbls            = Graph_LabelTypeCount(g);
	uint64_t labeled_node_count     = 0;
	uint64_t label_assignment_count = 0;
	bool *labeled_nodes             = NULL;

	//--------------------------------------------------------------------------
	// collect label statistics
	//--------------------------------------------------------------------------

	if(n_lbls > 0 && node_cap > 0) {
		labeled_nodes = rm_calloc(node_cap, sizeof(bool));
		ASSERT(labeled_nodes != NULL);

		_CollectLabeledNodes(g, labeled_nodes, node_cap, &labeled_node_count,
				&label_assignment_count);
	}

	bool overlapping = label_assignment_count > labeled_node_count;

	//--------------------------------------------------------------------------
	// check for unlabeled nodes
	//--------------------------------------------------------------------------

	bool has_unlabeled_nodes = Graph_NodeCount(g) > labeled_node_count;
	if(has_unlabeled_nodes) {
		uint64_t unlabeled_node_count = Graph_NodeCount(g) - labeled_node_count;
		node_memory_usage = _UnlabeledNodesMemory(g, labeled_nodes,
				unlabeled_node_count, samples);
		result->unlabeled_node_attr_sz = node_memory_usage;
	}

	if(overlapping) {
		_EstimateOverlapingNodeAttributeMemory(g, sample_size, result);
	} else {
		_EstimateNonOverlapingNodeAttributeMemory(g, sample_size, result);
	}

	rm_free(labeled_nodes);
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
		edge_memory_usage = (relation_memory_usage / n_sampled_edges)
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
	ASSERT(gc      != NULL);
	ASSERT(samples >= 0);
	ASSERT(result  != NULL);

	// a graph memory consumption is spread across multiple components:
	// 1. matrices
	// 2. datablocks
	// 3. attributes
	// 4. indices

	const Graph *g = GraphContext_GetGraph(gc);

	// collect graph's memory consumption
	Graph_memoryUsage(g, result);

	//--------------------------------------------------------------------------
	// estimate nodes & edges attribute-set memory consumption
	//--------------------------------------------------------------------------

	// add estimated nodes attribute set size
	_EstimateNodeAttributeMemory(gc, g, samples, result);

	// add estimated edges attribute set size
	_EstimateEdgeAttributeMemory(gc, g, samples, result);

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
		result->indices_sz += RediSearch_MemUsage(sp);
	}

	int n_edge_schema = GraphContext_SchemaCount(gc, SCHEMA_EDGE);
	for(int i = 0; i < n_edge_schema; i++) {
		Schema *s = GraphContext_GetSchemaByID(gc, i, SCHEMA_EDGE);

		if(!Schema_HasIndices(s)) {
			continue;
		}

		Index   idx = ACTIVE_IDX(s) ? ACTIVE_IDX(s) : PENDING_IDX(s);
		RSIndex *sp = Index_RSIndex(idx);
		result->indices_sz += RediSearch_MemUsage(sp);
	}

	// convert from bytes to mb
	result->indices_sz             /= MB;
	result->lbl_matrices_sz        /= MB;
	result->rel_matrices_sz        /= MB;
	result->node_block_storage_sz  /= MB;
	result->edge_block_storage_sz  /= MB;
	result->unlabeled_node_attr_sz /= MB;

	//--------------------------------------------------------------------------
	// compute the total graph memory usage
	//--------------------------------------------------------------------------

	// sum up node attributes
	for(int i = 0; i < arr_len(result->node_attr_by_label_sz); i++) {
		result->node_attr_by_label_sz[i] /= MB;
		result->total_graph_sz_mb += result->node_attr_by_label_sz[i];
	}

	// sum up edge attributes
	for(int i = 0; i < arr_len(result->edge_attr_by_type_sz); i++) {
		result->edge_attr_by_type_sz[i] /= MB;
		result->total_graph_sz_mb += result->edge_attr_by_type_sz[i];
	}

	// add up the rest of the components
	result->total_graph_sz_mb +=
			result->indices_sz             +
			result->lbl_matrices_sz        +
			result->rel_matrices_sz        +
			result->node_block_storage_sz  +
			result->edge_block_storage_sz  +
			result->unlabeled_node_attr_sz;
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
	Graph                    *g      = GraphContext_GetGraph (gc) ;
	int64_t                  samples = ctx->samples;
	RedisModuleBlockedClient *bc     = ctx->bc;

	//--------------------------------------------------------------------------
	// compute graph memory usage
	//--------------------------------------------------------------------------

	MemoryUsageResult result = {0};
	result.edge_attr_by_type_sz  = arr_new(size_t, 0);
	result.node_attr_by_label_sz = arr_new(size_t, 0);

	// acquire read lock
	Graph_AcquireReadLock(g);

	_estimate_memory_consumption(gc, samples, &result);

	// release read lock
	Graph_ReleaseLock(g);

	// counter to GraphContext_Retrieve
	GraphContext_Release(gc);

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

	RedisModuleCtx *rm_ctx = RedisModule_GetThreadSafeContext(bc);

	// six key value pairs
	RedisModule_ReplyWithMap(rm_ctx, 9);

	// total_graph_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "total_graph_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, result.total_graph_sz_mb);

	// label_matrices_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "label_matrices_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, result.lbl_matrices_sz);

	// relation_matrices_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "relation_matrices_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, result.rel_matrices_sz);

	// amortized_node_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "amortized_node_block_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, result.node_block_storage_sz);

	// amortized_node_by_label_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "amortized_node_attributes_by_label_sz_mb");
	RedisModule_ReplyWithMap(rm_ctx, arr_len(result.node_attr_by_label_sz));

	for(size_t i = 0; i < arr_len(result.node_attr_by_label_sz); i++) {
		Schema *s = GraphContext_GetSchemaByID(gc, i, SCHEMA_NODE);
		ASSERT(s != NULL);
	
		RedisModule_ReplyWithCString(rm_ctx, Schema_GetName(s));
		RedisModule_ReplyWithLongLong(rm_ctx, result.node_attr_by_label_sz[i]);
	}

	// amortized_unlabeled_nodes_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "amortized_unlabeled_nodes_attributes_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, result.unlabeled_node_attr_sz);

	// amortized_edge_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "amortized_edge_block_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, result.edge_block_storage_sz);

	// amortized_edge_attributes_by_type_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "amortized_edge_attributes_by_type_sz_mb");
	RedisModule_ReplyWithMap(rm_ctx, arr_len(result.edge_attr_by_type_sz));
	for(size_t i = 0; i < arr_len(result.edge_attr_by_type_sz); i++) {
		Schema *s = GraphContext_GetSchemaByID(gc, i, SCHEMA_EDGE);
		ASSERT(s != NULL);

		RedisModule_ReplyWithCString(rm_ctx, Schema_GetName(s));
		RedisModule_ReplyWithLongLong(rm_ctx, result.edge_attr_by_type_sz[i]);
	}

	// indices_sz_mb
	RedisModule_ReplyWithCString(rm_ctx, "indices_sz_mb");
	RedisModule_ReplyWithLongLong(rm_ctx, result.indices_sz);

	// unblock client
    RedisModule_UnblockClient(bc, NULL);

	// free redis module context
	RedisModule_FreeThreadSafeContext(rm_ctx);

	// free command context
	rm_free(ctx);
	arr_free(result.edge_attr_by_type_sz);
	arr_free(result.node_attr_by_label_sz);
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

	unsigned long long samples = 100;  // default number of samples
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
		if(RedisModule_StringToULongLong(_arg, &samples) == REDISMODULE_ERR) {
			RedisModule_ReplyWithErrorFormat(ctx, EMSG_MUST_BE_NON_NEGATIVE,
					"SAMPLES");
			return REDISMODULE_OK;
		}

		// restrict number of samples to max 10,000
		samples = MAX(1, MIN(samples, 10000)) ;
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

	ThreadPool_AddWork(_Graph_Memory, cmd_ctx, true);

	return REDISMODULE_OK;
}
