/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "cmd_memory.h"
#include "../errors/error_msgs.h"
#include "../graph/graphcontext.h"

extern RedisModuleType *GraphContextRedisModuleType;

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
	// pick #samples random graph entities and compute their memory consumption
	//--------------------------------------------------------------------------

	// node random set size = 66% of the entire sample size
	// assuming nodes have more attributes than edges
	uint64_t n_nodes          = Graph_NodeCount(g);
	int      sample_size      = MIN(n_nodes, 0.6 * samples);
	size_t   set_memory_usage = 0;

	// compute memory consumption of a random set of nodes
	for(int i = 0; i < sample_size; i++) {
		// pick a random node
		Node node;
		NodeID id = rand() % n_nodes;
		Graph_GetNode(g, id, &node);

		AttributeSet set = GraphEntity_GetAttributes((GraphEntity*)&node);
		set_memory_usage += AttributeSet_memoryUsage(set);
	}
	
	// compute average memory consumption of a node
	set_memory_usage /= MAX(1, sample_size);
	*node_storage_sz_mb += set_memory_usage * n_nodes;

	// random set of edges
	set_memory_usage = 0;
	uint64_t n_edges = Graph_EdgeCount(g);

	// edge random set size = 33% of the sample size
	sample_size = MIN(0.3 * samples, n_edges);

	for(int i = 0; i < sample_size; i++) {
		Edge edge;
		EdgeID id = rand() % n_edges;
		Graph_GetEdge(g, id, &edge);

		AttributeSet set = GraphEntity_GetAttributes((GraphEntity*)&edge);
		set_memory_usage += AttributeSet_memoryUsage(set);
	}

	// compute average memory consumption of a node
	set_memory_usage /= MAX(1, sample_size);
	*edge_storage_sz_mb += set_memory_usage * n_edges;

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
	*indices_sz_mb      /= 1000000;
	*lbl_matrices_sz_mb /= 1000000;
	*rel_matrices_sz_mb /= 1000000;
	*node_storage_sz_mb /= 1000000;
	*edge_storage_sz_mb /= 1000000;

	// return total memory consumption
	return (*lbl_matrices_sz_mb +
			*rel_matrices_sz_mb +
			*node_storage_sz_mb +
			*edge_storage_sz_mb +
			*indices_sz_mb);
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

	//--------------------------------------------------------------------------
	// compute graph memory usage
	//--------------------------------------------------------------------------

	size_t indices_sz_mb;       // indices memory usage
	size_t lbl_matrices_sz_mb;  // label matrices memory usage
	size_t rel_matrices_sz_mb;  // relation matrices memory usage
	size_t node_storage_sz_mb;  // node storage memory usage
	size_t edge_storage_sz_mb;  // edge storage memory usage

	size_t total_graph_sz_mb = _estimate_memory_consumption(gc, samples,
			&lbl_matrices_sz_mb, &rel_matrices_sz_mb, &node_storage_sz_mb,
			&edge_storage_sz_mb, &indices_sz_mb);

	// counter to GraphContext_Retrieve
	GraphContext_Release(gc);

	//--------------------------------------------------------------------------
	// reply to caller
	//--------------------------------------------------------------------------

	// reply structure:
	// [
	//    total_graph_sz_mb
	//    (integer) total_graph_sz_mb
	//
	//    label_matrices_sz_mb
	//    (integer) <label_matrices_sz_mb>

	//    relation_matrices_sz_mb
	//    (integer) <relation_matrices_sz_mb>

	//    amortized_node_storage_sz_mb
	//    (integer) <node_storage_sz_mb>

	//    amortized_edge_storage_sz_mb
	//    (integer) <edge_storage_sz_mb>
	//
	//    indices_sz_mb
	//    (integer) <indices_sz_mb>
	//
	// ]

	RedisModule_ReplyWithArray(ctx, 6 * 2);

	// total_graph_sz_mb
	RedisModule_ReplyWithCString(ctx, "total_graph_sz_mb");
	RedisModule_ReplyWithLongLong(ctx, total_graph_sz_mb);

	// label_matrices_sz_mb
	RedisModule_ReplyWithCString(ctx, "label_matrices_sz_mb");
	RedisModule_ReplyWithLongLong(ctx, lbl_matrices_sz_mb);

	// relation_matrices_sz_mb
	RedisModule_ReplyWithCString(ctx, "relation_matrices_sz_mb");
	RedisModule_ReplyWithLongLong(ctx, rel_matrices_sz_mb);

	// amortized_node_storage_sz_mb
	RedisModule_ReplyWithCString(ctx, "amortized_node_storage_sz_mb");
	RedisModule_ReplyWithLongLong(ctx, node_storage_sz_mb);

	// amortized_edge_storage_sz_mb
	RedisModule_ReplyWithCString(ctx, "amortized_edge_storage_sz_mb");
	RedisModule_ReplyWithLongLong(ctx, edge_storage_sz_mb);

	// indices_sz_mb
	RedisModule_ReplyWithCString(ctx, "indices_sz_mb");
	RedisModule_ReplyWithLongLong(ctx, indices_sz_mb);

	return REDISMODULE_OK;
}

