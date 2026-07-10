/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "cmd_memory.h"
#include "../errors/error_msgs.h"
#include "../util/thpool/pool.h"
#include "../graph/graphcontext.h"
#include "../graph/graph_memoryUsage.h"

// GRAPH.MEMORY command context
typedef struct {
	int64_t samples;               // number of samples to inspect
	RedisModuleString *graph_id;   // graph name
	RedisModuleBlockedClient *bc;  // blocked client
} GraphMemoryCtx;

// GRAPH.MEMORY USAGE internal command handler
// the function is executed on a reader thread to avoid blocking the main thread
static void _Graph_Memory
(
	void *_ctx  // command context
) {
	ASSERT (_ctx != NULL) ;

	GraphMemoryCtx           *ctx    = (GraphMemoryCtx*)_ctx ;
	int64_t                  samples = ctx->samples ;
	RedisModuleBlockedClient *bc     = ctx->bc ;
	RedisModuleCtx           *rm_ctx = RedisModule_GetThreadSafeContext (bc) ;

	//--------------------------------------------------------------------------
	// compute graph memory usage
	//--------------------------------------------------------------------------

	// declare result before any goto so cleanup can safely arr_free the arrays
	// (arr_free checks for NULL, so zero-initialized pointers are safe)
	MemoryUsageResult result = {0} ;

	//--------------------------------------------------------------------------
	// get graph key
	//--------------------------------------------------------------------------

	GraphContext *gc = NULL ;
	GraphContext_Retrieve (rm_ctx, ctx->graph_id, true, false, true, &gc) ;
	if (gc == NULL) {
		// error alreay emitted by GraphContext_Retrieve
		goto cleanup ;
	}

	result.edge_attr_by_type_sz  = arr_new (size_t, 0) ;
	result.node_attr_by_label_sz = arr_new (size_t, 0) ;

	// acquire read lock
	GraphContext_AcquireReadLock (gc) ;

	GraphContext_EstimateMemoryUsage (gc, samples, &result) ;

	// release read lock
	GraphContext_ReleaseReadLock (gc) ;

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

	// counter to GraphContext_Retrieve
	// held until here so schema name lookups above are not use-after-free
	GraphContext_DecreaseRefCount (gc) ;

cleanup:
	RedisModule_FreeString (rm_ctx, ctx->graph_id) ;

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

	// GRAPH.MEMORY might be an expensive operation to compute
	// to avoid blocking the main thread
	// delegate the computation to a dedicated thread

	// create command context to pass to worker thread
	GraphMemoryCtx *cmd_ctx = rm_calloc (1, sizeof (GraphMemoryCtx)) ;
	ASSERT (cmd_ctx != NULL) ;

	// block the client
	RedisModuleBlockedClient *bc = RedisModule_BlockClient (ctx, NULL, NULL,
			NULL, 0) ;

	// retain graph name
	RedisModule_RetainString (ctx, argv [2]) ;

	cmd_ctx->bc       = bc ;
	cmd_ctx->graph_id = argv [2] ;
	cmd_ctx->samples  = samples ;

	ThreadPool_AddWork (_Graph_Memory, cmd_ctx, true) ;

	return REDISMODULE_OK ;
}
