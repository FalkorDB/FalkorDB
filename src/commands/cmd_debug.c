/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../redismodule.h"
#include "../util/rmalloc.h"
#include "../util/thpool/pool.h"
#include "../module_event_handlers.h"
#include "../graph/graph.h"
#include "../graph/graphcontext.h"

#include <string.h>
#include <sys/mman.h>

void ModuleEventHandler_AUXAfterKeyspaceEvent(void);
void ModuleEventHandler_AUXBeforeKeyspaceEvent(void);

extern uint aux_field_counter;

static void Debug_AUX
(
	RedisModuleString **argv,
	int argc
) {
	if(argc < 2) return;

	const char *arg = RedisModule_StringPtrLen(argv[1], NULL);

	if(strcmp(arg, "START") == 0) {
		ModuleEventHandler_AUXBeforeKeyspaceEvent();
	} else if(strcmp(arg, "END") == 0) {
		ModuleEventHandler_AUXAfterKeyspaceEvent();
	}
}

// crash the server simulating an out-of-memory error
static void _Debug_OOM
(
	void *args  // unused
) {
	void *ptr = rm_malloc(SIZE_MAX/2); // should trigger an out of memory
	rm_free(ptr);
}

// crash the server with sigsegv
static void _Debug_SegFault
(
	void *args  // unused
) {
	// compiler gives warnings about writing to a random address
	// e.g "*((char*)-1) = 'x';"
	// as a workaround, we map a read-only area
	// and try to write there to trigger segmentation fault
	char* p = mmap(NULL, 4096, PROT_READ, MAP_PRIVATE | MAP_ANON, -1, 0);
	*p = 'x';
}

// crash by assertion failed
static void _Debug_ASSERT
(
	void *args  // unused
) {
	RedisModule_Assert(false && "DEBUG ASSERT");
}

// GRAPH.DEBUG SYNC <key>
//
// debugging aid: forces a full, synchronous flush of every delta matrix
// belonging to the named graph (adjacency, node-label, zero, per-label,
// per-relation) into their primary GraphBLAS matrices - i.e.
// Graph_ApplyAllPending with force_flush = true.
//
// this blocks: the flush runs inline on whichever thread is executing the
// command. GRAPH.DEBUG is dispatched directly (not routed through
// CommandDispatch's worker-thread offload the way GRAPH.QUERY is), so on
// the node handling a live client invocation this is the Redis main
// thread - meaning it blocks ALL client traffic on that node, for every
// graph, not just the one being synced, for as long as the flush takes.
//
// this command IS replicated (RedisModule_ReplicateVerbatim, same trick
// AUX uses above), so a single invocation against the master reaches
// every currently-attached replica (and chains further down to any of
// their own sub-replicas) without requiring the operator to reach each
// node individually. because a replicated command is dispatched on the
// replica's main thread too, to preserve ordering (see
// cmd_dispatcher.c), the same server-wide blocking applies there for the
// duration of the flush - the same tradeoff every replicated write
// already makes (e.g. GRAPH.EFFECT).
//
// the command stays "readonly"-flagged at the module level (see
// Graph_Debug's registration) so it remains directly invocable against a
// replica too, independent of the master.
static void Debug_Sync
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,  // argv[0] is the graph key
	int argc
) {
	if (argc != 1) {
		RedisModule_WrongArity (ctx) ;
		return ;
	}

	// true if this invocation arrived via the replication stream or is
	// being replayed from the AOF/RDB, rather than issued directly by a
	// live client
	bool from_replication_stream = (RedisModule_GetContextFlags (ctx) &
			(REDISMODULE_CTX_FLAGS_REPLICATED | REDISMODULE_CTX_FLAGS_LOADING)) != 0 ;

	const char *graph_name = RedisModule_StringPtrLen (argv[0], NULL) ;

	GraphContext *gc = NULL ;
	// open the key for writing (we're about to mutate the graph's matrices)
	// do not auto-create, do load from disk if the graph is offloaded
	GraphContext_Retrieve (ctx, argv[0], false, false, true, &gc) ;
	if (gc == NULL) {
		if (from_replication_stream) {
			// nothing to flush locally (graph not loaded/never existed
			// on this node), but the command must still propagate to
			// this node's own sub-replicas, if any
			RedisModule_Log (ctx, "warning",
					"GRAPH.DEBUG SYNC: graph '%s' not found on this node, "
					"nothing to flush - propagating", graph_name) ;
			RedisModule_ReplicateVerbatim (ctx) ;
		}
		// error already emitted by GraphContext_Retrieve for the
		// non-replicated (direct client) case
		return ;
	}

	RedisModule_Log (ctx, "notice",
			"GRAPH.DEBUG SYNC: forcing full delta matrix flush for graph "
			"'%s'", graph_name) ;
	mstime_t start = RedisModule_Milliseconds () ;

	GraphContext_AcquireWriteLock (gc) ;

	Graph *g = GraphContext_GetGraph (gc) ;
	Graph_ApplyAllPending (g, true) ;  // force full delta matrix sync

	GraphContext_ReleaseLock (gc) ;

	RedisModule_Log (ctx, "notice",
			"GRAPH.DEBUG SYNC: finished flushing graph '%s' in %lldms",
			graph_name, (long long) (RedisModule_Milliseconds () - start)) ;

	GraphContext_DecreaseRefCount (gc) ;

	RedisModule_ReplyWithSimpleString (ctx, "OK") ;
	RedisModule_ReplicateVerbatim (ctx) ;
}

int Graph_Debug
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT(ctx != NULL);

	if (argc < 2) {
		return RedisModule_WrongArity(ctx);
	}

	const char *sub_cmd = RedisModule_StringPtrLen(argv[1], NULL);

	if (!strcasecmp(sub_cmd, "AUX")) {
		Debug_AUX(argv + 1, argc - 1);
		RedisModule_ReplyWithLongLong(ctx, aux_field_counter);
		RedisModule_ReplicateVerbatim(ctx);
	}

	else if (!strcasecmp(sub_cmd, "OOM")) {
		// crash the server simulating an out-of-memory error
		ThreadPool_AddWork(_Debug_OOM, NULL, true);
		RedisModule_ReplyWithCString(ctx, "OK");
	}

	else if (!strcasecmp(sub_cmd, "ASSERT")) {
		// crash by assertion failed
		ThreadPool_AddWork(_Debug_ASSERT, NULL, true);
		RedisModule_ReplyWithCString(ctx, "OK");
	}

	else if (!strcasecmp(sub_cmd, "SEGFAULT")) {
		// crash the server with sigsegv
		ThreadPool_AddWork(_Debug_SegFault, NULL, true);
		RedisModule_ReplyWithCString(ctx, "OK");
	}

	else if (!strcasecmp(sub_cmd, "SYNC")) {
		// force a full delta matrix sync of the named graph, replicated
		// to every attached replica - see comment on Debug_Sync
		Debug_Sync(ctx, argv + 2, argc - 2);
	}

	else {
		RedisModule_ReplyWithError(ctx, "ERR unknown subcommand or wrong number of arguments");
	}

	return REDISMODULE_OK;
}

