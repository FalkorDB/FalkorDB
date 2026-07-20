/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graphcontext.h"
#include "graph_load_queue.h"
#include "graphcontext_retrieve.h"
#include "../redismodule.h"

extern uint aux_field_counter ;
extern pthread_t MAIN_THREAD_ID;  // redis main thread ID

// GraphContext type as registered with Redis.
extern RedisModuleType *GraphContextRedisModuleType ;

// stub type representing a graph that has been offloaded to disk
static RedisModuleType *GraphStubType = NULL ;

//------------------------------------------------------------------------------
// Enterprise - FalkorDB exported API — function pointer types
//------------------------------------------------------------------------------

typedef enum {
    GraphLoad_SUCCESS,      // graph restored from disk successfully
    GraphLoad_LOADING,      // another load for this key is already in progress
    GraphLoad_OFFLOADING,   // an offload for this key is already in progress
    GraphLoad_KEY_MISSING,  // key does not exist
    GraphLoad_NOT_STUB,     // key exists but is not an offloaded graph stub
    GraphLoad_DUMP_MISSING, // stub is valid but its dump file was not found
    GraphLoad_OOM,          // not enough memory to hold the loaded graph
    GraphLoad_ERR,          // all other failures
} GraphLoadResult ;

typedef RedisModuleType* (*GraphStubType_Get_t) (void) ;

typedef GraphLoadResult (*graph_load_t)
(
    RedisModuleCtx    *ctx,
    RedisModuleString *key_name,
    bool              from_thread,
	bool              force,
	bool              bypass_claim
) ;

//------------------------------------------------------------------------------
// Enterprise - FalkorDB exported API — function pointers
// (resolved lazily on first call)
//------------------------------------------------------------------------------

static graph_load_t        graph_load        = NULL ;
static GraphStubType_Get_t GraphStubType_Get = NULL ;

// Resolve all FalkorDB shared APIs required by this module.
// Returns true only when every required symbol is available.
static bool load_falkordb_apis
(
	RedisModuleCtx *ctx
) {
#define LOAD_API(name)                                                    \
	if (name == NULL) {                                                   \
		name = RedisModule_GetSharedAPI (ctx, #name) ;                    \
		if (name == NULL)                                                 \
		RedisModule_Log (ctx, "warning", "missing FalkorDB API: %s",      \
					#name) ;                                              \
	}

	if (graph_load != NULL && GraphStubType_Get != NULL) {
		return true ;
	}

	LOAD_API (graph_load)
	LOAD_API (GraphStubType_Get)

#undef LOAD_API

	if (GraphStubType_Get != NULL) {
		GraphStubType = GraphStubType_Get () ;
	}

	return (graph_load != NULL && GraphStubType_Get != NULL) ;
}

// creates and registers a new graph context under the given name
static GraphContext *_GraphContext_Create
(
	RedisModuleCtx *ctx,
	const char *graph_name
) {
	GraphContext *gc = GraphContext_New (graph_name) ;
	GraphContext_SetKey (ctx, gc) ;
	return gc ;
}

// attach graph context to a Redis key and register it with FalkorDB's
// global graph registry
void GraphContext_SetKey
(
	RedisModuleCtx *ctx,  // redis module context
    GraphContext *gc      // graph context
) {
	ASSERT (gc != NULL) ;

	const char *graph_name = GraphContext_GetName (gc) ;
	RedisModuleString *key_name = RedisModule_CreateString (ctx, graph_name,
			strlen (graph_name)) ;

	RedisModuleKey *key = RedisModule_OpenKey (ctx, key_name, REDISMODULE_WRITE) ;
	ASSERT (key != NULL) ;

	RedisModule_ModuleTypeSetValue (key, GraphContextRedisModuleType, gc) ;
	GraphContext_RegisterWithModule (gc) ;

	RedisModule_FreeString (ctx, key_name) ;
	RedisModule_CloseKey (key) ;
}

// emits an appropriate error reply for a graph_load failure result
// (never called with GraphLoad_SUCCESS); full diagnostic detail for the
// failure was already logged server-side by graph_load itself
static void _reply_graph_load_error
(
	RedisModuleCtx  *ctx,
	const char      *graph_name,
	GraphLoadResult result
) {
	switch (result) {
		case GraphLoad_LOADING:
			RedisModule_ReplyWithErrorFormat (ctx,
					"ERR graph: %s is currently being loaded, please retry",
					graph_name) ;
			break ;

		case GraphLoad_OFFLOADING:
			RedisModule_ReplyWithErrorFormat (ctx,
					"ERR graph: %s is currently being offloaded, please retry",
					graph_name) ;
			break ;

		case GraphLoad_KEY_MISSING:
			RedisModule_ReplyWithErrorFormat (ctx,
					"ERR graph: %s does not exist", graph_name) ;
			break ;

		case GraphLoad_NOT_STUB:
			RedisModule_ReplyWithErrorFormat (ctx,
					"ERR graph: %s is not an offloaded graph stub", graph_name) ;
			break ;

		case GraphLoad_DUMP_MISSING:
			RedisModule_ReplyWithErrorFormat (ctx,
					"ERR dump file for graph: %s not found", graph_name) ;
			break ;

		case GraphLoad_OOM:
			RedisModule_ReplyWithErrorFormat (ctx,
					"ERR not enough memory to load graph: %s", graph_name) ;
			break ;

		case GraphLoad_SUCCESS:
			ASSERT ("_reply_graph_load_error called with GraphLoad_SUCCESS" && false) ;
			break ;

		case GraphLoad_ERR:
		default:
			RedisModule_ReplyWithErrorFormat (ctx,
					"ERR failed to load graph: %s", graph_name) ;
			break ;
	}
}

// shared implementation backing GraphContext_Retrieve,
// GraphContext_RetrieveOrQueue and GraphContext_RetrieveOrForce
//
// command_ctx and bypass_claim are mutually exclusive - at most one may be
// set/true on any given call
//
// when command_ctx is NULL and bypass_claim is false: legacy behavior - on
// contention, an error is replied and GraphRetrieve_FAILED is returned
//
// when command_ctx is non-NULL (load_from_disk is implicitly true): this
// call either becomes the owner of the graph's load (and performs it), or
// parks `command_ctx` behind whichever thread already owns it - see
// graph_load_queue.h
//
// when bypass_claim is true (load_from_disk is implicitly true): never
// returns GraphRetrieve_FAILED due to another load being in flight - an
// independent load is attempted regardless (see graph_load.h in the
// enterprise repo); an in-flight offload is never bypassed, since there is
// nothing yet to load in that case
static GraphRetrieveStatus _GraphContext_Retrieve
(
	RedisModuleCtx    *ctx,             // Redis module context
	RedisModuleString *graphID,         // key identifying the graph
	bool               readOnly,        // if true, opens the key in read mode
	bool               shouldCreate,    // create new graph if the key is absent
	bool               load_from_disk,  // load graph from disk if offloaded
	CommandCtx        *command_ctx,     // parked if the graph is being loaded
	bool               bypass_claim,    // never fail due to contention
	GraphContext     **gc               // out: graph context on success
) {
	ASSERT (gc      != NULL) ;
	ASSERT (ctx     != NULL) ;
	ASSERT (graphID != NULL) ;
	ASSERT (!(command_ctx != NULL && bypass_claim)) ;

	*gc = NULL ;

	// reject all access while the module is replicating
	if (aux_field_counter > 0) {
		RedisModule_ReplyWithError (ctx,
				"ERR FalkorDB module is currently replicating") ;
		return GraphRetrieve_FAILED ;
	}

	bool from_thread = (pthread_equal (pthread_self(), MAIN_THREAD_ID) == 0) ;

	// when loading from disk the initial open is always READ (type check only);
	// a successful load re-fetches with the caller's readOnly flag
	int rw_flag = (readOnly || load_from_disk)
		? REDISMODULE_READ
		: REDISMODULE_WRITE ;

	//--------------------------------------------------------------------------
	// Phase 1: inspect key
	//--------------------------------------------------------------------------

	bool is_stub = false ;

	if (from_thread) {
		RedisModule_ThreadSafeContextLock (ctx) ;
	}

	RedisModuleKey *key      = RedisModule_OpenKey (ctx, graphID, rw_flag) ;
	int             key_type = RedisModule_KeyType (key) ;

	if (key_type == REDISMODULE_KEYTYPE_EMPTY) {
		if (shouldCreate) {
			const char *graph_name = RedisModule_StringPtrLen (graphID, NULL) ;
			*gc = _GraphContext_Create (ctx, graph_name) ;
		} else {
			RedisModule_ReplyWithError (ctx,
					"ERR Invalid graph operation on empty key") ;
		}
	} else {
		RedisModuleType *module_type = RedisModule_ModuleTypeGetType (key) ;

		if (module_type == GraphContextRedisModuleType) {
			*gc = RedisModule_ModuleTypeGetValue (key) ;
		} else if (load_falkordb_apis (ctx) && module_type == GraphStubType) {
			is_stub = true ;
		} else {
			RedisModule_ReplyWithError (ctx, REDISMODULE_ERRORMSG_WRONGTYPE) ;
		}
	}

	// Increment the ref count while still holding the GIL (or on the main
	// thread where no other command can run).  Moving it here closes the race
	// where a concurrent GRAPH.OFFLOAD Phase 3 could free the gc between the
	// pointer read above and the increment below.
	if (*gc != NULL) {
		GraphContext_IncreaseRefCount (*gc) ;
	}

	RedisModule_CloseKey (key) ;

	if (from_thread) {
		RedisModule_ThreadSafeContextUnlock (ctx) ;
	}

	//--------------------------------------------------------------------------
	// Phase 2: finalize if graph is already in memory
	//--------------------------------------------------------------------------

	if (*gc != NULL) {
		return GraphRetrieve_RETRIEVED ;
	}

	if (!is_stub) {
		// error already emitted (empty key with no create, or wrong type)
		return GraphRetrieve_FAILED ;
	}

	if (!load_from_disk) {
		return GraphRetrieve_OFFLOADED ;
	}

	//--------------------------------------------------------------------------
	// Phase 3: graph is offloaded — load from disk and re-fetch
	// graph_load manages GIL internally via from_thread;
	// on success, recursive call re-enters Phase 1 to fetch the live graph
	//--------------------------------------------------------------------------

	const char *graph_name = RedisModule_StringPtrLen (graphID, NULL) ;

	if (command_ctx == NULL) {
		// legacy path, or bypass_claim (GraphContext_RetrieveOrForce)
		GraphLoadResult result =
			graph_load (ctx, graphID, from_thread, false, bypass_claim) ;

		if (result == GraphLoad_SUCCESS) {
			return _GraphContext_Retrieve (ctx, graphID, readOnly, shouldCreate,
					false, NULL, false, gc) ;
		}

		_reply_graph_load_error (ctx, graph_name, result) ;

		return GraphRetrieve_FAILED ;
	}

	// queueing path: become the owner of this graph's load, or park behind
	// whichever thread already owns it
	GraphLoadQueueStatus qstatus = GraphLoadQueue_AcquireOrWait (graph_name,
			command_ctx, CommandCtx_ResumeAfterGraphLoad) ;

	if (qstatus == GraphLoadQueue_PARKED) {
		return GraphRetrieve_LOADING ;  // parked; caller does nothing further
	}

	if (qstatus == GraphLoadQueue_FULL) {
		RedisModule_ReplyWithErrorFormat (ctx,
				"ERR too many queries waiting for graph: %s to finish loading",
				graph_name) ;
		return GraphRetrieve_FAILED ;
	}

	// qstatus == GraphLoadQueue_OWNER: perform the load ourselves
	GraphLoadResult result = graph_load (ctx, graphID, from_thread, false,
			false) ;

	// wake everyone parked behind us with our own outcome
	GraphLoadQueue_Drain (graph_name, result == GraphLoad_SUCCESS) ;

	if (result == GraphLoad_SUCCESS) {
		return _GraphContext_Retrieve (ctx, graphID, readOnly, shouldCreate,
				false, NULL, false, gc) ;
	}

	_reply_graph_load_error (ctx, graph_name, result) ;

	return GraphRetrieve_FAILED ;
}

// retrieve the GraphContext for graphID
// on success sets *gc and returns GraphRetrieve_RETRIEVED
// on error emits a reply and returns GraphRetrieve_FAILED
// when load_from_disk=false and the graph is a stub, returns
// GraphRetrieve_OFFLOADED with no error reply
// may be called from any thread when load_from_disk=true
// must be called from the Redis main thread when load_from_disk=false
GraphRetrieveStatus GraphContext_Retrieve
(
	RedisModuleCtx    *ctx,             // Redis module context
	RedisModuleString *graphID,         // key identifying the graph
	bool               readOnly,        // if true, opens the key in read mode
	bool               shouldCreate,    // create new graph if the key is absent
	bool               load_from_disk,  // load graph from disk if offloaded
	GraphContext     **gc               // out: graph context on success
) {
	return _GraphContext_Retrieve (ctx, graphID, readOnly, shouldCreate,
			load_from_disk, NULL, false, gc) ;
}

// Like GraphContext_Retrieve, always attempting to load
// the graph from disk if it is offloaded, but with different handling of
// concurrent loads: if another thread is already loading this stub,
// `command_ctx` is parked instead of GraphRetrieve_LOADING being returned
// immediately. Once that in-flight load resolves,
// CommandCtx_ResumeAfterGraphLoad (see cmd_context.h) is invoked with
// `command_ctx` - the caller must not touch it again until then.
// may be called from any thread
GraphRetrieveStatus GraphContext_RetrieveOrQueue
(
	RedisModuleCtx    *ctx,
	RedisModuleString *graphID,
	bool               readOnly,
	bool               shouldCreate,
	CommandCtx        *command_ctx,
	GraphContext     **gc
) {
	ASSERT (command_ctx != NULL) ;

	return _GraphContext_Retrieve (ctx, graphID, readOnly, shouldCreate,
			true, command_ctx, false, gc) ;
}

// Like GraphContext_Retrieve, always attempting to load
// the graph from disk if it is offloaded, but never failing due to another
// load being in flight: an independent load is attempted regardless (see
// bypass_claim in the enterprise repo's graph_load.h), so
// GraphRetrieve_FAILED is only ever returned for a genuine failure (missing
// dump, OOM, corruption, etc.), never for mere contention with another
// load. An in-flight offload is never bypassed, since there is nothing yet
// to load in that case, so this may still fail while a concurrent offload
// of the same graph is in progress.
// For use by callers that cannot tolerate a transient retrieval failure -
// e.g. GRAPH.EFFECT, where a failure risks master/replica divergence.
// May be called from any thread.
GraphRetrieveStatus GraphContext_RetrieveOrForce
(
	RedisModuleCtx    *ctx,
	RedisModuleString *graphID,
	bool               readOnly,
	bool               shouldCreate,
	GraphContext     **gc
) {
	return _GraphContext_Retrieve (ctx, graphID, readOnly, shouldCreate,
			true, NULL, true, gc) ;
}

