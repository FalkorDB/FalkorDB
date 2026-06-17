/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "graphcontext.h"
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
    GraphLoad_SUCCESS,  // graph restored from disk successfully
    GraphLoad_LOADING,  // a load for this key is already in progress
    GraphLoad_MISSING,  // key or dump file does not exist
    GraphLoad_OOM,      // not enough memory to hold the loaded graph
    GraphLoad_ERR,      // all other failures
} GraphLoadResult ;

typedef RedisModuleType* (*GraphStubType_Get_t) (void) ;

typedef GraphLoadResult (*graph_load_t)
(
    RedisModuleCtx    *ctx,
    RedisModuleString *key_name,
    bool              from_thread,
	bool              force
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
	ASSERT (gc      != NULL) ;
	ASSERT (ctx     != NULL) ;
	ASSERT (graphID != NULL) ;

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

	if (graph_load (ctx, graphID, from_thread, false) == GraphLoad_SUCCESS) {
		return GraphContext_Retrieve (ctx, graphID, readOnly, shouldCreate,
				false, gc) ;
	}

	return GraphRetrieve_FAILED ;
}

