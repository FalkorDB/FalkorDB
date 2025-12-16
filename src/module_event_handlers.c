/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "module_event_handlers.h"
#include "LAGraph.h"
#include "globals.h"
#include "util/uuid.h"
#include "cron/cron.h"
#include "index/indexer.h"
#include "bolt/bolt_api.h"
#include "commands/cmd_acl.h"
#include "util/thpool/pool.h"
#include "util/redis_version.h"
#include "graph/graphcontext.h"
#include "configuration/config.h"
#include "serializers/graphmeta_type.h"
#include "serializers/graphcontext_type.h"
#include "commands/util/run_redis_command_as.h"

#include <pthread.h>
#include <stdbool.h>

// indicates the possibility of half-baked graphs in the keyspace
#define INTERMEDIATE_GRAPHS (aux_field_counter > 0)

// graphContext type as it is registered at Redis
extern RedisModuleType *GraphContextRedisModuleType;
// graph meta keys type as it is registered at Redis
extern RedisModuleType *GraphMetaRedisModuleType;

// both of the following fields are required to verify that the module is replicated
// in a successful manner. In a sharded environment, there could be a race condition between the decoding of
// the last key, and the last aux_fields, so both counters should be zeroed in order to verify
// that the module replicated properly.*/

// holds the number of aux fields encountered during decoding of RDB file
// this field is used to represent when the module is replicating its graphs
uint aux_field_counter = 0 ;

// holds the id of the Redis Main thread in order to figure out the context the fork is running on
static pthread_t redis_main_thread_id;

// this callback invokes once rename for a graph is done
// since the key value is a graph context
// which saves the name of the graph for later key accesses
// this data must be consistent with the key name
// otherwise the graph context will remain with the previous graph name
// and a key access to this name might yield an empty key or wrong value
// this method changes the graph name value at the graph context to be
// consistent with the key name
static int _GenericKeyspaceHandler
(
	RedisModuleCtx *ctx,
	int type,
	const char *event,
	RedisModuleString *key_name
) {
	if(type != REDISMODULE_NOTIFY_GENERIC) {
		return REDISMODULE_OK;
	}

	if(strcasecmp(event, "RENAME_TO") == 0) {
		RedisModuleKey *key = RedisModule_OpenKey(ctx, key_name,
				REDISMODULE_WRITE);
		if(RedisModule_ModuleTypeGetType(key) == GraphContextRedisModuleType) {
			GraphContext *gc = RedisModule_ModuleTypeGetValue(key);
			size_t len;
			const char *new_name = RedisModule_StringPtrLen(key_name, &len);
			GraphContext_Rename(ctx, gc, new_name);
		}
		RedisModule_CloseKey(key);
	}

	else if(strcasecmp(event, "DEL") == 0) {
		// TODO: in Redis 7.2 we could use REDISMODULE_EVENT_KEY to register
		// for a more convenient key event notification
		const char *graph_name = RedisModule_StringPtrLen(key_name, NULL);
		Globals_RemoveGraphByName(graph_name);
	}

	return REDISMODULE_OK;
}

//------------------------------------------------------------------------------
// Meta keys API
//------------------------------------------------------------------------------

// Checks if the graph name contains a hash tag between curly braces.
static bool _GraphContext_NameContainsTag
(
	const GraphContext *gc
) {
	const char *left_curly_brace = strstr(gc->graph_name, "{");
	if(left_curly_brace) {
		const char *right_curly_brace = strstr(left_curly_brace, "}");
		if(right_curly_brace) {
			return true;
		}
	}
	return false;
}

// Calculate how many virtual keys are needed to represent the graph.
static uint64_t _GraphContext_RequiredMetaKeys
(
	const GraphContext *gc
) {
	uint64_t vkey_entity_count;
	Config_Option_get(Config_VKEY_MAX_ENTITY_COUNT, &vkey_entity_count);
	gc->encoding_context->vkey_entity_count = vkey_entity_count;

	uint64_t entities_count = Graph_NodeCount(gc->g) + Graph_EdgeCount(gc->g) +
		Graph_DeletedNodeCount(gc->g) + Graph_DeletedEdgeCount(gc->g);

	if(entities_count == 0) return 0;

	// calculate the required keys
	// substruct one since there is also the graph context key
	uint64_t key_count = ceil((double)entities_count / vkey_entity_count) - 1;
	return MAX(key_count, 0);
}

static void _CreateGraphMetaKeys
(
	RedisModuleCtx *ctx,
	GraphContext *gc
) {
	uint meta_key_count = _GraphContext_RequiredMetaKeys(gc);
	bool graph_name_contains_tag = _GraphContext_NameContainsTag(gc);
	for(uint i = 1; i <= meta_key_count; i++) {
		char *uuid = UUID_New();
		RedisModuleString *meta_rm_string;
		// meta keys need to be in the exact shard/slot as the graph context key
		// to avoid graph sharding at the target db
		// we want to save all the graph keys on the same shard
		// for that, we need to that them In so their tag hash value will be
		// the same as the graph context key hash value
		// if the graph name already contains a tag, we can duplicate
		// the graph name completely for each meta key
		// if not, the meta keys tag will be the graph name, so
		// when hashing the graphcontext key name (graph name)
		// and the graph meta key tag (graph name)
		// the hash values will be the same
		if(graph_name_contains_tag) {
			// graph already has a tag, create a meta key of "graph_name_uuid"
			meta_rm_string = RedisModule_CreateStringPrintf(ctx, "%s_%s",
					gc->graph_name, uuid);
		} else {
			// graph is untagged, one must be introduced to ensure that
			// keys are propagated to the same node
			// create a meta key of "{graph_name}graph_name_i"
			meta_rm_string = RedisModule_CreateStringPrintf(ctx, "{%s}%s_%s",
					gc->graph_name, gc->graph_name, uuid);
		}

		const char *key_name = RedisModule_StringPtrLen(meta_rm_string, NULL);
		GraphEncodeContext_AddMetaKey(gc->encoding_context, key_name);
		RedisModuleKey *key = RedisModule_OpenKey(ctx, meta_rm_string,
				REDISMODULE_WRITE);

		// set value in key
		RedisModule_ModuleTypeSetValue(key, GraphMetaRedisModuleType, gc);
		RedisModule_CloseKey(key);
		
		// increase graph context ref count for each virtual key
		GraphContext_IncreaseRefCount(gc);
		RedisModule_FreeString(ctx, meta_rm_string);
		rm_free(uuid);
	}

	RedisModule_Log(ctx, "notice", "Created %d virtual keys for graph %s",
			meta_key_count, gc->graph_name);
}

// delete meta keys, upon RDB encode or decode finished event triggering
// the decode flag represent the event
static void _DeleteGraphMetaKeys
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	bool decode
) {
	uint key_count;
	unsigned char **keys;

	// get the meta keys required, according to the "decode" flag.
	if(decode) {
		keys = GraphDecodeContext_GetMetaKeys(gc->decoding_context);
	} else {
		keys = GraphEncodeContext_GetMetaKeys(gc->encoding_context);
	}

	key_count = array_len(keys);
	for(uint i = 0; i < key_count; i++) {
		RedisModuleString *meta_rm_string =
			RedisModule_CreateStringPrintf(ctx, "%s", keys[i]);

		RedisModuleKey *key =
			RedisModule_OpenKey(ctx, meta_rm_string, REDISMODULE_WRITE);

		RedisModule_DeleteKey(key);
		RedisModule_CloseKey(key);
		RedisModule_FreeString(ctx, meta_rm_string);

		rm_free(keys[i]);
	}

	array_free(keys);

	// clear the relevant context meta keys as they are no longer valid
	if(decode) {
		GraphDecodeContext_ClearMetaKeys(gc->decoding_context);
	} else {
		GraphEncodeContext_ClearMetaKeys(gc->encoding_context);
	}

	RedisModule_Log(ctx, "notice", "Deleted %d virtual keys for graph %s",
			key_count, gc->graph_name);
}

// create the meta keys for each graph in the keyspace
// used on RDB start event
static void _CreateKeySpaceMetaKeys
(
	RedisModuleCtx *ctx
) {
	KeySpaceGraphIterator it;
	GraphContext *gc = NULL;
	Globals_ScanGraphs(&it);

	while((gc = GraphIterator_Next(&it)) != NULL) {
		_CreateGraphMetaKeys(ctx, gc);
		GraphContext_DecreaseRefCount(gc);
	}
}

// delete the meta keys for each graph in the keyspace
// used on RDB finish (save/load/fail) event
// the decode flag represent if the graph is after encodeing or decodeing
static void _ClearKeySpaceMetaKeys
(
	RedisModuleCtx *ctx,
	bool decode
) {
	KeySpaceGraphIterator it;
	GraphContext *gc = NULL;
	Globals_ScanGraphs(&it);

	while((gc = GraphIterator_Next(&it)) != NULL) {
		_DeleteGraphMetaKeys(ctx, gc, decode);
		GraphContext_DecreaseRefCount(gc);
	}
}

static void _FlushDBHandler
(
	RedisModuleCtx *ctx,
	RedisModuleEvent eid,
	uint64_t subevent,
	void *data
) {
	ASSERT(eid.id == REDISMODULE_EVENT_FLUSHDB);

	if(subevent == REDISMODULE_SUBEVENT_FLUSHDB_START) {
		// clear global graphs tracking
		Globals_ClearGraphs(ctx);

		// reset `aux_field_counter`
		aux_field_counter = 0;
	}
}

// Checks if the event is persistence start event.
static bool _IsEventPersistenceStart
(
	RedisModuleEvent eid,
	uint64_t subevent
) {
	return eid.id == REDISMODULE_EVENT_PERSISTENCE  &&
		   (subevent == REDISMODULE_SUBEVENT_PERSISTENCE_RDB_START      ||    // Normal RDB.
			subevent == REDISMODULE_SUBEVENT_PERSISTENCE_AOF_START      ||    // Preamble AOF.
			subevent == REDISMODULE_SUBEVENT_PERSISTENCE_SYNC_RDB_START ||    // SAVE and DEBUG RELOAD.
			subevent == REDISMODULE_SUBEVENT_PERSISTENCE_SYNC_AOF_START       // 
		   );
}

// Checks if the event is persistence end event.
static bool _IsEventPersistenceEnd
(
	RedisModuleEvent eid,
	uint64_t subevent
) {
	return eid.id == REDISMODULE_EVENT_PERSISTENCE &&
		   (subevent == REDISMODULE_SUBEVENT_PERSISTENCE_ENDED ||  // Save ended.
			subevent == REDISMODULE_SUBEVENT_PERSISTENCE_FAILED    // Save failed.
		   );
}

static void _ReplicationRoleChangedEventHandler
(
	RedisModuleCtx *ctx,
	RedisModuleEvent eid,
	uint64_t subevent,
	void *data
) {
	//KeySpaceGraphIterator it;
	//GraphContext *gc = NULL;
	//Globals_ScanGraphs(&it);
	//if(subevent == REDISMODULE_EVENT_REPLROLECHANGED_NOW_MASTER) {
	//	// now master enable constraints
	//	while((gc = GraphIterator_Next(&it)) != NULL) {
	//		GraphContext_EnableConstrains(gc);
	//		GraphContext_DecreaseRefCount(gc);
	//	}
	//} else if (subevent == REDISMODULE_EVENT_REPLROLECHANGED_NOW_REPLICA) {
	//	// now slave disable constraints
	//	while((gc = GraphIterator_Next(&it)) != NULL) {
	//		GraphContext_DisableConstrains(gc);
	//		GraphContext_DecreaseRefCount(gc);
	//	}
	//}
}

// server persistence event handler
static void _PersistenceEventHandler
(
	RedisModuleCtx *ctx,
	RedisModuleEvent eid,
	uint64_t subevent,
	void *data
) {
	if(INTERMEDIATE_GRAPHS) {
		// check for half-baked graphs
		// indicated by `aux_field_counter` > 0
		// in such case we do not want to either perform backup nor do we want to
		// synchronize our replica, as such we're aborting by existing
		// assuming we're running on a fork process
		if(Globals_Get_ProcessIsChild()) {
			// intermediate graph(s) detected, exit!
			RedisModule_Log(NULL, REDISMODULE_LOGLEVEL_WARNING,
					"RedisGraph - aborting BGSAVE, detected intermediate graph(s)");

			exit(255);
		} else {
			// don't mess with the keyspace if we have half-baked graphs
			return;
		}
	}

	if(_IsEventPersistenceStart(eid, subevent)) {
		_CreateKeySpaceMetaKeys(ctx);
	} else if(_IsEventPersistenceEnd(eid, subevent)) {
		_ClearKeySpaceMetaKeys(ctx, false);
	}
}

// Perform clean-up upon server shutdown.
static void _ShutdownEventHandler
(
	RedisModuleCtx *ctx,
	RedisModuleEvent eid,
	uint64_t subevent,
	void *data
) {
	void RediSearch_CleanupModule();
	if (!getenv("RS_GLOBAL_DTORS")) {  // used only with sanitizer or valgrind
		return; 
	}

	// stop cron
	Cron_Stop();

	// stop indexer
	Indexer_Stop () ;

	// stop threads before finalize GraphBLAS
	ThreadPool_Destroy();

	// server is shutting down, finalize GraphBLAS
	LAGraph_Finalize(NULL);

	RedisModule_Log(ctx, "notice", "%s", "Clearing RediSearch resources on shutdown");
	RediSearch_CleanupModule();

	free_cmd_acl();
	free_run_cmd_as();

	BoltApi_Unregister();

	// free global variables
	Globals_Free();
}

static void _ModuleLoadedHandler
(
	RedisModuleCtx *ctx,
	RedisModuleEvent eid,
	uint64_t subevent,
	void *data
) {
	if(subevent == REDISMODULE_SUBEVENT_MODULE_LOADED) {
		// see which module been loaded
		RedisModuleModuleChange *pdata = (RedisModuleModuleChange*)data;
		if(strcmp(pdata->module_name, "graph") == 0) {
			Cron_Start();
			Cron_AddRecurringTasks();
		}
	}
}

static void _RegisterServerEvents
(
	RedisModuleCtx *ctx
) {
	int res;
	res = RedisModule_SubscribeToServerEvent(ctx,
			RedisModuleEvent_FlushDB,
			_FlushDBHandler);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_SubscribeToServerEvent(ctx,
			RedisModuleEvent_Shutdown,
			_ShutdownEventHandler);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_SubscribeToServerEvent(ctx,
			RedisModuleEvent_Persistence,
			_PersistenceEventHandler);
	ASSERT(res == REDISMODULE_OK);

	// TODO: try to use RedisModuleEvent_ModuleChange to start cron
	//res = RedisModule_SubscribeToServerEvent(ctx,
	//		RedisModuleEvent_ModuleChange,
	//		_ModuleLoadedHandler);
	//ASSERT(res == REDISMODULE_OK);

//	RedisModule_SubscribeToServerEvent (ctx,
//			RedisModuleEvent_ReplicationRoleChanged,
//			_ReplicationRoleChangedEventHandler) ;

	RedisModule_SubscribeToKeyspaceEvents(ctx,
			REDISMODULE_NOTIFY_GENERIC,
			_GenericKeyspaceHandler);
}

//------------------------------------------------------------------------------
// FORK callbacks
//------------------------------------------------------------------------------

// before fork at parent
static void RG_ForkPrepare() {
	// at this point, fork been issued, we assume that this is due to BGSAVE
	// or RedisSearch GC
	//
	// on BGSAVE acquire read lock for each graph to ensure no graph is being
	// modified, otherwise the child process might inherit a malformed matrix
	//
	// on BGSAVE: acquire read lock
	// flush all matrices such that child won't inherit locked matrix
	// release read lock immediately once forked
	//
	// in the case of RediSearch GC fork, quickly return

	// BGSAVE is invoked from Redis main thread
	if(!pthread_equal(pthread_self(), redis_main_thread_id)) {
		return;
	}

	// return if we have half-baked graphs
	if(INTERMEDIATE_GRAPHS) {
		return;
	}

	// measure and report prep time
	double tic[2];
	simple_tic(tic);

	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);

	// scan through each graph in the keyspace
	GraphContext *gc = NULL;
	KeySpaceGraphIterator it;
	Globals_ScanGraphs(&it);

	while((gc = GraphIterator_Next(&it)) != NULL) {
		// acquire read lock, guarantee graph isn't modified by a writer
		Graph *g = gc->g;
		Graph_AcquireReadLock(g);  // release in RG_AfterForkParent

		// set matrix synchronization policy to default
		Graph_SetMatrixPolicy(g, SYNC_POLICY_FLUSH_RESIZE);

		// synchronize all matrices, make sure they're in a consistent state
		// do not force-flush as this can take awhile

		//----------------------------------------------------------------------
		// sync graph's matrices
		//----------------------------------------------------------------------

		// calling Graph_Get* will sync the retrieved matrix

		Graph_GetAdjacencyMatrix(g, false);
		RedisModule_Yield(ctx, REDISMODULE_YIELD_FLAG_CLIENTS,
				"preparing to fork");

		Graph_GetNodeLabelMatrix(g);
		RedisModule_Yield(ctx, REDISMODULE_YIELD_FLAG_CLIENTS,
				"preparing to fork");

		int n_lbls = Graph_LabelTypeCount(g);
		for (int i = 0; i < n_lbls; i++) {
			Graph_GetLabelMatrix(g, i);
			RedisModule_Yield(ctx, REDISMODULE_YIELD_FLAG_CLIENTS,
					"preparing to fork");
		}

		int n_rels = Graph_RelationTypeCount(g);
		for (int i = 0; i < n_rels; i++) {
			Graph_GetRelationMatrix(g, i, false);
			RedisModule_Yield(ctx, REDISMODULE_YIELD_FLAG_CLIENTS,
					"preparing to fork");
		}

		// decrease graph context ref count
		GraphContext_DecreaseRefCount(gc);
	}

	RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_NOTICE,
			"Fork preparation time: %.6f sec\n", simple_toc(tic));

	// clean up
	RedisModule_FreeThreadSafeContext(ctx);
}

// after fork at parent
static void RG_AfterForkParent() {
	// BGSAVE is invoked from Redis main thread
	if(!pthread_equal(pthread_self(), redis_main_thread_id)) return;

	// return if we have half-baked graphs
	if(INTERMEDIATE_GRAPHS) return;

	// the child process forked, release all acquired locks
	GraphContext *gc = NULL;
	KeySpaceGraphIterator it;
	Globals_ScanGraphs(&it);

	while((gc = GraphIterator_Next(&it)) != NULL) {
		// release read lock
		Graph_ReleaseLock(gc->g);

		// decrease graph context ref count
		GraphContext_DecreaseRefCount(gc);
	}
}

// after fork at child
static void RG_AfterForkChild() {
	// mark that the child is a forked process so that it doesn't
	// attempt invalid accesses of POSIX primitives it doesn't own
	Globals_Set_ProcessIsChild(true);

	// restrict GraphBLAS to use a single thread this is done for 2 reasons:
	// 1. save resources
	// 2. avoid a bug in GNU OpenMP which hangs when performing parallel loop
	// in forked process
	GxB_set(GxB_NTHREADS, 1);

	GraphContext *gc = NULL;
	KeySpaceGraphIterator it;
	Globals_ScanGraphs(&it);

	while((gc = GraphIterator_Next(&it)) != NULL) {
		// matrices should be synced, don't waste time
		Graph_SetMatrixPolicy(gc->g, SYNC_POLICY_NOP);

		// decrease graph context ref count
		GraphContext_DecreaseRefCount(gc);
	}
}

static void _RegisterForkHooks() {
	redis_main_thread_id = pthread_self();  // This function is being called on the main thread context.

	// register handlers to control the behavior of fork calls
	int res = pthread_atfork(RG_ForkPrepare, RG_AfterForkParent, RG_AfterForkChild);
	ASSERT(res == 0);
}

static void _ModuleEventHandler_TryClearKeyspace(void) {
	// return if we have half-baked graphs
	if(INTERMEDIATE_GRAPHS) return;

	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
	_ClearKeySpaceMetaKeys(ctx, true);
	RedisModule_FreeThreadSafeContext(ctx);
}

// increase the number of aux fields encountered during rdb loading
// there could be more than one on multiple shards scenario
// so each shard is saving the aux field in its own RDB file
void ModuleEventHandler_AUXBeforeKeyspaceEvent(void) {
	aux_field_counter++;
}

// decrease the number of aux fields encountered during rdb loading
// there could be more than one on multiple shards scenario
// so each shard is saving the aux field in its own RDB file
// once the number is zero,
// the module finished replicating and the meta keys can be deleted
void ModuleEventHandler_AUXAfterKeyspaceEvent(void) {
	aux_field_counter--;
	_ModuleEventHandler_TryClearKeyspace();
}

void RegisterEventHandlers
(
	RedisModuleCtx *ctx
) {
	_RegisterForkHooks();       // set up hooks for forking logic to prevent bgsave deadlocks
	_RegisterServerEvents(ctx); // set up hooks for del/rename and server events
}

