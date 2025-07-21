/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "graphcontext_type.h"
#include "../version.h"
#include "../globals.h"
#include "encoding_version.h"
#include "encoder/encode_graph.h"
#include "decoders/decode_graph.h"
#include "decoders/decode_previous.h"
#include "../util/redis_version.h"

// forward declerations of the module event handler functions
void ModuleEventHandler_AUXBeforeKeyspaceEvent(void);
void ModuleEventHandler_AUXAfterKeyspaceEvent(void);

// declaration of the type for redis registration
RedisModuleType *GraphContextRedisModuleType;

static void *_GraphContextType_RdbLoad
(
	RedisModuleIO *rdb,
	int encver
) {
	GraphContext *gc = NULL;

	if(encver > GRAPH_ENCODING_VERSION_LATEST) {
		// not forward compatible
		printf("Failed loading Graph, RedisGraph version (%d) is not forward compatible.\n",
			   REDISGRAPH_MODULE_VERSION);
		return NULL;
		// not backward compatible
	} else if(encver < GRAPHCONTEXT_TYPE_DECODE_MIN_V) {
		printf("Failed loading Graph, RedisGraph version (%d) is not backward compatible with encoder version %d.\n",
			   REDISGRAPH_MODULE_VERSION, encver);
		return NULL;
		// previous version
	} else if(encver < GRAPH_ENCODING_VERSION_LATEST) {
		gc = Decode_Previous(rdb, encver);
	} else {
		// current version
		gc = RdbLoadGraph(rdb);
	}

	// add GraphContext to global array of graphs
	GraphContext_RegisterWithModule(gc);
	return gc;
}

// save RDB for Redis 6 and up
static void _GraphContextType_RdbSave(RedisModuleIO *rdb, void *value) {
	RdbSaveGraph(rdb, value);
}

// save an unsigned placeholder before and after the keyspace encoding
static void _GraphContextType_AuxSave
(
	RedisModuleIO *rdb,
	int when
) {
	RedisModule_SaveUnsigned(rdb, 0);
}

// save an unsigned placeholder before and after the keyspace encoding
static void _GraphContextType_AuxSave2
(
	RedisModuleIO *rdb,
	int when
) {
	// only write AUX field if there are graphs in the keyspace
	if(Globals_GetGraphCount() > 0) {
		RedisModule_SaveUnsigned(rdb, 0);
	}
}

// decode the unsigned placeholders saved before and after the keyspace values
// and call the module event handler
static int _GraphContextType_AuxLoad(RedisModuleIO *rdb, int encver, int when) {
	RedisModule_LoadUnsigned(rdb);
	if(when == REDISMODULE_AUX_BEFORE_RDB) ModuleEventHandler_AUXBeforeKeyspaceEvent();
	else ModuleEventHandler_AUXAfterKeyspaceEvent();
	return REDISMODULE_OK;
}

static void _GraphContextType_Free(void *value) {
	GraphContext *gc = value;
	Globals_RemoveGraph(gc);
	GraphContext_DecreaseRefCount(gc);
}

int GraphContextType_Register(RedisModuleCtx *ctx) {
	RedisModuleTypeMethods tm = { 0 };
	tm.free               =  _GraphContextType_Free;
	tm.version            =  REDISMODULE_TYPE_METHOD_VERSION;
	tm.rdb_load           =  _GraphContextType_RdbLoad;
	tm.rdb_save           =  _GraphContextType_RdbSave;
	tm.aux_save           =  _GraphContextType_AuxSave;
	tm.aux_load           =  _GraphContextType_AuxLoad;
	tm.aux_save_triggers  =  REDISMODULE_AUX_BEFORE_RDB | REDISMODULE_AUX_AFTER_RDB;

	// use aux_save2 if available
	Redis_Version redis_version = RG_GetRedisVersion();
	if(redis_version.major > 7 ||
	  (redis_version.major == 7 && redis_version.minor >= 2)) {
		tm.aux_save  = NULL;
		tm.aux_save2 = _GraphContextType_AuxSave2;
	}

	GraphContextRedisModuleType = RedisModule_CreateDataType(ctx, "graphdata",
			GRAPH_ENCODING_VERSION_LATEST, &tm);

	if(GraphContextRedisModuleType == NULL) return REDISMODULE_ERR;
	return REDISMODULE_OK;
}

