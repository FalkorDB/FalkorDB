/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <unistd.h>
#include <pthread.h>
#include "redismodule.h"
#include "debug.h"
#include "errors.h"
#include "version.h"
#include "globals.h"
#include "util/arr.h"
#include "cron/cron.h"
#include "query_ctx.h"
#include "index/indexer.h"
#include "redisearch_api.h"
#include "arithmetic/funcs.h"
#include "commands/commands.h"
#include "util/thpool/pools.h"
#include "graph/graphcontext.h"
#include "util/redis_version.h"
#include "ast/ast_validations.h"
#include "configuration/config.h"
#include "procedures/procedure.h"
#include "module_event_handlers.h"
#include "serializers/graphmeta_type.h"
#include "configuration/reconf_handler.h"
#include "serializers/graphcontext_type.h"
#include "arithmetic/arithmetic_expression.h"
#include "bolt/socket.h"
#include "bolt/bolt.h"
#include <byteswap.h>

// minimal supported Redis version
#define MIN_REDIS_VERION_MAJOR 6
#define MIN_REDIS_VERION_MINOR 2
#define MIN_REDIS_VERION_PATCH 0

static int _RegisterDataTypes(RedisModuleCtx *ctx) {
	if(GraphContextType_Register(ctx) == REDISMODULE_ERR) {
		printf("Failed to register GraphContext type\n");
		return REDISMODULE_ERR;
	}

	if(GraphMetaType_Register(ctx) == REDISMODULE_ERR) {
		printf("Failed to register GraphMeta type\n");
		return REDISMODULE_ERR;
	}
	return REDISMODULE_OK;
}

// starts cron and register recurring tasks
static bool _Cron_Start(void) {
	// start CRON
	bool res = Cron_Start();

	// register recurring tasks
	Cron_AddRecurringTasks();

	return res;
}

// print RedisGraph configuration
static void _Print_Config
(
	RedisModuleCtx *ctx
) {
	// TODO: consider adding Config_Print

	int ompThreadCount;
	Config_Option_get(Config_OPENMP_NTHREAD, &ompThreadCount);
	RedisModule_Log(ctx, "notice", "Maximum number of OpenMP threads set to %d", ompThreadCount);

	bool cmd_info_enabled = false;
	if(Config_Option_get(Config_CMD_INFO, &cmd_info_enabled) && cmd_info_enabled) {
		uint32_t info_max_query_count = 0;
		Config_Option_get(Config_CMD_INFO_MAX_QUERY_COUNT, &info_max_query_count);
		RedisModule_Log(ctx, "notice", "Query backlog size: %u", info_max_query_count);
	}
}

static int GraphBLAS_Init(RedisModuleCtx *ctx) {
	// GraphBLAS should use Redis allocator
	GrB_Info res = GxB_init(GrB_NONBLOCKING, RedisModule_Alloc,
			RedisModule_Calloc, RedisModule_Realloc, RedisModule_Free);
	if(res != GrB_SUCCESS) {
		RedisModule_Log(ctx, "warning", "Encountered error initializing GraphBLAS");
		return REDISMODULE_ERR;
	}

	// all matrices in CSR format
	GxB_set(GxB_FORMAT, GxB_BY_ROW);

	return REDISMODULE_OK;
}

void BoltRequestHandler
(
	int fd,
	void *user_data,
	int mask
) {
	bolt_client_t *client = (bolt_client_t*)user_data;
	int nread = socket_read(fd, client->read_buffer, 2);
	if(nread == -1) {
		RedisModule_Log(NULL, "warning", "Socket read error");
		return;
	}
	if(nread == 0) {
		socket_close(fd);
		RedisModule_EventLoopDel(fd, REDISMODULE_EVENTLOOP_READABLE);
		return;
	}

	uint16_t size = bswap_16(*(uint16_t*)client->read_buffer);

	nread = socket_read(fd, client->read_buffer, size + 2);
	if(nread == -1) {
		RedisModule_Log(NULL, "warning", "Socket read error");
		return;
	}

	if(client->read_buffer[size] != 0x00 || client->read_buffer[size + 1] != 0x00) {
		RedisModule_Log(NULL, "warning", "Socket read error");
		return;
	}

	switch (bolt_value_get_structure_type(client->read_buffer))
	{
		case BST_HELLO: {
			bolt_reply_structure(client, BST_SUCCESS, 1);
			bolt_reply_map(client, 1);
			bolt_reply_string(client, "server");
			bolt_reply_string(client, "Neo4j/5.7.0");
			bolt_client_send(client);
			break;
		}
		case BST_LOGON: {
			bolt_reply_structure(client, BST_SUCCESS, 1);
			bolt_reply_map(client, 0);
			bolt_client_send(client);
			break;
		}
		case BST_LOGOFF:
			break;
		case BST_GOODBYE:
			break;
		case BST_RESET: {
			bolt_reply_structure(client, BST_SUCCESS, 1);
			bolt_reply_map(client, 0);
			bolt_client_send(client);
			break;
		}
		case BST_RUN: {
			RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
			RedisModuleString *args[5];
			char *db = bolt_value_get_structure_value(client->read_buffer, 2);
			char *db_str;
			size_t db_len;
			if(bolt_value_get_map_size(db) == 0) {
				db_str = "x";
				db_len = 1;
			} else {
				db = bolt_value_get_map_value(db, 0);
				db_str = bolt_value_get_string(db);
				db_len = bolt_value_get_string_size(db);
			}
			char *query = bolt_value_get_structure_value(client->read_buffer, 0);
			char *parameters = bolt_value_get_structure_value(client->read_buffer, 1);
			uint32_t params_count = bolt_value_get_map_size(parameters);
			char *parameters_str = rm_malloc(512);
			sprintf(parameters_str, "CYPHER");
			args[0] = RedisModule_CreateString(ctx, "graph.QUERY", 11);
			args[1] = RedisModule_CreateString(ctx, db_str, db_len);
			if(params_count > 0) {
				for (uint i = 0; i < params_count; i++) {
					char *key = bolt_value_get_map_key(parameters, i);
					char *value = bolt_value_get_map_value(parameters, i);
					switch (bolt_value_get_type(value))
					{
					case BVT_STRING:
						sprintf(parameters_str, "%s %.*s='%.*s'", parameters_str, bolt_value_get_string_size(key), bolt_value_get_string(key), bolt_value_get_string_size(value), bolt_value_get_string(value));
						break;
					case BVT_STRUCTURE:
						if(bolt_value_get_structure_type(value) == BST_POINT2D) {
							// value = bolt_value_get_structure_value(value, 0);
							// sprintf(parameters_str, "%s %.*s=POINT({latitude: %f, longitude: %f})", parameters_str, bolt_value_get_string_size(key), bolt_value_get_string(key), bolt_value_get_string_size(value), bolt_value_get_string(value));
							break;
						}
						ASSERT(false);
						break;
					default:
						ASSERT(false);
						break;
					}
				}
				sprintf(parameters_str, "%s %.*s", parameters_str, bolt_value_get_string_size(query), bolt_value_get_string(query));
				args[2] = RedisModule_CreateString(ctx, parameters_str, strlen(parameters_str));
			} else {
				args[2] = RedisModule_CreateString(ctx, bolt_value_get_string(query), bolt_value_get_string_size(query));
			}
			args[3] = RedisModule_CreateString(ctx, "--bolt", 6);
			args[4] = RedisModule_CreateStringFromLongLong(ctx, (long long)client);

			CommandDispatch(ctx, args, 5);
			break;
		}
		case BST_DISCARD:
			break;
		case BST_PULL: {
			break;
		}
		case BST_BEGIN: {
			bolt_reply_structure(client, BST_SUCCESS, 1);
			bolt_reply_map(client, 0);
			bolt_client_send(client);
			break;
		}
		case BST_COMMIT:{
			bolt_reply_structure(client, BST_SUCCESS, 1);
			bolt_reply_map(client, 0);
			bolt_client_send(client);
			break;
		}
		case BST_ROLLBACK:{
			bolt_reply_structure(client, BST_SUCCESS, 1);
			bolt_reply_map(client, 0);
			bolt_client_send(client);
			break;
		}
		case BST_ROUTE:
			break;
		default:
			break;
	}
}

void BoltAcceptHandler
(
	int fd,
	void *user_data,
	int mask
) {
	socket_t client = socket_accept(fd);
	if(client == -1) return;

	if(!bolt_check_handshake(client)) {
		socket_close(client);
		return;
	}

	bolt_version_t version = bolt_read_supported_version(client);

	char data[4];
	data[0] = 0x00;
	data[1] = 0x00;
	data[2] = version.minor;
	data[3] = version.major;
	socket_write(client, data, 4);

	RedisModule_EventLoopAdd(client, REDISMODULE_EVENTLOOP_READABLE, BoltRequestHandler, bolt_client_new(client));
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
	if(RedisModule_Init(ctx, "graph", REDISGRAPH_MODULE_VERSION,
						REDISMODULE_APIVER_1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	// initialize GraphBLAS
	int res = GraphBLAS_Init(ctx);
	if(res != REDISMODULE_OK) return res;

	// validate minimum redis-server version
	if(!Redis_Version_GreaterOrEqual(MIN_REDIS_VERION_MAJOR,
									 MIN_REDIS_VERION_MINOR, MIN_REDIS_VERION_PATCH)) {
		RedisModule_Log(ctx, "warning", "RedisGraph requires redis-server version %d.%d.%d and up",
						MIN_REDIS_VERION_MAJOR, MIN_REDIS_VERION_MINOR, MIN_REDIS_VERION_PATCH);
		return REDISMODULE_ERR;
	}

	if(RediSearch_Init(ctx, REDISEARCH_INIT_LIBRARY) != REDISMODULE_OK) {
		return REDISMODULE_ERR;
	}

	RedisModule_Log(ctx, "notice", "Starting up RedisGraph version %d.%d.%d.",
					REDISGRAPH_VERSION_MAJOR, REDISGRAPH_VERSION_MINOR, REDISGRAPH_VERSION_PATCH);

	Proc_Register();     // register procedures
	AR_RegisterFuncs();  // register arithmetic functions

	// set up the module's configurable variables,
	// using user-defined values where provided
	// register for config updates
	Config_Subscribe_Changes(reconf_handler);
	if(Config_Init(ctx, argv, argc) != REDISMODULE_OK) return REDISMODULE_ERR;

	RegisterEventHandlers(ctx);

	// create thread local storage keys for query and error contexts
	if(!_Cron_Start())                return REDISMODULE_ERR;
	if(!QueryCtx_Init())              return REDISMODULE_ERR;
	if(!ErrorCtx_Init())              return REDISMODULE_ERR;
	if(!ThreadPools_Init())           return REDISMODULE_ERR;
	if(!Indexer_Init())               return REDISMODULE_ERR;
	if(!AST_ValidationsMappingInit()) return REDISMODULE_ERR;

	RedisModule_Log(ctx, "notice", "Thread pool created, using %d threads.",
			ThreadPools_ReadersCount());

	int ompThreadCount;
	Config_Option_get(Config_OPENMP_NTHREAD, &ompThreadCount);

	if(GxB_set(GxB_NTHREADS, ompThreadCount) != GrB_SUCCESS) {
		RedisModule_Log(ctx, "warning", "Failed to set OpenMP thread count to %d", ompThreadCount);
		return REDISMODULE_ERR;
	}

	// log configuration
	_Print_Config(ctx);

	if(_RegisterDataTypes(ctx) != REDISMODULE_OK) return REDISMODULE_ERR;

	if(RedisModule_CreateCommand(ctx, "graph.QUERY", CommandDispatch, "write deny-oom", 1, 1,
								 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.RO_QUERY", CommandDispatch, "readonly", 1, 1,
								 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.DELETE", Graph_Delete, "write", 1, 1,
								 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.EXPLAIN", CommandDispatch, "write deny-oom", 1, 1,
								 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.PROFILE", CommandDispatch, "write deny-oom", 1, 1,
								 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.BULK", Graph_BulkInsert, "write deny-oom", 1, 1,
								 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.CONSTRAINT", Graph_Constraint, "write deny-oom", 2, 2,
								 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.SLOWLOG", Graph_Slowlog, "readonly", 1, 1,
								 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.CONFIG", Graph_Config, "readonly", 0, 0,
								 0) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.LIST", Graph_List, "readonly", 0, 0,
								 0) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.DEBUG", Graph_Debug, "readonly", 0, 0,
								 0) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.INFO", Graph_Info, "readonly", 1, 1,
				1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx, "graph.EFFECT", Graph_Effect, "write", 1,
				1, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	socket_t bolt = socket_bind(7687);
	if(bolt == -1) {
		RedisModule_Log(NULL, "warning", "Failed to bind to port 7687");
		return REDISMODULE_ERR;
	}

	if(RedisModule_EventLoopAdd(bolt, REDISMODULE_EVENTLOOP_READABLE, BoltAcceptHandler, NULL) == REDISMODULE_ERR) {
		RedisModule_Log(NULL, "warning", "Failed to register socket accept handler");
		return REDISMODULE_ERR;
	}
	RedisModule_Log(NULL, "notice", "bolt server initialized");

	// set up global variables scoped to the entire module
	Globals_Init();

	setupCrashHandlers(ctx);

	return REDISMODULE_OK;
}

