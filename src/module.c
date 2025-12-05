/*
 * Copyright Redis Ltd. 2018 - present
 * Copyright FalkorDB Ltd. 2024 - present
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
#include "LAGraph.h"
#include "util/arr.h"
#include "cron/cron.h"
#include "query_ctx.h"
#include "udf/udf_ctx.h"
#include "udf/classes.h"
#include "util/roaring.h"
#include "bolt/bolt_api.h"
#include "index/indexer.h"
#include "udf/repository.h"
#include "redisearch_api.h"
#include "commands/cmd_acl.h"
#include "arithmetic/funcs.h"
#include "util/thpool/pool.h"
#include "commands/commands.h"
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
#include "commands/util/run_redis_command_as.h"

// minimal supported Redis version
#define MIN_REDIS_VERSION_MAJOR 7
#define MIN_REDIS_VERSION_MINOR 2
#define MIN_REDIS_VERSION_PATCH 0


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

// print FalkorDB configuration
static void _Print_Config
(
	RedisModuleCtx *ctx
) {
	// TODO: consider adding Config_Print

	uint64_t ompThreadCount;
	Config_Option_get(Config_OPENMP_NTHREAD, &ompThreadCount);
	RedisModule_Log(ctx, "notice", "Maximum number of OpenMP threads set to %"PRIu64, ompThreadCount);

	bool cmd_info_enabled = false;
	if(Config_Option_get(Config_CMD_INFO, &cmd_info_enabled) && cmd_info_enabled) {
		uint64_t info_max_query_count = 0;
		Config_Option_get(Config_CMD_INFO_MAX_QUERY_COUNT, &info_max_query_count);
		RedisModule_Log(ctx, "notice", "Query backlog size: %"PRIu64, info_max_query_count);
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

	// initialize LAGraph
	char msg [LAGRAPH_MSG_LEN];
	res = LAGr_Init(GrB_NONBLOCKING, RedisModule_Alloc, RedisModule_Calloc,
			RedisModule_Realloc, RedisModule_Free, msg);

	if(res != GrB_SUCCESS) {
		RedisModule_Log(ctx, "warning", "Encountered error initializing LAGraph: %s", msg);
		return REDISMODULE_ERR;
	}

	return REDISMODULE_OK;
}

int RedisModule_OnLoad
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	if(RedisModule_Init(ctx, "graph", REDISGRAPH_MODULE_VERSION,
						REDISMODULE_APIVER_1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	// initialize GraphBLAS
	int res = GraphBLAS_Init(ctx);
	if(res != REDISMODULE_OK) return res;

	roaring_init_memory_hook((roaring_memory_t) {
		.free           = rm_free,
		.malloc         = rm_malloc,
		.calloc         = rm_calloc,
		.realloc        = rm_realloc,
		.aligned_free   = rm_free,
		.aligned_malloc = rm_aligned_malloc
	});

	// validate minimum redis-server version
	if(!Redis_Version_GreaterOrEqual(MIN_REDIS_VERSION_MAJOR,
				MIN_REDIS_VERSION_MINOR, MIN_REDIS_VERSION_PATCH)) {
		RedisModule_Log(ctx, "warning",
				"FalkorDB requires redis-server version %d.%d.%d and up",
				MIN_REDIS_VERSION_MAJOR, MIN_REDIS_VERSION_MINOR,
				MIN_REDIS_VERSION_PATCH);
		return REDISMODULE_ERR;
	}

	if(RediSearch_Init(ctx, REDISEARCH_INIT_LIBRARY) != REDISMODULE_OK) {
		return REDISMODULE_ERR;
	}

	RedisModule_Log(ctx, "notice", "Starting up FalkorDB version %d.%d.%d.",
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

	if(!UDFCtx_Init())                return REDISMODULE_ERR;
	if(!UDF_RepoInit())               return REDISMODULE_ERR;
	UDF_InitClasses () ;

	if(!ErrorCtx_Init())              return REDISMODULE_ERR;
	if(!ThreadPool_Init())            return REDISMODULE_ERR;
	if(!Indexer_Init())               return REDISMODULE_ERR;
	if(!AST_ValidationsMappingInit()) return REDISMODULE_ERR;

	init_acl_admin_username(ctx);  // set ACL ADMIN username

	RedisModule_Log(ctx, "notice", "Thread pool created, using %d threads.",
			ThreadPool_ThreadCount());

	uint64_t ompThreadCount;
	Config_Option_get(Config_OPENMP_NTHREAD, &ompThreadCount);

	if(GxB_set(GxB_NTHREADS, ompThreadCount) != GrB_SUCCESS) {
		RedisModule_Log(ctx, "warning",
				"Failed to set OpenMP thread count to %" PRIu64, ompThreadCount);
		return REDISMODULE_ERR;
	}

	// log configuration
	_Print_Config(ctx);

	if(_RegisterDataTypes(ctx) != REDISMODULE_OK) return REDISMODULE_ERR;

	if(RedisModule_CreateCommand(ctx,
				"graph.QUERY",
				CommandDispatch,
				"write deny-oom deny-script blocking",
				1, 1, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.RO_QUERY",
				CommandDispatch,
				"readonly deny-script blocking",
				1, 1, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.DELETE",
				Graph_Delete,
				"write deny-script",
				1, 1, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.EXPLAIN",
				CommandDispatch,
				"write deny-oom deny-script",
				1, 1, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.PROFILE",
				CommandDispatch,
				"write deny-oom deny-script blocking",
				1, 1, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.BULK",
				Graph_BulkInsert,
				"write deny-oom deny-script",
				1, 1, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.CONSTRAINT",
				Graph_Constraint,
				"write deny-oom deny-script",
				2, 2, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.SLOWLOG",
				Graph_Slowlog,
				"readonly deny-script allow-busy",
				1, 1, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.CONFIG",
				Graph_Config,
				"readonly deny-script allow-busy",
				0, 0, 0) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.LIST",
				Graph_List,
				"readonly deny-script allow-busy",
				0, 0, 0) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.DEBUG",
				Graph_Debug,
				"readonly deny-script",
				0, 0, 0) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.INFO",
				Graph_Info,
				"readonly deny-script allow-busy",
				1, 1, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.EFFECT",
				Graph_Effect,
				"write deny-script",
				1, 1, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.COPY",
				Graph_Copy,
				"write deny-oom deny-script",
				1, 2, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.RESTORE",
				Graph_Restore,
				"write deny-oom deny-script",
				1, 1, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(init_cmd_acl(ctx) == REDISMODULE_OK) {
		if(RedisModule_CreateCommand(ctx,
					"graph.ACL",
					Graph_ACL,
					"write deny-oom deny-script",
					0, 0, 0) == REDISMODULE_ERR) {
			return REDISMODULE_ERR;
		}
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.PASSWORD",
				Graph_SetPassword,
				"write deny-oom deny-script",
				0, 0, 0) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.MEMORY",
				Graph_Memory,
				"readonly deny-script",
				2, 2, 1) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_CreateCommand(ctx,
				"graph.UDF",
				Graph_UDF,
				"deny-script",
				0, 0, 0) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(BoltApi_Register(ctx) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	// set up global variables scoped to the entire module
	Globals_Init();

	setupCrashHandlers(ctx);

	return REDISMODULE_OK;
}

