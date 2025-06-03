/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "config.h"
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include "util/rmalloc.h"
#include "reconf_handler.h"
#include "util/redis_version.h"
#include "../deps/GraphBLAS/Include/GraphBLAS.h"

// configuration object
typedef struct {
	uint64_t timeout;                  // the timeout for each query in milliseconds
	uint64_t timeout_default;          // default timeout for read and write queries
	uint64_t timeout_max;              // max timeout that can be enforced
	uint64_t cache_size;               // the cache size for each thread, per graph
	bool async_delete;                 // if true, graph deletion is done asynchronously
	uint64_t omp_thread_count;         // maximum number of OpenMP threads
	uint64_t thread_pool_size;         // thread count for thread pool
	uint64_t resultset_size;           // resultset maximum size, UINT64_MAX unlimited
	uint64_t vkey_entity_count;        // the limit of number of entities encoded at once for each RDB key
	uint64_t max_queued_queries;       // max number of queued queries
	int64_t query_mem_capacity;        // max mem(bytes) that query/thread can utilize at any given time
	int64_t delta_max_pending_changes; // number of pending changed before Delta_Matrix flushed
	uint64_t node_creation_buffer;     // number of extra node creations to buffer as margin in matrices
	bool cmd_info;                     // if true, the GRAPH.INFO is enabled
	uint64_t effects_threshold;        // replicate via effects when runtime exceeds threshold
	uint64_t max_info_queries_count;   // maximum number of query info elements
	int16_t bolt_port;                 // bolt protocol port
	bool delay_indexing;               // delay index construction when decoding
	char *import_folder;               // path to import folder, used for CSV loading
	bool deduplicate_strings;          // use string pool to deduplicate strings
} RG_Config;

// global module configuration
RG_Config config = {
	.timeout                   = CONFIG_TIMEOUT_DEFAULT,
	.timeout_default           = CONFIG_TIMEOUT_DEFAULT,
	.timeout_max               = CONFIG_TIMEOUT_DEFAULT,
	.cache_size                = CACHE_SIZE_DEFAULT,
	.async_delete              = ASYNC_DELETE_DEFAULT,
	.omp_thread_count          = OMP_THREAD_COUNT_DEFAULT,
	.thread_pool_size          = THREAD_COUNT_DEFAULT,
	.resultset_size            = RESULTSET_SIZE_DEFAULT,
	.vkey_entity_count         = VKEY_MAX_ENTITY_COUNT_DEFAULT,
	.max_queued_queries        = QUEUED_QUERIES_DEFAULT,
	.query_mem_capacity        = QUERY_MEM_CAPACITY_DEFAULT,
	.delta_max_pending_changes = DELTA_MAX_PENDING_CHANGES_DEFAULT,
	.node_creation_buffer      = NODE_CREATION_BUFFER_DEFAULT,
	.cmd_info                  = CMD_INFO_DEFAULT,
	.effects_threshold         = EFFECTS_THRESHOLD_DEFAULT,
	.max_info_queries_count    = CMD_INFO_QUERIES_MAX_COUNT_DEFAULT,
	.bolt_port                 = BOLT_PROTOCOL_PORT_DEFAULT,
	.delay_indexing            = DELAY_INDEXING_DEFAULT,
	.import_folder             = IMPORT_DIR_DEFAULT,
	.deduplicate_strings       = DEDUPLICATE_STRINGS_DEFAULT,
};

//------------------------------------------------------------------------------
// Config access functions
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// max queued queries
//------------------------------------------------------------------------------

static bool Config_max_queued_queries_set
(
	uint64_t max_queued_queries
) {
	config.max_queued_queries = max_queued_queries;
	return true;
}

static uint Config_max_queued_queries_get(void) {
	return config.max_queued_queries;
}

//------------------------------------------------------------------------------
// timeout
//------------------------------------------------------------------------------

static bool Config_timeout_set
(
	uint64_t timeout
) {
	config.timeout = timeout;
	return true;
}

// check if new(TIMEOUT_DEFAULT or TIMEOUT_MAX) are used
// log a deprecation message
static bool _Config_check_if_new_timeout_used() {
	bool new_timeout_set  = config.timeout_default != CONFIG_TIMEOUT_NO_TIMEOUT;
	new_timeout_set      |= config.timeout_max     != CONFIG_TIMEOUT_NO_TIMEOUT;

	if(new_timeout_set) {
		RedisModule_Log(NULL, "warning", "The TIMEOUT configuration parameter is deprecated. Please set TIMEOUT_MAX and TIMEOUT_DEFAULT instead");
		return false;
	}

	return true;
}

static bool Config_enforce_timeout_max
(
	uint64_t timeout_default,
	uint64_t timeout_max
) {
	if(timeout_max != CONFIG_TIMEOUT_NO_TIMEOUT &&
	   timeout_default > timeout_max) {
#ifdef __aarch64__
		RedisModule_Log(NULL, "warning", "The TIMEOUT_DEFAULT(%lld) configuration parameter value is higher than TIMEOUT_MAX(%lld).", timeout_default, timeout_max);
#else
		RedisModule_Log(NULL, "warning", "The TIMEOUT_DEFAULT(%ld) configuration parameter value is higher than TIMEOUT_MAX(%ld).", timeout_default, timeout_max);
#endif
		return false;
	}

	config.timeout_max     = timeout_max;
	config.timeout_default = timeout_default;
	return true;
}

static bool Config_timeout_default_set
(
	uint64_t timeout_default
) {
	return Config_enforce_timeout_max(timeout_default, config.timeout_max);
}

static bool Config_timeout_max_set
(
	uint64_t timeout_max
) {
	return Config_enforce_timeout_max(config.timeout_default, timeout_max);
}

static uint Config_timeout_get(void) {
	return config.timeout;
}

static uint Config_timeout_default_get(void) {
	return config.timeout_default;
}

static uint Config_timeout_max_get(void) {
	return config.timeout_max;
}

//------------------------------------------------------------------------------
// thread count
//------------------------------------------------------------------------------

static bool Config_thread_pool_size_set
(
	uint64_t nthreads
) {
	config.thread_pool_size = nthreads;
	return true;
}

static uint64_t Config_thread_pool_size_get(void) {
	return config.thread_pool_size;
}

//------------------------------------------------------------------------------
// OpenMP thread count
//------------------------------------------------------------------------------

static bool Config_omp_thread_count_set(uint64_t nthreads) {
	config.omp_thread_count = nthreads;
	return true;
}

static uint64_t Config_omp_thread_count_get(void) {
	return config.omp_thread_count;
}

//------------------------------------------------------------------------------
// virtual key entity count
//------------------------------------------------------------------------------

static bool Config_vkey_entity_count_set
(
	uint64_t entity_count
) {
	config.vkey_entity_count = entity_count;
	return true;
}

static uint64_t Config_vkey_entity_count_get(void) {
	return config.vkey_entity_count;
}

//------------------------------------------------------------------------------
// cache size
//------------------------------------------------------------------------------

static bool Config_cache_size_set
(
	uint64_t cache_size
) {
	config.cache_size = cache_size;
	return true;
}

static uint64_t Config_cache_size_get(void) {
	return config.cache_size;
}

//------------------------------------------------------------------------------
// async delete
//------------------------------------------------------------------------------

static bool Config_async_delete_set
(
	bool async_delete
) {
	config.async_delete = async_delete;
	return true;
}

static bool Config_async_delete_get(void) {
	return config.async_delete;
}

//------------------------------------------------------------------------------
// result-set max size
//------------------------------------------------------------------------------

static bool Config_resultset_size_set
(
	int64_t max_size
) {
	if(max_size < 0) config.resultset_size = RESULTSET_SIZE_UNLIMITED;
	else config.resultset_size = max_size;

	return true;
}

static uint64_t Config_resultset_size_get(void) {
	return config.resultset_size;
}

//------------------------------------------------------------------------------
// query mem capacity
//------------------------------------------------------------------------------

static bool Config_query_mem_capacity_set
(
	int64_t capacity
) {
	if(capacity <= 0)
		config.query_mem_capacity = QUERY_MEM_CAPACITY_UNLIMITED;
	else
		config.query_mem_capacity = capacity;

	return true;
}

static uint64_t Config_query_mem_capacity_get(void) {
	return config.query_mem_capacity;
}

//------------------------------------------------------------------------------
// delta max pending changes
//------------------------------------------------------------------------------

static bool Config_delta_max_pending_changes_set
(
	int64_t capacity
) {
	if(capacity == 0)
		config.delta_max_pending_changes = DELTA_MAX_PENDING_CHANGES_DEFAULT;
	else
		config.delta_max_pending_changes = capacity;

	return true;
}

static uint64_t Config_delta_max_pending_changes_get(void) {
	return config.delta_max_pending_changes;
}

//------------------------------------------------------------------------------
// node creation buffer
//------------------------------------------------------------------------------

static bool Config_node_creation_buffer_set
(
	long long buf_size
) {
	// node_creation_buffer should be at-least 128
	buf_size = (buf_size < 128) ? 128: buf_size;

	// retrieve the MSB of the value
	long long msb = (sizeof(long long) * 8) - __builtin_clzll(buf_size);
	long long set_msb = 1 << (msb - 1);

	// if the value is not a power of 2
	// (if any bits other than the MSB are 1),
	// raise it to the next power of 2
	if((~set_msb & buf_size) != 0) {
		buf_size = 1 << msb;
	}

	config.node_creation_buffer = buf_size;
	return true;
}

static uint64_t Config_node_creation_buffer_get(void) {
	return config.node_creation_buffer;
}

//------------------------------------------------------------------------------
// cmd info
//------------------------------------------------------------------------------

static bool Config_cmd_info_get(void) {
	return config.cmd_info;
}

static bool Config_cmd_info_set
(
	const bool cmd_info
) {
	config.cmd_info = cmd_info;
	return true;
}

static uint64_t Config_max_info_queries_count_get(void) {
	return config.max_info_queries_count;
}

static bool Config_max_info_queries_count_set
(
	const uint64_t count
) {
	if (count > CMD_INFO_QUERIES_MAX_COUNT_DEFAULT) {
		config.max_info_queries_count = CMD_INFO_QUERIES_MAX_COUNT_DEFAULT;
	} else {
		config.max_info_queries_count = count;
	}

	return true;
}

//------------------------------------------------------------------------------
// effects threshold
//------------------------------------------------------------------------------

static bool Config_effects_threshold_set
(
	uint64_t threshold
) {
	config.effects_threshold = threshold;
	return true;
}

static uint64_t Config_effects_threshold_get (void) {
	return config.effects_threshold;
}

//------------------------------------------------------------------------------
// bolt protocol port
//------------------------------------------------------------------------------

static bool Config_bolt_port_set
(
	int16_t port
) {
	int16_t p = (port < 0) ? BOLT_PROTOCOL_PORT_DEFAULT : port;
	config.bolt_port = p;
	return true;
}

static int16_t Config_bolt_port_get(void) {
	return config.bolt_port;
}

//------------------------------------------------------------------------------
// delay indexing
//------------------------------------------------------------------------------

static bool Config_delay_indexing_get(void) {
	return config.delay_indexing;
}

static bool Config_delay_indexing_set
(
	const bool delay_indexing
) {
	config.delay_indexing = delay_indexing;
	return true;
}

//------------------------------------------------------------------------------
// import folder
//------------------------------------------------------------------------------

static bool Config_import_folder_set
(
	const char *path
) {
	ASSERT(path != NULL);

	// free previous value
	if(strcmp(config.import_folder, IMPORT_DIR_DEFAULT) != 0) {
		rm_free(config.import_folder);
	}

	// copy new path
	config.import_folder = rm_strdup(path);

	return true;
}

static const char *Config_import_folder_get(void) {
	return config.import_folder;
}

//------------------------------------------------------------------------------
// deduplicate strings
//------------------------------------------------------------------------------

static bool Config_deduplicate_strings_set
(
	bool enabled
) {
	config.deduplicate_strings = enabled;
	return true;
}

static bool Config_deduplicate_strings_get(void) {
	return config.deduplicate_strings;
}

// generate config get function name for an individual configuration attribute
#define CONFIG_GET_BOOL_FUNC_NAME(config_attr) _Config_Get_##config_attr##_Bool
#define CONFIG_GET_STRING_FUNC_NAME(config_attr) _Config_Get_##config_attr##_String
#define CONFIG_GET_NUMERIC_FUNC_NAME(config_attr) _Config_Get_##config_attr##_Numeric

// generate a config get boolean function
// return the boolean value of the configuration key
#define CONFIG_GET_BOOL(config_attr)                          \
int CONFIG_GET_BOOL_FUNC_NAME(config_attr)                    \
(                                                             \
	const char *name,                                         \
	void *privdata                                            \
) {                                                           \
	return Config_##config_attr##_get();                      \
}

#define CONFIG_GET_NUMERIC(config_attr)                       \
long long CONFIG_GET_NUMERIC_FUNC_NAME(config_attr)           \
(                                                             \
	const char *name,                                         \
	void *privdata                                            \
) {                                                           \
	return Config_##config_attr##_get();                      \
}

#define CONFIG_GET_STRING(config_attr)                        \
RedisModuleString* CONFIG_GET_STRING_FUNC_NAME(config_attr)   \
(                                                             \
	const char *name,                                         \
	void *privdata                                            \
) {                                                           \
	const char *v = Config_##config_attr##_get();             \
	return RedisModule_CreateString(NULL,  v, strlen(v));     \
}

//------------------------------------------------------------------------------
// create config numeric value getters
//------------------------------------------------------------------------------

CONFIG_GET_BOOL (cmd_info)
CONFIG_GET_BOOL (async_delete)
CONFIG_GET_BOOL (delay_indexing)
CONFIG_GET_BOOL (deduplicate_strings)

CONFIG_GET_NUMERIC (timeout)
CONFIG_GET_NUMERIC (bolt_port)
CONFIG_GET_NUMERIC (cache_size)
CONFIG_GET_NUMERIC (timeout_max)
CONFIG_GET_NUMERIC (resultset_size)
CONFIG_GET_NUMERIC (timeout_default)
CONFIG_GET_NUMERIC (thread_pool_size)
CONFIG_GET_NUMERIC (omp_thread_count)
CONFIG_GET_NUMERIC (vkey_entity_count)
CONFIG_GET_NUMERIC (effects_threshold)
CONFIG_GET_NUMERIC (max_queued_queries)
CONFIG_GET_NUMERIC (query_mem_capacity)
CONFIG_GET_NUMERIC (node_creation_buffer)
CONFIG_GET_NUMERIC (max_info_queries_count)
CONFIG_GET_NUMERIC (delta_max_pending_changes)

CONFIG_GET_STRING (import_folder)

// check if field is a valid configuration option
bool Config_Contains_field
(
	const char *field_str,      // configuration option name
	Config_Option_Field *field  // [out] configuration field
) {
	ASSERT(field_str != NULL);

	Config_Option_Field f;

	if(!strcasecmp(field_str, THREAD_COUNT)) {
		f = Config_THREAD_POOL_SIZE;
	} else if(!strcasecmp(field_str, TIMEOUT)) {
		f = Config_TIMEOUT;
	} else if(!(strcasecmp(field_str, TIMEOUT_DEFAULT))) {
		f = Config_TIMEOUT_DEFAULT;
	} else if(!(strcasecmp(field_str, TIMEOUT_MAX))) {
		f = Config_TIMEOUT_MAX;
	} else if(!strcasecmp(field_str, OMP_THREAD_COUNT)) {
		f = Config_OPENMP_NTHREAD;
	} else if(!strcasecmp(field_str, VKEY_MAX_ENTITY_COUNT)) {
		f = Config_VKEY_MAX_ENTITY_COUNT;
	} else if(!(strcasecmp(field_str, CACHE_SIZE))) {
		f = Config_CACHE_SIZE;
	} else if(!(strcasecmp(field_str, RESULTSET_SIZE))) {
		f = Config_RESULTSET_MAX_SIZE;
	} else if(!(strcasecmp(field_str, MAX_QUEUED_QUERIES))) {
		f = Config_MAX_QUEUED_QUERIES;
	} else if(!(strcasecmp(field_str, QUERY_MEM_CAPACITY))) {
		f = Config_QUERY_MEM_CAPACITY;
	} else if(!(strcasecmp(field_str, DELTA_MAX_PENDING_CHANGES))) {
		f = Config_DELTA_MAX_PENDING_CHANGES;
	} else if(!(strcasecmp(field_str, NODE_CREATION_BUFFER))) {
		f = Config_NODE_CREATION_BUFFER;
	} else if(!(strcasecmp(field_str, ASYNC_DELETE))) {
		f = Config_ASYNC_DELETE;
	} else if(!(strcasecmp(field_str, CMD_INFO))) {
		f = Config_CMD_INFO;
	} else if(!(strcasecmp(field_str, CMD_INFO_MAX_QUERIES_COUNT_OPTION_NAME))) {
		f = Config_CMD_INFO_MAX_QUERY_COUNT;
	} else if (!(strcasecmp(field_str, EFFECTS_THRESHOLD))) {
		f = Config_EFFECTS_THRESHOLD;
	} else if (!(strcasecmp(field_str, BOLT_PORT))) {
		f = Config_BOLT_PORT;
	} else if (!(strcasecmp(field_str, DELAY_INDEXING))) {
		f = Config_DELAY_INDEXING;
	} else if (!(strcasecmp(field_str, IMPORT_FOLDER))) {
		f = Config_IMPORT_FOLDER;
	} else if (!(strcasecmp(field_str, DEDUPLICATE_STRINGS))) {
		f = Config_DEDUPLICATE_STRINGS;
	} else {
		return false;
	}

	if(field) *field = f;
	return true;
}

// returns the field type
SIType Config_Field_type
(
	Config_Option_Field field  // field
) {
	switch(field) {
		case Config_TIMEOUT:
			return T_INT64;

		case Config_TIMEOUT_DEFAULT:
			return T_INT64;

		case Config_TIMEOUT_MAX:
			return T_INT64;

		case Config_CACHE_SIZE:
			return T_INT64;

		case Config_OPENMP_NTHREAD:
			return T_INT64;

		case Config_THREAD_POOL_SIZE:
			return T_INT64;

		case Config_RESULTSET_MAX_SIZE:
			return T_INT64;

		case Config_VKEY_MAX_ENTITY_COUNT:
			return T_INT64;

		case Config_ASYNC_DELETE:
			return T_BOOL;

		case Config_MAX_QUEUED_QUERIES:
			return T_INT64;

		case Config_QUERY_MEM_CAPACITY:
			return T_INT64;

		case Config_DELTA_MAX_PENDING_CHANGES:
			return T_INT64;

		case Config_NODE_CREATION_BUFFER:
			return T_INT64;

		case Config_CMD_INFO:
			return T_BOOL;

		case Config_CMD_INFO_MAX_QUERY_COUNT:
			return T_INT64;

		case Config_EFFECTS_THRESHOLD:
			return T_INT64;

		case Config_BOLT_PORT:
			return T_INT64;

		case Config_DELAY_INDEXING:
			return T_BOOL;

		case Config_IMPORT_FOLDER:
			return T_STRING;

		case Config_DEDUPLICATE_STRINGS:
			return T_BOOL;	

		//----------------------------------------------------------------------
		// invalid option
		//----------------------------------------------------------------------

		default :
			ASSERT("invalid option field" && false);
			break;
	}

	return T_NULL;
}

// generate config set function name for an individual configuration attribute
#define CONFIG_SET_BOOL_FUNC_NAME(config_attr) _Config_Set_##config_attr##_Bool
#define CONFIG_SET_NUMERIC_FUNC_NAME(config_attr) _Config_Set_##config_attr##_Numeric
#define CONFIG_SET_STRING_FUNC_NAME(config_attr) _Config_Set_##config_attr##_String

// generate a config set boolean function
// sets the boolean value of the configuration key
#define CONFIG_SET_BOOL(config_attr)                                   \
int CONFIG_SET_BOOL_FUNC_NAME(config_attr) (                           \
	const char *name,                                                  \
	int val,                                                           \
	void *privdata,                                                    \
	RedisModuleString **err                                            \
) {                                                                    \
	/* call config internal setter */                                  \
	if(Config_##config_attr##_set(val)) {                              \
		return REDISMODULE_OK;                                         \
	}                                                                  \
                                                                       \
	/* failed to set config */                                         \
	*err = RedisModule_CreateStringPrintf(NULL,                        \
			"Failed to set configuration %s to %d", name, val);        \
   return REDISMODULE_ERR;                                             \
}

#define CONFIG_SET_NUMERIC(config_attr)                                \
int CONFIG_SET_NUMERIC_FUNC_NAME(config_attr) (                        \
	const char *name,                                                  \
	long long val,                                                     \
	void *privdata,                                                    \
	RedisModuleString **err                                            \
) {                                                                    \
	/* call config internal setter */                                  \
	if(Config_##config_attr##_set(val)) {                              \
		return REDISMODULE_OK;                                         \
	}                                                                  \
                                                                       \
	/* failed to set config */                                         \
	*err = RedisModule_CreateStringPrintf(NULL,                        \
			"Failed to set configuration %s to %lld", name, val);      \
   return REDISMODULE_ERR;                                             \
}

#define CONFIG_SET_STRING(config_attr)                                 \
int CONFIG_SET_STRING_FUNC_NAME(config_attr) (                         \
	const char *name,                                                  \
	RedisModuleString *val,                                            \
	void *privdata,                                                    \
	RedisModuleString **err                                            \
) {                                                                    \
	const char *value = RedisModule_StringPtrLen(val, NULL);           \
                                                                       \
	/* call config internal setter */                                  \
	if(Config_##config_attr##_set(value)) {                            \
		return REDISMODULE_OK;                                         \
	}                                                                  \
                                                                       \
	/* failed to set config */                                         \
	*err = RedisModule_CreateStringPrintf(NULL,                        \
			"Failed to set configuration %s to %s", name, value);      \
   return REDISMODULE_ERR;                                             \
}

//------------------------------------------------------------------------------
// create config numeric value setters
//------------------------------------------------------------------------------

CONFIG_SET_BOOL (cmd_info)
CONFIG_SET_BOOL (async_delete)
CONFIG_SET_BOOL (delay_indexing)
CONFIG_SET_BOOL (deduplicate_strings)

CONFIG_SET_NUMERIC (timeout)
CONFIG_SET_NUMERIC (bolt_port)
CONFIG_SET_NUMERIC (cache_size)
CONFIG_SET_NUMERIC (timeout_max)
CONFIG_SET_NUMERIC (resultset_size)
CONFIG_SET_NUMERIC (timeout_default)
CONFIG_SET_NUMERIC (thread_pool_size)
CONFIG_SET_NUMERIC (omp_thread_count)
CONFIG_SET_NUMERIC (vkey_entity_count)
CONFIG_SET_NUMERIC (effects_threshold)
CONFIG_SET_NUMERIC (max_queued_queries)
CONFIG_SET_NUMERIC (query_mem_capacity)
CONFIG_SET_NUMERIC (node_creation_buffer)
CONFIG_SET_NUMERIC (max_info_queries_count)
CONFIG_SET_NUMERIC (delta_max_pending_changes)

CONFIG_SET_STRING (import_folder)

static void _Config_Register
(
	RedisModuleCtx *ctx
) {
	//--------------------------------------------------------------------------
	// register graph configurations
	//--------------------------------------------------------------------------

	//--------------------------------------------------------------------------
	// register boolean configurations
	//--------------------------------------------------------------------------

	int res;

	res = RedisModule_RegisterBoolConfig(ctx,
			CMD_INFO,
			Config_cmd_info_get(),
			REDISMODULE_CONFIG_DEFAULT,
			CONFIG_GET_BOOL_FUNC_NAME(cmd_info),
			CONFIG_SET_BOOL_FUNC_NAME(cmd_info),
			reconf_cmd_info_apply,
			NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterBoolConfig(ctx,
			ASYNC_DELETE,
			Config_async_delete_get(),
			REDISMODULE_CONFIG_DEFAULT,
			CONFIG_GET_BOOL_FUNC_NAME(async_delete),
			CONFIG_SET_BOOL_FUNC_NAME(async_delete),
			NULL,
			NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterBoolConfig(ctx,
			DEDUPLICATE_STRINGS,
			Config_deduplicate_strings_get(),
			REDISMODULE_CONFIG_DEFAULT,
			CONFIG_GET_BOOL_FUNC_NAME(deduplicate_strings),
			CONFIG_SET_BOOL_FUNC_NAME(deduplicate_strings),
			reconf_deduplicate_strings_apply,
			NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterBoolConfig(ctx,
			DELAY_INDEXING,
			Config_delay_indexing_get(),
			REDISMODULE_CONFIG_DEFAULT,
			CONFIG_GET_BOOL_FUNC_NAME(delay_indexing),
			CONFIG_SET_BOOL_FUNC_NAME(delay_indexing),
			NULL,
			NULL);
	ASSERT(res == REDISMODULE_OK);

	//--------------------------------------------------------------------------
	// register numeric configurations
	//--------------------------------------------------------------------------

	res = RedisModule_RegisterNumericConfig(ctx,
										  TIMEOUT,
										  Config_timeout_get(),
										  REDISMODULE_CONFIG_DEFAULT,
										  CONFIG_TIMEOUT_MIN,
										  CONFIG_TIMEOUT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(timeout),
										  CONFIG_SET_NUMERIC_FUNC_NAME(timeout),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  BOLT_PORT,
										  Config_bolt_port_get(),
										  REDISMODULE_CONFIG_IMMUTABLE,
										  BOLT_PROTOCOL_PORT_MIN,
										  BOLT_PROTOCOL_PORT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(bolt_port),
										  CONFIG_SET_NUMERIC_FUNC_NAME(bolt_port),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  CACHE_SIZE,
										  Config_cache_size_get(),
										  REDISMODULE_CONFIG_IMMUTABLE,
										  CACHE_SIZE_MIN,
										  CACHE_SIZE_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(cache_size),
										  CONFIG_SET_NUMERIC_FUNC_NAME(cache_size),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  TIMEOUT_MAX,
										  Config_timeout_max_get(),
										  REDISMODULE_CONFIG_DEFAULT,
										  CONFIG_TIMEOUT_MIN,
										  CONFIG_TIMEOUT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(timeout_max),
										  CONFIG_SET_NUMERIC_FUNC_NAME(timeout_max),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  RESULTSET_SIZE,
										  Config_resultset_size_get(),
										  REDISMODULE_CONFIG_DEFAULT,
										  RESULTSET_SIZE_MIN,
										  RESULTSET_SIZE_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(resultset_size),
										  CONFIG_SET_NUMERIC_FUNC_NAME(resultset_size),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  TIMEOUT_DEFAULT,
										  Config_timeout_default_get(),
										  REDISMODULE_CONFIG_DEFAULT,
										  CONFIG_TIMEOUT_MIN,
										  CONFIG_TIMEOUT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(timeout_default),
										  CONFIG_SET_NUMERIC_FUNC_NAME(timeout_default),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	// the thread pool's default size is equal to the system's number of cores
	int CPUCount = sysconf(_SC_NPROCESSORS_ONLN);
	CPUCount = (CPUCount != -1) ? CPUCount : 1;

	res = RedisModule_RegisterNumericConfig(ctx,
										  THREAD_COUNT,
										  //THREAD_COUNT_DEFAULT,
										  CPUCount,
										  REDISMODULE_CONFIG_IMMUTABLE,
										  THREAD_COUNT_MIN,
										  THREAD_COUNT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(thread_pool_size),
										  CONFIG_SET_NUMERIC_FUNC_NAME(thread_pool_size),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	// use the GraphBLAS-defined number of OpenMP threads by default
	uint64_t omp_thread_count;
	GxB_get(GxB_NTHREADS, &omp_thread_count);

	res = RedisModule_RegisterNumericConfig(ctx,
										  OMP_THREAD_COUNT,
										  //OMP_THREAD_COUNT_DEFAULT,
										  omp_thread_count,
										  REDISMODULE_CONFIG_IMMUTABLE,
										  OMP_THREAD_COUNT_MIN,
										  OMP_THREAD_COUNT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(omp_thread_count),
										  CONFIG_SET_NUMERIC_FUNC_NAME(omp_thread_count),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  VKEY_MAX_ENTITY_COUNT,
										  Config_vkey_entity_count_get(),
										  REDISMODULE_CONFIG_DEFAULT,
										  VKEY_MAX_ENTITY_COUNT_MIN,
										  VKEY_MAX_ENTITY_COUNT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(vkey_entity_count),
										  CONFIG_SET_NUMERIC_FUNC_NAME(vkey_entity_count),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  EFFECTS_THRESHOLD,
										  Config_effects_threshold_get(),
										  REDISMODULE_CONFIG_DEFAULT,
										  EFFECTS_THRESHOLD_MIN,
										  EFFECTS_THRESHOLD_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(effects_threshold),
										  CONFIG_SET_NUMERIC_FUNC_NAME(effects_threshold),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  MAX_QUEUED_QUERIES,
										  Config_max_queued_queries_get(),
										  REDISMODULE_CONFIG_DEFAULT,
										  QUEUED_QUERIES_MIN,
										  QUEUED_QUERIES_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(max_queued_queries),
										  CONFIG_SET_NUMERIC_FUNC_NAME(max_queued_queries),
										  reconf_max_queued_queries_apply,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  QUERY_MEM_CAPACITY,
										  Config_query_mem_capacity_get(),
										  REDISMODULE_CONFIG_DEFAULT,
										  QUERY_MEM_CAPACITY_MIN,
										  QUERY_MEM_CAPACITY_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(query_mem_capacity),
										  CONFIG_SET_NUMERIC_FUNC_NAME(query_mem_capacity),
										  reconf_query_mem_cap_apply,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  NODE_CREATION_BUFFER,
										  Config_node_creation_buffer_get(),
										  REDISMODULE_CONFIG_IMMUTABLE,
										  NODE_CREATION_BUFFER_MIN,
										  NODE_CREATION_BUFFER_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(node_creation_buffer),
										  CONFIG_SET_NUMERIC_FUNC_NAME(node_creation_buffer),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  CMD_INFO_MAX_QUERIES_COUNT_OPTION_NAME,
										  Config_max_info_queries_count_get(),
										  REDISMODULE_CONFIG_DEFAULT,
										  CMD_INFO_QUERIES_MAX_COUNT_MIN,
										  CMD_INFO_QUERIES_MAX_COUNT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(max_info_queries_count),
										  CONFIG_SET_NUMERIC_FUNC_NAME(max_info_queries_count),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  DELTA_MAX_PENDING_CHANGES,
										  Config_delta_max_pending_changes_get(),
										  REDISMODULE_CONFIG_DEFAULT,
										  DELTA_MAX_PENDING_CHANGES_MIN,
										  DELTA_MAX_PENDING_CHANGES_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(delta_max_pending_changes),
										  CONFIG_SET_NUMERIC_FUNC_NAME(delta_max_pending_changes),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	//--------------------------------------------------------------------------
	// register string configurations
	//--------------------------------------------------------------------------

	res = RedisModule_RegisterStringConfig(ctx,
										  IMPORT_FOLDER,
										  Config_import_folder_get(),
										  REDISMODULE_CONFIG_IMMUTABLE,
										  CONFIG_GET_STRING_FUNC_NAME(import_folder),
										  CONFIG_SET_STRING_FUNC_NAME(import_folder),
										  NULL,
										  NULL);
}

bool Config_Option_get
(
	Config_Option_Field field,
	...
) {
	//--------------------------------------------------------------------------
	// get the option
	//--------------------------------------------------------------------------

	va_list ap;

	switch(field) {
		case Config_MAX_QUEUED_QUERIES: {
			va_start(ap, field);
			uint64_t *max_queued_queries = va_arg(ap, uint64_t *);
			va_end(ap);

			ASSERT(max_queued_queries != NULL);
			(*max_queued_queries) = Config_max_queued_queries_get();
		}
		break;

		//----------------------------------------------------------------------
		// timeout
		//----------------------------------------------------------------------

		case Config_TIMEOUT: {
			 va_start(ap, field);
			 uint64_t *timeout = va_arg(ap, uint64_t *);
			 va_end(ap);

			 ASSERT(timeout != NULL);
			 (*timeout) = Config_timeout_get();
		 }
		 break;

		//----------------------------------------------------------------------
		// timeout default
		//----------------------------------------------------------------------

		case Config_TIMEOUT_DEFAULT: {
			 va_start(ap, field);
			 uint64_t *timeout_default = va_arg(ap, uint64_t *);
			 va_end(ap);

			 ASSERT(timeout_default != NULL);
			 (*timeout_default) = Config_timeout_default_get();
		 }
		 break;

		//----------------------------------------------------------------------
		// timeout max
		//----------------------------------------------------------------------

		case Config_TIMEOUT_MAX: {
			 va_start(ap, field);
			 uint64_t *timeout_max = va_arg(ap, uint64_t *);
			 va_end(ap);

			 ASSERT(timeout_max != NULL);
			 (*timeout_max) = Config_timeout_max_get();
		 }
		 break;

		//----------------------------------------------------------------------
		// cache size
		//----------------------------------------------------------------------

		case Config_CACHE_SIZE: {
			va_start(ap, field);
			uint64_t *cache_size = va_arg(ap, uint64_t *);
			va_end(ap);

			ASSERT(cache_size != NULL);
			(*cache_size) = Config_cache_size_get();
		}
		break;

		//----------------------------------------------------------------------
		// OpenMP thread count
		//----------------------------------------------------------------------

		case Config_OPENMP_NTHREAD: {
			va_start(ap, field);
			uint64_t *omp_nthreads = va_arg(ap, uint64_t *);
			va_end(ap);

			ASSERT(omp_nthreads != NULL);
			(*omp_nthreads) = Config_omp_thread_count_get();
		}
		break;

		//----------------------------------------------------------------------
		// thread-pool size
		//----------------------------------------------------------------------

		case Config_THREAD_POOL_SIZE: {
			va_start(ap, field);
			uint64_t *pool_nthreads = va_arg(ap, uint64_t *);
			va_end(ap);

			ASSERT(pool_nthreads != NULL);
			(*pool_nthreads) = Config_thread_pool_size_get();
		}
		break;

		//----------------------------------------------------------------------
		// result-set size
		//----------------------------------------------------------------------

		case Config_RESULTSET_MAX_SIZE: {
			va_start(ap, field);
			uint64_t *resultset_max_size = va_arg(ap, uint64_t *);
			va_end(ap);

			ASSERT(resultset_max_size != NULL);
			(*resultset_max_size) = Config_resultset_size_get();
		}
		break;

		//----------------------------------------------------------------------
		// virtual key entity count
		//----------------------------------------------------------------------

		case Config_VKEY_MAX_ENTITY_COUNT: {
			va_start(ap, field);
			uint64_t *vkey_max_entity_count = va_arg(ap, uint64_t *);
			va_end(ap);

			ASSERT(vkey_max_entity_count != NULL);
			(*vkey_max_entity_count) = Config_vkey_entity_count_get();
		}
		break;

		//----------------------------------------------------------------------
		// async deleteion
		//----------------------------------------------------------------------

		case Config_ASYNC_DELETE: {
			va_start(ap, field);
			bool *async_delete = va_arg(ap, bool *);
			va_end(ap);

			ASSERT(async_delete != NULL);
			(*async_delete) = Config_async_delete_get();
		}
		break;

		//----------------------------------------------------------------------
		// query mem capacity
		//----------------------------------------------------------------------

		case Config_QUERY_MEM_CAPACITY: {
			va_start(ap, field);
			int64_t *query_mem_capacity = va_arg(ap, int64_t *);
			va_end(ap);

			ASSERT(query_mem_capacity != NULL);
			(*query_mem_capacity) = Config_query_mem_capacity_get();
		}
		break;

		//----------------------------------------------------------------------
		// number of pending changed before Delta_Matrix flushed
		//----------------------------------------------------------------------

		case Config_DELTA_MAX_PENDING_CHANGES: {
			va_start(ap, field);
			int64_t *delta_max_pending_changes = va_arg(ap, int64_t *);
			va_end(ap);

			ASSERT(delta_max_pending_changes != NULL);
			(*delta_max_pending_changes) = Config_delta_max_pending_changes_get();
		}
		break;

		//----------------------------------------------------------------------
		// size of buffer to maintain as margin in matrices
		//----------------------------------------------------------------------

		case Config_NODE_CREATION_BUFFER: {
			va_start(ap, field);
			uint64_t *node_creation_buffer = va_arg(ap, uint64_t *);
			va_end(ap);

			ASSERT(node_creation_buffer != NULL);
			(*node_creation_buffer) = Config_node_creation_buffer_get();
		}
		break;

		//----------------------------------------------------------------------
		// cmd info
		//----------------------------------------------------------------------

		case Config_CMD_INFO: {
			va_start(ap, field);
			bool *cmd_info_on = va_arg(ap, bool *);
			va_end(ap);

			ASSERT(cmd_info_on != NULL);
			(*cmd_info_on) = Config_cmd_info_get();
		}
		break;

		//----------------------------------------------------------------------
		// cmd info maximum queries count
		//----------------------------------------------------------------------

		case Config_CMD_INFO_MAX_QUERY_COUNT: {
			va_start(ap, field);
			uint64_t *count = va_arg(ap, uint64_t *);
			va_end(ap);

			ASSERT(count != NULL);
			(*count) = Config_max_info_queries_count_get();
		  }
		  break;

		//----------------------------------------------------------------------
		// effects threshold
		//----------------------------------------------------------------------

		case Config_EFFECTS_THRESHOLD: {
			va_start(ap, field);
			uint64_t *effects_threshold = va_arg(ap, uint64_t *);
			va_end(ap);

			ASSERT(effects_threshold != NULL);
			(*effects_threshold) = Config_effects_threshold_get();
	   }
	   break;

		//----------------------------------------------------------------------
		// bolt protocol port
		//----------------------------------------------------------------------

		case Config_BOLT_PORT: {
			va_start(ap, field);
			int16_t *bolt_port = va_arg(ap, int16_t *);
			va_end(ap);

			ASSERT(bolt_port != NULL);
			(*bolt_port) = Config_bolt_port_get();
	   }
	   break;

		//----------------------------------------------------------------------
		// delay indexing
		//----------------------------------------------------------------------

		case Config_DELAY_INDEXING: {
			va_start(ap, field);
			bool *delay_indexing = va_arg(ap, bool *);
			va_end(ap);

			ASSERT(delay_indexing != NULL);
			(*delay_indexing) = Config_delay_indexing_get();
		}
		break;

		//----------------------------------------------------------------------
		// import folder path
		//----------------------------------------------------------------------

		case Config_IMPORT_FOLDER: {
			va_start(ap, field);
			const char **import_folder = va_arg(ap, const char **);
			va_end(ap);

			ASSERT(import_folder != NULL);
			(*import_folder) = Config_import_folder_get();
		}
		break;

		//----------------------------------------------------------------------
		// deduplicate strings
		//----------------------------------------------------------------------

		case Config_DEDUPLICATE_STRINGS: {
			va_start(ap, field);
			bool *enabled = va_arg(ap, bool *);
			va_end(ap);

			ASSERT(enabled != NULL);
			(*enabled) = Config_deduplicate_strings_get();
		}
		break;

		//----------------------------------------------------------------------
		// invalid option
		//----------------------------------------------------------------------

		default :
			ASSERT("invalid option field" && false);
			return false;
	}

	return true;
}

int Set_Config_Init_Value(RedisModuleCtx *ctx,
						  Config_Option_Field field,
						  RedisModuleString *value) {
	ASSERT(ctx != NULL);
	ASSERT(value != NULL);

	switch(field) {
		case Config_TIMEOUT:
			if(RedisModule_StringToULongLong(value, &config.timeout) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								TIMEOUT, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;

		case Config_TIMEOUT_DEFAULT:
			if(RedisModule_StringToULongLong(value, &config.timeout_default) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								TIMEOUT_DEFAULT, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;
		
		case Config_TIMEOUT_MAX:
			if(RedisModule_StringToULongLong(value, &config.timeout_max) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								TIMEOUT_MAX, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;
		case Config_CACHE_SIZE:
			if(RedisModule_StringToULongLong(value, &config.cache_size) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								CACHE_SIZE, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;

		case Config_OPENMP_NTHREAD:
			if(RedisModule_StringToULongLong(value, &config.omp_thread_count) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								OMP_THREAD_COUNT, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;

		case Config_THREAD_POOL_SIZE:
			if(RedisModule_StringToULongLong(value, &config.thread_pool_size) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								THREAD_COUNT, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;

		case Config_RESULTSET_MAX_SIZE:
			if(RedisModule_StringToULongLong(value, &config.resultset_size) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								RESULTSET_SIZE, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;

		case Config_VKEY_MAX_ENTITY_COUNT:
			if(RedisModule_StringToULongLong(value, &config.vkey_entity_count) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								VKEY_MAX_ENTITY_COUNT, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;

		case Config_ASYNC_DELETE: {
			unsigned long long v = 0;
			if(RedisModule_StringToULongLong(value, &v) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								ASYNC_DELETE, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			config.async_delete = (v != 0);
			return REDISMODULE_OK;
		}

		case Config_MAX_QUEUED_QUERIES:
			if(RedisModule_StringToULongLong(value, &config.max_queued_queries) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								MAX_QUEUED_QUERIES, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;

		case Config_QUERY_MEM_CAPACITY:
			if(RedisModule_StringToLongLong(value, &config.query_mem_capacity) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								QUERY_MEM_CAPACITY, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;

		case Config_DELTA_MAX_PENDING_CHANGES: {
			unsigned long long v = 0;
			if(RedisModule_StringToULongLong(value, &v) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								DELTA_MAX_PENDING_CHANGES, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			config.delta_max_pending_changes = (int64_t)v;
			return REDISMODULE_OK;
		}

		case Config_NODE_CREATION_BUFFER:
			if(RedisModule_StringToULongLong(value, &config.node_creation_buffer) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								NODE_CREATION_BUFFER, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;

		case Config_CMD_INFO: {
			unsigned long long v = 0;
			if(RedisModule_StringToULongLong(value, &v) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								CMD_INFO, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			config.cmd_info = (v != 0);
			return REDISMODULE_OK;
		}

		case Config_CMD_INFO_MAX_QUERY_COUNT:
			if(RedisModule_StringToULongLong(value, &config.max_info_queries_count) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								CMD_INFO_MAX_QUERIES_COUNT_OPTION_NAME, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return T_INT64;

		case Config_EFFECTS_THRESHOLD:
			if(RedisModule_StringToULongLong(value, &config.effects_threshold) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								EFFECTS_THRESHOLD, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			return REDISMODULE_OK;

		case Config_BOLT_PORT: {
			unsigned long long v = 0;
			if(RedisModule_StringToULongLong(value, &v) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								BOLT_PORT, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			config.bolt_port = (int16_t)v;
			return REDISMODULE_OK;
		}

		case Config_DELAY_INDEXING: {
			unsigned long long v = 0;
			if(RedisModule_StringToULongLong(value, &v) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								DELAY_INDEXING, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			config.delay_indexing = (v != 0);
			return REDISMODULE_OK;
		}

		case Config_IMPORT_FOLDER: {
		   // allocate to c string copy the value and store in config import folder
			size_t len;
			const char *import_folder = RedisModule_StringPtrLen(value, &len);
			// copy the import folder path to config
			config.import_folder = RedisModule_Alloc(len + 1);
			memcpy(config.import_folder, import_folder, len);
			config.import_folder[len] = '\0'; // null-terminate the string
			return REDISMODULE_OK;
		}

		case Config_DEDUPLICATE_STRINGS: {
		    unsigned long long v = 0;
			if(RedisModule_StringToULongLong(value, &v) != REDISMODULE_OK) {
				RedisModule_Log(ctx, "error", "Failed to set configuration %s to %s",
								DEDUPLICATE_STRINGS, RedisModule_StringPtrLen(value, NULL));
				return REDISMODULE_ERR;
			}
			config.deduplicate_strings = (v != 0);
			return REDISMODULE_OK;
		}
		//----------------------------------------------------------------------
		// invalid option
		//----------------------------------------------------------------------

		default :
			ASSERT("invalid option field" && false);
			return REDISMODULE_ERR;
	}

	return REDISMODULE_OK;
}

int Config_Init
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT(ctx  != NULL);

	if(argc % 2) {
		// emit an error if we received an odd number of arguments,
		// as this indicates an invalid configuration
		RedisModule_Log(ctx, "warning",
						"FalkorDB received %d arguments, all configurations should be key-value pairs", argc);
		return REDISMODULE_ERR;
	}

	// ovveride default config from module arguments
	for(int i = 0; i < argc; i += 2) {
		// each configuration is a key-value pair. (K, V)

		//----------------------------------------------------------------------
		// get field
		//----------------------------------------------------------------------

		Config_Option_Field field;
		RedisModuleString *val = argv[i + 1];
		size_t key_len, val_len;
		const char *key_str = RedisModule_StringPtrLen(argv[i], &key_len);
		const char *val_str = RedisModule_StringPtrLen(val, &val_len);

		char *key_cstr = RedisModule_Alloc(key_len + 1);
		memcpy(key_cstr, key_str, key_len);
		key_cstr[key_len] = '\0';
		
		
		// exit if configuration is not aware of field
		if(!Config_Contains_field(key_cstr, &field)) {
			RedisModule_Log(ctx, "error",
							"Encountered unknown configuration field '%s'", key_cstr);
			RedisModule_Free(key_cstr);
			return REDISMODULE_ERR;
		}
		if (Set_Config_Init_Value(ctx, field, val) != REDISMODULE_OK) {
			RedisModule_Log(ctx, "error",
							"Failed to set configuration %s to %s", key_cstr, val_str);
			RedisModule_Free(key_cstr);
			return REDISMODULE_ERR;
		}
        
		RedisModule_Free(key_cstr);	 
	}

	_Config_Register(ctx);

	int res = RedisModule_LoadConfigs(ctx);
	ASSERT(res == REDISMODULE_OK);

	return REDISMODULE_OK;
}

