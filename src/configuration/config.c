/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "config.h"
#include "util/rmalloc.h"
#include "util/redis_version.h"
#include "../deps/GraphBLAS/Include/GraphBLAS.h"

#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include <stdint.h>
#include <sys/stat.h>

//-----------------------------------------------------------------------------
// Configuration parameters
//-----------------------------------------------------------------------------

// config param, the timeout for each query in milliseconds
#define TIMEOUT "TIMEOUT"

// default timeout for read and write queries
#define TIMEOUT_DEFAULT "TIMEOUT_DEFAULT"

// max timeout that can be enforced
#define TIMEOUT_MAX "TIMEOUT_MAX"

// config param, the size of each thread cache size, per graph
#define CACHE_SIZE "CACHE_SIZE"

// whether graphs should be deleted asynchronously
#define ASYNC_DELETE "ASYNC_DELETE"

// config param, number of threads in thread pool
#define THREAD_COUNT "THREAD_COUNT"

// resultset size limit
#define RESULTSET_SIZE "RESULTSET_SIZE"

// config param, max number of OpenMP threads
#define OMP_THREAD_COUNT "OMP_THREAD_COUNT"

// config param, max number of entities in each virtual key
#define VKEY_MAX_ENTITY_COUNT "VKEY_MAX_ENTITY_COUNT"

// config param, max number of queued queries
#define MAX_QUEUED_QUERIES "MAX_QUEUED_QUERIES"

// Max mem(bytes) that query/thread can utilize at any given time
#define QUERY_MEM_CAPACITY "QUERY_MEM_CAPACITY"

// number of pending changed before Delta_Matrix flushed
#define DELTA_MAX_PENDING_CHANGES "DELTA_MAX_PENDING_CHANGES"

// size of node creation buffer
#define NODE_CREATION_BUFFER "NODE_CREATION_BUFFER"

// The GRAPH.INFO command
#define CMD_INFO "CMD_INFO"

// The GRAPH.INFO QUERIES maximum element count
#define CMD_INFO_MAX_QUERIES_COUNT_OPTION_NAME "MAX_INFO_QUERIES"

// effects replication threshold
#define EFFECTS_THRESHOLD "EFFECTS_THRESHOLD"

// bolt protocol port
#define BOLT_PORT "BOLT_PORT"

// delay indexing
#define DELAY_INDEXING "DELAY_INDEXING"

// import folder
#define IMPORT_FOLDER "IMPORT_FOLDER"

// temp folder
#define TEMP_FOLDER "TEMP_FOLDER"

// quick js runtime heap max size
#define JS_HEAP_SIZE "JS_HEAP_SIZE"

// quick js runtime stack max size
#define JS_STACK_SIZE "JS_STACK_SIZE"

//------------------------------------------------------------------------------
// Configuration defaults
//------------------------------------------------------------------------------

#define CACHE_SIZE_DEFAULT 25
#define QUEUED_QUERIES_UNLIMITED INT64_MAX
#define VKEY_MAX_ENTITY_COUNT_DEFAULT 100000
#define CMD_INFO_DEFAULT true
#define CMD_INFO_QUERIES_MAX_COUNT_DEFAULT 1000
#define BOLT_PROTOCOL_PORT_DEFAULT -1 // disabled by default
#define DELAY_INDEXING_DEFAULT false
#define IMPORT_DIR_DEFAULT "/var/lib/FalkorDB/import/"
#define TEMP_DIR_DEFAULT "/tmp"
#define JS_HEAP_SIZE_DEFAULT  (256 * 1024 * 1024)  // 256 MB
#define JS_STACK_SIZE_DEFAULT (1024 * 1024)  // 1MB

// configuration object
typedef struct
{
	uint64_t timeout;				   // the timeout for each query in milliseconds
	uint64_t timeout_default;		   // default timeout for read and write queries
	uint64_t timeout_max;			   // max timeout that can be enforced
	uint64_t cache_size;			   // the cache size for each thread, per graph
	bool async_delete;				   // if true, graph deletion is done asynchronously
	uint64_t omp_thread_count;		   // maximum number of OpenMP threads
	uint64_t thread_pool_size;		   // thread count for thread pool
	uint64_t resultset_size;		   // resultset maximum size, UINT64_MAX unlimited
	uint64_t vkey_entity_count;		   // the limit of number of entities encoded at once for each RDB key
	uint64_t max_queued_queries;	   // max number of queued queries
	int64_t query_mem_capacity;		   // max mem(bytes) that query/thread can utilize at any given time
	int64_t delta_max_pending_changes; // number of pending changed before Delta_Matrix flushed
	uint64_t node_creation_buffer;	   // number of extra node creations to buffer as margin in matrices
	bool cmd_info_on;				   // if true, the GRAPH.INFO is enabled
	uint64_t effects_threshold;		   // replicate via effects when runtime exceeds threshold
	uint64_t max_info_queries_count;   // maximum number of query info elements
	int16_t bolt_port;				   // bolt protocol port
	bool delay_indexing;			   // delay index construction when decoding
	char *import_folder;			   // path to import folder, used for CSV loading
	char *temp_folder;				   // path to temp folder, used for storing temporary files
	size_t js_heap_size;               // quickjs runtime max heap size;
	size_t js_stack_size;              // quickjs runtime max stack size;
	Config_on_change cb;			   // callback function which being called when config param changed
} RG_Config;

RG_Config config; // global module configuration

//------------------------------------------------------------------------------
// Redis module configuration registration helpers
//------------------------------------------------------------------------------

static bool _Config_is_runtime(Config_Option_Field field) {
	for(size_t i = 0; i < RUNTIME_CONFIG_COUNT; i++) {
		if(RUNTIME_CONFIGS[i] == field) {
			return true;
		}
	}
	return false;
}

static unsigned int _Config_flags(Config_Option_Field field, bool memory) {
	unsigned int flags = _Config_is_runtime(field) ?
		REDISMODULE_CONFIG_DEFAULT : REDISMODULE_CONFIG_IMMUTABLE;
	if(memory) flags |= REDISMODULE_CONFIG_MEMORY;
	return flags;
}

static int _Config_set_from_string
(
	Config_Option_Field field,
	const char *val,
	RedisModuleString **err
) {
	char *error = NULL;
	if(!Config_Option_set(field, val, &error)) {
		if(err != NULL) {
			const char *name = Config_Field_name(field);
			if(error) {
				*err = RedisModule_CreateString(NULL, error, strlen(error));
			} else {
				char buf[128];
				snprintf(buf, sizeof(buf),
					"Failed to set config value %s to %s", name, val);
				*err = RedisModule_CreateString(NULL, buf, strlen(buf));
			}
		}
		return REDISMODULE_ERR;
	}
	return REDISMODULE_OK;
}

static RedisModuleString *_Config_get_string_value(Config_Option_Field field) {
	const char *value = NULL;
	bool res = Config_Option_get(field, &value);
	ASSERT(res);
	return RedisModule_CreateString(NULL, value, strlen(value));
}

static int _ModuleConfigSetNumeric(const char *name, long long val,
	void *privdata, RedisModuleString **err) {
	UNUSED(name);
	Config_Option_Field field = (Config_Option_Field)(intptr_t)privdata;
	char buf[64];
	snprintf(buf, sizeof(buf), "%lld", val);
	return _Config_set_from_string(field, buf, err);
}

static long long _ModuleConfigGetNumeric(const char *name, void *privdata) {
	UNUSED(name);
	Config_Option_Field field = (Config_Option_Field)(intptr_t)privdata;

	switch(field) {
	case Config_TIMEOUT:
	case Config_TIMEOUT_DEFAULT:
	case Config_TIMEOUT_MAX:
	case Config_CACHE_SIZE:
	case Config_OPENMP_NTHREAD:
	case Config_THREAD_POOL_SIZE:
	case Config_RESULTSET_MAX_SIZE:
	case Config_VKEY_MAX_ENTITY_COUNT:
	case Config_MAX_QUEUED_QUERIES:
	case Config_QUERY_MEM_CAPACITY:
	case Config_DELTA_MAX_PENDING_CHANGES:
	case Config_NODE_CREATION_BUFFER:
	case Config_CMD_INFO_MAX_QUERY_COUNT:
	case Config_EFFECTS_THRESHOLD:
	case Config_BOLT_PORT:
	case Config_IMPORT_FOLDER: // not numeric, handled below
	case Config_TEMP_FOLDER:   // not numeric, handled below
	case Config_JS_HEAP_SIZE:
	case Config_JS_STACK_SIZE:
	default:
		break;
	}

	long long n = 0;

	switch(field) {
	case Config_TIMEOUT:
	case Config_TIMEOUT_DEFAULT:
	case Config_TIMEOUT_MAX:
	case Config_MAX_QUEUED_QUERIES:
	case Config_QUERY_MEM_CAPACITY:
	case Config_DELTA_MAX_PENDING_CHANGES:
	case Config_EFFECTS_THRESHOLD:
	case Config_RESULTSET_MAX_SIZE:
	case Config_VKEY_MAX_ENTITY_COUNT:
	case Config_NODE_CREATION_BUFFER:
	case Config_CMD_INFO_MAX_QUERY_COUNT:
	case Config_CACHE_SIZE:
	case Config_OPENMP_NTHREAD:
	case Config_THREAD_POOL_SIZE:
	case Config_JS_HEAP_SIZE:
	case Config_JS_STACK_SIZE:
		Config_Option_get(field, &n);
		break;

	case Config_BOLT_PORT: {
		int16_t port;
		Config_Option_get(field, &port);
		n = port;
		break;
	}

	default:
		ASSERT(false && "invalid numeric config");
	}

	return n;
}

static int _ModuleConfigSetBool(const char *name, int val, void *privdata,
	RedisModuleString **err) {
	UNUSED(name);
	Config_Option_Field field = (Config_Option_Field)(intptr_t)privdata;
	const char *bool_str = val ? "yes" : "no";
	return _Config_set_from_string(field, bool_str, err);
}

static int _ModuleConfigGetBool(const char *name, void *privdata) {
	UNUSED(name);
	Config_Option_Field field = (Config_Option_Field)(intptr_t)privdata;
	bool flag = false;
	Config_Option_get(field, &flag);
	return flag ? 1 : 0;
}

static int _ModuleConfigSetString(const char *name, RedisModuleString *val,
	void *privdata, RedisModuleString **err) {
	UNUSED(name);
	Config_Option_Field field = (Config_Option_Field)(intptr_t)privdata;

	size_t len = 0;
	const char *cstr = RedisModule_StringPtrLen(val, &len);
	char *buf = rm_malloc(len + 1);
	memcpy(buf, cstr, len);
	buf[len] = '\0';

	int rc = _Config_set_from_string(field, buf, err);
	rm_free(buf);
	return rc;
}

static RedisModuleString *_ModuleConfigGetString(const char *name,
	void *privdata) {
	UNUSED(name);
	Config_Option_Field field = (Config_Option_Field)(intptr_t)privdata;
	return _Config_get_string_value(field);
}

static int _RegisterModuleConfigs(RedisModuleCtx *ctx) {
	// numeric configs
	if(RedisModule_RegisterNumericConfig(ctx, "timeout", config.timeout,
		_Config_flags(Config_TIMEOUT, false),
		0, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_TIMEOUT) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "timeout_default",
		config.timeout_default,
		_Config_flags(Config_TIMEOUT_DEFAULT, false),
		0, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_TIMEOUT_DEFAULT) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "timeout_max", config.timeout_max,
		_Config_flags(Config_TIMEOUT_MAX, false),
		0, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_TIMEOUT_MAX) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "cache_size", config.cache_size,
		_Config_flags(Config_CACHE_SIZE, false),
		1, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_CACHE_SIZE) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "thread_count",
		config.thread_pool_size,
		_Config_flags(Config_THREAD_POOL_SIZE, false),
		1, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_THREAD_POOL_SIZE) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "resultset_size",
		config.resultset_size,
		_Config_flags(Config_RESULTSET_MAX_SIZE, false),
		LLONG_MIN, LLONG_MAX, _ModuleConfigGetNumeric,
		_ModuleConfigSetNumeric, NULL,
		(void *)(intptr_t)Config_RESULTSET_MAX_SIZE) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "omp_thread_count",
		config.omp_thread_count,
		_Config_flags(Config_OPENMP_NTHREAD, false),
		1, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_OPENMP_NTHREAD) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "vkey_max_entity_count",
		config.vkey_entity_count,
		_Config_flags(Config_VKEY_MAX_ENTITY_COUNT, false),
		0, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_VKEY_MAX_ENTITY_COUNT) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "max_queued_queries",
		config.max_queued_queries,
		_Config_flags(Config_MAX_QUEUED_QUERIES, false),
		1, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_MAX_QUEUED_QUERIES) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "query_mem_capacity",
		config.query_mem_capacity,
		_Config_flags(Config_QUERY_MEM_CAPACITY, true),
		0, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_QUERY_MEM_CAPACITY) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "delta_max_pending_changes",
		config.delta_max_pending_changes,
		_Config_flags(Config_DELTA_MAX_PENDING_CHANGES, false),
		0, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_DELTA_MAX_PENDING_CHANGES) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "node_creation_buffer",
		config.node_creation_buffer,
		_Config_flags(Config_NODE_CREATION_BUFFER, false),
		0, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_NODE_CREATION_BUFFER) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "max_info_queries",
		config.max_info_queries_count,
		_Config_flags(Config_CMD_INFO_MAX_QUERY_COUNT, false),
		0, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_CMD_INFO_MAX_QUERY_COUNT) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "effects_threshold",
		config.effects_threshold,
		_Config_flags(Config_EFFECTS_THRESHOLD, false),
		0, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_EFFECTS_THRESHOLD) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "bolt_port", config.bolt_port,
		_Config_flags(Config_BOLT_PORT, false),
		-1, 65535, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_BOLT_PORT) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "js_heap_size",
		(long long)config.js_heap_size,
		_Config_flags(Config_JS_HEAP_SIZE, true),
		1048576, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_JS_HEAP_SIZE) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterNumericConfig(ctx, "js_stack_size",
		(long long)config.js_stack_size,
		_Config_flags(Config_JS_STACK_SIZE, true),
		1048576, LLONG_MAX, _ModuleConfigGetNumeric, _ModuleConfigSetNumeric,
		NULL, (void *)(intptr_t)Config_JS_STACK_SIZE) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	// bool configs
	if(RedisModule_RegisterBoolConfig(ctx, "async_delete", config.async_delete,
		_Config_flags(Config_ASYNC_DELETE, false),
		_ModuleConfigGetBool, _ModuleConfigSetBool, NULL,
		(void *)(intptr_t)Config_ASYNC_DELETE) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterBoolConfig(ctx, "cmd_info", config.cmd_info_on,
		_Config_flags(Config_CMD_INFO, false),
		_ModuleConfigGetBool, _ModuleConfigSetBool, NULL,
		(void *)(intptr_t)Config_CMD_INFO) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterBoolConfig(ctx, "delay_indexing", config.delay_indexing,
		_Config_flags(Config_DELAY_INDEXING, false),
		_ModuleConfigGetBool, _ModuleConfigSetBool, NULL,
		(void *)(intptr_t)Config_DELAY_INDEXING) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	// string configs
	if(RedisModule_RegisterStringConfig(ctx, "import_folder", config.import_folder,
		_Config_flags(Config_IMPORT_FOLDER, false),
		_ModuleConfigGetString, _ModuleConfigSetString, NULL,
		(void *)(intptr_t)Config_IMPORT_FOLDER) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	if(RedisModule_RegisterStringConfig(ctx, "temp_folder", config.temp_folder,
		_Config_flags(Config_TEMP_FOLDER, false),
		_ModuleConfigGetString, _ModuleConfigSetString, NULL,
		(void *)(intptr_t)Config_TEMP_FOLDER) == REDISMODULE_ERR) {
		return REDISMODULE_ERR;
	}

	return REDISMODULE_OK;
}

//------------------------------------------------------------------------------
// config value parsing
//------------------------------------------------------------------------------

// parse integer
// return true if string represents an integer
static inline bool _Config_ParseInteger(
	const char *integer_str,
	long long *value)
{
	char *endptr;
	errno = 0; // To distinguish success/failure after call
	*value = strtoll(integer_str, &endptr, 10);

	// Return an error code if integer parsing fails.
	return (errno == 0 && endptr != integer_str && *endptr == '\0');
}

// parse positive integer
// return true if string represents a positive integer > 0
static inline bool _Config_ParsePositiveInteger(
	const char *integer_str,
	long long *value)
{
	bool res = _Config_ParseInteger(integer_str, value);
	// Return an error code if integer parsing fails or value is not positive.
	return (res == true && *value > 0);
}

// parse non-negative integer
// return true if string represents an integer >= 0
static inline bool _Config_ParseNonNegativeInteger(
	const char *integer_str,
	long long *value)
{
	bool res = _Config_ParseInteger(integer_str, value);
	// Return an error code if integer parsing fails or value is negative.
	return (res == true && *value >= 0);
}

// return true if 'str' is either "yes" or "no" otherwise returns false
// sets 'value' to true if 'str' is "yes"
// sets 'value to false if 'str' is "no"
static inline bool _Config_ParseYesNo(
	const char *str,
	bool *value)
{
	bool res = false;

	if (!strcasecmp(str, "yes"))
	{
		res = true;
		*value = true;
	}
	else if (!strcasecmp(str, "no"))
	{
		res = true;
		*value = false;
	}

	return res;
}

//------------------------------------------------------------------------------
// Config access functions
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// max queued queries
//------------------------------------------------------------------------------

static void Config_max_queued_queries_set(
	uint64_t max_queued_queries)
{
	config.max_queued_queries = max_queued_queries;
}

static uint Config_max_queued_queries_get(void)
{
	return config.max_queued_queries;
}

//------------------------------------------------------------------------------
// timeout
//------------------------------------------------------------------------------

static void Config_timeout_set(
	uint64_t timeout)
{
	config.timeout = timeout;
}

// check if new(TIMEOUT_DEFAULT or TIMEOUT_MAX) are used
// log a deprecation message
static bool _Config_check_if_new_timeout_used()
{
	bool new_timeout_set = config.timeout_default != CONFIG_TIMEOUT_NO_TIMEOUT;
	new_timeout_set |= config.timeout_max != CONFIG_TIMEOUT_NO_TIMEOUT;

	if (new_timeout_set)
	{
		RedisModule_Log(NULL, "warning", "The TIMEOUT configuration parameter is deprecated. Please set TIMEOUT_MAX and TIMEOUT_DEFAULT instead");
		return false;
	}

	return true;
}

static bool Config_enforce_timeout_max(
	uint64_t timeout_default,
	uint64_t timeout_max)
{
	if (timeout_max != CONFIG_TIMEOUT_NO_TIMEOUT &&
		timeout_default > timeout_max)
	{
#ifdef __aarch64__
		RedisModule_Log(NULL, "warning", "The TIMEOUT_DEFAULT(%" PRId64 ") configuration parameter value is higher than TIMEOUT_MAX(%" PRId64 ").", timeout_default, timeout_max);
#else
		RedisModule_Log(NULL, "warning", "The TIMEOUT_DEFAULT(%ld) configuration parameter value is higher than TIMEOUT_MAX(%ld).", timeout_default, timeout_max);
#endif
		return false;
	}

	config.timeout_max = timeout_max;
	config.timeout_default = timeout_default;
	return true;
}

static bool Config_timeout_default_set(
	uint64_t timeout_default)
{
	return Config_enforce_timeout_max(timeout_default, config.timeout_max);
}

static bool Config_timeout_max_set(
	uint64_t timeout_max)
{
	return Config_enforce_timeout_max(config.timeout_default, timeout_max);
}

static uint Config_timeout_get(void)
{
	return config.timeout;
}

static uint Config_timeout_default_get(void)
{
	return config.timeout_default;
}

static uint Config_timeout_max_get(void)
{
	return config.timeout_max;
}

//------------------------------------------------------------------------------
// thread count
//------------------------------------------------------------------------------

static void Config_thread_pool_size_set(
	uint64_t nthreads)
{
	config.thread_pool_size = nthreads;
}

static uint64_t Config_thread_pool_size_get(void)
{
	return config.thread_pool_size;
}

//------------------------------------------------------------------------------
// OpenMP thread count
//------------------------------------------------------------------------------

static void Config_OMP_thread_count_set(uint64_t nthreads)
{
	config.omp_thread_count = nthreads;
}

static uint64_t Config_OMP_thread_count_get(void)
{
	return config.omp_thread_count;
}

//------------------------------------------------------------------------------
// virtual key entity count
//------------------------------------------------------------------------------

static void Config_virtual_key_entity_count_set(
	uint64_t entity_count)
{
	config.vkey_entity_count = entity_count;
}

static uint64_t Config_virtual_key_entity_count_get(void)
{
	return config.vkey_entity_count;
}

//------------------------------------------------------------------------------
// cache size
//------------------------------------------------------------------------------

static void Config_cache_size_set(
	uint64_t cache_size)
{
	config.cache_size = cache_size;
}

static uint64_t Config_cache_size_get(void)
{
	return config.cache_size;
}

//------------------------------------------------------------------------------
// async delete
//------------------------------------------------------------------------------

static void Config_async_delete_set(
	bool async_delete)
{
	config.async_delete = async_delete;
}

static bool Config_async_delete_get(void)
{
	return config.async_delete;
}

//------------------------------------------------------------------------------
// result-set max size
//------------------------------------------------------------------------------

static void Config_resultset_max_size_set(
	int64_t max_size)
{
	if (max_size < 0)
		config.resultset_size = RESULTSET_SIZE_UNLIMITED;
	else
		config.resultset_size = max_size;
}

static uint64_t Config_resultset_max_size_get(void)
{
	return config.resultset_size;
}

//------------------------------------------------------------------------------
// query mem capacity
//------------------------------------------------------------------------------

static void Config_query_mem_capacity_set(
	int64_t capacity)
{
	if (capacity <= 0)
		config.query_mem_capacity = QUERY_MEM_CAPACITY_UNLIMITED;
	else
		config.query_mem_capacity = capacity;
}

static uint64_t Config_query_mem_capacity_get(void)
{
	return config.query_mem_capacity;
}

//------------------------------------------------------------------------------
// delta max pending changes
//------------------------------------------------------------------------------

static void Config_delta_max_pending_changes_set(
	int64_t capacity)
{
	if (capacity == 0)
		config.delta_max_pending_changes = DELTA_MAX_PENDING_CHANGES_DEFAULT;
	else
		config.delta_max_pending_changes = capacity;
}

static uint64_t Config_delta_max_pending_changes_get(void)
{
	return config.delta_max_pending_changes;
}

//------------------------------------------------------------------------------
// node creation buffer
//------------------------------------------------------------------------------

static void Config_node_creation_buffer_set(
	uint64_t buf_size)
{
	config.node_creation_buffer = buf_size;
}

static uint64_t Config_node_creation_buffer_get(void)
{
	return config.node_creation_buffer;
}

//------------------------------------------------------------------------------
// cmd info
//------------------------------------------------------------------------------

static bool Config_cmd_info_get(void)
{
	return config.cmd_info_on;
}

static void Config_cmd_info_set(
	const bool cmd_info_on)
{
	config.cmd_info_on = cmd_info_on;
}

static uint64_t Config_cmd_info_max_queries_get(void)
{
	return config.max_info_queries_count;
}

static void Config_cmd_info_max_queries_set(
	const uint64_t count)
{
	if (count > CMD_INFO_QUERIES_MAX_COUNT_DEFAULT)
	{
		config.max_info_queries_count = CMD_INFO_QUERIES_MAX_COUNT_DEFAULT;
	}
	else
	{
		config.max_info_queries_count = count;
	}
}

//------------------------------------------------------------------------------
// effects threshold
//------------------------------------------------------------------------------

static void Config_effects_threshold_set(
	uint64_t threshold)
{
	config.effects_threshold = threshold;
}

static uint64_t Config_effects_threshold_get(void)
{
	return config.effects_threshold;
}

//------------------------------------------------------------------------------
// bolt protocol port
//------------------------------------------------------------------------------

static void Config_bolt_port_set(
	int16_t port)
{
	int16_t p = (port < 0) ? BOLT_PROTOCOL_PORT_DEFAULT : port;
	config.bolt_port = p;
}

static int16_t Config_bolt_port_get(void)
{
	return config.bolt_port;
}

//------------------------------------------------------------------------------
// delay indexing
//------------------------------------------------------------------------------

static bool Config_delay_indexing_get(void)
{
	return config.delay_indexing;
}

static void Config_delay_indexing_set(
	const bool delay_indexing)
{
	config.delay_indexing = delay_indexing;
}

//------------------------------------------------------------------------------
// import folder
//------------------------------------------------------------------------------

static void Config_import_folder_set(
	const char *path)
{
	ASSERT(path != NULL);

	// free previous value
	rm_free(config.import_folder);

	// copy new path
	config.import_folder = rm_strdup(path);
}

static const char *Config_import_folder_get(void)
{
	return config.import_folder;
}

//------------------------------------------------------------------------------
// temp folder
//------------------------------------------------------------------------------
static void Config_temp_folder_set(
	const char *path)
{
	ASSERT(path != NULL);

	// free previous value
	rm_free(config.temp_folder);

	// copy new path
	size_t len = strlen(path);
	// check if path ends with "/"
	if (len > 0 && path[len - 1] == '/')
	{
		// allocate and copy without trailing "/"
		config.temp_folder = rm_malloc(len);
		memcpy(config.temp_folder, path, len - 1);
		config.temp_folder[len - 1] = '\0';
	}
	else
	{
		config.temp_folder = rm_strdup(path);
	}
}

static const char *Config_temp_folder_get(void)
{
	return config.temp_folder;
}

//------------------------------------------------------------------------------
// JS Heap Size
//------------------------------------------------------------------------------

static void Config_js_heap_size_set
(
	size_t size
) {
    config.js_heap_size = size ;
}

static size_t Config_js_heap_size_get(void) {
    return config.js_heap_size ;
}

//------------------------------------------------------------------------------
// JS Stack Size
//------------------------------------------------------------------------------

static void Config_js_stack_size_set
(
	size_t size
) {
    config.js_stack_size = size ;
}

static size_t Config_js_stack_size_get(void) {
    return config.js_stack_size ;
}

// check if field is a valid configuration option
bool Config_Contains_field(
	const char *field_str,	   // configuration option name
	Config_Option_Field *field // [out] configuration field
)
{
	ASSERT(field_str != NULL);

	Config_Option_Field f;

	if (!strcasecmp(field_str, THREAD_COUNT))
	{
		f = Config_THREAD_POOL_SIZE;
	}
	else if (!strcasecmp(field_str, TIMEOUT))
	{
		f = Config_TIMEOUT;
	}
	else if (!(strcasecmp(field_str, TIMEOUT_DEFAULT)))
	{
		f = Config_TIMEOUT_DEFAULT;
	}
	else if (!(strcasecmp(field_str, TIMEOUT_MAX)))
	{
		f = Config_TIMEOUT_MAX;
	}
	else if (!strcasecmp(field_str, OMP_THREAD_COUNT))
	{
		f = Config_OPENMP_NTHREAD;
	}
	else if (!strcasecmp(field_str, VKEY_MAX_ENTITY_COUNT))
	{
		f = Config_VKEY_MAX_ENTITY_COUNT;
	}
	else if (!(strcasecmp(field_str, CACHE_SIZE)))
	{
		f = Config_CACHE_SIZE;
	}
	else if (!(strcasecmp(field_str, RESULTSET_SIZE)))
	{
		f = Config_RESULTSET_MAX_SIZE;
	}
	else if (!(strcasecmp(field_str, MAX_QUEUED_QUERIES)))
	{
		f = Config_MAX_QUEUED_QUERIES;
	}
	else if (!(strcasecmp(field_str, QUERY_MEM_CAPACITY)))
	{
		f = Config_QUERY_MEM_CAPACITY;
	}
	else if (!(strcasecmp(field_str, DELTA_MAX_PENDING_CHANGES)))
	{
		f = Config_DELTA_MAX_PENDING_CHANGES;
	}
	else if (!(strcasecmp(field_str, NODE_CREATION_BUFFER)))
	{
		f = Config_NODE_CREATION_BUFFER;
	}
	else if (!(strcasecmp(field_str, ASYNC_DELETE)))
	{
		f = Config_ASYNC_DELETE;
	}
	else if (!(strcasecmp(field_str, CMD_INFO)))
	{
		f = Config_CMD_INFO;
	}
	else if (!(strcasecmp(field_str, CMD_INFO_MAX_QUERIES_COUNT_OPTION_NAME)))
	{
		f = Config_CMD_INFO_MAX_QUERY_COUNT;
	}
	else if (!(strcasecmp(field_str, EFFECTS_THRESHOLD)))
	{
		f = Config_EFFECTS_THRESHOLD;
	}
	else if (!(strcasecmp(field_str, BOLT_PORT)))
	{
		f = Config_BOLT_PORT;
	}
	else if (!(strcasecmp(field_str, DELAY_INDEXING)))
	{
		f = Config_DELAY_INDEXING;
	}
	else if (!(strcasecmp(field_str, IMPORT_FOLDER)))
	{
		f = Config_IMPORT_FOLDER;
	}
	else if (!(strcasecmp(field_str, TEMP_FOLDER)))
	{
		f = Config_TEMP_FOLDER;
	}
	else if (!(strcasecmp (field_str, JS_HEAP_SIZE)))
    {
        f = Config_JS_HEAP_SIZE ;
    }
    else if (!(strcasecmp (field_str, JS_STACK_SIZE)))
    {
        f = Config_JS_STACK_SIZE ;
    }
	else
	{
		return false ;
	}

	if (field)
		*field = f;
	return true;
}

// returns the field type
SIType Config_Field_type(
	Config_Option_Field field // field
)
{
	switch (field)
	{
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

	case Config_TEMP_FOLDER:
		return T_STRING;

	case Config_JS_HEAP_SIZE:
        return T_INT64 ;

    case Config_JS_STACK_SIZE:
        return T_INT64 ;

		//----------------------------------------------------------------------
		// invalid option
		//----------------------------------------------------------------------

	default:
		ASSERT("invalid option field" && false);
		break;
	}

	return T_NULL;
}

const char *Config_Field_name(
	Config_Option_Field field)
{
	const char *name = NULL;
	switch (field)
	{
	case Config_TIMEOUT:
		name = TIMEOUT;
		break;

	case Config_TIMEOUT_DEFAULT:
		name = TIMEOUT_DEFAULT;
		break;

	case Config_TIMEOUT_MAX:
		name = TIMEOUT_MAX;
		break;

	case Config_CACHE_SIZE:
		name = CACHE_SIZE;
		break;

	case Config_OPENMP_NTHREAD:
		name = OMP_THREAD_COUNT;
		break;

	case Config_THREAD_POOL_SIZE:
		name = THREAD_COUNT;
		break;

	case Config_RESULTSET_MAX_SIZE:
		name = RESULTSET_SIZE;
		break;

	case Config_VKEY_MAX_ENTITY_COUNT:
		name = VKEY_MAX_ENTITY_COUNT;
		break;

	case Config_ASYNC_DELETE:
		name = ASYNC_DELETE;
		break;

	case Config_MAX_QUEUED_QUERIES:
		name = MAX_QUEUED_QUERIES;
		break;

	case Config_QUERY_MEM_CAPACITY:
		name = QUERY_MEM_CAPACITY;
		break;

	case Config_DELTA_MAX_PENDING_CHANGES:
		name = DELTA_MAX_PENDING_CHANGES;
		break;

	case Config_NODE_CREATION_BUFFER:
		name = NODE_CREATION_BUFFER;
		break;

	case Config_CMD_INFO:
		name = CMD_INFO;
		break;

	case Config_CMD_INFO_MAX_QUERY_COUNT:
		name = CMD_INFO_MAX_QUERIES_COUNT_OPTION_NAME;
		break;

	case Config_EFFECTS_THRESHOLD:
		name = EFFECTS_THRESHOLD;
		break;

	case Config_BOLT_PORT:
		name = BOLT_PORT;
		break;

	case Config_DELAY_INDEXING:
		name = DELAY_INDEXING;
		break;

	case Config_IMPORT_FOLDER:
		name = IMPORT_FOLDER;
		break;

	case Config_TEMP_FOLDER:
		name = TEMP_FOLDER;
		break;

	case Config_JS_HEAP_SIZE:
        name = JS_HEAP_SIZE ;
        break ;

    case Config_JS_STACK_SIZE:
        name = JS_STACK_SIZE ;
        break ;

		//----------------------------------------------------------------------
		// invalid option
		//----------------------------------------------------------------------

	default:
		ASSERT("invalid option field" && false);
		break;
	}

	return name;
}

// initialize every module-level configuration to its default value
static void _Config_SetToDefaults(void)
{
	// the thread pool's default size is equal to the system's number of cores
	int CPUCount = sysconf(_SC_NPROCESSORS_ONLN);
	config.thread_pool_size = (CPUCount != -1) ? CPUCount : 1;

	// use the GraphBLAS-defined number of OpenMP threads by default
	GxB_get(GxB_NTHREADS, &config.omp_thread_count);

	// the default entity count of virtual keys
	config.vkey_entity_count = VKEY_MAX_ENTITY_COUNT_DEFAULT;

// MEMCHECK compile flag;
#ifdef MEMCHECK
	// disable async delete during memcheck
	config.async_delete = false;
#else
	// always perform async delete when no checking for memory issues
	config.async_delete = true;
#endif

	config.cache_size = CACHE_SIZE_DEFAULT;

	// no limit on result-set size
	config.resultset_size = RESULTSET_SIZE_UNLIMITED;

	// no query timeout by default
	config.timeout = CONFIG_TIMEOUT_NO_TIMEOUT;

	// no max timeout by default
	config.timeout_max = CONFIG_TIMEOUT_NO_TIMEOUT;

	// no query timeout by default
	config.timeout_default = CONFIG_TIMEOUT_NO_TIMEOUT;

	// no limit on number of queued queries by default
	config.max_queued_queries = QUEUED_QUERIES_UNLIMITED;

	// no limit on query memory capacity
	config.query_mem_capacity = QUERY_MEM_CAPACITY_UNLIMITED;

	// number of pending changed before Delta_Matrix flushed
	config.delta_max_pending_changes = DELTA_MAX_PENDING_CHANGES_DEFAULT;

	// the amount of empty space to reserve for node creations in matrices
	config.node_creation_buffer = NODE_CREATION_BUFFER_DEFAULT;

	// GRAPH.INFO command on/off.
	config.cmd_info_on = CMD_INFO_DEFAULT;

	// GRAPH.INFO maximum queries count.
	config.max_info_queries_count = CMD_INFO_QUERIES_MAX_COUNT_DEFAULT;

	// replicate effects if avg change time μs > effects_threshold μs
	config.effects_threshold = 300;

	// bolt protocol port (disabled by default)
	config.bolt_port = BOLT_PROTOCOL_PORT_DEFAULT;

	// index entities as they're being decoded
	config.delay_indexing = DELAY_INDEXING_DEFAULT;

	// set default import folder path
	config.import_folder = rm_strdup(IMPORT_DIR_DEFAULT);

	// set default temp folder path
	config.temp_folder = rm_strdup(TEMP_DIR_DEFAULT);

	// set default JS runtime limits
	config.js_heap_size  = JS_HEAP_SIZE_DEFAULT ;
	config.js_stack_size = JS_STACK_SIZE_DEFAULT ;
}

// set module-level configurations to defaults or to user provided arguments
// returns REDISMODULE_OK on success
// emits an error and returns REDISMODULE_ERR on failure
int Config_Init
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // arguments
	int argc                   // number of arguments
) {
	// make sure reconfiguration callback is already registered
	ASSERT (config.cb != NULL) ;

	// initialize the configuration to its default values
	_Config_SetToDefaults () ;

	// register configurations with Redis CONFIG interface
	if(_RegisterModuleConfigs(ctx) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "warning",
				"Failed to register module configuration");
		return REDISMODULE_ERR;
	}

	// apply configurations provided via MODULE LOADEX / redis.conf
	if(RedisModule_LoadConfigs(ctx) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "warning",
				"Failed to load module configuration");
		return REDISMODULE_ERR;
	}

	if(argc > 0) {
		RedisModule_Log(ctx, "warning",
				"Passing FalkorDB configuration via loadmodule arguments is "
				"deprecated and will be removed in a future release. "
				"Use redis.conf module directives (for example: "
				"graph.max_queued_queries 10) instead.");
	}

	if (argc % 2)
	{
		// emit an error if we received an odd number of arguments,
		// as this indicates an invalid configuration
		RedisModule_Log (ctx, "warning",
						"FalkorDB received %d arguments, all configurations"
						"should be key-value pairs", argc) ;
		return REDISMODULE_ERR ;
	}

	bool old_timeout_specified = false ;
	bool new_timeout_specified = false ;

	for (int i = 0; i < argc; i += 2) {
		// each configuration is a key-value pair. (K, V)

		//----------------------------------------------------------------------
		// get field
		//----------------------------------------------------------------------

		Config_Option_Field field ;
		RedisModuleString *val = argv [i + 1] ;
		const char *val_str    = RedisModule_StringPtrLen (val, NULL) ;
		const char *field_str  = RedisModule_StringPtrLen (argv [i], NULL) ;

		// exit if configuration is not aware of field
		if (!Config_Contains_field (field_str, &field)) {
			RedisModule_Log (ctx, "error",
							"Encountered unknown configuration field '%s'",
							field_str) ;
			return REDISMODULE_ERR ;
		}

		if (field == Config_TIMEOUT_DEFAULT || field == Config_TIMEOUT_MAX) {
			new_timeout_specified = true ;
		}

		if (field == Config_TIMEOUT) {
			old_timeout_specified = true ;
		}

		// exit if encountered an error when setting configuration
		char *error = NULL ;
		if (!Config_Option_set (field, val_str, &error)) {
			if (error != NULL) {
				RedisModule_Log (ctx, "error",
								"Failed setting field '%s' with error: %s",
								field_str, error) ;
			} else {
				RedisModule_Log (ctx, "error",
								"Failed setting field '%s'", field_str) ;
			}
			return REDISMODULE_ERR ;
		}
	}

	if (old_timeout_specified && new_timeout_specified) {
		RedisModule_Log (ctx, "error",
						"The TIMEOUT configuration parameter should be removed"
						"when specifying TIMEOUT_DEFAULT and/or TIMEOUT_MAX") ;
		return REDISMODULE_ERR ;
	}

	return REDISMODULE_OK ;
}

bool Config_Option_get(
	Config_Option_Field field,
	...)
{
	//--------------------------------------------------------------------------
	// get the option
	//--------------------------------------------------------------------------

	va_list ap;

	switch (field)
	{
	case Config_MAX_QUEUED_QUERIES:
	{
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

	case Config_TIMEOUT:
	{
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

	case Config_TIMEOUT_DEFAULT:
	{
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

	case Config_TIMEOUT_MAX:
	{
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

	case Config_CACHE_SIZE:
	{
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

	case Config_OPENMP_NTHREAD:
	{
		va_start(ap, field);
		uint64_t *omp_nthreads = va_arg(ap, uint64_t *);
		va_end(ap);

		ASSERT(omp_nthreads != NULL);
		(*omp_nthreads) = Config_OMP_thread_count_get();
	}
	break;

		//----------------------------------------------------------------------
		// thread-pool size
		//----------------------------------------------------------------------

	case Config_THREAD_POOL_SIZE:
	{
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

	case Config_RESULTSET_MAX_SIZE:
	{
		va_start(ap, field);
		uint64_t *resultset_max_size = va_arg(ap, uint64_t *);
		va_end(ap);

		ASSERT(resultset_max_size != NULL);
		(*resultset_max_size) = Config_resultset_max_size_get();
	}
	break;

		//----------------------------------------------------------------------
		// virtual key entity count
		//----------------------------------------------------------------------

	case Config_VKEY_MAX_ENTITY_COUNT:
	{
		va_start(ap, field);
		uint64_t *vkey_max_entity_count = va_arg(ap, uint64_t *);
		va_end(ap);

		ASSERT(vkey_max_entity_count != NULL);
		(*vkey_max_entity_count) = Config_virtual_key_entity_count_get();
	}
	break;

		//----------------------------------------------------------------------
		// async deleteion
		//----------------------------------------------------------------------

	case Config_ASYNC_DELETE:
	{
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

	case Config_QUERY_MEM_CAPACITY:
	{
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

	case Config_DELTA_MAX_PENDING_CHANGES:
	{
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

	case Config_NODE_CREATION_BUFFER:
	{
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

	case Config_CMD_INFO:
	{
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

	case Config_CMD_INFO_MAX_QUERY_COUNT:
	{
		va_start(ap, field);
		uint64_t *count = va_arg(ap, uint64_t *);
		va_end(ap);

		ASSERT(count != NULL);
		(*count) = Config_cmd_info_max_queries_get();
	}
	break;

		//----------------------------------------------------------------------
		// effects threshold
		//----------------------------------------------------------------------

	case Config_EFFECTS_THRESHOLD:
	{
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

	case Config_BOLT_PORT:
	{
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

	case Config_DELAY_INDEXING:
	{
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

	case Config_IMPORT_FOLDER:
	{
		va_start(ap, field);
		const char **import_folder = va_arg(ap, const char **);
		va_end(ap);

		ASSERT(import_folder != NULL);
		(*import_folder) = Config_import_folder_get();
	}
	break;

		//----------------------------------------------------------------------
		// temp folder path
		//----------------------------------------------------------------------

	case Config_TEMP_FOLDER:
	{
		va_start(ap, field);
		const char **temp_folder = va_arg(ap, const char **);
		va_end(ap);

		ASSERT(temp_folder != NULL);
		(*temp_folder) = Config_temp_folder_get();
	}
	break;

	case Config_JS_HEAP_SIZE:
    {
        va_start (ap, field) ;
        size_t *js_heap_size = va_arg (ap, size_t *) ;
        va_end (ap) ;

        ASSERT (js_heap_size != NULL) ;
        (*js_heap_size) = Config_js_heap_size_get () ;
    }
    break ;

    case Config_JS_STACK_SIZE:
    {
        va_start (ap, field) ;
        size_t *js_stack_size = va_arg (ap, size_t *) ;
        va_end (ap) ;

        ASSERT (js_stack_size != NULL) ;
        (*js_stack_size) = Config_js_stack_size_get () ;
    }
    break ;

		//----------------------------------------------------------------------
		// invalid option
		//----------------------------------------------------------------------

	default:
		ASSERT("invalid option field" && false);
		return false;
	}

	return true;
}

bool Config_Option_set
(
	Config_Option_Field field,
	const char *val,
	char **err
) {
	bool updated = false ;  // indicate rather or not a value was updated

	//--------------------------------------------------------------------------
	// set the option
	//--------------------------------------------------------------------------

	switch (field)
	{
		//----------------------------------------------------------------------
		// max queued queries
		//----------------------------------------------------------------------

		case Config_MAX_QUEUED_QUERIES:
		{
			long long max_queued_queries;
			if (!_Config_ParsePositiveInteger(val, &max_queued_queries))
			{
				return false;
			}

			// update only if value changed
			if (Config_max_queued_queries_get () != max_queued_queries) {
				Config_max_queued_queries_set (max_queued_queries) ;
				updated = true ;
			}
		}
		break;

		//----------------------------------------------------------------------
		// timeout
		//----------------------------------------------------------------------

		case Config_TIMEOUT:
		{
			long long timeout ;
			if (!_Config_ParseNonNegativeInteger (val, &timeout)) {
				return false ;
			}

			if (!_Config_check_if_new_timeout_used ()) {
				if (err) {
					*err = "The TIMEOUT configuration parameter is deprecated."
						" Please set TIMEOUT_MAX and TIMEOUT_DEFAULT instead";
				}
				return false ;
			}
			if (Config_timeout_get () != timeout) {
				Config_timeout_set (timeout) ;
				updated = true ;
			}
		}
		break;

		//----------------------------------------------------------------------
		// timeout default
		//----------------------------------------------------------------------

		case Config_TIMEOUT_DEFAULT:
		{
			long long timeout_default ;
			if (!_Config_ParseNonNegativeInteger (val, &timeout_default)) {
				return false ;
			}

			if (Config_timeout_default_get () != timeout_default) {
				if (!Config_timeout_default_set (timeout_default)) {
					if (err) {
						*err = "TIMEOUT_DEFAULT configuration parameter cannot be"
							" set to a value higher than TIMEOUT_MAX" ;
					}
					return false ;
				}
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// timeout max
		//----------------------------------------------------------------------

		case Config_TIMEOUT_MAX:
		{
			long long timeout_max ;
			if (!_Config_ParseNonNegativeInteger (val, &timeout_max)) {
				return false ;
			}

			if (Config_timeout_max_get () != timeout_max) {
				if (!Config_timeout_max_set (timeout_max)) {
					if (err)
						*err = "TIMEOUT_MAX configuration parameter cannot be"
							" set to a value lower than TIMEOUT_DEFAULT" ;
					return false ;
				}
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// cache size
		//----------------------------------------------------------------------

		case Config_CACHE_SIZE:
		{
			long long cache_size ;
			if (!_Config_ParsePositiveInteger (val, &cache_size)) {
				return false ;
			}

			if (Config_cache_size_get () != cache_size) {
				Config_cache_size_set (cache_size) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// OpenMP thread count
		//----------------------------------------------------------------------

		case Config_OPENMP_NTHREAD:
		{
			long long omp_nthreads ;
			if (!_Config_ParsePositiveInteger (val, &omp_nthreads)) {
				return false ;
			}

			if (Config_OMP_thread_count_get () != omp_nthreads) {
				Config_OMP_thread_count_set (omp_nthreads) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// thread-pool size
		//----------------------------------------------------------------------

		case Config_THREAD_POOL_SIZE:
		{
			long long pool_nthreads ;
			if (!_Config_ParsePositiveInteger (val, &pool_nthreads)) {
				return false ;
			}

			if (Config_thread_pool_size_get () != pool_nthreads) {
				Config_thread_pool_size_set (pool_nthreads) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// result-set size
		//----------------------------------------------------------------------

		case Config_RESULTSET_MAX_SIZE:
		{
			long long resultset_max_size ;
			if (!_Config_ParseInteger (val, &resultset_max_size)) {
				return false ;
			}

			if (Config_resultset_max_size_get () != resultset_max_size) {
				Config_resultset_max_size_set(resultset_max_size);
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// virtual key entity count
		//----------------------------------------------------------------------

		case Config_VKEY_MAX_ENTITY_COUNT:
		{
			long long vkey_max_entity_count ;
			if (!_Config_ParseNonNegativeInteger (val, &vkey_max_entity_count)) {
				return false ;
			}

			if (Config_virtual_key_entity_count_get () != vkey_max_entity_count) {
				Config_virtual_key_entity_count_set(vkey_max_entity_count);
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// async deleteion
		//----------------------------------------------------------------------

		case Config_ASYNC_DELETE:
		{
			bool async_delete ;
			if (!_Config_ParseYesNo (val, &async_delete)) {
				return false ;
			}

			if (Config_async_delete_get () != async_delete) {
				Config_async_delete_set (async_delete) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// query mem capacity
		//----------------------------------------------------------------------

		case Config_QUERY_MEM_CAPACITY:
		{
			long long query_mem_capacity ;
			if (!_Config_ParseNonNegativeInteger (val, &query_mem_capacity)) {
				return false ;
			}

			if (Config_query_mem_capacity_get () != query_mem_capacity) {
				Config_query_mem_capacity_set (query_mem_capacity) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// number of pending changed befor Delta_Matrix flushed
		//----------------------------------------------------------------------

		case Config_DELTA_MAX_PENDING_CHANGES:
		{
			long long delta_max_pending_changes ;
			if (!_Config_ParseNonNegativeInteger (val, &delta_max_pending_changes)) {
				return false ;
			}

			if (Config_delta_max_pending_changes_get () != delta_max_pending_changes) {
				Config_delta_max_pending_changes_set (delta_max_pending_changes) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// size of buffer to maintain as margin in matrices
		//----------------------------------------------------------------------

		case Config_NODE_CREATION_BUFFER:
		{
			long long node_creation_buffer ;
			if (!_Config_ParseNonNegativeInteger (val, &node_creation_buffer)) {
				return false ;
			}

			// node_creation_buffer should be at-least 128
			node_creation_buffer = (node_creation_buffer < 128) ?
				128 :
				node_creation_buffer ;

			// retrieve the MSB of the value
			long long msb = (sizeof(long long) * 8) - __builtin_clzll (node_creation_buffer) ;
			long long set_msb = 1 << (msb - 1) ;

			// if the value is not a power of 2
			// (if any bits other than the MSB are 1),
			// raise it to the next power of 2
			if ((~set_msb & node_creation_buffer) != 0) {
				node_creation_buffer = 1 << msb ;
			}
			if (Config_node_creation_buffer_get () != node_creation_buffer) {
				Config_node_creation_buffer_set (node_creation_buffer) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// cmd info
		//----------------------------------------------------------------------

		case Config_CMD_INFO:
		{
			bool cmd_info_on = false ;
			if (!_Config_ParseYesNo (val, &cmd_info_on)) {
				return false ;
			}

			if (Config_cmd_info_get () != cmd_info_on) {
				Config_cmd_info_set (cmd_info_on) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// cmd info max queries count
		//----------------------------------------------------------------------

		case Config_CMD_INFO_MAX_QUERY_COUNT:
		{
			long long count = 0 ;
			if (!_Config_ParseNonNegativeInteger (val, &count)) {
				return false ;
			}
			if (count > UINT64_MAX) {
				return false ;
			}

			if (Config_cmd_info_max_queries_get () != count) {
				Config_cmd_info_max_queries_set (count) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// effects threshold
		//----------------------------------------------------------------------

		case Config_EFFECTS_THRESHOLD:
		{
			long long threshold ;
			if (!_Config_ParseNonNegativeInteger (val, &threshold)) {
				return false ;
			}
			if (Config_effects_threshold_get () != threshold) {
				Config_effects_threshold_set (threshold) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// bolt protocol port
		//----------------------------------------------------------------------

		case Config_BOLT_PORT:
		{
			long long port ;
			if (!_Config_ParseInteger (val, &port)) {
				return false ;
			}
			if (Config_bolt_port_get () != port) {
				Config_bolt_port_set (port) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// delay indexing
		//----------------------------------------------------------------------

		case Config_DELAY_INDEXING:
		{
			bool delay_indexing ;
			if (!_Config_ParseYesNo (val, &delay_indexing)) {
				return false ;
			}

			if (Config_delay_indexing_get () != delay_indexing) {
				Config_delay_indexing_set (delay_indexing) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// import folder path
		//----------------------------------------------------------------------

		case Config_IMPORT_FOLDER:
		{
			ASSERT (val != NULL) ;
			if (strcmp (Config_import_folder_get (), val) != 0) {
				Config_import_folder_set (val) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// temp folder path
		//----------------------------------------------------------------------

		case Config_TEMP_FOLDER:
		{
			ASSERT (val != NULL) ;

			// check that path exists
			if (access (val, F_OK) != 0)
			{
				if (err)
					*err = "TEMP_FOLDER path does not exist";
				return false;
			}

			// check that path is a directory
			struct stat st;
			if (stat(val, &st) != 0 || !S_ISDIR(st.st_mode))
			{
				if (err)
					*err = "TEMP_FOLDER path is not a directory";
				return false;
			}

			// check that process has write access
			if (access(val, W_OK) != 0)
			{
				if (err)
					*err = "No write access to TEMP_FOLDER path";
				return false;
			}

			if (strcmp (Config_temp_folder_get (), val) != 0) {
				Config_temp_folder_set (val) ;
				updated = true ;
			}
		}
		break ;

		case Config_JS_HEAP_SIZE:
		{
			long long heap_size ;
			if (!_Config_ParsePositiveInteger (val, &heap_size) ||
				heap_size < 1048576) {
				if (err) {
					*err = "JS_HEAP_SIZE must be at least 1MB (1048576)" ;
				}
				return false ;
			}

			if (Config_js_heap_size_get () != heap_size) {
				Config_js_heap_size_set ((size_t)heap_size) ;
				updated = true ;
			}
		}
		break ;

		case Config_JS_STACK_SIZE:
		{
			long long stack_size ;
			if (!_Config_ParsePositiveInteger (val, &stack_size) || stack_size < 1048576) {
				if (err) {
					*err = "JS_STACK_SIZE must be at least 1MB (1048576)" ;
				}
				return false ;
			}

			if (Config_js_stack_size_get () != stack_size) {
				Config_js_stack_size_set ((size_t)stack_size) ;
				updated = true ;
			}
		}
		break ;

		//----------------------------------------------------------------------
		// invalid option
		//----------------------------------------------------------------------

	default:
		return false ;
	}

	if (updated && config.cb) {
		config.cb (field) ;
	}

	return true ;
}

// dry run configuration change
bool Config_Option_dryrun(
	Config_Option_Field field,
	const char *val,
	char **err)
{
	// clone configuration
	RG_Config config_clone = config;

	// disable configuration notification
	config.cb = NULL;

	// NOTE: for a short period of time
	// whoever might query the configuration WILL see this modification
	bool valid = Config_Option_set(field, val, err);

	// restore original configuration
	config = config_clone;

	return valid;
}

void Config_Subscribe_Changes(
	Config_on_change cb)
{
	ASSERT(cb != NULL);
	config.cb = cb;
}
