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
	Config_on_change cb;               // callback function which being called when config param changed
} RG_Config;

RG_Config config; // global module configuration

//------------------------------------------------------------------------------
// config value parsing
//------------------------------------------------------------------------------

// parse integer
// return true if string represents an integer
static inline bool _Config_ParseInteger
(
	const char *integer_str,
	long long *value
) {
	char *endptr;
	errno = 0;    // To distinguish success/failure after call
	*value = strtoll(integer_str, &endptr, 10);

	// Return an error code if integer parsing fails.
	return (errno == 0 && endptr != integer_str && *endptr == '\0');
}

// parse positive integer
// return true if string represents a positive integer > 0
static inline bool _Config_ParsePositiveInteger
(
	const char *integer_str,
	long long *value
) {
	bool res = _Config_ParseInteger(integer_str, value);
	// Return an error code if integer parsing fails or value is not positive.
	return (res == true && *value > 0);
}

// parse non-negative integer
// return true if string represents an integer >= 0
static inline bool _Config_ParseNonNegativeInteger
(
	const char *integer_str,
	long long *value
) {
	bool res = _Config_ParseInteger(integer_str, value);
	// Return an error code if integer parsing fails or value is negative.
	return (res == true && *value >= 0);
}

// return true if 'str' is either "yes" or "no" otherwise returns false
// sets 'value' to true if 'str' is "yes"
// sets 'value to false if 'str' is "no"
static inline bool _Config_ParseYesNo
(
	const char *str,
	bool *value
) {
	bool res = false;

	if(!strcasecmp(str, "yes")) {
		res = true;
		*value = true;
	} else if(!strcasecmp(str, "no")) {
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

static uint64_t Config_resultset_max_size_get(void) {
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
	uint64_t buf_size
) {
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

static uint64_t Config_max_info_queries_get(void) {
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
	rm_free(config.import_folder);

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

const char *Config_Field_name
(
	Config_Option_Field field
) {
	const char *name = NULL;
	switch(field) {
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

		case Config_DEDUPLICATE_STRINGS:
			name = DEDUPLICATE_STRINGS;
			break;

		//----------------------------------------------------------------------
		// invalid option
		//----------------------------------------------------------------------

		default :
			ASSERT("invalid option field" && false);
			break;
	}

	return name;
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
	return config.config_attr;                                \
}

#define CONFIG_GET_NUMERIC(config_attr)                       \
long long CONFIG_GET_NUMERIC_FUNC_NAME(config_attr)           \
(                                                             \
	const char *name,                                         \
	void *privdata                                            \
) {                                                           \
	return config.config_attr;                                \
}

#define CONFIG_GET_STRING(config_attr)                        \
RedisModuleString* CONFIG_GET_STRING_FUNC_NAME(config_attr)   \
(                                                             \
	const char *name,                                         \
	void *privdata                                            \
) {                                                           \
	return RedisModule_CreateString(NULL,                     \
			config.config_attr, strlen(config.config_attr));  \
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


// generate config set function name for an individual configuration attribute
#define CONFIG_SET_BOOL_FUNC_NAME(config_attr) _Config_Set_##config_attr##_Bool
#define CONFIG_SET_NUMERIC_FUNC_NAME(config_attr) _Config_Set_##config_attr##_String
#define CONFIG_SET_STRING_FUNC_NAME(config_attr) _Config_Set_##config_attr##_Numeric

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
			CMD_INFO_DEFAULT,
			REDISMODULE_CONFIG_DEFAULT,
			CONFIG_GET_BOOL_FUNC_NAME(cmd_info),
			CONFIG_SET_BOOL_FUNC_NAME(cmd_info),
			NULL,
			NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterBoolConfig(ctx,
			ASYNC_DELETE,
			ASYNC_DELETE_DEFAULT,
			REDISMODULE_CONFIG_DEFAULT,
			CONFIG_GET_BOOL_FUNC_NAME(async_delete),
			CONFIG_SET_BOOL_FUNC_NAME(async_delete),
			NULL,
			NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterBoolConfig(ctx,
			DEDUPLICATE_STRINGS,
			DEDUPLICATE_STRINGS_DEFAULT,
			REDISMODULE_CONFIG_DEFAULT,
			CONFIG_GET_BOOL_FUNC_NAME(deduplicate_strings),
			CONFIG_SET_BOOL_FUNC_NAME(deduplicate_strings),
			NULL,
			NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterBoolConfig(ctx,
			DELAY_INDEXING,
			DELAY_INDEXING_DEFAULT,
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
										  CONFIG_TIMEOUT_DEFAULT,
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
										  BOLT_PROTOCOL_PORT_DEFAULT,
										  REDISMODULE_CONFIG_DEFAULT,
										  BOLT_PROTOCOL_PORT_MIN,
										  BOLT_PROTOCOL_PORT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(bolt_port),
										  CONFIG_SET_NUMERIC_FUNC_NAME(bolt_port),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  CACHE_SIZE,
										  CACHE_SIZE_DEFAULT,
										  REDISMODULE_CONFIG_DEFAULT,
										  CACHE_SIZE_MIN,
										  CACHE_SIZE_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(cache_size),
										  CONFIG_SET_NUMERIC_FUNC_NAME(cache_size),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  TIMEOUT_MAX,
										  CONFIG_TIMEOUT_DEFAULT,
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
										  RESULTSET_SIZE_DEFAULT,
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
										  CONFIG_TIMEOUT_DEFAULT,
										  REDISMODULE_CONFIG_DEFAULT,
										  CONFIG_TIMEOUT_MIN,
										  CONFIG_TIMEOUT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(timeout_default),
										  CONFIG_SET_NUMERIC_FUNC_NAME(timeout_default),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  THREAD_COUNT,
										  THREAD_COUNT_DEFAULT,
										  REDISMODULE_CONFIG_DEFAULT,
										  THREAD_COUNT_MIN,
										  THREAD_COUNT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(thread_pool_size),
										  CONFIG_SET_NUMERIC_FUNC_NAME(thread_pool_size),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  OMP_THREAD_COUNT,
										  OMP_THREAD_COUNT_DEFAULT,
										  REDISMODULE_CONFIG_DEFAULT,
										  OMP_THREAD_COUNT_MIN,
										  OMP_THREAD_COUNT_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(omp_thread_count),
										  CONFIG_SET_NUMERIC_FUNC_NAME(omp_thread_count),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  VKEY_MAX_ENTITY_COUNT,
										  VKEY_MAX_ENTITY_COUNT_DEFAULT,
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
										  EFFECTS_THRESHOLD_DEFAULT,
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
										  QUEUED_QUERIES_DEFAULT,
										  REDISMODULE_CONFIG_DEFAULT,
										  QUEUED_QUERIES_MIN,
										  QUEUED_QUERIES_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(max_queued_queries),
										  CONFIG_SET_NUMERIC_FUNC_NAME(max_queued_queries),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  QUERY_MEM_CAPACITY,
										  QUERY_MEM_CAPACITY_DEFAULT,
										  REDISMODULE_CONFIG_DEFAULT,
										  QUERY_MEM_CAPACITY_MIN,
										  QUERY_MEM_CAPACITY_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(query_mem_capacity),
										  CONFIG_SET_NUMERIC_FUNC_NAME(query_mem_capacity),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  NODE_CREATION_BUFFER,
										  NODE_CREATION_BUFFER_DEFAULT,
										  REDISMODULE_CONFIG_DEFAULT,
										  NODE_CREATION_BUFFER_MIN,
										  NODE_CREATION_BUFFER_MAX,
										  CONFIG_GET_NUMERIC_FUNC_NAME(node_creation_buffer),
										  CONFIG_SET_NUMERIC_FUNC_NAME(node_creation_buffer),
										  NULL,
										  NULL);
	ASSERT(res == REDISMODULE_OK);

	res = RedisModule_RegisterNumericConfig(ctx,
										  CMD_INFO_MAX_QUERIES_COUNT_OPTION_NAME,
										  CMD_INFO_QUERIES_MAX_COUNT_DEFAULT,
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
										  DELTA_MAX_PENDING_CHANGES_DEFAULT,
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
										  IMPORT_DIR_DEFAULT,
										  REDISMODULE_CONFIG_DEFAULT,
										  CONFIG_GET_STRING_FUNC_NAME(import_folder),
										  CONFIG_SET_STRING_FUNC_NAME(import_folder),
										  NULL,
										  NULL);

}

// initialize every module-level configuration to its default value
static void _Config_SetToDefaults(void) {
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
	config.cmd_info = CMD_INFO_DEFAULT;

	// GRAPH.INFO maximum queries count.
	config.max_info_queries_count = CMD_INFO_QUERIES_MAX_COUNT_DEFAULT;

	// replicate effects if avg change time μs > effects_threshold μs
	config.effects_threshold = 300 ;

	// bolt protocol port (disabled by default)
	config.bolt_port = BOLT_PROTOCOL_PORT_DEFAULT;

	// index entities as they're being decoded
	config.delay_indexing = DELAY_INDEXING_DEFAULT;

	// set default import folder path
	config.import_folder = rm_strdup(IMPORT_DIR_DEFAULT);

	// set default deduplicate strings
	config.deduplicate_strings = DEDUPLICATE_STRINGS_DEFAULT;
}

int Config_Init
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	// make sure reconfiguration callback is already registered
	ASSERT(config.cb != NULL);

	_Config_Register(ctx);

	// initialize the configuration to its default values
	_Config_SetToDefaults();

	int res = RedisModule_LoadConfigs(ctx);
	ASSERT(res == REDISMODULE_OK);

	if(argc % 2) {
		// emit an error if we received an odd number of arguments,
		// as this indicates an invalid configuration
		RedisModule_Log(ctx, "warning",
						"FalkorDB received %d arguments, all configurations should be key-value pairs", argc);
		return REDISMODULE_ERR;
	}

	bool old_timeout_specified = false;
	bool new_timeout_specified = false;

	for(int i = 0; i < argc; i += 2) {
		// each configuration is a key-value pair. (K, V)

		//----------------------------------------------------------------------
		// get field
		//----------------------------------------------------------------------

		Config_Option_Field field;
		RedisModuleString *val = argv[i + 1];
		const char *field_str = RedisModule_StringPtrLen(argv[i], NULL);
		const char *val_str = RedisModule_StringPtrLen(val, NULL);

		// exit if configuration is not aware of field
		if(!Config_Contains_field(field_str, &field)) {
			RedisModule_Log(ctx, "error",
							"Encountered unknown configuration field '%s'", field_str);
			return REDISMODULE_ERR;
		}

		if(field == Config_TIMEOUT_DEFAULT || field == Config_TIMEOUT_MAX) {
			new_timeout_specified = true;
		}

		if(field == Config_TIMEOUT) {
			old_timeout_specified = true;
		}

		// exit if encountered an error when setting configuration
		char *error = NULL;
		if(!Config_Option_set(field, val_str, &error)) {
			if(error != NULL) {
				RedisModule_Log(ctx, "error",
							"Failed setting field '%s' with error: %s",
							field_str, error);
			} else {
				RedisModule_Log(ctx, "error",
						"Failed setting field '%s'", field_str);
			}
			return REDISMODULE_ERR;
		}
	}

	if(old_timeout_specified && new_timeout_specified) {
		RedisModule_Log(ctx, "error",
						"The TIMEOUT configuration parameter should be removed when specifying TIMEOUT_DEFAULT and/or TIMEOUT_MAX");
		return REDISMODULE_ERR;
	}

	return REDISMODULE_OK;
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
			(*resultset_max_size) = Config_resultset_max_size_get();
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
			bool *cmd_info = va_arg(ap, bool *);
			va_end(ap);

			ASSERT(cmd_info != NULL);
			(*cmd_info) = Config_cmd_info_get();
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
			(*count) = Config_max_info_queries_get();
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

bool Config_Option_set
(
	Config_Option_Field field,
	const char *val,
	char **err
) {
	//--------------------------------------------------------------------------
	// set the option
	//--------------------------------------------------------------------------

	switch(field) {
		//----------------------------------------------------------------------
		// max queued queries
		//----------------------------------------------------------------------

		case Config_MAX_QUEUED_QUERIES: {
			long long max_queued_queries;
			if(!_Config_ParsePositiveInteger(val, &max_queued_queries)) {
				return false;
			}
			Config_max_queued_queries_set(max_queued_queries);
		}
		break;

		//----------------------------------------------------------------------
		// timeout
		//----------------------------------------------------------------------

		case Config_TIMEOUT: {
			long long timeout;
			if(!_Config_ParseNonNegativeInteger(val, &timeout)) return false;
			if(!_Config_check_if_new_timeout_used()) {
				if(err) *err = "The TIMEOUT configuration parameter is deprecated. Please set TIMEOUT_MAX and TIMEOUT_DEFAULT instead";
				return false;
			}
			Config_timeout_set(timeout);
		}
		break;

		//----------------------------------------------------------------------
		// timeout default
		//----------------------------------------------------------------------

		case Config_TIMEOUT_DEFAULT: {
			long long timeout_default;
			if(!_Config_ParseNonNegativeInteger(val, &timeout_default)) return false;
			if(!Config_timeout_default_set(timeout_default)) {
				if(err) *err = "TIMEOUT_DEFAULT configuration parameter cannot be set to a value higher than TIMEOUT_MAX";
				return false;
			}
		}
		break;

		//----------------------------------------------------------------------
		// timeout max
		//----------------------------------------------------------------------

		case Config_TIMEOUT_MAX: {
			long long timeout_max;
			if(!_Config_ParseNonNegativeInteger(val, &timeout_max)) return false;
			if(!Config_timeout_max_set(timeout_max)) {
				if(err) *err = "TIMEOUT_MAX configuration parameter cannot be set to a value lower than TIMEOUT_DEFAULT";
				return false;
			}
		}
		break;

		//----------------------------------------------------------------------
		// cache size
		//----------------------------------------------------------------------

		case Config_CACHE_SIZE: {
			long long cache_size;
			if(!_Config_ParsePositiveInteger(val, &cache_size)) return false;
			Config_cache_size_set(cache_size);
		}
		break;

		//----------------------------------------------------------------------
		// OpenMP thread count
		//----------------------------------------------------------------------

		case Config_OPENMP_NTHREAD: {
			long long omp_nthreads;
			if(!_Config_ParsePositiveInteger(val, &omp_nthreads)) return false;

			Config_omp_thread_count_set(omp_nthreads);
		}
		break;

		//----------------------------------------------------------------------
		// thread-pool size
		//----------------------------------------------------------------------

		case Config_THREAD_POOL_SIZE: {
			long long pool_nthreads;
			if(!_Config_ParsePositiveInteger(val, &pool_nthreads)) return false;

			Config_thread_pool_size_set(pool_nthreads);
		}
		break;

		//----------------------------------------------------------------------
		// result-set size
		//----------------------------------------------------------------------

		case Config_RESULTSET_MAX_SIZE: {
			long long resultset_max_size;
			if(!_Config_ParseInteger(val, &resultset_max_size)) return false;

			Config_resultset_size_set(resultset_max_size);
		}
		break;

		//----------------------------------------------------------------------
		// virtual key entity count
		//----------------------------------------------------------------------

		case Config_VKEY_MAX_ENTITY_COUNT: {
			long long vkey_max_entity_count;
			if(!_Config_ParseNonNegativeInteger(val, &vkey_max_entity_count)) return false;

			Config_vkey_entity_count_set(vkey_max_entity_count);
		}
		break;

		//----------------------------------------------------------------------
		// async deleteion
		//----------------------------------------------------------------------

		case Config_ASYNC_DELETE: {
			bool async_delete;
			if(!_Config_ParseYesNo(val, &async_delete)) return false;

			Config_async_delete_set(async_delete);
		}
		break;

		//----------------------------------------------------------------------
		// query mem capacity
		//----------------------------------------------------------------------

		case Config_QUERY_MEM_CAPACITY: {
			long long query_mem_capacity;
			if(!_Config_ParseNonNegativeInteger(val, &query_mem_capacity)) return false;

			Config_query_mem_capacity_set(query_mem_capacity);
		}
		break;

		//----------------------------------------------------------------------
		// number of pending changed befor Delta_Matrix flushed
		//----------------------------------------------------------------------

		case Config_DELTA_MAX_PENDING_CHANGES: {
			long long delta_max_pending_changes;
			if(!_Config_ParseNonNegativeInteger(val, &delta_max_pending_changes)) return false;

			Config_delta_max_pending_changes_set(delta_max_pending_changes);
		}
		break;

		//----------------------------------------------------------------------
		// size of buffer to maintain as margin in matrices
		//----------------------------------------------------------------------

		case Config_NODE_CREATION_BUFFER: {
			long long node_creation_buffer;
			if(!_Config_ParseNonNegativeInteger(val, &node_creation_buffer)) return false;

			// node_creation_buffer should be at-least 128
			node_creation_buffer =
				(node_creation_buffer < 128) ? 128: node_creation_buffer;

			// retrieve the MSB of the value
			long long msb = (sizeof(long long) * 8) - __builtin_clzll(node_creation_buffer);
			long long set_msb = 1 << (msb - 1);

			// if the value is not a power of 2
			// (if any bits other than the MSB are 1),
			// raise it to the next power of 2
			if((~set_msb & node_creation_buffer) != 0) {
				node_creation_buffer = 1 << msb;
			}
			Config_node_creation_buffer_set(node_creation_buffer);
		}
		break;

		//----------------------------------------------------------------------
		// cmd info
		//----------------------------------------------------------------------

		case Config_CMD_INFO: {
			bool cmd_info = false;
			if (!_Config_ParseYesNo(val, &cmd_info)) {
				return false;
			}

			Config_cmd_info_set(cmd_info);
		}
		break;

		//----------------------------------------------------------------------
		// cmd info max queries count
		//----------------------------------------------------------------------

		case Config_CMD_INFO_MAX_QUERY_COUNT: {
			long long count = 0;
			if (!_Config_ParseNonNegativeInteger(val, &count)) return false;
			if (count > UINT64_MAX) return false;

			Config_max_info_queries_count_set(count);
		}
  		break;

		//----------------------------------------------------------------------
		// effects threshold
		//----------------------------------------------------------------------
				
		case Config_EFFECTS_THRESHOLD: {
			long long threshold;
			if(!_Config_ParseNonNegativeInteger(val, &threshold)) {
				return false;
			}
			Config_effects_threshold_set(threshold);
		}
		break;

		//----------------------------------------------------------------------
		// bolt protocol port
		//----------------------------------------------------------------------

		case Config_BOLT_PORT: {
			long long port;
			if(!_Config_ParseInteger(val, &port)) {
				return false;
			}
			Config_bolt_port_set(port);
		}
		break;

		//----------------------------------------------------------------------
		// delay indexing
		//----------------------------------------------------------------------

		case Config_DELAY_INDEXING: {
			bool delay_indexing;
			if(!_Config_ParseYesNo(val, &delay_indexing)) return false;

			Config_delay_indexing_set(delay_indexing);
		}
		break;

		//----------------------------------------------------------------------
		// import folder path
		//----------------------------------------------------------------------

		case Config_IMPORT_FOLDER: {
			ASSERT(val != NULL);
			Config_import_folder_set(val);
		}
		break;

		//----------------------------------------------------------------------
		// deduplicate strings
		//----------------------------------------------------------------------

		case Config_DEDUPLICATE_STRINGS: {
			bool enabled;
			if(!_Config_ParseYesNo(val, &enabled)) return false;

			Config_deduplicate_strings_set(enabled);
		}
		break;

		//----------------------------------------------------------------------
		// invalid option
		//----------------------------------------------------------------------

		default:
			return false;
	}

	if(config.cb) config.cb(field);

	return true;
}

// dry run configuration change
bool Config_Option_dryrun
(
	Config_Option_Field field,
	const char *val,
	char **err
) {
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

void Config_Subscribe_Changes
(
	Config_on_change cb
) {
	ASSERT(cb != NULL);
	config.cb = cb;
}

