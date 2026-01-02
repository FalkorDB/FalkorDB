/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"
#include "redismodule.h"

#include <stdbool.h>

#define RESULTSET_SIZE_UNLIMITED          UINT64_MAX
#define CONFIG_TIMEOUT_NO_TIMEOUT         0
#define VKEY_ENTITY_COUNT_UNLIMITED       UINT64_MAX
#define QUERY_MEM_CAPACITY_UNLIMITED      0
#define NODE_CREATION_BUFFER_DEFAULT      16384
#define DELTA_MAX_PENDING_CHANGES_DEFAULT 10000

typedef enum {
	Config_TIMEOUT                   = 0,   // timeout value for queries
	Config_TIMEOUT_DEFAULT           = 1,   // default timeout for read and write queries
	Config_TIMEOUT_MAX               = 2,   // max timeout that can be enforced
	Config_CACHE_SIZE                = 3,   // number of entries in cache
	Config_ASYNC_DELETE              = 4,   // delete graph asynchronously
	Config_OPENMP_NTHREAD            = 5,   // max number of OpenMP threads to use
	Config_THREAD_POOL_SIZE          = 6,   // number of threads in thread pool
	Config_RESULTSET_MAX_SIZE        = 7,   // max number of records in result-set
	Config_VKEY_MAX_ENTITY_COUNT     = 8,   // max number of elements in vkey
	Config_MAX_QUEUED_QUERIES        = 9,   // max number of queued queries
	Config_QUERY_MEM_CAPACITY        = 10,  // max mem(bytes) that query/thread can utilize at any given time
	Config_DELTA_MAX_PENDING_CHANGES = 11,  // number of pending changes before Delta_Matrix flushed
	Config_NODE_CREATION_BUFFER      = 12,  // size of buffer to maintain as margin in matrices
	Config_CMD_INFO                  = 13,  // toggle on/off the GRAPH.INFO
	Config_CMD_INFO_MAX_QUERY_COUNT  = 14,  // the max number of info queries count
	Config_EFFECTS_THRESHOLD         = 15,  // bolt protocol port
	Config_BOLT_PORT                 = 16,  // replicate queries via effects
	Config_DELAY_INDEXING            = 17,  // delay index construction when decoding
	Config_IMPORT_FOLDER             = 18,  // path to CSV import folder
	Config_TEMP_FOLDER               = 19,  // path to temp folder
	Config_END_MARKER                = 20
} Config_Option_Field;

// callback function, invoked once configuration changes as a result of
// successfully executing GRAPH.CONFIG SET
typedef void (*Config_on_change)(Config_Option_Field type);

// Run-time configurable fields
static const Config_Option_Field RUNTIME_CONFIGS[] = {
	Config_TIMEOUT,
	Config_TIMEOUT_MAX,
	Config_ASYNC_DELETE,
	Config_TIMEOUT_DEFAULT,
	Config_RESULTSET_MAX_SIZE,
	Config_MAX_QUEUED_QUERIES,
	Config_QUERY_MEM_CAPACITY,
	Config_VKEY_MAX_ENTITY_COUNT,
	Config_DELTA_MAX_PENDING_CHANGES,
	Config_CMD_INFO,
	Config_CMD_INFO_MAX_QUERY_COUNT,
	Config_EFFECTS_THRESHOLD,
	Config_DELAY_INDEXING
};
static const size_t RUNTIME_CONFIG_COUNT = sizeof(RUNTIME_CONFIGS) / sizeof(RUNTIME_CONFIGS[0]);

// set module-level configurations to defaults or to user provided arguments
// returns REDISMODULE_OK on success
// emits an error and returns REDISMODULE_ERR on failure
int Config_Init
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
);

// returns true if 'field_str' reffers to a configuration field and sets
// 'field' accordingly
bool Config_Contains_field
(
	const char *field_str,
	Config_Option_Field *field
);

// returns the field type
SIType Config_Field_type
(
	Config_Option_Field field  // field
);

// returns field name
const char *Config_Field_name
(
	Config_Option_Field field
);

bool Config_Option_get
(
	Config_Option_Field field,
	...
);

bool Config_Option_set
(
	Config_Option_Field field,
	const char *val,
	char **err
);

// dryrun configuration
bool Config_Option_dryrun
(
	Config_Option_Field field,
	const char *val,
	char **err
);

// sets config update callback function
void Config_Subscribe_Changes
(
	Config_on_change cb
);

