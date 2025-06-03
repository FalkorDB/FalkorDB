/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"
#include "redismodule.h"

#include <stdbool.h>

//-----------------------------------------------------------------------------
// Configuration parameters
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// timeout config
//-----------------------------------------------------------------------------

#define TIMEOUT         "TIMEOUT"          // query timeout in milliseconds
#define TIMEOUT_MAX     "TIMEOUT_MAX"      // max timeout that can be enforced
#define TIMEOUT_DEFAULT "TIMEOUT_DEFAULT"  // default timeout

#define CONFIG_TIMEOUT_MIN                 CONFIG_TIMEOUT_DEFAULT
#define CONFIG_TIMEOUT_MAX                 LLONG_MAX
#define CONFIG_TIMEOUT_DEFAULT             CONFIG_TIMEOUT_NO_TIMEOUT
#define CONFIG_TIMEOUT_NO_TIMEOUT          0


//-----------------------------------------------------------------------------
// cache size config
//-----------------------------------------------------------------------------

// size of each thread cache size, per graph
#define CACHE_SIZE "CACHE_SIZE"

#define CACHE_SIZE_MIN                     0
#define CACHE_SIZE_MAX                     512
#define CACHE_SIZE_DEFAULT                 25

//-----------------------------------------------------------------------------
// async delete config
//-----------------------------------------------------------------------------

// whether graphs should be deleted asynchronously
#define ASYNC_DELETE "ASYNC_DELETE"
#ifdef MEMCHECK
	#define ASYNC_DELETE_DEFAULT           false
#else
	#define ASYNC_DELETE_DEFAULT           true
#endif

//-----------------------------------------------------------------------------
// thread pool size config
//-----------------------------------------------------------------------------

// number of threads in thread pool
#define THREAD_COUNT "THREAD_COUNT"
#define THREAD_COUNT_MIN                   THREAD_COUNT_DEFAULT
#define THREAD_COUNT_MAX                   128
#define THREAD_COUNT_DEFAULT               1


//-----------------------------------------------------------------------------
// result-set size config
//-----------------------------------------------------------------------------

// resultset size limit
#define RESULTSET_SIZE "RESULTSET_SIZE"
#define RESULTSET_SIZE_UNLIMITED           LLONG_MAX
#define RESULTSET_SIZE_MIN                 -1
#define RESULTSET_SIZE_MAX                 RESULTSET_SIZE_UNLIMITED
#define RESULTSET_SIZE_DEFAULT             RESULTSET_SIZE_UNLIMITED


//-----------------------------------------------------------------------------
// OpenMP thread count config
//-----------------------------------------------------------------------------

// max number of OpenMP threads
#define OMP_THREAD_COUNT "OMP_THREAD_COUNT"
#define OMP_THREAD_COUNT_DEFAULT           1
#define OMP_THREAD_COUNT_MIN               OMP_THREAD_COUNT_DEFAULT
#define OMP_THREAD_COUNT_MAX               128


//-----------------------------------------------------------------------------
// virtual key size config
//-----------------------------------------------------------------------------

// max number of entities in each virtual key
#define VKEY_MAX_ENTITY_COUNT "VKEY_MAX_ENTITY_COUNT"
#define VKEY_MAX_ENTITY_COUNT_MIN          1
#define VKEY_MAX_ENTITY_COUNT_MAX          LLONG_MAX
#define VKEY_MAX_ENTITY_COUNT_DEFAULT      100000
#define VKEY_ENTITY_COUNT_UNLIMITED        LLONG_MAX

//-----------------------------------------------------------------------------
// max queued queries config
//-----------------------------------------------------------------------------

// max number of queued queries
#define MAX_QUEUED_QUERIES "MAX_QUEUED_QUERIES"
#define QUEUED_QUERIES_UNLIMITED           LLONG_MAX  // unlimited
#define QUEUED_QUERIES_DEFAULT             QUEUED_QUERIES_UNLIMITED
#define QUEUED_QUERIES_MIN                 1
#define QUEUED_QUERIES_MAX                 QUEUED_QUERIES_UNLIMITED


//-----------------------------------------------------------------------------
// query memory capacity config
//-----------------------------------------------------------------------------

// max mem (bytes) that query/thread can utilize at any given time
#define QUERY_MEM_CAPACITY "QUERY_MEM_CAPACITY"
#define QUERY_MEM_CAPACITY_UNLIMITED       0
#define QUERY_MEM_CAPACITY_DEFAULT         QUERY_MEM_CAPACITY_UNLIMITED
#define QUERY_MEM_CAPACITY_MIN             QUERY_MEM_CAPACITY_UNLIMITED
#define QUERY_MEM_CAPACITY_MAX             LLONG_MAX


//-----------------------------------------------------------------------------
// delta matrix max pending changes config
//-----------------------------------------------------------------------------

// number of pending changed befor RG_Matrix flushed
#define DELTA_MAX_PENDING_CHANGES "DELTA_MAX_PENDING_CHANGES"
#define DELTA_MAX_PENDING_CHANGES_DEFAULT  10000
#define DELTA_MAX_PENDING_CHANGES_MIN      1
#define DELTA_MAX_PENDING_CHANGES_MAX      LLONG_MAX

//-----------------------------------------------------------------------------
// node buffer initial size config
//-----------------------------------------------------------------------------

// size of node creation buffer
#define NODE_CREATION_BUFFER "NODE_CREATION_BUFFER"
#define NODE_CREATION_BUFFER_DEFAULT       16384
#define NODE_CREATION_BUFFER_MIN           1
#define NODE_CREATION_BUFFER_MAX           LLONG_MAX


//-----------------------------------------------------------------------------
// GRAPH.INFO max size
//-----------------------------------------------------------------------------

// The GRAPH.INFO QUERIES maximum element count
#define CMD_INFO_MAX_QUERIES_COUNT_OPTION_NAME "MAX_INFO_QUERIES"
#define CMD_INFO_QUERIES_MAX_COUNT_DEFAULT 1000
#define CMD_INFO_QUERIES_MAX_COUNT_MIN     0
#define CMD_INFO_QUERIES_MAX_COUNT_MAX     5000

//-----------------------------------------------------------------------------
// effect replication threashold config
//-----------------------------------------------------------------------------

// effects replication threshold
#define EFFECTS_THRESHOLD "EFFECTS_THRESHOLD"
#define EFFECTS_THRESHOLD_DEFAULT          300
#define EFFECTS_THRESHOLD_MIN              0
#define EFFECTS_THRESHOLD_MAX              LLONG_MAX


//-----------------------------------------------------------------------------
// bolt protocol port config
//-----------------------------------------------------------------------------

// bolt protocol port
#define BOLT_PORT "BOLT_PORT"
#define BOLT_PROTOCOL_PORT_DISABLE         -1  // disabled by default
#define BOLT_PROTOCOL_PORT_DEFAULT         BOLT_PROTOCOL_PORT_DISABLE
#define BOLT_PROTOCOL_PORT_MIN             BOLT_PROTOCOL_PORT_DISABLE
#define BOLT_PROTOCOL_PORT_MAX             LLONG_MAX


//-----------------------------------------------------------------------------
// delay indexing config
//-----------------------------------------------------------------------------

// delay indexing
#define DELAY_INDEXING "DELAY_INDEXING"
#define DELAY_INDEXING_DEFAULT             false


//-----------------------------------------------------------------------------
// import folder path config
//-----------------------------------------------------------------------------

// import folder
#define IMPORT_FOLDER "IMPORT_FOLDER"
#define IMPORT_DIR_DEFAULT                 "/var/lib/FalkorDB/import/"


//-----------------------------------------------------------------------------
// deduplicate string config
//-----------------------------------------------------------------------------

// deduplicate string
#define DEDUPLICATE_STRINGS "DEDUPLICATE_STRINGS"
#define DEDUPLICATE_STRINGS_DEFAULT        false

//-----------------------------------------------------------------------------
// cmd info config
//-----------------------------------------------------------------------------

// The GRAPH.INFO command
#define CMD_INFO "CMD_INFO"
#define CMD_INFO_DEFAULT                   true


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
	Config_DEDUPLICATE_STRINGS       = 19,  // use string pool for string dedup
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

bool Config_Option_get
(
	Config_Option_Field field,
	...
);

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
