/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "redismodule.h"

#include <stdbool.h>

//------------------------------------------------------------------------------
// Enterprise - FalkorDB exported API
//
// function pointer types for APIs the enterprise module exports via
// RedisModule_GetSharedAPI. Any translation unit that needs to resolve one
// of these should include this header rather than redeclaring the typedef
// locally, so the signatures can't drift out of sync between call sites.
//------------------------------------------------------------------------------

// result of an attempt to load an offloaded graph stub from disk
typedef enum {
    GraphLoad_SUCCESS,      // graph restored from disk successfully
    GraphLoad_LOADING,      // another load for this key is already in progress
    GraphLoad_OFFLOADING,   // an offload for this key is already in progress
    GraphLoad_KEY_MISSING,  // key does not exist
    GraphLoad_NOT_STUB,     // key exists but is not an offloaded graph stub
    GraphLoad_DUMP_MISSING, // stub is valid but its dump file was not found
    GraphLoad_OOM,          // not enough memory to hold the loaded graph
    GraphLoad_ERR,          // all other failures
} GraphLoadResult ;

// "GraphStubType_Get" - returns the RedisModuleType representing an
// offloaded graph stub
typedef RedisModuleType *(*GraphStubType_Get_t) (void) ;

// "graph_load" - loads an offloaded graph stub from disk, replacing the
// stub key's value with a live GraphContext
typedef GraphLoadResult (*graph_load_t)
(
    RedisModuleCtx    *ctx,
    RedisModuleString *key_name,
    bool              from_thread,
	bool              force,
	bool              bypass_claim
) ;
