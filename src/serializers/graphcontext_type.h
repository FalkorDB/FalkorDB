/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "serializers_include.h"

int _GraphContextType_Defrag
(
	RedisModuleDefragCtx *ctx,
	RedisModuleString *key,
	void **value
);

int GraphContextType_Register
(
	RedisModuleCtx *ctx
);

// returns the RedisModuleType* used to tag keys holding a live GraphContext
// exported as a shared API so the enterprise graph-offloading module can
// test RedisModule_ModuleTypeGetType(key) == GraphContextRedisModuleType_Get()
// to tell "this key is a live graph" apart from "this key holds some other,
// unrelated Redis type" - mirrors GraphStubType_Get, which exposes the
// enterprise module's stub type to core in the opposite direction
RedisModuleType *GraphContextRedisModuleType_Get (void) ;

