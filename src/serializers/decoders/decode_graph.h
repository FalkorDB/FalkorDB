/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>

#include "../serializers_include.h"

// load RDB
GraphContext *RdbLoadGraph(RedisModuleIO *rdb);

// log why a graph key failed to load, once, naming the key
// silent for short reads, which Redis reports itself
void RdbLoadGraph_LogFailure
(
	RedisModuleIO *rdb,  // redis IO the key was read from
	SerializerIO io      // serializer that failed
);

// decoder for a graph encoding version, matching RdbLoadGraphContext_latest
typedef GraphContext *(*RdbLoadGraphContext_t)
(
	SerializerIO rdb,
	const RedisModuleString *rm_key_name,
	bool detached
);

// resolve the SerializerIO-based decoder for a given encoding version, or NULL
// when this build ships no such decoder for it
// the graph-offloading module records the encoding version in every dump and
// asks for the matching decoder on load
RdbLoadGraphContext_t Graph_GetDecoder(uint32_t version);
