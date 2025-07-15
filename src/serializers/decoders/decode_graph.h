/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../serializers_include.h"

// load a graph from SerializerIO
GraphContext *SerializerLoadGraph
(
	SerializerIO io,       // serializer
	const char *key_name,  // load graph under this key name
	int encver             // encoder version
);

// load a graph from RDB
GraphContext *RdbLoadGraph
(
	RedisModuleIO *rdb,  // RDB
	int encver           // encoder version
);

// load a graph virtual key from RDB
void *RdbLoadMetaGraph
(
	RedisModuleIO *rdb,  // RDB
	int encver           // encoder version
);

