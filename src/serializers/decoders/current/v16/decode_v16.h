/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../../serializers_include.h"

GraphContext *RdbLoadGraphContext_latest
(
	SerializerIO rdb,
	const RedisModuleString *rm_key_name
);

void RdbLoadNodes_v16
(
	SerializerIO rdb,
	GraphContext *gc
);

void RdbLoadDeletedNodes_v16
(
	SerializerIO rdb,
	GraphContext *gc
);

void RdbLoadEdges_v16
(
	SerializerIO rdb,
	GraphContext *gc
);

void RdbLoadDeletedEdges_v16
(
	SerializerIO rdb,
	GraphContext *gc
);

void RdbLoadGraphSchema_v16
(
	SerializerIO rdb,
	GraphContext *gc,
	bool already_loaded
);

