/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
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

