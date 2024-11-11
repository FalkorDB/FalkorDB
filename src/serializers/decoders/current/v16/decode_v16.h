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
	GraphContext *gc,
	const uint64_t node_count
);

void RdbLoadDeletedNodes_v16
(
	SerializerIO rdb,
	GraphContext *gc,
	const uint64_t deleted_node_count
);

void RdbLoadEdges_v16
(
	SerializerIO rdb,
	GraphContext *gc,
	const uint64_t n
);

void RdbLoadDeletedEdges_v16
(
	SerializerIO rdb,
	GraphContext *gc,
	const uint64_t deleted_edge_count
);

void RdbLoadGraphSchema_v16
(
	SerializerIO rdb,
	GraphContext *gc,
	bool already_loaded
);

