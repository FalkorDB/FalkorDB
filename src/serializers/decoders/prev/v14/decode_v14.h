/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../../serializers_include.h"
#include "../../../../graph/graph_limits.h"  // GRAPH_MAX_* decode bounds

GraphContext *RdbLoadGraphContext_v14
(
	SerializerIO rdb,
	const RedisModuleString *rm_key_name
);

void RdbLoadNodes_v14
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t node_count,
	uint64_t id_limit        // node ids are below this (header counts)
);

void RdbLoadDeletedNodes_v14
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t deleted_node_count,
	uint64_t id_limit        // node ids are below this
);

void RdbLoadEdges_v14
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t edge_count,
	uint64_t node_id_limit,  // src/dest node ids are below this
	uint64_t edge_id_limit   // edge ids are below this
);

void RdbLoadDeletedEdges_v14
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t deleted_edge_count,
	uint64_t id_limit        // edge ids are below this
);

void RdbLoadGraphSchema_v14
(
	SerializerIO rdb,
	GraphContext *gc,
	bool already_loaded
);

