/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../../serializers_include.h"
#include "../../../../graph/graph_limits.h"  // GRAPH_MAX_* decode bounds

GraphContext *RdbLoadGraphContext_v16
(
	SerializerIO rdb,
	const RedisModuleString *rm_key_name
);

// decode nodes
void RdbLoadNodes_v16
(
	SerializerIO rdb,           // RDB
	GraphContext *gc,           // graph context
	const uint64_t node_count,  // number of nodes to decode
	const uint64_t id_limit     // node ids are below this (header counts)
);

// decode deleted nodes
void RdbLoadDeletedNodes_v16
(
	SerializerIO rdb,                   // RDB
	GraphContext *gc,                   // graph context
	const uint64_t deleted_node_count,  // number of deleted nodes
	const uint64_t id_limit             // node ids are below this
);

// decode edges
void RdbLoadEdges_v16
(
	SerializerIO rdb,             // RDB
	GraphContext *gc,             // graph context
	const uint64_t n,             // number of edges to decode
	const uint64_t node_id_limit, // src/dest node ids are below this
	const uint64_t edge_id_limit  // edge ids are below this
);

// decode deleted edges
void RdbLoadDeletedEdges_v16
(
	SerializerIO rdb,                   // RDB
	GraphContext *gc,                   // graph context
	const uint64_t deleted_edge_count,  // number of deleted edges
	const uint64_t id_limit             // edge ids are below this
);

void RdbLoadGraphSchema_v16
(
	SerializerIO rdb,
	GraphContext *gc,
	bool already_loaded
);

