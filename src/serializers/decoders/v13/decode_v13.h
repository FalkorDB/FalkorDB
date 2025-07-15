/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../serializers_include.h"

void RdbLoadGraphContext_v13
(
	SerializerIO rdb,
	GraphContext *gc
);

void RdbLoadNodes_v13
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t node_count
);

void RdbLoadDeletedNodes_v13
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t deleted_node_count
);

void RdbLoadEdges_v13
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t edge_count
);

void RdbLoadDeletedEdges_v13
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t deleted_edge_count
);

void RdbLoadGraphSchema_v13
(
	SerializerIO rdb,
	GraphContext *gc,
	bool already_loaded
);

