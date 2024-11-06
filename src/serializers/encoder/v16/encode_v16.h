/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../serializers_include.h"

void RdbSaveGraph_latest
(
	SerializerIO rdb,
	void *value
);

// encode nodes
// returns number of nodes encoded
uint64_t RdbSaveNodes_v16
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // iterator offset
	uint64_t n         // number of nodes to encode
);

// encode deleted node IDs
// return number of elements encoded
uint64_t RdbSaveDeletedNodes_v16
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	uint64_t n         // number of deleted nodes to encode
);

// encode edges
// returns number of encoded edges.
uint64_t RdbSaveEdges_v16
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	uint64_t n         // number of edges to encode
);

// encode deleted edges IDs
// return number of elements encoded
uint64_t RdbSaveDeletedEdges_v16
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	uint64_t n         // number of deleted edges to encode
);

void RdbSaveGraphSchema_v16
(
	SerializerIO rdb,
	GraphContext *gc
);
