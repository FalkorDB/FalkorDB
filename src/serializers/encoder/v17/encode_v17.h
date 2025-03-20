/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../serializers_include.h"

void RdbSaveGraph_latest
(
	SerializerIO rdb,
	void *value
);

// encode nodes
void RdbSaveNodes_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // iterator offset
	const uint64_t n   // number of nodes to encode
);

// encode deleted node IDs
void RdbSaveDeletedNodes_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	const uint64_t n   // number of deleted nodes to encode
);

// encode edges
void RdbSaveEdges_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	const uint64_t n   // number of edges to encode
);

// encode deleted edges IDs
void RdbSaveDeletedEdges_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	const uint64_t n   // number of deleted edges to encode
);

void RdbSaveGraphSchema_v17
(
	SerializerIO rdb,
	GraphContext *gc
);

// encode label matrices to rdb
void RdbSaveLabelMatrices_v17
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
);

// encode relationship matrices to rdb
void RdbSaveRelationMatrices_v17
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
);

// encode graph's adjacency matrix
void RdbSaveAdjMatrix_v17
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
);

// encode graph's labels matrix
void RdbSaveLblsMatrix_v17
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
);

