/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../../serializers_include.h"

void RdbLoadGraphContext_latest
(
	SerializerIO rdb,
	GraphContext *gc
);

// decode nodes
void RdbLoadNodes_v17
(
	SerializerIO rdb,          // RDB
	Graph *g,                  // graph context
	const uint64_t node_count  // number of nodes to decode
);

// decode deleted nodes
void RdbLoadDeletedNodes_v17
(
	SerializerIO rdb,                  // RDB
	Graph *g,                          // graph context
	const uint64_t deleted_node_count  // number of deleted nodes
);

// decode edges
void RdbLoadEdges_v17
(
	SerializerIO rdb,  // RDB
	Graph *g,          // graph context
	const uint64_t n   // virtual key capacity
);

// decode deleted edges
void RdbLoadDeletedEdges_v17
(
	SerializerIO rdb,                  // RDB
	Graph *g,                          // graph context
	const uint64_t deleted_edge_count  // number of deleted edges
);

void RdbLoadGraphSchema_v17
(
	SerializerIO rdb,
	GraphContext *gc,
	bool already_loaded
);

void RdbLoadLabelMatrices_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
);

void RdbLoadRelationMatrices_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
);

// decode adjacency matrix
void RdbLoadAdjMatrix_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
);

void RdbLoadLblsMatrix_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc   // graph context
);

