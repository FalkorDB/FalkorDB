/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../../serializers_include.h"

GraphContext *RdbLoadGraphContext_latest
(
	SerializerIO io,
	const RedisModuleString *rm_key_name
);

// decode nodes
bool RdbLoadNodes_v18
(
	SerializerIO io,           // stream
	Graph *g,                  // graph context
	const uint64_t node_count  // number of nodes to decode
);

// decode deleted nodes
bool RdbLoadDeletedNodes_v18
(
	SerializerIO io,                   // stream
	Graph *g,                          // graph context
	const uint64_t deleted_node_count  // number of deleted nodes
);

// decode edges
bool RdbLoadEdges_v18
(
	SerializerIO io,   // stream
	Graph *g,          // graph context
	const uint64_t n   // virtual key capacity
);

// decode deleted edges
bool RdbLoadDeletedEdges_v18
(
	SerializerIO io,                   // stream
	Graph *g,                          // graph context
	const uint64_t deleted_edge_count  // number of deleted edges
);

bool RdbLoadGraphSchema_v18
(
	SerializerIO io,
	GraphContext *gc,
	bool already_loaded
);

bool LoadLabelMatrices_v18
(
	SerializerIO io,   // stream
	GraphContext *gc   // graph context
);

bool LoadRelationMatrices_v18
(
	SerializerIO io,   // stream
	GraphContext *gc   // graph context
);

// decode adjacency matrix
bool LoadAdjMatrix_v18
(
	SerializerIO io,   // stream
	GraphContext *gc   // graph context
);

bool LoadLblsMatrix_v18
(
	SerializerIO io,   // stream
	GraphContext *gc   // graph context
);

