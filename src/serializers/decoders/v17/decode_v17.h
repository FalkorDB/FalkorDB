/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../serializers_include.h"

bool RdbLoadGraphContext_v17
(
	SerializerIO io,
	GraphContext *gc
);

// decode nodes
bool RdbLoadNodes_v17
(
	SerializerIO io,           // RDB
	Graph *g,                  // graph context
	const uint64_t node_count  // number of nodes to decode
);

// decode deleted nodes
bool RdbLoadDeletedNodes_v17
(
	SerializerIO io,                   // RDB
	Graph *g,                          // graph context
	const uint64_t deleted_node_count  // number of deleted nodes
);

// decode edges
bool RdbLoadEdges_v17
(
	SerializerIO io,  // RDB
	Graph *g,         // graph context
	const uint64_t n  // virtual key capacity
);

// decode deleted edges
bool RdbLoadDeletedEdges_v17
(
	SerializerIO io,                   // RDB
	Graph *g,                          // graph context
	const uint64_t deleted_edge_count  // number of deleted edges
);

bool RdbLoadGraphSchema_v17
(
	SerializerIO io,
	GraphContext *gc,
	bool already_loaded
);

bool RdbLoadLabelMatrices_v17
(
	SerializerIO io,  // RDB
	GraphContext *gc  // graph context
);

bool RdbLoadRelationMatrices_v17
(
	SerializerIO io,  // RDB
	GraphContext *gc  // graph context
);

// decode adjacency matrix
bool RdbLoadAdjMatrix_v17
(
	SerializerIO io,  // RDB
	GraphContext *gc  // graph context
);

bool RdbLoadLblsMatrix_v17
(
	SerializerIO io,  // RDB
	GraphContext *gc  // graph context
);

