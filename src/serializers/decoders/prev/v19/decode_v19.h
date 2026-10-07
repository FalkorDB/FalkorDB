/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../../serializers_include.h"

GraphContext *RdbLoadGraphContext_v19
(
	SerializerIO rdb,
	const RedisModuleString *rm_key_name,
	bool detached
);

// encode DB UDFs
void AUXLoadUDF_v19
(
	RedisModuleIO *io  // IO
);

// decode nodes
void RdbLoadNodes_v19
(
	SerializerIO rdb,          // RDB
	Graph *g,                  // graph context
	const uint64_t node_count, // number of nodes to decode
	const uint64_t id_limit    // node ids are below this (header counts)
);

// decode deleted nodes
void RdbLoadDeletedNodes_v19
(
	SerializerIO rdb,                   // RDB
	Graph *g,                           // graph context
	const uint64_t deleted_node_count,  // number of deleted nodes
	const uint64_t id_limit             // node ids are below this
);

// decode edges
void RdbLoadEdges_v19
(
	SerializerIO rdb,         // RDB
	Graph *g,                 // graph context
	const uint64_t n,         // virtual key capacity
	const uint64_t id_limit   // edge ids are below this (header counts)
);

// decode deleted edges
void RdbLoadDeletedEdges_v19
(
	SerializerIO rdb,                   // RDB
	Graph *g,                           // graph context
	const uint64_t deleted_edge_count,  // number of deleted edges
	const uint64_t id_limit             // edge ids are below this
);

void RdbLoadGraphSchema_v19
(
	SerializerIO rdb,
	GraphContext *gc,
	bool already_loaded
);

void RdbLoadLabelMatrices_v19
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
);

void RdbLoadRelationMatrices_v19
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
);

// decode adjacency matrix
void RdbLoadAdjMatrix_v19
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
);

void RdbLoadLblsMatrix_v19
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
);

