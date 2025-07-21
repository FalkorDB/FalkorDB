/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../../serializers_include.h"

void RdbSaveGraph_v13
(
	RedisModuleIO *rdb,
	void *value
);

void RdbSaveNodes_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	uint64_t nodes_to_encode
);

void RdbSaveDeletedNodes_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	uint64_t deleted_nodes_to_encode
);

void RdbSaveEdges_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	uint64_t edges_to_encode
);

void RdbSaveDeletedEdges_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	uint64_t deleted_edges_to_encode
);

void RdbSaveGraphSchema_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc
);

