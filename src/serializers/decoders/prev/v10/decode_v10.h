/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../../../serializers_include.h"

GraphContext *RdbLoadGraphContext_v10(RedisModuleIO *rdb);
void RdbLoadNodes_v10(RedisModuleIO *rdb, GraphContext *gc, uint64_t node_count);
void RdbLoadDeletedNodes_v10(RedisModuleIO *rdb, GraphContext *gc, uint64_t deleted_node_count);
void RdbLoadEdges_v10(RedisModuleIO *rdb, GraphContext *gc, uint64_t edge_count);
void RdbLoadDeletedEdges_v10(RedisModuleIO *rdb, GraphContext *gc, uint64_t deleted_edge_count);
void RdbLoadGraphSchema_v10(RedisModuleIO *rdb, GraphContext *gc);

