/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_graph.h"
#include "current/v14/decode_v14.h"

GraphContext *RdbLoadGraph(RedisModuleIO *rdb) {
	return RdbLoadGraphContext_v14(rdb);
}

