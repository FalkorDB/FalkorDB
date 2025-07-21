/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "decode_graph.h"
#include "current/v13/decode_v13.h"

GraphContext *RdbLoadGraph(RedisModuleIO *rdb) {
	return RdbLoadGraphContext_v13(rdb);
}

