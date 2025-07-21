/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"

// create blocked client and report start time
RedisModuleBlockedClient *RedisGraph_BlockClient
(
    RedisModuleCtx *ctx
);

// unblock blocked client and report end time
void RedisGraph_UnblockClient
(
    RedisModuleBlockedClient *bc
);