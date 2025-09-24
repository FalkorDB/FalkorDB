/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "blocked_client.h"

RedisModuleBlockedClient *RedisGraph_BlockClient
(
    RedisModuleCtx *ctx
) {
    ASSERT(ctx != NULL);

    RedisModuleBlockedClient *bc = RedisModule_BlockClient(ctx, NULL, NULL, NULL, 0);
    if(RedisModule_BlockedClientMeasureTimeStart) {
        // report block client start time
        RedisModule_BlockedClientMeasureTimeStart(bc);
    }
    return bc;
}

void RedisGraph_UnblockClient
(
    RedisModuleBlockedClient *bc
) {
    ASSERT(bc != NULL);

    if(RedisModule_BlockedClientMeasureTimeEnd) {
        // report block client end time
        RedisModule_BlockedClientMeasureTimeEnd(bc);
    }
    RedisModule_UnblockClient(bc, NULL);
}
