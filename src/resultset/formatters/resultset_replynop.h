/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// Formatter for compact (client-parsed) replies
void ResultSet_EmitNOPHeader
(
    RedisModuleCtx *ctx,
    bolt_client_t *bolt_client,
    const char **columns,
    uint *col_rec_map
);

void ResultSet_EmitNOPRow
(
    RedisModuleCtx *ctx,
    bolt_client_t *bolt_client,
    GraphContext *gc,
    SIValue **row,
    uint numcols
);

