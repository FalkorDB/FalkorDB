/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "redismodule.h"

// listen to bolt port 7687
// add the socket to the event loop
int BoltApi_Register
(
    RedisModuleCtx *ctx  // redis context
);