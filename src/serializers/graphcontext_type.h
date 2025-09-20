/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "serializers_include.h"

int _GraphContextType_Defrag
(
	RedisModuleDefragCtx *ctx,
	RedisModuleString *key,
	void **value
);

int GraphContextType_Register
(
	RedisModuleCtx *ctx
);

