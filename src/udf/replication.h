/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "../redismodule.h"

// register callback receivers for cluster UDF messages 
void UDF_ReplicationRegisterReceiver
(
	RedisModuleCtx *ctx  // redis module context
);

// counter part of UDF_ReplicationRegisterReceiver
// unregister callback receivers
void UDF_ReplicationUnRegisterReceiver
(
	RedisModuleCtx *ctx  // redis module context
);

// replicate a UDF command to the cluster
void UDF_ReplicationSendCmd
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) ;	

