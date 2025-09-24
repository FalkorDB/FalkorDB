/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "execution_plan.h"
#include "../RG.h"
#include "./ops/ops.h"

void _ExecutionPlan_Print
(
	const OpBase *op,
	RedisModuleCtx *ctx,
	sds *buffer,
	int ident,
	int *op_count
) {
	if(!op) return;

	*op_count += 1; // account for current operation

	// construct operation string representation
	sdsclear(*buffer);
	*buffer = sdscatprintf(*buffer, "%*s", ident, "");
	OpBase_ToString(op, buffer);

	RedisModule_ReplyWithStringBuffer(ctx, *buffer, sdslen(*buffer));

	// recurse over child operations
	for(int i = 0; i < op->childCount; i++) {
		_ExecutionPlan_Print(op->children[i], ctx, buffer, ident + 4, op_count);
	}
}

// reply with a string representation of given execution plan
void ExecutionPlan_Print
(
	const ExecutionPlan *plan,
	RedisModuleCtx *ctx
) {
	ASSERT(plan && ctx);

	int op_count = 0;   // Number of operations printed.
	sds buffer = sdsempty();

	// No idea how many operation are in execution plan.
	RedisModule_ReplyWithArray(ctx, REDISMODULE_POSTPONED_ARRAY_LEN);
	_ExecutionPlan_Print(plan->root, ctx, &buffer, 0, &op_count);

	RedisModule_ReplySetArrayLength(ctx, op_count);
	sdsfree(buffer);
}

