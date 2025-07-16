/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "../execution_plan.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"

/* A distinct operation following an aggregation operation
 * is unnecessary, as aggregation groups are guaranteed to be unique.
 * this optimization will try to look for an aggregation operation
 * followed by a distinct operation, in which case we can omit distinct
 * from the execution plan. */

void reduceDistinct(ExecutionPlan *plan) {
	// Look for Distinct operations.
	OpBase **distinct_ops = ExecutionPlan_CollectOps(plan->root, OPType_DISTINCT);

	for(uint i = 0; i < array_len(distinct_ops); i++) {
		OpBase *distinct = distinct_ops[i];
		ASSERT(distinct->childCount == 1);
		if(distinct->children[0]->type == OPType_AGGREGATE) {
			// We can remove the Distinct op if its child is an aggregate operation,
			// as its results will inherently be unique.
			ExecutionPlan_RemoveOp(plan, distinct);
			OpBase_Free(distinct);
		}
	}

	array_free(distinct_ops);
}

