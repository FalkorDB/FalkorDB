/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "../ops/op.h"
#include "../ops/op_sort.h"
#include "../ops/op_limit.h"
#include "../ops/op_expand_into.h"
#include "../ops/op_conditional_traverse.h"

/* applyLimit will traverse the given execution plan looking for Limit operations.
 * Once one is found, all relevant child operations (e.g. Sort) will be
 * notified about the current limit value.
 * This is beneficial as a number of different optimizations can be applied
 * once a limit is known. */

static void notify_limit(OpBase *op, uint limit) {
	OPType t = op->type;

	switch(t) {
		// reset limit on eager operation
		case OPType_MERGE:
		case OPType_CREATE:
		case OPType_UPDATE:
		case OPType_DELETE:
		case OPType_AGGREGATE:
			limit = UNLIMITED;
			break;
		case OPType_LIMIT:
			// update limit
			limit = ((OpLimit *)op)->limit;
			break;
		case OPType_SORT:
			((OpSort *)op)->limit = limit;
			break;
		case OPType_EXPAND_INTO:
			((OpExpandInto *)op)->record_cap = limit;
			break;
		case OPType_CONDITIONAL_TRAVERSE:
			((OpCondTraverse *)op)->record_cap = limit;
			break;
		default:
			break;
	}

	for(uint i = 0; i < op->childCount; i++) {
		notify_limit(op->children[i], limit);
	}
}

void applyLimit(ExecutionPlan *plan) {
	notify_limit(plan->root, UNLIMITED);
}

