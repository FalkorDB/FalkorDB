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
#include "../ops/op_skip.h"

/* applySkip will traverse the given execution plan looking for Skip operations.
 * Once one is found, all relevant child operations (e.g. Sort) will be
 * notified about the current skip value.
 * This is beneficial as a number of different optimizations can be applied
 * once a skip is known */

static void notify_skip(OpBase *op, uint skip) {
	OPType t = op->type;

	switch(t) {
		case OPType_SKIP:
			// update skip
			skip = ((OpSkip *)op)->skip;
			break;
		case OPType_SORT:
			((OpSort *)op)->skip = skip;
			break;
		default:
			break;
	}

	for(uint i = 0; i < op->childCount; i++) {
		notify_skip(op->children[i], skip);
	}
}

void applySkip(ExecutionPlan *plan) {
	notify_skip(plan->root, 0);
}

