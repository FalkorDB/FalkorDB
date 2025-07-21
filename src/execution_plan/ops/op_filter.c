/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "op_filter.h"
#include "RG.h"

/* Forward declarations. */
static Record FilterConsume(OpBase *opBase);
static OpBase *FilterClone(const ExecutionPlan *plan, const OpBase *opBase);
static void FilterFree(OpBase *opBase);

OpBase *NewFilterOp(const ExecutionPlan *plan, FT_FilterNode *filterTree) {
	OpFilter *op = rm_malloc(sizeof(OpFilter));
	op->filterTree = filterTree;

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_FILTER, "Filter", NULL, FilterConsume,
				NULL, NULL, FilterClone, FilterFree, false, plan);

	return (OpBase *)op;
}

/* FilterConsume next operation
 * returns OP_OK when graph passes filter tree. */
static Record FilterConsume(OpBase *opBase) {
	Record r = NULL;
	OpFilter *filter = (OpFilter *)opBase;
	OpBase *child = filter->op.children[0];

	while(true) {
		r = OpBase_Consume(child);
		if(!r) break;

		/* Pass record through filter tree */
		if(FilterTree_applyFilters(filter->filterTree, r) == FILTER_PASS) break;
		else OpBase_DeleteRecord(r);
	}

	return r;
}

static inline OpBase *FilterClone(const ExecutionPlan *plan, const OpBase *opBase) {
	ASSERT(opBase->type == OPType_FILTER);
	OpFilter *op = (OpFilter *)opBase;
	return NewFilterOp(plan, FilterTree_Clone(op->filterTree));
}

/* Frees OpFilter*/
static void FilterFree(OpBase *ctx) {
	OpFilter *filter = (OpFilter *)ctx;
	if(filter->filterTree) {
		FilterTree_Free(filter->filterTree);
		filter->filterTree = NULL;
	}
}

