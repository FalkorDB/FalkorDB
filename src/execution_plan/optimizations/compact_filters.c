/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../../RG.h"
#include "../../query_ctx.h"
#include "../ops/op_filter.h"
#include "../../errors/errors.h"
#include "../../filter_tree/filter_tree.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"

// the compact filters optimizer scans an execution plan for filters that can be
// compressed. in case the filter is compressed into a final constant 'true'
// value, the filter operation will be removed from the execution plan

// try to compact a filter
static inline bool _compactFilter
(
	OpBase *op
) {
	ASSERT(op->type == OPType_FILTER);

	OpFilter *filter_op = (OpFilter *)op;
	return FilterTree_Compact(filter_op->filterTree);
}

// in case the compacted filter resolved to 'true'
// remove it from the execution plan
static void _removeTrueFilter
(
	ExecutionPlan *plan,
	OpBase *op
) {
	ASSERT(OpBase_Type(op) == OPType_FILTER);

	OpFilter *filter_op = (OpFilter *)op;
	FT_FilterNode *root = filter_op->filterTree;

	// we can only have a contant expression at this point (after compaction)
	ASSERT(root->t == FT_N_EXP);

	// evaluate the expression, and check if it is a 'true' value
	SIValue bool_val = AR_EXP_Evaluate(root->exp.exp, NULL);
	if(SI_TYPE(bool_val) != T_BOOL && SI_TYPE(bool_val) != T_NULL) {
		// Value did not resolve to boolean, emit an error
		Error_SITypeMismatch (SI_TYPE (bool_val), T_BOOL) ;
		SIValue_Free(bool_val);
		return;
	}

	if(!SIValue_IsNull(bool_val) && SIValue_IsTrue(bool_val)) {
		ExecutionPlan_RemoveOp(plan, op);
		OpBase_Free(op);
	}
}

static void _compactFilters
(
	ExecutionPlan *plan,
	OpBase *op
) {
	OpBase **filters = ExecutionPlan_CollectOps(op, OPType_FILTER);
	int n = array_len(filters);

	for(int i = 0; i < n; i++) {
		OpBase *op = filters[i];
		// try to compact the filter
		if(_compactFilter(op)) {
			// if there was a compaction, try to remove 'true' filters
			_removeTrueFilter(plan, op);
		}
	}
	array_free(filters);
}

void compactFilters
(
	ExecutionPlan *plan
) {
	_compactFilters(plan, plan->root);
}

