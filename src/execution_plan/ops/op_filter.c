/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_filter.h"

// forward declarations
static RecordBatch FilterConsume(OpBase *opBase);
static OpBase *FilterClone(const ExecutionPlan *plan, const OpBase *opBase);
static void FilterFree(OpBase *opBase);

OpBase *NewFilterOp
(
	const ExecutionPlan *plan,
	FT_FilterNode *filterTree
) {
	OpFilter *op = rm_calloc (1, sizeof(OpFilter)) ;
	op->filterTree = filterTree;

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_FILTER, "Filter", NULL, FilterConsume,
				NULL, NULL, FilterClone, FilterFree, false, plan);

	return (OpBase *)op;
}

// FilterConsume next operation
// returns OP_OK when graph passes filter tree
static RecordBatch FilterConsume
(
	OpBase *opBase
) {
	OpFilter   *filter = (OpFilter *)opBase ;
	OpBase     *child  = filter->op.children[0] ;
	RecordBatch batch  = NULL ;

pull:
	// get batch
	batch = OpBase_Consume (child) ;
	if (batch == NULL) {
		// depleted
		return NULL ;
	}

	uint16_t batch_size = RecordBatch_Size (batch) ;

	for (uint16_t i = 0 ; i < batch_size ; i++) {
		Record r = batch[i] ;

		// TODO: batch filter
		// pass record through filter
		if (!FilterTree_applyFilters (filter->filterTree, r) == FILTER_PASS) {
			// record did not passed filter
			RecordBatch_RemoveRecord (batch, i) ;
			i-- ;
			batch_size-- ;
		}
	}

	// pull again if batch is empty (no record passed the filter)
	if (unlikely (RecordBatch_Size (batch) == 0)) {
		RecordBatch_Free (&batch) ;
		goto pull ;
	}

	return batch ;
}

static inline OpBase *FilterClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_FILTER);
	OpFilter *op = (OpFilter *)opBase;
	return NewFilterOp(plan, FilterTree_Clone(op->filterTree));
}

// frees OpFilter
static void FilterFree
(
	OpBase *ctx
) {
	OpFilter *filter = (OpFilter *)ctx;
	if(filter->filterTree) {
		FilterTree_Free(filter->filterTree);
		filter->filterTree = NULL;
	}
}

