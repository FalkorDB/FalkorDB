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

static RecordBatch FilterConsume
(
	OpBase *opBase
) {
    OpFilter *filter = (OpFilter *)opBase ;
    OpBase *child = filter->op.children[0] ;

    while (true) {
		// pull batch
        RecordBatch batch = OpBase_Consume (child) ;
        if (batch == NULL) {
			// depleted
			return NULL ;
		}

        uint16_t n = RecordBatch_Size (batch) ;

        FT_Result pass[n] ; // TODO: move to a pre allocated workspace
        FilterTree_applyBatchFilters (pass, filter->filterTree, batch, n) ;

        // compaction Logic: move 'passing' records to the front
        uint16_t write_idx = 0 ;
        for (uint16_t read_idx = 0 ; read_idx < n ; read_idx++) {
            if (pass[read_idx] == FILTER_PASS) {
                batch[write_idx] = batch[read_idx] ;
                write_idx++ ;
            } else {
				OpBase_DeleteRecord (batch + read_idx) ;
			}
        }

		// update batch size
		RecordBatch_SetSize (batch, write_idx) ;

		if (write_idx == 0) {
			// entire batch was filtered out, loop again to get next batch
			RecordBatch_Free (&batch) ;
			continue ;
		} 

		return batch ;
    }
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

