/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_join.h"
#include "op_results.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../configuration/config.h"
#include "../../arithmetic/arithmetic_expression.h"
#include "../execution_plan_build/execution_plan_util.h"

// forward declarations
static RecordBatch ResultsConsume(OpBase *opBase);
static OpResult ResultsInit(OpBase *opBase);
static OpBase *ResultsClone(const ExecutionPlan *plan, const OpBase *opBase);

OpBase *NewResultsOp
(
	const ExecutionPlan *plan
) {
	Results *op = rm_calloc (1, sizeof(Results)) ;

	// set our Op operations
	OpBase_Init ((OpBase *)op, OPType_RESULTS, "Results", ResultsInit,
			ResultsConsume, NULL, NULL, ResultsClone, NULL, false, plan) ;

	return (OpBase *)op ;
}

static OpResult ResultsInit
(
	OpBase *opBase
) {
	Results *op = (Results *)opBase ;
	op->result_set = QueryCtx_GetResultSet () ;
	Config_Option_get (Config_RESULTSET_MAX_SIZE, &op->result_set_size_limit) ;

	// map resultset columns to record entries
	OpBase *join = ExecutionPlan_LocateOpDepth (opBase, OPType_JOIN, 2) ;
	if (op->result_set != NULL && (join == NULL || !JoinGetUpdateColumnMap (join))) {
		rax *mapping = ExecutionPlan_GetMappings (opBase->plan) ;
		ResultSet_MapProjection (op->result_set, mapping) ;
	}

	return OP_OK ;
}

// results consume operation
// called each time a new result record is required
static RecordBatch ResultsConsume
(
	OpBase *opBase
) {
	Results *op = (Results *)opBase ;

	// enforce result-set size limit
	if (op->result_set_size_limit == 0) {
		return NULL ;
	}

	OpBase *child = op->op.children[0] ;
	RecordBatch batch = OpBase_Consume (child) ;
	if (batch == NULL) {
		return NULL ;
	}

	uint32_t batch_size = RecordBatch_Size (batch) ;
	ASSERT (batch_size > 0) ;

	if (unlikely (batch_size > op->result_set_size_limit)) {
		RecordBatch_DeleteRecords (batch,
				batch_size - op->result_set_size_limit) ;
		batch_size = op->result_set_size_limit ;
	}

	op->result_set_size_limit -= batch_size ;

	// append to final result set
	ResultSet_AddBatch (op->result_set, batch) ;
	return batch ;
}

static inline OpBase *ResultsClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT (opBase->type == OPType_RESULTS) ;
	return NewResultsOp (plan) ;
}

