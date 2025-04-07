/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// Results populates the query's result-set
// the operation enforces the configured maximum result-set size
// if that limit been reached query execution terminate
// otherwise the current record is added to the result-set

#include "RG.h"
#include "op_results.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../configuration/config.h"

// forward declarations
static Record ResultsConsume(OpBase *opBase);
static OpResult ResultsInit(OpBase *opBase);
static OpBase *ResultsClone(const ExecutionPlan *plan, const OpBase *opBase);

OpBase *NewResultsOp
(
	const ExecutionPlan *plan
) {
	Results *op = rm_calloc(1, sizeof(Results));

	// Set our Op operations
	OpBase_Init((OpBase *)op, OPType_RESULTS, "Results", ResultsInit, ResultsConsume,
				NULL, NULL, ResultsClone, NULL, false, plan);

	return (OpBase *)op;
}

static OpResult ResultsInit
(
	OpBase *opBase
) {
	Results *op = (Results *)opBase;
	Config_Option_get(Config_RESULTSET_MAX_SIZE, &op->result_set_size_limit);

	// map resultset columns to record entries
	op->result_set = QueryCtx_GetResultSet();
	if(op->result_set != NULL) {
		rax *mapping = ExecutionPlan_GetMappings(opBase->plan);
		ResultSet_MapProjection(op->result_set, mapping);
	}

	return OP_OK;
}

// results consume operation
// called each time a new result record is required
static Record ResultsConsume
(
	OpBase *opBase
) {
	Record r = NULL;
	Results *op = (Results*)opBase;

	// enforce result-set size limit
	if(unlikely(op->result_set_size_limit == 0)) {
		return NULL;
	}

	op->result_set_size_limit--;

	OpBase *child = op->op.children[0];
	r = OpBase_Consume(child);
	if(!r) {
		return NULL;
	}

	// append to final result set
	ResultSet_AddRecord(op->result_set, r);

	return r;
}

static inline OpBase *ResultsClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_RESULTS);
	return NewResultsOp(plan);
}

