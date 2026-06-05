/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../arithmetic/arithmetic_expression.h"

// OP Unwind
typedef struct {
	OpBase op;
	uint listIdx;          // current list index
	uint listLen;          // length of the list currently being traversed
	SIValue list;          // list which the unwind operation is performed on
	bool rangeMode;        // true if iterating direct range(...) expression
	int64_t rangeStart;    // initial range value (for resets)
	int64_t rangeEnd;      // final range value
	int64_t rangeCurrent;  // current range value
	int64_t rangeStep;     // range step
	bool rangeDepleted;    // true once all range values were emitted
	AR_ExpNode *exp;       // arithmetic expression (evaluated as an SIArray)
	int unwindRecIdx;      // update record at this index
	Record currentRecord;  // record to clone and add a value from the list
} OpUnwind;

// creates a new Unwind operation
OpBase *NewUnwindOp
(
	const ExecutionPlan *plan,
	AR_ExpNode *exp
);
