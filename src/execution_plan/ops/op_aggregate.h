/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../../util/dict.h"
#include "../execution_plan.h"
#include "../../grouping/group.h"
#include "../../arithmetic/arithmetic_expression.h"

typedef struct {
	OpBase op;
	Record r;                     // Input Record being read from (stored to free if we encounter an error).
	uint *record_offsets;         // record IDs for key and aggregate exps
	AR_ExpNode **key_exps;        // array of expressions used to calculate the group key
	AR_ExpNode **aggregate_exps;  // array of expressions that aggregate data for each key
	dict *groups;                 // map of all groups built by this operation
	dictIterator *group_iter;     // iterator for walking all groups
	uint key_count;               // number of key expressions
	uint aggregate_count;         // number of aggregating expressions
} OpAggregate;

OpBase *NewAggregateOp
(
	const ExecutionPlan *plan,
	AR_ExpNode **exps
);

// bind the Aggregate operation to the execution plan
void AggregateBindToPlan
(
	OpBase *opBase,            // op to bind
	const ExecutionPlan *plan  // plan to bind the op to
);
