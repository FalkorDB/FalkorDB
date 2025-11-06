/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../../util/heap.h"
#include "../execution_plan.h"
#include "../../arithmetic/arithmetic_expression.h"

typedef struct {
	OpBase op;
	Record *buffer;        // holds all records
	heap_t *heap;          // holds top n records
	bool first;            // first visit to consume func
	uint64_t skip;         // total number of records to skip
	uint record_idx;       // index of current record to return
	uint64_t limit;        // total number of records to produce
	uint *sort_offsets;    // all record offsets containing values to sort by
	int *directions;       // array of sort directions(ascending / descending)
	AR_ExpNode **exps;     // sort expressions

	AR_ExpNode **to_eval;  // sort expression to eval
	uint *record_offsets;  // sort expressions record position
} OpSort;

// creates a new Sort operation
OpBase *NewSortOp
(
	const ExecutionPlan *plan,
	AR_ExpNode **exps,
	int *directions
);

void SortBindToPlan
(
	OpBase *opBase,            // op to bind
	const ExecutionPlan *plan  // plan to bind the op to
);

