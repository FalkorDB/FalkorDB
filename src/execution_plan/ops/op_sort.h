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
	Record *buffer;        // Holds all records.
	heap_t *heap;          // Holds top n records.
	bool first;            // first visit to consume func
	uint64_t skip;         // Total number of records to skip
	uint record_idx;       // index of current record to return
	uint64_t limit;        // Total number of records to produce
	uint *record_offsets;  // All Record offsets containing values to sort by
	int *directions;       // Array of sort directions(ascending / descending)
	AR_ExpNode **exps;     // Projected expressons.
} OpSort;

/* Creates a new Sort operation */
OpBase *NewSortOp(const ExecutionPlan *plan, AR_ExpNode **exps, int *directions);
