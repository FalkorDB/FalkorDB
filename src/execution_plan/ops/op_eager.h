/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

//  OpEager
//
//  the Eager operation fully materializes all records from its child operator
//  before producing any output
//  this breaks the normal pipelined execution flow,
//  forcing the entire input stream to be consumed and buffered in memory
//
//  once the child is fully exhausted, Eager yields the buffered records one by
//  one to its parent operator
//
//  this operator is typically used to:
//    - enforce evaluation barriers within the execution plan
//    - support operators that require all input data to be available before
//      processing (e.g., global aggregation or transformations)
//
//  note:
//    the Eager operator may significantly increase memory usage for large
//    inputs, as it stores all records in memory prior to yielding any result

typedef struct {
	OpBase op;        // base operator
	Record *records;  // materialized records from child
	uint rec_idx;     // current record index during iteration
} OpEager;

// construct a new Eager operator
// return pointer to a fully initialized Eager operator
OpBase *NewEagerOp
(
	const ExecutionPlan *plan  // execution plan to which this operator belongs
);

