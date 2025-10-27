/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"

//  OpSubQueryForeach
//
//  the SubQueryForeach operation executes a subquery for each record produced
//  by its child operator
//  for every input record, the operator evaluates the
//  subquery once, binding the record’s variables into the subquery context
//
//  this operator is used to implement Cypher constructs of the form:
//      CALL { ... }
//  or iterative subqueries like:
//      UNWIND list AS x CALL { WITH x ... }
//
//  each "tap" represents an entry point into the subquery that needs to be
//  re-executed for every input record
//  the operator coordinates execution across these taps to ensure proper
//  isolation and data flow between the outer and inner query scopes
//
//  NOTE:
//    since each input record triggers a full subquery execution, this operator
//    may be computationally expensive for large input streams

typedef struct {
	OpBase op;          // base operator
	OpArgument **taps;  // entry points (arguments) into the subquery to execute
	uint n_taps;        // number of taps (subquery entry points)
} OpSubQueryForeach;

// construct a new SubQueryForeach operator
// return pointer to a fully initialized SubQueryForeach operator
OpBase *NewSubQueryForeach
(
	const ExecutionPlan *plan  // execution plan to which this operator belongs
);

