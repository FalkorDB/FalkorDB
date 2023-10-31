/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../arithmetic/arithmetic_expression.h"

typedef struct {
	OpBase op;
	Record rhs_rec;                   // right hand side record
	AR_ExpNode *lhs_exp;              // left hand side expression to join on
	AR_ExpNode *rhs_exp;              // right hand side expression to join on
	int64_t intersect_idx;            // current intersection, < number_of_intersections
	Record *cached_records;           // cached left hand side records
	uint join_value_rec_idx;          // position on joined expression within record
	int64_t number_of_intersections;  // number of intersections located
} OpValueHashJoin;

// creates a new ValueHashJoin operation
OpBase *NewValueHashJoin
(
	ExecutionPlan *plan,
	AR_ExpNode *lhs_exp,
	AR_ExpNode *rhs_exp
);

