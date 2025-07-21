/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../arithmetic/arithmetic_expression.h"

typedef struct {
	OpBase op;
	Record rhs_rec;                     // Right hand side record.
	AR_ExpNode *lhs_exp;                // Left hand side expression to join on.
	AR_ExpNode *rhs_exp;                // Right hand side expression to join on.
	int64_t intersect_idx;              // Current intersection, < number_of_intersections
	Record *cached_records;             // Cached left hand side records.
	uint join_value_rec_idx;            // position on joined expression within record.
	int64_t number_of_intersections;    // Number of intersections located.
} OpValueHashJoin;

/* Creates a new ValueHashJoin operation */
OpBase *NewValueHashJoin(const ExecutionPlan *plan, AR_ExpNode *lhs_exp, AR_ExpNode *rhs_exp);
