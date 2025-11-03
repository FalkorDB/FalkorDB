/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../execution_plan.h"
#include "../../filter_tree/filter_tree.h"
#include "../../arithmetic/algebraic_expression.h"

// reorders exps such that exp[i] is the ith expression to evaluate
void orderExpressions(
	QueryGraph *qg,                 // queryGraph containing expression entity data
	AlgebraicExpression **exps,     // expressions to order
	uint *exps_count,               // number of expressions
	const FT_FilterNode *filters,   // filters
	rax *bound_vars                 // previously-bound variables
);

void compactFilters(ExecutionPlan *plan);
void reduceScans(ExecutionPlan *plan);
void utilizeIndices(ExecutionPlan *plan);
void seekByID(ExecutionPlan *plan);
void filterVariableLengthEdges(ExecutionPlan *plan);
void reduceCartesianProductStreamCount(ExecutionPlan *plan);
void applyJoin(ExecutionPlan *plan);
void reduceFilters(ExecutionPlan *plan);
void reduceTraversal(ExecutionPlan *plan);
void reduceDistinct(ExecutionPlan *plan);
void reduceCount(ExecutionPlan *plan);
void costBaseLabelScan(ExecutionPlan *plan);
void batchOptionalMatch(ExecutionPlan *plan);

