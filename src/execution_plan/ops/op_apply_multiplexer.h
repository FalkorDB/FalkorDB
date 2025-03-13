/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "op_argument.h"
#include "../execution_plan.h"

/* ApplyMultiplexer operation tests for condition satisfaction over multiple execution plan
 * branches. The branches can be a simple filter operation, SemiApply operations,
 * or ApplyMultiplexer operations. The logic applied by the operation is defined by the
 * boolean operators OR and AND.
 * ORApplyMultiplexer: Starts by pulling on the bounded branch,
 * for each record received it tries to get a record from the filter branch, if exists. If
 * the filter is not exists, or did not return any record, the operation will check each of its branches
 * until a record is found. If no data is produced from any branch,it will try to fetch a new data point
 * from the bounded branch, otherwise the bounded branch record is passed onward.
 * ANDApplyMultiplexer: Starts by pulling on the bounded branch,
 * for each record received it tries to get a record from the filter branch, if exists. If
 * the filter is not exists, or the filter returned a record, the ANDApplyMultiplexer should make sure that
 * each branch was able to produce a record. If one branch did not produced any data, the operation will try to fetch
 * a new data point from the bounded branch, otherwise the bounded branch
 * record is passed onward.
 *                  .                                                        _______________
 *                  .                                                       || argument op ||
 *                  .                                                        ______________
 *                  |                                                           |
 *                  |             _______________      _______________     _____|_________
 *                  |            || argument op ||    || argument op ||   || match branch ||
 *                  |             _______________      ______________      ________________
 *           _______|______             |                   |                   |
 *          ||bound branch||            |                   |                   |
 *           ______________      _______|_____          ____|_________          |
 *                  |           || filter op ||        || semi apply ||---------+
 *                  |            _____________          ______________
 *                  |                   |                   |
 *                  |                   |                   |
 *       ___________|_________          |                   |
 *      || apply multiplexer ||---------+-------------------+--------- . . .
 *       _____________________
 * */

typedef struct OpApplyMultiplexer {
	OpBase op;
	Record r;                       // bound branch record
	OpBase *bound_branch;           // bound branch root
	OpArgument **branch_arguments;  // branches taps
	AST_Operator boolean_operator;  // defines the operation logic - OR/AND
} OpApplyMultiplexer;

OpBase *NewApplyMultiplexerOp
(
	const ExecutionPlan *plan,
	AST_Operator boolean_operator
);

