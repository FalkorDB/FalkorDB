/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "../ops/op_skip.h"
#include "../execution_plan.h"
#include "../../arithmetic/arithmetic_expression_construct.h"

OpBase *buildSkipOp(ExecutionPlan *plan, const cypher_astnode_t *skip_clause) {
	// build skip expression
	AR_ExpNode *exp = AR_EXP_FromASTNode(skip_clause);
	OpBase *op = NewSkipOp(plan, exp);
	return op;
}

