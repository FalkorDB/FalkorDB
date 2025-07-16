/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "../execution_plan.h"
#include "../ops/op_limit.h"
#include "../../arithmetic/arithmetic_expression_construct.h"

OpBase *buildLimitOp(ExecutionPlan *plan, const cypher_astnode_t *limit_clause) {
	// build limit expression
	AR_ExpNode *exp = AR_EXP_FromASTNode(limit_clause);
	OpBase *op = NewLimitOp(plan, exp);
	return op;
}

