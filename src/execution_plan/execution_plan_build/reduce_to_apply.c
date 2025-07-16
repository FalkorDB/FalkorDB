/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "../ops/ops.h"
#include "../../query_ctx.h"
#include "../execution_plan.h"
#include "../../ast/ast_mock.h"
#include "execution_plan_util.h"
#include "execution_plan_modify.h"
#include "execution_plan_construct.h"
#include "../../util/rax_extensions.h"
#include "../optimizations/optimizer.h"

/* Swap operation on operation children. If two valid indices, a and b, are given, this operation
 * swap the child in index a with the child in index b. */
static inline void _OpBaseSwapChildren(OpBase *op, int a, int b) {
	ASSERT(a >= 0 && b >= 0 && a < op->childCount && b < op->childCount);
	OpBase *tmp = op->children[a];
	op->children[a] = op->children[b];
	op->children[b] = tmp;
}

static void _CreateBoundBranch(OpBase *op, ExecutionPlan *plan, const char **vars) {
	OpBase *arg = NewArgumentOp(plan, vars);

	/* In case of Apply operations, we need to append, the new argument branch and
	 * set it as the first child. */
	ExecutionPlan_AddOp(op, arg);

	if(OP_IS_APPLY(op)) _OpBaseSwapChildren(op, 0, op->childCount - 1);
}


/* Builds a semi apply operation out of path filter expression. */
static OpBase *_ApplyOpFromPathExpression(ExecutionPlan *plan, const char **vars,
										  AR_ExpNode *expression) {
	bool anti = false;
	if(strcasecmp(AR_EXP_GetFuncName(expression), "not") == 0) {
		anti = true;
		expression = expression->op.children[0];
	}
	// Build new Semi Apply op.
	OpBase *op_semi_apply = NewSemiApplyOp(plan, anti);
	const cypher_astnode_t *path = expression->op.children[0]->operand.constant.ptrval;
	// Add a match branch as a Semi Apply op child.
	OpBase *match_branch = ExecutionPlan_BuildOpsFromPath(plan, vars, path);
	ExecutionPlan_AddOp(op_semi_apply, match_branch);
	return op_semi_apply;
}

/* This method reduces a filter tree into an OpBase.
 * The method perfrom post-order traversal over the filter tree,
 * and checks if the current subtree rooted at the visited node
 * contains path filter or not, and either reduces the root 
 * or continue traversal and reduction.
 * The three possible operations could be:
 * 1. OpFilter - In the case the subtree originated from the root does not contains any path filters.
 * 2. OpSemiApply - In case of the current root is an expression which contains a path filter.
 * 3. ApplyMultiplexer - In case the current root is an operator (OR or AND) and at least one of its children
 * has been reduced to OpSemiApply or ApplyMultiplexer. */
static OpBase *_ReduceFilterToOp(ExecutionPlan *plan, const char **vars,
								 FT_FilterNode *filter_root) {
	FT_FilterNode *node;
	// Case of an expression which contains path filter.
	if(filter_root->t == FT_N_EXP && FilterTree_ContainsFunc(filter_root, "path_filter", &node)) {
		/* If an expression filter tree node contains "path_filter" function, it can be either as
		 * a single function, or a single child of a single "not" function. */
		AR_ExpNode *expression = filter_root->exp.exp;
		return _ApplyOpFromPathExpression(plan, vars, expression);
	}
	// Case of an operator (Or or And) which its subtree contains path filter
	if(filter_root->t == FT_N_COND && FilterTree_ContainsFunc(filter_root, "path_filter", &node)) {
		// Create the relevant LHS branch and set a bounded branch for it.
		OpBase *lhs = _ReduceFilterToOp(plan, vars, filter_root->cond.left);
		if(lhs->type == OPType_FILTER) filter_root->cond.left = NULL;
		_CreateBoundBranch(lhs, plan, vars);
		// Create the relevant RHS branch and set a bounded branch for it.
		OpBase *rhs = _ReduceFilterToOp(plan, vars, filter_root->cond.right);
		if(rhs->type == OPType_FILTER) filter_root->cond.right = NULL;
		_CreateBoundBranch(rhs, plan, vars);
		// Check that at least one of the branches is not a filter.
		ASSERT(!(rhs->type == OPType_FILTER && lhs->type == OPType_FILTER));
		// Create multiplexer op and set the branches as its children.
		OpBase *apply_multiplexer = NewApplyMultiplexerOp(plan, filter_root->cond.op);
		ExecutionPlan_AddOp(apply_multiplexer, lhs);
		ExecutionPlan_AddOp(apply_multiplexer, rhs);
		return apply_multiplexer;
	}
	// In the case of a simple filter.
	return NewFilterOp(plan, filter_root);
}

/* Reduces a filter operation to a SemiApply/ApplyMultiplexer operation, according to the filter tree */
void ExecutionPlan_ReduceFilterToApply(ExecutionPlan *plan, OpFilter *filter) {
	// Collect all variables that are bound at this position in the op tree.
	rax *bound_vars = raxNew();
	ExecutionPlan_BoundVariables((OpBase *)filter, bound_vars,
		((OpBase *)filter)->plan);
	// Collect the variable names from bound_vars to populate the Argument ops we will build.
	const char **vars = (const char **)raxValues(bound_vars);

	ExecutionPlan *filter_plan = (ExecutionPlan *)filter->op.plan;
	// Reduce.
	OpBase *apply_op = _ReduceFilterToOp(filter_plan, vars, filter->filterTree);
	// Replace operations.
	ExecutionPlan_ReplaceOp(plan, (OpBase *)filter, apply_op);
	// Bounded branch is now the last child (after ops replacement). Make it the first.
	_OpBaseSwapChildren(apply_op, 0, apply_op->childCount - 1);
	if((OpBase *)filter == plan->root) plan->root = apply_op;

	// Free filter op.
	OpBase_Free((OpBase *)filter);
	raxFree(bound_vars);
	array_free(vars);
}

