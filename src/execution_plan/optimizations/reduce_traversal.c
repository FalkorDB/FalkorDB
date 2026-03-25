/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../util/arr.h"
#include "../ops/op_expand_into.h"
#include "../ops/op_conditional_traverse.h"
#include "../ops/op_cond_var_len_traverse.h"
#include "../execution_plan_build/execution_plan_util.h"
#include "../execution_plan_build/execution_plan_modify.h"

/* Reduce traversal searches for traversal operations where
 * both the src and destination nodes in the traversal are already
 * resolved by former operation, in which case we need to make sure
 * src is connected to dest via the current expression.
 *
 * Consider the following query, execution plan:
 * MATCH (A)-[X]->(B)-[Y]->(A) RETURN A,B
 * SCAN (A)
 * TRAVERSE-1 (A)-[X]->(B)
 * TRAVERSE-2 (B)-[Y]->(A)
 * TRAVERSE-2 tries to see if B is connected to A via Y
 * but A and B are known, we just need to make sure there's an edge
 * of type Y connecting B to A
 * this is done by the EXPAND-INTO operation. */

static inline bool _isInSubExecutionPlan(OpBase *op) {
	return ExecutionPlan_LocateOp(op, OPType_ARGUMENT) != NULL;
}

static void _removeRedundantTraversal(ExecutionPlan *plan, OpCondTraverse *traverse) {
	AlgebraicExpression *ae = traverse->ae;
	const char *src = AlgebraicExpression_Src(ae);
	const char *dest = AlgebraicExpression_Dest(ae);
	if(ae && AlgebraicExpression_OperandCount(ae) == 1 &&
	   src && dest && !strcmp(src, dest)) {
		ExecutionPlan_RemoveOp(plan, (OpBase *)traverse);
		OpBase_Free((OpBase *)traverse);
	}
}

/* Inspect each traverse operation T,
 * For each T see if T's source and destination nodes
 * are already resolved, in which case replace traversal operation
 * with expand-into op. */
void reduceTraversal(ExecutionPlan *plan) {
	OpBase **traversals = ExecutionPlan_CollectOpsMatchingTypes(plan->root, TRAVERSE_OPS,
															   TRAVERSE_OP_COUNT);
	uint traversals_count = array_len(traversals);

	/* Keep track of redundant traversals which will be removed
	 * once we'll inspect every traversal operation. */
	uint redundantTraversalsCount = 0;
	OpCondTraverse *redundantTraversals[traversals_count];

	for(uint i = 0; i < traversals_count; i++) {
		OpBase *op = traversals[i];
		AlgebraicExpression *ae = NULL;
		if(op->type == OPType_CONDITIONAL_TRAVERSE) {
			OpCondTraverse *traverse = (OpCondTraverse *)op;
			ae = traverse->ae;
		} else if(op->type == OPType_CONDITIONAL_VAR_LEN_TRAVERSE) {
			CondVarLenTraverse *traverse = (CondVarLenTraverse *)op;
			ae = traverse->ae;
		} else {
			ASSERT(false);
		}

		/* If traverse src and dest nodes are the same,
		 * number of hops is 1 and the matrix being used is a label matrix, than
		 * traverse acts as a filter which make sure the node is of a specific type
		 * e.g. MATCH (a:A)-[e:R]->(b:B) RETURN e
		 * in this case there will be a traverse operation which will
		 * filter our dest nodes (b) which aren't of type B. */

		const char *src = AlgebraicExpression_Src(ae);
		const char *dest = AlgebraicExpression_Dest(ae);

		if(src && dest && !strcmp(src, dest) &&
		   AlgebraicExpression_OperandCount(ae) == 1 &&
		   AlgebraicExpression_DiagonalOperand(ae, 0)) continue;

		// collect variables bound before this op
		rax *bound_vars = raxNew();
		for(int i = 0; i < op->childCount; i ++) {
			ExecutionPlan_BoundVariables(op->children[i], bound_vars,
				op->children[i]->plan);
		}

		const char *dest_alias = AlgebraicExpression_Dest(ae);
		if(dest_alias == NULL || raxFind(bound_vars, (unsigned char *)dest_alias, strlen(dest_alias)) == raxNotFound) {
			// The destination could not be resolved, cannot optimize.
			raxFree(bound_vars);
			continue;
		}

		/* Both src and dest are already known
		 * perform expand into instead of traverse. */
		if(op->type == OPType_CONDITIONAL_TRAVERSE) {
			OpCondTraverse *traverse = (OpCondTraverse *)op;
			const ExecutionPlan *traverse_plan = traverse->op.plan;
			OpBase *expand_into = NewExpandIntoOp(traverse_plan, traverse->graph, traverse->ae);

			// Set traverse algebraic_expression to NULL to avoid early free.
			traverse->ae = NULL;
			ExecutionPlan_ReplaceOp(plan, (OpBase *)traverse, expand_into);
			OpBase_Free((OpBase *)traverse);
		} else {
			CondVarLenTraverse *traverse = (CondVarLenTraverse *)op;
			const ExecutionPlan *traverse_plan = traverse->op.plan;
			CondVarLenTraverseOp_ExpandInto(traverse);
			/* Conditional variable length traversal do not perform
			 * label filtering by matrix matrix multiplication
			 * it introduces conditional traverse operation in order
			 * to perform label filtering, but in case a node is already
			 * resolved this filtering is redundent and should be removed. */
			OpBase *t;
			const char *src_alias = AlgebraicExpression_Src(ae);
			QGNode *src = src_alias ? QueryGraph_GetNodeByAlias(traverse_plan->query_graph, src_alias) : NULL;
			if(src && QGNode_Labeled(src)) {
				t = op->children[0];
				if(t->type == OPType_CONDITIONAL_TRAVERSE && !_isInSubExecutionPlan(op)) {
					// Queue traversal for removal.
					redundantTraversals[redundantTraversalsCount++] = (OpCondTraverse *)t;
				}
			}
			const char *dest_alias_2 = AlgebraicExpression_Dest(ae);
			QGNode *dest = dest_alias_2 ? QueryGraph_GetNodeByAlias(traverse_plan->query_graph,
													 dest_alias_2) : NULL;
			if(dest && QGNode_Labeled(dest)) {
				t = op->parent;
				if(t->type == OPType_CONDITIONAL_TRAVERSE && !_isInSubExecutionPlan(op)) {
					// Queue traversal for removal.
					redundantTraversals[redundantTraversalsCount++] = (OpCondTraverse *)t;
				}
			}
		}
		raxFree(bound_vars);
	}

	// Remove redundant traversals
	for(uint i = 0; i < redundantTraversalsCount; i++)
		_removeRedundantTraversal(plan, redundantTraversals[i]);

	// Clean up.
	array_free(traversals);
}

