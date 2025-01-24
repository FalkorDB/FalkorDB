/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../../ast/ast.h"
#include "../ops/op_join.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../execution_plan.h"
#include "../ops/op_project.h"
#include "../ops/op_argument.h"
#include "execution_plan_util.h"
#include "../ops/op_aggregate.h"
#include "execution_plan_modify.h"
#include "../ops/op_argument_list.h"
#include "../ops/op_call_subquery.h"

// adds an empty projection as the child of parent, such that the records passed
// to parent are "filtered" to contain no bound vars
static OpBase *_add_empty_projection
(
	OpBase *parent
) {
	OpBase *empty_proj =
		NewProjectOp(parent->plan, array_new(AR_ExpNode *, 0));

	OPType type = OpBase_Type(parent);
	if(type == OPType_CALLSUBQUERY || type == OPType_FOREACH) {
		ExecutionPlan_AddOpInd(parent, empty_proj, 0);
	} else {
		ExecutionPlan_AddOp(parent, empty_proj);
	}

	return empty_proj;
}

// returns true if op is effectively a deepest op (i.e., no lhs)
static inline bool _is_deepest_call_foreach
(
	OpBase *op  // op to check
) {
	OPType type = OpBase_Type(op);
	return (type == OPType_CALLSUBQUERY || type == OPType_FOREACH) &&
			op->childCount == 1;
}

// finds the deepest operation starting from root, and appends it to deepest_ops
// if a CALL op with one child is found, it is appended to deepest_ops
static void _get_deepest
(
	OpBase *root,          // root op from which to look for the deepest op
	OpBase ***deepest_ops  // target array to which the deepest op is appended
) {
	OpBase *deepest = root;

	// check root
	if(_is_deepest_call_foreach(deepest)) {
		array_append(*deepest_ops, deepest);
		return;
	}

	// traverse children
	OPType type;
	while(OpBase_ChildCount(deepest) > 0) {
		deepest = deepest->children[0];
		// in case of a CallSubquery op with no lhs, we want to stop
		// here, as the added op should be its first child (instead of
		// the current child, which will be moved to be the second)
		// Example:
		// "CALL {CALL {RETURN 1 AS one} RETURN one} RETURN one"
		type = OpBase_Type(deepest);
		if(_is_deepest_call_foreach(deepest)) {
			array_append(*deepest_ops, deepest);
			return;
		}
	}

	array_append(*deepest_ops, deepest);
}

// returns an array with the deepest ops of an execution plan
// note: it's that caller's responsibility to free the array
static OpBase **_find_feeding_points
(
	const ExecutionPlan *plan
) {
	ASSERT(plan != NULL);

	// the root is a Results op if the subquery is returning
	// search for a Join op in its first child or its first child's first child
	// (depending on whether there is a `UNION` or `UNION ALL` clause)
	OpBase *join = ExecutionPlan_LocateOpDepth(plan->root, OPType_JOIN, 3);

	// get the deepest op(s)
	uint n_branches = 1;
	OpBase **feeding_points = array_new(OpBase *, n_branches);

	if(join != NULL) {
		n_branches = OpBase_ChildCount(join);
		for(uint i = 0; i < n_branches; i++) {
			OpBase *branch = OpBase_GetChild(join, i);
			_get_deepest(branch, &feeding_points);
		}
	} else {
		_get_deepest(plan->root, &feeding_points);
	}

	ASSERT(array_len(feeding_points) == n_branches);
	return feeding_points;
}

// binds the projecting ops (effectively, all ops between the first
// Project\Aggregate and CallSubquery in every branch other than the Join op,
// inclusive) in embedded_plan to plan
// returns true if `embedded_plan` should be free'd after binding its root to
// the call {} op (if there are no more ops left in it), false otherwise
static bool _bind_projecting_ops_to_plan
(
	ExecutionPlan *embedded_plan,  // embedded plan
	const ExecutionPlan *plan      // plan to migrate ops to
) {
	// check if there is a Join operation (from UNION or UNION ALL)
	OpBase *proj_op = NULL;
	OpBase *root    = embedded_plan->root;
	OpBase *join_op = ExecutionPlan_LocateOpDepth(root, OPType_JOIN, 3);

	// 5 place-holders are allocated for a maximum of 5 operations between the
	// returning op (Project/Aggregate) and the CallSubquery op
	// (Sort, Join, Distinct, Skip, Limit)
	OpBase *ops[5];
	OPType proj_types[] = {OPType_PROJECT, OPType_AGGREGATE};

	if(join_op == NULL) {
		// only one projection/aggregation operation
		proj_op =
			ExecutionPlan_LocateOpMatchingTypes(root, proj_types, 2, NULL, 0);

		// if the projecting op has no children, we need to free its exec-plan
		// after binding it to a new plan
		ExecutionPlan *old_plan = (ExecutionPlan *)proj_op->plan;

		// collect parent operations e.g. DISTINCT, SORT, SKIP, LIMIT
		uint n_ops = ExecutionPlan_CollectUpwards(ops, 5, proj_op);
		ExecutionPlan_MigrateOpsExcludeType(ops, OPType_JOIN, n_ops, plan);

		if(proj_op->childCount == 0) {
			if(old_plan == embedded_plan) {
				return true;
			} else {
				old_plan->root = NULL;
				ExecutionPlan_Free(old_plan);
				return false;
			}
		}

		// there are more ops in the plan of the binded op, so we don't free its
		// plan
		return false;
	} else {
		// there's a UNION operation
		// migrate all operations within the UNION plan
		// that includes projections / distinct & join

		// collect all reachable operations within the join plan
		uint n;  // number of ops
		OpBase **ops = ExecutionPlan_CollectAllOps(embedded_plan, &n);

		ASSERT(n   > 0);
		ASSERT(ops != NULL);

		// migrate ops from inner-plan to outter-plan
		for(uint i = 0; i < n; i++) {
			OpBase_BindOpToPlan(ops[i], plan);
		}
		rm_free(ops);

		// return true, indicating the later free of the inner-plan
		return true;
	}
}

// add empty projections to the branches which do not contain an importing WITH
// clause, in order to 'reset' the bound-vars environment
static void _add_empty_projections
(
	OpBase **feeding_points  // deepest op in each of the UNION branches
) {
	uint n_branches = array_len(feeding_points);
	for(uint i = 0; i < n_branches; i++) {
		if(OpBase_Type(feeding_points[i]) != OPType_PROJECT) {
			feeding_points[i] = _add_empty_projection(feeding_points[i]);
		}
	}
}

// construct the execution-plan corresponding to a call {} clause
void buildCallSubqueryPlan
(
	ExecutionPlan *plan,            // execution plan to add plan to
	const cypher_astnode_t *clause  // call subquery clause
) {
	//--------------------------------------------------------------------------
	// build an AST from the subquery
	//--------------------------------------------------------------------------

	// save the original AST
	AST *orig_ast = QueryCtx_GetAST();

	// create an AST from the body of the subquery
	AST subquery_ast = {0};
	subquery_ast.root = cypher_ast_call_subquery_get_query(clause);
	subquery_ast.anot_ctx_collection = orig_ast->anot_ctx_collection;

	//--------------------------------------------------------------------------
	// build the embedded execution plan corresponding to the subquery
	//--------------------------------------------------------------------------

	QueryCtx_SetAST(&subquery_ast);
	ExecutionPlan *embedded_plan = ExecutionPlan_FromTLS_AST();

	// restore AST
	QueryCtx_SetAST(orig_ast);

	// characterize whether the query is eager or not
	bool is_eager = ExecutionPlan_isEager(embedded_plan->root);

	// characterize whether the query is returning or not
	bool is_returning = (OpBase_Type(embedded_plan->root) == OPType_RESULTS);

	// find the feeding points, to which we will add the projections and feeders
	OpBase **feeding_points = _find_feeding_points(embedded_plan);

	// if no variables are imported, add an 'empty' projection so that the
	// records within the subquery will be cleared from the outer-context
	_add_empty_projections(feeding_points);

	//--------------------------------------------------------------------------
	// Bind returning projection(s)\aggregation(s) to the outer plan
	//--------------------------------------------------------------------------

	bool free_embedded_plan = false;
	if(is_returning) {
		// remove the Results op from the embedded execution-plan
		OpBase *results_op = embedded_plan->root;
		ASSERT(OpBase_Type(results_op) == OPType_RESULTS);

		ExecutionPlan_RemoveOp(embedded_plan, embedded_plan->root);
		OpBase_Free(results_op);

		// bind the projecting ops to the outer plan
		free_embedded_plan = _bind_projecting_ops_to_plan(embedded_plan, plan);
	}

	//--------------------------------------------------------------------------
	// plant feeders
	//--------------------------------------------------------------------------

	uint n_feeding_points = array_len(feeding_points);
	if(is_eager) {
		for(uint i = 0; i < n_feeding_points; i++) {
			OpBase *argument_list = NewArgumentListOp(plan, NULL);
			ExecutionPlan_AddOp(feeding_points[i], argument_list);
		}
	} else {
		for(uint i = 0; i < n_feeding_points; i++) {
			OpBase *argument = NewArgumentOp(plan, NULL);
			ExecutionPlan_AddOp(feeding_points[i], argument);
		}
	}

	array_free(feeding_points);

	//--------------------------------------------------------------------------
	// introduce a Call-Subquery op
	//--------------------------------------------------------------------------

	OpBase *call_op = NewCallSubqueryOp(plan, is_eager, is_returning);
	ExecutionPlan_UpdateRoot(plan, call_op);

	// add the embedded plan as a child of the Call-Subquery op
	ExecutionPlan_AddOp(call_op, embedded_plan->root);

	if(free_embedded_plan) {
		embedded_plan->root = NULL;
		ExecutionPlan_Free(embedded_plan);
	}
}

