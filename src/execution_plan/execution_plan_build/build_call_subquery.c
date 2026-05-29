/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../../ast/ast.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../ops/op_eager.h"
#include "../ops/op_apply.h"
#include "../ops/op_argument.h"
#include "../ops/op_empty_row.h"
#include "../ops/op_subquery_foreach.h"
#include "../execution_plan.h"
#include "execution_plan_util.h"
#include "execution_plan_modify.h"

#include <string.h>

// looks for a Join operation at root or root->children[0] and returns it, or
// NULL if not found
static OpBase *_get_join
(
	OpBase *root  // root op from which to look for the Join op
) {
	// check if there is a Join operation (from UNION or UNION ALL)
	OpBase *join_op = NULL;
	if(root->type == OPType_JOIN) {
		join_op = root;
	} else if(root->childCount > 0 && root->children[0]->type == OPType_JOIN) {
		join_op = root->children[0];
	}

	return join_op;
}

// collect variables imported by leading WITH clauses in a CALL {} subquery
static const char **_collect_imported_variables
(
	const cypher_astnode_t *clause  // call subquery clause
) {
	ASSERT(clause != NULL);

	const cypher_astnode_t *query =
		cypher_ast_call_subquery_get_query(clause);
	uint n_clauses = cypher_ast_query_nclauses(query);
	const char **variables = arr_new(const char *, 0);

	// A WITH clause imports outer variables only when it is the first clause of
	// the subquery, or the first clause after a UNION.
	bool expecting_import = true;
	for(uint i = 0; i < n_clauses; i++) {
		const cypher_astnode_t *inner_clause =
			cypher_ast_query_get_clause(query, i);
		cypher_astnode_type_t t = cypher_astnode_type(inner_clause);

		if(t == CYPHER_AST_UNION) {
			expecting_import = true;
			continue;
		}

		if(!expecting_import || t != CYPHER_AST_WITH) {
			expecting_import = false;
			continue;
		}

		uint n_projections = cypher_ast_with_nprojections(inner_clause);
		for(uint j = 0; j < n_projections; j++) {
			const cypher_astnode_t *projection =
				cypher_ast_with_get_projection(inner_clause, j);
			const cypher_astnode_t *exp =
				cypher_ast_projection_get_expression(projection);

			// Empty WITH * imports are rewritten to a NULL projection.
			if(cypher_astnode_type(exp) != CYPHER_AST_IDENTIFIER) {
				continue;
			}

			const char *identifier = cypher_ast_identifier_get_name(exp);
			bool exists = false;
			for(uint k = 0; k < arr_len(variables); k++) {
				if(strcmp(variables[k], identifier) == 0) {
					exists = true;
					break;
				}
			}

			if(!exists) {
				arr_append(variables, identifier);
			}
		}

		expecting_import = false;
	}

	return variables;
}

static bool _variable_in_array
(
	const char *variable,
	const char **variables
) {
	ASSERT(variable != NULL);

	uint n = arr_len(variables);
	for(uint i = 0; i < n; i++) {
		if(strcmp(variable, variables[i]) == 0) {
			return true;
		}
	}

	return false;
}

// return the imported variables projected by a specific feeder point
static const char **_feeding_point_imported_variables
(
	OpBase *feeding_point,
	const char **imported_variables
) {
	ASSERT(feeding_point != NULL);

	const char **variables = arr_new(const char *, 0);
	uint n_modified = arr_len(feeding_point->modifies);

	for(uint i = 0; i < n_modified; i++) {
		const char *modified = feeding_point->modifies[i];
		if(_variable_in_array(modified, imported_variables)) {
			arr_append(variables, modified);
		}
	}

	return variables;
}

// returns an array with the deepest ops of an execution plan
// note: it's that caller's responsibility to free the array
// TODO: make a utility function
static OpBase **_find_feeding_points
(
	const ExecutionPlan *plan
) {
	ASSERT(plan != NULL);

	OpBase **ops  = arr_new (OpBase*, 1) ;
	OpBase **taps = arr_new (OpBase*, 1) ;

	OpBase *sub_query_root = plan->root ;
	arr_append (ops, sub_query_root) ;

	while (arr_len (ops) > 0) {
		OpBase *child = arr_pop (ops) ;
		OPType t = OpBase_Type (child) ;

		// tap located
		if ((OpBase_ChildCount (child) == 0) && t == OPType_PROJECT) {
			arr_append (taps, child) ;
		}

		// join op, traverse each branch
		else if (OP_JOIN_MULTIPLE_STREAMS (child)) {
			for (uint i = 0; i < OpBase_ChildCount (child); i++) {
				arr_append (ops, OpBase_GetChild (child, i)) ;
			}
		}

		// go "left"
		else if (OpBase_ChildCount (child) > 0) {
			arr_append (ops, OpBase_GetChild (child, 0)) ;
		}
	}

	arr_free (ops) ;
	return taps ;
}

// binds the returning ops (effectively, all ops between the first
// Project\Aggregate and CallSubquery in every branch other than the Join op,
// inclusive) in embedded_plan to plan
// returns true if `embedded_plan` should be free'd after binding its root to
// the call {} op (if there are no more ops left in it), false otherwise
static bool _bind_returning_ops_to_plan
(
	const ExecutionPlan *embedded_plan,  // embedded plan
	const ExecutionPlan *plan            // plan to migrate ops to
) {
	// check if there is a Join operation (from UNION or UNION ALL)
	OpBase *root = embedded_plan->root;
	OpBase *join_op = _get_join(root);

	// 5 place-holders are allocated for a maximum of 5 operations between the
	// returning op (Project/Aggregate) and the CallSubquery op
	// (Sort, Join, Distinct, Skip, Limit)
	uint depth = 0 ;  // depth of returning_op from root
	OPType return_types[] = {OPType_PROJECT, OPType_AGGREGATE};

	if(join_op == NULL) {
		// only one returning projection/aggregation
		OpBase *returning_op =
			ExecutionPlan_LocateOpMatchingTypes (root, return_types, 2, &depth) ;

		ASSERT (returning_op != NULL) ;
		depth++ ;  // accommodate root

		// if the returning op has no children, we need to free its exec-plan
		// after binding it to a new plan
		ExecutionPlan *old_plan = (ExecutionPlan *)returning_op->plan;

		OpBase *ops[depth];
		uint n_ops = ExecutionPlan_CollectUpwards(ops, returning_op);
		ExecutionPlan_MigrateOpsExcludeType(ops, OPType_JOIN, n_ops, plan);
		if(returning_op->childCount == 0) {
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
		// if there is a Union operation, we need to look at all of its branches
		for(uint i = 0; i < join_op->childCount; i++) {
			OpBase *child = join_op->children[i];

			while (true) {
				OPType t = OpBase_Type (child) ;

				// do not go beyond a project / aggregate operation
				bool stop = (t == OPType_PROJECT || t == OPType_AGGREGATE) ;

				// consumed the entire join stream
				// free its plan
				if (stop && OpBase_ChildCount (child) == 0) {
						ExecutionPlan *old_plan = (ExecutionPlan *)child->plan ;
						old_plan->root = NULL ;
						ExecutionPlan_Free (old_plan) ;
				}

				OpBase_BindOpToPlan (child, plan) ;

				if (stop) {
					break ;
				}

				// continue on to the LHS
				child = OpBase_GetChild (child, 0) ;
			}
		}

		// if there is a join op, we never free the embedded plan
		return false;
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
	AST *orig_ast = QueryCtx_GetAST () ;

	// create an AST from the body of the subquery
	AST subquery_ast = {0} ;
	subquery_ast.root = cypher_ast_call_subquery_get_query (clause) ;
	subquery_ast.anot_ctx_collection = orig_ast->anot_ctx_collection ;

	//--------------------------------------------------------------------------
	// build the embedded execution plan corresponding to the subquery
	//--------------------------------------------------------------------------

	QueryCtx_SetAST (&subquery_ast) ;
	ExecutionPlan *embedded_plan = ExecutionPlan_FromTLS_AST () ;

	// restore original AST
	QueryCtx_SetAST (orig_ast) ;

	// characterize whether the sub-query performs modifications or not
	// if it is the call sub-query becomes eager, consuming all possible records
	// before running
	bool is_eager = ExecutionPlan_LocateOpMatchingTypes (embedded_plan->root,
			MODIFYING_OPERATIONS, MODIFYING_OP_COUNT, NULL) != NULL ;

	// characterize whether the query is returning or not
	bool is_returning = (OpBase_Type(embedded_plan->root) == OPType_RESULTS) ;

	//--------------------------------------------------------------------------
	// bind returning projection(s)\aggregation(s) to the outer plan
	//--------------------------------------------------------------------------

	bool free_embedded_plan = false;
	if (is_returning) {
		// remove the Results op from the embedded execution-plan
		OpBase *results_op = embedded_plan->root ;
		ASSERT (OpBase_Type (results_op) == OPType_RESULTS) ;
		ExecutionPlan_RemoveOp (embedded_plan, embedded_plan->root) ;
		OpBase_Free (results_op) ;

		// bind the returning ops to the outer plan
		free_embedded_plan = _bind_returning_ops_to_plan (embedded_plan, plan) ;
	}

	//--------------------------------------------------------------------------
	// plant feeders
	//--------------------------------------------------------------------------

	// find the feeding points, to which we will add the projections and feeders
	OpBase **feeding_points = _find_feeding_points (embedded_plan) ;
	const char **imported_variables = _collect_imported_variables (clause) ;

	uint n_feeding_points = arr_len (feeding_points) ;
	for (uint i = 0; i < n_feeding_points; i++) {
		const ExecutionPlan *feeding_plan = feeding_points[i]->plan ;
		const char **variables =
			_feeding_point_imported_variables (feeding_points[i],
					imported_variables) ;
		OpBase *argument = NewArgumentOp (feeding_plan, variables) ;
		ExecutionPlan_AddOp (feeding_points[i], argument) ;
		arr_free (variables) ;
	}

	arr_free(imported_variables);
	arr_free(feeding_points);

	//--------------------------------------------------------------------------
	// connect the embedded plan
	//--------------------------------------------------------------------------

	OpBase *call_op;

	// in case the sub-query returns data use the Apply op
	// to merge records, otherwise the sub-query doesn't return anything
	// and we can simply ignore its emitted records, in this case use the
	// SubQueryForeach op
	if (is_returning) {
		call_op = NewApplyOp (plan) ;
	} else {
		call_op = NewSubQueryForeach (plan) ;
	}

	ExecutionPlan_UpdateRoot (plan, call_op) ;

	// in case there's no operation feeding the `call_op`
	// add EmptyRow op as an input
	if (OpBase_ChildCount (call_op) == 0) {
		ExecutionPlan_AddOp (call_op, NewEmptyRow (plan)) ;
	}

	// in case there's an input stream ChildCount > 0
	// and the sub-query has write operation(s)
	// add an eager operation to drain the input stream before any modifications
	// are applied
	else if (is_eager) {
		ExecutionPlan_PushBelow (OpBase_GetChild(call_op, 0), NewEagerOp (plan)) ;
	}

	// attach the embedded plan
	ExecutionPlan_AddOp (call_op, embedded_plan->root) ;

	if (is_eager) {
		ExecutionPlan_UpdateRoot (plan, NewEagerOp (plan)) ;
	}

	if (free_embedded_plan) {
		embedded_plan->root = NULL;
		ExecutionPlan_Free (embedded_plan) ;
	}
}
