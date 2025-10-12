/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "execution_plan_construct.h"
#include "execution_plan_modify.h"
#include "../../RG.h"
#include "../ops/ops.h"
#include "../../query_ctx.h"
#include "../execution_plan.h"
#include "../../ast/ast_build_op_contexts.h"
#include "../../arithmetic/arithmetic_expression_construct.h"

// handle ORDER entities
static AR_ExpNode **_BuildOrderExpressions
(
	const cypher_astnode_t *order_clause
) {
	uint count = cypher_ast_order_by_nitems (order_clause) ;
	AR_ExpNode **order_exps = array_new (AR_ExpNode *, count) ;

	for (uint i = 0; i < count; i++) {
		const cypher_astnode_t *item =
			cypher_ast_order_by_get_item (order_clause, i) ;

		const cypher_astnode_t *ast_exp =
			cypher_ast_sort_item_get_expression (item) ;

		AR_ExpNode *exp = AR_EXP_FromASTNode (ast_exp) ;
		exp->resolved_name = AST_ToString (ast_exp) ;
		array_append (order_exps, exp) ;
	}

	return order_exps ;
}

// handle projected entities
// (this function is not static because it is relied upon by unit tests)
AR_ExpNode **_BuildProjectionExpressions
(
	const cypher_astnode_t *clause
) {
	uint count = 0 ;
	AR_ExpNode **expressions = NULL ;
	cypher_astnode_type_t t = cypher_astnode_type (clause) ;

	ASSERT (t == CYPHER_AST_RETURN || t == CYPHER_AST_WITH) ;

	if (t == CYPHER_AST_RETURN) {
		// if we have a "RETURN *" at this point, it is because we raised 
		// an error in AST rewriting
		if (cypher_ast_return_has_include_existing(clause)) {
			return NULL ;
		}
		count = cypher_ast_return_nprojections (clause) ;
	} else {
		ASSERT (cypher_ast_with_has_include_existing (clause) == false) ;
		count = cypher_ast_with_nprojections (clause) ;
	}

	expressions = array_new (AR_ExpNode *, count) ;

	rax *rax = raxNew();

	for (uint i = 0; i < count; i++) {
		const cypher_astnode_t *projection = (t == CYPHER_AST_RETURN) ?
			cypher_ast_return_get_projection (clause, i) :
			cypher_ast_with_get_projection   (clause, i) ;

		// the AST expression can be an identifier, function call, or constant
		const cypher_astnode_t *ast_exp =
			cypher_ast_projection_get_expression (projection) ;

		// find the resolved name of the entity - its alias,
		// its identifier if referring to a full entity,
		// the entity.prop combination ("a.val"),
		// or the function call ("MAX(a.val)")
		const char *identifier = NULL ;
		const cypher_astnode_t *alias_node =
			cypher_ast_projection_get_alias (projection) ;

		if (alias_node) {
			// the projection either has an alias (AS), is a function call,
			// or is a property specification (e.name).
			identifier = cypher_ast_identifier_get_name (alias_node) ;
		} else {
			// this expression did not have an alias
			// so it must be an identifier
			ASSERT (cypher_astnode_type(ast_exp) == CYPHER_AST_IDENTIFIER) ;
			// retrieve "a" from "RETURN a" or "RETURN a AS e"
			// (theoretically; the latter case is already handled)
			identifier = cypher_ast_identifier_get_name (ast_exp) ;
		}

		if (raxTryInsert(rax, (unsigned char *)identifier, strlen(identifier),
					NULL, NULL) != 0) {
			// construction an AR_ExpNode to represent this projected entity
			AR_ExpNode *exp = AR_EXP_FromASTNode (ast_exp) ;
			exp->resolved_name = identifier ;
			array_append (expressions, exp) ;
		}
	}

	raxFree(rax);

	return expressions;
}

// build an aggregate or project operation and any required modifying operations
// this logic applies for both WITH and RETURN projections
static inline void _buildProjectionOps
(
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	OpBase                 *op              = NULL  ;
	OpBase                 *distinct_op     = NULL  ;
	bool                   distinct         = false ;
	bool                   aggregate        = false ;
	int                    *sort_directions = NULL  ;
	AR_ExpNode             **order_exps     = NULL  ;
	AR_ExpNode             **projections    = NULL  ;
	const cypher_astnode_t *skip_clause     = NULL  ;
	const cypher_astnode_t *limit_clause    = NULL  ;
	const cypher_astnode_t *order_clause    = NULL  ;

	cypher_astnode_type_t t = cypher_astnode_type (clause) ;
	ASSERT (t == CYPHER_AST_WITH || t == CYPHER_AST_RETURN) ;

	aggregate = AST_ClauseContainsAggregation (clause) ;
	projections = _BuildProjectionExpressions (clause) ;

	if (t == CYPHER_AST_WITH) {
		distinct     = cypher_ast_with_is_distinct  (clause) ;
		skip_clause  = cypher_ast_with_get_skip     (clause) ;
		limit_clause = cypher_ast_with_get_limit    (clause) ;
		order_clause = cypher_ast_with_get_order_by (clause) ;
	} else {
		distinct     = cypher_ast_return_is_distinct  (clause) ;
		skip_clause  = cypher_ast_return_get_skip     (clause) ;
		limit_clause = cypher_ast_return_get_limit    (clause) ;
		order_clause = cypher_ast_return_get_order_by (clause) ;
	}

	if (distinct) {
		// prepare the distinct op but do not add it to op tree
		// this is required so that it does not operate on order expressions
		// e.g.
		//
		// MATCH (a), (b) WITH b.v AS v ORDER BY a.v
		// in which case `a.v` is added to the projected expressions
		uint n = array_len (projections) ;

		// populate a stack array with the aliases to perform Distinct on
		const char *aliases[n] ;
		for (uint i = 0; i < n; i ++) {
			aliases[i] = projections[i]->resolved_name ;
		}

		distinct_op = NewDistinctOp (plan, aliases, n) ;
	}

	if (order_clause) {
		AST_PrepareSortOp (order_clause, &sort_directions) ;
		order_exps = _BuildOrderExpressions (order_clause) ;
		normalize_sort_exps (&projections, order_exps, aggregate) ;
	}

	// our fundamental operation will be a projection or aggregation
	if (aggregate) {
		op = NewAggregateOp (plan, projections) ;
	} else {
		op = NewProjectOp (plan, projections) ;
	}
	ExecutionPlan_UpdateRoot (plan, op) ;

	// add modifier operations in order such that the final execution plan will
	// follow the sequence:
	// Limit -> Skip -> Sort -> Distinct -> Project/Aggregate

	if (distinct_op) {
		ExecutionPlan_UpdateRoot (plan, distinct_op) ;
	}

	if (sort_directions) {
		op = NewSortOp (plan, order_exps, sort_directions) ;
		ExecutionPlan_UpdateRoot (plan, op) ;
	}

	if (skip_clause) {
		op = buildSkipOp (plan, skip_clause) ;
		ExecutionPlan_UpdateRoot (plan, op) ;
	}

	if (limit_clause) {
		op = buildLimitOp (plan, limit_clause) ;
		ExecutionPlan_UpdateRoot (plan, op) ;
	}
}

// RETURN builds a subtree of projection ops with Results operation as the root
void buildReturnOps
(
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	_buildProjectionOps (plan, clause) ;

	// follow up with a Result operation
	OpBase *op = NewResultsOp (plan) ;
	ExecutionPlan_UpdateRoot (plan, op) ;
}

// WITH builds a subtree of projection ops
void buildWithOps
(
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	_buildProjectionOps (plan, clause) ;
}

