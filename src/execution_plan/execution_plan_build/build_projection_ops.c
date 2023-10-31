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
#include "../../ast/ast_build_filter_tree.h"
#include "../../ast/ast_build_op_contexts.h"
#include "../../arithmetic/arithmetic_expression_construct.h"

// handle ORDER entities
static AR_ExpNode **_BuildOrderExpressions
(
	AR_ExpNode **projections,
	const cypher_astnode_t *order_clause
) {
	uint count = cypher_ast_order_by_nitems(order_clause);
	AR_ExpNode **order_exps = array_new(AR_ExpNode *, count);

	for(uint i = 0; i < count; i++) {
		const cypher_astnode_t *item = cypher_ast_order_by_get_item(order_clause, i);
		const cypher_astnode_t *ast_exp = cypher_ast_sort_item_get_expression(item);
		AR_ExpNode *exp = AR_EXP_FromASTNode(ast_exp);
		exp->resolved_name = AST_ToString(ast_exp);
		array_append(order_exps, exp);
	}

	return order_exps;
}

// handle projected entities
// (this function is not static because it is relied upon by unit tests)
AR_ExpNode **_BuildProjectionExpressions
(
	const cypher_astnode_t *clause
) {
	uint count = 0;
	AR_ExpNode **expressions = NULL;
	cypher_astnode_type_t t = cypher_astnode_type(clause);

	ASSERT(t == CYPHER_AST_RETURN || t == CYPHER_AST_WITH);

	if(t == CYPHER_AST_RETURN) {
		// if we have a "RETURN *" at this point, it is because we raised 
		// an error in AST rewriting
		if(cypher_ast_return_has_include_existing(clause)) return NULL;
		count = cypher_ast_return_nprojections(clause);
	} else {
		ASSERT(cypher_ast_with_has_include_existing(clause) == false);
		count = cypher_ast_with_nprojections(clause);
	}

	expressions = array_new(AR_ExpNode *, count);

	rax *rax = raxNew();

	for(uint i = 0; i < count; i++) {
		const cypher_astnode_t *projection = NULL;
		if(t == CYPHER_AST_RETURN) {
			projection = cypher_ast_return_get_projection(clause, i);
		} else {
			projection = cypher_ast_with_get_projection(clause, i);
		}

		// the AST expression can be an identifier, function call, or constant
		const cypher_astnode_t *ast_exp =
			cypher_ast_projection_get_expression(projection);

		// find the resolved name of the entity - its alias,
		// its identifier if referring to a full entity,
		// the entity.prop combination ("a.val"),
		// or the function call ("MAX(a.val)")
		const char *identifier = NULL;
		const cypher_astnode_t *alias_node =
			cypher_ast_projection_get_alias(projection);

		if(alias_node) {
			// the projection either has an alias (AS), is a function call,
			// or is a property specification (e.name)
			identifier = cypher_ast_identifier_get_name(alias_node);
		} else {
			// this expression did not have an alias,
			// so it must be an identifier
			ASSERT(cypher_astnode_type(ast_exp) == CYPHER_AST_IDENTIFIER);
			// retrieve "a" from "RETURN a" or "RETURN a AS e"
			// (theoretically; the latter case is already handled)
			identifier = cypher_ast_identifier_get_name(ast_exp);
		}

		if(raxTryInsert(rax, (unsigned char *)identifier, strlen(identifier),
					NULL, NULL) != 0) {
			// construction an AR_ExpNode to represent this projected entity
			AR_ExpNode *exp = AR_EXP_FromASTNode(ast_exp);
			exp->resolved_name = identifier;
			array_append(expressions, exp);
		}
	}

	raxFree(rax);

	return expressions;
}

// build WITH projection filter operation
static OpBase* _buildProjectionFilters
(
	ExecutionPlan *plan,           // execution plan
	const cypher_astnode_t *with,  // WITH clause
	AR_ExpNode **projections       // projection expressions
) {
	cypher_astnode_type_t t = cypher_astnode_type(with);
	ASSERT(t == CYPHER_AST_WITH);

	// unfortunately, the following query is legal
	//
	// MATCH (a), (b)
	// WITH b AS c
	// WHERE c.v = a.v
	// RETURN c
	//
	// 'a' is accessible to the WITH clause filter
	// to enable this a new projection operation is required
	// it is as if the user had specified the following
	//
	// MATCH (a), (b)
	// WITH a, b AS c
	// WHERE c.v = a.v
	// WITH c
	// RETURN c

	//--------------------------------------------------------------------------
	// construct filter tree
	//--------------------------------------------------------------------------

	FT_FilterNode *ft = AST_BuildFilterTreeFromClauses(NULL, &with, 1);
	OpBase *filter_op = NewFilterOp(plan, ft);

	//--------------------------------------------------------------------------
	// collect projected identifiers
	//--------------------------------------------------------------------------

	rax *projected_identifiers = raxNew();
	uint n = array_len(projections);
	for(uint i = 0; i < n; i++) {
		AR_ExpNode *exp = projections[i];
		const char *identifier = exp->resolved_name;
		raxInsert(projected_identifiers, (unsigned char *)identifier,
				strlen(identifier), NULL, NULL);
	}

	//--------------------------------------------------------------------------
	// collect filtered identifiers
	//--------------------------------------------------------------------------

	const char **filter_identifiers = array_new(const char *, 1);
	const cypher_astnode_t *predicate = cypher_ast_with_get_predicate(with);
	AST_CollectUniqueAliases(&filter_identifiers, predicate);

	// determine if all filtered identifiers are projected
	// if so, there's no need for an additional projection operation
	// if not, we need to add another projection operation
	n = array_len(filter_identifiers);
	bool new_projection_required = false;
	for(uint i = 0; i < n; i++) {
		const char *identifier = filter_identifiers[i];
		if(raxFind(projected_identifiers, (unsigned char *)identifier,
					strlen(identifier)) == raxNotFound) {
			new_projection_required = true;
			break;
		}
	}

	if(new_projection_required) {
		// a new projection is required, add missing identifiers

		// clone original projections
		AR_ExpNode **new_projections;
		array_clone_with_cb(new_projections, projections, AR_EXP_Clone);

		for(uint i = 0; i < n; i++) {
			const char *identifier = filter_identifiers[i];
			if(raxFind(projected_identifiers, (unsigned char *)identifier,
						strlen(identifier)) == raxNotFound) {
				// add missing identifier
				AR_ExpNode *exp = AR_EXP_NewVariableOperandNode(identifier);
				array_append(new_projections, exp);
			}
		}

		// introduce filter operation to the execution plan
		// and connect it to the new projection operation
		// ProjectOp -> FilterOp
		ExecutionPlan_UpdateRoot(plan, filter_op);

		// create the new projection operation
		OpBase *project_op = NewProjectOp(plan, new_projections);
		OpBase_AddChild(filter_op, project_op);

		// return NULL to caller
		filter_op = NULL;
	}

	array_free(filter_identifiers);
	raxFree(projected_identifiers);

	return filter_op;
}

// merge all order expressions into the projections array without duplicates
static void _combine_projection_arrays
(
	AR_ExpNode ***exps_ptr,
	AR_ExpNode **order_exps
) {
	rax        *projection_names = raxNew();
	AR_ExpNode **project_exps    = *exps_ptr;
	uint       order_count       = array_len(order_exps);
	uint       project_count     = array_len(project_exps);

	// add all WITH/RETURN projection names to rax
	for(uint i = 0; i < project_count; i ++) {
		const char *name = project_exps[i]->resolved_name;
		raxTryInsert(projection_names, (unsigned char *)name, strlen(name),
				NULL, NULL);
	}

	// merge non-duplicate order expressions into projection array
	for(uint i = 0; i < order_count; i ++) {
		const char *name = order_exps[i]->resolved_name;
		int new_name = raxTryInsert(projection_names, (unsigned char *)name,
				strlen(name), NULL, NULL);
		// if it is a new projection, add a clone to the array
		if(new_name) array_append(project_exps, AR_EXP_Clone(order_exps[i]));
	}

	raxFree(projection_names);
	*exps_ptr = project_exps;
}

// build an aggregate or project operation and any required modifying operations
// this logic applies for both WITH and RETURN projections
static inline void _buildProjectionOps
(
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	ASSERT(plan   != NULL);
	ASSERT(clause != NULL);

	OpBase                 *op              = NULL;
	OpBase                 *distinct_op     = NULL;
	OpBase                 *filter_op       = NULL;
	bool                   distinct         = false;
	bool                   aggregate        = false;
	bool				   filters          = false;
	int                    *sort_directions = NULL;
	AR_ExpNode             **order_exps     = NULL;
	AR_ExpNode             **projections    = NULL;
	const cypher_astnode_t *skip_clause     = NULL;
	const cypher_astnode_t *limit_clause    = NULL;
	const cypher_astnode_t *order_clause    = NULL;

	cypher_astnode_type_t t = cypher_astnode_type(clause);
	ASSERT(t == CYPHER_AST_WITH || t == CYPHER_AST_RETURN);

	aggregate = AST_ClauseContainsAggregation(clause);
	projections = _BuildProjectionExpressions(clause);

	if(t == CYPHER_AST_WITH) {
		distinct     = cypher_ast_with_is_distinct(clause);
		skip_clause  = cypher_ast_with_get_skip(clause);
		limit_clause = cypher_ast_with_get_limit(clause);
		order_clause = cypher_ast_with_get_order_by(clause);
		filters      = cypher_ast_with_get_predicate(clause) != NULL;
	} else {
		distinct     = cypher_ast_return_is_distinct(clause);
		skip_clause  = cypher_ast_return_get_skip(clause);
		limit_clause = cypher_ast_return_get_limit(clause);
		order_clause = cypher_ast_return_get_order_by(clause);
	}
	
	if(filters) {
		filter_op = _buildProjectionFilters(plan, clause, projections);
	}

	if(distinct) {
		// prepare the distinct op but do not add it to op tree
		// this is required so that it does not operate on order expressions
		uint n = array_len(projections);

		// populate a stack array with the aliases to perform Distinct on
		const char *aliases[n];
		for(uint i = 0; i < n; i ++) aliases[i] = projections[i]->resolved_name;
		distinct_op = NewDistinctOp(plan, aliases, n);
	}

	if(order_clause) {
		AST_PrepareSortOp(order_clause, &sort_directions);
		order_exps = _BuildOrderExpressions(projections, order_clause);
		// merge order expressions into the projections array
		_combine_projection_arrays(&projections, order_exps);
	}

	// our fundamental operation will be a projection or aggregation
	if(aggregate) {
		// an aggregate op's caching policy depends on
		// whether its results will be sorted
		op = NewAggregateOp(plan, projections);
	} else {
		op = NewProjectOp(plan, projections);
	}
	ExecutionPlan_UpdateRoot(plan, op);

	// add modifier operations in order such that the final execution plan
	// will follow the sequence:
	// Limit -> Skip -> Sort -> Distinct -> Filters -> Project/Aggregate

	if(filter_op) {
		ExecutionPlan_UpdateRoot(plan, filter_op);
	}
	
	if(distinct_op) {
		ExecutionPlan_UpdateRoot(plan, distinct_op);
	}

	if(sort_directions) {
		// the sort operation will obey a specified limit
		// but must account for skipped records
		op = NewSortOp(plan, order_exps, sort_directions);
		ExecutionPlan_UpdateRoot(plan, op);
	}

	if(skip_clause) {
		op = buildSkipOp(plan, skip_clause);
		ExecutionPlan_UpdateRoot(plan, op);
	}

	if(limit_clause) {
		op = buildLimitOp(plan, limit_clause);
		ExecutionPlan_UpdateRoot(plan, op);
	}
}

// builds a subtree of projection ops with Results as the root
void buildReturnOps
(
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	_buildProjectionOps(plan, clause);

	// follow up with a Result operation
	OpBase *op = NewResultsOp(plan);
	ExecutionPlan_UpdateRoot(plan, op);
}

// builds a subtree of projection ops
void buildWithOps
(
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
) {
	_buildProjectionOps(plan, clause);
}

