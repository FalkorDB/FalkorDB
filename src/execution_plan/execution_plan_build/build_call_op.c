/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../ast/ast.h"
#include "../../util/arr.h"
#include "../../procedures/procedure.h"
#include "../../ast/ast_build_filter_tree.h"
#include "../../execution_plan/execution_plan.h"
#include "../../execution_plan/ops/op_procedure_call.h"
#include "../../arithmetic/arithmetic_expression_construct.h"
#include "../../execution_plan/execution_plan_build/execution_plan_modify.h"
#include "../../execution_plan/execution_plan_build/execution_plan_construct.h"

/* Strings enclosed in the parentheses of a CALL clause represent the arguments to the procedure.
 * _BuildCallArguments creates a string array holding all of these arguments. */
static AR_ExpNode **_BuildCallArguments(const cypher_astnode_t *call_clause) {
	// Handle argument entities
	uint arg_count = cypher_ast_call_narguments(call_clause);
	AR_ExpNode **arguments = array_new(AR_ExpNode *, arg_count);
	for(uint i = 0; i < arg_count; i ++) {
		const cypher_astnode_t *exp = cypher_ast_call_get_argument(call_clause, i);
		AR_ExpNode *arg = AR_EXP_FromASTNode(exp);
		array_append(arguments, arg);
	}

	return arguments;
}

/* _BuildCallProjections creates an array of expression nodes to populate a Project operation with.
 * All Strings in the YIELD block of a CALL clause are represented, or the procedure-registered
 * outputs if the YIELD block is missing. */
static AR_ExpNode **_BuildCallProjections(const cypher_astnode_t *call_clause) {
	// Handle yield entities
	uint yield_count = cypher_ast_call_nprojections(call_clause);
	AR_ExpNode **expressions = array_new(AR_ExpNode *, yield_count);

	for(uint i = 0; i < yield_count; i ++) {
		const cypher_astnode_t *projection = cypher_ast_call_get_projection(call_clause, i);
		const cypher_astnode_t *ast_exp = cypher_ast_projection_get_expression(projection);

		// Construction an AR_ExpNode to represent this entity.
		AR_ExpNode *exp = AR_EXP_FromASTNode(ast_exp);

		const char *identifier = NULL;
		const cypher_astnode_t *alias_node = cypher_ast_projection_get_alias(projection);
		if(alias_node) {
			// The projection either has an alias (AS), is a function call, or is a property specification (e.name).
			identifier = cypher_ast_identifier_get_name(alias_node);
		} else {
			// This expression did not have an alias, so it must be an identifier
			ASSERT(cypher_astnode_type(ast_exp) == CYPHER_AST_IDENTIFIER);
			// Retrieve "a" from "RETURN a" or "RETURN a AS e" (theoretically; the latter case is already handled)
			identifier = cypher_ast_identifier_get_name(ast_exp);
		}

		exp->resolved_name = identifier;
		array_append(expressions, exp);
	}

	// If the procedure call is missing its yield part, include procedure outputs.
	if(yield_count == 0) {
		const char *proc_name = cypher_ast_proc_name_get_value(cypher_ast_call_get_proc_name(call_clause));
		ProcedureCtx *proc = Proc_Get(proc_name);
		ASSERT(proc != NULL);

		unsigned int output_count = Procedure_OutputCount(proc);
		for(uint i = 0; i < output_count; i++) {
			const char *name = Procedure_GetOutput(proc, i);
			AR_ExpNode *exp = AR_EXP_NewVariableOperandNode(name);
			exp->resolved_name = name;
			array_append(expressions, exp);
		}
		Proc_Free(proc);
	}

	return expressions;
}

// convert a CALL clause into a procedure call operation
void buildCallOp
(
	AST *ast,
	ExecutionPlan *plan,
	const cypher_astnode_t *call_clause
) {
	// a call clause has a procedure name
	// 0+ arguments (parenthesized expressions)
	// and a projection if YIELD is included
	const cypher_astnode_t *proc = cypher_ast_call_get_proc_name(call_clause);
	const char *proc_name = cypher_ast_proc_name_get_value(proc);
	AR_ExpNode **arguments = _BuildCallArguments(call_clause);
	AR_ExpNode **yield_exps = _BuildCallProjections(call_clause); // TODO only need strings

	OpBase *op = NewProcCallOp(plan, proc_name, arguments, yield_exps);
	ExecutionPlan_UpdateRoot(plan, op);

	// Build the FilterTree to model any WHERE predicates on this clause
	// and place ops appropriately.
	FT_FilterNode *sub_ft = AST_BuildFilterTreeFromClauses(ast, &call_clause, 1);
	ExecutionPlan_PlaceFilterOps(plan, plan->root, sub_ft);
}

