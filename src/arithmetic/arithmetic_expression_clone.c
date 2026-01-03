/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../util/rmalloc.h"
#include "./arithmetic_expression.h"

// clone operand
static AR_ExpNode *_AR_EXP_CloneOperand
(
	const AR_ExpNode *exp  // operand to clone
) {
	AR_ExpNode *clone = rm_malloc (sizeof (AR_ExpNode)) ;
	memcpy (clone, exp, sizeof (AR_ExpNode)) ;

	if (exp->operand.type == AR_EXP_CONSTANT) {
		clone->operand.constant = SI_ShallowCloneValue(exp->operand.constant);
	}

	return clone;
}

// clone an 'OP' node
static AR_ExpNode *_AR_EXP_CloneOp
(
	const AR_ExpNode *exp
) {
	ASSERT (exp       != NULL) ;
	ASSERT (exp->op.f != NULL) ;
	ASSERT (exp->type == AR_EXP_OP) ;

	AR_ExpNode *clone = rm_malloc (sizeof (AR_ExpNode)) ;
	memcpy (clone, exp, sizeof (AR_ExpNode)) ;

	//--------------------------------------------------------------------------
	// clone cached constants
	//--------------------------------------------------------------------------

	size_t cached_constants_size = sizeof (SIValue) * clone->op.child_count ;
	clone->op.cached_constants = rm_malloc (cached_constants_size) ;
	memcpy (clone->op.cached_constants, exp->op.cached_constants,
			cached_constants_size) ;

	uint child_count   = exp->op.child_count ;
	clone->op.children = rm_calloc (child_count, sizeof (AR_ExpNode *)) ;

	AR_FuncDesc *f = clone->op.f ;

	// not expecting both callbacks
	ASSERT (! (f->callbacks.private_data != NULL &&
			   f->callbacks.clone != NULL)) ;

	// add aggregation context as function private data
	if (f->aggregate) {
		// generate aggregation context and store it in node's private data
		ASSERT (f->callbacks.private_data != NULL) ;
		clone->op.private_data = f->callbacks.private_data () ;
	}

	if (f->callbacks.clone) {
		// clone callback specified, use it to duplicate function's private data
		AR_Func_Clone clone_cb = f->callbacks.clone ;
		void *pdata = exp->op.private_data ;
		clone->op.private_data = clone_cb (pdata) ;
	}

	// clone child nodes
	for (uint i = 0; i < exp->op.child_count; i++) {
		clone->op.children[i] = AR_EXP_Clone (exp->op.children[i]) ;
	}

	return clone ;
}

// clone arithmetic expression
AR_ExpNode *AR_EXP_Clone
(
	const AR_ExpNode *exp  // expression to clone
) {
	if (exp == NULL) {
		return NULL ;
	}

	AR_ExpNode *clone = NULL;

	switch (exp->type) {
		case AR_EXP_OPERAND:
			clone = _AR_EXP_CloneOperand (exp) ;
			break;

		case AR_EXP_OP:
			clone = _AR_EXP_CloneOp (exp) ;
			break;

		default:
			ASSERT (false) ;
			break;
	}

	clone->resolved_name = exp->resolved_name ;

	return clone ;
}

