/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../value.h"
#include "arithmetic_expression.h"
#include "arithmetic_expression_eval.h"

//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------

// clear an op node internals, without freeing the node allocation itself
void _AR_EXP_FreeOpInternals(AR_ExpNode *op_node);

static bool _ReduceOperand
(
	AR_ExpNode *root,    // expression to reduce
	bool reduce_params,  // should reduce params
	SIValue *val         // value representing reduced expression
) {
	ASSERT (root->type == AR_EXP_OPERAND) ;

	// in runtime, parameters are set so they can be evaluated
	if (reduce_params && AR_EXP_IsParameter (root)) {
		SIValue v = AR_EXP_Evaluate (root, NULL) ;

		if (val != NULL) {
			*val = v ;
		}
		return true ;
	}

	if (AR_EXP_IsConstant (root)) {
		// root is already a constant
		if (val != NULL) {
			*val = root->operand.constant ;
		}
		return true ;
	}

	// root is variadic, no way to reduce
	ASSERT (root->operand.type == AR_EXP_VARIADIC) ;
	return false ;
}

static bool _ReduceOperation
(
	AR_ExpNode *root,    // expression to reduce
	bool reduce_params,  // should reduce params
	SIValue *val         // value representing reduced expression
) {
	ASSERT (root->type == AR_EXP_OP) ;

	//--------------------------------------------------------------------------
	// reduce children
	//--------------------------------------------------------------------------

	AR_OpNode *op = &root->op ;
	bool all_children_reduced = true ;
	size_t n = op->child_count;

	op->constant_mask = 0 ;  // clear constant_mask

	for (int i = 0; i < n; i++) {
		SIValue v ;
		AR_ExpNode *child = op->children[i] ;

		if (AR_EXP_ReduceToScalar (child, reduce_params, &v)) {
			op->constant_mask |= 1ULL << i ;
			op->cached_constants[i] = v ;
		} else {
			all_children_reduced = false ;
		}
	}

	// can't reduce root as one of its children is not a constant
	if (!all_children_reduced) {
		return false ;
	}

	// all child nodes been reduced, reduce root only if its function is
	// marked as reducible
	if (!op->f->reducible) {
		return false ;
	}

	// evaluate function
	SIValue res = AR_EXP_Evaluate (root, NULL) ;
	if (val != NULL) {
		*val = res ;
	}

	// why ?
	if (SIValue_IsNull (res)) {
		return false ;
	}

	// reduce
	// clear children and function context
	_AR_EXP_FreeOpInternals (root) ;

	// in-place update, set as constant
	root->type             = AR_EXP_OPERAND ;
	root->operand.type     = AR_EXP_CONSTANT ;
	root->operand.constant = res ;

	return true ;

}

// compact tree by evaluating constant expressions
// e.g. MINUS(X) where X is a constant number will be reduced to
// a single node with the value -X
// PLUS(MINUS(A), B) will be reduced to a single constant: B-A
bool AR_EXP_ReduceToScalar
(
	AR_ExpNode *root,    // expression to reduce
	bool reduce_params,  // should reduce params
	SIValue *val         // value representing reduced expression
) {
	if (val != NULL) {
		*val = SI_NullVal () ;
	}

	switch (root->type) {
		case AR_EXP_OPERAND :
			return _ReduceOperand (root, reduce_params, val) ;

		case AR_EXP_OP:
			return _ReduceOperation (root, reduce_params, val) ;

		default:
			assert (false && "unknown arithmetic expression type") ;
			return false ;
	}
}

