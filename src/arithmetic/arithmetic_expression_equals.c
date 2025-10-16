/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "arithmetic_expression.h"

static bool _OpEquals
(
	const AR_OpNode *a,  // first op expression
	const AR_OpNode *b   // second op expression
) {
	// check if a and b represent the same function
	// and have the same number of arguments
	bool eq = (a->f == b->f && a->child_count == b->child_count) ;
	if (!eq) {
		return false ;
	}

	for (int i = 0; i < a->child_count; i++) {
		if (!AR_EXP_Equals (a->children[i], b->children[i])) {
			return false ;
		}
	}

	return true ;
}

static bool _OperandEquals
(
	const AR_OperandNode *a,  // first operand expression
	const AR_OperandNode *b   // second operand expression
) {
	if (a->type != b->type) {
		return false ;
	}

	switch (a->type) {
		case AR_EXP_CONSTANT:
			return SIValue_Compare (a->constant, b->constant, NULL) == 0 ;
		case AR_EXP_VARIADIC:
			return strcmp (a->variadic.entity_alias,
						   b->variadic.entity_alias) == 0 ;
		case AR_EXP_PARAM:
			return strcmp (a->param_name, b->param_name) == 0 ;
		case AR_EXP_BORROW_RECORD:
			return true ;
		default:
			ASSERT ("unknown operand type" && false) ;
			return false ;
	}
}

// compare two expressions
// returns true if `a` and `b` represent the same expression
bool AR_EXP_Equals
(
	const AR_ExpNode *a,  // first expression
	const AR_ExpNode *b   // second expression
) {
	ASSERT (a != NULL) ;
	ASSERT (b != NULL) ;

	bool res = false ;

	if (a->type == b->type) {
		switch (a->type) {
			case AR_EXP_OP:
				res = _OpEquals (&a->op, &b->op) ;
				break ;
			case AR_EXP_OPERAND:
				res = _OperandEquals (&a->operand, &b->operand) ;
				break ;
			default:
				ASSERT ("unknown expression type" && false) ;
				return false ;
		}
	}

	// i hate having this last condition but some edge cases e.g.
	//
	// MATCH p = (a) RETURN length(p) ORDER BY length(p)
	//
	// would create two different expressions:
	// 1. length(to_path([a]))
	// 2. length(p)
	//
	// semantically the two are the same, moreover the user specified them
	// as the same, and so we're falling back on string comparison
	// as a last resort
	if (res == false && a->resolved_name != NULL && b->resolved_name != NULL) {
		res = strcmp (a->resolved_name, b->resolved_name) == 0 ;
	}

	return res ;
}

