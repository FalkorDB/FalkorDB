/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "utils.h"
#include "../algebraic_expression.h"

static const AlgebraicExpression *_AlgebraicExpression_SrcOperand
(
	const AlgebraicExpression *root,
	bool *transposed
) {
	ASSERT(root != NULL);
	ASSERT(root->type == AL_OPERAND || root->type == AL_OPERATION);

	bool t = *transposed;
	AlgebraicExpression *exp = (AlgebraicExpression *)root;

	while(exp->type == AL_OPERATION) {
		switch(exp->operation.op) {
			case AL_EXP_ADD:
				// Src (A+B) = Src(A)
				// Src (Transpose(A+B)) = Src (Transpose(A)+Transpose(B)) = Src (Transpose(A))
				exp = FIRST_CHILD(exp);
				break;
			case AL_EXP_MUL:
				// Src (A*B) = Src(A)
				// Src (Transpose(A*B)) = Src (Transpose(B)*Transpose(A)) = Src (Transpose(B))
				exp = (t) ? LAST_CHILD(exp) : FIRST_CHILD(exp);
				break;
			case AL_EXP_TRANSPOSE:
				// Src (Transpose(Transpose(A))) = Src(A)
				// negate transpose
				t = !t;
				exp = FIRST_CHILD(exp);
				break;
			default:
				ASSERT("Unknown algebraic expression operation" && false);
				return NULL;
		}
	}

	*transposed = t;
	return exp;
}

const AlgebraicExpression *AlgebraicExpression_SrcOperand
(
	const AlgebraicExpression *root   // root of expression
) {
	ASSERT(root != NULL);

	bool transposed = false;
	return _AlgebraicExpression_SrcOperand(root, &transposed);
}

const AlgebraicExpression *AlgebraicExpression_DestOperand
(
	const AlgebraicExpression *root   // root of expression
) {
	ASSERT(root != NULL);

	bool transposed = true;
	return _AlgebraicExpression_SrcOperand(root, &transposed);
}

// returns the source entity alias, row domain
const char *AlgebraicExpression_Src
(
	const AlgebraicExpression *root  // root of expression
) {
	ASSERT(root != NULL);

	bool transposed = false;
	const AlgebraicExpression *exp = NULL;

	exp = _AlgebraicExpression_SrcOperand(root, &transposed);
	return (transposed) ? exp->operand.dest : exp->operand.src;
}

// returns the destination entity alias represented by the right-most operand
// column domain
const char *AlgebraicExpression_Dest
(
	const AlgebraicExpression *root   // root of expression
) {
	ASSERT(root);
	// Dest(exp) = Src(Transpose(exp))
	// Gotta love it!

	bool transposed = true;
	const AlgebraicExpression *exp = NULL;

	exp = _AlgebraicExpression_SrcOperand(root, &transposed);
	return (transposed) ? exp->operand.dest : exp->operand.src;
}

