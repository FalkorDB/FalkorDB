/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
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

// collect operands originating at root
// the operands are collected in order from the leftmost to the rightmost
// such that the leftmost will be stored in the retuned array at position 0
// and the rightmost will be stored as the last element of the array
AlgebraicExpression **AlgebraicExpression_CollectOperandsInOrder
(
	const AlgebraicExpression *root,  // root from which to collect operands
	uint *n                           // [output] number of operands collected
) {
	ASSERT(n    != NULL);
	ASSERT(root != NULL);

	// determine size of output array
	uint idx = 0;
	uint _n  = AlgebraicExpression_OperandCount(root);

	// allocate output array
	AlgebraicExpression **operands =
		rm_calloc(_n, sizeof(AlgebraicExpression*));

	// create local queue for DFS style traversal
	// visiting nodes from left to right
	// NOTE: this function doesn't treat operands switch by transpose operation
	// e.g. T(A*B) -> Bt * At
	// will treat A as the leftmost operand and B as the rightmost
	AlgebraicExpression **queue = array_new(AlgebraicExpression*, 1);
	array_append(queue, (AlgebraicExpression*)root);

	// as long as there are nodes to visit
	while(array_len(queue) > 0) {
		// get the newest node added to the queue
		AlgebraicExpression *current = array_pop(queue);

		if(current->type == AL_OPERATION) {
			// push children to queue
			uint child_count = AlgebraicExpression_ChildCount(current);
			for(uint i = 0; i < child_count; i++) {
				array_append(queue, CHILD_AT(current, i));
			}
		} else {
			// node is an operand add it to the output array
			operands[idx++] = current;
		}
	}
	array_free(queue);

	// set output 'n'
	ASSERT(idx == _n);
	*n = _n;

	return operands;
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

