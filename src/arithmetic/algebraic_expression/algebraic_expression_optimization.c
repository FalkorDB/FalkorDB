/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */
#include "utils.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../algebraic_expression.h"
#include "../../configuration/config.h"

static inline bool _AlgebraicExpression_IsMultiplicationNode(const AlgebraicExpression *node) {
	return (node->type == AL_OPERATION && node->operation.op == AL_EXP_MUL);
}

static inline bool _AlgebraicExpression_IsAdditionNode(const AlgebraicExpression *node) {
	return (node->type == AL_OPERATION && node->operation.op == AL_EXP_ADD);
}

/* Collect all operands under given root by performing a left to right scan
 * adding each encountered operand to the `operands` array. */
static void _AlgebraicExpression_CollectOperands(AlgebraicExpression *root,
												 AlgebraicExpression ***operands) {
	uint child_count = 0;

	switch(root->type) {
	case AL_OPERAND:
		array_append(*operands, AlgebraicExpression_Clone(root));
		root->operand.bfree = false; // The caller is the new owner of this operand.
		break;
	case AL_OPERATION:
		switch(root->operation.op) {
		case AL_EXP_TRANSPOSE:
			// Transpose is considered as an operand.
			array_append(*operands, AlgebraicExpression_Clone(root));
			break;
		case AL_EXP_ADD:
		case AL_EXP_MUL:
			child_count = AlgebraicExpression_ChildCount(root);
			for(uint i = 0; i < child_count; i++) {
				_AlgebraicExpression_CollectOperands(CHILD_AT(root, i), operands);
			}
			break;
		default:
			ASSERT("Unknown algebraic expression operation type" && false);
			break;
		}
		break;
	default:
		ASSERT("Unknown algebraic expression node type" && false);
		break;
	}
}

static bool __AlgebraicExpression_MulOverAdd(AlgebraicExpression **root) {
	if(_AlgebraicExpression_IsMultiplicationNode(*root)) {
		AlgebraicExpression *l = CHILD_AT((*root), 0);
		AlgebraicExpression *r = CHILD_AT((*root), 1);

		if(_AlgebraicExpression_IsAdditionNode(l) && _AlgebraicExpression_IsAdditionNode(r)) {
			// MATCH ()-[:A|B]->()-[:C|D]->()
			// (A+B)*(C+D) =
			// = (A*C)+(A*D)+(B*C)+(B*D)

			uint left_op_count = AlgebraicExpression_ChildCount(l);
			uint right_op_count = AlgebraicExpression_ChildCount(r);
			AlgebraicExpression **left_ops = array_new(AlgebraicExpression *, left_op_count);
			AlgebraicExpression **right_ops = array_new(AlgebraicExpression *, right_op_count);

			for(uint i = 0; i < left_op_count; i++) {
				array_append(left_ops, _AlgebraicExpression_OperationRemoveSource(l));
			}
			for(uint i = 0; i < right_op_count; i++) {
				array_append(right_ops, _AlgebraicExpression_OperationRemoveSource(r));
			}

			ASSERT(AlgebraicExpression_ChildCount(l) == 0 && AlgebraicExpression_ChildCount(r) == 0);

			// Multiply each left op by right op: (A*C), (A*D), (B*C), (B*D).
			AlgebraicExpression **multiplications = array_new(AlgebraicExpression *,
															  left_op_count * right_op_count);
			for(uint i = 0; i < left_op_count; i++) {
				AlgebraicExpression *l_op = left_ops[i];
				for(uint j = 0; j < right_op_count; j++) {
					// Clone op as it's being reused: A*C, A*D.
					if(j > 0) l_op = AlgebraicExpression_Clone(l_op);
					AlgebraicExpression *r_op = right_ops[j];
					// Clone op as it's being reused B*C, B*D.
					if(i > 0) r_op = AlgebraicExpression_Clone(r_op);
					AlgebraicExpression *mul = _AlgebraicExpression_MultiplyToTheRight(l_op, r_op);
					array_append(multiplications, mul);
				}
			}
			array_free(left_ops);
			array_free(right_ops);

			// Sum all multiplications: (A*C)+(A*D)+(B*C)+(B*D).
			AlgebraicExpression *add = multiplications[0];
			for(uint i = 1; i < (left_op_count * right_op_count); i++) {
				add = _AlgebraicExpression_AddToTheRight(add, multiplications[i]);
			}

			array_free(multiplications);

			// Free original root and overwrite it with new addition root.
			AlgebraicExpression_Free(*root);
			// Update root.
			*root = add;
			return true;
		}

		else if((_AlgebraicExpression_IsAdditionNode(l) && !_AlgebraicExpression_IsAdditionNode(r)) ||
				(_AlgebraicExpression_IsAdditionNode(r) && !_AlgebraicExpression_IsAdditionNode(l))) {

			// disconnect left and right children from root
			r = _AlgebraicExpression_OperationRemoveDest((*root));
			l = _AlgebraicExpression_OperationRemoveDest((*root));
			ASSERT(AlgebraicExpression_ChildCount(*root) == 0);

			AlgebraicExpression *A;
			AlgebraicExpression *B;
			AlgebraicExpression *add = AlgebraicExpression_NewOperation(AL_EXP_ADD);

			if(_AlgebraicExpression_IsAdditionNode(l)) {
				// lefthand side is addition
				// (A + B + C) * D = (A * D) + (B * D) + (C * D)
				B = r;
				uint child_count = AlgebraicExpression_ChildCount(l);
				for(uint i = 0; i < child_count; i++) {
					A = _AlgebraicExpression_OperationRemoveDest(l);
					AlgebraicExpression *mul = AlgebraicExpression_NewOperation(AL_EXP_MUL);
					AlgebraicExpression_AddChild(mul, A);
					if(i == 0) AlgebraicExpression_AddChild(mul, B);
					else AlgebraicExpression_AddChild(mul, AlgebraicExpression_Clone(B));
					AlgebraicExpression_AddChild(add, mul);
				}
				ASSERT(AlgebraicExpression_ChildCount(l) == 0);
				AlgebraicExpression_Free(l);
			} else {
				// righthand side is addition
				// D * (A + B + C) = (D * A) + (D * B) + (D * C)
				A = l;
				uint child_count = AlgebraicExpression_ChildCount(r);
				for(uint i = 0; i < child_count; i++) {
					B = _AlgebraicExpression_OperationRemoveDest(r);
					AlgebraicExpression *mul = AlgebraicExpression_NewOperation(AL_EXP_MUL);
					if(i == 0) AlgebraicExpression_AddChild(mul, A);
					else AlgebraicExpression_AddChild(mul, AlgebraicExpression_Clone(A));
					AlgebraicExpression_AddChild(mul, B);
					AlgebraicExpression_AddChild(add, mul);
				}
				ASSERT(AlgebraicExpression_ChildCount(r) == 0);
				AlgebraicExpression_Free(r);
			}
			// Free original root and overwrite it with new addition root.
			AlgebraicExpression_Free(*root);
			// Update root.
			*root = add;
			return true;
		}
	}

	// recurse
	uint child_count = AlgebraicExpression_ChildCount(*root);
	for(uint i = 0; i < child_count; i++) {
		if(__AlgebraicExpression_MulOverAdd((*root)->operation.children + i)) return true;
	}
	return false;
}

// Distributive, multiplication over addition:
// A * (B + C) = (A * B) + (A * C)
//
//           (*)
//   (A)             (+)
//            (B)          (C)
//
// Becomes
//
//               (+)
//       (*)                (*)
// (A)        (B)     (A)        (C)
//
// Whenever we encounter a multiplication operation
// where one child is an addition operation and the other child
// is a multiplication operation, we'll replace root multiplication
// operation with an addition operation with two multiplication operations
// one for each child of the original addition operation, as can be seen above.
// we'll want to reuse the left handside of the multiplication.
static void _AlgebraicExpression_MulOverAdd(AlgebraicExpression **root) {
	// As long as the tree changes keep modifying.
	while(__AlgebraicExpression_MulOverAdd(root));
}

/* Collapse multiplication operation under a single multiplication op
 * exp = A * B * C
 * exp can be computed in two ways:
 * 1. (A * B) * C
 * 2. A * (B * C)
 * by flattening the expression both 1 and 2 are represented by the
 * same tree structure. */
static void _AlgebraicExpression_FlattenMultiplications(AlgebraicExpression *root) {
	ASSERT(root);
	uint child_count;

	switch(root->type) {
	case AL_OPERATION:
		switch(root->operation.op) {
		case AL_EXP_ADD:
		case AL_EXP_TRANSPOSE:
			// Keep searching for a multiplication operation.
			child_count = AlgebraicExpression_ChildCount(root);
			for(int i = 0; i < child_count; i++) {
				_AlgebraicExpression_FlattenMultiplications(CHILD_AT(root, i));
			}
			break;

		case AL_EXP_MUL:
			// Root has sub multiplication node(s).
			if(AlgebraicExpression_OperationCount(root, AL_EXP_MUL) > 1) {
				child_count = AlgebraicExpression_OperandCount(root);
				AlgebraicExpression **flat_children = array_new(AlgebraicExpression *, child_count);
				_AlgebraicExpression_CollectOperands(root, &flat_children);
				_AlgebraicExpression_FreeOperation(root);
				root->operation.children = flat_children;
			}

			break;
		default:
			ASSERT("Unknown algebraic operation type" && false);
			break;
		}
	default:
		break;
	}
}

//------------------------------------------------------------------------------
// Transpose pushdown
//------------------------------------------------------------------------------

// Forward declaration.
static void _Pushdown_TransposeExp(AlgebraicExpression *exp);

// Transpose addition.
static void _Pushdown_TransposeAddition
(
	AlgebraicExpression *exp
) {
	// T(A + B) = T(A) + T(B)
	// Transpose children.
	uint child_count = AlgebraicExpression_ChildCount(exp);
	for(uint i = 0; i < child_count; i++) _Pushdown_TransposeExp(exp->operation.children[i]);
}

// Transpose multiplication.
static void _Pushdown_TransposeMultiplication
(
	AlgebraicExpression *exp
) {
	// Swap children, Transpose(A * B) = Transpose(B) * Transpose(A)
	array_reverse(exp->operation.children);
	// Transpose children.
	uint child_count = AlgebraicExpression_ChildCount(exp);
	for(uint i = 0; i < child_count; i++) _Pushdown_TransposeExp(exp->operation.children[i]);
}

// Transpose transpose.
static void _Pushdown_TransposeTranspose
(
	AlgebraicExpression *exp
) {
	// T(T(A)) = A
	// Expecting just a single operand.
	ASSERT(AlgebraicExpression_ChildCount(exp) == 1);
	AlgebraicExpression *only_child = _AlgebraicExpression_OperationRemoveDest(exp);

	// Replace Transpose operation with its child.
	_AlgebraicExpression_InplaceRepurpose(exp, only_child);
}

// Transpose operation.
static void _Pushdown_TransposeOperation
(
	AlgebraicExpression *exp
) {
	switch(exp->operation.op) {
	case AL_EXP_ADD:
		// T(A + B) = T(A) + T(B)
		_Pushdown_TransposeAddition(exp);
		break;
	case AL_EXP_MUL:
		_Pushdown_TransposeMultiplication(exp);
		break;
	case AL_EXP_TRANSPOSE:
		_Pushdown_TransposeTranspose(exp);
		break;
	default:
		ASSERT("Unknown algebraic expression operation" && false);
		break;
	}
}

// Transpose operand.
static void _Pushdown_TransposeOperand
(
	AlgebraicExpression *exp
) {
	// No need to transpose a diagonal matrix.
	if(exp->operand.diagonal) return;

	exp->operand.transpose = !exp->operand.transpose;
}

static void _Pushdown_TransposeExp
(
	AlgebraicExpression *exp
) {
	switch(exp->type) {
	case AL_OPERATION:
		_Pushdown_TransposeOperation(exp);
		break;
	case AL_OPERAND:
		_Pushdown_TransposeOperand(exp);
		break;
	default:
		ASSERT("unknown algebraic expression node type" && false);
		break;
	}
}

/* Push down transpose operations to the point where they are applied to individual operands
 * once this optimization is applied there shouldn't be instances of transpose acting on
 * operation nodes such as multiplication and addition.
 *
 * Consider Exp = Transpose(A + B)
 *
 *           (transpose)
 *               (+)
 *         (A)          (B)
 *
 * PushDownTranspose will transform Exp to: Transpose(A) + Transpose(B)
 *
 *                (+)
 *    (transpose)     (transpose)
 *         (A)            (B)
 *
 * Another example, Exp = Transpose(A * B)
 *
 *           (transpose)
 *               (*)
 *         (A)          (B)
 *
 * Would become Transpose(B) * Transpose(A)
 *
 *                (*)
 *    (transpose)     (transpose)
 *         (B)            (A)
 * */
void AlgebraicExpression_PushDownTranspose(AlgebraicExpression *root) {
	uint i = 0;
	uint child_count = 0;
	AlgebraicExpression *child = NULL;

	switch(root->type) {
	case AL_OPERAND:
		break;  // Nothing to be done.

	case AL_OPERATION:
		switch(root->operation.op) {
		case AL_EXP_ADD:    // Fall through.
		case AL_EXP_MUL:    // Fall through.
		case AL_EXP_POW:    // Fall through.
			child_count = AlgebraicExpression_ChildCount(root);
			for(; i < child_count; i++) {
				AlgebraicExpression *child = root->operation.children[i];
				AlgebraicExpression_PushDownTranspose(child);
			}
			break;

		case AL_EXP_TRANSPOSE:
			child = root->operation.children[0];
			if(child->type == AL_OPERATION) {
				/* Transpose operation:
				 * Transpose(A + B) = Transpose(A) + Transpose(B)
				 * Transpose(A * B) = Transpose(B) * Transpose(A) */
				_Pushdown_TransposeExp(child);
				/* Replace Transpose root with transposed expression.
				 * Remove root only child. */
				_AlgebraicExpression_OperationRemoveDest(root);
				_AlgebraicExpression_InplaceRepurpose(root, child);

				/* It is possible for `root` to contain a transpose subexpression
				 * push it further down. */
				AlgebraicExpression_PushDownTranspose(root);
			} else {
				child->operand.transpose = !child->operand.transpose;
				_AlgebraicExpression_OperationRemoveDest(root);
				_AlgebraicExpression_InplaceRepurpose(root, child);
			}
			break;
		default:
			ASSERT("Unknown operation" && false);
		}
		break;  // Break out of case AL_OPERATION.
	default:
		ASSERT("Unknown algebraic expression node type" && false);
	}
}

//------------------------------------------------------------------------------
// AlgebraicExpression optimizations
//------------------------------------------------------------------------------
void AlgebraicExpression_Optimize
(
	AlgebraicExpression **exp
) {
	ASSERT(exp);

	AlgebraicExpression_PushDownTranspose(*exp);
	_AlgebraicExpression_MulOverAdd(exp);
	_AlgebraicExpression_FlattenMultiplications(*exp);

	// Retrieve all operands now that they are guaranteed to be leaves.
	_AlgebraicExpression_PopulateOperands(*exp, QueryCtx_GetGraphCtx());
}

