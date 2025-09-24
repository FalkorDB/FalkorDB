/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../algebraic_expression.h"
#include "../../graph/graphcontext.h"

#define CHILDREN(node) (node)->operation.children
#define CHILD_AT(node, idx) CHILDREN(node)[idx]
#define FIRST_CHILD(node) CHILD_AT(node, 0)
#define LAST_CHILD(node) CHILD_AT(node, AlgebraicExpression_ChildCount(node) - 1)

// Performs inplace re-purposing of an operand into an operation
void _InplaceRepurposeOperandToOperation
(
	AlgebraicExpression *operand,   // Operand to repurpose.
	AL_EXP_OP op                    // Operation to turn operand into.
);

// Performs inplace re-purposing of expression,
// exp mustn't contain any children.
void _AlgebraicExpression_InplaceRepurpose
(
	AlgebraicExpression *exp,           // Expression to repurpose.
	AlgebraicExpression *replacement    // Replacement expression taking over `exp`.
);

// Remove 'child' from 'parent'
void _AlgebraicExpression_OperationRemoveChild
(
	AlgebraicExpression *parent,      // parent to remove child from
	const AlgebraicExpression *child  // child to remove
);

// Removes the rightmost direct child node of root.
AlgebraicExpression *_AlgebraicExpression_OperationRemoveDest
(
	AlgebraicExpression *root  // Root from which to remove a child.
);

// Removes the leftmost direct child node of root.
AlgebraicExpression *_AlgebraicExpression_OperationRemoveSource
(
	AlgebraicExpression *root   // Root from which to remove a child.
);

/* Multiplies `exp` to the left by `lhs`.
 * Returns new expression root.
 * `lhs` = (A + B)
 * `exp` = Transpose(C)
 * Returns (A + B) * Transpose(C) where `*` is the new root. */
AlgebraicExpression *_AlgebraicExpression_MultiplyToTheLeft
(
	AlgebraicExpression *lhs,
	AlgebraicExpression *exp
);

/* Multiplies `exp` to the right by `rhs`.
 * Returns new expression root.
 * `exp` = Transpose(C)
 * `rhs` = (A + B)
 * Returns Transpose(C) * (A + B) where `*` is the new root. */
AlgebraicExpression *_AlgebraicExpression_MultiplyToTheRight
(
	AlgebraicExpression *exp,
	AlgebraicExpression *rhs
);

/* Adds `exp` to the left by `lhs`.
 * Returns new expression root.
 * `lhs` = (A * B)
 * `exp` = Transpose(C)
 * Returns (A * B) + Transpose(C) where `+` is the new root. */
AlgebraicExpression *_AlgebraicExpression_AddToTheLeft
(
	AlgebraicExpression *lhs,
	AlgebraicExpression *exp
);

/* Adds `exp` to the right by `rhs`.
 * Returns new expression root.
 * `exp` = Transpose(C)
 * `rhs` = (A * B)
 * Returns Transpose(C) + (A * B) where `+` is the new root. */
AlgebraicExpression *_AlgebraicExpression_AddToTheRight
(
	AlgebraicExpression *exp,
	AlgebraicExpression *rhs
);

// Free Operation internals.
void _AlgebraicExpression_FreeOperation
(
	AlgebraicExpression *node   // Operation node to free internals.
);

// Free operand internals.
void _AlgebraicExpression_FreeOperand
(
	AlgebraicExpression *node   // Operand node to free internals.
);

// Locate operand at position `operand_idx` counting from left to right.
AlgebraicExpression *_AlgebraicExpression_GetOperand
(
	const AlgebraicExpression *root,    // Root of expression.
	uint operand_idx                    // Operand position (LTR, zero based).
);

// Resolves all missing operands, replacing transpose operations with
// transposed operands if they are available.
void _AlgebraicExpression_PopulateOperands
(
	AlgebraicExpression *exp,   // Expression to resolve operands for.
	const GraphContext *gc      // Graph context.
);

// given a list of algebraic expressions remove redundent label operands
// a redundent label operand is one which is already resolved by a previoius
// expression
void _AlgebraicExpression_RemoveRedundentOperands
(
	AlgebraicExpression **exps,
	const QueryGraph *qg
);

