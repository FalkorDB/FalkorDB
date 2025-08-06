/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "utils.h"
#include "../algebraic_expression.h"
#include "../arithmetic_expression.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../util/rmalloc.h"
#include "../../algorithms/algorithms.h"

//------------------------------------------------------------------------------
// Static internal functions.
//------------------------------------------------------------------------------

// Locate the left most node in `exp`
static AlgebraicExpression *_leftMostNode(AlgebraicExpression *exp) {
	AlgebraicExpression *left_most = exp;
	while(left_most->type == AL_OPERATION &&
		  AlgebraicExpression_ChildCount(left_most) > 0) {
		left_most = FIRST_CHILD(left_most);
	}
	return left_most;
}

// Locate the right most node in `exp`
static AlgebraicExpression *_rightMostNode(AlgebraicExpression *exp) {
	AlgebraicExpression *right_most = exp;
	while(right_most->type == AL_OPERATION &&
			AlgebraicExpression_ChildCount(right_most) > 0) {
		right_most = LAST_CHILD(right_most);
	}
	return right_most;
}

static AlgebraicExpression *_AlgebraicExpression_CloneOperation
(
	const AlgebraicExpression *exp
) {
	AlgebraicExpression *clone =
		AlgebraicExpression_NewOperation(exp->operation.op);

	uint child_count = AlgebraicExpression_ChildCount(exp);
	for(uint i = 0; i < child_count; i++) {
		AlgebraicExpression_AddChild(clone,
				AlgebraicExpression_Clone(exp->operation.children[i]));
	}
	return clone;
}

static AlgebraicExpression *_AlgebraicExpression_CloneOperand
(
	const AlgebraicExpression *exp
) {
	AlgebraicExpression *clone = rm_malloc(sizeof(AlgebraicExpression));
	memcpy(clone, exp, sizeof(AlgebraicExpression));
	return clone;
}

static void _AlgebraicExpression_PropagateOperandRemoval
(
	AlgebraicExpression **root,   // expression's root
	AlgebraicExpression **stack,  // stack of operations leading to operand
	AlgebraicExpression *operand  // operand to remove
) {
	ASSERT(root    != NULL);
	ASSERT(stack   != NULL);
	ASSERT(operand != NULL);

	// expression is just a single operand, set root to NULL
	if(array_len(stack) == 0) {
		*root = NULL;
		return;
	}

	// propagate operand removal upwards
	// when removing A from MUL(A,B) root should become B
	// when removing A from T(T(A)) root should become NULL
	// when removing A from ADD(MUL(T(A),B),C) root should become ADD(B,C)

	AlgebraicExpression *parent      = NULL;
	AlgebraicExpression *current     = operand;
	AlgebraicExpression *replacement = NULL;

	while(array_len(stack) > 0) {
		parent = array_pop(stack);
		_AlgebraicExpression_OperationRemoveChild(parent, current);

		// do not free operand
		if(current != operand) {
			AlgebraicExpression_Free(current);
		}

		AL_EXP_OP op = parent->operation.op;
		// binary operation with a single child, replace operation with child
		// removing A from A+B should become B
		if(op == AL_EXP_ADD || op == AL_EXP_MUL) {
			if(AlgebraicExpression_ChildCount(parent) == 1) {
				// replace operation with only child
				replacement = _AlgebraicExpression_OperationRemoveDest(parent);
				_AlgebraicExpression_InplaceRepurpose(parent, replacement);
			}
			// stop here, no need to propagate further
			break;
		}

		current = parent;
	}

	// handle last parent situation, e.g. removing A from T(T(T(A)))
	if(parent && parent->type == AL_OPERATION) {
		if(AlgebraicExpression_ChildCount(parent) == 0) {
			AlgebraicExpression_Free(parent);
			*root = NULL;
		}
	}
}

// remove leftmost child node from 'root' if 'src' is set to true
// rightmost child node otherwise
static AlgebraicExpression *_AlgebraicExpression_RemoveOperand
(
	AlgebraicExpression **root,  // root from which to remove left/right most child
	bool src                     // remove src operand if set, dest otherwise
) {
	ASSERT(*root);

	bool transpose               = false;
	AlgebraicExpression *ret     = NULL;
	AlgebraicExpression *parent  = NULL;
	AlgebraicExpression *current = *root;
	AlgebraicExpression **stack  = array_new(AlgebraicExpression*, 1);

	// search for operand
	while(current->type == AL_OPERATION) {
		array_append(stack, current);

		switch(current->operation.op) {
			case AL_EXP_TRANSPOSE:
				ASSERT(AlgebraicExpression_ChildCount(current) == 1 && "Transpose operation had invalid number of children");
				transpose = !transpose;
				current   = FIRST_CHILD(current); // transpose has only one child
				break;

			case AL_EXP_ADD:
				// addition order of operands is not effected by transpose
				if(src) current = FIRST_CHILD(current);
				else    current = LAST_CHILD(current);
				break;

			case AL_EXP_MUL:
				// multiplication order of operands depends on transpose
				// | transpose     | src     | get dest |
				// | transpose     | not src | get src  |
				// | not transpose | src     | get src  |
				// | not transpose | not src | get dest |
				if(transpose && src)        current = LAST_CHILD (current);
				else if(transpose  && !src) current = FIRST_CHILD(current);
				else if(!transpose && src)  current = FIRST_CHILD(current);
				else if(!transpose && !src) current = LAST_CHILD (current);

				break;

			default:
				ASSERT("Unknown algebraic expression operation" && false);
		}
	}

	ret = current;
	ASSERT(current->type == AL_OPERAND);

	_AlgebraicExpression_PropagateOperandRemoval(root, stack, current);

	array_free(stack);
	return ret;
}

//------------------------------------------------------------------------------
// AlgebraicExpression Node creation functions.
//------------------------------------------------------------------------------

// Create a new AlgebraicExpression operation node
AlgebraicExpression *AlgebraicExpression_NewOperation
(
	AL_EXP_OP op    // Operation to perform.
) {
	AlgebraicExpression *node = rm_malloc(sizeof(AlgebraicExpression));
	node->type = AL_OPERATION;
	node->operation.op = op;
	node->operation.children = array_new(AlgebraicExpression *, 0);
	return node;
}

// Create a new AlgebraicExpression operand node
AlgebraicExpression *AlgebraicExpression_NewOperand
(
	Delta_Matrix mat,      // Matrix
	bool diagonal,      // Is operand a diagonal matrix?
	const char *src,    // Operand row domain (src node)
	const char *dest,   // Operand column domain (destination node)
	const char *edge,   // Operand alias (edge)
	const char *label   // Label attached to matrix
) {
	AlgebraicExpression *node = rm_malloc(sizeof(AlgebraicExpression));

	node->type             = AL_OPERAND;
	node->operand.src      = src;
	node->operand.dest     = dest;
	node->operand.edge     = edge;
	node->operand.label    = label;
	node->operand.bfree    = false;
	node->operand.diagonal = diagonal;
	node->operand.matrix   = mat;

	return node;
}

// Clone algebraic expression node
AlgebraicExpression *AlgebraicExpression_Clone
(
	const AlgebraicExpression *exp  // Expression to clone
) {
	ASSERT(exp);
	switch(exp->type) {
	case AL_OPERATION:
		return _AlgebraicExpression_CloneOperation(exp);
		break;
	case AL_OPERAND:
		return _AlgebraicExpression_CloneOperand(exp);
		break;
	default:
		ASSERT("Unknown algebraic expression node type" && false);
		break;
	}
	return NULL;
}

//------------------------------------------------------------------------------
// AlgebraicExpression attributes.
//------------------------------------------------------------------------------

// Returns the first edge alias encountered.
// if no edge alias is found NULL is returned
const char *AlgebraicExpression_Edge
(
	const AlgebraicExpression *root   // Root of expression
) {
	ASSERT(root);

	uint child_count = 0;
	const char *edge = NULL;
	switch(root->type) {
	case AL_OPERATION:
		child_count = AlgebraicExpression_ChildCount(root);
		for(uint i = 0; i < child_count; i++) {
			edge = AlgebraicExpression_Edge(CHILD_AT(root, i));
			if(edge) return edge;
		}
		break;
	case AL_OPERAND:
		return root->operand.edge;
	}

	return NULL;
}

const char *AlgebraicExpression_Label
(
  const AlgebraicExpression *exp
) {
	ASSERT(exp       != NULL);
	ASSERT(exp->type == AL_OPERAND);

  return exp->operand.label;
}

// Returns the number of child nodes directly under root
uint AlgebraicExpression_ChildCount
(
	const AlgebraicExpression *root   // Root of expression
) {
	// empty expression
	if(root == NULL) return 0;

	if(root->type == AL_OPERATION) {
		return array_len(root->operation.children);
	} else {
		return 0;
	}
}

// returns the number of operands in expression
uint AlgebraicExpression_OperandCount
(
	const AlgebraicExpression *root
) {
	// empty expression
	if(!root) {
		return 0;
	}

	uint child_count   = 0;
	uint operand_count = 0;

	switch(root->type) {
		case AL_OPERATION:
			child_count = AlgebraicExpression_ChildCount(root);
			for(uint i = 0; i < child_count; i++) {
				operand_count += AlgebraicExpression_OperandCount(CHILD_AT(root, i));
			}
			break;
		case AL_OPERAND:
			operand_count = 1;
			break;
		default:
			ASSERT("Unknown algebraic expression node type" && false);
			break;
	}

	return operand_count;
}

// Returns the number of operations in expression
uint AlgebraicExpression_OperationCount
(
	const AlgebraicExpression *root,
	AL_EXP_OP op_type
) {
	// Empty expression.
	if(!root) return 0;

	uint op_count = 0;
	if(root->type == AL_OPERATION) {
		if(root->operation.op & op_type) op_count = 1;
		uint child_count = AlgebraicExpression_ChildCount(root);
		for(uint i = 0; i < child_count; i++) {
			op_count += AlgebraicExpression_OperationCount(CHILD_AT(root, i), op_type);
		}
	}
	return op_count;
}

// Returns true if entire expression is transposed
bool AlgebraicExpression_Transposed
(
	const AlgebraicExpression *root   // Root of expression.
) {
	// Empty expression.
	if(root == NULL) return false;

	const AlgebraicExpression *n = root;

	// handle directly nested transposes, e.g. T(T(T(X)))
	bool transposed = false;
	while(n->type == AL_OPERATION && n->operation.op == AL_EXP_TRANSPOSE) {
		ASSERT(AlgebraicExpression_ChildCount(n) == 1 && "Transpose operation had invalid number of children");
		transposed = !transposed;
		n = FIRST_CHILD(n);
	}

	// TODO: handle cases such as T(A) + T(B).
	return transposed;
}

// returns true if expression contains operation
bool AlgebraicExpression_ContainsOp
(
	const AlgebraicExpression *root,
	AL_EXP_OP op
) {
	// empty expression
	if(!root) return false;

	if(root->type == AL_OPERATION) {
		if(root->operation.op == op) return true;
		uint child_count = AlgebraicExpression_ChildCount(root);
		for(uint i = 0; i < child_count; i++) {
			if(AlgebraicExpression_ContainsOp(CHILD_AT(root, i), op)) return true;
		}
	}
	return false;
}

// returns true if expression contains matrix 'm'
bool AlgebraicExpression_ContainsMatrix
(
	const AlgebraicExpression *root,  // root of expression
	Delta_Matrix m                    // matrix to look for
) {
	// empty expression
	if(!root) return false;

	uint child_count = 0;
	switch(root->type) {
		case AL_OPERATION:
			child_count = AlgebraicExpression_ChildCount(root);
			for(uint i = 0; i < child_count; i++) {
				if(AlgebraicExpression_ContainsMatrix(CHILD_AT(root, i), m)) {
					return true;
				}
			}

			break;

		case AL_OPERAND:
			return (root->operand.matrix == m);

		default:
			ASSERT("unknown node type" && false);
			break;
	}

	return false;
}

// returns true if operand represents a diagonal matrix
bool AlgebraicExpression_Diagonal
(
	const AlgebraicExpression *operand
) {
	ASSERT(operand != NULL);
	ASSERT(operand->type == AL_OPERAND);

	return operand->operand.diagonal;
}

// Checks to see if operand at position `operand_idx` is a diagonal matrix
bool AlgebraicExpression_DiagonalOperand
(
	const AlgebraicExpression *root,    // Root of expression.
	uint operand_idx                    // Operand position (LTR, zero based).
) {
	// Empty expression.
	if(!root) return false;

	const AlgebraicExpression *op =
		_AlgebraicExpression_GetOperand(root, operand_idx);
	ASSERT(op && op->type == AL_OPERAND);
	return op->operand.diagonal;
}

bool _AlgebraicExpression_LocateOperand
(
	AlgebraicExpression *root,      // Root to search
	AlgebraicExpression *proot,     // Root to search
	AlgebraicExpression **operand,  // [output] set to operand
	AlgebraicExpression **parent,   // [output] set to operand parent
	const char *row_domain,         // operand row domain
	const char *column_domain,      // operand column domain
	const char *edge,               // operand edge name
	const char *label               // operand label name
) {
	ASSERT(!(edge && label));
	if(root == NULL) return false;

	if(root->type == AL_OPERAND) {
		// check row domain
		if(row_domain != NULL && root->operand.src != NULL) {
			if(strcmp(row_domain, root->operand.src) != 0) {
				return false;
			}
		} else if(row_domain != root->operand.src) {
			return false;
		}

		// check column domain
		if(column_domain != NULL && root->operand.dest != NULL) {
			if(strcmp(column_domain, root->operand.dest) != 0) {
				return false;
			}
		} else if(column_domain != root->operand.dest) {
			return false;
		}

		// check edge
		if(edge != NULL && root->operand.edge != NULL) {
			if(strcmp(edge, root->operand.edge) != 0) {
				return false;
			}
		} else if (edge != root->operand.edge) {
			return false;
		} else if(label != NULL && root->operand.label != NULL) {
			// check label
			if(strcmp(label, root->operand.label) != 0) {
				return false;
			}
		}

		// found seeked operand
		*operand = root;
		if(parent != NULL) *parent = proot;
		return true;
	}

	if(root->type == AL_OPERATION) {
		uint child_count = AlgebraicExpression_ChildCount(root);
		for(uint i = 0; i < child_count; i++) {
			AlgebraicExpression *child = CHILD_AT(root, i);
			if(_AlgebraicExpression_LocateOperand(child, root, operand, parent,
						row_domain, column_domain, edge, label)) {
				return true;
			}
		}
	}

	return false;
}

bool AlgebraicExpression_LocateOperand
(
	AlgebraicExpression *root,       // Root to search
	AlgebraicExpression **operand,   // [output] set to operand, NULL if missing
	AlgebraicExpression **parent,    // [output] set to operand parent
	const char *row_domain,          // operand row domain
	const char *column_domain,       // operand column domain
	const char *edge,                // operand edge name
	const char *label                // operand label name
) {
	ASSERT(root != NULL);
	ASSERT(operand != NULL);
	ASSERT(!(edge && label));

	*operand = NULL;
	if(parent) *parent = NULL;

	return _AlgebraicExpression_LocateOperand(root, NULL, operand, parent,
			row_domain, column_domain, edge, label);
}

//------------------------------------------------------------------------------
// AlgebraicExpression modification functions.
//------------------------------------------------------------------------------

// Adds child node to root children list.
void AlgebraicExpression_AddChild
(
	AlgebraicExpression *root,  // Root to attach child to.
	AlgebraicExpression *child  // Child node to attach.
) {
	ASSERT(root && root->type == AL_OPERATION);
	array_append(root->operation.children, child);
}

// Remove leftmost child node from root.
AlgebraicExpression *AlgebraicExpression_RemoveSource
(
	AlgebraicExpression **root  // Root from which to remove left most child.
) {
	bool src = true;
	return _AlgebraicExpression_RemoveOperand(root, src);
}

// Remove right most child node from root.
AlgebraicExpression *AlgebraicExpression_RemoveDest
(
	AlgebraicExpression **root  // Root from which to remove left most child.
) {
	bool src = false;
	return _AlgebraicExpression_RemoveOperand(root, src);
}

// construct path from root to operand
static bool _AlgebraicExpression_PathToOperand
(
	AlgebraicExpression ***path,
	const AlgebraicExpression *operand,
	AlgebraicExpression *root
) {
	// did we reached operand?
	if(root == operand) {
		// yes, add it to path and return true
		array_append(*path, root);
		return true;
	}

	// root is an operation, search for operand in each operation's child node
	if(root->type == AL_OPERATION) {
		uint n = AlgebraicExpression_ChildCount(root);

		for(uint i = 0; i < n; i++) {
			AlgebraicExpression *child = CHILD_AT(root, i);
			if(_AlgebraicExpression_PathToOperand(path, operand, child)) {
				// operand was found, add child to path
				array_append(*path, root);
				return true;
			}
		}
	}

	// root is an operand, not the one we're looking for
	return false;
}

static AlgebraicExpression **AlgebraicExpression_PathToOperand
(
	const AlgebraicExpression *operand,
	AlgebraicExpression *root
) {
	AlgebraicExpression **path = array_new(AlgebraicExpression*, 1);

	bool found = _AlgebraicExpression_PathToOperand(&path, operand, root);
	ASSERT(found == true);

	// path is constructed in reverse order operand -> root
	// reverse the order, root -> operand
	array_reverse(path);

	return path;
}

// remove operand
void AlgebraicExpression_RemoveOperand
(
	AlgebraicExpression **root,   // expression root
	AlgebraicExpression *operand  // operand to remove
) {
	ASSERT(root          != NULL);
	ASSERT(*root         != NULL);
	ASSERT(operand       != NULL);
	ASSERT(operand->type == AL_OPERAND);

	//--------------------------------------------------------------------------
	// build path from root to operand
	//--------------------------------------------------------------------------

	AlgebraicExpression **path = AlgebraicExpression_PathToOperand(operand,
			*root);
	ASSERT(path != NULL);

	// make sure operand is the last element on the path
	uint n = array_len(path);
	ASSERT(path[n-1] == operand);

	//--------------------------------------------------------------------------
	// remove the operand
	//--------------------------------------------------------------------------

	_AlgebraicExpression_PropagateOperandRemoval(root, path, operand);
	array_free(path);
}

// Multiply root to the left with op.
// Updates root
void AlgebraicExpression_MultiplyToTheLeft
(
	AlgebraicExpression **root,
	Delta_Matrix m
) {
	ASSERT(m    != NULL);
	ASSERT(root != NULL);

	AlgebraicExpression *rhs = *root;
	// assuming new operand inherits (src, dest and edge) from
	// from the current left most operand
	AlgebraicExpression *left_most_operand = _leftMostNode(rhs);
	AlgebraicExpression *lhs = AlgebraicExpression_NewOperand(m, false, left_most_operand->operand.src,
															  left_most_operand->operand.dest, NULL, NULL);

	*root = _AlgebraicExpression_MultiplyToTheLeft(lhs, rhs);
}

// Multiply root to the right with op.
// Updates root
void AlgebraicExpression_MultiplyToTheRight
(
	AlgebraicExpression **root,
	Delta_Matrix m
) {
	ASSERT(root && m);
	AlgebraicExpression *lhs = *root;
	/* Assuming new operand inherits (src, dest and edge) from
	 * from the current right most operand. */
	AlgebraicExpression *right_most_operand = _rightMostNode(lhs);
	AlgebraicExpression *rhs = AlgebraicExpression_NewOperand(m, false, right_most_operand->operand.src,
															  right_most_operand->operand.dest, NULL, NULL);

	*root = _AlgebraicExpression_MultiplyToTheRight(lhs, rhs);
}

// Add expression to the left by operand
// A + (exp)
void AlgebraicExpression_AddToTheLeft
(
	AlgebraicExpression **root,
	Delta_Matrix m
) {
	ASSERT(root && m);
	AlgebraicExpression *rhs = *root;
	/* Assuming new operand inherits (src, dest and edge) from
	 * from the current left most operand. */
	AlgebraicExpression *left_most_operand = _leftMostNode(rhs);
	AlgebraicExpression *lhs = AlgebraicExpression_NewOperand(m, false,
			left_most_operand->operand.src, left_most_operand->operand.dest,
			left_most_operand->operand.edge, NULL);

	*root = _AlgebraicExpression_AddToTheLeft(lhs, rhs);
}

// Add expression to the right by operand
// (exp) + A
void AlgebraicExpression_AddToTheRight
(
	AlgebraicExpression **root,
	Delta_Matrix m
) {
	ASSERT(root && m);
	AlgebraicExpression *lhs = *root;
	/* Assuming new operand inherits (src, dest and edge) from
	 * from the current right most operand. */
	AlgebraicExpression *right_most_operand = _rightMostNode(lhs);
	AlgebraicExpression *rhs = AlgebraicExpression_NewOperand(m, false, right_most_operand->operand.src,
															  right_most_operand->operand.dest, right_most_operand->operand.edge, NULL);

	*root = _AlgebraicExpression_AddToTheRight(lhs, rhs);
}

//------------------------------------------------------------------------------
// AlgebraicExpression free
//------------------------------------------------------------------------------

// Free algebraic expression.
void AlgebraicExpression_Free
(
	AlgebraicExpression *root  // Root node.
) {
	ASSERT(root != NULL);
	switch(root->type) {
	case AL_OPERATION:
		_AlgebraicExpression_FreeOperation(root);
		break;
	case AL_OPERAND:
		_AlgebraicExpression_FreeOperand(root);
		break;
	default:
		ASSERT("Unknown algebraic expression node type" && false);
	}
	rm_free(root);
}

