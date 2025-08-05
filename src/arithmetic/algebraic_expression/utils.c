/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "utils.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"
#include "../../configuration/config.h"

// performs inplace re-purposing of an operand into an operation
void _InplaceRepurposeOperandToOperation
(
	AlgebraicExpression *operand,
	AL_EXP_OP op
) {
	ASSERT(operand && operand->type == AL_OPERAND);
	AlgebraicExpression *operation = AlgebraicExpression_NewOperation(op);
	// turn operand into an operation.
	memcpy(operand, operation, sizeof(AlgebraicExpression));

	// Don't free op internals!
	rm_free(operation);
}

// performs inplace re-purposing of an operation into an operand
void _AlgebraicExpression_InplaceRepurpose
(
	AlgebraicExpression *exp,
	AlgebraicExpression *replacement
) {
	ASSERT(exp && replacement && AlgebraicExpression_ChildCount(exp) == 0);
	// free internals
	if(exp->type == AL_OPERATION) {
		_AlgebraicExpression_FreeOperation(exp);
	} else if(exp->type == AL_OPERAND) {
		_AlgebraicExpression_FreeOperand(exp);
	} else {
		ASSERT("Unknown algebraic expression type" && false);
	}

	// replace
	memcpy(exp, replacement, sizeof(AlgebraicExpression));

	// free the memory of the migrated replacement
	rm_free(replacement);
}

void _AlgebraicExpression_OperationRemoveChild
(
	AlgebraicExpression *parent,
	const AlgebraicExpression *child
) {
	ASSERT(child  != NULL);
	ASSERT(parent != NULL);

	if(parent->type != AL_OPERATION) return;

	uint child_count = AlgebraicExpression_ChildCount(parent);
	// no child nodes to remove
	if(child_count == 0) return;

	// search for child in parent
	for(uint i = 0; i < child_count; i++) {
		if(parent->operation.children[i] != child) continue;

		// child found, remove it
		// shift-left following children
		for(uint j = i; j < child_count - 1; j++) {
			parent->operation.children[j] = parent->operation.children[j+1];
		}

		array_pop(parent->operation.children);
		break;
	}
}

// removes the rightmost direct child node of root
AlgebraicExpression *_AlgebraicExpression_OperationRemoveDest
(
	AlgebraicExpression *root  // Root from which to remove a child.
) {
	ASSERT(root);
	if(root->type != AL_OPERATION) return NULL;

	// No child nodes to remove.
	if(AlgebraicExpression_ChildCount(root) == 0) return NULL;

	// Remove rightmost child.
	AlgebraicExpression *child = array_pop(root->operation.children);
	return child;
}

// removes the leftmost direct child node of root
AlgebraicExpression *_AlgebraicExpression_OperationRemoveSource
(
	AlgebraicExpression *root   // Root from which to remove a child.
) {
	ASSERT(root);
	if(root->type != AL_OPERATION) return NULL;

	uint child_count = AlgebraicExpression_ChildCount(root);
	// No child nodes to remove.
	if(child_count == 0) return NULL;

	// Remove leftmost child.
	AlgebraicExpression *child = root->operation.children[0];

	// Shift left by 1.
	for(uint i = 0; i < child_count - 1; i++) {
		root->operation.children[i] = root->operation.children[i + 1];
	}
	array_pop(root->operation.children);

	return child;
}

// multiplies `exp` to the left by `lhs`
// returns new expression root
// `lhs` = (A + B)
// `exp` = Transpose(C)
// returns (A + B) * Transpose(C) where `*` is the new root
AlgebraicExpression *_AlgebraicExpression_MultiplyToTheLeft
(
	AlgebraicExpression *lhs,
	AlgebraicExpression *exp
) {
	ASSERT(lhs && exp);
	AlgebraicExpression *mul = AlgebraicExpression_NewOperation(AL_EXP_MUL);
	AlgebraicExpression_AddChild(mul, lhs);
	AlgebraicExpression_AddChild(mul, exp);
	return mul;
}

// multiplies `exp` to the right by `rhs`
// returns new expression root
// `exp` = Transpose(C)
// `rhs` = (A + B)
// returns Transpose(C) * (A + B) where `*` is the new root
AlgebraicExpression *_AlgebraicExpression_MultiplyToTheRight
(
	AlgebraicExpression *exp,
	AlgebraicExpression *rhs
) {
	ASSERT(exp && rhs);
	AlgebraicExpression *mul = AlgebraicExpression_NewOperation(AL_EXP_MUL);
	AlgebraicExpression_AddChild(mul, exp);
	AlgebraicExpression_AddChild(mul, rhs);
	return mul;
}

// adds `exp` to the left by `lhs`
// returns new expression root
// `lhs` = (A * B)
// `exp` = Transpose(C)
// returns (A * B) + Transpose(C) where `+` is the new root
AlgebraicExpression *_AlgebraicExpression_AddToTheLeft
(
	AlgebraicExpression *lhs,
	AlgebraicExpression *exp
) {
	ASSERT(lhs && exp);
	AlgebraicExpression *add = AlgebraicExpression_NewOperation(AL_EXP_ADD);
	AlgebraicExpression_AddChild(add, lhs);
	AlgebraicExpression_AddChild(add, exp);
	return add;
}

// adds `exp` to the right by `rhs`
// returns new expression root
// `exp` = Transpose(C)
// `rhs` = (A * B)
// returns Transpose(C) + (A * B) where `+` is the new root
AlgebraicExpression *_AlgebraicExpression_AddToTheRight
(
	AlgebraicExpression *exp,
	AlgebraicExpression *rhs
) {
	ASSERT(exp && rhs);
	AlgebraicExpression *add = AlgebraicExpression_NewOperation(AL_EXP_ADD);
	AlgebraicExpression_AddChild(add, exp);
	AlgebraicExpression_AddChild(add, rhs);
	return add;
}

void _AlgebraicExpression_FreeOperation
(
	AlgebraicExpression *node
) {
	ASSERT(node && node->type == AL_OPERATION);

	if(node->operation.children) {
		uint child_count = AlgebraicExpression_ChildCount(node);
		for(uint i = 0; i < child_count; i++) {
			AlgebraicExpression_Free(node->operation.children[i]);
		}
		array_free(node->operation.children);
		node->operation.children = NULL;
	}
}

void _AlgebraicExpression_FreeOperand
(
	AlgebraicExpression *node
) {
	ASSERT(node && node->type == AL_OPERAND);

	if(node->operand.bfree) {
		Delta_Matrix_free(&node->operand.matrix);
	}
}

// locate operand at position `operand_idx` counting from left to right
AlgebraicExpression *__AlgebraicExpression_GetOperand
(
	const AlgebraicExpression *root,  // root of expression
	uint operand_idx,                 // operand position (LTR, zero based)
	uint *current_operand_idx
) {
	// `operand_idx` must be within [0, AlgebraicExpression_OperandCount(root)).
	ASSERT(root);

	uint child_count = 0;
	AlgebraicExpression *operand = NULL;

	switch(root->type) {
		case AL_OPERAND:
			if(operand_idx == *current_operand_idx) {
				return (AlgebraicExpression *)root;
			}
			*current_operand_idx += 1;
			break;
		case AL_OPERATION:
			child_count = AlgebraicExpression_ChildCount(root);
			for(uint i = 0; i < child_count; i++) {
				operand = __AlgebraicExpression_GetOperand(CHILD_AT(root, i),
						operand_idx, current_operand_idx);
				if(operand) {
					return operand;
				}
			}
			break;
		default:
			ASSERT("unknown algebraic expression node type" && false);
	}

	return NULL;
}

AlgebraicExpression *_AlgebraicExpression_GetOperand
(
	const AlgebraicExpression *root,  // root of expression
	uint operand_idx                  // operand position (LTR, zero based)
) {
	uint current_operand_idx = 0;
	return __AlgebraicExpression_GetOperand(root, operand_idx, &current_operand_idx);
}

// populate an operand with a standard matrix
static void _AlgebraicExpression_PopulateOperand
(
	AlgebraicExpression *operand,
	const GraphContext *gc
) {
	// do not update matrix if already set,
	// as algebraic expression test depends on this behavior
	// TODO: Redesign _AlgebraicExpression_FromString to remove this condition
	Graph *g = gc->g;
	Delta_Matrix a = operand->operand.matrix;
	Delta_Matrix z = Graph_GetZeroMatrix(g);
	if(a != NULL && a != z) return;

	Schema *s      = NULL;
	Delta_Matrix m = NULL;
	const char *label = operand->operand.label;

	if(label == NULL) {
		// no label, use THE adjacency matrix
		m = Graph_GetRelationMatrix(g, GRAPH_NO_RELATION, false);
	} else if(operand->operand.diagonal) {
		// diagonal operand refers to label matrix
		s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
		if(s) m = Graph_GetLabelMatrix(g, s->id);
	} else {
		// none diagonal matrix, use relationship matrix
		s = GraphContext_GetSchema(gc, label, SCHEMA_EDGE);
		if(s) m = Graph_GetRelationMatrix(g, s->id, false);
	}

	// m is unset, use zero matrix
	if(m == NULL) m = z;

	// set operand matrix
	operand->operand.matrix = m;
}

// populate a transposed operand with a transposed relationship matrix
// and swap the row/col domains
static void _AlgebraicExpression_PopulateTransposedOperand
(
	AlgebraicExpression *operand,
	const GraphContext *gc
) {
	// swap the row and column domains of the operand
	const char *tmp = operand->operand.dest;
	operand->operand.dest = operand->operand.src;
	operand->operand.src = tmp;

	// diagonal matrices do not need to be transposed
	if(operand->operand.diagonal == true) return;

	// do not update matrix if already set
	// as algebraic expression test depends on this behavior
	// TODO: Redesign _AlgebraicExpression_FromString to remove this condition
	if(operand->operand.matrix != NULL) return;

	Schema *s = NULL;
	Delta_Matrix m = NULL;
	const char *label = operand->operand.label;

	if(label == NULL) {
		m = Graph_GetAdjacencyMatrix(gc->g, true);
	} else {
		s = GraphContext_GetSchema(gc, operand->operand.label, SCHEMA_EDGE);
		if(!s) m = Graph_GetZeroMatrix(gc->g);
		else m = Graph_GetRelationMatrix(gc->g, s->id, true);
	}

	operand->operand.matrix = m;
}

// fetch all operands, replacing transpose operations with transposed operands
// if they are available
void _AlgebraicExpression_PopulateOperands
(
	AlgebraicExpression *root,
	const GraphContext *gc
) {
	uint child_count = 0;

	switch(root->type) {
		case AL_OPERATION:
			child_count = AlgebraicExpression_ChildCount(root);
			if(root->operation.op == AL_EXP_TRANSPOSE) {
				ASSERT(child_count == 1 && "Transpose operation had invalid number of children");
				AlgebraicExpression *child = _AlgebraicExpression_OperationRemoveDest(root);
				// fetch the transposed matrix and update the operand
				_AlgebraicExpression_PopulateTransposedOperand(child, gc);
				// replace this operation with the transposed operand
				_AlgebraicExpression_InplaceRepurpose(root, child);
				break;
			}
			for(uint i = 0; i < child_count; i++) {
				_AlgebraicExpression_PopulateOperands(CHILD_AT(root, i), gc);
			}
			break;
		case AL_OPERAND:
			_AlgebraicExpression_PopulateOperand(root, gc);
			break;
		default:
			ASSERT("Unknown algebraic expression node type" && false);
			break;
	}
}

void _AlgebraicExpression_RemoveRedundentOperands
(
	AlgebraicExpression **exps,
	const QueryGraph *qg
) {
	// remove redundent label(s) matrices
	// MATCH (a:A)-[]->(b:B)-[]->(c:C) RETURN a,b,c
	// will result in 2 algebraic expressions:
	// 1. A * ADJ * B
	// 2. B * ADJ * C
	// as node 'b' is shared between the two expressions
	// its operand(s) can be discarded in the later expression
	// as a general rule for every expression I where a former expression J
	// I > J resolves I's source we should remove I's source label operands

	ASSERT(qg   != NULL);
	ASSERT(exps != NULL);

	int exp_count = array_len(exps);
	if(exp_count < 2) return;

	for(int i = 1; i < exp_count; i++) {
		AlgebraicExpression *exp = exps[i];

		// in case source operand isn't a label matrix continue
		if(!AlgebraicExpression_DiagonalOperand(
					AlgebraicExpression_SrcOperand(exp), 0)) continue;

		const char *src_alias = AlgebraicExpression_Src(exp);
		ASSERT(src_alias != NULL);
		QGNode *src_node = QueryGraph_GetNodeByAlias(qg, src_alias);
		ASSERT(src_node != NULL);

		uint label_count = QGNode_LabelCount(src_node);
		ASSERT(label_count > 0);

		// see if source is resolved by a previous expression
		bool resolved = false;
		for(int j = i-1; j >= 0; j--) {
			AlgebraicExpression *prev_exp = exps[j];
			const char *dest_alias = AlgebraicExpression_Dest(prev_exp);
			if(dest_alias == NULL || strcmp(src_alias, dest_alias)) continue;

			resolved = (AlgebraicExpression_DiagonalOperand(
			AlgebraicExpression_DestOperand(prev_exp), 0));
			if(resolved) break;
		}

		if(!resolved) continue;

		// remove source label matrices
		for(int i = 0; i < label_count; i++) {
			AlgebraicExpression *redundent =
				AlgebraicExpression_RemoveSource(&exp);
			AlgebraicExpression_Free(redundent);
		}

		if(AlgebraicExpression_OperandCount(exp) == 0) {
			// reduced to an empty expression
			// delete expression from list
			array_del(exps, i);
			exp_count--;
			i--;
		}
	}
}

