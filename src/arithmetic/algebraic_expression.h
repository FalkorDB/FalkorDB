/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../graph/graph.h"
#include "../graph/query_graph.h"

static RG_Matrix IDENTITY_MATRIX = (RG_Matrix)0x31032017;  // identity matrix

// Matrix, vector operations
typedef enum {
	AL_EXP_ADD = 1,                 // Matrix addition
	AL_EXP_MUL = (1 << 1),          // Matrix multiplication
	AL_EXP_POW = (1 << 2),          // Matrix raised to a power
	AL_EXP_TRANSPOSE = (1 << 3),    // Matrix transpose
} AL_EXP_OP;

#define AL_EXP_ALL (AL_EXP_ADD | AL_EXP_MUL | AL_EXP_POW | AL_EXP_TRANSPOSE)

// Type of node within an algebraic expression
typedef enum {
	AL_OPERAND = 1,
	AL_OPERATION  = (1 << 1),
} AlgebraicExpressionType;

/* Forward declarations. */
typedef struct AlgebraicExpression AlgebraicExpression;

struct AlgebraicExpression {
	AlgebraicExpressionType type;   // type of node, either an operation or an operand
	union {
		struct {
			bool bfree;                 // should matrix be free
			bool diagonal;              // diagonal matrix
			const char *src;            // alias given to operand's rows
			const char *dest;           // alias given to operand's columns
			const char *edge;           // alias given to operand (edge)
			const char *label;          // label attached to matrix
			RG_Matrix matrix;           // matrix
		} operand;
		struct {
			AL_EXP_OP op;                   // operation: `*`,`+`,`transpose`
			AlgebraicExpression **children; // child nodes
		} operation;
	};
};

//------------------------------------------------------------------------------
// AlgebraicExpression construction.
//------------------------------------------------------------------------------

// Construct algebraic expression form query graph.
AlgebraicExpression **AlgebraicExpression_FromQueryGraph
(
	const QueryGraph *qg    // Query-graph to process
);

//------------------------------------------------------------------------------
// AlgebraicExpression Node creation functions.
//------------------------------------------------------------------------------

// Create a new AlgebraicExpression operation node.
AlgebraicExpression *AlgebraicExpression_NewOperation
(
	AL_EXP_OP op    // Operation to perform.
);

// Create a new AlgebraicExpression operand node.
AlgebraicExpression *AlgebraicExpression_NewOperand
(
	RG_Matrix mat,      // Matrix.
	bool diagonal,      // Is operand a diagonal matrix?
	const char *src,    // Operand row domain (src node).
	const char *dest,   // Operand column domain (destination node).
	const char *edge,   // Operand alias (edge).
	const char *label   // Label attached to matrix.
);

// Clone algebraic expression node.
AlgebraicExpression *AlgebraicExpression_Clone
(
	const AlgebraicExpression *exp  // Expression to clone.
);

//------------------------------------------------------------------------------
// AlgebraicExpression attributes.
//------------------------------------------------------------------------------

// Returns the source entity alias represented by the left-most operand
// row domain
const char *AlgebraicExpression_Src
(
	const AlgebraicExpression *root   // Root of expression.
);

// Returns the destination entity alias represented by the right-most operand
// column domain
const char *AlgebraicExpression_Dest
(
	const AlgebraicExpression *root   // Root of expression.
);

// Returns the first edge alias encountered
// if no edge alias is found NULL is returned
const char *AlgebraicExpression_Edge
(
	const AlgebraicExpression *root   // Root of expression.
);

// returns expression label
// exp must be an operand
const char *AlgebraicExpression_Label
(
	const AlgebraicExpression *exp
);

// Returns the number of child nodes directly under root
uint AlgebraicExpression_ChildCount
(
	const AlgebraicExpression *root   // Root of expression.
);

// Returns the number of operands in expression
uint AlgebraicExpression_OperandCount
(
	const AlgebraicExpression *root   // Root of expression.
);

// Returns the number of operations of given type in expression
uint AlgebraicExpression_OperationCount
(
	const AlgebraicExpression *root,    // Root of expression.
	AL_EXP_OP op_type                   // Type of operation.
);

// Returns true if entire expression is transposed
bool AlgebraicExpression_Transposed
(
	const AlgebraicExpression *root   // Root of expression.
);

// Returns true if expression contains an operation of type `op`
bool AlgebraicExpression_ContainsOp
(
	const AlgebraicExpression *root,    // Root of expression.
	AL_EXP_OP op                        // Operation to look for.
);

// Checks to see if operand at position `operand_idx` is a diagonal matrix
bool AlgebraicExpression_DiagonalOperand
(
	const AlgebraicExpression *root,    // Root of expression.
	uint operand_idx                    // Operand position (LTR, zero based).
);

//------------------------------------------------------------------------------
// AlgebraicExpression modification functions.
//------------------------------------------------------------------------------

// Adds child node to root children list.
void AlgebraicExpression_AddChild
(
	AlgebraicExpression *root,  // Root to attach child to.
	AlgebraicExpression *child  // Child node to attach.
);

// Remove source of algebraic expression from root
AlgebraicExpression *AlgebraicExpression_RemoveSource
(
	AlgebraicExpression **root   // Root from which to remove a child.
);

// Remove destination of algebraic expression from root
AlgebraicExpression *AlgebraicExpression_RemoveDest
(
	AlgebraicExpression **root   // Root from which to remove a child.
);

// Multiply expression to the left by operand
// m * (exp)
void AlgebraicExpression_MultiplyToTheLeft
(
	AlgebraicExpression **root,
	RG_Matrix m
);

// Multiply expression to the right by operand
// (exp) * m
void AlgebraicExpression_MultiplyToTheRight
(
	AlgebraicExpression **root,
	RG_Matrix m
);

// Add expression to the left by operand
// m + (exp)
void AlgebraicExpression_AddToTheLeft
(
	AlgebraicExpression **root,
	RG_Matrix m
);

// Add expression to the right by operand
// (exp) + m
void AlgebraicExpression_AddToTheRight
(
	AlgebraicExpression **root,
	RG_Matrix m
);

// Transpose expression
// By wrapping exp in a transpose root node
void AlgebraicExpression_Transpose
(
	AlgebraicExpression **exp    // Expression to transpose
);

// Evaluate expression tree.
RG_Matrix AlgebraicExpression_Eval
(
	const AlgebraicExpression *exp, // Root node
	RG_Matrix res                   // Result output
);

// locates operand based on row,column domain and edge or label
// sets 'operand' if found otherwise set it to NULL
// sets 'parent' if requested, parent can still be set to NULL
// if 'root' is the seeked operand
bool AlgebraicExpression_LocateOperand
(
	AlgebraicExpression *root,       // Root to search
	AlgebraicExpression **operand,   // [output] set to operand, NULL if missing
	AlgebraicExpression **parent,    // [output] set to operand parent
	const char *row_domain,          // operand row domain
	const char *column_domain,       // operand column domain
	const char *edge,                // operand edge name
	const char *label                // operand label name
);

const AlgebraicExpression *AlgebraicExpression_SrcOperand
(
	const AlgebraicExpression *root   // Root of expression.
);

const AlgebraicExpression *AlgebraicExpression_DestOperand
(
	const AlgebraicExpression *root   // Root of expression.
);

//------------------------------------------------------------------------------
// AlgebraicExpression debugging utilities.
//------------------------------------------------------------------------------

// Create an algebraic expression from string
// e.g. B*T(B+A)
AlgebraicExpression *AlgebraicExpression_FromString
(
	const char *exp,    // String representation of expression.
	rax *matrices       // Map of matrices referred to in expression.
);

// Print a tree structure of algebraic expression to stdout
void AlgebraicExpression_PrintTree
(
	const AlgebraicExpression *exp  // Root node.
);

// Print algebraic expression to stdout
void AlgebraicExpression_Print
(
	const AlgebraicExpression *exp  // Root node.
);

// Return a string representation of expression
char *AlgebraicExpression_ToString
(
	const AlgebraicExpression *exp  // Root node.
);

//------------------------------------------------------------------------------
// AlgebraicExpression optimizations
//------------------------------------------------------------------------------
void AlgebraicExpression_Optimize
(
	AlgebraicExpression **exp   // Expression to optimize.
);

// Push down transpose operations to individual operands.
void AlgebraicExpression_PushDownTranspose
(
	AlgebraicExpression *root   // Expression to modify
);

//------------------------------------------------------------------------------
// AlgebraicExpression free
//------------------------------------------------------------------------------

// Free algebraic expression
void AlgebraicExpression_Free
(
	AlgebraicExpression *root  // Root node.
);
