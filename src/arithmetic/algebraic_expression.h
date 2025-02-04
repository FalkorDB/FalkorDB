/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/graph.h"
#include "../graph/query_graph.h"

static Delta_Matrix IDENTITY_MATRIX = (Delta_Matrix)0x31032017;  // identity matrix

// matrix, vector operations
typedef enum {
	AL_EXP_ADD = 1,               // matrix addition
	AL_EXP_MUL = (1 << 1),        // matrix multiplication
	AL_EXP_POW = (1 << 2),        // matrix raised to a power
	AL_EXP_TRANSPOSE = (1 << 3),  // matrix transpose
} AL_EXP_OP;

#define AL_EXP_ALL (AL_EXP_ADD | AL_EXP_MUL | AL_EXP_POW | AL_EXP_TRANSPOSE)

// type of node within an algebraic expression
typedef enum {
	AL_OPERAND = 1,
	AL_OPERATION  = (1 << 1),
} AlgebraicExpressionType;

// forward declarations
typedef struct AlgebraicExpression AlgebraicExpression;

struct AlgebraicExpression {
	AlgebraicExpressionType type;            // type of node, either an operation or an operand
	union {
		struct {
			bool bfree;                      // should matrix be free
			bool diagonal;                   // diagonal matrix
			const char *src;                 // alias given to operand's rows
			const char *dest;                // alias given to operand's columns
			const char *edge;                // alias given to operand (edge)
			const char *label;               // label attached to matrix
			Delta_Matrix matrix;             // matrix
		} operand;
		struct {
			AL_EXP_OP op;                    // operation: `*`,`+`,`transpose`
			AlgebraicExpression **children;  // child nodes
		} operation;
	};
};

//------------------------------------------------------------------------------
// AlgebraicExpression construction
//------------------------------------------------------------------------------

// construct algebraic expression form query graph
AlgebraicExpression **AlgebraicExpression_FromQueryGraph
(
	const QueryGraph *qg    // query-graph to process
);

//------------------------------------------------------------------------------
// AlgebraicExpression Node creation functions
//------------------------------------------------------------------------------

// create a new AlgebraicExpression operation node
AlgebraicExpression *AlgebraicExpression_NewOperation
(
	AL_EXP_OP op  // operation to perform
);

// create a new AlgebraicExpression operand node
AlgebraicExpression *AlgebraicExpression_NewOperand
(
	Delta_Matrix mat,  // matrix
	bool diagonal,     // is operand a diagonal matrix?
	const char *src,   // operand row domain (src node)
	const char *dest,  // operand column domain (destination node)
	const char *edge,  // operand alias (edge)
	const char *label  // label attached to matrix
);

// clone algebraic expression node
AlgebraicExpression *AlgebraicExpression_Clone
(
	const AlgebraicExpression *exp  // expression to clone
);

//------------------------------------------------------------------------------
// AlgebraicExpression attributes.
//------------------------------------------------------------------------------

// returns the source entity alias represented by the left-most operand
// row domain
const char *AlgebraicExpression_Src
(
	const AlgebraicExpression *root   // root of expression
);

// returns the destination entity alias represented by the right-most operand
// column domain
const char *AlgebraicExpression_Dest
(
	const AlgebraicExpression *root   // root of expression
);

// returns the first edge alias encountered
// if no edge alias is found NULL is returned
const char *AlgebraicExpression_Edge
(
	const AlgebraicExpression *root  // root of expression
);

// returns expression label
// exp must be an operand
const char *AlgebraicExpression_Label
(
	const AlgebraicExpression *exp
);

// returns the number of child nodes directly under root
uint AlgebraicExpression_ChildCount
(
	const AlgebraicExpression *root  // root of expression
);

// returns the number of operands in expression
uint AlgebraicExpression_OperandCount
(
	const AlgebraicExpression *root  // root of expression
);

// returns the number of operations of given type in expression
uint AlgebraicExpression_OperationCount
(
	const AlgebraicExpression *root,  // root of expression
	AL_EXP_OP op_type                 // type of operation
);

// returns true if entire expression is transposed
bool AlgebraicExpression_Transposed
(
	const AlgebraicExpression *root  // root of expression
);

// returns true if expression contains an operation of type `op`
bool AlgebraicExpression_ContainsOp
(
	const AlgebraicExpression *root,  // root of expression
	AL_EXP_OP op                      // operation to look for
);

// returns true if operand represents a diagonal matrix
bool AlgebraicExpression_Diagonal
(
	const AlgebraicExpression *operand
);

// checks to see if operand at position `operand_idx` is a diagonal matrix
bool AlgebraicExpression_DiagonalOperand
(
	const AlgebraicExpression *root,  // root of expression
	uint operand_idx                  // operand position (LTR, zero based)
);

//------------------------------------------------------------------------------
// AlgebraicExpression modification functions.
//------------------------------------------------------------------------------

// adds child node to root children list
void AlgebraicExpression_AddChild
(
	AlgebraicExpression *root,  // root to attach child to
	AlgebraicExpression *child  // child node to attach
);

// remove source of algebraic expression from root
AlgebraicExpression *AlgebraicExpression_RemoveSource
(
	AlgebraicExpression **root  // root from which to remove a child
);

// remove destination of algebraic expression from root
AlgebraicExpression *AlgebraicExpression_RemoveDest
(
	AlgebraicExpression **root  // root from which to remove a child
);

// remove operand
void AlgebraicExpression_RemoveOperand
(
	AlgebraicExpression **root,   // expression root
	AlgebraicExpression *operand  // operand to remove
);

// multiply expression to the left by operand
// m * (exp)
void AlgebraicExpression_MultiplyToTheLeft
(
	AlgebraicExpression **root,
	Delta_Matrix m
);

// multiply expression to the right by operand
// (exp) * m
void AlgebraicExpression_MultiplyToTheRight
(
	AlgebraicExpression **root,
	Delta_Matrix m
);

// add expression to the left by operand
// m + (exp)
void AlgebraicExpression_AddToTheLeft
(
	AlgebraicExpression **root,
	Delta_Matrix m
);

// add expression to the right by operand
// (exp) + m
void AlgebraicExpression_AddToTheRight
(
	AlgebraicExpression **root,
	Delta_Matrix m
);

// transpose expression
// By wrapping exp in a transpose root node
void AlgebraicExpression_Transpose
(
	AlgebraicExpression **exp  // expression to transpose
);

// evaluate expression tree
Delta_Matrix AlgebraicExpression_Eval
(
	const AlgebraicExpression *exp, // root node
	Delta_Matrix res                // result output
);

//------------------------------------------------------------------------------
// utils
//------------------------------------------------------------------------------

// locates operand based on row, column domain and edge or label
// sets 'operand' if found otherwise set it to NULL
// sets 'parent' if requested, parent can still be set to NULL
// if 'root' is the seeked operand
bool AlgebraicExpression_LocateOperand
(
	AlgebraicExpression *root,      // root to search
	AlgebraicExpression **operand,  // [output] set to operand, NULL if missing
	AlgebraicExpression **parent,   // [output] set to operand parent
	const char *row_domain,         // operand row domain
	const char *column_domain,      // operand column domain
	const char *edge,               // operand edge name
	const char *label               // operand label name
);

const AlgebraicExpression *AlgebraicExpression_SrcOperand
(
	const AlgebraicExpression *root  // root of expression
);

const AlgebraicExpression *AlgebraicExpression_DestOperand
(
	const AlgebraicExpression *root  // root of expression
);

// collect operands originating at root
// the operands are collected in order from the leftmost to the rightmost
// such that the leftmost will be stored in the retuned array at position 0
// and the rightmost will be stored as the last element of the array
AlgebraicExpression **AlgebraicExpression_CollectOperandsInOrder
(
	const AlgebraicExpression *root,  // root from which to collect operands
	uint *n                           // [output] number of operands collected
);

//------------------------------------------------------------------------------
// AlgebraicExpression debugging utilities.
//------------------------------------------------------------------------------

// create an algebraic expression from string
// e.g. B*T(B+A)
AlgebraicExpression *AlgebraicExpression_FromString
(
	const char *exp,  // string representation of expression
	rax *matrices     // map of matrices referred to in expression
);

// print a tree structure of algebraic expression to stdout
void AlgebraicExpression_PrintTree
(
	const AlgebraicExpression *exp  // root node
);

// print algebraic expression to stdout
void AlgebraicExpression_Print
(
	const AlgebraicExpression *exp  // root node
);

// return a string representation of expression
char *AlgebraicExpression_ToString
(
	const AlgebraicExpression *exp  // root node
);

//------------------------------------------------------------------------------
// AlgebraicExpression optimizations
//------------------------------------------------------------------------------

void AlgebraicExpression_Optimize
(
	AlgebraicExpression **exp  // expression to optimize
);

// push down transpose operations to individual operands
void AlgebraicExpression_PushDownTranspose
(
	AlgebraicExpression *root  // expression to modify
);

//------------------------------------------------------------------------------
// AlgebraicExpression free
//------------------------------------------------------------------------------

// free algebraic expression
void AlgebraicExpression_Free
(
	AlgebraicExpression *root  // root node
);

