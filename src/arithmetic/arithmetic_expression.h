/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>
#include "rax.h"
#include "./func_desc.h"
#include "../../deps/rax/rax.h"
#include "../execution_plan/record.h"
#include "../graph/entities/graph_entity.h"

// AR_ExpNodeType lists the type of nodes within
// an arithmetic expression tree
typedef enum {
	AR_EXP_UNKNOWN,
	AR_EXP_OP,
	AR_EXP_OPERAND
} AR_ExpNodeType;

// AR_OperandNodeType - Type of an expression tree's leaf node
typedef enum {
	AR_EXP_OP_UNKNOWN,     // should not occur
	AR_EXP_CONSTANT,       // a constant, e.g. 3
	AR_EXP_VARIADIC,       // a variable, e.g. n
	AR_EXP_PARAM,          // a parameter, e.g. $p
	AR_EXP_BORROW_RECORD   // a directive to store the current record
} AR_OperandNodeType;

// success of an evaluation
typedef enum {
	EVAL_OK = 0,
	EVAL_ERR = (1 << 0),
	EVAL_FOUND_PARAM = (1 << 1),
} AR_EXP_Result;

// op represents an operation applied to child args
typedef struct {
	uint8_t child_count;            // number of children
	uint16_t constant_mask;         // bitmask where bit i is set if
									// children[i] is a constant
	SIValue *cached_constants;      // pre-evaluated constants
	AR_FuncDesc *f;                 // operation to perform on children
	void *private_data;             // additional data associated with function
	struct AR_ExpNode **children;   // child nodes
} AR_OpNode;

// OperandNode represents either constant, parameter, or graph entity
typedef struct {
	union {
		SIValue constant;
		const char *param_name;
		struct {
			const char *entity_alias;
			int entity_alias_idx;
		} variadic;
	};
	AR_OperandNodeType type;
} AR_OperandNode;

// AR_ExpNode a node within an arithmetic expression tree,
// This node can take one of two forms:
// 1. OpNode
// 2. OperandNode
typedef struct AR_ExpNode {
	union {
		AR_OperandNode operand;
		AR_OpNode op;
	};
	AR_ExpNodeType type;
	// the string representation of the node, such as the literal string "ID(a) + 5"
	const char *resolved_name;
} AR_ExpNode;

// creates a new Arithmetic expression operation node
AR_ExpNode *AR_EXP_NewOpNode
(
	const char *func_name,
	bool include_internal,
	uint child_count
);

// creates a new Arithmetic expression variable operand node
AR_ExpNode *AR_EXP_NewVariableOperandNode(const char *alias);

// creates a new Arithmetic expression extracting an attribute from an entity
AR_ExpNode *AR_EXP_NewAttributeAccessNode(AR_ExpNode *entity, const char *attr);

// creates a new Arithmetic expression constant operand node
AR_ExpNode *AR_EXP_NewConstOperandNode(SIValue constant);

// creates a new Arithmetic expression parameter operand node
AR_ExpNode *AR_EXP_NewParameterOperandNode(const char *param_name);

// creates a new Arithmetic expression that will resolve to the current Record
AR_ExpNode *AR_EXP_NewRecordNode(void);

// set node private data
void AR_SetPrivateData(AR_ExpNode *node, void *pdata);

// compact tree by evaluating constant expressions
// e.g. MINUS(X) where X is a constant number will be reduced to
// a single node with the value -X
// PLUS(MINUS(A), B) will be reduced to a single constant: B-A
bool AR_EXP_ReduceToScalar
(
	AR_ExpNode *root,    // expression to reduce
	bool reduce_params,  // should reduce params
	SIValue *val         // value representing reduced expression
);

// resolve variables to constants
void AR_EXP_ResolveVariables(AR_ExpNode *root, const Record r);

// evaluate arithmetic expression tree
// this function raise exception
SIValue AR_EXP_Evaluate(AR_ExpNode *root, const Record r);

// evaluate arithmetic exception tree for a number of records
// this function raise exception
// populates res with each evaluated value
void AR_EXP_Evaluate_Batch
(
	SIValue *restrict res,        // [input/output] results
	AR_ExpNode *restrict root,    // root of expression tree to evaluate
	const Record *restrict recs,  // batch of records
	size_t n                      // number of records
);

// evaluate arithmmetic expression tree
// this function will not raise exception in case of error
// use it in arithmetic function for example comprehension function
SIValue AR_EXP_Evaluate_NoThrow(AR_ExpNode *root, const Record r);

// evaluate aggregate functions in expression tree
void AR_EXP_Aggregate(AR_ExpNode *root, const Record r);

// reduce aggregation functions to their scalar values
// and evaluates the expression
SIValue AR_EXP_FinalizeAggregations(AR_ExpNode *root, const Record r);

//------------------------------------------------------------------------------
// Utility functions
//------------------------------------------------------------------------------

// set the ith child of parent
// asserts if idx is out of bounds
// returns previous child at `idx` position
AR_ExpNode *AR_EXP_setChild
(
	AR_ExpNode *parent,  // parent node
	AR_ExpNode *child,   // child to add
	uint idx             // child index
);

// get the ith child of root
// in case root isn't a parent or idx > number of children NULL is returned
AR_ExpNode *AR_EXP_getChild
(
	const AR_ExpNode *root,  // arithmetic expression node
	uint idx                 // child index to return
);

// traverse an expression tree and add all entity aliases to a rax
void AR_EXP_CollectEntities
(
	const AR_ExpNode *root,  // expression root node
	rax *aliases             // collected aliases
);

// collect accessed attribute for a given entity
// e.g. person.first_name + person.last_name
// will populate attributes with both 'first_name' and 'last_name'
// if entity is 'person'
void AR_EXP_CollectAttributes
(
	AR_ExpNode *root,    // expression to collect attributes from
	const char *entity,  // accessed entity
	rax *attributes      // collected attributes
);

// collect each variable operand node expands from root
AR_ExpNode **AR_EXP_CollectVariableOperands
(
	AR_ExpNode *root  // expression root
);

AR_ExpNode **AR_EXP_CollectFunctions
(
	AR_ExpNode *root
);

// collect every aggregation node within expression tree
// returns: dynamically allocated array of AR_ExpNode pointers caller must free
// with array_free()
AR_ExpNode **AR_EXP_CollectAggregations
(
	AR_ExpNode *root  // expression root node
);

// search for an aggregation node within the expression tree
// return 1 if one exists
// please note an expression tree can't contain nested aggregation nodes
bool AR_EXP_ContainsAggregation
(
	const AR_ExpNode *root
);

// constructs string representation of arithmetic expression tree
void AR_EXP_ToString(const AR_ExpNode *root, char **str);

// checks to see if expression contains given function
// root - expression root to traverse
// func - function name to lookup
bool AR_EXP_ContainsFunc(const AR_ExpNode *root, const char *func);

// checks to see if expression contains a variable
bool AR_EXP_ContainsVariadic(const AR_ExpNode *root);

// returns true if an arithmetic expression node is a constant
bool AR_EXP_IsConstant
(
	const AR_ExpNode *exp
);

// returns true if an arithmetic expression node is variadic
bool AR_EXP_IsVariadic(const AR_ExpNode *exp);

// returns true if an arithmetic expression node is a parameter
bool AR_EXP_IsParameter
(
	const AR_ExpNode *exp
);

// returns true if an arithmetic expression node is an operation
bool AR_EXP_IsOperation
(
	const AR_ExpNode *exp
);

// returns true if 'exp' represent attribute extraction
// sets 'attr' to attribute name if provided
bool AR_EXP_IsAttribute(const AR_ExpNode *exp, char **attr);

// returns true if `exp` is an aggregation
bool AR_EXP_IsAggregation
(
	const AR_ExpNode *exp  // expression to inspect
);

// check to see if the function operates on distinct values
bool AR_EXP_PerformsDistinct(AR_ExpNode *exp);

// return type of expression
// e.g. the expression: `1+3` return type is SI_NUMERIC
// e.g. the expression : `ToString(4+3)` return type is T_STRING
SIType AR_EXP_ReturnType(const AR_ExpNode *exp);

// returns true if the arithmetic expression returns
// a boolean value and false otherwise
bool AR_EXP_ReturnsBoolean(const AR_ExpNode *exp);

// get the function name of op node
const char *AR_EXP_GetFuncName(const AR_ExpNode *exp);

// clones given expression
AR_ExpNode *AR_EXP_Clone
(
	const AR_ExpNode *exp
);

// copies the content of `src` into `dest`
void AR_EXP_Overwrite
(
	AR_ExpNode *dest,      // node being overwritten
	const AR_ExpNode *src  // node to overwrite with
);

// compare two expressions
// returns true if `a` and `b` represent the same expression
bool AR_EXP_Equals
(
	const AR_ExpNode *a,  // first expression
	const AR_ExpNode *b   // second expression
) ;

// free arithmetic expression tree
void AR_EXP_Free(AR_ExpNode *root);

