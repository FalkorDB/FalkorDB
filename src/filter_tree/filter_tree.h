/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>
#include "../redismodule.h"
#include "../ast/ast_shared.h"
#include "../../deps/rax/rax.h"
#include "../execution_plan/record.h"
#include "../arithmetic/arithmetic_expression.h"

// filter tree evaluation return values
typedef enum {
	FILTER_NULL = -1,  // indicates filter evaluated to NULL
	FILTER_FAIL =  0,  // indicates filter evaluated to FALSE
	FILTER_PASS =  1   // indicates filter evaluated to TRUE
} FT_Result;

// nodes within the filter tree are one of two types
// either a predicate node or a condition node
typedef enum {
	FT_N_EXP,   // Expression node
	FT_N_PRED,  // Predicate node
	FT_N_COND,  // Conditional node
} FT_FilterNodeType;

struct FT_FilterNode;

/* The FT_ExpressionNode represent a leaf node within the filter tree
 * it holds a single boolean arithmetic expression to evaluate
 * e.g. `WHERE false` in which case `false` is the expression. */
typedef struct {
	AR_ExpNode *exp;    /* Boolean expression to evaluate. */
} FT_ExpressionNode;

/* The FT_PredicateNode represents a leaf node within the filter tree
 * it holds an operator: [<. <=, =, <>, >, >=]
 * a left and right hand-side arithmetic expressions
 * which are evaluated and compared to one another using the operator. */
typedef struct {
	AR_ExpNode *lhs;
	AR_ExpNode *rhs;
	AST_Operator op;	/* Can validly be an operation (<, <=, =, =>, >, <>, maybe NOT). */
} FT_PredicateNode;

/* The FT_ConditionNode is a top level node in the filter tree
 * it holds a conditional operator: [OR, AND]
 * a left and right hand-side filter nodes which can be further extended */
typedef struct {
	struct FT_FilterNode *left;
	struct FT_FilterNode *right;
	AST_Operator op;	/* Can validly be OR, AND (and later, XOR and maybe NOT) */
} FT_ConditionNode;

/* All nodes within the filter tree are of type FT_FilterNode. */
struct FT_FilterNode {
	union {
		FT_ExpressionNode exp;
		FT_PredicateNode pred;
		FT_ConditionNode cond;
	};
	FT_FilterNodeType t;	/* Determines the actual type of this node. */
};

typedef struct FT_FilterNode FT_FilterNode;

// determines if node is a predicate node
int IsNodePredicate
(
	const FT_FilterNode *node
);

// appends a left hand-side node to root
FT_FilterNode *FilterTree_AppendLeftChild
(
	FT_FilterNode *root,
	FT_FilterNode *child
);

// appends a right hand-side node to root
FT_FilterNode *FilterTree_AppendRightChild
(
	FT_FilterNode *root,
	FT_FilterNode *child
);

// creates a new expression node
FT_FilterNode *FilterTree_CreateExpressionFilter
(
	AR_ExpNode *exp
);

// creates a new predicate node
FT_FilterNode *FilterTree_CreatePredicateFilter
(
	AST_Operator op,
	AR_ExpNode *lhs,
	AR_ExpNode *rhs
);

// creates a new condition node
FT_FilterNode *FilterTree_CreateConditionFilter
(
	AST_Operator op
);

// runs val through the filter tree
FT_Result FilterTree_applyFilters
(
	const FT_FilterNode *root,
	const Record r
);

// extract every modified record ID mentioned in the tree
// without duplications
rax *FilterTree_CollectModified
(
	const FT_FilterNode *root
);

// collect filtered attribute for a given entity
// e.g. person.first_name = 'a' OR person.last_name = 'b'
// will collect both 'first_name' and 'last_name'
// if filtered_entity is 'person'
rax *FilterTree_CollectAttributes
(
	const FT_FilterNode *root,  // filter tree
	const char *entity          // filtered entity
);

// check if any filtered variable is an alias
bool FilterTree_FiltersAlias
(
	const FT_FilterNode *root,
	const cypher_astnode_t *ast
);

// checks to see if tree contains given operation
bool FilterTree_containsOp
(
	const FT_FilterNode *root,
	AST_Operator op
);

// checks to see if tree contains given function
// root - tree root to traverse
// func - function name to lookup
// node - point to tree node in which func was located, null if func isn't located
bool FilterTree_ContainsFunc
(
	const FT_FilterNode *root,
	const char *func,
	FT_FilterNode **node
);

// prints tree
void FilterTree_Print
(
	const FT_FilterNode *root
);

// break filter tree into sub filter trees as follows:
// sub trees under an OR operator are returned,
// sub trees under an AND operator are broken down to the smallest
// components possible following the two rules above
// sub trees reffer to the original tree subsections
const FT_FilterNode **FilterTree_SubTrees
(
	const FT_FilterNode *root
);

// combines filters usign AND conditions
FT_FilterNode *FilterTree_Combine
(
	const FT_FilterNode **filters,
	uint count
);

// verifies tree structure
// a condition or predicate node can't be childless
bool FilterTree_Valid
(
	const FT_FilterNode *root
);

// remove NOT nodes by applying DeMorgan laws
void FilterTree_DeMorgan
(
	FT_FilterNode **root
);

// try to compress a given filter tree
bool FilterTree_Compact
(
	FT_FilterNode *root
);

// resolve variables to constants
void FilterTree_ResolveVariables
(
	FT_FilterNode *root,
	const Record r
);

// clones tree
FT_FilterNode *FilterTree_Clone
(
	const FT_FilterNode *root
);

// prints tree
void FilterTree_Print
(
	const FT_FilterNode *root
);

// free tree
void FilterTree_Free
(
	FT_FilterNode *root
);

