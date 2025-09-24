/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "ast.h"

// ast_visitor is responsible for traversing an ast-node
// using the visitor design pattern performing some operations along the
// traversal (e.g., validation)

typedef struct ast_visitor ast_visitor;

// possible return values of a visiting function
typedef enum {
	VISITOR_BREAK,     // fast-fold
	VISITOR_CONTINUE,  // continue traversing the next sibling, without visiting children
	VISITOR_RECURSE    // visit children as well as next sibling
} VISITOR_STRATEGY;

// visit function signature called for each registered AST node type
typedef VISITOR_STRATEGY (*visit)(const cypher_astnode_t *n, bool start, ast_visitor *visitor);

// ast_visitor_mapping maps between an ast-node-type and a visiting function
typedef visit* ast_visitor_mapping;

// ast_visitor traverses an ast-node as long as its state is valid (not VISITOR_BREAK).
// the ast_visitor holds a context which it can populate and use throughout the traversal,
// and a mapping between ast-node-types to their corresponding visiting functions.
typedef struct ast_visitor {
	void *ctx;                    // context
	ast_visitor_mapping mapping;  // mapping between ast-node-types to visiting functions
} ast_visitor;

// get the context of a visitor
void *AST_Visitor_GetContext
(
	const ast_visitor *visitor  // visitor
);

// visits an ast-node
void AST_Visitor_visit
(
	const cypher_astnode_t *node,  // node to visit
	ast_visitor *visitor           // visitor
);

// registers a function to a visitor
void AST_Visitor_mapping_register
(
	ast_visitor_mapping mapping, // visitor to register the function to
	cypher_astnode_type_t type,  // type of ast-node
	visit cb                     // visit function to register
);
