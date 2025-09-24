/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "ast_visitor.h"

// build the global mapping from ast-node-type to visiting functions
bool AST_ValidationsMappingInit(void);

// default visit function
VISITOR_STRATEGY _default_visit
(
	const cypher_astnode_t *n,  // ast-node
	bool start,                 // first traversal
	ast_visitor *visitor        // visitor
);
