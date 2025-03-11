/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"
#include "cypher-parser.h"

// convert an AST node into an SIValue
void AST_ToSIValue
(
	const cypher_astnode_t *node,  // AST node to convert
	SIValue *v                     // [output]
);

