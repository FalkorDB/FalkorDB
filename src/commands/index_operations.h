/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../ast/ast.h"
#include "execution_ctx.h"
#include "../graph/graphcontext.h"

// handle index creation/deletion
void IndexOperation_Run
(
	GraphContext *gc,  // graph context
	AST *ast,          // AST
	ExecutionType op   // operation type
);

