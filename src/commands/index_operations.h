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

// extract index level configuration (language, stopwords) from options map
// shared by index_operations.c (client-facing CREATE INDEX) and
// effects_apply.c (replica-side EFFECT_CREATE_INDEX reconstruction), which
// must apply the exact same index-level configuration exactly once per
// CREATE INDEX statement
bool IndexOperation_ExtractLevelConfig
(
	char ***stopwords,  // index stopwords
	char **language,    // index language
	SIValue options     // options map
);

