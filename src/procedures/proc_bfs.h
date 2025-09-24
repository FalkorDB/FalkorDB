/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "proc_ctx.h"
#include "../arithmetic/arithmetic_expression.h"

// Perform BFS from a single source node.
ProcedureCtx *Proc_BFS_Ctx();

