/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */


#pragma once

#include "proc_ctx.h"

// Returns all the system functions. 
// This procedure yields the function name description and signature.
ProcedureCtx *Proc_FunctionsCtx();