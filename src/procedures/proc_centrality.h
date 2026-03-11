/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "proc_ctx.h"

// run harmonic closeness centrality on sub graph using HLL BFS propagation
ProcedureCtx *Proc_CentralityCtx(void);
