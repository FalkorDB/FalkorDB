/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "proc_ctx.h"

// algo.CCH -- Customizable Contraction Hierarchies preprocessing.
// Builds the CCH (metric-independent order + chordal triangulation +
// customization) for the given metric, then commits the result entirely to the
// graph as SHORTCUT-typed edges (carrying the customized weight) plus a rank
// property per node. No CCH data structure is retained afterwards; a subsequent
// rank-aware bidirectional Dijkstra query reuses the shortcuts + ranks.
ProcedureCtx *Proc_CCHCtx(void);
