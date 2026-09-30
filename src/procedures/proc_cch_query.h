/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "proc_ctx.h"

// algo.CCH.query -- point-to-point shortest path over a Customizable
// Contraction Hierarchy previously built by algo.CCH. Runs a rank-aware
// bidirectional Dijkstra over the ROAD + SHORTCUT edges (pruned by the node
// rank property), then unpacks every shortcut on the winning path back into
// the original ROAD edges via each shortcut's stored middle node, yielding the
// real road path and its weight.
ProcedureCtx *Proc_CCHQueryCtx(void);
