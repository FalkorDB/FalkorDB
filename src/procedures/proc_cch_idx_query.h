/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "proc_ctx.h"

// db.idx.cch.query -- answer a point-to-point shortest path using a CCH path
// index built by db.idx.cch.create. Runs a rank-pruned bidirectional search
// entirely over the index's in-memory hierarchy (no graph traversal), then
// unpacks the resulting shortcut arcs back into the real road edges they stand
// for. Read-only and stateless: many queries run concurrently against the same
// index. Yields pathWeight (total metric weight) and path (the road path).
ProcedureCtx *Proc_CCHIdxQueryCtx(void);
