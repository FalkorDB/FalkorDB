/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "proc_ctx.h"

// Returns all the procedures in the systems. This procedure yields the procedures
// names and/or modes (R/W).
ProcedureCtx *Proc_ProceduresCtx();
