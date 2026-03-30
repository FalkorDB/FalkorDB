/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../../util/dict.h"
#include "../execution_plan.h"

typedef struct {
	OpBase op;                         // base op
	raxIterator it;                    // iterator for traversing update ctxs
	uint64_t rec_idx;                  // emit record index
	Record *records;                   // updated records
	GraphContext *gc;                  // graph context
	rax *update_ctxs;                  // entities to update and their exps
	StagedUpdatesCtx *staged_updates;  // staged updates
} OpUpdate;

// create a new update operation
OpBase *NewUpdateOp
(
	const ExecutionPlan *plan,  // execution plan
	rax *update_exps            // update expressions
);

