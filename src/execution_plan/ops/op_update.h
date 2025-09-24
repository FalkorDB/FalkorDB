/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../../util/dict.h"
#include "../execution_plan.h"
#include "shared/update_functions.h"
#include "../../resultset/resultset_statistics.h"

typedef struct {
	OpBase op;
	raxIterator it;      // iterator for traversing update contexts
	uint64_t rec_idx;    // emit record index
	Record *records;     // updated records
	GraphContext *gc;    // graph context
	rax *update_ctxs;    // entities to update and their expressions
	dict *node_updates;  // enqueued node updates
	dict *edge_updates;  // enqueued edge updates
} OpUpdate;

OpBase *NewUpdateOp
(
	const ExecutionPlan *plan,
	rax *update_exps
);

