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
	raxIterator it;          // iterator for traversing update contexts
	Record *records;         // updated records
	GraphContext *gc;
	rax *update_ctxs;        // entities to update and their expressions
	bool updates_committed;  // true if we've already committed updates and are now in handoff mode.
	dict *node_updates;      // enqueued node updates
	dict *edge_updates;      // enqueued edge updates
} OpUpdate;

OpBase *NewUpdateOp
(
	ExecutionPlan *plan,
	rax *update_exps
);

