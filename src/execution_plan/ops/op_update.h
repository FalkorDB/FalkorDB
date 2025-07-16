/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "op.h"
#include "../../util/dict.h"
#include "../execution_plan.h"
#include "shared/update_functions.h"
#include "../../resultset/resultset_statistics.h"

typedef struct {
	OpBase op;
	raxIterator it;                 // Iterator for traversing update contexts
	Record *records;                // Updated records
	GraphContext *gc;
	rax *update_ctxs;               // Entities to update and their expressions
	bool updates_committed;         // True if we've already committed updates and are now in handoff mode.
	dict *node_updates;             // Enqueued node updates
	dict *edge_updates;             // Enqueued edge updates
} OpUpdate;

OpBase *NewUpdateOp(const ExecutionPlan *plan, rax *update_exps);
