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
#include "../execution_plan.h"
#include "../../ast/ast_shared.h"
#include "shared/create_functions.h"
#include "../../graph/entities/node.h"
#include "../../graph/entities/edge.h"

/* Create new graph entities without introducing any non-unique patterns. */
typedef struct {
	OpBase op;                 // The base operation.
	bool handoff_mode;         // Flag denoting whether the op is in Record creation or handoff mode.
	Record *records;           // Array of Records created by this operation.
	rax *unique_entities;      // A map of each unique pending set of creations.
	XXH64_state_t *hash_state; // Reusable hash state for determining creation uniqueness.
	PendingCreations pending;  // Container struct for all graph changes to be committed.
	GraphContext *gc;
} OpMergeCreate;

OpBase *NewMergeCreateOp(const ExecutionPlan *plan, NodeCreateCtx *nodes, EdgeCreateCtx *edges);

// Commit all pending creations and switch to Record handoff mode.
void MergeCreate_Commit(OpBase *opBase);
