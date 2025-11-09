/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../ast/ast_shared.h"
#include "shared/create_functions.h"
#include "../../graph/entities/node.h"
#include "../../graph/entities/edge.h"

// create new graph entities without introducing any non-unique patterns
typedef struct {
	OpBase op;                  // the base operation
	bool handoff_mode;          // flag denoting whether the op is in Record creation or handoff mode
	Record *records;            // array of Records created by this operation
	rax *unique_entities;       // a map of each unique pending set of creations
	XXH64_state_t *hash_state;  // reusable hash state for determining creation uniqueness
	PendingCreations pending;   // container struct for all graph changes to be committed
	GraphContext *gc;
} OpMergeCreate;

OpBase *NewMergeCreateOp
(
	const ExecutionPlan *plan,
	NodeCreateCtx *nodes,
	EdgeCreateCtx *edges
);

// commit accumulated changes and switch to Record handoff mode
void MergeCreate_Commit
(
	OpBase *opBase
);

