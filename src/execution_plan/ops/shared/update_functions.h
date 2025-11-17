/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../execution_plan.h"
#include "../../../util/dict.h"

// context representing a single update to perform on an entity
typedef struct {
	GraphEntity *ge;             // entity to be updated
	AttributeSet attributes;     // attributes to update
	const char **add_labels;     // labels to add to the node
	const char **remove_labels;  // labels to remove from the node
} PendingUpdateCtx;

// commit all updates described in the array of pending updates
void CommitUpdates
(
	GraphContext *gc,
	dict *updates,
	EntityType type
);

// build pending updates in the 'updates' array to match all
// AST-level updates described in the context
// NULL values are allowed in SET clauses but not in MERGE clauses
void EvalEntityUpdates
(
	GraphContext *gc,
	dict *node_updates,
	dict *edge_updates,
	const Record r,
	const EntityUpdateEvalCtx *ctx,
	bool allow_null
);

void PendingUpdateCtx_Free
(
	PendingUpdateCtx *ctx
);

