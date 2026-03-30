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
	GraphEntity *ge;           // entity to be updated
	AttributeSet attributes;  // attributes to update
} PendingUpdateCtx;

// opaque StagedUpdatesCtx struct
typedef struct StagedUpdatesCtx StagedUpdatesCtx ;

// create a new StagedUpdatesCtx
StagedUpdatesCtx *StagedUpdatesCtx_New (void);

// returns true if there are pending node updates
bool StagedUpdatesCtx_HasNodeUpdates
(
	const StagedUpdatesCtx *ctx  // staged updates context
) ;

// returns true if there are pending edge updates
bool StagedUpdatesCtx_HasEdgeUpdates
(
	const StagedUpdatesCtx *ctx  // staged updates context
) ;

// free staged update context
void StagedUpdatesCtx_Free
(
	StagedUpdatesCtx **ctx  // staged updates context to free
) ;

// build pending updates in the 'updates' array to match all
// AST-level updates described in the context
// NULL values are allowed in SET clauses but not in MERGE clauses
void EvalEntityUpdates
(
	GraphContext *gc,
	StagedUpdatesCtx *staged_updates
	const Record r,
	const EntityUpdateDesc *desc,
	bool allow_null
);

// commit all updates described in the array of pending updates
void CommitUpdates
(
	GraphContext *gc,
	StagedUpdatesCtx *updates,
	EntityType type
);

// make sure label matrices used in SET n:L
// are of the correct dimensions NxN
void ensureMatrixDim
(
	GraphContext *gc,
	rax *blueprints
);

void PendingUpdateCtx_Free
(
	PendingUpdateCtx *ctx
);

