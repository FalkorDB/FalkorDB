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
	GraphEntity *ge;          // entity to be updated
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

// retrieve an entity update context
PendingUpdateCtx *StagedUpdatesCtx_GetEntityUpdateCtx
(
	StagedUpdatesCtx *ctx,  // staged updates
	GraphEntity *e,         // entity
	GraphEntityType t      // entity type Node/Edge
) ;

// free staged update context
void StagedUpdatesCtx_Free
(
	StagedUpdatesCtx **ctx  // staged updates context to free
) ;

// build pending updates in the 'updates' array to match all
// AST-level updates described in the context
// NULL values are allowed in SET clauses but not in MERGE clauses
void EvalUpdates
(
	GraphContext *gc,                 // graph context
	StagedUpdatesCtx *staged_updates  // staged updates context
	const Record *recs,               // records
	uint32_t n_recs,                  // number of records
	EntityUpdateDesc **descs,         // update descriptors
	uint32_t n_descs,                 // number of update descriptors
	bool allow_null                   // allow nulls
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

