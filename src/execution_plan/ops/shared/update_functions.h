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

// get staged node updates
dict *StagedUpdatesCtx_NodeUpdates
(
    StagedUpdatesCtx *ctx  // staged updates context
);

// get staged edge updates
dict *StagedUpdatesCtx_EdgeUpdates
(
    StagedUpdatesCtx *ctx  // staged updates context
);

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
	GraphEntityType t       // entity type Node/Edge
) ;

// stage a label association with a set of nodes
void StagedUpdatesCtx_LabelNodes
(
	StagedUpdatesCtx *ctx,  // staged update context
	NodeID *node_ids,       // node IDs
	uint32_t node_count,    // number of nodes
	char *label             // label to add
) ;

// stage a label removal from a set of nodes
void StagedUpdatesCtx_UnLabelNodes
(
	StagedUpdatesCtx *ctx,  // staged update context
	NodeID *node_ids,       // node IDs
	uint32_t node_count,    // number of nodes
	char *label             // label to remove
) ;

// returns number of different added labels
uint8_t StagedUpdatesCtx_AddLabelCount
(
    const StagedUpdatesCtx *ctx  // staged update context
);

// returns number of different removed labels
uint8_t StagedUpdatesCtx_RmvLabelCount
(
    const StagedUpdatesCtx *ctx  // staged update context
);

// returns an array of added label vectors
GrB_Vector *StagedUpdatesCtx_AddLabels
(
    StagedUpdatesCtx *ctx  // staged update context
) ;

// returns an array of removed label vectors
GrB_Vector *StagedUpdatesCtx_RmvLabels
(
    StagedUpdatesCtx *ctx  // staged update context
) ;

// free staged update context
void StagedUpdatesCtx_Free
(
	StagedUpdatesCtx **ctx  // staged updates context to free
) ;

// build pending updates in the 'updates' array to match all
// AST-level updates described in the context
// NULL values are allowed in SET clauses but not in MERGE clauses
bool EvalUpdates
(
	GraphContext *gc,                  // graph context
	StagedUpdatesCtx *staged_updates,  // staged updates context
	const Record *recs,                // records
	uint32_t n_recs,                   // number of records
	EntityUpdateDesc **descs,          // update descriptors
	uint32_t n_descs,                  // number of update descriptors
	bool allow_null                    // allow nulls
);

// commit all updates described in the array of pending updates
void CommitUpdates
(
	GraphContext *gc,
	StagedUpdatesCtx *updates
);

// make sure label matrices used in SET n:L and REMOVE n:M
// are of the correct dimensions NxN
void ensureMatrixDim
(
	GraphContext *gc,
	StagedUpdatesCtx *ctx
);

void PendingUpdateCtx_Free
(
	PendingUpdateCtx *ctx
);

