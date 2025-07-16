/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../index/index.h"
#include "../graph/graph.h"
#include "../graph/query_graph.h"
#include "../graph/entities/graph_entity.h"
#include "../graph/entities/attribute_set.h"

// forward declaration of opaque constraint structure
typedef struct _Constraint *Constraint;

// constraint enforcement callback function
typedef bool (*Constraint_EnforcementCB)
(
	const Constraint c,
	const GraphEntity *e,
	char **err_msg
);

typedef void (*Constraint_SetPrivateDataCB)
(
	Constraint c,  // constraint to update
	void *pdata    // private data
);

typedef void* (*Constraint_GetPrivateDataCB)
(
	Constraint c  // constraint to get private data from
);

// different states a constraint can be at
// starting as pending and transitioning to either active or failed
typedef enum ConstraintStatus {
    CT_ACTIVE = 0,
    CT_PENDING,
    CT_FAILED
} ConstraintStatus;

// type of constraint
// we're currently supporting two types of constraints
// 1. unique constraint
// 2. mandatory constraint
typedef enum {
	CT_UNIQUE,
	CT_MANDATORY
} ConstraintType;

// create a new constraint
struct GraphContext;

Constraint Constraint_New
(
	struct GraphContext *gc,
	ConstraintType t,         // type of constraint
	LabelID l,                // label/relation ID
	Attribute_ID *fields,     // enforced fields
	const char **attr_names,  // enforced attribute names
	uint8_t n_fields,         // number of fields
	GraphEntityType et,       // entity type
	const char **err_msg      // error message
);

// enable constraint
void Constraint_Enable
(
	Constraint c  // constraint to enable
);

// disable constraint
void Constraint_Disable
(
	Constraint c  // constraint to disable
);

// returns constraint type
ConstraintType Constraint_GetType
(
	const Constraint c  // constraint to query
);

// returns constraint entity type
GraphEntityType Constraint_GetEntityType
(
	const Constraint c  // constraint to query
);

// returns constraint schema ID
int Constraint_GetSchemaID
(
	const Constraint c  // constraint to query
);

// returns constraint status
ConstraintStatus Constraint_GetStatus
(
	const Constraint c
);

// set constraint status
// status can change from:
// 1. CT_PENDING to CT_ACTIVE
// 2. CT_PENDING to CT_FAILED
void Constraint_SetStatus
(
	Constraint c,            // constraint to update
	ConstraintStatus status  // new status
);

// sets constraint private data
void Constraint_SetPrivateData
(
	Constraint c,  // constraint to update
	void *pdata    // private data
);

// get constraint private data
void *Constraint_GetPrivateData
(
	Constraint c  // constraint from which to get private data
);

// returns a shallow copy of constraint attributes
uint8_t Constraint_GetAttributes
(
	const Constraint c,             // constraint from which to get attributes
	const Attribute_ID **attr_ids,  // array of constraint attribute IDs
	const char ***attr_names        // array of constraint attribute names
);

// checks if constraint enforces attribute
bool Constraint_ContainsAttribute
(
	Constraint c,         // constraint to query
	Attribute_ID attr_id  // enforced attribute
);

// returns number of pending changes
int Constraint_PendingChanges
(
	const Constraint c  // constraint to inquery
);

// increment number of pending changes
void Constraint_IncPendingChanges
(
	Constraint c  // constraint to update
);

// decrement number of pending changes
void Constraint_DecPendingChanges
(
	Constraint c  // constraint to update
);

// replicate constraint to both persistency and replicas
void Constraint_Replicate
(
	RedisModuleCtx *ctx,           // redis module context
	const Constraint c,            // constraint to replicate
	const struct GraphContext *gc  // graph context
);

// tries to enforce constraint on all relevant entities
// sets constraint status to pending
void Constraint_Enforce
(
	Constraint c,            // constraint to enforce
	struct GraphContext *gc  // graph context
);

// enforce constraint on all relevant nodes
void Constraint_EnforceNodes
(
	Constraint c,  // constraint to enforce
	Graph *g       // graph
);

// enforce constraint on all relevant edges
void Constraint_EnforceEdges
(
	Constraint c,  // constraint to enforce
	Graph *g       // graph
);

// enforce constraint on entity
// returns true if entity satisfies the constraint
// false otherwise
bool Constraint_EnforceEntity
(
	Constraint c,          // constraint to enforce
	const GraphEntity *e,  // enforced entity
	char **err_msg         // report error message
);

// free constraint
void Constraint_Free
(
	Constraint *c  // constraint to free
);

