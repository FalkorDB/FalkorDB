/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "./index.h"
#include "../graph/graphcontext.h"
#include "../constraint/constraint.h"

// Indexer handels asynchronously index population
// in the system there's a single instance of it, initialized on module init
//
// whenever an index is to be created, e.g. issuing: CREATE INDEX ON :C(v)
// or updating an existing index by either adding or removing fields from it
// the index population is going to be performed asynchronously by the indexer
// internal working thread, this alows for the system to remain functional
// during index population time

// initialize indexer
bool Indexer_Init(void);

// populates index
// adds the task for populating the given index to the indexer
void Indexer_PopulateIndex
(
	GraphContext *gc, // graph to operate on
	Schema *s,        // schema containing the idx
	Index idx         // index to populate
);

// drops index asynchronously
// this function simply place the drop request onto a queue
// eventually the indexer working thread will pick it up and drop the index
void Indexer_DropIndex
(
	Index idx,        // index to drop
	GraphContext *gc  // graph context
);

// TODO: Indexer should be incorparated into our scheduler

// enforces constraint
// adds the task for enforcing the given constraint to the indexer
void Indexer_EnforceConstraint
(
	Constraint c,  // constraint to enforce
	GraphContext *gc  // graph context
);

// drops constraint asynchronously
// this function simply place the drop request onto a queue
// eventually the indexer working thread will pick it up and drop the constraint
void Indexer_DropConstraint
(
	Constraint c,     // constraint to drop
	GraphContext *gc  // graph context
);

// stop and free indexer
void Indexer_Stop(void);

