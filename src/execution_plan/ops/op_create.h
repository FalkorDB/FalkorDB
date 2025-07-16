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

/* Creates new entities according to the CREATE clause. */

typedef struct {
	OpBase op;                 // The base operation.
	Record *records;           // Array of Records created by this operation.
	PendingCreations pending;  // Container struct for all graph changes to be committed.
} OpCreate;

OpBase *NewCreateOp(const ExecutionPlan *plan, NodeCreateCtx *nodes, EdgeCreateCtx *edges);
