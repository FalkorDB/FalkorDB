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

// creates new entities according to the CREATE clause

typedef struct {
	OpBase op;                 // the base operation
	uint64_t rec_idx;          // emit record index
	Record *records;           // array of Records created by this operation
	PendingCreations pending;  // container struct for all graph changes to be committed
} OpCreate;

OpBase *NewCreateOp
(
	const ExecutionPlan *plan,
	NodeCreateCtx *nodes,
	EdgeCreateCtx *edges
);

