/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../../execution_plan.h"
#include "../../../ast/ast_shared.h"
#include "../../../resultset/resultset_statistics.h"

typedef struct {
	NodeCreateCtx *nodes_to_create;
	EdgeCreateCtx *edges_to_create;

	AttributeSet *node_attributes;
	AttributeSet *edge_attributes;

	int **node_labels;
	Node **created_nodes;
	Edge **created_edges;
} PendingCreations;

// initialize all variables for storing pending creations
void NewPendingCreationsContainer
(
	PendingCreations *pending,
	NodeCreateCtx *nodes,
	EdgeCreateCtx *edges
);

// lock the graph and commit all changes introduced by the operation
void CommitNewEntities
(
	OpBase *op,
	PendingCreations *pending
);

// resolve the properties specified in the query into constant values
void ConvertPropertyMap
(
	GraphContext* gc,
	AttributeSet *attributes,
	Record r,
	PropertyMap *map,
	bool fail_on_null
);

// free all data associated with a completed create operation
void PendingCreationsFree
(
	PendingCreations *pending
);

