/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../execution_plan.h"
#include "../../../ast/ast_shared.h"
#include "../../../resultset/resultset_statistics.h"

// nodes commited one at a time as we reserved the node ids and need to be commited in order
typedef struct {
	NodeCreateCtx *nodes_to_create;  // array of node blueprints
	AttributeSet *node_attributes;   // array of node attribute sets created
	LabelID      **node_labels;      // array of node labels
	Node         **created_nodes;    // array of created nodes
} PendingNodeCreations;

// edges commited to the graph in baches all the edges from the same blueprint
// are commited together
typedef struct {
	EdgeCreateCtx edges_to_create;  // edge blueprints
	AttributeSet *edge_attributes;  // array of edge attribute sets created
	Edge **created_edges;           // array of created edges
} PendingEdgeCreations;

typedef struct {
	PendingNodeCreations nodes;   // pending node creations
	PendingEdgeCreations *edges;  // array of pending edge creations
} PendingCreations;

// initialize all variables for storing pending creations
void NewPendingCreationsContainer
(
	PendingCreations *pending,
	NodeCreateCtx *nodes,
	EdgeCreateCtx *edges
);

void PendingCreations_Reset
(
	PendingCreations *ctx
);

// lock the graph and commit all pending changes
void CommitNewEntities
(
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

