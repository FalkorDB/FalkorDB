/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../datatypes/path/path.h"
#include "../graph/graph.h"
#include "../graph/entities/node.h"
#include "../graph/entities/edge.h"
#include "../graph/tensor/tensor.h"

typedef struct {
	TensorIterator it;
	NodeID src;
	int relnum;
} LevelIt;

typedef struct {
	LevelIt *levels;
	TensorIterator *it_arr;
	const RelationID *rels;
	int rel_count;
	int it_n;
} Graph_dfs_stack;

void dfs_stack_new (
	Graph_dfs_stack *stk,
	const Graph *g,
	const RelationID *rels,
	int n_rels,
	GRAPH_EDGE_DIR dir,
	int cap
);

void dfs_stack_push_neighbors (
	Graph_dfs_stack *stk,
	NodeID n
);

bool dfs_stack_pop (
	Graph_dfs_stack *stk,
	NodeID *frontier,
	Edge *e
);

Path *dfs_stack_to_path (
	const Graph_dfs_stack *stk,
	const Graph *g
);

// Checks if the stack contains the given edge (excludes the last edge)
bool dfs_stack_contains_edge
(
	const Graph_dfs_stack *stk,
	EdgeID eid
);

// Checks if the stack contains the given node (excludes the last node)
bool dfs_stack_contains_node
(
	const Graph_dfs_stack *stk,
	NodeID nid
);

void dfs_stack_clear
(
	Graph_dfs_stack *stk
);

bool dfs_stack_empty
(
	const Graph_dfs_stack *stk
);

void dfs_stack_free (
	Graph_dfs_stack stk
);
