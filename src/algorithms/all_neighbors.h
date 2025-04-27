/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "GraphBLAS.h"
#include "../util/hashmap.h"
#include "../graph/entities/node.h"
#include "../graph/delta_matrix/delta_matrix.h"
#include "../graph/delta_matrix/delta_matrix_iter.h"

// performs iterative DFS from 'src'
// each iteration (call to AllNeighborsCtx_NextNeighbor)
// returns the newly discovered destination node
// it is possible for the same destination node to be returned multiple times
// if it is on multiple different paths from src
// we allow cycles to be closed, but we don't expand once a cycle been closed
// path: (a)->(b)->(a), 'a' will not be expanded again during traversal of this
// current path

typedef struct {
	EntityID src;                   // traverse begin here
	Delta_Matrix M;                 // adjacency matrix
	uint minLen;                    // minimum required depth
	uint maxLen;                    // maximum allowed depth
	int current_level;              // current depth
	bool first_pull;                // first call to Next
	EntityID *visited;              // visited nodes
	Delta_MatrixTupleIter *levels;  // array of neighbors iterator
	struct hashmap *visited_nodes;            // visited nodes
} AllNeighborsCtx;

void AllNeighborsCtx_Reset
(
	AllNeighborsCtx *ctx,  // all neighbors context to reset
	EntityID src,          // source node from which to traverse
	Delta_Matrix M,        // matrix describing connections
	uint minLen,           // minimum traversal depth
	uint maxLen            // maximum traversal depth
);

AllNeighborsCtx *AllNeighborsCtx_New
(
	EntityID src,    // source node from which to traverse
	Delta_Matrix M,  // matrix describing connections
	uint minLen,     // minimum traversal depth
	uint maxLen      // maximum traversal depth
);

// produce next reachable destination node
EntityID AllNeighborsCtx_NextNeighbor
(
	AllNeighborsCtx *ctx
);

void AllNeighborsCtx_Free
(
	AllNeighborsCtx *ctx
);

