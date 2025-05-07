/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "LAGraph.h"
#include "../../value.h"
#include "../../deps/GraphBLAS/Include/GraphBLAS.h"

// context struct containing traversal data for shortestPath function calls
typedef struct {
	uint minHops;                // minimum number of edges traversed by this path
	uint maxHops;                // maximum number of edges traversed by this path
	LAGraph_Graph G;             // LAGraph graph object
	int *reltypes;               // relationship type IDs
	const char **reltype_names;  // relationship type names
	uint reltype_count;          // number of traversed relationship types
} ShortestPathCtx;

void Register_PathFuncs();

