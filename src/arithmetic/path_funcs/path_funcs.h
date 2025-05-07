/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// context struct containing traversal data for shortestPath function calls
typedef struct {
	unsigned int minHops;        // minimum number of edges traversed by this path
	unsigned int maxHops;        // maximum number of edges traversed by this path
	int *reltypes;               // relationship type IDs
	const char **reltype_names;  // relationship type names
	unsigned int reltype_count;  // number of traversed relationship types
} ShortestPathCtx;

void Register_PathFuncs();

