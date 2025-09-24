/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/entities/qg_node.h"
#include <limits.h>

#define BFS_LOWEST_LEVEL INT_MAX    // Return leaf nodes at deepest level.

/* Perform BFS scan from node S,
 * Returns a set of nodes reached at given level.
 * Incase level is set to BFS_LOWEST_LEVEL
 * well scan all the way to the bottom and set level
 * to indicate the level reached. */
QGNode **BFS(
	QGNode *s,  // Node from which BFS scan begins.
	int *level  // Stop scanning at level.
);

