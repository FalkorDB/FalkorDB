/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "graph/entities/qg_node.h"
#include "graph/entities/qg_edge.h"
#include <limits.h>

/* Perform DFS scan from node S,
 * Returns a single path from S to a reachable node at distance level. */
QGEdge **DFS(
	QGNode *s,          // Node from which DFS scan begins.
	int level,          // Stop scanning once reached level.
    bool close_cycle    // Allow DFS scan to close a cycle.
);

