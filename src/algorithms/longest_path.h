/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/query_graph.h"
#include "../graph/entities/qg_node.h"

// Determins the longest path length within a tree structured graph.
QGNode *LongestPathTree(const QueryGraph *g, int *level);
// Determins the longest path length within a graph (containing cycles).
QGNode *LongestPathGraph(const QueryGraph *g, int *level);
