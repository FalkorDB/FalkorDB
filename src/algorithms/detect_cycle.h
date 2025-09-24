/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/query_graph.h"

/* Detects if graph contains a cycle. 
 * assuming graph is fully connected. */
bool IsAcyclicGraph(
	const QueryGraph *qg
);
