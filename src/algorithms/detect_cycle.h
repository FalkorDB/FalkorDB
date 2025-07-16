/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../graph/query_graph.h"

/* Detects if graph contains a cycle. 
 * assuming graph is fully connected. */
bool IsAcyclicGraph(
	const QueryGraph *qg
);
