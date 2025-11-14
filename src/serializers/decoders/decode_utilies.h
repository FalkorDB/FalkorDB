/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../serializers_include.h"

// if the rdb we are loading is old, then we must recalculate the number of
// edges connecting ech pair of nodes
// precondition: relation matricies have been calculated and fully synced
void RdbNormalizeAdjMatrix
(
	const Graph *g  // graph
) ;

