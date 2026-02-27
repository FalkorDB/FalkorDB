/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// if the rdb we are loading is old, then we must recalculate the number of
// edges connecting each pair of nodes
// precondition: relation matricies have been calculated and fully synced
void RdbNormalizeAdjMatrix
(
	const Graph *g  // graph
) ;

