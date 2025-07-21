/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "GraphBLAS.h"

int LG_BreadthFirstSearch_SSGrB  // push-pull BFS, or push-only if AT = NULL
(
	GrB_Vector    *level,     // v(i) is the BFS level of node i in the graph
	GrB_Vector    *parent,    // pi(i) = p+1 if p is the parent of node i.
	GrB_Matrix    A,          // input graph, treated as if boolean in semiring
	GrB_Index     src,        // starting node of the BFS
	GrB_Index     *dest,      // [optional] stop traversing upon reaching dest
	GrB_Index     max_level   // optional limit of # levels to search
);
