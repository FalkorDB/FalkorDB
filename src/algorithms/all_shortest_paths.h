/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "all_paths.h"

int AllShortestPaths_FindMinimumLength(
	AllPathsCtx *ctx,  // shortest path context
	Node *src,         // start traversing from `src`
	Node *dest         // end traversing when `dest` is reached
);

// get next shortest path
// returns NULL when all shortest paths from `src` to `dest`
// had been consumed
Path *AllShortestPaths_NextPath
(
	AllPathsCtx *ctx
);

