/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

// computes x + y
// returns to true if x+y overflowed
bool safe_add
(
	int x,  // x
	int y,  // y
	int *z  // x+y
);

// computes x * y
// returns to true if x*y overflowed
bool safe_mul
(
	int x,  // x
	int y,  // y
	int *z  // x*y
);

