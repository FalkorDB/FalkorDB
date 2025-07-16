/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../value.h"
#include "../arithmetic/arithmetic_expression.h"

typedef struct {
	SIValue *keys;     // SIValues that form the key associated with group
	AR_ExpNode **agg;  // aggregate functions
	uint key_count;    // number of keys
	uint func_count;   // number of aggregation functions
} Group;

// creates a new group
Group *Group_New
(
	SIValue *keys,     // group keys
	uint key_count,    // number of keys
	AR_ExpNode **agg,  // aggregation functions
	uint func_count    // number of aggregation functions
);

// free group
void Group_Free
(
	Group *g  // group to free
);

