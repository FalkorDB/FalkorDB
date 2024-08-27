/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"
#include "../arithmetic/arithmetic_expression.h"

typedef struct {
	AR_ExpNode **agg;  // aggregate functions
	uint func_count;   // number of aggregation functions
	Record r;          // representative record
} Group;

// creates a new group
Group *Group_New
(
	AR_ExpNode **agg,  // aggregation functions
	uint func_count,   // number of aggregation functions
	Record r           // representative record
);

// free group
void Group_Free
(
	Group *g  // group to free
);

