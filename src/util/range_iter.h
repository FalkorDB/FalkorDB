/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "../arithmetic/arithmetic_expression.h"

typedef struct {
	bool active;
	int64_t start;
	int64_t end;
	int64_t current;
	int64_t step;
	bool depleted;
} RangeIter;

// create a new range iterator from an expression
bool RangeIter_fromRangeExp
(
	RangeIter *iter,
	const AR_ExpNode *exp
);

// iterator next
bool RangeIter_next
(
	RangeIter *iter,  // iterator to increment
	int64_t *value    // the value before incrementing
);

void RangeIter_reset
(
	RangeIter *iter  // iterator to reset
);

void RangeIter_free
(
	RangeIter *iter // iterator to free
);
