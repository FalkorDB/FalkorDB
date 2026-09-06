/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../arithmetic/arithmetic_expression.h"

// iterates a range() function
typedef struct {
	int64_t start;    // inclusive: the start of the range
	int64_t end;      // inclusive: the end of the range
	int64_t current;  // current value in the range
	int64_t step;     // stride legth of range
} RangeIter;

// create a new range iterator from an expression
bool RangeIter_fromRangeExp
(
	RangeIter *iter,       // iterator to write
	const AR_ExpNode *exp  // AR tree to expand
);

// iterator next
bool RangeIter_next
(
	RangeIter *iter,  // iterator to increment
	int64_t *value    // the value before incrementing
);

// set current back to start
void RangeIter_reset
(
	RangeIter *iter  // iterator to reset
);

// the number of next calls until the iterator is exhausted
int64_t RangeIter_len
(
	const RangeIter iter  // iterator to query
);

