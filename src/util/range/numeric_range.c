/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "numeric_range.h"
#include "RG.h"
#include "../rmalloc.h"
#include "../../ast/ast_shared.h"
#include <math.h>

NumericRange *NumericRange_New(void) {
	NumericRange *range = rm_malloc(sizeof(NumericRange));
	range->valid = true;
	range->max = INFINITY;
	range->min = -INFINITY;
	range->include_min = false;
	range->include_max = false;
	return range;
}

bool NumericRange_IsValid(const NumericRange *range) {
	if(!range->valid) return false;
	if(range->include_min && range->include_max) {
		// X >= y AND X <= z
		return (range->min <= range->max);
	} else {
		// X >= y AND X < z
		// X > y AND X <= z
		// X > y AND X < z
		return (range->min < range->max);
	}
}

bool NumericRange_ContainsValue(const NumericRange *range, double v) {
	if(!range->valid) return false;

	// Make sure v is <= max.
	if(range->include_max) {
		if(v > range->max) return false;
	} else {
		if(v >= range->max) return false;
	}

	// Make sure v >= min.
	if(range->include_min) {
		if(v < range->min) return false;
	} else {
		if(v <= range->min) return false;
	}

	return true;
}

void NumericRange_TightenRange(NumericRange *range, int op, double v) {
	if(!range->valid) return;

	switch(op) {
	case OP_LT:    // <
		if(range->max >= v) {
			range->include_max = false;
			range->max = v;
		}
		break;
	case OP_LE:    // <=
		if(range->max > v) {
			range->include_max = true;
			range->max = v;
		}
		break;
	case OP_GT:    // >
		if(range->min <= v) {
			range->include_min = false;
			range->min = v;
		}
		break;
	case OP_GE:    // >=
		if(range->min < v) {
			range->include_min = true;
			range->min = v;
		}
		break;
	case OP_EQUAL:  // =
		// Make sure v is within range.
		if(!NumericRange_ContainsValue(range, v)) {
			range->valid = false;
			return;
		}

		range->include_min = true;
		range->include_max = true;
		range->min = v;
		range->max = v;
		break;
	default:
		ASSERT(false && "operation not supported");
		break;
	}

	// See if range is still valid.
	range->valid = NumericRange_IsValid(range);
}

void NumericRange_ToString(const NumericRange *range) {
	ASSERT(range != NULL);
	int offset = 0;
	char buff[1024];

	if(range->include_min) offset += sprintf(buff + offset, "[");
	else offset += sprintf(buff + offset, "(");

	if(range->min == -INFINITY) offset += sprintf(buff + offset, "-inf");
	else offset += sprintf(buff + offset, "%f", range->min);

	offset += sprintf(buff + offset, ",");

	if(range->max == -INFINITY) offset += sprintf(buff + offset, "inf");
	else offset += sprintf(buff + offset, "%f", range->max);

	if(range->include_max) offset += sprintf(buff + offset, "]");
	else offset += sprintf(buff + offset, ")");
	printf("%s\n", buff);
}

void NumericRange_Free(NumericRange *range) {
	ASSERT(range != NULL);
	rm_free(range);
}

