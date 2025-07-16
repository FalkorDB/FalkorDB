/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "unsigned_range.h"
#include "RG.h"
#include "../rmalloc.h"
#include "../../ast/ast_shared.h"
#include <math.h>

UnsignedRange *UnsignedRange_New(void) {
	UnsignedRange *range = rm_malloc(sizeof(UnsignedRange));

	range->min         = 0;
	range->max         = UINT64_MAX;
	range->valid       = true;
	range->include_min = true;
	range->include_max = true;

	return range;
}

UnsignedRange *UnsignedRange_Clone
(
	const UnsignedRange *range
) {
	UnsignedRange *clone = rm_malloc(sizeof(UnsignedRange));

	clone->min         = range->min;
	clone->max         = range->max;
	clone->valid       = range->valid;
	clone->include_min = range->include_min;
	clone->include_max = range->include_max;

	return clone;
}

bool UnsignedRange_IsValid
(
	const UnsignedRange *range
) {
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

bool UnsignedRange_ContainsValue
(
	const UnsignedRange *range,
	uint64_t v
) {
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

void UnsignedRange_TightenRange
(
	UnsignedRange *range,
	int op,
	uint64_t v
) {
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
		if(!UnsignedRange_ContainsValue(range, v)) {
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

	// see if range is still valid
	range->valid = UnsignedRange_IsValid(range);
}

void UnsignedRange_ToString
(
	const UnsignedRange *range
) {
	ASSERT(range != NULL);
	int offset = 0;
	char buff[1024];

	if(range->include_min) offset += sprintf(buff + offset, "[");
	else offset += sprintf(buff + offset, "(");

	offset += sprintf(buff + offset, "%" PRIu64, range->min);

	offset += sprintf(buff + offset, ",");

	if(range->max == UINT64_MAX) offset += sprintf(buff + offset, "UINT64_MAX");
	else offset += sprintf(buff + offset, "%" PRIu64, range->max);

	if(range->include_max) offset += sprintf(buff + offset, "]");
	else offset += sprintf(buff + offset, ")");
	printf("%s\n", buff);
}

void UnsignedRange_Free
(
	UnsignedRange *range
) {
	ASSERT(range != NULL);
	rm_free(range);
}

