/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../arr.h"
#include "bitmap_range.h"
#include "../../query_ctx.h"

// tighten the range
// e.g.
// 3 < n < 10 && 1 < n < 8
// will result in the range: 3 < n < 8
bool BitmapRange_Tighten (
	uint64_t v,       // value
	AST_Operator op,  // <, <=, =, >=, >
    uint64_t *min,    // minimum value
    uint64_t *max     // maximum value
) {
	ASSERT (min != NULL) ;
	ASSERT (max != NULL) ;

	// tighten range acording to operator
	switch (op) {
		case OP_LT:    // <
			if (*max >= v) {
				*max = v - 1;
			}
			break;

		case OP_LE:    // <=
			if (*max > v) {
				*max = v;
			}
			break;

		case OP_GT:    // >
			if (*min <= v) {
				*min = v + 1;
			}
			break;

		case OP_GE:    // >=
			if (*min < v) {
				*min = v;
			}
			break;

		case OP_EQUAL:  // =
			if (v < *min || v > *max) {
				return false;
			}

			*min = v;
			*max = v;
			break;

		default:
			ASSERT (false && "operation not supported");
			break;
	}

	return true;
}

// combine multiple ranges into a single range object
bool BitmapRange_FromRanges (
    const RangeExpression *ranges,  // ranges to tighten
    roaring64_bitmap_t **bitmap,    // tighten range
    Record r,                       // record to evaluate range expressions
	uint64_t min,                   // initial minumum value
	uint64_t max                    // initial maximum value
) {
	ASSERT (min    <= max) ;
	ASSERT (ranges != NULL) ;
	ASSERT (bitmap != NULL && *bitmap != NULL) ;

	// clear range
	roaring64_bitmap_free (*bitmap) ;
	*bitmap = roaring64_bitmap_create () ;

	// evaluate range expressions and tighten
    int n = array_len ((RangeExpression *)ranges) ;

	for (int i = 0; i < n; i++) {
		SIValue v = AR_EXP_Evaluate (ranges[i].exp, r) ;

		// fail if range expression isn't an integer
		if (SI_TYPE(v) != T_INT64) {
			return false ;
		}

		if (!BitmapRange_Tighten (v.longval, ranges[i].op, &min, &max)) {
			return false ;
		}
	}

	roaring64_bitmap_add_range_closed (*bitmap, min, max) ;
    roaring64_bitmap_run_optimize (*bitmap) ;
    return true ;
}

