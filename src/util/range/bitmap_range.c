/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include <math.h>

#include "../arr.h"
#include "bitmap_range.h"
#include "../../query_ctx.h"

// convert a double-valued bound to an equivalent integer bound
// returns true if the converted bound was applied (or is a no-op)
// returns false if the predicate is unsatisfiable for any node id
// (in which case 'min'/'max' may have been left unchanged)
static bool _Tighten_Double (
	double v,         // double value
	AST_Operator op,  // <, <=, =, >=, >
	uint64_t *min,    // [in/out] minimum value
	uint64_t *max     // [in/out] maximum value
) {
	ASSERT (min != NULL) ;
	ASSERT (max != NULL) ;

	// NaN never compares true
	if (isnan (v)) {
		return false ;
	}

	// node ids are non-negative integers in the uint64_t range.
	// 2^64 is one past UINT64_MAX; it is used here as the exclusive upper
	// bound so doubles compared against it stay safely castable to uint64_t.
	const double UINT64_LIMIT_AS_DOUBLE = 18446744073709551616.0 ;  // 2^64

	switch (op) {
		case OP_EQUAL:  // id = v
			// only an exact non-negative integer can match
			if (v < 0.0 || v >= UINT64_LIMIT_AS_DOUBLE || floor (v) != v) {
				return false ;
			}
			return BitmapRange_Tighten ((uint64_t) v, OP_EQUAL, min, max) ;

		case OP_GE:  // id >= v
			if (v <= 0.0) {
				// no constraint, ids are >= 0
				return true ;
			}
			if (v >= UINT64_LIMIT_AS_DOUBLE) {
				// no id can be that large
				return false ;
			}
			return BitmapRange_Tighten ((uint64_t) ceil (v), OP_GE, min, max) ;

		case OP_GT:  // id > v
			if (v < 0.0) {
				return true ;
			}
			if (v >= UINT64_LIMIT_AS_DOUBLE - 1.0) {
				return false ;
			}
			{
				uint64_t iv = (floor (v) == v)
					? (uint64_t) v + 1
					: (uint64_t) ceil (v) ;
				return BitmapRange_Tighten (iv, OP_GE, min, max) ;
			}

		case OP_LE:  // id <= v
			if (v < 0.0) {
				return false ;
			}
			if (v >= UINT64_LIMIT_AS_DOUBLE) {
				return true ;
			}
			return BitmapRange_Tighten ((uint64_t) floor (v), OP_LE, min, max) ;

		case OP_LT:  // id < v
			if (v <= 0.0) {
				return false ;
			}
			if (v >= UINT64_LIMIT_AS_DOUBLE) {
				return true ;
			}
			{
				uint64_t iv = (floor (v) == v)
					? (uint64_t) v - 1
					: (uint64_t) floor (v) ;
				return BitmapRange_Tighten (iv, OP_LE, min, max) ;
			}

		default:
			ASSERT (false && "operation not supported") ;
			return false ;
	}
}

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
    roaring64_bitmap_t *bitmap,     // tighten range
    Record r,                       // record to evaluate range expressions
	uint64_t min,                   // initial minumum value
	uint64_t max                    // initial maximum value
) {
	ASSERT (min    <= max) ;
	ASSERT (ranges != NULL) ;
	ASSERT (bitmap != NULL) ;

	// clear range
	roaring64_bitmap_clear (bitmap) ;

	// evaluate range expressions and tighten
    int n = arr_len ((RangeExpression *)ranges) ;

	for (int i = 0; i < n; i++) {
		SIValue v = AR_EXP_Evaluate (ranges[i].exp, r) ;

		if (SI_TYPE(v) == T_INT64) {
			if (!BitmapRange_Tighten (v.longval, ranges[i].op, &min, &max)) {
				return false ;
			}
		} else if (SI_TYPE(v) == T_DOUBLE) {
			// e.g. WHERE id(n) >= 0.0
			// translate the double bound into an equivalent integer bound
			if (!_Tighten_Double (v.doubleval, ranges[i].op, &min, &max)) {
				return false ;
			}
		} else {
			// fail if range expression isn't numeric
			return false ;
		}
	}

	roaring64_bitmap_add_range_closed (bitmap, min, max) ;
    roaring64_bitmap_run_optimize (bitmap) ;
    return true ;
}

