/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../arr.h"
#include "bitmap_range.h"
#include "../../query_ctx.h"

#include <math.h>

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
		AST_Operator op = ranges[i].op ;

		// the range is over integer node ids, so only a numeric bound can
		// constrain it; a non-numeric value (string, list, map, bool, null)
		// equals no id and yields an empty range
		if (!(SI_TYPE(v) & SI_NUMERIC)) {
			return false ;
		}

		// resolve the bound to the tightest integer that preserves the
		// comparison's meaning against integer ids: an integer is used as-is,
		// a floating-point value is rounded toward the range interior per the
		// operator, e.g. ID(n) > 2.0 -> id >= 3, ID(n) <= 2.5 -> id <= 2,
		// while ID(n) = 2.5 matches no id
		int64_t bound ;
		if (SI_TYPE(v) == T_INT64) {
			bound = v.longval ;
		} else {  // T_DOUBLE
			double d = v.doubleval ;

			// NaN compares false against every id -> empty range
			if (isnan(d)) {
				return false ;
			}

			// clamp to a window that converts exactly to int64 (node ids never
			// exceed 2^53); a bound outside it saturates and is then clamped
			// against [min, max] by BitmapRange_Tighten
			const double LIM = 9007199254740992.0 ;  // 2^53
			if (d >  LIM) d =  LIM ;
			if (d < -LIM) d = -LIM ;

			switch (op) {
				case OP_LT:  bound = (int64_t) ceil  (d) ; break ;  // id <  d
				case OP_GE:  bound = (int64_t) ceil  (d) ; break ;  // id >= d
				case OP_LE:  bound = (int64_t) floor (d) ; break ;  // id <= d
				case OP_GT:  bound = (int64_t) floor (d) ; break ;  // id >  d
				case OP_EQUAL:
					// no integer equals a fractional value -> empty range
					if (floor (d) != d) {
						return false ;
					}
					bound = (int64_t) d ;
					break ;
				default:
					return false ;
			}
		}

		if (!BitmapRange_Tighten (bound, op, &min, &max)) {
			return false ;
		}
	}

	roaring64_bitmap_add_range_closed (bitmap, min, max) ;
    roaring64_bitmap_run_optimize (bitmap) ;
    return true ;
}

