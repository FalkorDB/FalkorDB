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
		SIType t = SI_TYPE (v) ;

		// only a numeric bound can constrain an integer id range; a non-numeric
		// value (string, list, map, bool, null) equals no id -> empty range
		if (!(t & SI_NUMERIC)) {
			SIValue_Free (v) ;  // free heap-backed values (string/list/map)
			return false ;
		}

		// capture the numeric value, then release v: it is not needed past this
		// point, and freeing it every iteration avoids leaking a heap-backed
		// SIValue per incoming record. numeric SIValues are inline so this is a
		// no-op for them, but it keeps a single, uniform ownership boundary
		int64_t ival = v.longval ;
		double  dval = v.doubleval ;
		SIValue_Free (v) ;

		// resolve the bound to the tightest integer preserving the comparison
		// against integer ids: an integer is used as-is; a floating-point value
		// is rounded toward the range interior per operator (ID(n) > 2.0 ->
		// id >= 3, ID(n) <= 2.5 -> id <= 2), while ID(n) = 2.5 matches no id
		int64_t bound ;
		if (t == T_INT64) {
			bound = ival ;
		} else {  // T_DOUBLE
			double d = dval ;

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

		// node ids are non-negative and BitmapRange_Tighten operates on
		// uint64_t, so a bound at/below zero would underflow (OP_LT computes
		// bound - 1). normalize: a lower bound below zero is vacuous - every id
		// satisfies it - while an upper/equality bound below the first id makes
		// the range empty
		if (bound < 0 || (bound == 0 && op == OP_LT)) {
			if (op == OP_GT || op == OP_GE) {
				continue ;      // vacuous lower bound, nothing to tighten
			}
			return false ;      // impossible upper/equality bound -> empty
		}

		if (!BitmapRange_Tighten ((uint64_t) bound, op, &min, &max)) {
			return false ;
		}
	}

	roaring64_bitmap_add_range_closed (bitmap, min, max) ;
    roaring64_bitmap_run_optimize (bitmap) ;
    return true ;
}

