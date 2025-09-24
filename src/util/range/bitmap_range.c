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
    roaring64_bitmap_t *range,  // tighten range
	uint64_t v,                 // value
	AST_Operator op             // <, <=, =, >=, >
) {
	// tighten range acording to operator
	switch(op) {
		case OP_LT:    // <
			if(roaring64_bitmap_maximum(range) >= v) {
				roaring64_bitmap_remove_range_closed(range, v,
						roaring64_bitmap_maximum(range));
			}
			break;

		case OP_LE:    // <=
			if(roaring64_bitmap_maximum(range) > v) {
				roaring64_bitmap_remove_range_closed(range, v + 1,
						roaring64_bitmap_maximum(range));
			}
			break;

		case OP_GT:    // >
			if(roaring64_bitmap_minimum(range) <= v) {
				roaring64_bitmap_remove_range_closed(range,
						roaring64_bitmap_minimum(range), v);
			}
			break;

		case OP_GE:    // >=
			if(roaring64_bitmap_minimum(range) < v) {
				roaring64_bitmap_remove_range(range,
						roaring64_bitmap_minimum(range), v);
			}
			break;

		case OP_EQUAL:  // =
			if(!roaring64_bitmap_contains(range, v)) {
				return false;
			}

			roaring64_bitmap_remove_range_closed(range, v + 1,
					roaring64_bitmap_maximum(range));
			roaring64_bitmap_remove_range(range,
					roaring64_bitmap_minimum(range), v);
			break;

		default:
			ASSERT(false && "operation not supported");
			break;
	}

	return true;
}

// combine multiple ranges into a single range object
bool BitmapRange_FromRanges (
    const RangeExpression *ranges,  // ranges to tighten
    roaring64_bitmap_t *range,      // tighten range
    Record r,                       // record to evaluate range expressions
	uint64_t min,                   // initial minumum value
	uint64_t max                    // initial maximum value
) {
	ASSERT(min <= max);
	ASSERT(range  != NULL);
	ASSERT(ranges != NULL);

	// start with [0..node_count) range
    // size_t node_count = Graph_UncompactedNodeCount(QueryCtx_GetGraph());
    roaring64_bitmap_add_range_closed(range, min, max);

	// evaluate range expressions and tighten
    int n = array_len((RangeExpression *)ranges);
    for(int i = 0; i < n; i++) {
        SIValue v = AR_EXP_Evaluate(ranges[i].exp, r);

		// fail if range expression isn't an integer
        if(SI_TYPE(v) != T_INT64) {
            return false;
        }

		if(!BitmapRange_Tighten(range, v.longval, ranges[i].op)) {
			return false;
		}
    }

    roaring64_bitmap_run_optimize(range);
    return true;
}

