/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "filter_functions.h"
#include "../../../util/arr.h"

// evaluate the ids according to filters
bool FilterExpression_Resolve
(
    Graph *g,                   // graph to get max id
    FilterExpression *filters,  // filters to consider
    roaring64_bitmap_t *ids,    // output ids
    Record r                    // record to evaluate filters
){
    size_t node_count = Graph_UncompactedNodeCount(g);
    int count = array_len(filters);
    roaring64_bitmap_add_range_closed(ids, 0, node_count);
    for(int i = 0; i < count; i++) {
        SIValue v = AR_EXP_Evaluate(filters[i].id_exp, r);
        if(SI_TYPE(v) != T_INT64) {
            return false;
        }
        switch(filters[i].op) {
            case OP_LT:    // <
                if(roaring64_bitmap_maximum(ids) >= v.longval) {
                    roaring64_bitmap_remove_range_closed(ids, v.longval, roaring64_bitmap_maximum(ids));
                }
                break;
            case OP_LE:    // <=
                if(roaring64_bitmap_maximum(ids) > v.longval) {
                    roaring64_bitmap_remove_range_closed(ids, v.longval + 1, roaring64_bitmap_maximum(ids));
                }
                break;
            case OP_GT:    // >
                if(roaring64_bitmap_minimum(ids) <= v.longval) {
                    roaring64_bitmap_remove_range_closed(ids, roaring64_bitmap_minimum(ids), v.longval);
                }
                break;
            case OP_GE:    // >=
                if(roaring64_bitmap_minimum(ids) < v.longval) {
                    roaring64_bitmap_remove_range(ids, roaring64_bitmap_minimum(ids), v.longval);
                }
                break;
            case OP_EQUAL:  // =
                if(!roaring64_bitmap_contains(ids, v.longval)) {
                    return false;
                }

                roaring64_bitmap_remove_range_closed(ids, v.longval + 1, roaring64_bitmap_maximum(ids));
                roaring64_bitmap_remove_range(ids, roaring64_bitmap_minimum(ids), v.longval);
                break;
            default:
                ASSERT(false && "operation not supported");
                break;
        }
    }

    roaring64_bitmap_run_optimize(ids);
    return true;
}