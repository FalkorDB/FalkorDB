/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../../ast/ast_shared.h"
#include "../../../graph/graph.h"
#include "../../../util/roaring.h"

// Filter expression used in NodeByIdSeek and Node By Label And ID Seek
typedef struct {
	AST_Operator op;
	AR_ExpNode  *id_exp;
} FilterExpression;

// evaluate the ids according to filters
bool FilterExpression_Resolve
(
    Graph *g,                   // graph to get max id
    FilterExpression *filters,  // filters to consider
    roaring64_bitmap_t *ids,    // output ids
    Record r                    // record to evaluate filters
);
