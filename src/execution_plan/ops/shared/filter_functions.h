/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../../ast/ast_shared.h"
#include "../../../graph/graph.h"
#include "../../../util/roaring.h"

typedef struct {
	AST_Operator op;
	AR_ExpNode  *id_exp;
} FilterExpression;

bool FilterExpression_Resolve
(
    Graph *g,
    FilterExpression *filters,
    roaring64_bitmap_t *ids,
    Record r
);
