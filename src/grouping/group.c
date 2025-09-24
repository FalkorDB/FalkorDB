/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <stdio.h>
#include "group.h"
#include "../redismodule.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../execution_plan/ops/op.h"

// creates a new group
Group *Group_New
(
	AR_ExpNode **agg,  // aggregation functions
	uint func_count,   // number of aggregation functions
	Record r           // representative record
) {
	Group *g = rm_malloc(sizeof(Group));

	g->r          = r;
	g->agg        = agg;
	g->func_count = func_count;

	return g;
}

// free group
void Group_Free
(
	Group *g  // group to free
) {
	if(g == NULL) {
		return;
	}

	if(g->r != NULL) {
		OpBase_DeleteRecord(&(g->r));
	}

	for(uint i = 0; i < g->func_count; i++) {
		AR_EXP_Free(g->agg[i]);
	}

	rm_free(g->agg);
	rm_free(g);
}

