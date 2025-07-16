/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
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
	SIValue *keys,     // group keys
	uint key_count,    // number of keys
	AR_ExpNode **agg,  // aggregation functions
	uint func_count    // number of aggregation functions
) {
	Group *g = rm_malloc(sizeof(Group));

	g->keys       = keys;
	g->agg        = agg;
	g->key_count  = key_count;
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

	if(g->keys != NULL) {
		for(int i = 0; i < g->key_count; i ++) {
			SIValue_Free(g->keys[i]);
		}
		rm_free(g->keys);
	}

	for(uint i = 0; i < g->func_count; i++) {
		AR_EXP_Free(g->agg[i]);
	}

	rm_free(g->agg);
	rm_free(g);
}

