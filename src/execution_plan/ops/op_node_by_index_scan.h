/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../graph/graph.h"
#include "../../index/index.h"
#include "shared/scan_functions.h"
#include "redisearch_api.h"

typedef struct {
	OpBase op;
	Graph *g;
	bool rebuild_index_query;           // should we rebuild RediSearch index query for each input record
	RSIndex *idx;                       // index to query
	NodeScanCtx *n;                     // label data of node being scanned
	uint nodeRecIdx;                    // index of the node being scanned in the Record
	RSResultsIterator *iter;            // rediSearch iterator over an index with the appropriate filters
	FT_FilterNode *filter;              // filter from which to compose index query
	FT_FilterNode *unresolved_filters;  // subset of filter, contains filters that couldn't be resolved by index
	Record child_record;                // the Record this op acts on if it is not a tap
} IndexScan;

// creates a new IndexScan operation
OpBase *NewIndexScanOp(const ExecutionPlan *plan, Graph *g, NodeScanCtx *n,
		RSIndex *idx, FT_FilterNode *filter);

