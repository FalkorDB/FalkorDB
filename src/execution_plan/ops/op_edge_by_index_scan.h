/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../graph/graph.h"
#include "../../index/index.h"

typedef struct {
	OpBase op;
	Graph *g;
	bool rebuild_index_query;           // should we rebuild RediSearch index query for each input record
	Index idx;                          // index to query
	QGEdge *edge;                       // edge scanned
	int edgeRecIdx;                     // record index of source node
	int srcRecIdx;                      // record index of destination node
	int destRecIdx;                     // record index of edge
	bool srcAware;                      // src node already resolved
	bool destAware;                     // dest node already resolved
	RSResultsIterator *iter;            // iterator over an index
	FT_FilterNode *filter;              // index query
	AR_ExpNode *current_src_node_id;    // current source node id
	AR_ExpNode *current_dest_node_id;   // current destination node id
	FT_FilterNode *unresolved_filters;  // subset of filter, contains filters that couldn't be resolved by index
	Record child_record;                // input record in case op ins't a tap
} OpEdgeIndexScan;

// creates a new OpEdgeIndexScan operation
OpBase *NewEdgeIndexScanOp
(
	const ExecutionPlan *plan,
	Graph *g,
	QGEdge *e,
	Index idx,
	FT_FilterNode *filter
);

