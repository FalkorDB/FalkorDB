/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#ifndef BULKINSERT_H
#define BULKINSERT_H

#include "../redismodule.h"
#include "../graph/graph.h"
#include "../graph/graphcontext.h"

#define BULK_OK 1
#define BULK_FAIL 0

/*
 * Bulk insert performs fast insertion of large amount of data,
 * it's an alternative to Cypher's CREATE query, one should prefer using
 * bulk insert over CREATE queries when constructing a fairly large
 * (thousands of entities) new graph. */

/* Parse bulk insert format and inserts new entities */
int BulkInsert(
	RedisModuleCtx *ctx,        // Redis thread-safe context.
	GraphContext *gc,           // GraphContext hosting schemas and Graph.
	RedisModuleString **argv,   // Arguments passed to bulk insert command.
	int argc,                   // Number of elements in argv.
	uint node_count,            // Number of nodes to be created.
	uint edge_count             // Number of edges to be created.
);

#endif

