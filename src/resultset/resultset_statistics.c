/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "../query_ctx.h"
#include "resultset_statistics.h"
#include <string.h>

bool ResultSetStat_IndicateModification
(
	const ResultSetStatistics *stats
) {
	ASSERT(stats != NULL);

	return (
			stats->labels_added          |
			stats->nodes_created         |
			stats->nodes_deleted         |
			stats->properties_set        |
			stats->labels_removed        |
			stats->indices_deleted       |
			stats->indices_created       |
			stats->properties_removed    |
			stats->relationships_created |
			stats->relationships_deleted
		);
}

// initialize resultset statistics
void ResultSetStat_init
(
	ResultSetStatistics *stats  // resultset statistics to initialize
) {
	ASSERT(stats != NULL);

	memset(stats, 0, sizeof(ResultSetStatistics));
}

// emit resultset statistics
void ResultSetStat_emit
(
	RedisModuleCtx *ctx,              // redis module context
	const ResultSetStatistics *stats  // statistics to emit
) {
	int buflen;
	char buff[512] = {0};
	size_t resultset_size = 2; // execution time, cached

	// compute required space for resultset statistics
	if(stats->index_creation)            resultset_size++;
	if(stats->index_deletion)            resultset_size++;
	if(stats->constraint_creation)       resultset_size++;
	if(stats->constraint_deletion)       resultset_size++;
	if(stats->labels_added          > 0) resultset_size++;
	if(stats->nodes_created         > 0) resultset_size++;
	if(stats->nodes_deleted         > 0) resultset_size++;
	if(stats->labels_removed        > 0) resultset_size++;
	if(stats->properties_set        > 0) resultset_size++;
	if(stats->properties_removed    > 0) resultset_size++;
	if(stats->relationships_deleted > 0) resultset_size++;
	if(stats->relationships_created > 0) resultset_size++;

	RedisModule_ReplyWithArray(ctx, resultset_size);

	if(stats->labels_added > 0) {
		buflen = sprintf(buff, "Labels added: %d", stats->labels_added);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->labels_removed > 0) {
		buflen = sprintf(buff, "Labels removed: %d", stats->labels_removed);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->nodes_created > 0) {
		buflen = sprintf(buff, "Nodes created: %d", stats->nodes_created);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->properties_set > 0) {
		buflen = sprintf(buff, "Properties set: %d", stats->properties_set);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->properties_removed > 0) {
		buflen = sprintf(buff, "Properties removed: %d", stats->properties_removed);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->relationships_created > 0) {
		buflen = sprintf(buff, "Relationships created: %d", stats->relationships_created);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->nodes_deleted > 0) {
		buflen = sprintf(buff, "Nodes deleted: %d", stats->nodes_deleted);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->relationships_deleted > 0) {
		buflen = sprintf(buff, "Relationships deleted: %d", stats->relationships_deleted);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->index_creation) {
		buflen = sprintf(buff, "Indices created: %d", stats->indices_created);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->index_deletion) {
		buflen = sprintf(buff, "Indices deleted: %d", stats->indices_deleted);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->constraint_creation) {
		buflen = sprintf(buff, "Constraints created: %d", stats->constraints_created);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	if(stats->constraint_deletion) {
		buflen = sprintf(buff, "Constraints deleted: %d", stats->constraints_deleted);
		RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);
	}

	buflen = sprintf(buff, "Cached execution: %d", stats->cached ? 1 : 0);
	RedisModule_ReplyWithStringBuffer(ctx, (const char *)buff, buflen);

	// emit query execution time
	double t = QueryCtx_GetRuntime();
	buflen = sprintf(buff, "Query internal execution time: %.6f milliseconds", t);
	RedisModule_ReplyWithStringBuffer(ctx, buff, buflen);
}

void ResultSetStat_Clear
(
	ResultSetStatistics *stats
) {
	ASSERT(stats != NULL);

	stats->index_creation = false;
	stats->index_deletion = false;
	stats->constraint_creation = false;
	stats->constraint_deletion = false;
	stats->labels_added          = 0;
	stats->nodes_deleted         = 0;
	stats->nodes_created         = 0;
	stats->properties_set        = 0;
	stats->labels_removed        = 0;
	stats->indices_created       = 0;
	stats->indices_deleted       = 0;
	stats->constraints_created   = 0;
	stats->constraints_deleted   = 0;
	stats->properties_removed    = 0;
	stats->relationships_created = 0;
	stats->relationships_deleted = 0;
}

