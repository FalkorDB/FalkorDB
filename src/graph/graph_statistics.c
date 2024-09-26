/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "graph_statistics.h"

// Initialize the node_count and edge_count arrays
void GraphStatistics_init
(
	GraphStatistics *stats
) {
	ASSERT(stats);
	stats->node_count = array_new(uint64_t, 0);
	stats->edge_count = array_new(uint64_t, 0);
}

void GraphStatistics_IntroduceRelationship
(
	GraphStatistics *stats
) {
	ASSERT(stats && stats->edge_count);
	array_append(stats->edge_count, 0);
}

void GraphStatistics_IntroduceLabel
(
	GraphStatistics *stats
) {
	ASSERT(stats && stats->node_count);
	array_append(stats->node_count, 0);
}

uint64_t GraphStatistics_EdgeCount
(
	const GraphStatistics *stats,
	RelationID r
) {
	ASSERT(stats != NULL);
	ASSERT(r < ((RelationID)array_len(stats->edge_count)));

	if(r < 0) {
		return 0;
	}

	return stats->edge_count[r];
}

uint64_t GraphStatistics_NodeCount
(
	const GraphStatistics *stats,
	LabelID l
) {
	ASSERT(stats != NULL);
	ASSERT(l < ((LabelID)array_len(stats->node_count)));

	if(l < 0) {
		return 0;
	}

	return stats->node_count[l];
}

void GraphStatistics_FreeInternals
(
	GraphStatistics *stats
) {
	ASSERT(stats);
	if(stats->node_count) array_free(stats->node_count);
	if(stats->edge_count) array_free(stats->edge_count);
}
