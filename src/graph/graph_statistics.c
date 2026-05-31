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
	stats->node_count = arr_new(uint64_t, 0);
	stats->edge_count = arr_new(uint64_t, 0);
}

void GraphStatistics_IntroduceRelationship
(
	GraphStatistics *stats
) {
	ASSERT(stats && stats->edge_count);
	arr_append(stats->edge_count, 0);
}

// removes a relationship from graph's statistics
void GraphStatistics_RemoveRelationship
(
	GraphStatistics *stats,  // graph statistics
	RelationID rel_id        // relation id to remove
) {
	ASSERT (stats != NULL) ;
	ASSERT (stats->edge_count != NULL) ;
	ASSERT (rel_id == arr_len (stats->edge_count) - 1) ;

	stats->edge_count = arr_del (stats->edge_count, rel_id) ;
}

void GraphStatistics_IntroduceLabel
(
	GraphStatistics *stats
) {
	ASSERT (stats && stats->node_count) ;
	arr_append (stats->node_count, 0) ;
}

// removes a label from graph's statistics
void GraphStatistics_RemoveLabel
(
	GraphStatistics *stats,  // graph statistics
	LabelID lbl_id           // label id
) {
	ASSERT (stats != NULL) ;
	ASSERT (stats->node_count != NULL) ;
	ASSERT (lbl_id == arr_len (stats->node_count) - 1) ;

	stats->node_count = arr_del (stats->node_count, lbl_id) ;
}

uint64_t GraphStatistics_EdgeCount
(
	const GraphStatistics *stats,
	RelationID r
) {
	ASSERT(stats != NULL);
	ASSERT(r < ((RelationID)arr_len(stats->edge_count)));

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
	ASSERT(l < ((LabelID)arr_len(stats->node_count)));

	// none existing / unknown label id
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
	if(stats->node_count) arr_free(stats->node_count);
	if(stats->edge_count) arr_free(stats->edge_count);
}
