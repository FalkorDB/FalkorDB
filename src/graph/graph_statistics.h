/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>
#include "../util/arr.h"
#include "entities/node.h"
#include "entities/edge.h"

// graph related statistics

typedef struct {
	uint64_t *node_count; // array of node count per label matrix
	uint64_t *edge_count; // array of edge count per relationship matrix
} GraphStatistics;

// initialize the node_count and edge_count arrays
void GraphStatistics_init
(
	GraphStatistics *stats
);

// new relationship is added, resize the edge_count array
void GraphStatistics_IntroduceRelationship
(
	GraphStatistics *stats
);

// new label is added, resize the node_count array
void GraphStatistics_IntroduceLabel
(
	GraphStatistics *stats
);

// increment the edge counter by amount
static inline void GraphStatistics_IncEdgeCount
(
	GraphStatistics *stats,
	RelationID r,
	uint64_t amount
) {
	ASSERT(r < array_len(stats->edge_count));
	stats->edge_count[r] += amount;
}

// decrement the edge counter by amount
static inline void GraphStatistics_DecEdgeCount
(
	GraphStatistics *stats,
	RelationID r,
	uint64_t amount
) {
	ASSERT(r < array_len(stats->edge_count) && stats->edge_count[r] >= amount);
	stats->edge_count[r] -= amount;
}

// increment the node counter by amount
static inline void GraphStatistics_IncNodeCount
(
	GraphStatistics *stats,
	LabelID l,
	uint64_t amount
) {
	ASSERT(l < array_len(stats->node_count));
	stats->node_count[l] += amount;
}

// decrement the node counter by amount
static inline void GraphStatistics_DecNodeCount
(
	GraphStatistics *stats,
	int l,
	uint64_t amount
) {
	ASSERT(l < array_len(stats->node_count) && stats->node_count[l] >= amount);
	stats->node_count[l] -= amount;
}

// retrieves edge count for given relationship type
uint64_t GraphStatistics_EdgeCount
(
	const GraphStatistics *stats,
	RelationID r
);

// retrieves node count for given label
uint64_t GraphStatistics_NodeCount
(
	const GraphStatistics *stats,
	LabelID l
);

// free the internal structures
void GraphStatistics_FreeInternals
(
	GraphStatistics *stats
);
