/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "graph.h"

// memory usage report struct
typedef struct {
	size_t lbl_matrices_sz;    // label matrices memory usage
	size_t rel_matrices_sz;    // relation matrices memory usage
	size_t node_storage_sz;    // node memory usage
	size_t edge_storage_sz;    // edge memory usage
	size_t total_graph_sz_mb;  // total size in MB
	size_t unlabeled_node_sz;  // unlabeled nodes memory usage
	size_t *node_by_label_sz;  // node memory usage by label
	size_t *edge_by_type_sz;   // edge memory usage by type
	size_t indices_sz;         // indices memory usage
} MemoryUsageResult;

// get graph's memory usage
void Graph_memoryUsage
(
	const Graph *g,            // graph
	MemoryUsageResult *result  // [output] memory usage
);

