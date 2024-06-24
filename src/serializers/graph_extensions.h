/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/graph.h"

// sets a node in the graph
void Serializer_Graph_SetNode
(
	Graph *g,               // graph to add node to
	NodeID id,              // node ID
	LabelID *labels,        // node labels
	uint label_count,       // label count
	Node *n                 // pointer to node
);

// sets graph's node labels matrix
void Serializer_Graph_SetNodeLabels
(
	Graph *g
);

// set a given edge in the graph
void Serializer_Graph_SetEdge
(
	Graph *g,               // graph to add edge to
	bool multi_edge,        // true if graph supports multi-edge
	EdgeID edge_id,         // edge ID
	NodeID src,             // edge source
	NodeID dest,            // edge destination
	int r,                  // edge relationship-type
	Edge *e                 // pointer to edge
);

// marks a node ID as deleted
void Serializer_Graph_MarkNodeDeleted
(
	Graph *g,               // graph from which to mark node as deleted
	NodeID ID               // node ID
);

// marks a edge ID as deleted
void Serializer_Graph_MarkEdgeDeleted
(
	Graph *g,               // graph from which to mark edge as deleted
	EdgeID ID               // edge ID
);

// returns the graph deleted nodes list
uint64_t *Serializer_Graph_GetDeletedNodesList
(
	Graph *g
);

// returns the graph deleted nodes list
uint64_t *Serializer_Graph_GetDeletedEdgesList
(
	Graph *g
);

