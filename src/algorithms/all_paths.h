/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

/*
 * Finds all paths starting at given source node
 * We're computing one path at a time, this is done
 * to take advantage of scenarios where a query specifies LIMIT.
 * To implement this kind of iterative path finding using DFS
 * we're keeping track after:
 * 1. the last path computed, which we'll try to expand
 * 2. neighboring nodes discovered, each placed within a "level"
 * array containing all nodes discovered at a specific level.
 * */

#pragma once

#include "../datatypes/path/path.h"
#include "../graph/graph.h"
#include "../graph/entities/node.h"
#include "../filter_tree/filter_tree.h"

typedef struct {
	Node node;
	Edge edge;
} LevelConnection;

typedef struct {
	LevelConnection **levels;   // nodes reached at depth i, and edges leading to them
	Path *path;                 // current path
	Graph *g;                   // graph to traverse
	Edge *neighbors;            // reusable buffer of edges along the current path
	RelationID *relationIDs;    // edge type(s) to traverse (owned, may be expanded from GRAPH_NO_RELATION)
	int relationCount;          // length of relationIDs
	Tensor *matrices;           // cached relation matrices, matrices[i] for relationIDs[i]
	bool *multi_edge;           // multi_edge[i] true if matrices[i] contains multi-edges
	GRAPH_EDGE_DIR dir;         // traverse direction
	uint minLen;                // path minimum length
	uint maxLen;                // path max length
	Node *dst;                  // destination node, defaults to NULL in case of general all paths execution
	Record r;                   // record the traversal is being performed upon, only used for edge filtering
	FT_FilterNode *ft;          // filterTree of predicates to be applied to traversed edges
	int edge_idx;               // record index of the edge alias; -1 if edge is not referenced
	bool fetch_edges;           // true when edge attributes must be populated (filter or referenced edge)
	bool shortest_paths;        // only collect shortest paths
	GrB_Vector visited;         // visited nodes in shortest path
} AllPathsCtx;

// Create a new All paths context object.
AllPathsCtx *AllPathsCtx_New(
	Node *src,           // Source node to traverse.
	Node *dst,           // Destination node of the paths
	Graph *g,            // Graph to traverse.
	RelationID *relationIDs,  // Edge type(s) on which we'll traverse.
	int relationCount,   // Length of relationIDs.
	GRAPH_EDGE_DIR dir,  // Traversal direction.
	uint minLen,         // Path length must contain at least minLen + 1 nodes.
	uint maxLen,         // Path length must not exceed maxLen + 1 nodes.
	Record r,            // Record the traversal is being performed upon.
	FT_FilterNode *ft,   // FilterTree of predicates to be applied to traversed edges.
	int edge_idx,        // Record index of the edge alias; -1 if unreferenced.
	bool shortest_paths  // Only collect shortest paths.
);

void addNeighbors
(
	AllPathsCtx *ctx,
	LevelConnection *frontier,
	uint32_t depth,
	GRAPH_EDGE_DIR dir
);

// reset context for a new source node, reusing all allocations
// must not be called when shortest_paths is true
void AllPathsCtx_Reset
(
	AllPathsCtx *ctx,
	Node *src,
	Node *dst,
	Record r
);

// tries to produce a new path from given context
// If no additional path can be computed return NULL
Path *AllPathsCtx_NextPath
(
	AllPathsCtx *ctx
);

// free context object
void AllPathsCtx_Free
(
	AllPathsCtx *ctx
);

