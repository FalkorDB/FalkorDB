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
	LevelConnection **levels;   // Nodes reached at depth i, and edges leading to them.
	Path *path;                 // Current path.
	Graph *g;                   // Graph to traverse.
	Edge *neighbors;            // Reusable buffer of edges along the current path.
	int *relationIDs;           // edge type(s) to traverse.
	int relationCount;          // length of relationIDs.
	GRAPH_EDGE_DIR dir;         // traverse direction.
	uint minLen;                // Path minimum length.
	uint maxLen;                // Path max length.
	Node *dst;                  // Destination node, defaults to NULL in case of general all paths execution.
	Record r;                   // Record the traversal is being performed upon, only used for edge filtering.
	FT_FilterNode *ft;          // FilterTree of predicates to be applied to traversed edges.
	uint edge_idx;              // Record index of the edge alias, only used for edge filtering.
	bool shortest_paths;        // Only collect shortest paths.
	GrB_Vector visited;         // Visited nodes in shortest path.
} AllPathsCtx;

// Create a new All paths context object.
AllPathsCtx *AllPathsCtx_New(
	Node *src,           // Source node to traverse.
	Node *dst,           // Destination node of the paths
	Graph *g,            // Graph to traverse.
	int *relationIDs,    // Edge type(s) on which we'll traverse.
	int relationCount,   // Length of relationIDs.
	GRAPH_EDGE_DIR dir,  // Traversal direction.
	uint minLen,         // Path length must contain be at least minLen + 1 nodes.
	uint maxLen,         // Path length must not exceed maxLen + 1 nodes.
	Record r,            // Record the traversal is being performed upon.
	FT_FilterNode *ft,   // FilterTree of predicates to be applied to traversed edges.
	uint edge_idx,       // Record index of the edge alias.
	bool shortest_paths  // Only collect shortest paths.
);

void addNeighbors
(
	AllPathsCtx *ctx,
	LevelConnection *frontier,
	uint32_t depth,
	GRAPH_EDGE_DIR dir
);

// Tries to produce a new path from given context
// If no additional path can be computed return NULL.
Path *AllPathsCtx_NextPath(AllPathsCtx *ctx);

// Free context object.
void AllPathsCtx_Free(AllPathsCtx *ctx);
