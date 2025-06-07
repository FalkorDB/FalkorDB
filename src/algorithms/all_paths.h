/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// finds all paths starting at given source node
// we're computing one path at a time, this is done
// to take advantage of scenarios where a query specifies LIMIT
// to implement this kind of iterative path finding using DFS
// we're keeping track after:
// 1. the last path computed, which we'll try to expand
// 2. neighboring nodes discovered, each placed within a "level"
// array containing all nodes discovered at a specific level

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
	int *relationIDs;           // edge type(s) to traverse
	int relationCount;          // length of relationIDs
	GRAPH_EDGE_DIR dir;         // traverse direction
	uint minLen;                // path minimum length
	uint maxLen;                // path max length
	Node *dst;                  // destination node, defaults to NULL in case of general all paths execution
	Record r;                   // record the traversal is being performed upon, only used for edge filtering
	FT_FilterNode *ft;          // filterTree of predicates to be applied to traversed edges
	int path_idx;               // record index of the variable length edge path object, only used for path filtering
	int edge_idx;               // record index of last edge added to the path, only used for edge filtering
	bool shortest_paths;        // only collect shortest paths
	GrB_Vector visited;         // visited nodes in shortest path
} AllPathsCtx;

// create a new All paths context object
AllPathsCtx *AllPathsCtx_New(
	Node *src,           // source node to traverse
	Node *dst,           // destination node of the paths
	Graph *g,            // graph to traverse
	int *relationIDs,    // edge type(s) on which we'll traverse
	int relationCount,   // length of relationIDs
	GRAPH_EDGE_DIR dir,  // traversal direction
	uint minLen,         // path length must contain be at least minLen + 1 nodes
	uint maxLen,         // path length must not exceed maxLen + 1 nodes
	Record r,            // record the traversal is being performed upon
	FT_FilterNode *ft,   // filterTree of predicates to be applied to traversed edges
	int path_idx,        // record index of the variable length edge path object
	int edge_idx,        // record index of the edge alias
	bool shortest_paths  // only collect shortest paths
);

void addNeighbors
(
	AllPathsCtx *ctx,
	LevelConnection *frontier,
	uint32_t depth,
	GRAPH_EDGE_DIR dir
);

// tries to produce a new path from given context
// if no additional path can be computed return NULL
Path *AllPathsCtx_NextPath
(
	AllPathsCtx *ctx
);

// free context object
void AllPathsCtx_Free
(
	AllPathsCtx *ctx
);

