/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "./longest_path.h"
#include "./bfs.h"
#include "./dfs.h"
#include "../util/arr.h"
#include"./detect_cycle.h"

// Scans the graph in a DFS fashion, keeps track after the longest path length.
static void __DFSMaxDepth(QGNode *n, int level, int *max_depth, rax *visited, rax *used_edges) {
	if(level > *max_depth) *max_depth = level;

	// Mark n as visited, return if node already marked.
	if(!raxInsert(visited, (unsigned char *)n->alias, strlen(n->alias), NULL, NULL)) {
		// We've already processed n.
		return;
	}

	// Expand node N by visiting all of its neighbors
	for(uint i = 0; i < array_len(n->outgoing_edges); i++) {
		QGEdge *e = n->outgoing_edges[i];
		if(!raxInsert(used_edges, (unsigned char *)e->alias, strlen(e->alias), NULL, NULL)) continue;
		__DFSMaxDepth(e->dest, level + 1, max_depth, visited, used_edges);
		raxRemove(used_edges, (unsigned char *)e->alias, strlen(e->alias), NULL);
	}

	// Expand node N by visiting all of its neighbors
	for(uint i = 0; i < array_len(n->incoming_edges); i++) {
		QGEdge *e = n->incoming_edges[i];
		if(!raxInsert(used_edges, (unsigned char *)e->alias, strlen(e->alias), NULL, NULL)) continue;
		__DFSMaxDepth(e->src, level + 1, max_depth, visited, used_edges);
		raxRemove(used_edges, (unsigned char *)e->alias, strlen(e->alias), NULL);
	}

	raxRemove(visited, (unsigned char *)n->alias, strlen(n->alias), NULL);
}

// Finds out the longest path distance from given node.
static int _DFSMaxDepth(QGNode *n) {
	int level = 0;              // Starting at level 0.
	int max_depth = 0;          // Longest path length.
	rax *visited = raxNew();    // Dictionary of visited nodes.
	rax *used_edges = raxNew(); // Dictionary of used edges.

	__DFSMaxDepth(n, level, &max_depth, visited, used_edges);

	raxFree(visited);
	raxFree(used_edges);
	return max_depth;
}

// Finds the longest path in an cyclic graph.
QGNode *LongestPathGraph(const QueryGraph *g, int *level) {
	/* To find the longest path in a graph containing a cycle
	 * where we do not expand from a visited node:
	 * 1. the entire graph is a cycle, in which case it doesn't matter
	 * which node we pick to begin out traversal.
	 * 2. there's a node with in-degree of out-degree 0, as we know
	 * this node resided on the "edge" of the graph from which the longest path
	 * begins/ends. */

	QGNode *n = NULL;  // Node from which the longest path expand.
	uint node_count = QueryGraph_NodeCount(g);
	for(uint i = 0; i < node_count; i++) {
		n = g->nodes[i];
		if(QGNode_IncomeDegree(n) == 0 || QGNode_OutgoingDegree(n) == 0) {
			*level = _DFSMaxDepth(n);
			return n;
		}
	}

	// All nodes are part of a cycle, pick one randomly.
	n = g->nodes[0];
	*level = _DFSMaxDepth(n);
	return n;
}

// Finds the longest path in an acyclic graph.
QGNode *LongestPathTree(const QueryGraph *g, int *level) {
	int l = BFS_LOWEST_LEVEL;
	QGNode **leafs = BFS(g->nodes[0], &l);
	QGNode *leaf = leafs[0];
	array_free(leafs);

	l = BFS_LOWEST_LEVEL;
	leafs = BFS(leaf, &l);

	ASSERT(array_len(leafs) > 0 && l >= 0);
	QGNode *n = leafs[0];
	array_free(leafs);

	*level = l;
	return n;
}

