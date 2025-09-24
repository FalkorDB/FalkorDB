/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "./bfs.h"
#include "../util/arr.h"
#include "rax.h"
#include "../graph/entities/qg_edge.h"

bool _DFS(QGNode *n, int level, bool close_cycle, int current_level, rax *visited, rax *used_edges,
		  QGEdge ***path) {
	// As long as we've yet to reach required level and there are nodes to process.
	if(current_level >= level) return true;

	// Mark n as visited, return if node already marked.
	if(!raxInsert(visited, (unsigned char *)n->alias, strlen(n->alias), NULL, NULL)) {
		// We've already processed n.
		return false;
	}

	// Expand node N by visiting all of its neighbors
	bool not_seen;
	for(uint i = 0; i < array_len(n->outgoing_edges); i++) {
		QGEdge *e = n->outgoing_edges[i];
		not_seen = raxFind(visited, (unsigned char *)e->dest->alias, strlen(e->dest->alias)) == raxNotFound;
		if(not_seen || close_cycle) {
			if(!raxInsert(used_edges, (unsigned char *)e->alias, strlen(e->alias), NULL, NULL)) continue;
			array_append(*path, e);
			if(_DFS(e->dest, level, close_cycle, current_level + 1, visited, used_edges, path)) return true;
			array_pop(*path);
			raxRemove(used_edges, (unsigned char *)e->alias, strlen(e->alias), NULL);
		}
	}

	for(uint i = 0; i < array_len(n->incoming_edges); i++) {
		QGEdge *e = n->incoming_edges[i];
		not_seen = raxFind(visited, (unsigned char *)e->src->alias, strlen(e->src->alias)) == raxNotFound;
		if(not_seen || close_cycle) {
			if(!raxInsert(used_edges, (unsigned char *)e->alias, strlen(e->alias), NULL, NULL)) continue;
			array_append(*path, e);
			if(_DFS(e->src, level, close_cycle, current_level + 1, visited, used_edges, path)) return true;
			array_pop(*path);
			raxRemove(used_edges, (unsigned char *)e->alias, strlen(e->alias), NULL);
		}
	}

	raxRemove(visited, (unsigned char *)n->alias, strlen(n->alias), NULL);
	return false;
}

// Returns a single path from S to a reachable node at distance level.
QGEdge **DFS(QGNode *s, int level, bool close_cycle) {
	int current_level = 0;                  // Tracks BFS level.
	rax *visited = raxNew();                // Dictionary of visited nodes.
	rax *used_edges = raxNew();             // Dictionary of used edges.
	QGEdge **path = array_new(QGEdge *, 0); // Path found.

	_DFS(s, level, close_cycle, current_level, visited, used_edges, &path);
	raxFree(visited);
	raxFree(used_edges);
	return path;
}
