/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "./bfs.h"
#include "../util/arr.h"
#include "../graph/entities/qg_edge.h"

#include <stddef.h>
#include "rax.h"

// Returns a set of nodes reached at given level from S.
QGNode **BFS(QGNode *s, int *level) {
	void *seen;                             // Has node been visited?
	int current_level = 0;                  // Tracks BFS level.
	rax *visited = raxNew();                // Dictionary of visited nodes.
	QGNode **next = array_new(QGNode *, 0);     // Nodes to explore next.
	QGNode **current = array_new(QGNode *, 1);  // Nodes currently explored.

	array_append(current, s);

	// As long as we've yet to reach required level and there are nodes to process.
	while(current_level < *level && array_len(current)) {
		// As long as there are nodes in the frontier.
		for(int i = 0; i < array_len(current); i++) {
			QGNode *n = current[i];

			// Have we already processed n?
			seen = raxFind(visited, (unsigned char *)n->alias, strlen(n->alias));
			if(seen != raxNotFound) continue;

			// Expand node N by visiting all of its neighbors
			for(int j = 0; j < array_len(n->outgoing_edges); j++) {
				QGEdge *e = n->outgoing_edges[j];
				seen = raxFind(visited, (unsigned char *)e->dest->alias, strlen(e->dest->alias));
				if(seen == raxNotFound) {
					array_append(next, e->dest);
				}
			}
			for(int j = 0; j < array_len(n->incoming_edges); j++) {
				QGEdge *e = n->incoming_edges[j];
				seen = raxFind(visited, (unsigned char *)e->src->alias, strlen(e->src->alias));
				if(seen == raxNotFound) {
					array_append(next, e->src);
				}
			}

			// Mark n as visited.
			raxInsert(visited, (unsigned char *)n->alias, strlen(n->alias), NULL, NULL);
		}

		/* No way to progress and we're interested in the lowest level leafs
		 * do not clear current queue. */
		if(array_len(next) == 0 && *level == BFS_LOWEST_LEVEL) {
			*level = current_level;
			break;
		}

		// Queue consumed, swap queues.
		array_clear(current);
		QGNode **tmp = current;
		current = next;
		next = tmp;
		current_level++;
	}

	raxFree(visited);
	array_free(next);
	return current;
}

