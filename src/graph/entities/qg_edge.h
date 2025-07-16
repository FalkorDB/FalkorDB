/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "qg_node.h"
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

struct QGEdge {
	const char *alias;      // user-provided alias attached to edge
	const char **reltypes;  // relationship type strings
	int *reltypeIDs;        // relationship type IDs // TODO: can be uint save for GRAPH_NO_RELATION
	QGNode *src;            // pointer to source node
	QGNode *dest;           // pointer to destination node
	uint minHops;           // minimum number of hops this edge represents
	uint maxHops;           // maximum number of hops this edge represents
	bool bidirectional;     // edge doesn't have a direction
	bool shortest_path;     // only edges in the shortest paths should be collected
};

typedef struct QGEdge QGEdge;

// creates a new edge without forming connections
QGEdge *QGEdge_New
(
	const char *relationship,
	const char *alias
);

// returns edge alias
const char *QGEdge_Alias
(
	const QGEdge *e
);

// returns edge source node
QGNode *QGEdge_Src
(
	const QGEdge *e
);

// returns edge destination node
QGNode *QGEdge_Dest
(
	const QGEdge *e
);

// create a duplicate of an edge containing all of the original's data
QGEdge *QGEdge_Clone
(
	const QGEdge *orig
);

// determine whether this is a variable length edge
bool QGEdge_VariableLength
(
	const QGEdge *e
);

// determine whether this is a "ghost" edge
// an edge of length zero
// e.g. ()-[*0]->()
bool QGEdge_GhostEdge
(
	const QGEdge *e
);

// determine whether this edge represent a single hop
bool QGEdge_SingleHop
(
	const QGEdge *e
);

// determine whether this is part of an allShortestPaths query
bool QGEdge_IsShortestPath
(
	const QGEdge *e
);

// number of relationships associated with edge
int QGEdge_RelationCount
(
	const QGEdge *e
);

// return relationship for relation at position 'idx'
const char *QGEdge_Relation
(
	const QGEdge *e,
	int idx
);

// return relationship id for relation at position 'idx'
int QGEdge_RelationID
(
	const QGEdge *e,
	int idx
);

// reverse edge direction
void QGEdge_Reverse
(
	QGEdge *e
);

// gets a string representation of given edge
void QGEdge_ToString
(
	const QGEdge *e,
	sds *buff
);

// free allocations associated with the given edge
void QGEdge_Free
(
	QGEdge *e
);

