/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "qg_edge.h"
#include "qg_node.h"
#include "../graph.h"
#include "../../util/arr.h"

QGEdge *QGEdge_New
(
	const char *relationship,
	const char *alias
) {
	QGEdge *e = rm_malloc(sizeof(QGEdge));

	e->alias         = alias;
	e->reltypes      = array_new(const char*, 1);
	e->reltypeIDs    = array_new(int, 1);
	e->src           = NULL;
	e->dest          = NULL;
	e->minHops       = 1;
	e->maxHops       = 1;
	e->bidirectional = false;
	e->shortest_path = false;

	return e;
}

const char *QGEdge_Alias
(
	const QGEdge *e
) {
	ASSERT(e != NULL);

	return e->alias;
}

QGNode *QGEdge_Src
(
	const QGEdge *e
) {
	ASSERT(e != NULL);

	return e->src;
}

QGNode *QGEdge_Dest
(
	const QGEdge *e
) {
	ASSERT(e != NULL);

	return e->dest;
}

QGEdge *QGEdge_Clone
(
	const QGEdge *orig
) {
	QGEdge *e = rm_malloc(sizeof(QGEdge));
	memcpy(e, orig, sizeof(QGEdge));
	e->src = NULL;
	e->dest = NULL;
	array_clone(e->reltypes, orig->reltypes);
	array_clone(e->reltypeIDs, orig->reltypeIDs);

	return e;
}

// returns true if edge represents a variable length path
// e.g. ()-[*]->()
bool QGEdge_VariableLength
(
	const QGEdge *e  // edge to check
) {
	ASSERT(e != NULL);

	return e->minHops != e->maxHops;
}

// determine whether this is a "ghost" edge
// an edge of length zero
// e.g. ()-[*0]->()
bool QGEdge_GhostEdge
(
	const QGEdge *e
) {
	return (e->minHops == e->maxHops && e->minHops == 0);
}

bool QGEdge_SingleHop
(
	const QGEdge *e
) {
	ASSERT(e);
	return (e->minHops == 1 && e->maxHops == 1);
}

bool QGEdge_IsShortestPath
(
	const QGEdge *e
) {
	ASSERT(e);
	return e->shortest_path;
}

int QGEdge_RelationCount
(
	const QGEdge *e
) {
	ASSERT(e);
	return array_len(e->reltypes);
}

const char *QGEdge_Relation
(
	const QGEdge *e,
	int idx
) {
	ASSERT(e != NULL);
	ASSERT(idx < QGEdge_RelationCount(e));

	return e->reltypes[idx];
}

int QGEdge_RelationID
(
	const QGEdge *e,
	int idx
) {
	ASSERT(e != NULL && idx < QGEdge_RelationCount(e));
	return e->reltypeIDs[idx];
}

void QGEdge_Reverse
(
	QGEdge *e
) {
	QGNode *src = e->src;
	QGNode *dest = e->dest;

	QGNode_RemoveOutgoingEdge(src, e);
	QGNode_RemoveIncomingEdge(dest, e);

	// Reconnect nodes with the source and destination reversed.
	e->src = dest;
	e->dest = src;

	QGNode_ConnectNode(e->src, e->dest, e);
}

void QGEdge_ToString
(
	const QGEdge *e,
	sds *buff
) {
	ASSERT(e && buff && *buff);

	*buff = sdscatprintf(*buff, "[");

	if(e->alias) *buff = sdscatprintf(*buff, "%s", e->alias);
	uint reltype_count = QGEdge_RelationCount(e);
	for(uint i = 0; i < reltype_count; i ++) {
		// Multiple relationship types are printed separated by pipe characters
		if(i > 0) *buff = sdscatprintf(*buff, "|");
		*buff = sdscatprintf(*buff, ":%s", e->reltypes[i]);
	}
	if(e->minHops != 1 || e->maxHops != 1) {
		if(e->maxHops == EDGE_LENGTH_INF)
			*buff = sdscatprintf(*buff, "*%u..INF", e->minHops);
		else
			*buff = sdscatprintf(*buff, "*%u..%u", e->minHops, e->maxHops);
	}

	*buff = sdscatprintf(*buff, "]");
}

void QGEdge_Free
(
	QGEdge *e
) {
	if(!e) return;

	array_free(e->reltypes);
	array_free(e->reltypeIDs);

	rm_free(e);
}

