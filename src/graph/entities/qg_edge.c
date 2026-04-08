/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "qg_edge.h"
#include "qg_node.h"
#include "../graph.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../graphcontext.h"
#include "../../schema/schema.h"

QGEdge *QGEdge_New
(
	const char *relationship,
	const char *alias
) {
	QGEdge *e = rm_malloc(sizeof(QGEdge));

	e->alias         = (alias != NULL) ? rm_strdup(alias) : NULL;
	e->reltypes      = arr_new(const char*, 1);
	e->reltypeIDs    = arr_new(int, 1);
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

	// deep-copy alias so clone doesn't point into AST parse tree
	if(orig->alias != NULL) {
		e->alias = rm_strdup(orig->alias);
	}

	// deep-copy reltype strings
	arr_clone(e->reltypes, orig->reltypes);
	uint reltype_count = arr_len(orig->reltypes);
	for(uint i = 0; i < reltype_count; i++) {
		e->reltypes[i] = rm_strdup(orig->reltypes[i]);
	}

	arr_clone(e->reltypeIDs, orig->reltypeIDs);

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
	return arr_len(e->reltypes);
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

// tries to resolves unknown relationship types
// return false if at least one relationship type remained unresolved
// otherwise all relationship types are resolved and true is returned
bool QGEdge_ResolveUnknownRelIDS
(
	QGEdge *e  // edge to update
) {
	ASSERT (e != NULL) ;

	GraphContext *gc = QueryCtx_GetGraphCtx () ;
	bool    res = true ;  // assuming all relationship types are resolved
	Schema *s   = NULL ;
	uint    n   = arr_len (e->reltypeIDs) ;

	for (uint i = 0; i < n; i++) {
		if (e->reltypeIDs [i] == GRAPH_UNKNOWN_RELATION) {
			// try to resolve an unknown relationship type
			s = GraphContext_GetSchema (gc, e->reltypes [i], SCHEMA_EDGE) ;
			if (s != NULL) {
				// update relationship type
				e->reltypeIDs[i] = s->id ;
			} else {
				// cannot update the unkown relationship
				res = false ;
			}
		}
	}

	return res ;
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

	// free owned alias string
	if(e->alias != NULL) {
		rm_free((char *)e->alias);
	}

	// free owned reltype strings
	uint reltype_count = arr_len(e->reltypes);
	for(uint i = 0; i < reltype_count; i++) {
		rm_free((char *)e->reltypes[i]);
	}

	arr_free(e->reltypes);
	arr_free(e->reltypeIDs);

	rm_free(e);
}

