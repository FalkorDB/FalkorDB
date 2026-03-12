/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "./path.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"

Path *Path_New
(
	size_t len
) {
	Path *path  = rm_malloc (sizeof (Path)) ;
	path->edges = array_new(Edge, len);
	path->nodes = array_new(Node, len + 1);

	return path ;
}

void Path_EnsureLen
(
	Path *p,
	size_t len
) {
	p->nodes = array_ensure_len (p->nodes, len) ;
	p->edges = array_ensure_len (p->edges, len - 1) ;
}

void Path_AppendNode
(
	Path *p,
	Node n
) {
	array_append (p->nodes, n) ;
}

void Path_AppendEdge
(
	Path *p,
	Edge e
) {
	array_append (p->edges, e) ;
}

void Path_SetNode
(
	Path *p,
	uint i,
	Node n
) {
	ASSERT (p != NULL) ;
	ASSERT (i < array_len (p->nodes)) ;

	p->nodes [i] = n ;
}

void Path_SetEdge
(
	Path *p,
	uint i,
	Edge e
) {
	ASSERT (p != NULL) ;
	ASSERT (i < array_len (p->edges)) ;

	p->edges [i] = e ;
}

// returns a refernce to a node in the specific index
Node *Path_GetNode
(
	const Path *p,
	uint index
) {
	ASSERT (p != NULL) ;
	ASSERT (index < Path_NodeCount (p)) ;
	return &p->nodes [index] ;
}

// returns a refernce to an edge in the specific index
Edge *Path_GetEdge
(
	const Path *p,
	uint index
) {
	ASSERT (p != NULL) ;
	ASSERT (index < Path_EdgeCount (p)) ;
	return &p->edges [index] ;
}

// removes the last node from the path
Node Path_PopNode
(
	Path *p
) {
	ASSERT (p != NULL) ;
	return array_pop (p->nodes) ;
}

// removes the last edge from the path
Edge Path_PopEdge
(
	Path *p
) {
	ASSERT (p != NULL) ;
	return array_pop (p->edges) ;
}

// returns the amount of nodes in the path
inline size_t Path_NodeCount
(
	const Path *p
) {
	ASSERT (p != NULL) ;
	return array_len (p->nodes) ;
}

// returns the amount of edges in the path
inline size_t Path_EdgeCount
(
	const Path *p
) {
	ASSERT (p != NULL) ;
	return array_len (p->edges) ;
}

// returns the last node in the path
Node Path_Head
(
	Path *p
) {
	ASSERT (p != NULL) ;
	return p->nodes [array_len (p->nodes) - 1] ;
}

size_t Path_Len(const Path *p) {
	return Path_EdgeCount(p);
}

bool Path_ContainsNode
(
	const Path *p,
	Node *n
) {
	ASSERT (p != NULL) ;
	ASSERT (n != NULL) ;

	uint32_t pathDepth = Path_NodeCount (p) ;
	EntityID nId = ENTITY_GET_ID (n) ;

	for (int i = 0; i < pathDepth; i++) {
		if (ENTITY_GET_ID (p->nodes + i) == nId) {
			return true ;
		}
	}

	return false ;
}

// clones a path
Path *Path_Clone
(
	const Path *p
) {
	ASSERT (p != NULL) ;

	Path *clone = rm_malloc (sizeof (Path)) ;

	array_clone (clone->nodes, p->nodes) ;
	array_clone (clone->edges, p->edges) ;

	return clone ;
}

// reverse the order of the path
void Path_Reverse
(
	Path *p
) {
	ASSERT (p != NULL) ;

	array_reverse (p->nodes) ;
	array_reverse (p->edges) ;
}

// clear the path
void Path_Clear
(
	Path *p
) {
	ASSERT (p != NULL) ;

	array_clear (p->nodes) ;
	array_clear (p->edges) ;
}

// deletes the path nodes and edges arrays
void Path_Free
(
	Path *p
) {
	ASSERT (p != NULL) ;

	array_free (p->nodes) ;
	array_free (p->edges) ;
	rm_free (p) ;
}

