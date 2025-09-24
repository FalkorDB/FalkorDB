/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "./path.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"

Path *Path_New(size_t len) {
	Path *path = rm_malloc(sizeof(Path));
	path->edges = array_new(Edge, len);
	path->nodes = array_new(Node, len + 1);
	return path;
}

void Path_EnsureLen(Path *p, size_t len) {
	p->nodes = array_ensure_len(p->nodes, len);
	p->edges = array_ensure_len(p->edges, len - 1);
}

void Path_AppendNode(Path *p, Node n) {
	array_append(p->nodes, n);
}

void Path_AppendEdge(Path *p, Edge e) {
	array_append(p->edges, e);
}

void Path_SetNode(Path *p, uint i, Node n) {
	p->nodes[i] = n;
}

void Path_SetEdge(Path *p, uint i, Edge e) {
	p->edges[i] = e;
}

Node *Path_GetNode(const Path *p, int index) {
	ASSERT(index >= 0 && index < Path_NodeCount(p));
	return &p->nodes[index];
}

Edge *Path_GetEdge(const Path *p, int index) {
	ASSERT(index >= 0 && index < Path_EdgeCount(p));
	return &p->edges[index];
}

Node Path_PopNode(Path *p) {
	return array_pop(p->nodes);
}
Edge Path_PopEdge(Path *p) {
	return array_pop(p->edges);
}

size_t Path_NodeCount(const Path *p) {
	return array_len(p->nodes);
}

size_t Path_EdgeCount(const Path *p) {
	return array_len(p->edges);
}

Node Path_Head(Path *p) {
	return p->nodes[array_len(p->nodes) - 1];
}

size_t Path_Len(const Path *p) {
	return Path_EdgeCount(p);
}

bool Path_ContainsNode(const Path *p, Node *n) {
	uint32_t pathDepth = Path_NodeCount(p);
	EntityID nId = ENTITY_GET_ID(n);
	for(int i = 0; i < pathDepth; i++) {
		if(ENTITY_GET_ID(p->nodes + i) == nId) return true;
	}
	return false;
}

Path *Path_Clone(const Path *p) {
	Path *clone = rm_malloc(sizeof(Path));
	array_clone(clone->nodes, p->nodes);
	array_clone(clone->edges, p->edges);
	return clone;
}

void Path_Reverse(Path *p) {
	array_reverse(p->nodes);
	array_reverse(p->edges);
}

void Path_Clear(Path *p) {
	array_clear(p->nodes);
	array_clear(p->edges);
}

void Path_Free(Path *p) {
	array_free(p->nodes);
	array_free(p->edges);
	rm_free(p);
}
