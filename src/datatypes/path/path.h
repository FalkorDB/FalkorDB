/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../../graph/entities/node.h"
#include "../../graph/entities/edge.h"

typedef struct {
	Node *nodes;    // Nodes in paths.
	Edge *edges;    // Edges in path.
} Path;

// creates a new Path with given capacity
Path *Path_New(size_t len);

// ensure the nodes and edge array in a specific len
void Path_EnsureLen(Path *p, size_t len);

// appends a node to the path
void Path_AppendNode(Path *p, Node n);

// appends a relationship to the path
void Path_AppendEdge(Path *p, Edge e);

// set node in a position
void Path_SetNode(Path *p, uint i, Node n);

// set edge in a position
void Path_SetEdge(Path *p, uint i, Edge e);

// returns a refernce to a node in the specific index
Node *Path_GetNode(const Path *p, int index);

// returns a refernce to an edge in the specific index
Edge *Path_GetEdge(const Path *p, int index);

// removes the last node from the path
Node Path_PopNode(Path *p);

// removes the last edge from the path
Edge Path_PopEdge(Path *p);

// returns the last node in the path
Node Path_Head(Path *p);

// returns the amount of nodes in the path
size_t Path_NodeCount(const Path *p);

// returns the amount of edges in the path
size_t Path_EdgeCount(const Path *p);

// returns the path length - amount of edges
size_t Path_Len(const Path *p);

// returns if a path contains a node
bool Path_ContainsNode(const Path *p, Node *n);

// clones a path
Path *Path_Clone(const Path *p);

// reverse the order of the path
void Path_Reverse(Path *p);

// clear the path
void Path_Clear(Path *p);

// deletes the path nodes and edges arrays
void Path_Free(Path *p);

