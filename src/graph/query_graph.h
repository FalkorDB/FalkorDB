/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../ast/ast.h"
#include "entities/node.h"
#include "entities/edge.h"
#include "entities/qg_node.h"
#include "entities/qg_edge.h"
#include "../ast/ast_shared.h"
#include "../../deps/GraphBLAS/Include/GraphBLAS.h"

typedef struct {
	QGNode **nodes;             // Nodes contained in QueryGraph
	QGEdge **edges;             // Edges contained in QueryGraph
	bool unknown_reltype_ids;   // Indicates if the query graph contains unknown relationship ids.
} QueryGraph;

typedef enum {
	ENTITY_UNKNOWN,
	ENTITY_NODE,
	ENTITY_EDGE,
} EntityType;

/* Prepare a new query graph with initial allocations for
 * the provided node and edge counts. */
QueryGraph *QueryGraph_New(uint node_cap, uint edge_cap);

/* Adds a new node to the graph */
void QueryGraph_AddNode(QueryGraph *g, QGNode *n);

/* Adds a new edge to the graph */
void QueryGraph_ConnectNodes(QueryGraph *qg, QGNode *src, QGNode *dest, QGEdge *e);

/* Extract a sub-graph of 'qg' according to the path(s) definitions within
 * 'paths' variable, elements missing from 'qg' will be created */
QueryGraph *QueryGraph_ExtractPaths(const QueryGraph *qg,
		const cypher_astnode_t **paths, uint n);

/* Extract a sub-graph of 'qg' according to the path(s) difinitions within
 * 'patterns' variable, elements missing from 'qg' will be created */
QueryGraph *QueryGraph_ExtractPatterns(const QueryGraph *qg,
		const cypher_astnode_t **patterns, uint n);

/* Adds all paths described in an AST pattern node (from a
 * MATCH or MERGE clause) to a meta-graph that describes all
 * nodes and relationships in a query. */
QueryGraph *BuildQueryGraph(const AST *ast);

// Make sure that all entities in the "from" QueryGraph are represented in the "to" QueryGraph.
void QueryGraph_MergeGraphs(QueryGraph *to, QueryGraph *from);

/* Retrieve node by alias */
QGNode *QueryGraph_GetNodeByAlias(const QueryGraph *qg, const char *alias);

/* Retrieve edge by alias */
QGEdge *QueryGraph_GetEdgeByAlias(const QueryGraph *qg, const char *alias);

/* Determine whether a given alias refers to a node or relation. */
EntityType QueryGraph_GetEntityTypeByAlias(const QueryGraph *qg, const char *alias);

/* Tries to update the query graph's unknown relationship ids. */
void QueryGraph_ResolveUnknownRelIDs(QueryGraph *g);

/* Performs deep copy of input query graph. */
QueryGraph *QueryGraph_Clone(const QueryGraph *g);

/* Remove given node from query graph. */
QGNode *QueryGraph_RemoveNode(QueryGraph *g, QGNode *n);

/* Remove given edge from query graph. */
QGEdge *QueryGraph_RemoveEdge(QueryGraph *g, QGEdge *e);

/* Breaks up query graph into its connected components.
 * Returns an array object */
QueryGraph **QueryGraph_ConnectedComponents(const QueryGraph *qg);

/* Retrieve the number of nodes in a QueryGraph. */
uint QueryGraph_NodeCount(const QueryGraph *qg);

/* Retrieve the number of edges in a QueryGraph. */
uint QueryGraph_EdgeCount(const QueryGraph *qg);

/* Build a matrix representation of query graph. */
GrB_Matrix QueryGraph_MatrixRepresentation(const QueryGraph *qg);

/* Returns a string representation of query graph.
 * http://viz-js.com/ */
void QueryGraph_Print(const QueryGraph *qg);

/* Frees entire graph */
void QueryGraph_Free(QueryGraph *qg);

