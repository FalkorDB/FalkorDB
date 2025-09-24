/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "query_graph.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../schema/schema.h"
#include "../../deps/rax/rax.h"

// sets node label and label ID
static void _QueryGraphSetNodeLabel
(
	QGNode *n,
	const cypher_astnode_t *ast_entity
) {
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// retrieve node labels from the AST entity
	uint nlabels = cypher_ast_node_pattern_nlabels(ast_entity);

	for(uint i = 0; i < nlabels; i++) {
		const char *l = cypher_ast_label_get_name(
							cypher_ast_node_pattern_get_label(ast_entity, i));

		// if a schema is found, the AST refers to an existing label
		Schema *s = GraphContext_GetSchema(gc, l, SCHEMA_NODE);
		int l_id = (s) ? Schema_GetID(s) : GRAPH_UNKNOWN_LABEL;
		QGNode_AddLabel(n, l, l_id);
	}
}

// adds node to query graph
static void _QueryGraphAddNode
(
	QueryGraph *qg,
	const cypher_astnode_t *ast_entity
) {
	const char *alias = AST_ToString(ast_entity);

	// look up this alias in the QueryGraph
	// this node may already exist
	// if it appears multiple times in query patterns
	QGNode *n = QueryGraph_GetNodeByAlias(qg, alias);
	if(n == NULL) {
		// node has not been mapped; create it
		n = QGNode_New(alias);
		QueryGraph_AddNode(qg, n);
	}

	_QueryGraphSetNodeLabel(n, ast_entity);
}

// adds edge to query graph
static void _QueryGraphAddEdge
(
	QueryGraph *qg,                      // query graph to add edge to
	const cypher_astnode_t *ast_entity,  // edge entity
	QGNode *src,                         // src node
	QGNode *dest,                        // dest node
	bool only_shortest                   // edge is part of a shortest path
) {
	GraphContext *gc = QueryCtx_GetGraphCtx();
	const char *alias = AST_ToString(ast_entity);
	enum cypher_rel_direction dir =
		cypher_ast_rel_pattern_get_direction(ast_entity);

	// each edge can only appear once in a QueryGraph
	ASSERT(QueryGraph_GetEdgeByAlias(qg, alias) == NULL);

	QGEdge *edge = QGEdge_New(NULL, alias);
	edge->bidirectional = (dir == CYPHER_REL_BIDIRECTIONAL);
	edge->shortest_path = only_shortest;

	// add the IDs of all reltype matrixes
	uint nreltypes = cypher_ast_rel_pattern_nreltypes(ast_entity);
	for(uint i = 0; i < nreltypes; i ++) {
		const char *reltype = cypher_ast_reltype_get_name(cypher_ast_rel_pattern_get_reltype(ast_entity,
														  i));
		bool found = false;
		Schema *s = GraphContext_GetSchema(gc, reltype, SCHEMA_EDGE);
		if(!s) {
			// unknown relationship
			// search if reltype exists in edge->reltypes to don't insert duplicated reltype
			int len = array_len(edge->reltypes);
			for (int j = 0; j < len; j++) {
				if(edge->reltypeIDs[j] == GRAPH_UNKNOWN_RELATION) {
					if(strcasecmp(edge->reltypes[j],reltype) == 0) {
						found = true;
						break;
					}
				}
			}
			if(!found) {
				array_append(edge->reltypes, reltype);
				array_append(edge->reltypeIDs, GRAPH_UNKNOWN_RELATION);
				qg->unknown_reltype_ids = true;
			}
			continue;
		}
		// search if s-id exists in edge->reltypeIDs to don't insert duplicated ids
		int len = array_len(edge->reltypeIDs);
		for (int j = 0; j < len; j++) {
			if(edge->reltypeIDs[j] == s->id) {
				found = true;
				break;
			}
		}
		if(!found) {
			array_append(edge->reltypes, reltype);
			array_append(edge->reltypeIDs, s->id);
		}
	}

	// incase of a variable length edge, set edge min/max hops
	const cypher_astnode_t *range = cypher_ast_rel_pattern_get_varlength(ast_entity);
	if(range) {
		const cypher_astnode_t *start = cypher_ast_range_get_start(range);
		const cypher_astnode_t *end = cypher_ast_range_get_end(range);
		if(start) edge->minHops = AST_ParseIntegerNode(start);
		if(end) edge->maxHops = AST_ParseIntegerNode(end);
		else edge->maxHops = EDGE_LENGTH_INF;
	}

	// build and add a QGEdge representing this entity to the QueryGraph
	// swap the source and destination for left-pointing relations
	if(dir != CYPHER_REL_INBOUND) QueryGraph_ConnectNodes(qg, src, dest, edge);
	else QueryGraph_ConnectNodes(qg, dest, src, edge);
}

// extracts node from 'qg' and places a copy of into 'graph'
static void _QueryGraph_ExtractNode
(
	const QueryGraph *qg,
	QueryGraph *graph,
	const cypher_astnode_t *ast_node
) {

	// validate inputs
	ASSERT(qg        !=  NULL);
	ASSERT(graph     !=  NULL);
	ASSERT(ast_node  !=  NULL);

	// see if node is already in 'graph'
	const char *alias = AST_ToString(ast_node);
	QGNode *n = QueryGraph_GetNodeByAlias(graph, alias);

	if(n == NULL) {
		// node is missing from 'graph', try getting it from 'qg'
		n = QueryGraph_GetNodeByAlias(qg, alias);
		if(n == NULL) {
			// node is missing from 'qg', create it
			// it is possible to get into a situation where we try to extract
			// a path which contains entities that are missing from the
			// "holistic" query graph consider:
			// MATCH (a) WITH a WHERE (a)-[]->(:L1) in this case due to
			// clause scoping only node 'a' is in 'qg' the filtered pattern
			// which is being extracted from 'qg' has additional entities:
			// an anonymous edge and node
			_QueryGraphAddNode(graph, ast_node);
		} else {
			// add a clone of the original node
			n = QGNode_Clone(n);

			// clear node label information
			array_clear(n->labels);
			array_clear(n->labelsID);

			QueryGraph_AddNode(graph, n);
			// set node label information
			_QueryGraphSetNodeLabel(n, ast_node);
		}
	} else {
		// set node label information
		_QueryGraphSetNodeLabel(n, ast_node);
	}
}

// extracts edge from 'qg' and places a copy of into 'graph'
static void _QueryGraph_ExtractEdge
(
	const QueryGraph *qg,
	QueryGraph *graph,
	QGNode *left,
	QGNode *right,
	const cypher_astnode_t *ast_edge
) {
	const char *alias = AST_ToString(ast_edge);

	// validate input, edge shouldn't be in graph
	ASSERT(left != NULL);
	ASSERT(right != NULL);
	ASSERT(QueryGraph_GetEdgeByAlias(graph, alias) == NULL);

	QGEdge *e = QueryGraph_GetEdgeByAlias(qg, alias);
	bool shortest_path = e ? e->shortest_path : false;
	_QueryGraphAddEdge(graph, ast_edge, left, right, shortest_path);
}

// clones path from 'qg' into 'graph'
static void _QueryGraph_ExtractPath
(
	const QueryGraph *qg,
	QueryGraph *graph,
	const cypher_astnode_t *path
) {

	// validate input
	ASSERT(qg != NULL && graph != NULL && path != NULL);

	const char *alias;
	const cypher_astnode_t *ast_node;
	uint nelems = cypher_ast_pattern_path_nelements(path);

	// introduce nodes to graph
	// nodes are at even indices
	for(uint i = 0; i < nelems; i += 2) {
		ast_node = cypher_ast_pattern_path_get_element(path, i);
		_QueryGraph_ExtractNode(qg, graph, ast_node);
	}

	// introduce edges to graph
	// edges are at odd indices
	for(uint i = 1; i < nelems; i += 2) {
		// retrieve the QGNode corresponding to the node left of this edge
		const cypher_astnode_t *l_node = cypher_ast_pattern_path_get_element(path, i - 1);
		const char *l_alias = AST_ToString(l_node);
		QGNode *left = QueryGraph_GetNodeByAlias(graph, l_alias);

		// retrieve the QGNode corresponding to the node right of this edge
		const cypher_astnode_t *r_node = cypher_ast_pattern_path_get_element(path, i + 1);
		const char *r_alias = AST_ToString(r_node);
		QGNode *right = QueryGraph_GetNodeByAlias(graph, r_alias);

		ast_node = cypher_ast_pattern_path_get_element(path, i);
		_QueryGraph_ExtractEdge(qg, graph, left, right, ast_node);
	}
}

QueryGraph *QueryGraph_New
(
	uint node_cap,
	uint edge_cap
) {
	QueryGraph *qg = rm_malloc(sizeof(QueryGraph));

	qg->nodes = array_new(QGNode *, node_cap);
	qg->edges = array_new(QGEdge *, edge_cap);
	qg->unknown_reltype_ids = false;

	return qg;
}

void QueryGraph_AddNode
(
	QueryGraph *qg,
	QGNode *n
) {
	array_append(qg->nodes, n);
}

void QueryGraph_ConnectNodes
(
	QueryGraph *qg,
	QGNode *src,
	QGNode *dest,
	QGEdge *e
) {
	QGNode_ConnectNode(src, dest, e);
	e->src = src;
	e->dest = dest;
	array_append(qg->edges, e);
}

void QueryGraph_AddPath
(
	QueryGraph *qg,                // query graph to add path to
	const cypher_astnode_t *path,  // path to add
	bool only_shortest             // interested only in the shortest paths
) {
	AST *ast = QueryCtx_GetAST();
	uint nelems = cypher_ast_pattern_path_nelements(path);
	// introduce nodes first
	// nodes are positioned at every even offset
	// into the path (0, 2, ...)
	for(uint i = 0; i < nelems; i += 2) {
		const cypher_astnode_t *ast_node =
			cypher_ast_pattern_path_get_element(path, i);
		_QueryGraphAddNode(qg, ast_node);
	}

	// every odd offset corresponds to an edge in a path
	for(uint i = 1; i < nelems; i += 2) {
		// retrieve the QGNode corresponding to the node left of this edge
		const cypher_astnode_t *l_node =
			cypher_ast_pattern_path_get_element(path, i - 1);
		const char *l_alias = AST_ToString(l_node);
		QGNode *left = QueryGraph_GetNodeByAlias(qg, l_alias);

		// retrieve the QGNode corresponding to the node right of this edge
		const cypher_astnode_t *r_node =
			cypher_ast_pattern_path_get_element(path, i + 1);
		const char *r_alias = AST_ToString(r_node);
		QGNode *right = QueryGraph_GetNodeByAlias(qg, r_alias);

		// retrieve the AST reference to this edge
		const cypher_astnode_t *edge =
			cypher_ast_pattern_path_get_element(path, i);
		_QueryGraphAddEdge(qg, edge, left, right, only_shortest);
	}
}

// clones path from 'qg' into 'graph'
QueryGraph *QueryGraph_ExtractPaths
(
	const QueryGraph *qg,
	const cypher_astnode_t **paths,
	uint n
) {
	// validate input
	ASSERT(qg != NULL && paths != NULL);

	// create an empty query graph
	uint node_count = QueryGraph_NodeCount(qg);
	uint edge_count = QueryGraph_EdgeCount(qg);
	QueryGraph *graph = QueryGraph_New(node_count, edge_count);
	ASSERT(graph != NULL);

	for(int i = 0; i < n; i++) {
		const cypher_astnode_t *path = paths[i];
		_QueryGraph_ExtractPath(qg, graph, path);
	}

	return graph;
}

// clones patterns from 'qg' into 'graph'
QueryGraph *QueryGraph_ExtractPatterns
(
	const QueryGraph *qg,
	const cypher_astnode_t **patterns,
	uint n
) {

	// validate inputs
	ASSERT(qg != NULL && patterns != NULL);

	// create an empty query graph
	uint node_count = QueryGraph_NodeCount(qg);
	uint edge_count = QueryGraph_EdgeCount(qg);
	QueryGraph *graph = QueryGraph_New(node_count, edge_count);
	ASSERT(graph != NULL);

	// extract paths described by each pattern
	for(uint i = 0; i < n; i++) {
		const cypher_astnode_t *pattern = patterns[i];
		uint npaths = cypher_ast_pattern_npaths(pattern);
		for(uint j = 0; j < npaths; j ++) {
			const cypher_astnode_t *path = cypher_ast_pattern_get_path(pattern, j);
			_QueryGraph_ExtractPath(qg, graph, path);
		}
	}

	return graph;
}

// build a query graph from AST
QueryGraph *BuildQueryGraph
(
	const AST *ast
) {
	uint node_count;
	uint edge_count;

	// AST clauses containing path objects
	cypher_astnode_type_t clause_types [2] = {CYPHER_AST_MATCH, CYPHER_AST_MERGE};

	// the initial node and edge arrays will be large enough to accommodate
	// all AST entities (which is overkill, consider reducing)
	node_count = edge_count = raxSize(ast->referenced_entities);
	QueryGraph *qg = QueryGraph_New(node_count, edge_count);

	// for each relevant clause type
	for(int i = 0; i < 2; i ++) {
		const uint8_t clause_type = clause_types[i];
		// collect all path objects
		const cypher_astnode_t **clauses = AST_GetTypedNodes(ast->root,
															 clause_type);
		uint clause_count = array_len(clauses);

		// for each clause of the current type
		for(uint j = 0; j < clause_count; j ++) {
			const cypher_astnode_t *clause = clauses[j];
			// collect path objects
			const cypher_astnode_t **paths =
				AST_GetTypedNodes(clause, CYPHER_AST_PATTERN_PATH);
			const cypher_astnode_t **shortest_paths =
				AST_GetTypedNodes(clause, CYPHER_AST_SHORTEST_PATH);

			// differentiate between regular paths and shortest paths
			// as a path can be marked as shortest
			uint  path_count           =  array_len(paths);
			uint  shortest_path_count  =  array_len(shortest_paths);
			bool only_shortest[path_count];
			memset(only_shortest, 0, path_count*sizeof(bool));

			uint l = 0;  // index to paths array
			uint k = 0;  // index to shortest paths array
			for (; k < shortest_path_count; k++) {
				const cypher_astnode_t *shortest = shortest_paths[k];
				// single shortest paths are handled by a procedure
				if(cypher_ast_shortest_path_is_single(shortest)) continue;

				const cypher_astnode_t *path =
					cypher_ast_shortest_path_get_path(shortest);

				// seek the matching path in the paths array
				while(paths[l] != path) l++;
				ASSERT(l < path_count);

				// each shortest must find its counterpart path
				only_shortest[l] = true;
				l++; // advance for next match
			}
			array_free(shortest_paths);

			// introduce each path object to the query graph
			for(uint k = 0; k < path_count; k ++) {
				QueryGraph_AddPath(qg, paths[k], only_shortest[k]);
			}
			array_free(paths);
		}
		array_free(clauses);
	}

	return qg;
}

void QueryGraph_MergeGraphs
(
	QueryGraph *to,
	QueryGraph *from
) {
	uint node_count = QueryGraph_NodeCount(from);
	uint edge_count = QueryGraph_EdgeCount(from);

	for(uint i = 0; i < node_count; i++) {
		QGNode *n = from->nodes[i];
		// if the entity already exists in the "to" graph, do nothing
		// we could have more complex logic to merge entity data, but this is not
		// currently necessary as this logic only benefits toString calls like EXPLAIN
		if(QueryGraph_GetNodeByAlias(to, n->alias)) continue;
		// new entity, clone and add it
		QueryGraph_AddNode(to, QGNode_Clone(n));
	}

	for(uint i = 0; i < edge_count; i++) {
		QGEdge *e = from->edges[i];
		// if the entity already exists in the "to" graph, do nothing
		// we could have more complex logic to merge entity data, but this is not
		// currently necessary as this logic only benefits toString calls like EXPLAIN
		if(QueryGraph_GetEdgeByAlias(to, e->alias)) continue;

		// retrieve the edge's endpoints in the "to" graph
		QGNode *src = QueryGraph_GetNodeByAlias(to, e->src->alias);
		QGNode *dest = QueryGraph_GetNodeByAlias(to, e->dest->alias);
		// clone and add the unmatched edge
		QGEdge *clone_edge = QGEdge_Clone(e);
		QueryGraph_ConnectNodes(to, src, dest, clone_edge);
	}
}

QGNode *QueryGraph_GetNodeByAlias
(
	const QueryGraph *qg,
	const char *alias
) {
	uint node_count = QueryGraph_NodeCount(qg);
	for(uint i = 0; i < node_count; i ++) {
		if(!strcmp(qg->nodes[i]->alias, alias)) return qg->nodes[i];
	}
	return NULL;
}

QGEdge *QueryGraph_GetEdgeByAlias
(
	const QueryGraph *qg,
	const char *alias
) {
	uint edge_count = QueryGraph_EdgeCount(qg);
	for(uint i = 0; i < edge_count; i ++) {
		if(!strcmp(qg->edges[i]->alias, alias)) return qg->edges[i];
	}
	return NULL;
}

EntityType QueryGraph_GetEntityTypeByAlias
(
	const QueryGraph *qg,
	const char *alias
) {
	if(QueryGraph_GetNodeByAlias(qg, alias) != NULL) return ENTITY_NODE;
	if(QueryGraph_GetEdgeByAlias(qg, alias) != NULL) return ENTITY_EDGE;
	return ENTITY_UNKNOWN;
}

void QueryGraph_ResolveUnknownRelIDs
(
	QueryGraph *qg
) {
	// no unknown relationships - no need to updated
	if(!qg->unknown_reltype_ids) return;

	Schema *s = NULL;
	bool unkown_relationships = false;
	GraphContext *gc = QueryCtx_GetGraphCtx();
	uint edge_count = QueryGraph_EdgeCount(qg);

	// update edges
	for(uint i = 0; i < edge_count; i++) {
		QGEdge *edge = qg->edges[i];
		uint rel_types_count = array_len(edge->reltypeIDs);
		for(uint j = 0; j < rel_types_count; j++) {
			if(edge->reltypeIDs[j] == GRAPH_UNKNOWN_RELATION) {
				s = GraphContext_GetSchema(gc, edge->reltypes[j], SCHEMA_EDGE);
				if(s) edge->reltypeIDs[j] = s->id;
				else unkown_relationships = true; // cannot update the unkown relationship
			}
		}
	}

	qg->unknown_reltype_ids = unkown_relationships;
}

QueryGraph *QueryGraph_Clone
(
	const QueryGraph *qg
) {
	uint        node_count  =  QueryGraph_NodeCount(qg);
	uint        edge_count  =  QueryGraph_EdgeCount(qg);
	QueryGraph  *clone      =  QueryGraph_New(node_count, edge_count);

	// clone nodes
	for(uint i = 0; i < node_count; i++) {
		// clones node without its edges
		QGNode *n = QGNode_Clone(qg->nodes[i]);
		QueryGraph_AddNode(clone, n);
	}

	// clone edges
	for(uint i = 0; i < edge_count; i++) {
		QGEdge  *e     =  qg->edges[i];
		QGNode  *src   =  QueryGraph_GetNodeByAlias(clone, e->src->alias);
		QGNode  *dest  =  QueryGraph_GetNodeByAlias(clone, e->dest->alias);
		ASSERT(src != NULL && dest != NULL);

		QGEdge *clone_edge = QGEdge_Clone(e);
		QueryGraph_ConnectNodes(clone, src, dest, clone_edge);
	}

	return clone;
}

QGNode *QueryGraph_RemoveNode
(
	QueryGraph *qg,
	QGNode *n
) {
	ASSERT(qg != NULL && n != NULL);

	// remove node from query graph
	// remove and free all edges associated with node
	uint incoming_edge_count = array_len(n->incoming_edges);
	for(uint i = 0; i < incoming_edge_count; i++) {
		QGEdge *e = n->incoming_edges[i];
		QueryGraph_RemoveEdge(qg, e);
		QGEdge_Free(e);
	}

	uint outgoing_edge_count = array_len(n->outgoing_edges);
	for(uint i = 0; i < outgoing_edge_count; i++) {
		QGEdge *e = n->outgoing_edges[i];
		QueryGraph_RemoveEdge(qg, e);
		QGEdge_Free(e);
	}

	// remove node from graph nodes
	uint node_count = QueryGraph_NodeCount(qg);
	uint i = 0;
	for(; i < node_count; i++) {
		if(n == qg->nodes[i]) {
			array_del_fast(qg->nodes, i);
			break;
		}
	}

	return n;
}

QGEdge *QueryGraph_RemoveEdge
(
	QueryGraph *qg,
	QGEdge *e
) {
	ASSERT(qg != NULL && e != NULL);

	// disconnect nodes connected by edge
	QGNode_RemoveOutgoingEdge(e->src, e);
	QGNode_RemoveIncomingEdge(e->dest, e);

	// remove edge from query graph
	uint edge_count = QueryGraph_EdgeCount(qg);
	uint i = 0;
	for(; i < edge_count; i++) {
		if(e == qg->edges[i]) {
			array_del_fast(qg->edges, i);
			break;
		}
	}
	return e;
}

QueryGraph **QueryGraph_ConnectedComponents
(
	const QueryGraph *qg
) {
	QGNode *n;                              // current node
	QGNode **q = array_new(QGNode *, 1);    // node frontier
	void *seen;                             // has node been visited?
	QueryGraph *g = QueryGraph_Clone(qg);   // clone query graph
	rax *visited;                           // dictionary of visited nodes
	QueryGraph **connected_components;      // list of connected components

	// at least one connected component (the original graph)
	connected_components = array_new(QueryGraph *, 1);

	// as long as there are nodes to process
	while(true) {
		visited = raxNew();

		// get a random node and add it to the frontier
		QGNode *s = g->nodes[0];
		array_append(q, s);

		// as long as there are nodes in the frontier
		while(array_len(q) > 0) {
			n = array_pop(q);

			// mark n as visited
			if(!raxInsert(visited, (unsigned char *)n->alias, strlen(n->alias),
						  NULL, NULL)) {
				// we've already processed n
				continue;
			}

			// expand node N by visiting all of its neighbors
			for(int i = 0; i < array_len(n->outgoing_edges); i++) {
				QGEdge *e = n->outgoing_edges[i];
				seen = raxFind(visited, (unsigned char *)e->dest->alias,
						strlen(e->dest->alias));
				if(seen == raxNotFound) array_append(q, e->dest);
			}

			for(int i = 0; i < array_len(n->incoming_edges); i++) {
				QGEdge *e = n->incoming_edges[i];
				seen = raxFind(visited, (unsigned char *)e->src->alias,
						strlen(e->src->alias));
				if(seen == raxNotFound) array_append(q, e->src);
			}
		}

		// visited comprise the connected component defined by S
		// remove all non-reachable nodes from current connected component.
		// remove connected component from graph
		QueryGraph *cc = QueryGraph_Clone(g);
		uint node_count = QueryGraph_NodeCount(g);
		for(uint i = 0; i < node_count; i++) {
			void *reachable;
			n = g->nodes[i];
			reachable = raxFind(visited, (unsigned char *)n->alias,
					strlen(n->alias));

			// if node is reachable, which means it is part of the
			// connected component, then remove it from the graph,
			// otherwise, node isn't reachable, not part of the
			// connected component
			if(reachable != raxNotFound) {
				n = QueryGraph_RemoveNode(g, n);
				QGNode_Free(n);
			} else {
				n = QueryGraph_GetNodeByAlias(cc, n->alias);
				QueryGraph_RemoveNode(cc, n);
				QGNode_Free(n);
			}
		}

		array_append(connected_components, cc);

		// clear visited dict for next iteration
		raxFree(visited);

		// exit when graph is empty
		if(QueryGraph_NodeCount(g) == 0) break;
	}

	array_free(q);
	QueryGraph_Free(g);

	return connected_components;
}

uint QueryGraph_NodeCount
(
	const QueryGraph *qg
) {
	return array_len(qg->nodes);
}

// retrieve the number of edges in a QueryGraph
uint QueryGraph_EdgeCount
(
	const QueryGraph *qg
) {
	return array_len(qg->edges);
}

GrB_Matrix QueryGraph_MatrixRepresentation
(
	const QueryGraph *qg
) {
	ASSERT(qg != NULL);

	// make a clone of the given graph as we're about to modify it
	QueryGraph *qg_clone = QueryGraph_Clone(qg);

	// give an ID for each node, abuse of `labelID`
	uint node_count = QueryGraph_NodeCount(qg_clone);
	for(uint i = 0; i < node_count; i++) {
		QGNode *n = qg_clone->nodes[i];
		if(QGNode_LabelCount(n) == 0) QGNode_AddLabel(n, "", i);
		else n->labelsID[0] = i;
	}

	GrB_Matrix m;   // matrix representation of QueryGraph
	GrB_Info res = GrB_Matrix_new(&m, GrB_BOOL, node_count, node_count);
	UNUSED(res);
	ASSERT(res == GrB_SUCCESS);

	// build matrix representation of query graph
	for(uint i = 0; i < node_count; i++) {
		const QGNode *n = qg_clone->nodes[i];
		GrB_Index src = QGNode_GetLabelID(n, 0);
		uint outgoing_degree = QGNode_OutgoingDegree(n);

		for(uint j = 0; j < outgoing_degree; j++) {
			const QGEdge *e = n->outgoing_edges[j];
			GrB_Index dest = QGNode_GetLabelID(e->dest, 0);

			// populate `m`
			res = GrB_Matrix_setElement_BOOL(m, true, src, dest);
			ASSERT(res == GrB_SUCCESS);
		}
	}

	QueryGraph_Free(qg_clone);
	return m;
}

void QueryGraph_Print
(
	const QueryGraph *qg
) {
	char *buff = calloc(1024, sizeof(char));

	uint node_count = QueryGraph_NodeCount(qg);
	uint edge_count = QueryGraph_EdgeCount(qg);

	for(int i = 0; i < node_count; i++) {
		QGNode *n = qg->nodes[i];
		if(QGNode_IncomeDegree(n) + QGNode_OutgoingDegree(n) == 0) {
			// floating node
			int rc __attribute__((unused));
			rc = asprintf(&buff, "%s%s;\n", buff, n->alias);
		}
	}

	for(int i = 0; i < edge_count; i++) {
		QGEdge *e = qg->edges[i];
        int rc __attribute__((unused));
		rc = asprintf(&buff, "%s%s -> %s;\n", buff, e->src->alias, e->dest->alias);
	}

	printf("%s\n", buff);
	free(buff);
}

// frees entire graph
void QueryGraph_Free
(
	QueryGraph *qg
) {
	if(qg == NULL) return;

	// free QueryGraph nodes
	uint nodeCount = QueryGraph_NodeCount(qg);
	for(uint i = 0; i < nodeCount; i++) {
		QGNode_Free(qg->nodes[i]);
	}

	// free QueryGraph edges
	uint edgeCount = QueryGraph_EdgeCount(qg);
	for(uint i = 0; i < edgeCount; i++) {
		QGEdge_Free(qg->edges[i]);
	}

	array_free(qg->nodes);
	array_free(qg->edges);
	rm_free(qg);
}

