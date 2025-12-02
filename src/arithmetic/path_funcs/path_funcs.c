/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "path_funcs.h"
#include "LAGraphX.h"
#include "../func_desc.h"
#include "../../ast/ast.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../util/rmalloc.h"
#include "../../configuration/config.h"
#include "../../procedures/utility/internal.h"
#include "../../datatypes/path/sipath_builder.h"

// creates a path from a given sequence of graph entities
// the first argument is the ast node represents the path
// arguments 2...n are the sequence of graph entities combines the path
// the sequence is always in odd length and defined as:
// odd indices members are always representing the value of a single node
// even indices members are either representing the value of a single edge,
// or an sipath, in case of variable length traversal
SIValue AR_TOPATH
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	const cypher_astnode_t *ast_path = argv[0].ptrval;
	uint nelements = cypher_ast_pattern_path_nelements(ast_path);
	ASSERT(argc == (nelements + 1));

	uint n = 0;
	uint path_elements = 0;
	SIValue arr[nelements];
	// collect path elements
	// and calculate how much space needed for the returned path
	for(uint i = 0; i < nelements; i++) {
		SIValue element = argv[i + 1];
		if(SI_TYPE(element) == T_NULL) {
			// if any element of the path does not exist
			// the entire path is invalid
			return SI_NullVal();
		}

		if(i % 2 == 0) {
			path_elements++;
		} else {
			// edges and paths are in odd positions
			// element type can be either edge, or path
			if(SI_TYPE(element) == T_EDGE) {
				path_elements++;
			} else { // if element is not an edge, it is a path
				// path with 0 edges should not be appended
				// their source and destination nodes are the same
				// and the source node already appended.
				size_t len = SIPath_Length(element);
				if(len == 0) {
					i++;
					continue;
				}
				// len - 1 nodes and len edges from this path
				// will be added to the returned path 
				path_elements += len * 2 - 1;
			}
		}
		arr[n++] = element;
	}

	SIValue path = SIPathBuilder_New(path_elements);
	for(uint i = 0; i < n; i++) {
		SIValue element = arr[i];

		if(i % 2 == 0) {
			// nodes are in even position
			SIPathBuilder_AppendNode(path, element);
		} else {
			// edges and paths are in odd positions
			const cypher_astnode_t *ast_rel_pattern = cypher_ast_pattern_path_get_element(ast_path, i);
			bool RTL_pattern = cypher_ast_rel_pattern_get_direction(ast_rel_pattern) == CYPHER_REL_INBOUND;
			// element type can be either edge, or path
			if(SI_TYPE(element) == T_EDGE) {
				SIPathBuilder_AppendEdge(path, element, RTL_pattern);
			} else { // if element is not an edge, it is a path
				// the build should continue to the next edge/path value
				// consider the following query
				// for the graph in the form of (:L1)-[]->(:L2):
				// "MATCH p=(a:L1)-[*0..]->(b:L1)-[]->(c:L2)"
				// the path build should
				// return a path with with the relevant entities.
				SIPathBuilder_AppendPath(path, element, RTL_pattern);
			}
		}
	}

	return path;
}

// routine for freeing a shortest path function's private data
void ShortestPath_Free
(
	void *ctx_ptr
) {
	ShortestPathCtx *ctx = (ShortestPathCtx*)ctx_ptr;

	if(ctx->reltypes)      array_free(ctx->reltypes);
	if(ctx->reltype_names) array_free(ctx->reltype_names);

	rm_free(ctx);
}

// routine for cloning a shortest path function's private data
void *ShortestPath_Clone
(
	void *orig
) {
	ShortestPathCtx *ctx = (ShortestPathCtx*)orig;

	// allocate space for the clone
	ShortestPathCtx *ctx_clone = rm_calloc(1, sizeof(ShortestPathCtx));

	ctx_clone->minHops = ctx->minHops;
	ctx_clone->maxHops = ctx->maxHops;

	// clone reltype names but not IDs, to avoid
	// a scenario in which a traversed type is created after the
	// shortestPath query is cached
	ctx_clone->reltype_count = ctx->reltype_count;

	if(ctx->reltype_names) {
		array_clone(ctx_clone->reltype_names, ctx->reltype_names);
	}

	return ctx_clone;
}

SIValue AR_SHORTEST_PATH
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	// expecting both source node and destination node as arguments
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	if(SI_TYPE(argv[1]) == T_NULL) return SI_NullVal();

	Node *srcNode  = argv[0].ptrval;
	Node *destNode = argv[1].ptrval;

	ShortestPathCtx  *ctx     = (ShortestPathCtx*)private_data;
	GrB_Index         src_id  = ENTITY_GET_ID(srcNode);
	GrB_Index         dest_id = ENTITY_GET_ID(destNode);

	GrB_Info info;

	GrB_Matrix M     = NULL;  // adjacency matrix
	GrB_Vector V     = NULL;  // vector of results
	GrB_Vector PI    = NULL;  // vector backtracking results to their parents
	Edge *edges      = NULL;
	GraphContext *gc = QueryCtx_GetGraphCtx();

	// BFS max depth
	int64_t max_level = (ctx->maxHops == EDGE_LENGTH_INF) ? -1 : ctx->maxHops;

	if(ctx->reltype_count > 0) {
		// retrieve IDs of traversed relationship types
		ctx->reltypes = array_new(RelationID, ctx->reltype_count);

		for(uint i = 0; i < ctx->reltype_count; i++) {
			Schema *s = GraphContext_GetSchema(gc, ctx->reltype_names[i],
					SCHEMA_EDGE);
			// skip missing schemas
			if(s != NULL) {
				array_append(ctx->reltypes, Schema_GetID(s));
			}
		}

		// update the reltype count
		// as it may have changed due to missing schemas
		ctx->reltype_count = array_len(ctx->reltypes);
	}

	// build the adjacency matrix BFS will be executed on
	if(ctx->reltype_names != NULL && ctx->reltype_count == 0) {
		// if edge types were specified but none were valid,
		// use the zero matrix
		info = Delta_Matrix_export(&M, Graph_GetZeroMatrix(gc->g), GrB_BOOL);
		ASSERT(info == GrB_SUCCESS);
	} else {
		info = Build_Matrix(&M, NULL, gc->g, NULL, 0, ctx->reltypes,
				ctx->reltype_count, false, false);
		ASSERT(info == GrB_SUCCESS);
	}

	//--------------------------------------------------------------------------
	// invoke the BFS algorithm
	//--------------------------------------------------------------------------

	char msg[LAGRAPH_MSG_LEN];
	LAGraph_Graph G;

	info = LAGraph_New(&G, &M, LAGraph_ADJACENCY_DIRECTED, msg);
	ASSERT(info == GrB_SUCCESS);

	info = LAGr_BreadthFirstSearch_Extended(&V, &PI, G, src_id, max_level,
			dest_id, false, msg);
	ASSERT(info == GrB_SUCCESS);

	ASSERT(V    != NULL);
	ASSERT(PI   != NULL);
	ASSERT(info == GrB_SUCCESS);

	info = LAGraph_Delete(&G, msg);
	ASSERT(info == GrB_SUCCESS);

	SIValue p = SI_NullVal();

	// the length of the path is equal to the level of the destination node
	GrB_Index path_len;
	info = GrB_Vector_extractElement(&path_len, V, dest_id);
	if(info == GrB_NO_VALUE) goto cleanup; // no path found

	// only emit a path with no edges if minHops is 0
	if(path_len == 0 && ctx->minHops != 0) goto cleanup;

	// build path in reverse, starting by appending the destination node
	// the path is built in reverse because we have the destination's parent
	// in the PI array, and can use this to backtrack until we reach the source
	p = SIPathBuilder_New(path_len);
	SIPathBuilder_AppendNode(p, SI_Node(destNode));

	edges = array_new(Edge, 1);

	NodeID id = destNode->id;
	for(uint i = 0; i < path_len; i++) {
		array_clear(edges);
		GrB_Index parent_id;

		// find the parent of the reached node
		info = GrB_Vector_extractElement(&parent_id, PI, id);
		ASSERT(info == GrB_SUCCESS);

		// retrieve edges connecting the parent node to the current node
		if(ctx->reltype_count == 0) {
			Graph_GetEdgesConnectingNodes(gc->g, parent_id, id,
					GRAPH_NO_RELATION, &edges);
		} else {
			for(uint j = 0; j < ctx->reltype_count; j++) {
				Graph_GetEdgesConnectingNodes(gc->g, parent_id, id,
						ctx->reltypes[j], &edges);
				if(array_len(edges) > 0) break;
			}
		}

		ASSERT(array_len(edges) > 0);

		// append the edge to the path
		SIPathBuilder_AppendEdge(p, SI_Edge(&edges[0]), false);

		// append the reached node to the path
		id = Edge_GetSrcNodeID(&edges[0]);
		Node n = GE_NEW_NODE();
		Graph_GetNode(gc->g, id, &n);
		SIPathBuilder_AppendNode(p, SI_Node(&n));
	}

	// reverse the path so it starts at the source
	Path_Reverse(p.ptrval);

cleanup:
	if(V)     GrB_free(&V);
	if(PI)    GrB_free(&PI);
	if(edges) array_free(edges);

	return p;
}

SIValue AR_PATH_NODES
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	return SIPath_Nodes(argv[0]);
}

SIValue AR_PATH_RELATIONSHIPS
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	return SIPath_Relationships(argv[0]);
}

SIValue AR_PATH_LENGTH
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	return SI_LongVal(SIPath_Length(argv[0]));
}

void Register_PathFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 2);
	array_append(types, T_PTR);
	array_append(types, T_NULL | T_NODE | T_EDGE | T_PATH);
	ret_type = T_PATH | T_NULL;
	func_desc = AR_FuncDescNew("topath", AR_TOPATH, 1, VAR_ARG_LEN, types,
			ret_type, true, false, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 3);
	array_append(types, T_NULL | T_NODE);
	array_append(types, T_NULL | T_NODE);
	ret_type = T_PATH | T_NULL;
	func_desc = AR_FuncDescNew("shortestpath", AR_SHORTEST_PATH, 2, 2, types,
			ret_type, true, false, true);
	AR_SetPrivateDataRoutines(func_desc, ShortestPath_Free, ShortestPath_Clone);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_PATH);
	ret_type = T_ARRAY | T_NULL;
	func_desc = AR_FuncDescNew("nodes", AR_PATH_NODES, 1, 1, types, ret_type,
			false, false, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_PATH);
	ret_type = T_ARRAY | T_NULL;
	func_desc = AR_FuncDescNew("relationships", AR_PATH_RELATIONSHIPS, 1, 1,
			types, ret_type, false, false, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_PATH);
	ret_type = T_INT64 | T_NULL;
	func_desc = AR_FuncDescNew("length", AR_PATH_LENGTH, 1, 1, types, ret_type,
			false, false, true);
	AR_RegFunc(func_desc);
}

