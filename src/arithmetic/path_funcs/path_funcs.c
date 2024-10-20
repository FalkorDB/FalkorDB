/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "path_funcs.h"
#include "../func_desc.h"
#include "../../ast/ast.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../util/rmalloc.h"
#include "../../configuration/config.h"
#include "../../datatypes/path/sipath_builder.h"
#include "../../algorithms/LAGraph/LAGraph_bfs.h"

/* Creates a path from a given sequence of graph entities.
 * The first argument is the ast node represents the path.
 * Arguments 2...n are the sequence of graph entities combines the path.
 * The sequence is always in odd length and defined as:
 * Odd indices members are always representing the value of a single node.
 * Even indices members are either representing the value of a single edge,
 * or an sipath, in case of variable length traversal. */
SIValue AR_TOPATH(SIValue *argv, int argc, void *private_data) {
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

// Routine for freeing a shortest path function's private data.
void ShortestPath_Free(void *ctx_ptr) {
	ShortestPathCtx *ctx = ctx_ptr;
	if(ctx->reltypes) array_free(ctx->reltypes);
	if(ctx->reltype_names) array_free(ctx->reltype_names);
	if(ctx->free_matrices) {
		GrB_free(&ctx->R);
	}
	rm_free(ctx);
}

// Routine for cloning a shortest path function's private data.
void *ShortestPath_Clone(void *orig) {
	ShortestPathCtx *ctx = orig;
	// Allocate space for the clone
	ShortestPathCtx *ctx_clone = rm_malloc(sizeof(ShortestPathCtx));
	ctx_clone->minHops = ctx->minHops;
	ctx_clone->maxHops = ctx->maxHops;
	/* Clone reltype names but not IDs, to avoid
	 * a scenario in which a traversed type is created after the
	 * shortestPath query is cached. */
	ctx_clone->reltype_count = ctx->reltype_count;
	ctx_clone->reltypes = NULL;
	if(ctx->reltype_names) array_clone(ctx_clone->reltype_names, ctx->reltype_names);
	else ctx_clone->reltype_names = NULL;
	// Do not clone matrix data
	ctx_clone->R = GrB_NULL;
	ctx_clone->free_matrices = false;

	return ctx_clone;
}

SIValue AR_SHORTEST_PATH(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	if(SI_TYPE(argv[1]) == T_NULL) return SI_NullVal();

	Node             *srcNode   =  argv[0].ptrval;
	Node             *destNode  =  argv[1].ptrval;
	ShortestPathCtx  *ctx       =  private_data;
	GrB_Index src_id            =  ENTITY_GET_ID(srcNode);
	GrB_Index dest_id           =  ENTITY_GET_ID(destNode);

	GrB_Info res;
	UNUSED(res);
	Edge *edges = NULL;
	GrB_Vector V = GrB_NULL;  // vector of results
	GrB_Vector PI = GrB_NULL; // vector backtracking results to their parents
	GraphContext *gc = QueryCtx_GetGraphCtx();

	GrB_Index max_level = (ctx->maxHops == EDGE_LENGTH_INF) ? 0 : ctx->maxHops;

	if(ctx->R == GrB_NULL) {
		// First invocation, initialize unset context members.
		if(ctx->reltype_count > 0) {
			// Retrieve IDs of traversed relationship types.
			ctx->reltypes = array_new(int, ctx->reltype_count);
			for(uint i = 0; i < ctx->reltype_count; i ++) {
				Schema *s = GraphContext_GetSchema(gc, ctx->reltype_names[i], SCHEMA_EDGE);
				// Skip missing schemas
				if(s) array_append(ctx->reltypes, Schema_GetID(s));
			}

			// Update the reltype count, as it may have changed due to missing schemas
			ctx->reltype_count = array_len(ctx->reltypes);
		}

		// Get edge matrix and transpose matrix, if available.
		if(ctx->reltypes == NULL) {
			// No edge types were specified, use the overall adjacency matrix.
			ctx->free_matrices = true;
			res = Delta_Matrix_export(&ctx->R, Graph_GetAdjacencyMatrix(gc->g,
						false));
			ASSERT(res == GrB_SUCCESS);
		} else if(ctx->reltype_count == 0) {
			// If edge types were specified but none were valid,
			// use the zero matrix
			ctx->free_matrices = true;
			res = Delta_Matrix_export(&ctx->R, Graph_GetZeroMatrix(gc->g));
			ASSERT(res == GrB_SUCCESS);
		} else if(ctx->reltype_count == 1) {
			ctx->free_matrices = true;
			res = Delta_Matrix_export(&ctx->R, Graph_GetRelationMatrix(gc->g,
						ctx->reltypes[0], false));
			ASSERT(res == GrB_SUCCESS);
		} else {
			// we have multiple edge types, combine them into a boolean matrix
			ctx->free_matrices = true;
			GrB_Index dims = Graph_RequiredMatrixDim(gc->g);
			res = GrB_Matrix_new(&ctx->R, GrB_BOOL, dims, dims);
			ASSERT(res == GrB_SUCCESS);

			for(uint i = 0; i < ctx->reltype_count; i ++) {
				GrB_Matrix adj;
				res = Delta_Matrix_export(&adj, Graph_GetRelationMatrix(gc->g,
							ctx->reltypes[i], false));
				ASSERT(res == GrB_SUCCESS);
				res = GrB_eWiseAdd(ctx->R, GrB_NULL, GrB_NULL,
						GxB_ANY_PAIR_BOOL, ctx->R, adj, GrB_NULL);
				ASSERT(res == GrB_SUCCESS);
				res = GrB_Matrix_free(&adj);
				ASSERT(res == GrB_SUCCESS);
			}

			GrB_Index nrows;
			res = GrB_Matrix_nrows(&nrows, ctx->R);
			ASSERT(res == GrB_SUCCESS);
		}
	}

	// Invoke the BFS algorithm
	res = LG_BreadthFirstSearch_SSGrB(&V, &PI, ctx->R, src_id, &dest_id,
		max_level);
	ASSERT(res == GrB_SUCCESS);
	ASSERT(V != GrB_NULL);
	ASSERT(PI != GrB_NULL);

	SIValue p = SI_NullVal();

	// The length of the path is equal to the level of the destination node
	GrB_Index path_len;
	res = GrB_Vector_extractElement(&path_len, V, dest_id);
	if(res == GrB_NO_VALUE) goto cleanup; // no path found

	// Only emit a path with no edges if minHops is 0
	if(path_len == 0 && ctx->minHops != 0) goto cleanup;

	/* Build path in reverse, starting by appending the destination node.
	 * The path is built in reverse because we have the destination's parent
	 * in the PI array, and can use this to backtrack until we reach the source. */
	p = SIPathBuilder_New(path_len);
	SIPathBuilder_AppendNode(p, SI_Node(destNode));

	edges = array_new(Edge, 1);

	NodeID id = destNode->id;
	for(uint i = 0; i < path_len; i ++) {
		array_clear(edges);
		GrB_Index parent_id;
		// Find the parent of the reached node.
		GrB_Info res = GrB_Vector_extractElement(&parent_id, PI, id);
		ASSERT(res == GrB_SUCCESS);

		// Retrieve edges connecting the parent node to the current node.
		if(ctx->reltype_count == 0) {
			Graph_GetEdgesConnectingNodes(gc->g, parent_id, id, GRAPH_NO_RELATION, &edges);
		} else {
			for(uint j = 0; j < ctx->reltype_count; j ++) {
				Graph_GetEdgesConnectingNodes(gc->g, parent_id, id, ctx->reltypes[j], &edges);
				if(array_len(edges) > 0) break;
			}
		}
		ASSERT(array_len(edges) > 0);
		// Append the edge to the path
		SIPathBuilder_AppendEdge(p, SI_Edge(&edges[0]), false);

		// Append the reached node to the path.
		id = Edge_GetSrcNodeID(&edges[0]);
		Node n = GE_NEW_NODE();
		Graph_GetNode(gc->g, id, &n);
		SIPathBuilder_AppendNode(p, SI_Node(&n));
	}

	// Reverse the path so it starts at the source
	Path_Reverse(p.ptrval);

cleanup:
	if(V) GrB_free(&V);
	if(PI) GrB_free(&PI);
	if(edges) array_free(edges);

	return p;
}

SIValue AR_PATH_NODES(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	return SIPath_Nodes(argv[0]);
}

SIValue AR_PATH_RELATIONSHIPS(SIValue *argv, int argc, void *private_data) {
	if(SI_TYPE(argv[0]) == T_NULL) return SI_NullVal();
	return SIPath_Relationships(argv[0]);
}

SIValue AR_PATH_LENGTH(SIValue *argv, int argc, void *private_data) {
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
	func_desc = AR_FuncDescNew("topath", AR_TOPATH, 1, VAR_ARG_LEN, types, ret_type, true, false, "topath");
	AR_RegFunc(func_desc);

	types = array_new(SIType, 3);
	array_append(types, T_NULL | T_NODE);
	array_append(types, T_NULL | T_NODE);
	ret_type = T_PATH | T_NULL;
	func_desc = AR_FuncDescNew("shortestpath", AR_SHORTEST_PATH, 2, 2, types, ret_type, true, false, "Return the shortest path that resolves the given pattern");
	AR_SetPrivateDataRoutines(func_desc, ShortestPath_Free, ShortestPath_Clone);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_PATH);
	ret_type = T_ARRAY | T_NULL;
	func_desc = AR_FuncDescNew("nodes", AR_PATH_NODES, 1, 1, types, ret_type, false, false, "Returns a list containing all the nodes in path\nReturns null if path evaluates to null");
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_PATH);
	ret_type = T_ARRAY | T_NULL;
	func_desc = AR_FuncDescNew("relationships", AR_PATH_RELATIONSHIPS, 1, 1, types, ret_type, false, false, "Returns a list containing all the relationships in path\nReturns null if path evaluates to null");
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_PATH);
	ret_type = T_INT64 | T_NULL;
	func_desc = AR_FuncDescNew("length", AR_PATH_LENGTH, 1, 1, types, ret_type, false, false, "Return the length (number of edges) of path\nReturns null if path evaluates to null");
	AR_RegFunc(func_desc);
}

