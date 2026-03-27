/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "udf_ctx.h"
#include "classes.h"
#include "traverse.h"
#include "GraphBLAS.h"
#include "repository.h"
#include "../query_ctx.h"
#include "../arithmetic/func_desc.h"

//------------------------------------------------------------------------------
// external class IDs
//------------------------------------------------------------------------------

extern JSClassID js_node_class_id;  // JS Node class

// return a node iterator over the specified label
// usage: const it = graph.iterateNodes('Country');
static JSValue graph_iterate_nodes
(
	JSContext *js_ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
	const Graph *g = QueryCtx_GetGraph () ;
	ASSERT (g != NULL) ;

    // validate arguments: ensure a label string is provided
    if (argc < 1 || !JS_IsString (argv [0])) {
        return JS_ThrowTypeError (js_ctx,
				"iterateNodes expects a label string as the first argument") ;
    }

    const char *label = JS_ToCString (js_ctx, argv [0]) ;
    if (!label) {
		return JS_EXCEPTION ;
	}

    // get the label matrix from the graph
	Delta_Matrix L = NULL ;
	GraphContext *gc = QueryCtx_GetGraphCtx () ;
	Schema *s = GraphContext_GetSchema (gc, label, SCHEMA_NODE) ;

	if (s == NULL) {
		L = Graph_GetZeroMatrix (g) ;
	} else {
		L = Graph_GetLabelMatrix (g, Schema_GetID (s)) ;
	}
	ASSERT (L != NULL) ;

    // free the C string immediately after use
    JS_FreeCString (js_ctx, label) ;

    // create the DeltaMatrix Iterator
    Delta_MatrixTupleIter *it = rm_malloc (sizeof (Delta_MatrixTupleIter)) ;
    Delta_MatrixTupleIter_attach (it, L) ;

    // wrap it in our custom NodeIterator JS Object
    // we pass 'g' as well so the iterator knows which graph to query for node data
    return UDF_CreateNodeIterator (js_ctx, g, it) ;
}

// return an edge iterator over the specified relationship-type
// usage: const it = graph.iterateEdges('FOLLOWS');
static JSValue graph_iterate_edges
(
	JSContext *js_ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
	const Graph *g = QueryCtx_GetGraph () ;
	ASSERT (g != NULL) ;

	// validate arguments: ensure a label string is provided
	if (argc < 1 || !JS_IsString (argv [0])) {
		return JS_ThrowTypeError (js_ctx,
				"iterateEdges expects a relationship-type string as the first argument") ;
	}

	const char *rel = JS_ToCString (js_ctx, argv [0]) ;
	if (!rel) {
		return JS_EXCEPTION ;
	}

	// get the label matrix from the graph
	Tensor R = NULL ;
	RelationID rel_id = GRAPH_UNKNOWN_RELATION ;
	GraphContext *gc = QueryCtx_GetGraphCtx () ;
	Schema *s = GraphContext_GetSchema (gc, rel, SCHEMA_EDGE) ;

	if (s == NULL) {
		R = Graph_GetZeroMatrix (g) ;
	} else {
		rel_id = Schema_GetID (s) ;
		R = Graph_GetRelationMatrix (g, rel_id, false) ;
	}
	ASSERT (R != NULL) ;

	// free the C string immediately after use
	JS_FreeCString (js_ctx, rel) ;

	// create the Tensor Iterator
	TensorIterator *it = rm_malloc (sizeof (TensorIterator)) ;
	TensorIterator_ScanRange (it, R, 0, UINT64_MAX, false) ;

	// wrap it in our custom EdgeIterator JS Object
	// we pass 'g' as well so the iterator knows which graph to query for edge data
	return UDF_CreateEdgeIterator (js_ctx, g, rel_id, it) ;
}

//------------------------------------------------------------------------------
// graph.getNodeById implementations
//------------------------------------------------------------------------------

// retrieves a node from the graph by its integer ID
// JS Usage: const node = graph.getNodeById(nodeId);
// argv[0] the integer ID of the node
// return JSValue The Node object if found, JS_NULL if not found,
// or a TypeError if arguments are invalid
static JSValue graph_get_node_by_id
(
	JSContext *ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
    // validation: ensure an argument is provided
    if (argc < 1) {
        return JS_ThrowTypeError (ctx, "getNodeById: At least one "
				"argument (ID) expected") ;
    }

	int64_t node_id = -1 ;
	JSValueConst val = argv [0] ;

	// check if it's a BigInt (common for 64-bit IDs in JS)
	if (JS_IsBigInt (ctx, val)) {
		if (JS_ToInt64 (ctx, &node_id, val)) {
			return JS_EXCEPTION ;
		}
	}

	// check if it's a standard JS Number
	else if (JS_IsNumber (val)) {
		double d;
		JS_ToFloat64 (ctx, &d, val) ;

		// ensure the number has no fractional part and is within uint64 range
		// fmod(d, 1.0) != 0 checks if it's a "true" float (e.g., 1.5)
		if (d < 0 || fmod (d, 1.0) != 0) {
			return JS_ThrowTypeError (ctx,
					"getNodeById: ID must be a positive integer") ;
		}

		// safe to convert now that we've validated it's a whole number
		JS_ToInt64 (ctx, &node_id, val) ;
	}

	else {
		return JS_ThrowTypeError (ctx,
				"getNodeById: Argument must be a positive integer") ;
	}

	// final safety check for the logic
	if (node_id < 0) {
		return JS_ThrowTypeError (ctx,
				"getNodeById: ID must be a positive integer") ;
	}

	const Graph *graph = QueryCtx_GetGraph () ;
	ASSERT (graph != NULL) ;

    // fetch the node from the underlying C engine
    Node node ;
    if (Graph_GetNode (graph, (NodeID) node_id, &node)) {
        // successfully found: return the JS wrapper for the Node
        return UDF_CreateNode (ctx, &node) ;
    }

    // fallback: Node not found
    return JS_NULL ;
}

//------------------------------------------------------------------------------
// graph.traverse implementations
//------------------------------------------------------------------------------

// non-runtime implementation of `graph.*`
static JSValue non_runtime_function
(
	JSContext *js_ctx,      // JavaScript context
	JSValueConst this_val,  // 'this' value passed by the caller
	int argc,               // number of arguments
	JSValueConst *argv      // function arguments
) {
	ASSERT (argv   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	return JS_ThrowTypeError (js_ctx,
			"graph API shouldn't be called in a global context") ;
}

// traverse from multiple sources
//
// example:
// let nodes = graph.traverse([a,b]);
// nodes[0] contains a's neighbors
// nodes[1] contains b's neighbors
//
// accepts an optional config map:
// {
//   direction:  string   - 'incoming' / 'outgoing' / 'both',
//   types:      string[] - ['KNOWS', 'WORKS_AT'],
//   labels:     string[] - ['Person', 'City'],
//   distance:   number   - traversal depth,
//   returnType: string   - 'nodes' / 'edges'
// }
//
// all fields in map are optional
//
// returns an array of array of Nodes
static JSValue graph_traverse
(
	JSContext *js_ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
	ASSERT (js_ctx != NULL) ;

	if (argc == 0) {
		return JS_ThrowTypeError (js_ctx,
				"graph.traverse requires at least one argument") ;
	}

	//--------------------------------------------------------------------------
	// extract nodes
	//--------------------------------------------------------------------------

	// expecting an array of nodes
	JSValueConst js_arr = argv[0] ;
	if (!JS_IsArray (js_ctx, js_arr)) {
		return JS_ThrowTypeError (js_ctx,
				"graph.traverse first argument should be an array of nodes") ;
	}

	// process array
	uint32_t source_count = 0 ;
	JSValue len_val = JS_GetPropertyStr (js_ctx, js_arr, "length") ;
	JS_ToUint32  (js_ctx, &source_count, len_val) ;
	JS_FreeValue (js_ctx, len_val) ;

	if (source_count == 0) {
		return JS_NewArray (js_ctx) ;
	}

	// extract each node
	EntityID *sources = rm_malloc (sizeof (EntityID) * source_count) ;
	for (uint32_t i = 0 ; i < source_count ; i++) {
		JSValue   elem = JS_GetPropertyUint32 (js_ctx, js_arr, i) ;
		JSClassID cid  = JS_GetClassID (elem) ;

		if (cid != js_node_class_id) {
			rm_free (sources) ;
			JS_FreeValue (js_ctx, elem) ;
			return JS_ThrowTypeError (js_ctx,
					"graph.traverse first argument should be an array of nodes") ;
		}

		Node *node = JS_GetOpaque2 (js_ctx, elem, js_node_class_id) ;
		if (!node) {
			rm_free (sources) ;
			JS_FreeValue (js_ctx, elem) ;
			return JS_EXCEPTION ;
		}

		sources[i] = ENTITY_GET_ID (node) ;
		JS_FreeValue (js_ctx, elem) ;
	}

	//--------------------------------------------------------------------------
	// default config
	//--------------------------------------------------------------------------

	uint distance            = 1 ;                        // direct neighbors
	char **labels            = NULL ;                     // neighbors labels
	char **rel_types         = NULL ;                     // edge types
	GRAPH_EDGE_DIR dir       = GRAPH_EDGE_DIR_OUTGOING ;  // edge direction
	GraphEntityType ret_type = GETYPE_NODE ;              // returned type

	//----------------------------------------------------------------------
	// parse the provided options object
	//----------------------------------------------------------------------

	if (argc > 1) {
		const char *err_msg = NULL ;
		if (!traverse_init_config (js_ctx, argc - 1, argv + 1, &distance,
					&labels, &rel_types, &dir, &ret_type, &err_msg)) {
			// parsing config map failed
			ASSERT (err_msg != NULL) ;
			rm_free (sources) ;
			return JS_ThrowTypeError (js_ctx, "%s", err_msg) ;
		}
	}

	//--------------------------------------------------------------------------
	// traverse
	//--------------------------------------------------------------------------

	// neighbors is an array with a single element:
	// an array of all reachable entities (Nodes / Edges)
	uint *neighbors_count = rm_malloc (sizeof (uint) * source_count)  ;

	GraphEntity **neighbors = traverse (neighbors_count, sources, source_count,
			distance, (const char **)labels, (const char **)rel_types, dir,
			ret_type) ;
	ASSERT (neighbors != NULL) ;

	//--------------------------------------------------------------------------
	// compose output
	//--------------------------------------------------------------------------

	JSValue output = JS_NewArray (js_ctx) ;
	for (uint i = 0 ; i < source_count ; i++) {
		// populate output javascript array
		JSValue js_neighbors = JS_NewArray (js_ctx) ;

		if (ret_type == GETYPE_NODE) {
			// add node to neighbors
			Node *nodes = (Node*) neighbors[i] ;

			for (uint j = 0; j < neighbors_count[i] ; j++) {
				JS_SetPropertyUint32 (js_ctx, js_neighbors, j,
						UDF_CreateNode (js_ctx, nodes + j)) ;
			}
			rm_free (nodes) ;
		}
		else {
			// add edge to neighbors
			Edge *edges = (Edge*) neighbors[i] ;

			for (uint j = 0; j < neighbors_count[i] ; j++) {
				JS_SetPropertyUint32 (js_ctx, js_neighbors, j,
						UDF_CreateEdge (js_ctx, edges + j)) ;
			}
			array_free (edges) ;
		}

		JS_SetPropertyUint32 (js_ctx, output, i, js_neighbors) ;
	}

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	rm_free (sources) ;
	rm_free (neighbors) ;
	rm_free (neighbors_count) ;

	if (labels != NULL) {
		for (int i = 0; i < array_len (labels) ; i++) {
			free (labels[i]) ;
		}
		array_free (labels) ;
	}

	if (rel_types != NULL) {
		for (int i = 0; i < array_len (rel_types) ; i++) {
			free (rel_types[i]) ;
		}
		array_free (rel_types) ;
	}

	return output ;
}

//------------------------------------------------------------------------------
// graph object setup
//------------------------------------------------------------------------------

// register the global graph object in the given QuickJS context
void UDF_RegisterGraphObject
(
	JSContext *js_ctx  // the QuickJS context in which to register the object
) {
	ASSERT (js_ctx != NULL) ;

    // create a plain namespace object: const graph = {};
    JSValue graph_obj = JS_NewObject (js_ctx) ;

	// register graph.traverse
//	JSValue func_obj = JS_NewCFunction (js_ctx, graph_traverse, "traverse", 2) ;
//
//    int def_res = JS_DefinePropertyValueStr (js_ctx, graph_obj, "traverse",
//			func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;
//	ASSERT (def_res >= 0) ;

    // expose the namespace globally as "graph"
    JSValue global_obj = JS_GetGlobalObject (js_ctx) ;

    JS_SetPropertyStr (js_ctx, global_obj, "graph", graph_obj) ;

    // free what we got from GetGlobalObject
    JS_FreeValue (js_ctx, global_obj) ;
}

// set the implementation of the `graph.traverse` function
void UDF_SetGraphAPI
(
	JSContext *js_ctx,              // the QuickJS context
	UDF_JSCtxRegisterFuncMode mode  // the registration mode
) {
	ASSERT(js_ctx != NULL);

    // get global.graph
    JSValue global_obj = JS_GetGlobalObject (js_ctx) ;
    JSValue graph_obj = JS_GetPropertyStr  (js_ctx, global_obj, "graph") ;

	ASSERT (JS_IsObject (graph_obj));

	// pick implementation
    JSValue traverse_func_obj      = JS_UNDEFINED ;
    JSValue get_node_func_obj      = JS_UNDEFINED ;
	JSValue iterate_nodes_func_obj = JS_UNDEFINED ;
	JSValue iterate_edges_func_obj = JS_UNDEFINED ;

	if (mode == UDF_FUNC_REG_MODE_LOCAL) {
		traverse_func_obj =
			JS_NewCFunction (js_ctx, graph_traverse,       "traverse",     2) ;
		get_node_func_obj =
			JS_NewCFunction (js_ctx, graph_get_node_by_id, "getNodeById",  1) ;
		iterate_nodes_func_obj =
			JS_NewCFunction (js_ctx, graph_iterate_nodes,  "iterateNodes", 1) ;
		iterate_edges_func_obj =
			JS_NewCFunction (js_ctx, graph_iterate_edges,  "iterateEdges", 1) ;
	} else {
		traverse_func_obj =
			JS_NewCFunction (js_ctx, non_runtime_function, "traverse",     2) ;
		get_node_func_obj =
			JS_NewCFunction (js_ctx, non_runtime_function, "getNodeById",  1) ;
		iterate_nodes_func_obj =
			JS_NewCFunction (js_ctx, non_runtime_function, "iterateNodes", 1) ;
		iterate_edges_func_obj =
			JS_NewCFunction (js_ctx, non_runtime_function, "iterateEdges", 1) ;
	}

	ASSERT (JS_IsFunction (js_ctx, traverse_func_obj)) ;
	ASSERT (JS_IsFunction (js_ctx, get_node_func_obj)) ;
	ASSERT (JS_IsFunction (js_ctx, iterate_nodes_func_obj)) ;
	ASSERT (JS_IsFunction (js_ctx, iterate_edges_func_obj)) ;

    // define property with explicit flags (writable, configurable)
    int def_res ;

    def_res = JS_DefinePropertyValueStr (js_ctx, graph_obj, "traverse",
			traverse_func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;
	ASSERT (def_res >= 0) ;

    def_res = JS_DefinePropertyValueStr (js_ctx, graph_obj, "getNodeById",
			get_node_func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;
	ASSERT (def_res >= 0) ;

    def_res = JS_DefinePropertyValueStr (js_ctx, graph_obj, "iterateNodes",
			iterate_nodes_func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;
	ASSERT (def_res >= 0) ;

    def_res = JS_DefinePropertyValueStr (js_ctx, graph_obj, "iterateEdges",
			iterate_edges_func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;
	ASSERT (def_res >= 0) ;

    // clean up
    JS_FreeValue (js_ctx, global_obj) ;
    JS_FreeValue (js_ctx, graph_obj) ;
}

