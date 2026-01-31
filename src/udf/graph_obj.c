/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "udf_ctx.h"
#include "classes.h"
#include "traverse.h"
#include "repository.h"
#include "../query_ctx.h"
#include "../arithmetic/func_desc.h"

//------------------------------------------------------------------------------
// external class IDs
//------------------------------------------------------------------------------

extern JSClassID js_node_class_id;  // JS Node class

//------------------------------------------------------------------------------
// graph.traverse implementations
//------------------------------------------------------------------------------

// non-runtime implementation of `graph.traverse`
// JS call: graph.traverse();
static JSValue non_runtime_traverse
(
	JSContext *js_ctx,      // JavaScript context
	JSValueConst this_val,  // 'this' value passed by the caller
	int argc,               // number of arguments
	JSValueConst *argv      // function arguments
) {
	ASSERT (argv   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	return JS_ThrowTypeError (js_ctx,
			"graph.traverse shouldn't be called in a global context") ;
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
	JSValue func_obj = JS_NewCFunction (js_ctx, graph_traverse, "traverse", 2) ;

    int def_res = JS_DefinePropertyValueStr (js_ctx, graph_obj, "traverse",
			func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;
	ASSERT (def_res >= 0) ;

    // expose the namespace globally as "graph"
    JSValue global_obj = JS_GetGlobalObject (js_ctx) ;

    JS_SetPropertyStr (js_ctx, global_obj, "graph", graph_obj) ;

    // free what we got from GetGlobalObject
    JS_FreeValue (js_ctx, global_obj) ;
}

// set the implementation of the `graph.traverse` function
void UDF_SetGraphTraverseImpl
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
    JSValue func_obj = JS_UNDEFINED;
	if (mode == UDF_FUNC_REG_MODE_LOCAL) {
		func_obj = JS_NewCFunction (js_ctx, graph_traverse, "traverse", 2) ;
	} else {
		func_obj = JS_NewCFunction (js_ctx, non_runtime_traverse, "traverse", 2) ;
	}

	ASSERT (JS_IsFunction (js_ctx, func_obj)) ;

    // define property with explicit flags (writable, configurable)
    int def_res = JS_DefinePropertyValueStr (js_ctx, graph_obj, "traverse",
			func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;

	ASSERT (def_res >= 0) ;

    // clean up
    JS_FreeValue (js_ctx, global_obj) ;
    JS_FreeValue (js_ctx, graph_obj) ;
}

