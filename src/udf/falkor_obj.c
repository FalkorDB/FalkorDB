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

// global UDF library name used when registering functions globally
extern const char *UDF_LIB ;

//------------------------------------------------------------------------------
// external class IDs
//------------------------------------------------------------------------------

extern JSClassID js_node_class_id;  // JS Node class

//------------------------------------------------------------------------------
// falkor.register implementations
//------------------------------------------------------------------------------

// validation-only implementation of `falkor.register`
// ensures the function signature is correct and not already defined
// but does not persist the function
// JS call: falkor.register("func_name", function);
static JSValue validate_register_udf
(
	JSContext *js_ctx,      // JavaScript context
	JSValueConst this_val,  // 'this' value passed by the caller
	int argc,               // number of arguments
	JSValueConst *argv      // function arguments
) {
	ASSERT (argv   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	if (argc != 2) {
		return JS_ThrowTypeError (js_ctx, "falkor.register() expects 2 arguments") ;
	}

	if (!JS_IsString (argv[0])) {
		return JS_ThrowTypeError (js_ctx,
				"first argument must be a string (function name)") ;
	}

	if (!JS_IsFunction (js_ctx, argv[1])) {
		return JS_ThrowTypeError (js_ctx,
				"second argument must be a function") ;
	}

	JSValue res;
	const char *func_name = JS_ToCString(js_ctx, argv[0]) ;

	//--------------------------------------------------------------------------
	// fail if UDF is a registered function
	//--------------------------------------------------------------------------

	// check both UDF repository & general functions repo
	if (UDF_RepoContainsFunc (UDF_LIB, func_name)) {
		res = JS_ThrowTypeError (js_ctx, "function: '%s.%s' already registered",
				UDF_LIB, func_name) ;

		goto cleanup ;
	}

	char *fullname = NULL ;
	asprintf (&fullname, "%s.%s", UDF_LIB, func_name) ;
	if (AR_FuncExists (fullname)) {
		res = JS_ThrowTypeError (js_ctx, "function: '%s' already registered",
				fullname) ;

		goto cleanup ;
	}

	res = JS_NewBool (js_ctx, true) ;

cleanup:
	if (fullname != NULL) {
		free (fullname) ;
	}
	JS_FreeCString (js_ctx, func_name) ;

	return res ;
}

// thread-local implementation of `falkor.register`
// functions are stored in the TLS UDF context
// JS call: falkor.register("func_name", function);
static JSValue local_register_udf
(
	JSContext *js_ctx,      // JavaScript context
	JSValueConst this_val,  // 'this' value passed by the caller
	int argc,               // number of arguments
	JSValueConst *argv      // function arguments
) {
	ASSERT (argv   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	if (argc != 2) {
		return JS_ThrowTypeError (js_ctx, "falkor.register() expects 2 arguments") ;
	}

	if (!JS_IsString (argv[0])) {
		return JS_ThrowTypeError (js_ctx,
				"first argument must be a string (function name)") ;
	}

	const char *func_name = JS_ToCString(js_ctx, argv[0]) ;

	JSValueConst func = argv[1] ;
	if (!JS_IsFunction (js_ctx, func)) {
		return JS_ThrowTypeError (js_ctx,
				"second argument must be a function") ;
	}

	// register function in TLS UDF context
	UDFCtx_RegisterFunction (JS_DupValue (js_ctx, func), func_name) ;

cleanup:
	JS_FreeCString (js_ctx, func_name) ;

	return JS_NewBool (js_ctx, true) ;
}

// global implementation of `falkor.register`
// functions are persisted in the global repository
// JS call: falkor.register("func_name", function);
static JSValue global_register_udf
(
	JSContext *js_ctx,      // JavaScript context
	JSValueConst this_val,  // 'this' value passed by the caller
	int argc,               // number of arguments
	JSValueConst *argv      // function arguments
) {
	ASSERT (argv   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	if (argc != 2) {
		return JS_ThrowTypeError (js_ctx, "falkor.register() expects 2 arguments") ;
	}

	if (!JS_IsString (argv[0])) {
		return JS_ThrowTypeError (js_ctx,
				"first argument must be a string (function name)") ;
	}

	if (!JS_IsFunction (js_ctx, argv[1])) {
		return JS_ThrowTypeError (js_ctx,
				"second argument must be a function") ;
	}

	JSValue res ;
	const char *func_name = JS_ToCString (js_ctx, argv[0]) ;

	if (!UDF_RepoRegisterFunc (UDF_LIB, func_name)) {
		res = JS_ThrowTypeError (js_ctx, "function: '%s' already registered",
				func_name) ;

		goto cleanup ;
	}

	res = JS_NewBool (js_ctx, true) ;

cleanup:
	JS_FreeCString (js_ctx, func_name) ;

	return res ;
}

// similar to console.log
// prints argument to stdout
static JSValue falkor_log
(
	JSContext *ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
    for (int i = 0 ; i < argc ; i++) {
        const char *str ;
        JSValue val = argv[i] ;

        // check if the value is an object (and not null) to use JSON stringify
        if (JS_IsObject (val) && !JS_IsNull (val) && !JS_IsFunction (ctx, val)) {
            // JS_JSONStringify(ctx, value, replacer, space0)
            // using a space value of 2 for "pretty-printing"
            JSValue json_str_val =
				JS_JSONStringify(ctx, val, JS_UNDEFINED, JS_NewInt32(ctx, 2));

            if (JS_IsException (json_str_val)) {
                // this usually happens with circular references
                str = "[Circular or Un-stringifiable Object]" ;
            } else {
                str = JS_ToCString (ctx, json_str_val) ;
                JS_FreeValue (ctx, json_str_val) ;
            }
        } else {
            // For strings, numbers, booleans, null, and undefined
            str = JS_ToCString (ctx, val) ;
        }

        if (!str) {
            return JS_EXCEPTION ;
        }

        printf ("%s%s", str, (i < argc - 1) ? " " : "") ;

        // only free the string if it was successfully allocated by QuickJS
        // (avoid freeing the static fallback string used in the exception check)
        if (str[0] != '[' || str[1] != 'C') {
            JS_FreeCString (ctx, str) ;
        }
    }

    printf ("\n") ;
    return JS_UNDEFINED ;
}

// traverse from multiple sources
//
// example:
// let nodes = falkor.traverse([a,b]);
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
static JSValue falkor_traverse
(
	JSContext *js_ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
	ASSERT (js_ctx != NULL) ;

	if (argc == 0) {
		return JS_ThrowTypeError (js_ctx,
				"falkor.traverse requires at least one argument") ;
	}

	//--------------------------------------------------------------------------
	// extract nodes
	//--------------------------------------------------------------------------

	// expecting an array of nodes
	JSValueConst js_arr = argv[0] ;
	if (!JS_IsArray (js_ctx, js_arr)) {
		return JS_ThrowTypeError (js_ctx,
				"falkor.traverse first argument should be an array of nodes") ;
	}

	// process array
	uint32_t source_count = 0 ;
	JSValue len_val = JS_GetPropertyStr (js_ctx, js_arr, "length") ;
	JS_ToUint32  (js_ctx, &source_count, len_val) ;
	JS_FreeValue (js_ctx, len_val) ;

	// extract each node
	EntityID *sources = rm_malloc (sizeof (EntityID) * source_count) ;
	for (uint32_t i = 0 ; i < source_count ; i++) {
		JSValue   elem = JS_GetPropertyUint32 (js_ctx, js_arr, i) ;
		JSClassID cid  = JS_GetClassID (elem) ;

		if (cid != js_node_class_id) {
			rm_free (sources) ;
			JS_FreeValue (js_ctx, elem) ;
			return JS_ThrowTypeError (js_ctx,
					"falkor.traverse first argument should be an array of nodes") ;
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
// falkor object setup
//------------------------------------------------------------------------------

// register the global falkor object in the given QuickJS context
void UDF_RegisterFalkorObject
(
	JSContext *js_ctx  // the QuickJS context in which to register the object
) {
	ASSERT (js_ctx != NULL) ;

    // create a plain namespace object: const falkor = {};
    JSValue falkor_obj = JS_NewObject (js_ctx) ;

	// register falkor.log
	JSValue func_obj = JS_NewCFunction (js_ctx, falkor_log, "log", 1) ;

    int def_res = JS_DefinePropertyValueStr (js_ctx, falkor_obj, "log",
			func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;
	ASSERT (def_res >= 0) ;

	// register falkor.traverse
	func_obj = JS_NewCFunction (js_ctx, falkor_traverse, "traverse", 2) ;

    def_res = JS_DefinePropertyValueStr (js_ctx, falkor_obj, "traverse",
			func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;
	ASSERT (def_res >= 0) ;

    // expose the namespace globally as "falkor"
    JSValue global_obj = JS_GetGlobalObject (js_ctx) ;

    JS_SetPropertyStr (js_ctx, global_obj, "falkor", falkor_obj) ;

    // free what we got from GetGlobalObject
    JS_FreeValue (js_ctx, global_obj) ;
}

// set the implementation of the `falkor.register` function
void UDF_SetFalkorRegisterImpl
(
	JSContext *js_ctx,              // the QuickJS context
	UDF_JSCtxRegisterFuncMode mode  // the registration mode
) {
	ASSERT(js_ctx != NULL);

    // get global.falkor
    JSValue global_obj = JS_GetGlobalObject (js_ctx) ;
    JSValue falkor_obj = JS_GetPropertyStr  (js_ctx, global_obj, "falkor") ;

	ASSERT (JS_IsObject (falkor_obj));

	// pick implementation
    JSValue func_obj = JS_UNDEFINED;
    switch (mode) {
        case UDF_FUNC_REG_MODE_VALIDATE:
            func_obj = JS_NewCFunction (js_ctx, validate_register_udf, "register", 1) ;
            break ;

        case UDF_FUNC_REG_MODE_LOCAL:
            func_obj = JS_NewCFunction (js_ctx, local_register_udf, "register", 1) ;
            break ;

        case UDF_FUNC_REG_MODE_GLOBAL:
            func_obj = JS_NewCFunction (js_ctx, global_register_udf, "register", 1) ;
            break ;

        default:
            assert (false && "unknown mode") ;
    }

	ASSERT (JS_IsFunction (js_ctx, func_obj)) ;

    // define property with explicit flags (writable, configurable)
    int def_res = JS_DefinePropertyValueStr (js_ctx, falkor_obj, "register",
			func_obj, JS_PROP_WRITABLE | JS_PROP_CONFIGURABLE) ;

	ASSERT (def_res >= 0) ;

    // clean up
    JS_FreeValue (js_ctx, global_obj) ;
    JS_FreeValue (js_ctx, falkor_obj) ;
}

