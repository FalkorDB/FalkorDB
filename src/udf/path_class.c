/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "classes.h"
#include "node_class.h"
#include "edge_class.h"
#include "path_class.h"

extern JSClassID js_path_class_id ;  // JS path class

// Define class + prototype
static JSClassDef js_path_class = {
    "Path",
} ;

// retrieve nodes from a Path
// returns a JavaScript array of Node objects
static JSValue js_path_nodes
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    Path *p = JS_GetOpaque2 (js_ctx, this_val, js_path_class_id) ;
    if (!p) {
        return JS_EXCEPTION ;
	}

	size_t l = Path_NodeCount (p) ;
	JSValue nodes = JS_NewArray (js_ctx) ;

	for (int i = 0; i < l; i++) {
		Node *n = Path_GetNode (p, i) ;
		JS_SetPropertyUint32 (js_ctx, nodes, i, UDF_CreateNode (js_ctx, n)) ;
	}

    return nodes ;
}

// retrieve relationships (edges) from a Path
// returns a JavaScript array of Edge objects
static JSValue js_path_relationships
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    Path *p = JS_GetOpaque2 (js_ctx, this_val, js_path_class_id) ;
    if (!p) {
        return JS_EXCEPTION ;
	}

	size_t l = Path_EdgeCount (p) ;
	JSValue edges = JS_NewArray (js_ctx) ;

	for (int i = 0; i < l; i++) {
		Edge *e = Path_GetEdge (p, i) ;
		JS_SetPropertyUint32 (js_ctx, edges, i, UDF_CreateEdge (js_ctx, e)) ;
	}

    return edges ;
}

// get the length of a Path
// returns the number of edges in the path as a JS integer
static JSValue js_path_length
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    Path *p = JS_GetOpaque2 (js_ctx, this_val, js_path_class_id) ;
    if (!p) {
        return JS_EXCEPTION ;
	}

    return JS_NewInt64 (js_ctx, Path_Len (p)) ;
}

// create a JSValue of type Path
// create a JavaScript Path object from a FalkorDB Path
// wraps a native FalkorDB Path into a QuickJS JSValue instance
// return JSValue representing the Path in QuickJS
JSValue UDF_CreatePath
(
	JSContext *js_ctx,  // JavaScript context
	const Path *path    // pointer to the native FalkorDB Path
) {
    JSValue obj = JS_NewObjectClass (js_ctx, js_path_class_id) ;
    if (JS_IsException (obj)) {
        return obj ;
    }

    JS_SetOpaque (obj, (void*) path) ;

    return obj ;
}

// register the Path class with a QuickJS runtime
// associates the Path class definition with the given QuickJS runtime
void UDF_RegisterPathClass
(
	JSRuntime *js_runtime  // JavaScript runtime
) {
	ASSERT (js_runtime != NULL) ;

	// register for each runtime
    int res = JS_NewClass (js_runtime, js_path_class_id, &js_path_class) ;
	ASSERT (res == 0) ;
}

// register the Path class with a QuickJS context
// makes the Path class available within the provided QuickJS context

static const JSCFunctionListEntry path_proto_func_list[] = {
	JS_CGETSET_DEF ("nodes", js_path_nodes, NULL),
	JS_CGETSET_DEF ("length", js_path_length, NULL),
	JS_CGETSET_DEF ("relationships", js_path_relationships, NULL)
} ;

void UDF_RegisterPathProto
(
	JSContext *js_ctx  // JavaScript context
) {
	ASSERT (js_ctx != NULL) ;

	// prototype object
    JSValue proto = JS_NewObject (js_ctx) ;

    int res =
		JS_SetPropertyFunctionList (js_ctx, proto, path_proto_func_list, 3) ;
	ASSERT (res == 0) ;

    JS_SetClassProto (js_ctx, js_path_class_id, proto) ;
}

