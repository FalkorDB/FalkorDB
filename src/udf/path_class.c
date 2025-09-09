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
		JS_SetPropertyUint32 (js_ctx, nodes, i, js_create_node (js_ctx, n)) ;
	}

    return nodes ;
}

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
		JS_SetPropertyUint32 (js_ctx, edges, i, js_create_edge (js_ctx, e)) ;
	}

    return edges ;
}

static JSValue js_path_length
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    Path *p = JS_GetOpaque2 (js_ctx, this_val, js_path_class_id) ;
    if (!p) {
        return JS_EXCEPTION ;
	}

	size_t l = Path_Len(p);
    JSValue obj = JS_NewInt64 (js_ctx, l) ;

    return obj ;
}

// register the path class with the js-runtime
void rt_register_path_class
(
	JSRuntime *js_runtime
) {
	ASSERT (js_runtime != NULL) ;

	// register for each runtime
    int res = JS_NewClass (js_runtime, js_path_class_id, &js_path_class) ;
	ASSERT (res == 0) ;
}

// register the path class with the js-context
void ctx_register_path_class
(
	JSContext *js_ctx
) {
	ASSERT (js_ctx != NULL) ;

	// prototype object
    JSValue proto = JS_NewObject (js_ctx) ;

    int res = JS_SetPropertyFunctionList (js_ctx, proto,
			(JSCFunctionListEntry[]) {
            JS_CGETSET_DEF ("nodes", js_path_nodes, NULL),
            JS_CGETSET_DEF ("length", js_path_length, NULL),
            JS_CGETSET_DEF ("relationships", js_path_relationships, NULL)
        },
        3
    ) ;
	ASSERT (res == 0) ;

    JS_SetClassProto (js_ctx, js_path_class_id, proto) ;
}

JSValue js_create_path
(
	JSContext *js_ctx,
	const Path *path
) {
    JSValue obj = JS_NewObjectClass (js_ctx, js_path_class_id) ;
    if (JS_IsException (obj)) {
        return obj ;
    }

    JS_SetOpaque (obj, (void*) path) ;

    return obj ;
}

