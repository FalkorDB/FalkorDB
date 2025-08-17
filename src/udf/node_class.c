/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "node_class.h"

// Define class + prototype
static JSClassDef js_node_class = {
    "Node",
};

// create a JSValue of type Node
static JSValue js_create_node
(
	JSContext *js_ctx,
	const Node *node
) {
    JSValue obj = JS_NewObjectClass (js_ctx, js_node_class_id) ;
    if (JS_IsException (obj)) {
        return obj ;
    }

    JS_SetOpaque (obj, node) ;

    return obj ;
}

void register_node_class
(
	JSContext *js_ctx
) {
	ASSERT (js_ctx != NULL) ;

    // register class
    JS_NewClassID (&js_node_class_id) ;
    JS_NewClass (JS_GetRuntime (js_ctx), js_node_class_id, &js_node_class) ;

	// prototype object
    JSValue proto = JS_NewObject (js_ctx) ;

    JS_SetPropertyFunctionList (js_ctx, proto,
        (JSCFunctionListEntry[]) {
            JS_CGETSET_DEF("attributes", js_node_get_attributes, NULL),
        },
        1
    ) ;

    JS_SetClassProto (js_ctx, js_node_class_id, proto) ;
}

