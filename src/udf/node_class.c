/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "classes.h"
#include "node_class.h"
#include "attributes_class.h"

// Define class + prototype
static JSClassDef js_node_class = {
    "Node",
};

// create a JSValue of type Node
void register_node_class
(
	JSRuntime *js_runtime, 
	JSContext *js_ctx
) {
	ASSERT (js_ctx     != NULL) ;
	ASSERT (js_runtime != NULL) ;

	// register for each runtime
	printf("After JS_NewClass: %u\n", js_node_class_id);
    int res = JS_NewClass (js_runtime, js_node_class_id, &js_node_class) ;
	ASSERT (res == 0) ;

	// prototype object
    JSValue proto = JS_NewObject (js_ctx) ;

    res = JS_SetPropertyFunctionList (js_ctx, proto,
        (JSCFunctionListEntry[]) {
            JS_CGETSET_DEF("attributes", js_entity_get_attributes, NULL),
        },
        1
    ) ;
	ASSERT (res == 0) ;

    JS_SetClassProto (js_ctx, js_node_class_id, proto) ;
}

JSValue js_create_node
(
	JSContext *js_ctx,
	const Node *node
) {
    JSValue obj = JS_NewObjectClass (js_ctx, js_node_class_id) ;
    if (JS_IsException (obj)) {
        return obj ;
    }

    JS_SetOpaque (obj, (void*) node) ;

    return obj ;
}

