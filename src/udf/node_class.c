/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "classes.h"
#include "node_class.h"
#include "../query_ctx.h"
#include "attributes_class.h"

extern JSClassID js_node_class_id ;  // JS node class

// forward declaration
static int js_node_get_property (JSContext *js_ctx, JSPropertyDescriptor *desc,
		JSValueConst obj, JSAtom prop) ;

static JSClassExoticMethods js_node_exotic = {
	.get_own_property = js_node_get_property,
} ;

// Define class + prototype
static JSClassDef js_node_class = {
    "Node",
	.exotic = &js_node_exotic
} ;

// retrieve requested attribute of a node
// e.g. return n.attributes.height ;
static int js_node_get_property
(
	JSContext *js_ctx,
	JSPropertyDescriptor *desc,
	JSValueConst obj,
	JSAtom prop
) {
	GraphEntity *e = JS_GetOpaque (obj, js_node_class_id) ;
    if (!e) {
        return 0 ;  // property not found
    }

    const char *key = JS_AtomToCString (js_ctx, prop) ;
    if (!key) {
        return -1 ; // exception
    }

	// reject built-in node attributes
	if (strcmp (key, "id")         == 0 ||
		strcmp (key, "attributes") == 0 ||
		strcmp (key, "labels")     == 0) {
		// reject direct access to the 'id', 'attributes' & 'labels'
		// e.g. n.id
		// this will cause quickjs to invoke a dedicated function
		// in case access to a user defined 'id' attribute is required
		// access it via n.attributes.id
		return 0 ;
	}

    // search for attribute
    GraphContext *gc = QueryCtx_GetGraphCtx () ;
    ASSERT (gc != NULL) ;

    // get attribute id
    AttributeID attr_id = GraphContext_GetAttributeID (gc, key) ;
    JS_FreeCString (js_ctx, key) ;

    // unknown attribute
    if (attr_id == ATTRIBUTE_ID_NONE) {
        return 0 ;  // property not found
    }

    // get attribute from node object
    SIValue *v = GraphEntity_GetProperty (e, attr_id) ;

    if (v != ATTRIBUTE_NOTFOUND) {
        // key found -> convert to JSValue
        if (desc) {
            desc->flags  = JS_PROP_ENUMERABLE ;  // configurable, etc. as needed
            desc->value  = UDF_SIValueToJS (js_ctx, *v) ;
            desc->getter = JS_UNDEFINED ;
            desc->setter = JS_UNDEFINED ;
        }
        return 1 ;  // property exists
    }

    // missing attribute
    return 0 ;  // property not found
}

static JSValue js_entity_get_id
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    GraphEntity *entity = JS_GetOpaque2 (js_ctx, this_val, js_node_class_id) ;
    if (!entity) {
        return JS_EXCEPTION ;
	}

    JSValue obj = JS_NewInt64 (js_ctx, ENTITY_GET_ID (entity)) ;

    return obj ;
}

static JSValue js_entity_get_labels
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    Node *node = JS_GetOpaque2 (js_ctx, this_val, js_node_class_id) ;
    if (!node) {
        return JS_EXCEPTION ;
	}

	// get node labels
	uint n_lbl ;
	Graph        *g  = QueryCtx_GetGraph () ;
	GraphContext *gc = QueryCtx_GetGraphCtx () ;

	NODE_GET_LABELS (g, node, n_lbl) ;

	// populate js array
	JSValue obj = JS_NewArray (js_ctx) ;
	for (int i = 0; i < n_lbl; i++) {
		// convert label id to string
		Schema *s = GraphContext_GetSchemaByID (gc, labels[i], SCHEMA_NODE) ;
		ASSERT (s != NULL) ;

		const char *lbl = Schema_GetName (s) ;
		JS_SetPropertyUint32 (js_ctx, obj, i, JS_NewString (js_ctx, lbl)) ;
	}

    return obj ;
}

// register the node class with the js-runtime
void rt_register_node_class
(
	JSRuntime *js_runtime
) {
	ASSERT (js_runtime != NULL) ;

	// register for each runtime
    int res = JS_NewClass (js_runtime, js_node_class_id, &js_node_class) ;
	ASSERT (res == 0) ;
}

// register the node class with the js-context
void ctx_register_node_class
(
	JSContext *js_ctx
) {
	ASSERT (js_ctx != NULL) ;

	// prototype object
    JSValue proto = JS_NewObject (js_ctx) ;

    int res = JS_SetPropertyFunctionList (js_ctx, proto,
			(JSCFunctionListEntry[]) {
            JS_CGETSET_DEF ("id", js_entity_get_id, NULL),
            JS_CGETSET_DEF ("labels", js_entity_get_labels, NULL),
            JS_CGETSET_DEF ("attributes", js_entity_get_attributes, NULL)
        },
        3
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

