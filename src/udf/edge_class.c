/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "classes.h"
#include "edge_class.h"
#include "../query_ctx.h"
#include "attributes_class.h"

extern JSClassID js_edge_class_id ;  // JS edge class

// forward declaration
static int js_edge_get_property (JSContext *js_ctx, JSPropertyDescriptor *desc,
		JSValueConst obj, JSAtom prop) ;

static JSClassExoticMethods js_edge_exotic = {
	.get_own_property = js_edge_get_property,
} ;

// Define class + prototype
static JSClassDef js_edge_class = {
    "Edge",
	.exotic = &js_edge_exotic
};

// retrieve requested attribute of an edge
// e.g. return n.attributes.height ;
static int js_edge_get_property
(
	JSContext *js_ctx,
	JSPropertyDescriptor *desc,
	JSValueConst obj,
	JSAtom prop
) {
	GraphEntity *e = JS_GetOpaque (obj, js_edge_class_id) ;
    if (!e) {
        return 0 ;  // property not found
    }

    const char *key = JS_AtomToCString (js_ctx, prop) ;
    if (!key) {
        return -1 ; // exception
    }

	// reject built-in node attributes
	if (strcmp (key, "id")         == 0 ||
		strcmp (key, "type")       == 0 ||
		strcmp (key, "endNode")    == 0 ||
		strcmp (key, "startNode")  == 0 ||
		strcmp (key, "attributes") == 0 ) {
		// reject direct access to edge native properties
		// e.g. e.id
		// this will cause quickjs to invoke a dedicated function
		// in case access to a user defined 'id' attribute is required
		// access it via e.attributes.id
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

    // get attribute from edge object
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

static JSValue js_edge_id
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    GraphEntity *e = JS_GetOpaque2 (js_ctx, this_val, js_edge_class_id) ;
    if (!e) {
        return JS_EXCEPTION ;
	}

    JSValue obj = JS_NewInt64 (js_ctx, ENTITY_GET_ID (e)) ;

    return obj ;
}

static JSValue js_edge_type
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    Edge *e = JS_GetOpaque2 (js_ctx, this_val, js_edge_class_id) ;
    if (!e) {
        return JS_EXCEPTION ;
	}

	RelationID r = Edge_GetRelationID (e) ;

	const GraphContext *gc = QueryCtx_GetGraphCtx () ;
	Schema *s = GraphContext_GetSchemaByID (gc, r, SCHEMA_EDGE) ;
	ASSERT (s != NULL) ;

    JSValue obj = JS_NewString (js_ctx, Schema_GetName (s)) ;

    return obj ;
}

static JSValue js_edge_endNode
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    Edge *e = JS_GetOpaque2 (js_ctx, this_val, js_edge_class_id) ;
    if (!e) {
        return JS_EXCEPTION ;
	}

	Graph *g = QueryCtx_GetGraph () ;
	NodeID n_id = Edge_GetDestNodeID (e) ;

	Node n ;
	bool found = Graph_GetNode (g, n_id, &n) ;
	ASSERT (found) ;

	assert ("implement" && false) ;
	return JS_NULL ;
	// return js_create_node (js_ctx, val.ptrval) ;
}

static JSValue js_edge_startNode
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    Edge *e = JS_GetOpaque2 (js_ctx, this_val, js_edge_class_id) ;
    if (!e) {
        return JS_EXCEPTION ;
	}

	Graph *g = QueryCtx_GetGraph () ;
	NodeID n_id = Edge_GetSrcNodeID (e) ;

	Node n ;
	bool found = Graph_GetNode (g, n_id, &n) ;
	ASSERT (found) ;

	assert ("implement" && false) ;
	return JS_NULL ;
	//return js_create_node (js_ctx, val.ptrval) ;
}

// register the edge class with the js-runtime
void rt_register_edge_class
(
	JSRuntime *js_runtime
) {
	ASSERT (js_runtime != NULL) ;

	// register for each runtime
    int res = JS_NewClass (js_runtime, js_edge_class_id, &js_edge_class) ;
	ASSERT (res == 0) ;
}

// register the edge class with the js-context
void ctx_register_edge_class
(
	JSContext *js_ctx
) {
	ASSERT (js_ctx != NULL) ;

	// prototype object
    JSValue proto = JS_NewObject (js_ctx) ;

    int res = JS_SetPropertyFunctionList (js_ctx, proto,
			(JSCFunctionListEntry[]) {
            JS_CGETSET_DEF ("id",         js_edge_id,               NULL),
            JS_CGETSET_DEF ("type",       js_edge_type,             NULL),
            JS_CGETSET_DEF ("endNode",    js_edge_endNode,          NULL),
            JS_CGETSET_DEF ("startNode",  js_edge_startNode,        NULL),
            JS_CGETSET_DEF ("attributes", js_entity_get_attributes, NULL)
        },
        5
    ) ;
	ASSERT (res == 0) ;

    JS_SetClassProto (js_ctx, js_edge_class_id, proto) ;
}

JSValue js_create_edge
(
	JSContext *js_ctx,
	const Edge *edge
) {
    JSValue obj = JS_NewObjectClass (js_ctx, js_edge_class_id) ;
    if (JS_IsException (obj)) {
        return obj ;
    }

    JS_SetOpaque (obj, (void*) edge) ;

    return obj ;
}

