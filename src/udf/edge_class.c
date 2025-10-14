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

extern JSClassID js_edge_class_id ;  // QuickJS class ID for Edge

// forward declaration
static int js_edge_get_property (JSContext *js_ctx, JSPropertyDescriptor *desc,
		JSValueConst obj, JSAtom prop) ;

//------------------------------------------------------------------------------
// class definition
//------------------------------------------------------------------------------

static JSClassExoticMethods js_edge_exotic = {
	.get_own_property = js_edge_get_property,
} ;

static JSClassDef js_edge_class = {
    "Edge",
	.exotic = &js_edge_exotic
};

//------------------------------------------------------------------------------
// property resolution
//------------------------------------------------------------------------------

// resolve an edge property access
//
// example:
//   e.attributes.height
//
// return 1 if property exists, 0 if not found, -1 on exception
static int js_edge_get_property
(
	JSContext *js_ctx,           // the QuickJS context
	JSPropertyDescriptor *desc,  // descriptor to populate if the property exists
	JSValueConst obj,            // the JS object wrapping the edge
	JSAtom prop                  // the property (attribute name)
) {
	GraphEntity *e = JS_GetOpaque (obj, js_edge_class_id) ;
    if (!e) {
        return 0 ;  // property not found
    }

    const char *key = JS_AtomToCString (js_ctx, prop) ;
    if (!key) {
        return -1 ; // exception
    }

	// reject built-in edge attributes, which are exposed via dedicated accessors
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

    // fetch property from edge
    SIValue *v = GraphEntity_GetProperty (e, attr_id) ;

	if (v == ATTRIBUTE_NOTFOUND) {
		return 0;  // not set
    }

	// key found -> convert to JSValue
	if (desc) {
		desc->flags  = JS_PROP_ENUMERABLE ;  // configurable, etc. as needed
		desc->value  = UDF_SIValueToJS (js_ctx, *v) ;
		desc->getter = JS_UNDEFINED ;
		desc->setter = JS_UNDEFINED ;
	}
	return 1 ;  // property exists
}

//------------------------------------------------------------------------------
// edge accessors
//------------------------------------------------------------------------------

// return the ID of the edge
static JSValue js_edge_id
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    GraphEntity *e = JS_GetOpaque2 (js_ctx, this_val, js_edge_class_id) ;
    if (!e) {
        return JS_EXCEPTION ;
	}

    return JS_NewInt64 (js_ctx, ENTITY_GET_ID (e)) ;
}

// return the type (relation name) of the edge
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

    return JS_NewString (js_ctx, Schema_GetName (s)) ;
}

// return the destination (end) node of the edge
// todo: implement proper Node wrapping (currently returns JS_NULL)
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

	return JS_NULL ;  // TODO: wrap `n` as a Node JS object
}

// return the source (start) node of the edge
// todo: implement proper Node wrapping (currently returns JS_NULL)
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

	return JS_NULL ;  // TODO: wrap `n` as a Node JS object
}

//------------------------------------------------------------------------------
// edge object creation
//------------------------------------------------------------------------------

// create a JavaScript Edge object from a FalkorDB Edge
// wraps a native FalkorDB Edge into a QuickJS JSValue instance
// return JSValue representing the Edge in QuickJS
JSValue UDF_CreateEdge
(
	JSContext *js_ctx,  // JavaScript context
	const Edge *edge    // pointer to the native FalkorDB Edge
) {
    JSValue obj = JS_NewObjectClass (js_ctx, js_edge_class_id) ;
    if (JS_IsException (obj)) {
        return obj ;
    }

    JS_SetOpaque (obj, (void*) edge) ;

    return obj ;
}

//------------------------------------------------------------------------------
// class registration
//------------------------------------------------------------------------------

// register the Edge class with a JavaScript runtime
// associates the Edge class definition with the given QuickJS runtime
void UDF_RegisterEdgeClass
(
	JSRuntime *js_runtime  // the QuickJS runtime in which to register the class
) {
	ASSERT (js_runtime != NULL) ;

	// register for each runtime
    int res = JS_NewClass (js_runtime, js_edge_class_id, &js_edge_class) ;
	ASSERT (res == 0) ;
}

// register the Edge class with a JavaScript context
// makes the Edge class available within the provided QuickJS context
void UDF_RegisterEdgeProto
(
	JSContext *js_ctx  // JavaScript context
) {
	ASSERT (js_ctx != NULL) ;

	// prototype object
    JSValue proto = JS_NewObject (js_ctx) ;

    int res = JS_SetPropertyFunctionList (js_ctx, proto,
			(JSCFunctionListEntry[]) {
            JS_CGETSET_DEF ("id",         js_edge_id,              NULL),
            JS_CGETSET_DEF ("type",       js_edge_type,            NULL),
            JS_CGETSET_DEF ("endNode",    js_edge_endNode,         NULL),
            JS_CGETSET_DEF ("startNode",  js_edge_startNode,       NULL),
            JS_CGETSET_DEF ("attributes", UDF_EntityGetAttributes, NULL)
        },
        5
    ) ;
	ASSERT (res == 0) ;

    JS_SetClassProto (js_ctx, js_edge_class_id, proto) ;
}

