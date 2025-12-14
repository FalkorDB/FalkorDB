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

//------------------------------------------------------------------------------
// forward declaration
//------------------------------------------------------------------------------

static int js_edge_get_property (JSContext *js_ctx, JSPropertyDescriptor *desc,
		JSValueConst obj, JSAtom prop) ;

static void js_edge_finalizer (JSRuntime *rt, JSValue val) ;

//------------------------------------------------------------------------------
// class definition
//------------------------------------------------------------------------------

static JSClassExoticMethods js_edge_exotic = {
	.get_own_property = js_edge_get_property,
} ;

static JSClassDef js_edge_class = {
    "Edge",
	.exotic = &js_edge_exotic,
	.finalizer = js_edge_finalizer
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
		strcmp (key, "target")     == 0 ||
		strcmp (key, "source")     == 0 ||
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
    SIValue v ;
	if (!GraphEntity_GetProperty (e, attr_id, &v)) {
		return 0;  // not set
    }

	// key found -> convert to JSValue
	if (desc) {
		desc->flags  = JS_PROP_ENUMERABLE ;  // configurable, etc. as needed
		desc->value  = UDF_SIValueToJS (js_ctx, v) ;
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

// return the target (destination) node of the edge
// todo: implement proper Node wrapping (currently returns JS_NULL)
static JSValue js_edge_target
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

	return UDF_CreateNode (js_ctx, &n) ;
}

// return the source node of the edge
// todo: implement proper Node wrapping (currently returns JS_NULL)
static JSValue js_edge_source
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

	return UDF_CreateNode (js_ctx, &n) ;
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
	ASSERT (edge   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	// clone edge
	Edge *_edge = rm_malloc (sizeof (Edge)) ;
	memcpy (_edge, edge, sizeof (Edge)) ;

    JSValue obj = JS_NewObjectClass (js_ctx, js_edge_class_id) ;
    if (JS_IsException (obj)) {
        return obj ;
    }

    JS_SetOpaque (obj, (void*) _edge) ;

    return obj ;
}

// destructor for the Edge JS object
// frees the underlying native Edge when the JS object is garbage collected
static void js_edge_finalizer
(
	JSRuntime *rt,
	JSValue val
) {
    // get the opaque pointer
    GraphEntity *edge = JS_GetOpaque (val, js_edge_class_id) ;

    // check if the pointer exists and free the native object
    if (edge) {
		rm_free (edge) ;
    }
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
static const JSCFunctionListEntry edge_proto_func_list[] = {
	JS_CGETSET_DEF ("id",         js_edge_id,              NULL),
	JS_CGETSET_DEF ("type",       js_edge_type,            NULL),
	JS_CGETSET_DEF ("source",     js_edge_source,          NULL),
	JS_CGETSET_DEF ("target",     js_edge_target,          NULL),
	JS_CGETSET_DEF ("attributes", UDF_EntityGetAttributes, NULL)
} ;

void UDF_RegisterEdgeProto
(
	JSContext *js_ctx  // JavaScript context
) {
	ASSERT (js_ctx != NULL) ;

	// prototype object
    JSValue proto = JS_NewObject (js_ctx) ;

    int res =
		JS_SetPropertyFunctionList (js_ctx, proto, edge_proto_func_list, 5) ;
	ASSERT (res == 0) ;

    JS_SetClassProto (js_ctx, js_edge_class_id, proto) ;
}

