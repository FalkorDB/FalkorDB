/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "classes.h"
#include "../query_ctx.h"
#include "../graph/entities/graph_entity.h"

//------------------------------------------------------------------------------
// class IDs
//------------------------------------------------------------------------------

extern JSClassID js_node_class_id;        // QuickJS class ID for Node
extern JSClassID js_edge_class_id;        // QuickJS class ID for Edge
extern JSClassID js_attributes_class_id;  // QuickJS class ID for Attributes

//------------------------------------------------------------------------------
// forward declaration
//------------------------------------------------------------------------------

static int js_attributes_get_property (JSContext *js_ctx,
		JSPropertyDescriptor *desc, JSValueConst obj, JSAtom prop) ;

static JSClassExoticMethods js_attributes_exotic = {
    .get_own_property = js_attributes_get_property,
};

//------------------------------------------------------------------------------
// exotic property handler for the Attributes class
//------------------------------------------------------------------------------

static JSClassDef js_attributes_class = {
    "Attributes",
	.exotic = &js_attributes_exotic,
};

//------------------------------------------------------------------------------
// attribute property resolution
//------------------------------------------------------------------------------

// retrieve a requested attribute from a graph entity
//
// example:
// n.attributes.height
//
// returns:  1 if property exists and was resolved,
//           0 if not found,
//           -1 if an exception occurred

// retrieve requested attribute of a graph entity
// e.g. return n.attributes.height ;
static int js_attributes_get_property
(
	JSContext *js_ctx,           // the QuickJS context
	JSPropertyDescriptor *desc,  // [optional] descriptor to populate
	JSValueConst obj,            // the JS object wrapping the graph entity
	JSAtom prop                  // the property (attribute name) to resolve
) {
	GraphEntity *e = JS_GetOpaque (obj, js_attributes_class_id) ;
    if (!e) {
        return 0;  // property not found
    }

    const char *key = JS_AtomToCString(js_ctx, prop);
    if (!key) {
        return -1; // exception
    }

    // search for attribute
    GraphContext *gc = QueryCtx_GetGraphCtx();
    ASSERT(gc != NULL);

    // get attribute id
    AttributeID attr_id = GraphContext_GetAttributeID(gc, key);
    JS_FreeCString(js_ctx, key);

    // unknown attribute
    if (attr_id == ATTRIBUTE_ID_NONE) {
        return 0;  // property not found
    }

    // get attribute from node object
    SIValue *v = GraphEntity_GetProperty(e, attr_id);
	if (v == ATTRIBUTE_NOTFOUND) {
		return 0 ; // attribute not set
	}

	// key found -> convert to JSValue
	if (desc) {
		desc->flags  = JS_PROP_ENUMERABLE ;  // configurable, etc. as needed
		desc->value  = UDF_SIValueToJS (js_ctx, *v) ;
		desc->getter = JS_UNDEFINED ;
		desc->setter = JS_UNDEFINED ;
	}

	return 1 ;
}

// -----------------------------------------------------------------------------
// Attributes object factory
// -----------------------------------------------------------------------------

// create an `Attributes` object for a given graph entity
// returns A new JS object of class `Attributes`, or JS_EXCEPTION on error
JSValue UDF_EntityGetAttributes
(
	JSContext *js_ctx,      // the QuickJS context
	JSValueConst this_val   // the JavaScript object representing the entity
) {
	JSClassID cid = JS_GetClassID (this_val) ;
	ASSERT (cid == js_node_class_id || cid == js_edge_class_id) ;

    GraphEntity *entity = JS_GetOpaque2 (js_ctx, this_val, cid) ;
    if (!entity) {
        return JS_EXCEPTION ;
	}

    JSValue obj = JS_NewObjectClass (js_ctx, js_attributes_class_id) ;
    JS_SetOpaque (obj, entity) ;

    return obj;
}

//------------------------------------------------------------------------------
// Class registration
//------------------------------------------------------------------------------

// register the `Attributes` class with the provided QuickJS runtime
void UDF_RegisterAttributesClass
(
	JSRuntime *js_runtime  // the QuickJS runtime in which to register the class
) {
	ASSERT (js_runtime != NULL) ;

	// register for each runtime
	int res = JS_NewClass (js_runtime, js_attributes_class_id,
			&js_attributes_class) ;
	ASSERT (res == 0) ;
}

