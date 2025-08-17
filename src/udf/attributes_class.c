/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "attributes_class.h"
#include "../graph/entities/graph_entity.h"

// forward declaration
static int js_attributes_get_property (JSContext *ctx,
		JSPropertyDescriptor *desc, JSValueConst obj, JSAtom prop) ;

static JSClassExoticMethods js_attributes_exotic = {
    .get_own_property = js_attributes_get_property,
};

static JSClassDef js_attributes_class = {
    "Attributes",
	.exotic = &js_attributes_exotic,
};

static JSValue js_entity_get_attributes
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    GraphEntity *entity = JS_GetOpaque2 (ctx, this_val, js_node_class_id) ;
    if (!entity) {
        return JS_EXCEPTION ;
	}

    JSValue obj = JS_NewObjectClass (js_ctx, js_attributes_class_id) ;
    JS_SetOpaque (obj, entity) ;

    return obj;
}

// retrieve requested attribute of a graph entity
// e.g. return n.attributes.height ;
static JSValue js_attributes_get_property
(
	JSContext *js_ctx,
	JSPropertyDescriptor *desc,
	JSValueConst obj,
	JSAtom prop
) {
    GraphEntity *e = JS_GetOpaque (obj, js_attributes_class_id) ;
    if (!e) {
        return JS_UNDEFINED ;
	}

    const char *key = JS_AtomToCString (js_ctx, prop) ;
    if (!key) {
        return JS_UNDEFINED ;
	}

    // search for attribute
	GraphContext *gc = QueryCtx_GetGraphCtx () ;
	ASSERT (gc != NULL) ;

	// get attribute id
	AttributeID attr_id = GraphContext_GetAttributeID (gc, key) ;
	JS_FreeCString (js_ctx, key) ;

	// unknown attribute
	if (attr_id == ATTRIBUTE_ID_NONE) {
		return JS_UNDEFINED ;	
	}

	// get attribute from node object
	SIValue *v = GraphEntity_GetProperty (e, attr_id) ;

	// key found convert to JSValue
	if (v != ATTRIBUTE_NOTFOUND) {
		return UDF_SIValueToJS (js_ctx, *v) ;
	}

	// missing attribute
    return JS_UNDEFINED ;
}

