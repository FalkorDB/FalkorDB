/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "GraphBLAS.h"
#include "node_class.h"
#include "../query_ctx.h"
#include "../graph/delta_matrix/delta_matrix_iter.h"

extern JSClassID js_node_it_class_id ;  // JS node iterator class

// internal state tracker
typedef struct {
    const Graph *g;             // reference to the graph for Graph_GetNode
    Delta_MatrixTupleIter *it;  // node iterator
} NodeIteratorState ;

//------------------------------------------------------------------------------
// Function Prototypes
//------------------------------------------------------------------------------

static void js_node_it_finalizer (JSRuntime *rt, JSValue val) ;
static JSValue js_node_it_next (JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) ;
static JSValue js_node_it_get_self (JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) ;

//------------------------------------------------------------------------------
// Class Definition
//------------------------------------------------------------------------------

static JSClassDef js_node_it_class = {
    "NodeIterator",
	.finalizer = js_node_it_finalizer
} ;

// prototype function list
static const JSCFunctionListEntry node_it_proto_func_list[] = {
	// standard iterator .next() method
	JS_CFUNC_DEF("next", 0, js_node_it_next),
};

//------------------------------------------------------------------------------
// Iterator Logic
//------------------------------------------------------------------------------

// the [Symbol.iterator] method simply returns the iterator itself
// this allows: for (let node of graph.iterateNodes('label'))
static JSValue js_node_it_get_self
(
	JSContext *ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
    return JS_DupValue (ctx, this_val) ;
}

// the .next() method follows the ECMAScript protocol:
// returns { value: Node|undefined, done: boolean }
static JSValue js_node_it_next
(
	JSContext *js_ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
	NodeIteratorState *state = JS_GetOpaque (this_val, js_node_it_class_id) ;
    if (!state) {
		return JS_EXCEPTION ;
	}

    JSValue res = JS_NewObject (js_ctx) ;
    if (JS_IsException (res)) {
		return res ;
	}

	// get the current element
	GrB_Index id ;
	GrB_Info info =
		Delta_MatrixTupleIter_next_BOOL (state->it, &id, NULL, NULL) ;
	if (info == GxB_EXHAUSTED) {
		JS_SetPropertyStr (js_ctx, res, "done",  JS_TRUE) ;
		JS_SetPropertyStr (js_ctx, res, "value", JS_UNDEFINED) ;
		return res ;
	}

	Node n ;
	bool found = Graph_GetNode (state->g, id, &n) ;
	ASSERT (found == true) ;

	// create a JS Node object from the current iterator position
	// this assumes you have a helper in node_class.h to wrap the current element
	JS_SetPropertyStr (js_ctx, res, "done",  JS_FALSE) ;
	JS_SetPropertyStr (js_ctx, res, "value", UDF_CreateNode (js_ctx, &n)) ;

    return res ;
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------

// create a JavaScript Node Iterator object from a GraphBLAS iterator
// return JSValue representing the Node Iterator in QuickJS
JSValue UDF_CreateNodeIterator
(
	JSContext *js_ctx,         // JavaScript context
	const Graph *g,            // graph
	Delta_MatrixTupleIter *it  // GraphBLAS iterator over a label matrix
) {
	ASSERT (g      != NULL) ;
	ASSERT (it     != NULL) ;
	ASSERT (js_ctx != NULL) ;

    JSValue obj = JS_NewObjectClass (js_ctx, js_node_it_class_id) ;
    if (JS_IsException (obj)) {
		GrB_OK (Delta_MatrixTupleIter_detach (it)) ;
		rm_free (it) ;
        return obj ;
    }

	NodeIteratorState *state = rm_malloc (sizeof (NodeIteratorState)) ;
    state->g  = g;
    state->it = it;

    JS_SetOpaque (obj, state) ;

    return obj ;
}

// destructor: frees the underlying GraphBLAS iterator
static void js_node_it_finalizer 
(
	JSRuntime *rt,
	JSValue val
) {
    // get the opaque pointer
    NodeIteratorState *state = JS_GetOpaque (val, js_node_it_class_id) ;

    // check if the pointer exists and free the native object
    if (state) {
		if (state->it) {
			GrB_OK (Delta_MatrixTupleIter_detach (state->it)) ;
			rm_free (state->it) ;
		}
		rm_free (state) ;
    }
}

// class Registration: Called during runtime initialization
void UDF_RegisterNodeIteratorClass
(
	JSRuntime *js_runtime  // JavaScript runtime
) {
	ASSERT (js_runtime != NULL) ;

    int res = JS_NewClass (js_runtime, js_node_it_class_id, &js_node_it_class) ;
	ASSERT (res == 0) ;
}

// prototype registration: called during context initialization
void UDF_RegisterNodeIteratorProto
(
	JSContext *js_ctx  // JavaScript context
) {
	ASSERT (js_ctx != NULL) ;

	JSValue proto = JS_NewObject (js_ctx) ;
	if (JS_IsException (proto)) {
		return ;
	}

	// register only .next() via the function list
	int res = JS_SetPropertyFunctionList (js_ctx, proto,
			node_it_proto_func_list,
			sizeof (node_it_proto_func_list) / sizeof (node_it_proto_func_list[0])) ;
	ASSERT (res == 0) ;

	// resolve Symbol.iterator at runtime via the global Symbol object
	JSValue global      = JS_GetGlobalObject (js_ctx) ;
	JSValue symbol_ctor = JS_GetPropertyStr  (js_ctx, global,      "Symbol") ;
	JSValue sym_iter    = JS_GetPropertyStr  (js_ctx, symbol_ctor, "iterator") ;

	JSAtom iter_atom = JS_ValueToAtom (js_ctx, sym_iter) ;

	JSValue self_fn = JS_NewCFunction (js_ctx, js_node_it_get_self,
	                                   "[Symbol.iterator]", 0) ;

	JS_DefinePropertyValue (js_ctx, proto, iter_atom, self_fn,
	                        JS_PROP_CONFIGURABLE | JS_PROP_WRITABLE) ;

	// cleanup
	JS_FreeAtom  (js_ctx, iter_atom) ;
	JS_FreeValue (js_ctx, sym_iter) ;
	JS_FreeValue (js_ctx, symbol_ctor) ;
	JS_FreeValue (js_ctx, global) ;

	JS_SetClassProto (js_ctx, js_node_it_class_id, proto) ;
}

