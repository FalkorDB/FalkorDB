/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "edge_class.h"
#include "../graph/graph.h"
#include "../graph/tensor/tensor.h"

extern JSClassID js_edge_it_class_id ;  // JS edge iterator class

// internal state tracker
typedef struct {
	const Graph *g;      // reference to the graph for Graph_GetEdge
	RelationID rel_id;   // relationship-type ID to filter by
	TensorIterator *it;  // edge iterator over the relationship matrix
} EdgeIteratorState ;

//------------------------------------------------------------------------------
// Function Prototypes
//------------------------------------------------------------------------------

static void    js_edge_it_finalizer (JSRuntime *rt, JSValue val) ;
static JSValue js_edge_it_next      (JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) ;
static JSValue js_edge_it_get_self  (JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) ;

//------------------------------------------------------------------------------
// Class Definition
//------------------------------------------------------------------------------

static JSClassDef js_edge_it_class = {
	"EdgeIterator",
	.finalizer = js_edge_it_finalizer
} ;

// prototype function list
static const JSCFunctionListEntry edge_it_proto_func_list[] = {
	// standard iterator .next() method
	JS_CFUNC_DEF("next", 0, js_edge_it_next),
} ;

//------------------------------------------------------------------------------
// Iterator Logic
//------------------------------------------------------------------------------

// the [Symbol.iterator] method simply returns the iterator itself
// this allows: for (let edge of graph.iterateEdges('KNOWS'))
static JSValue js_edge_it_get_self
(
	JSContext *ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
	return JS_DupValue (ctx, this_val) ;
}

// the .next() method follows the ECMAScript iterator protocol:
// returns { value: Edge|undefined, done: boolean }
static JSValue js_edge_it_next
(
	JSContext *js_ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
	EdgeIteratorState *state = JS_GetOpaque (this_val, js_edge_it_class_id) ;
	if (!state) {
		return JS_EXCEPTION ;
	}

	JSValue res = JS_NewObject (js_ctx) ;
	if (JS_IsException (res)) {
		return res ;
	}

	// relationship matrices store edge IDs as UINT64 values
	// keyed by (src_node_id, dst_node_id)
	GrB_Index src_id ;
	GrB_Index dst_id ;
	GrB_Index edge_id ;
	if (!TensorIterator_next  (state->it, &src_id, &dst_id, &edge_id, NULL)) {
		JS_SetPropertyStr (js_ctx, res, "done",  JS_TRUE) ;
		JS_SetPropertyStr (js_ctx, res, "value", JS_UNDEFINED) ;
		return res ;
	}

	Edge e ;
	bool found = Graph_GetEdge (state->g, edge_id, &e) ;
	ASSERT (found == true) ;

	// populate src/dst so callers can access edge endpoints
	Edge_SetSrcNodeID  (&e, src_id) ;
	Edge_SetDestNodeID (&e, dst_id) ;
	Edge_SetRelationID (&e, state->rel_id) ;

	JS_SetPropertyStr (js_ctx, res, "done",  JS_FALSE) ;
	JS_SetPropertyStr (js_ctx, res, "value", UDF_CreateEdge (js_ctx, &e)) ;

	return res ;
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------

// create a JavaScript Edge Iterator object from a GraphBLAS iterator
// over a single relationship-type matrix
// returns JSValue representing the Edge Iterator in QuickJS
JSValue UDF_CreateEdgeIterator
(
	JSContext *js_ctx,  // JavaScript context
	const Graph *g,     // graph
	RelationID rel_id,  // relationship-type ID
	TensorIterator *it  // GraphBLAS iterator over a rel-type matrix
) {
	ASSERT (g      != NULL) ;
	ASSERT (it     != NULL) ;
	ASSERT (js_ctx != NULL) ;

	JSValue obj = JS_NewObjectClass (js_ctx, js_edge_it_class_id) ;
	if (JS_IsException (obj)) {
		rm_free (it) ;
		return obj ;
	}

	EdgeIteratorState *state = rm_malloc (sizeof (EdgeIteratorState)) ;
	state->g      = g ;
	state->it     = it ;
	state->rel_id = rel_id ;

	JS_SetOpaque (obj, state) ;

	return obj ;
}

// destructor: frees the underlying GraphBLAS iterator
static void js_edge_it_finalizer
(
	JSRuntime *rt,
	JSValue val
) {
	EdgeIteratorState *state = JS_GetOpaque (val, js_edge_it_class_id) ;

	if (state) {
		if (state->it) {
			rm_free (state->it) ;
		}
		rm_free (state) ;
	}
}

// class registration: called during runtime initialization
void UDF_RegisterEdgeIteratorClass
(
	JSRuntime *js_runtime  // JavaScript runtime
) {
	ASSERT (js_runtime != NULL) ;

	int res = JS_NewClass (js_runtime, js_edge_it_class_id, &js_edge_it_class) ;
	ASSERT (res == 0) ;
}

// prototype registration: called during context initialization
void UDF_RegisterEdgeIteratorProto
(
	JSContext *js_ctx  // JavaScript context
) {
	ASSERT (js_ctx != NULL) ;

	JSValue proto = JS_NewObject (js_ctx) ;
	if (JS_IsException (proto)) {
		return ;
	}

	// register .next() via the function list
	int res = JS_SetPropertyFunctionList (js_ctx, proto,
			edge_it_proto_func_list,
			sizeof (edge_it_proto_func_list) / sizeof (edge_it_proto_func_list[0])) ;
	ASSERT (res == 0) ;

	// resolve Symbol.iterator at runtime via the global Symbol object
	JSValue global      = JS_GetGlobalObject (js_ctx) ;
	JSValue symbol_ctor = JS_GetPropertyStr  (js_ctx, global, "Symbol") ;
	JSValue sym_iter    = JS_GetPropertyStr  (js_ctx, symbol_ctor, "iterator") ;

	JSAtom  iter_atom = JS_ValueToAtom (js_ctx, sym_iter) ;

	JSValue self_fn = JS_NewCFunction (js_ctx, js_edge_it_get_self,
			"[Symbol.iterator]", 0) ;

	JS_DefinePropertyValue (js_ctx, proto, iter_atom, self_fn,
			JS_PROP_CONFIGURABLE | JS_PROP_WRITABLE) ;

	// cleanup
	JS_FreeAtom  (js_ctx, iter_atom) ;
	JS_FreeValue (js_ctx, sym_iter) ;
	JS_FreeValue (js_ctx, symbol_ctor) ;
	JS_FreeValue (js_ctx, global) ;

	JS_SetClassProto (js_ctx, js_edge_it_class_id, proto) ;
}

