/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "quickjs.h"
#include "../graph/graph.h"
#include "../graph/entities/node.h"

extern JSClassID js_node_it_class_id ;  // JS node iterator class

//------------------------------------------------------------------------------
// node iterator class binding
//------------------------------------------------------------------------------

// create a JavaScript Node Iterator object from a GraphBLAS iterator
// return JSValue representing the Node Iterator in QuickJS
JSValue UDF_CreateNodeIterator
(
	JSContext *js_ctx,         // JavaScript context
	const Graph *g,            // graph
	Delta_MatrixTupleIter *it  // GraphBLAS iterator over a label matrix
);

// register the Node Iterator class with a JavaScript runtime
// associates the Node Iterator class definition with the given QuickJS runtime
void UDF_RegisterNodeIteratorClass
(
	JSRuntime *js_runtime  // JavaScript runtime
);

// register the Node Iterator class with a JavaScript context
// makes the Node Iterator class available within the provided QuickJS context
void UDF_RegisterNodeIteratorProto
(
	JSContext *js_ctx  // JavaScript context
);

