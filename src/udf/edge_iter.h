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
// edge iterator class binding
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
);

// class registration: called during runtime initialization
void UDF_RegisterEdgeIteratorClass
(
	JSRuntime *js_runtime  // JavaScript runtime
);

// prototype registration: called during context initialization
void UDF_RegisterEdgeIteratorProto
(
	JSContext *js_ctx  // JavaScript context
);

