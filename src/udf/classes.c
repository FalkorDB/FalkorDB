/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "classes.h"

JSClassID js_node_class_id;        // JS node class
JSClassID js_edge_class_id;        // JS edge class
JSClassID js_path_class_id;        // JS path class
JSClassID js_attributes_class_id;  // JS attributes class

// initialize all QuickJS classes required by the UDF subsystem
// this should be called once during application startup
// before any QuickJS runtime or context is created
void UDF_InitClasses(void) {
    JS_NewClassID (&js_node_class_id) ;
    JS_NewClassID (&js_edge_class_id) ;
    JS_NewClassID (&js_path_class_id) ;
    JS_NewClassID (&js_attributes_class_id) ;
}

// register all FalkorDB classes with the given QuickJS runtime
void UDF_RT_RegisterClasses
(
	JSRuntime *js_rt  // the QuickJS runtime in which to register classes
) {
	ASSERT (js_rt != NULL) ;

	UDF_RegisterNodeClass       (js_rt) ;
	UDF_RegisterEdgeClass       (js_rt) ;
	UDF_RegisterPathClass       (js_rt) ;
	UDF_RegisterAttributesClass (js_rt) ;
}

// register all FalkorDB classes with the given QuickJS context
void UDF_CTX_RegisterClasses
(
	JSContext *js_ctx  // the QuickJS context in which to register classes.
) {
	UDF_RegisterNodeProto       (js_ctx) ;
	UDF_RegisterEdgeProto       (js_ctx) ;
	UDF_RegisterPathProto       (js_ctx) ;
	UDF_RegisterGraphObject     (js_ctx) ;
	UDF_RegisterFalkorObject    (js_ctx) ;
	UDF_RegisterAttributesProto (js_ctx) ;
}

