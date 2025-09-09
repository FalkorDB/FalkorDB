/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "classes.h"
#include "node_class.h"
#include "edge_class.h"
#include "path_class.h"
#include "attributes_class.h"

JSClassID js_node_class_id;        // JS node class
JSClassID js_edge_class_id;        // JS edge class
JSClassID js_path_class_id;        // JS path class
JSClassID js_attributes_class_id;  // JS attributes class

// register all classes
void UDF_InitClasses(void) {
    JS_NewClassID (&js_node_class_id) ;
    JS_NewClassID (&js_edge_class_id) ;
    JS_NewClassID (&js_path_class_id) ;
    JS_NewClassID (&js_attributes_class_id) ;
}

void UDF_RT_RegisterClasses
(
	JSRuntime *js_runtime
) {
	rt_register_node_class (js_runtime) ;
	rt_register_edge_class (js_runtime) ;
	rt_register_path_class (js_runtime) ;
	rt_register_attributes_class (js_runtime) ;
}

void UDF_CTX_RegisterClasses
(
	JSContext *js_ctx
) {
	ctx_register_node_class (js_ctx) ;
	ctx_register_edge_class (js_ctx) ;
	ctx_register_path_class (js_ctx) ;
}

