/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "classes.h"
#include "node_class.h"
#include "attributes_class.h"

// register all classes
void UDF_InitClasses(void) {
    JS_NewClassID (&js_node_class_id) ;
    JS_NewClassID (&js_edge_class_id) ;
    JS_NewClassID (&js_attributes_class_id) ;
}

void UDF_RegisterClasses
(
	JSRuntime *js_runtime,
	JSContext *js_ctx
) {
	register_node_class (js_runtime, js_ctx) ;
}
