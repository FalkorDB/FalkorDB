/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../udf/utils.h"
#include "../../udf/udf_ctx.h"
#include "../../udf/repository.h"
#include "../../errors/errors.h"

// execute a JavaScript UDF function
// the function is execute is specified as a string at argv[0]
SIValue AR_UDF
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	ASSERT (argv != NULL) ;
	ASSERT (argc >= 1) ;
	ASSERT (private_data == NULL) ;

	const char *func_name = argv[0].stringval ;
	SIValue *args = argv + 1 ;

	//--------------------------------------------------------------------------
	// locate function
	//--------------------------------------------------------------------------

	// get thread local JS context
	JSContext *js_ctx = UDFCtx_GetJSContext () ;
	ASSERT (js_ctx != NULL) ;

	// locate function
	JSValueConst *fn = UDFCtx_GetFunction (func_name) ;
	if (fn == NULL) {
		// it is possible for the function to be missing
		// this can happen if a query is trying to access a UDF as it is being
		// removed via GRAPH.UDF DELETE
		ErrorCtx_SetError(EMSG_UNKNOWN_FUNCTION, func_name) ; 	
		return SI_NullVal() ;
	}

	//--------------------------------------------------------------------------
	// convert arguments
	//--------------------------------------------------------------------------

	JSValue js_argv[argc - 1] ;

	for (int i = 0; i < argc - 1; i++) {
		js_argv[i] = UDF_SIValueToJS (js_ctx, args[i]) ;
	}

	JSValue res = JS_Call (js_ctx, *fn, JS_UNDEFINED, argc-1, js_argv) ;

	SIValue si_res = UDF_JSToSIValue (js_ctx, res) ;
	
	return si_res ;
}

