/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../udf/utils.h"
#include "../../udf/udf_ctx.h"
#include "../../udf/repository.h"
#include "../../errors/errors.h"

SIValue AR_UDF
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	const char *func_name = argv[0].stringval ;
	SIValue *args = argv + 1 ;

	//--------------------------------------------------------------------------
	// locate function
	//--------------------------------------------------------------------------

	JSContext *js_ctx = UDFCtx_GetJSContext () ;

	JSValue fn = UDF_GetFunction (func_name, js_ctx) ;
	if (JS_IsNull (fn)) {
		const char *script = UDF_RepoGetScript (func_name) ;

		JS_Eval (js_ctx, script, strlen (script), "<input>",
				JS_EVAL_TYPE_GLOBAL) ;

		fn = UDF_GetFunction (func_name, js_ctx) ;
		ASSERT (!JS_IsNull (fn)) ;
	}

	//--------------------------------------------------------------------------
	// convert arguments
	//--------------------------------------------------------------------------

	JSValue js_argv[argc - 1] ;

	for (int i = 0; i < argc - 1; i++) {
		js_argv[i] = UDF_SIValueToJS (js_ctx, args[i]) ;
	}

	JSValue res = JS_Call (js_ctx, fn, JS_UNDEFINED, argc-1, js_argv) ;

	SIValue si_res = UDF_JSToSIValue (js_ctx, res) ;
	
	return si_res ;
}

void Register_UDFFuncs (void) {
	SIType *types ;
	AR_FuncDesc *func_desc ;
	SIType ret_type = SI_ALL ;

	types = array_new (SIType, 1) ;
	// array_append (types, T_STRING) ;
	array_append (types, SI_ALL) ;

	// TODO: UDF is marked as not internal (it is!)
	// this should changed, currently if UDF is set as internal
	// calling functions such as greet() won't work
	func_desc = AR_FuncDescNew ("UDF", AR_UDF, 0, VAR_ARG_LEN, types, ret_type,
			false, false) ;

	AR_SetUDF  (func_desc) ;  // mark function as UDF
	AR_RegFunc (func_desc) ;  // register function
}

