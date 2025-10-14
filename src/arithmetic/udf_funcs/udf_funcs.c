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

#include <time.h>

// get current time in ms as a unix timestamp
static int64_t _current_time_in_ms(void) {
	struct timespec ts;

    int res = clock_gettime (CLOCK_REALTIME, &ts) ;
	ASSERT (res != -1) ;

    // convert to milliseconds
    int64_t ms = (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000 ;

    return ms;
}

// JS interrupt callback
static int js_interrupt_handler
(
	JSRuntime *rt,  // js runtime
	void *opaque    // handler private data
) {
    int64_t now = _current_time_in_ms () ;
	int64_t deadline_ms = *(int64_t*)(opaque) ;

	// check to see if we're passed the deadline
    if (now > deadline_ms) {
        // returning non-zero aborts execution
        return 1 ;
    }

	// aborts execution
    return 0 ;
}

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
	JSRuntime *js_rt  = UDFCtx_GetJSRuntime () ;
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

	// setup interrupt handler
	int64_t deadline_ms = _current_time_in_ms() + 3; // 3 seconds
	JS_SetInterruptHandler(js_rt, js_interrupt_handler, &deadline_ms) ;

	// invoke UDF
	JSValue res = JS_Call (js_ctx, *fn, JS_UNDEFINED, argc-1, js_argv) ;

	// disable the interrupt handler
	JS_SetInterruptHandler(js_rt, NULL, NULL);

	// free args
	for (int i = 0; i < argc - 1; i++) {
		JS_FreeValue (js_ctx, js_argv[i]) ;
	}

	SIValue si_res = UDF_JSToSIValue (js_ctx, res) ;

	JS_FreeValue (js_ctx, res) ;
	
	return si_res ;
}

