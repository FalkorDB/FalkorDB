/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../../src/redismodule.h"

#include <stdio.h>
#include <stdarg.h>

#define SILENT 1

// mock the RedisModule_Log function for unit tests
// running outside of Redis
void _mock_RedisModule_Log
(
	RedisModuleCtx *ctx,
	const char *level,
	const char *fmt,
	...
) {
#ifdef SILENT
#else
    // initialize the variadic argument list
    va_list args ;
    va_start (args, fmt) ;

    // print a prefix to identify this as a mock log (useful for test debugging)
    printf ("[Mock Log - %s] ", level) ;

    // use vprintf to handle the format string and the variable arguments
    vprintf (fmt, args) ;
    printf ("\n") ;

    // clean up
    va_end (args) ;
#endif
}

// reset RedisModule_Log to use a mock version of it capable of running
// outside of Redis
void Logging_Reset (void) {
	RedisModule_Log = _mock_RedisModule_Log ;
}
