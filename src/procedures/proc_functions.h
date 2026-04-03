/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "proc_ctx.h"

// returns all functions in the systems
//
// this procedure yields:
// 1. function name 
// 2. function return type
// 3. function arguments and their types
// 4. is the function internal
// 5. is the function reducible
// 6. is the function performs aggregation
// 7. is the function a user defined function
ProcedureCtx *Proc_FunctionsCtx (void) ;

