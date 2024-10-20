/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */


#pragma once

#include "proc_ctx.h"

// Returns all the system functions. 
// This procedure yields the function name, description and signature.

// Example:
// 1) "tostring"
// 2) "tostring: Datetime|Duration -> String|Null"
// 3) "Returns a string when expr evaluates to a string\nConverts an integer, float, Boolean, string, or point to a string representation\nReturns null when expr evaluates to null\nEmit an error on other types"
// 4) "List|Null"
// 5) "[Datetime|Duration]" 

ProcedureCtx *Proc_FunctionsCtx();


// format into buf the signature of a function with return type 'ret' and argument types 'args'
// the format of the result is: 'name: t1 ... tn -> tn+1'
// Example: "tostring: Datetime|Duration -> String|Null"
void SITypes_SignatureToString(const char * fName, SIType ret, SIType *args, char *buf, size_t bufferLen);
