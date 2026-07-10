/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>
#include "func_desc.h"

// number of built-in functions
#define NUM_BUILTIN_FUNCS 144

// flat array of every built-in descriptor pointer, used by procedures
// that enumerate all functions (e.g. dbms.functions())
extern AR_FuncDesc * const AR_BUILTIN_FUNCS[NUM_BUILTIN_FUNCS];

// Perfect-hash lookup for built-in arithmetic function names.
//
// str  - already lowercased name (e.g. "abs", "starts with")
// len  - strlen(str)
//
// returns a pointer to the static AR_FuncDesc for that function,
// or NULL if the name is not a known built-in
AR_FuncDesc *AR_BuiltinFuncLookup
(
    const char *str,
    size_t len
);
