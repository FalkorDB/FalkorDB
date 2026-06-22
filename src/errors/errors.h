# Fix for Issue #636: Crash found in fuzzer

/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>
#include <stdbool.h>

// Error message for same alias with different labels
#define EMSG_SAME_ALIAS_DIFFERENT_LABELS "Variable '%s' used with contradictory labels in variable-length path pattern"

// Error context functions
void ErrorCtx_SetError(const char *fmt, ...);
bool ErrorCtx_EncounteredError(void);
char *ErrorCtx_GetError(void);
void ErrorCtx_Clear(void);