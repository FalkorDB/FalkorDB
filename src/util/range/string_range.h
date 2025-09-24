/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdbool.h>

typedef struct {
    char *min;
    char *max;
    bool include_min;
    bool include_max;
    bool valid;
} StringRange;

StringRange* StringRange_New(void);
bool StringRange_IsValid(const StringRange *range);
bool StringRange_ContainsValue(const StringRange *range, const char *v);
void StringRange_TightenRange(StringRange *range, int op, const char *v);
void StringRange_ToString(const StringRange *range);
void StringRange_Free(StringRange *range);
