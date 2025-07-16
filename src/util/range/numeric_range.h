/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include <stdbool.h>

typedef struct {
    double min;
    double max;
    bool include_min;
    bool include_max;
    bool valid;
} NumericRange;

NumericRange* NumericRange_New(void);
bool NumericRange_IsValid(const NumericRange *range);
bool NumericRange_ContainsValue(const NumericRange *range, double v);
void NumericRange_TightenRange(NumericRange *range, int op, double v);
void NumericRange_ToString(const NumericRange *range);
void NumericRange_Free(NumericRange *range);
