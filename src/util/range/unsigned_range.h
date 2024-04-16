/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

typedef struct {
	uint64_t min;
	uint64_t max;
	bool include_min;
	bool include_max;
	bool valid;
} UnsignedRange;

UnsignedRange *UnsignedRange_New(void);
void UnsignedRange_Reset(UnsignedRange *range);
UnsignedRange *UnsignedRange_Clone(const UnsignedRange *range);
bool UnsignedRange_IsValid(const UnsignedRange *range);
bool UnsignedRange_ContainsValue(const UnsignedRange *range, uint64_t v);
void UnsignedRange_TightenRange(UnsignedRange *range, int op, uint64_t v);
void UnsignedRange_ToString(const UnsignedRange *range);
void UnsignedRange_Free(UnsignedRange *range);
