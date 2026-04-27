/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include <stdint.h>
#include <stdbool.h>

// returns true if year is a leap year (ISO 8601 / Gregorian calendar)
static inline bool is_leap_year
(
	int year
) {
	return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

// new temporal values
/* Create a new timestamp - millis from epoch */
int64_t TemporalValue_NewTimestamp();
