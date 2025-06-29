/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"

// Duration represents a human-readable span of time,
// broken down into calendar and clock components
//
// notes:
// - each field stores the amount in that unit
//   (e.g., 1.5 hours means 1 hour and 30 minutes)
//
// - units are stored as float to allow fractional values (e.g., 2.5 days)
// - calendar units (years, months) are variable in actual length 
//   depending on context (e.g., leap years, different month lengths)
//   so they should be interpreted accordingly
//
// - overlapping units (e.g., weeks and days) are not mutually exclusive;
//   they are additive unless otherwise specified by the application logic
//
// - this struct is useful for representing durations in a way similar to
//   ISO 8601 durations (e.g., "P1Y2M3W4DT5H6M7.8S") but does not enforce
//   normalization or conversion to exact time spans
typedef struct {
	float years;    // number of years
	float months;   // number of months
	float weeks;    // number of weeks
	float days;     // number of days
	float hours;    // number of hours
	float minutes;  // number of minutes
	float seconds;  // number of seconds
} Duration;

// create a new duration SIValue
SIValue Duration_New
(
	float years,    // number of years
	float months,   // number of months
	float weeks,    // number of weeks
	float days,     // number of days
	float hours,    // number of hours
	float minutes,  // number of minutes
	float seconds   // number of seconds
);

// extract component from duration object
bool Duration_getComponent
(
    const SIValue *duration,  // duration object
    const char *component,    // duration component to get
    float *value              // [output] component value
);

// apply a duration to epoch
time_t duration_from_epoch_utc
(
	const Duration *d  // duration added to epoch
);

// convert UTC time_t to duration since epoch
Duration duration_from_time_t_utc
(
	time_t target  // target = epoch + duration
);

// get a string representation of duration
void Duration_toString
(
	const SIValue *duration,  // duration object
	char **buf,               // print buffer
	size_t *bufferLen,        // print buffer length
	size_t *bytesWritten      // actual number of bytes written to the buffer
);

