/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"

// create a new datetime object representing the current datetime
SIValue DateTime_now(void);

// create a new datetime object from a ISO-8601 string datetime representation
SIValue DateTime_fromString
(
	const char *datetime_str  // datetime ISO-8601 string representation
);

// DateTime_fromWeekDate
//
// constructs a datetime object from an ISO 8601 week date representation,
// which includes the year, ISO week number, and day of the week.
//
// according to ISO 8601:
// - Weeks start on Monday (dayOfWeek = 1)
// - Week 1 is the week that contains January 4th (i.e., the first week with a Thursday)
// this function calculates the exact calendar date corresponding to the given
// week and day, and constructs a datetime value including optional time
// components (hour, minute, second, millisecond, microsecond, nanosecond).
//
// example: DateTime_fromWeekDate(1984, 10, 1, 0, 0, 0, 0, 0, 0)
//          → returns March 5, 1984 (Monday of week 10)
SIValue DateTime_fromWeekDate
(
    int year,         // year
    int week,         // ISO week number (1–53), defaults to 1
    int dayOfWeek,    // ISO weekday (1=Mon .. 7=Sun), defaults to 1
    int hour,         // hour defaults to 0
    int minute,       // minute defaults to 0
    int second,       // second defaults to 0
    int millisecond,  // millisecond defaults to 0
    int microsecond,  // microsecond defaults to 0
    int nanosecond    // nanosecond defaults to 0
);

// DateTime_fromQuarterDate
//
// constructs a datetime object from the given year and quarter, where each
// quarter spans three months:
//
// Q1: Jan–Mar, Q2: Apr–Jun, Q3: Jul–Sep, Q4: Oct–Dec
//
// the dayOfQuarter is 1-based and counts total days into the quarter,
// e.g., Q2 day 32 corresponds to June 1.
//
// this function computes the correct calendar date for that day and sets
// the specified time components. The result is returned as an SIValue.
//
// example:
// DateTime_fromQuarterDate(2024, 2, 32, 14, 0, 0, 0, 0, 0)
// → returns June 1, 2024 at 14:00:00
SIValue DateTime_fromQuarterDate
(
	int year,          // year
	int quarter,       // quarter (1-4)
	int dayOfQuarter,  // day within quarter (starts at 1)
	int hour,          // hour defaults to 0
	int minute,        // minute defaults to 0
	int second,        // second defaults to 0
	int millisecond,   // mili-second defaults to 0
	int microsecond,   // micro-second defaults to 0
	int nanosecond     // nano-second defaults to 0
);

// create a new datetime object from individual datetime components
SIValue DateTime_fromComponents
(
	int year,         // year
	int month,        // month defaults to 1
	int day,          // day defaults to 1
	int hour,         // hour defaults to 0
	int minute,       // minute defaults to 0
	int second,       // second defaults to 0
	int millisecond,  // mili-second defaults to 0
	int microsecond,  // micro-second defaults to 0
	int nanosecond    // nano-second defaults to 0
);

// extract component from datetime objects
// available components:
// second, minute, hour, day, month, year, dayOfWeek, ordinalDay
bool DateTime_getComponent
(
	const SIValue *datetime,  // datetime object
	const char *component,    // datetime component to get
	int *value                // [output] component value
);

// get a string representation of datetime
void DateTime_toString
(
	const SIValue *datetime,  // datetime object
	char **buf,               // print buffer
	size_t *bufferLen,        // print buffer length
	size_t *bytesWritten      // the actual number of bytes written to the buffer
);

