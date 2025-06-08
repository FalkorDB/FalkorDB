/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"

// create a new datetime object representing the current datetime
SIValue DateTime_now(void);

// create a new date object representing the current date
SIValue Date_now(void);

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
	char buffer[30]           // [output] string buffer
);

