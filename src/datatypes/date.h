/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"

// create a new date object representing the current date
SIValue Date_now(void);

// create a new date object from a ISO-8601 string time representation
SIValue Date_fromString
(
	const char *date_str  // date string representation
);

// extract component from date objects
// available components:
// year, quarter, month, week, weekYear, dayOfQuarter, quarterDay, day,
// ordinalDay, dayOfWeek, weekDay
bool Date_getComponent
(
	const SIValue *date,    // date object
	const char *component,  // date component to get
	int *value              // [output] component value
);

// get a string representation of date
void Date_toString
(
	const SIValue *date,  // date object
	char **buf,           // print buffer
	size_t *bufferLen,    // print buffer length
	size_t *bytesWritten  // the actual number of bytes written to the buffer
);

