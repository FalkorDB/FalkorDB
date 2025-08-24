/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// create a new time object representing the current time
SIValue Time_now(void);

// create a new time object from a ISO-8601 string time representation
SIValue Time_fromString
(
	const char *time_str  // time ISO-8601 string representation
);

// extract component from time objects
// available components:
// second, minute, hour
bool Time_getComponent
(
	const SIValue *time,    // time object
	const char *component,  // time component to get
	int *value              // [output] component value
);

// get a string representation of time
void Time_toString
(
	const SIValue *time,  // time object
	char **buf,           // print buffer
	size_t *bufferLen,    // print buffer length
	size_t *bytesWritten  // the actual number of bytes written to the buffer
);

