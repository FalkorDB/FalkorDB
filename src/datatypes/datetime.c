/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../value.h"

#include <time.h>

// create a new datetime object representing the current date & time
SIValue DateTime_now(void) {
	return (SIValue) {
		.datetimeval = time(NULL), .type = T_DATETIME, .allocation = M_NONE
	};
}

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
) {
	// validate components
	ASSERT(year        >= -999999999 && year        <= 999999999);
	ASSERT(month       >= 1          && month       <= 12);
	ASSERT(day         >= 1          && day         <= 31);
	ASSERT(hour        >= 0          && hour        <= 23);
	ASSERT(minute      >= 0          && minute      <= 59);
	ASSERT(second      >= 0          && second      <= 59);
	ASSERT(millisecond >= 0          && millisecond <= 999);
	ASSERT(microsecond >= 0          && microsecond <= 999999);
	ASSERT(nanosecond  >= 0          && nanosecond  <= 999999999);

	struct tm timeinfo;
    memset(&timeinfo, 0, sizeof(timeinfo));

    timeinfo.tm_year = year - 1900; // tm_year is years since 1900
    timeinfo.tm_mon  = month - 1;
    timeinfo.tm_mday = day;
    timeinfo.tm_hour = hour;
    timeinfo.tm_min  = minute;
    timeinfo.tm_sec  = second;

    // convert to time_t
    time_t sec_since_epoch = mktime(&timeinfo);

	return (SIValue) {
		.datetimeval = sec_since_epoch, .type = T_DATETIME, .allocation = M_NONE
	};
}

// extract component from datetime objects
// available components:
// year, quarter, month, week, weekYear, dayOfQuarter, quarterDay, day,
// ordinalDay, dayOfWeek, weekDay, hour, minute, second, millisecond,
// microsecond and nanosecond
bool DateTime_getComponent
(
	const SIValue *datetime,  // datetime object
	const char *component,    // datetime component to get
	int *value                // [output] component value
) {
	ASSERT(value              != NULL);
	ASSERT(datetime           != NULL);
	ASSERT(component          != NULL);
	ASSERT(SI_TYPE(*datetime) == T_DATETIME);

	// set output
	*value = -1;

	//--------------------------------------------------------------------------
	// convert from time_t to tm
	//--------------------------------------------------------------------------

	struct tm time;
	time_t rawtime = datetime->datetimeval;
	gmtime_r(&rawtime, &time);

	//--------------------------------------------------------------------------
	// extract component
	//--------------------------------------------------------------------------

	if(strcasecmp(component, "second")) {
		// seconds after the minute — [0, 60]
		*value = time.tm_sec;
	} else if(strcasecmp(component, "minute")) {
		// minutes after the hour — [0, 59]
		*value = time.tm_min;
	} else if(strcasecmp(component, "hour")) {
		// hours since midnight — [0, 23]
		*value = time.tm_hour;
	} else if(strcasecmp(component, "day")) {
		// day of the month — [1, 31]
		*value = time.tm_mday;
	} else if(strcasecmp(component, "month")) {
		// months since January — [0, 11]
		*value = time.tm_mon;
	} else if(strcasecmp(component, "year")) {
		// years since 1900
		*value = time.tm_year + 1900;
	} else if(strcasecmp(component, "dayOfWeek")) {
		// days since Sunday — [0, 6]
		*value = time.tm_wday;
	} else if(strcasecmp(component, "ordinalDay")) {
		// days since January 1 — [0, 365]
		*value = time.tm_yday;
	} else {
		// not supported
	}

	return (*value != -1);
}

// get a string representation of datetime
void DateTime_toString
(
	const SIValue *datetime,  // datetime object
	char buffer[30]           // [output] string buffer
) {
	ASSERT(buffer             != NULL);
	ASSERT(datetime           != NULL);
	ASSERT(SI_TYPE(*datetime) == T_DATETIME);

	// get a tm object from time_t
	struct tm time;
	time_t rawtime = datetime->datetimeval;
	gmtime_r(&rawtime, &time);

	// format the date and time up to seconds: 2025-04-14T06:08:21
    strftime(buffer, 30, "%Y-%m-%dT%H:%M:%S", &time);
}

