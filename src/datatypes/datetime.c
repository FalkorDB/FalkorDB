/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../value.h"
#include "../util/rmalloc.h"

#define _XOPEN_SOURCE
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

// create a new datetime object representing the current date & time
SIValue DateTime_now(void) {
	return SI_DateTime(time(NULL));
}

// create a new date object representing the current date
SIValue Date_now(void) {
	return (SIValue) {
		.datetimeval = time(NULL), .type = T_DATE, .allocation = M_NONE
	};
}

// parse ISO 8601 datetime string into time_t
// handles various ISO 8601 formats:
// - YYYY-MM-DD
// - YYYY-MM-DDThh:mm
// - YYYY-MM-DDThh:mm:ss
// - YYYY-MM-DDThh:mm:ss.sss
// - with or without timezone indicators (Z, +hh:mm, -hh:mm)
static time_t _parse_iso8601
(
	char *datetime_str
) {
	ASSERT(datetime_str != NULL);

    char *result;
    struct tm tm          = {0};
    bool has_timezone     = false;
    int tz_offset_hours   = 0;
    int tz_offset_minutes = 0;
    
    // check for timezone indicator
    char *tz_indicator = NULL;
    if((tz_indicator = strrchr(datetime_str, 'Z')) != NULL &&
	   *(tz_indicator+1) == '\0') {
        // UTC timezone (Z)
        has_timezone = true;
    } else if((tz_indicator = strrchr(datetime_str, '+')) != NULL) {
        // positive timezone offset
        has_timezone = true;
        sscanf(tz_indicator, "+%d:%d", &tz_offset_hours, &tz_offset_minutes);
    } else if((tz_indicator = strrchr(datetime_str, '-')) != NULL &&
			   tz_indicator > datetime_str + 4) {
        // negative timezone offset
		// (checking position to avoid confusing with date separators)
        has_timezone = true;
        sscanf(tz_indicator, "-%d:%d", &tz_offset_hours, &tz_offset_minutes);
    }
    
    // truncate the timezone part if present
    if(has_timezone && tz_indicator != NULL) {
       datetime_str[tz_indicator - datetime_str] = '\0';
    }
    
    // try different ISO 8601 formats, from most specific to least
    const char *formats[] = {
        "%Y-%m-%dT%H:%M:%S.%f",  // with fractional seconds
        "%Y-%m-%dT%H:%M:%S",     // without fractional seconds
        "%Y-%m-%dT%H:%M",        // without seconds
        "%Y-%m-%d",              // date only
        "%Y%m%dT%H%M%S",         // basic format (no separators)
        "%Y%m%d"                 // basic format, date only
    };
    
    // try each format
    for(int i = 0; i < sizeof(formats)/sizeof(formats[0]); i++) {
        memset(&tm, 0, sizeof(struct tm));
        result = strptime(datetime_str, formats[i], &tm);
        if(result != NULL) {
            // parsed successfully
            // apply timezone offset
            if(has_timezone) {
                // for Z or positive offset, subtract from local time
                // for negative offset, add to local time
                if(tz_indicator && *tz_indicator == '-') {
                    tm.tm_hour += tz_offset_hours;
                    tm.tm_min  += tz_offset_minutes;
                } else {
                    tm.tm_hour -= tz_offset_hours;
                    tm.tm_min  -= tz_offset_minutes;
                }
            }
            
            // convert to time_t
            return mktime(&tm);
        }
    }
    
    return (time_t)-1;  // return error if no format matched
}

// create a new datetime object from a ISO-8601 string datetime representation
SIValue DateTime_fromString
(
	char *datetime_str  // datetime ISO-8601 string representation
) {
	time_t parsed_time = _parse_iso8601(datetime_str);
	if(parsed_time != (time_t)-1) {
		return SI_DateTime(parsed_time);
	}

	// failed to parse datetime, return NULL
	return SI_NullVal();
}

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
) {
    // Validate inputs
    ASSERT(year        >= -999999999 && year        <= 999999999);
    ASSERT(week        >= 1          && week        <= 53);
    ASSERT(dayOfWeek   >= 1          && dayOfWeek   <= 7);  // ISO: 1=Mon, ..., 7=Sun
    ASSERT(hour        >= 0          && hour        <= 23);
    ASSERT(minute      >= 0          && minute      <= 59);
    ASSERT(second      >= 0          && second      <= 59);
    ASSERT(millisecond >= 0          && millisecond <= 999);
    ASSERT(microsecond >= 0          && microsecond <= 999999);
    ASSERT(nanosecond  >= 0          && nanosecond  <= 999999999);

    struct tm tm_anchor = {0};

    // step 1: January 4th defines ISO week 1
    tm_anchor.tm_year = year - 1900;
    tm_anchor.tm_mon  = 0;  // January
    tm_anchor.tm_mday = 4;
    tm_anchor.tm_hour = 0;
    tm_anchor.tm_min  = 0;
    tm_anchor.tm_sec  = 0;

    // normalize to get weekday
    timegm(&tm_anchor);

    // step 2: calculate ISO weekday (Monday = 1, Sunday = 7)
    int iso_wday = tm_anchor.tm_wday == 0 ? 7 : tm_anchor.tm_wday;

    // step 3: move to Monday of ISO week 1
    tm_anchor.tm_mday -= (iso_wday - 1);
    timegm(&tm_anchor); // normalize again

    // step 4: add (week - 1) * 7 + (dayOfWeek - 1) days
    tm_anchor.tm_mday += (week - 1) * 7 + (dayOfWeek - 1);
    tm_anchor.tm_hour = hour;
    tm_anchor.tm_min = minute;
    tm_anchor.tm_sec = second;

    // final normalization
    time_t sec_since_epoch = timegm(&tm_anchor);

    // optional: use or store sub-second fields if your system supports them
    // Note: SI_DateTime might need to be extended to accept high precision

    return SI_DateTime(sec_since_epoch);
}

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
) {
	// Validate inputs
    ASSERT(year         >= -999999999 && year        <= 999999999);
    ASSERT(quarter      >= 1          && quarter     <= 4);
    ASSERT(dayOfQuarter >= 1          && dayOfQuarter <= 92);  // conservative limit
    ASSERT(hour         >= 0          && hour        <= 23);
    ASSERT(minute       >= 0          && minute      <= 59);
    ASSERT(second       >= 0          && second      <= 59);
    ASSERT(millisecond  >= 0          && millisecond <= 999);
    ASSERT(microsecond  >= 0          && microsecond <= 999999);
    ASSERT(nanosecond   >= 0          && nanosecond  <= 999999999);

    struct tm tm_date = {0};

    // quarter base month: Q1=Jan (0), Q2=Apr (3), Q3=Jul (6), Q4=Oct (9)
    int base_month = (quarter - 1) * 3;

    tm_date.tm_year = year - 1900;
    tm_date.tm_mon  = base_month;
    tm_date.tm_mday = dayOfQuarter;  // may overflow into later months
    tm_date.tm_hour = hour;
    tm_date.tm_min  = minute;
    tm_date.tm_sec  = second;

    // convert to time_t using UTC to avoid local time shifts
    time_t sec_since_epoch = timegm(&tm_date);

    return SI_DateTime(sec_since_epoch);
}

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

	// mktime doesn not support dates before 1900
    // time_t sec_since_epoch = mktime(&timeinfo);  

    time_t sec_since_epoch = timegm(&timeinfo);

	return SI_DateTime(sec_since_epoch);
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

	if(strcasecmp(component, "second") == 0) {
		// seconds after the minute — [0, 60]
		*value = time.tm_sec;
	} else if(strcasecmp(component, "minute") == 0) {
		// minutes after the hour — [0, 59]
		*value = time.tm_min;
	} else if(strcasecmp(component, "hour") == 0) {
		// hours since midnight — [0, 23]
		*value = time.tm_hour;
	} else if(strcasecmp(component, "day") == 0) {
		// day of the month — [1, 31]
		*value = time.tm_mday;
	} else if(strcasecmp(component, "month") == 0) {
		// months since January — [0, 11]
		*value = time.tm_mon + 1;
	} else if(strcasecmp(component, "year") == 0) {
		// years since 1900
		*value = time.tm_year + 1900;
	} else if(strcasecmp(component, "dayOfWeek") == 0) {
		// days since Sunday — [0, 6]
		*value = time.tm_wday;
	} else if(strcasecmp(component, "ordinalDay") == 0) {
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
	char **buf,               // print buffer
	size_t *bufferLen,        // print buffer length
	size_t *bytesWritten      // the actual number of bytes written to the buffer
) {
	ASSERT(buf                != NULL);
	ASSERT(datetime           != NULL);
	ASSERT(SI_TYPE(*datetime) == T_DATETIME);

	if(*bufferLen - *bytesWritten < 32) {
		*bufferLen += 32;
		*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);
	}

	// get a tm object from time_t
	struct tm time;
	time_t rawtime = datetime->datetimeval;
	gmtime_r(&rawtime, &time);

	// format the date and time up to seconds: 2025-04-14T06:08:21
	*bytesWritten += strftime(*buf + *bytesWritten, *bufferLen,
			"%Y-%m-%dT%H:%M:%S", &time);
	ASSERT(*bytesWritten > 0);
}

// get a string representation of date
void Date_toString
(
	const SIValue *date,  // date object
	char **buf,           // print buffer
	size_t *bufferLen,    // print buffer length
	size_t *bytesWritten  // the actual number of bytes written to the buffer
) {
	ASSERT(buf            != NULL);
	ASSERT(date           != NULL);
	ASSERT(SI_TYPE(*date) == T_DATE);

	if(*bufferLen - *bytesWritten < 32) {
		*bufferLen += 32;
		*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);
	}

	// get a tm object from time_t
	struct tm time;
	time_t rawtime = date->datetimeval;
	gmtime_r(&rawtime, &time);

	// format the date and time up to seconds: 2025-04-14T06:08:21
	*bytesWritten += strftime(*buf + *bytesWritten, *bufferLen,
			"%Y-%m-%d", &time);
	ASSERT(*bytesWritten > 0);
}

