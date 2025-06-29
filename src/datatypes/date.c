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
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// helper: construct an SIValue date from a struct tm
static SIValue _create_date_from_tm
(
	struct tm *t
) {
    // force UTC interpretation
    time_t ts = timegm(t);

    return (SIValue) {.datetimeval = ts, .type = T_DATE, .allocation = M_NONE};
}

// parse ordinal date: YYYYDDD
static bool _parse_ordinal_date
(
	const char *s,
	struct tm *out
) {
    if (strlen(s) != 7) {
		return false;
	}

    char year_buf[5], day_buf[4];
    strncpy(year_buf, s, 4);
	year_buf[4] = '\0';

    strncpy(day_buf, s + 4, 3);
	day_buf[3] = '\0';

    int year = atoi(year_buf);
    int day_of_year = atoi(day_buf);
    if (day_of_year < 1 || day_of_year > 366) {
		return false;
	}

    memset(out, 0, sizeof(struct tm));
    out->tm_year = year - 1900;
    out->tm_mday = day_of_year;

    // normalize day-of-year into calendar date
    timegm(out);
    return true;
}

// parse week date: YYYY-Www or YYYY-Www-D
static bool _parse_week_date
(
	const char *s,
	struct tm *out
) {
    int year, week, weekday = 1;  // default to Monday
    int parsed = sscanf(s, "%4d-W%2d-%1d", &year, &week, &weekday);

    if (parsed < 2) {
        return false;
    }

    if (weekday < 1 || weekday > 7 || week < 1 || week > 53) {
        return false;
    }

    // start from Jan 4 of that year (always in week 1)
    memset(out, 0, sizeof(struct tm));
    out->tm_year = year - 1900;
    out->tm_mon  = 0;
    out->tm_mday = 4;
    timegm(out);

    int jan4_wday = out->tm_wday;
    if (jan4_wday == 0) {
        jan4_wday = 7;  // sunday is 7 in ISO week
    }

    int days_offset = (week - 1) * 7 + (weekday - jan4_wday);
    out->tm_mday += days_offset;
    timegm(out);

    return true;
}

// create a new date object representing the current date
SIValue Date_now(void) {
	return SI_Date(time(NULL));
}

// create a new date object from a ISO-8601 string time representation
SIValue Date_fromString
(
	char *date_str  // date string representation
) {
    struct tm t;
    memset(&t, 0, sizeof(struct tm));

    // try full date: YYYY-MM-DD
    if (strptime(date_str, "%Y-%m-%d", &t)) {
        return _create_date_from_tm(&t);
	}

    // try short calendar: YYYY-MM
    if (strptime(date_str, "%Y-%m", &t)) {
        t.tm_mday = 1;
        return _create_date_from_tm(&t);
    }

    // try basic format: YYYYMMDD
    if (strlen(date_str) == 8 && strptime(date_str, "%Y%m%d", &t)) {
        return _create_date_from_tm(&t);
	}

    // try basic format: YYYYMM
    if (strlen(date_str) == 6 && strptime(date_str, "%Y%m", &t)) {
        t.tm_mday = 1;
        return _create_date_from_tm(&t);
    }

    // try year only
    if (strlen(date_str) == 4 && isdigit(date_str[0])) {
        int year = atoi(date_str);

        t.tm_year = year - 1900;
        t.tm_mon  = 0;
        t.tm_mday = 1;
        return _create_date_from_tm(&t);
    }

    // try ordinal date: YYYYDDD
    if (_parse_ordinal_date(date_str, &t)) {
        return _create_date_from_tm(&t);
	}

    // try week date: YYYY-Www-D or YYYY-Www
    if (_parse_week_date(date_str, &t)) {
        return _create_date_from_tm(&t);
	}

    // invalid input: return NULL
    return SI_NullVal();
}

// extract component from date objects
// available components:
// year, quarter, month, week, weekYear, dayOfQuarter, quarterDay, day,
// ordinalDay, dayOfWeek, weekDay
bool Date_getComponent
(
	const SIValue *date,    // date object
	const char *component,  // date component to get
	int *value              // [output] component value
) {
    ASSERT(value          != NULL);
    ASSERT(date           != NULL);
    ASSERT(component      != NULL);
    ASSERT(SI_TYPE(*date) == T_DATE);

    *value = -1;

    struct tm time;
    time_t rawtime = date->datetimeval;
    gmtime_r(&rawtime, &time);

    int year  = time.tm_year + 1900;
    int month = time.tm_mon + 1;      // 1–12
    int day   = time.tm_mday;
    int yday  = time.tm_yday + 1;     // 1–366
    int wday  = time.tm_wday;         // 0–6 (Sun–Sat)

    if(strcasecmp(component, "day") == 0) {
        *value = day;
    } else if(strcasecmp(component, "month") == 0) {
        *value = month;
    } else if(strcasecmp(component, "year") == 0) {
        *value = year;
    } else if(strcasecmp(component, "dayOfWeek") == 0) {
        *value = wday;
    } else if(strcasecmp(component, "ordinalDay") == 0) {
        *value = yday;
    } else if(strcasecmp(component, "weekDay") == 0) {
        // ISO weekday: Monday=1, Sunday=7
        *value = wday == 0 ? 7 : wday;
    } else if(strcasecmp(component, "week")     == 0 ||
			  strcasecmp(component, "weekYear") == 0) {
        // ISO 8601 week and week-based year
        struct tm temp = time;
        // adjust day to Thursday of the current week
        temp.tm_mday -= (wday + 6) % 7 - 3;
        mktime(&temp);
        int week = (temp.tm_yday / 7) + 1;

        if(strcasecmp(component, "week") == 0) {
            *value = week;
		} else {
            *value = temp.tm_year + 1900;
		}
    } else if(strcasecmp(component, "quarter") == 0) {
        *value = (month - 1) / 3 + 1;
    } else if(strcasecmp(component, "dayOfQuarter") == 0 ||
			  strcasecmp(component, "quarterDay")   == 0) {
        int doy = yday;
        int q = (month - 1) / 3 + 1;
        int dayOfQuarter = doy;

        if(q > 1) {
            // subtract days of prior months in current year
            static const int daysUntilQuarter[] = {0, 0, 90, 181, 273};
            dayOfQuarter = doy - daysUntilQuarter[q];
        }
        *value = dayOfQuarter + 1;
    }

    return (*value != -1);
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

