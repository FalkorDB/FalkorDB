/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "duration.h"
#include "../util/rmalloc.h"

#include <stdio.h>
#include <time.h>
#include <math.h>

#define SECONDS_IN_DAY    86400
#define SECONDS_IN_HOUR   3600
#define SECONDS_IN_MINUTE 60

// apply a duration to epoch
time_t duration_from_epoch_utc
(
	const Duration *d  // duration added to epoch
) {
	ASSERT(d != NULL);

	// start from the Unix epoch: Jan 1, 1970 UTC
    struct tm base_tm = {0};
    base_tm.tm_year   = 70;
	base_tm.tm_mday   = 1;

	// separate integral and fractional parts
    int years_int  = (int)truncf(d->years);
    int months_int = (int)truncf(d->months);

	// add whole years and months
    base_tm.tm_year += years_int;
    base_tm.tm_mon  += months_int;

	// normalize struct tm (handles overflowed months)
    time_t base_time = timegm(&base_tm);

	// add fractional year/month durations using average values
    double extra_days = (d->years - years_int) * 365.25 +
                        (d->months - months_int) * 30.44;

	// accumulate all time components into total seconds
    double total_seconds = 0.0;
    total_seconds += (d->weeks * 7 + d->days + extra_days) * SECONDS_IN_DAY;
    total_seconds += d->hours   * SECONDS_IN_HOUR;
    total_seconds += d->minutes * SECONDS_IN_MINUTE;
    total_seconds += d->seconds;

	// add to base_time, ensuring we stay in time_t range
    time_t delta = (time_t)total_seconds;
    return base_time + delta;
}

// convert UTC time_t to duration since epoch
Duration duration_from_time_t_utc
(
	time_t target  // target = epoch + duration
) {
	// epoch: Jan 1, 1970 UTC
    struct tm epoch_tm = {0};
    epoch_tm.tm_year   = 70;
	epoch_tm.tm_mday   = 1;

	// convert target time_t to UTC broken-down time
	struct tm target_tm;
    gmtime_r(&target, &target_tm);

    Duration d = {0};

	// year and month difference
    int year_diff  = target_tm.tm_year - epoch_tm.tm_year;
    int month_diff = target_tm.tm_mon  - epoch_tm.tm_mon;

	// normalize months to [0, 11]
    if (month_diff < 0) {
        year_diff  -= 1;
        month_diff += 12;
    }

    d.years  = (float)year_diff;
    d.months = (float)month_diff;

	// construct anchor date with just year/month components
    struct tm anchor_tm = epoch_tm;
    anchor_tm.tm_year += year_diff;
    anchor_tm.tm_mon  += month_diff;

	// normalize anchor time
    time_t anchor_time = timegm(&anchor_tm);

	// remaining seconds beyond anchor
    time_t delta = target - anchor_time;

    d.days = delta / SECONDS_IN_DAY;
    delta -= (time_t)d.days * SECONDS_IN_DAY;

    d.hours = delta / SECONDS_IN_HOUR;
    delta -= (time_t)d.hours * SECONDS_IN_HOUR;

    d.minutes = delta / SECONDS_IN_MINUTE;
    delta -= (time_t)d.minutes * SECONDS_IN_MINUTE;

    d.seconds = (float)delta;
    return d;
}

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
) {
	SIValue duration;

	Duration d;

	d.years   = years;
	d.months  = months;
	d.weeks   = weeks;
	d.days    = days;
	d.hours   = hours;
	d.minutes = minutes;
	d.seconds = seconds;

	// compute number of seconds from epoch represented by this duration
	// e.g. 
	// duration = {
	//    years:   1,
	//    months:  1,
	//    weeks:   0,
	//    days:    2,
	//    hours:   2,
	//    minutes: 30,
	//    seconds: 20
	// }
	//
	// 1970/01/01-00:00:00 + duration = 1971/02/03-02:30:20
	time_t t = duration_from_epoch_utc(&d);

	return (SIValue) {.longval = t, .type = T_DURATION, .allocation = M_NONE};
}

// extract component from duration object
bool Duration_getComponent
(
    const SIValue *duration,  // duration object
    const char *component,    // duration component to get
    float *value              // [output] component value
) {
    ASSERT(value              != NULL);
    ASSERT(duration           != NULL);
    ASSERT(component          != NULL);
    ASSERT(SI_TYPE(*duration) == T_DURATION);

    // set output
    *value = -1;

    //--------------------------------------------------------------------------
    // convert from time_t to tm
    //--------------------------------------------------------------------------

	Duration d = duration_from_time_t_utc(duration->longval);

    //--------------------------------------------------------------------------
    // extract component
    //--------------------------------------------------------------------------

    if(strcasecmp(component, "seconds") == 0) {
        *value = d.seconds;
    } else if(strcasecmp(component, "minutes") == 0) {
        *value = d.minutes;
    } else if(strcasecmp(component, "hours") == 0) {
        *value = d.hours;
    } else if(strcasecmp(component, "days") == 0) {
        *value = d.days;
    } else if(strcasecmp(component, "weeks") == 0) {
        *value = d.weeks;
    } else if(strcasecmp(component, "months") == 0) {
        *value = d.months;
    } else if(strcasecmp(component, "years") == 0) {
        *value = d.years;
    } else {
        // not supported
        return false;
    }

    return true;
}

// helper macro to safely append to buffer
#define APPEND(fmt, ...)                                                                      \
    do {                                                                                      \
		if((*bufferLen - *bytesWritten) < 32) {                                               \
			*bufferLen += 32;                                                                 \
			*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);                               \
		}                                                                                     \
        int n = snprintf(*buf + *bytesWritten, *bufferLen - *bytesWritten, fmt, __VA_ARGS__); \
        *bytesWritten += n;                                                                   \
    } while (0)

// get a string representation of duration
void Duration_toString
(
	const SIValue *duration,  // duration object
	char **buf,               // print buffer
	size_t *bufferLen,        // print buffer length
	size_t *bytesWritten      // actual number of bytes written to the buffer
) {
    ASSERT(duration           != NULL);
	ASSERT(buf                != NULL);
	ASSERT(bufferLen          != NULL);
	ASSERT(bytesWritten       != NULL);
	ASSERT(SI_TYPE(*duration) == T_DURATION);

	Duration d = duration_from_time_t_utc(duration->longval);

    // print in a simplified ISO-8601-like format: PnYnMnWnDTnHnMnS
    APPEND("%s", "P");

    const float years   = d.years;
    const float months  = d.months;
    const float weeks   = d.weeks;
    const float days    = d.days;
    const float hours   = d.hours;
    const float minutes = d.minutes;
    const float seconds = d.seconds;

    // calendar part
    if(years  != 0.0f) APPEND("%.9gY", years);
    if(months != 0.0f) APPEND("%.9gM", months);
    if(weeks  != 0.0f) APPEND("%.9gW", weeks);
    if(days   != 0.0f) APPEND("%.9gD", days);

    // time part
    if(hours != 0.0f || minutes != 0.0f || seconds != 0.0f) {
        APPEND("%s", "T");
        if(hours   != 0.0f) APPEND("%.9gH", hours);
        if(minutes != 0.0f) APPEND("%.9gM", minutes);
        if(seconds != 0.0f) APPEND("%.9gS", seconds);
    }
}

// negate duration object
// duration.years = -duration.years
// duration.months = -duration.months
// ...
// duration.seconds = -duration.seconds
void Duration_Negate
(
	SIValue *duration
) {
	ASSERT(duration != NULL);
	ASSERT(SI_TYPE(*duration) == T_DURATION);

	Duration d = duration_from_time_t_utc(duration->longval);

	d.years   = -d.years;
	d.months  = -d.months;
	d.weeks   = -d.weeks;
	d.days    = -d.days;
	d.hours   = -d.hours;
	d.minutes = -d.minutes;
	d.seconds = -d.seconds;

	// apply a duration to epoch
	duration->longval = duration_from_epoch_utc(&d);
}

