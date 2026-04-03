/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../value.h"
#include "../../datatypes/duration.h"

#include <time.h>

// negate duration object
// duration.years = -duration.years
// duration.months = -duration.months
// ...
// duration.seconds = -duration.seconds
static void _Duration_Negate
(
	Duration *d
) {
	ASSERT(d != NULL);

	d->years   *= -1;
	d->months  *= -1;
	d->weeks   *= -1;
	d->days    *= -1;
	d->hours   *= -1;
	d->minutes *= -1;
	d->seconds *= -1;
}

// add two duration objects
// c = a + b
static void _Duration_Add
(
	Duration * restrict c,        // [output] a + b
	const Duration * restrict a,  // a
	const Duration * restrict b   // b
) {
	// add duration components
	c->years   = a->years   + b->years;
	c->months  = a->months  + b->months;
	c->weeks   = a->weeks   + b->weeks;
	c->days    = a->days    + b->days;
	c->hours   = a->hours   + b->hours;
	c->minutes = a->minutes + b->minutes;
	c->seconds = a->seconds + b->seconds;
}

// add two duration objects
// return a + b
static SIValue _AddDurations
(
	const SIValue a,  // duration object
	const SIValue b   // duration object
) {
	SIType a_type = SI_TYPE(a);
	SIType b_type = SI_TYPE(b);

	// both a and b must be of type duration
	ASSERT(a_type == T_DURATION && b_type == T_DURATION);

	Duration _c;
	Duration _a = duration_from_time_t_utc(a.longval);
	Duration _b = duration_from_time_t_utc(b.longval);
	_Duration_Add(&_c, &_a, &_b);

	SIValue c = a;
	c.longval = duration_from_epoch_utc(&_c);
	return c;
}

// subtract two duration objects
// return a - b
static SIValue _SubDurations
(
	const SIValue a,  // duration object
	const SIValue b   // duration object
) {
	SIType a_type = SI_TYPE(a);
	SIType b_type = SI_TYPE(b);

	// both a and b must be of type duration
	ASSERT(a_type == T_DURATION && b_type == T_DURATION);

	Duration _c;
	Duration _a = duration_from_time_t_utc(a.longval);
	Duration _b = duration_from_time_t_utc(b.longval);

	// compute _c = _a - _b
	_Duration_Negate(&_b);         // _b = -1 * _b
	_Duration_Add(&_c, &_a, &_b);  // _c = _a + _b

	SIValue c = a;
	c.longval = duration_from_epoch_utc(&_c);
	return c;
}

// adds a Duration to a temporal value (Date, Datetime, Time, or Duration)
// one operand must be a Duration
// returns a new temporal value of the same type as the non-Duration operand
SIValue Temporal_AddDuration
(
	SIValue a,  // Temporal lhs
	SIValue b   // Temporal rhs
) {
	SIType a_type = SI_TYPE(a);
	SIType b_type = SI_TYPE(b);

	ASSERT(a_type & (SI_TEMPORAL | T_DURATION));
	ASSERT(b_type & (SI_TEMPORAL | T_DURATION));
	ASSERT(a_type == T_DURATION || b_type == T_DURATION);

	if(a_type == T_DURATION && b_type == T_DURATION) {
		return _AddDurations(a, b);
	}

	// normalize to temporal + duration
	if(b_type != T_DURATION) {
		SIValue t = a;
		a = b;
		b = t;

		a_type = SI_TYPE(a);
		b_type = SI_TYPE(b);
	}

	SIType ret_type = a_type;
	Duration d = duration_from_time_t_utc(b.longval);

	struct tm temporal;
	time_t t = a.longval;
    gmtime_r(&t, &temporal);

	// apply calendar-based parts
	if(ret_type & (T_DATE | T_DATETIME)) {
		temporal.tm_year += d.years;
		temporal.tm_mon	 += d.months;
		temporal.tm_mday += d.days;
	}

	// apply clock-based parts
	if(ret_type & (T_TIME | T_DATETIME)) {
		temporal.tm_hour += d.hours;
		temporal.tm_min	 += d.minutes;
		temporal.tm_sec  += d.seconds;
	}

	// normalize the result
	time_t normalized = timegm(&temporal);
	a.longval = normalized;

	return a;
}

// subtracts one temporal value from another
// where one operand must be a duration
// returns a new temporal value of the same type as the non-Duration operand
// valid combinations:
//   - Duration/Date/Datetime/Time - Duration
//   - Duration - Duration
// invalid combinations (error):
//   - Duration - Date/Datetime/Time
//   - Date - Date, Datetime - Datetime, etc`
SIValue Temporal_SubDuration
(
	SIValue a,  // Temporal lhs
	SIValue b   // Temporal rhs
) {
	SIType a_type = SI_TYPE(a);
	SIType b_type = SI_TYPE(b);

	// a must be any temporal type
	// b must be duration
	
	ASSERT(a_type & SI_TEMPORAL);
	ASSERT(b_type == T_DURATION);

	//-------------------------------------------------------------------------
	// duration - duration
	//-------------------------------------------------------------------------

	if(a_type == T_DURATION && b_type == T_DURATION) {
		return _SubDurations(a, b);
	}

	//-------------------------------------------------------------------------
	// temporal - duration
	//-------------------------------------------------------------------------

	SIType ret_type = a_type;
	Duration d = duration_from_time_t_utc(b.longval);

	struct tm temporal;
	time_t t = a.longval;
    gmtime_r(&t, &temporal);

	// apply calendar-based parts
	if(ret_type & (T_DATE | T_DATETIME)) {
		temporal.tm_year -= d.years;
		temporal.tm_mon	 -= d.months;
		temporal.tm_mday -= d.days;
	}

	// apply clock-based parts
	if(ret_type & (T_TIME | T_DATETIME)) {
		temporal.tm_hour -= d.hours;
		temporal.tm_min	 -= d.minutes;
		temporal.tm_sec  -= d.seconds;
	}

	// normalize the result
	time_t normalized = timegm(&temporal);
	a.longval = normalized;

	return a;
}

