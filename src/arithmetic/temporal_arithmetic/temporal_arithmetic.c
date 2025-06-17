/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../value.h"
#include "../../datatypes/duration.h"

#include <time.h>

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
	
	if(!(a_type & SI_TEMPORAL)) {
		// error!
	}

	if(b_type != T_DURATION) {
		// error!
	}
	
	// return a - b
	// negate b
	Duration_Negate(&b);

	return Temporal_AddDuration(a, b);
}

