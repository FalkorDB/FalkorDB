/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "time_funcs.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../errors/errors.h"
#include "../../datatypes/map.h"
#include "../../datatypes/time.h"
#include "../../datatypes/date.h"
#include "../../datatypes/duration.h"
#include "../../datatypes/datetime.h"
#include "../../datatypes/temporal_value.h"

// returns a timestamp - millis from epoch
SIValue AR_TIMESTAMP
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	return SI_LongVal(TemporalValue_NewTimestamp());
}

// extract datetime component from map
static bool _get_component
(
	const SIValue *map,     // map
	const char *component,  // key to extract
	int *v,                 // [output] value
	int min_val,            // minimum valid value
	int max_val             // maximum valid value
) {
	ASSERT(v         != NULL);
	ASSERT(map       != NULL);
	ASSERT(component != NULL);

	// extract component
	SIValue _v;
	bool exists = MAP_GETCASEINSENSITIVE(*map, component, _v);

	// validate component type
	if(exists) {
		if(SI_TYPE(_v) != T_INT64) {
			ErrorCtx_SetError("%s must be an integer value", component);
		} else {
			*v = _v.longval;
			if(*v < min_val || *v > max_val) {
				ErrorCtx_SetError("Invalid value for %s (valid values %d - %d)",
						component, min_val, max_val);
			}
		}
	}

	return exists;
}

// extract duration component from map
static bool _duration_get_component
(
	const SIValue *map,     // map
	const char *component,  // key to extract
	float *v                // [output] value
) {
	ASSERT(v         != NULL);
	ASSERT(map       != NULL);
	ASSERT(component != NULL);

	// extract component
	SIValue _v;
	bool exists = MAP_GETCASEINSENSITIVE(*map, component, _v);

	// validate component type
	if(exists) {
		if(!(SI_TYPE(_v) & (SI_NUMERIC))) {
			ErrorCtx_SetError("%s must be a numerical value", component);
		} else {
			*v = SI_GET_NUMERIC(_v);
		}
	}

	return exists;
}

// create a new localtime object
SIValue AR_LOCALTIME
(
	SIValue *argv,      // arguments
	int argc,           // number of arguments
	void *private_data  // private data
) {
	if(argc == 0) {
		// return the current time
		return Time_now();
	}

	SIValue arg = argv[0];

	if(SIValue_IsNull(arg)) {
		return SI_NullVal();
	}

	SIValue time;  // time value

	SIType t = SI_TYPE(arg);
	ASSERT(t == T_STRING || t == T_MAP);

	if(t == T_STRING) {
		time = Time_fromString(arg.stringval);
		if(SIValue_IsNull(time)) {
			ErrorCtx_SetError("Failed to parse datetime");
		}
		return time;
	} else {
		// create datetime object from map
		// datetime components
		int hour         = 0;  // hour defaults to 0
		int minute       = 0;  // minute defaults to 0
		int second       = 0;  // second defaults to 0
		int millisecond  = 0;  // mili-second defaults to 0
		int nanosecond   = 0;  // nano-second defaults to 0
		int microsecond  = 0;  // micro-second defaults to 0

		uint n_keys = Map_KeyCount(arg);

		//----------------------------------------------------------------------
		// extract individual datetime elements
		//----------------------------------------------------------------------

		bool hour_specified         = _get_component(&arg, "hour",         &hour,         0, 23);
		bool minute_specified       = _get_component(&arg, "minute",       &minute,       0, 59);
		bool second_specified       = _get_component(&arg, "second",       &second,       0, 59);
		bool milisecond_specified   = _get_component(&arg, "millisecond",  &millisecond,  0, 999);
		bool microsecond_specified  = _get_component(&arg, "microsecond",  &microsecond,  0, 999999);
		bool nanosecond_specified   = _get_component(&arg, "nanosecond",   &nanosecond,   0, 999999999);

		// make sure no unexpected keys exists in the map
		n_keys -= hour_specified + minute_specified +
				  second_specified + milisecond_specified +
				  microsecond_specified + nanosecond_specified;

		if(hour_specified == false) {
			ErrorCtx_SetError("hour must be specified");
			return SI_NullVal();
		}

		if(minute_specified == false && second_specified == true) {
			ErrorCtx_SetError("second cannot be specified without minute");
			return SI_NullVal();
		}

		// error incase components map contains an unknown key
		if(n_keys > 0) {
			ErrorCtx_SetError("datetime components map contains an unknown key");
			return SI_NullVal();
		}

		time = DateTime_fromComponents(1900, 1, 1, hour, minute, second,
				millisecond, microsecond, nanosecond);

		time.type = T_TIME;
	}

	return time;
}

// create a new datetime object
SIValue AR_DATE
(
	SIValue *argv,      // arguments
	int argc,           // number of arguments
	void *private_data  // private data
) {
	if(argc == 0) {
		// return datetime representing the current date and time
		return Date_now();
	}

	SIValue arg = argv[0];
	
	if(SIValue_IsNull(arg)) {
		return SI_NullVal();
	}

	SIValue d;  // date value

	SIType t = SI_TYPE(arg);
	ASSERT(t == T_STRING || t == T_MAP);

	if(t == T_STRING) {
		d = Date_fromString(arg.stringval);
		if(SIValue_IsNull(d)) {
			ErrorCtx_SetError("Failed to parse date");
		}
		return d;
	} else {
		// create datetime object from map
		// datetime components
		int year;              // mandatory
		int month        = 1;  // month defaults to 1
		int quarter      = 1;  // quarter defaults to 1
		int dayOfQuarter = 1;  // day of quarter defaults to 1
		int week         = 1;  // week defaults to 1
		int day          = 1;  // day defaults to 1
		int dayOfWeek    = 1;  // day of week defaults to 1

		uint n_keys = Map_KeyCount(arg);

		//----------------------------------------------------------------------
		// extract individual datetime elements
		//----------------------------------------------------------------------

		// extract year
		bool year_specified = _get_component(&arg, "year", &year, -999999999, 999999999);

		if(!year_specified) {
			// year is mandatory
			ErrorCtx_SetError("year must be specified");
			return SI_NullVal();
		}

		bool quarter_specified      = _get_component(&arg, "quarter",      &quarter,      1, 4);
		bool dayOfQuarter_specified = _get_component(&arg, "dayOfQuarter", &dayOfQuarter, 1, 92);
		bool month_specified        = _get_component(&arg, "month",        &month,        1, 12);
		bool week_specified         = _get_component(&arg, "week",         &week,         1, 53);
		bool day_specified          = _get_component(&arg, "day",          &day,          1, 31);
		bool dayOfWeek_specified    = _get_component(&arg, "dayOfWeek",    &dayOfWeek,    1, 7);

		n_keys -= year_specified + quarter_specified + dayOfQuarter_specified +
				  month_specified + week_specified + dayOfWeek_specified +
				  day_specified;

		// can't specify both week and month
		// as month is determine from the week number
		if(week_specified == true && month_specified == true) {
			ErrorCtx_SetError("month cannot be specified with week");	
			return SI_NullVal();
		}

		// can't specify dayOfQuarter_specified without specifying quarter
		if(dayOfQuarter_specified == true && quarter_specified == false) {
			ErrorCtx_SetError("quarter/dayOfQuarter cannot be specified with day/month/week");
			return SI_NullVal();
		}

		// can't specify quarter with day or week
		if((dayOfQuarter_specified == true || quarter_specified == true) && (
					day_specified       == true ||
					month_specified     == true ||
					week_specified      == true ||
					dayOfWeek_specified == true)) {
			ErrorCtx_SetError("dayOfWeek cannot be specified without week");	
			return SI_NullVal();
		}

		// can't specify day of week without specifying week number
		if(dayOfWeek_specified == true && week_specified == false) {
			ErrorCtx_SetError("dayOfWeek cannot be specified without week");	
			return SI_NullVal();
		}

		// if day is specified month must be specified as well
		if(day_specified == true && month_specified == false) {
			ErrorCtx_SetError("day cannot be specified without month");	
			return SI_NullVal();
		}

		// error incase components map contains an unknown key
		if(n_keys > 0) {
			ErrorCtx_SetError("datetime components map contains an unknown key");	
			return SI_NullVal();
		}

		if(week_specified) {
			d = DateTime_fromWeekDate(year, week, dayOfWeek, 0, 0, 0, 0, 0, 0);
		}

		else if(quarter_specified) {
			d = DateTime_fromQuarterDate(year, quarter, dayOfQuarter, 0, 0, 0,
					0, 0, 0);
		}

		else {
			d = DateTime_fromComponents(year, month, day, 0, 0, 0, 0, 0, 0);
		}
	}

	d.type = T_DATE;
	return d;
}

// create a new datetime object
SIValue AR_LOCALDATETIME
(
	SIValue *argv,      // arguments
	int argc,           // number of arguments
	void *private_data  // private data
) {
	if(argc == 0) {
		// return datetime representing the current date and time
		return DateTime_now();
	}

	SIValue arg = argv[0];
	
	if(SIValue_IsNull(arg)) {
		return SI_NullVal();
	}

	SIType t = SI_TYPE(arg);
	ASSERT(t == T_STRING || t == T_MAP);

	if(t == T_STRING) {
		SIValue dt = DateTime_fromString(arg.stringval);
		if(SIValue_IsNull(dt)) {
			ErrorCtx_SetError("Failed to parse datetime");
		}
		return dt;
	} else {
		// create datetime object from map
		// datetime components
		int year;              // mandatory
		int month        = 1;  // month defaults to 1
		int quarter      = 1;  // quarter defaults to 1
		int dayOfQuarter = 1;  // day of quarter defaults to 1
		int week         = 1;  // week defaults to 1
		int day          = 1;  // day defaults to 1
		int dayOfWeek    = 1;  // day of week defaults to 1
		int hour         = 0;  // hour defaults to 0
		int minute       = 0;  // minute defaults to 0
		int second       = 0;  // second defaults to 0
		int millisecond  = 0;  // mili-second defaults to 0
		int nanosecond   = 0;  // nano-second defaults to 0
		int microsecond  = 0;  // micro-second defaults to 0

		uint n_keys = Map_KeyCount(arg);

		//----------------------------------------------------------------------
		// extract individual datetime elements
		//----------------------------------------------------------------------

		// extract year
		bool year_specified = _get_component(&arg, "year", &year, -999999999, 999999999);

		if(!year_specified) {
			// year is mandatory
			ErrorCtx_SetError("year must be specified");
			return SI_NullVal();
		}

		bool quarter_specified      = _get_component(&arg, "quarter",      &quarter,      1, 4);
		bool dayOfQuarter_specified = _get_component(&arg, "dayOfQuarter", &dayOfQuarter, 1, 92);
		bool month_specified        = _get_component(&arg, "month",        &month,        1, 12);
		bool week_specified         = _get_component(&arg, "week",         &week,         1, 53);
		bool day_specified          = _get_component(&arg, "day",          &day,          1, 31);
		bool dayOfWeek_specified    = _get_component(&arg, "dayOfWeek",    &dayOfWeek,    1, 7);
		bool hour_specified         = _get_component(&arg, "hour",         &hour,         0, 23);
		bool minute_specified       = _get_component(&arg, "minute",       &minute,       0, 59);
		bool second_specified       = _get_component(&arg, "second",       &second,       0, 59);
		bool milisecond_specified   = _get_component(&arg, "millisecond",  &millisecond,  0, 999);
		bool microsecond_specified  = _get_component(&arg, "microsecond",  &microsecond,  0, 999999);
		bool nanosecond_specified   = _get_component(&arg, "nanosecond",   &nanosecond,   0, 999999999);

		n_keys -= year_specified + quarter_specified + dayOfQuarter_specified +
				  month_specified + week_specified + dayOfWeek_specified +
				  day_specified + hour_specified + minute_specified +
				  second_specified + milisecond_specified +
				  microsecond_specified + nanosecond_specified;

		// can't specify both week and month
		// as month is determine from the week number
		if(week_specified == true && month_specified == true) {
			ErrorCtx_SetError("month cannot be specified with week");	
			return SI_NullVal();
		}

		// can't specify dayOfQuarter_specified without specifying quarter
		if(dayOfQuarter_specified == true && quarter_specified == false) {
			ErrorCtx_SetError("dayOfQuarter_specified cannot be specified without quarter");	
			return SI_NullVal();
		}

		// can't specify quarter with day or week
		if((dayOfQuarter_specified == true || quarter_specified == true) && (
					day_specified       == true ||
					month_specified     == true ||
					week_specified      == true ||
					dayOfWeek_specified == true)) {
			ErrorCtx_SetError("dayOfWeek cannot be specified without week");	
			return SI_NullVal();
		}

		// can't specify day of week without specifying week number
		if(dayOfWeek_specified == true && week_specified == false) {
			ErrorCtx_SetError("dayOfWeek cannot be specified without week");	
			return SI_NullVal();
		}

		// if day is specified month must be specified as well
		if(day_specified == true && month_specified == false) {
			ErrorCtx_SetError("day cannot be specified without month");	
			return SI_NullVal();
		}

		// error incase components map contains an unknown key
		if(n_keys > 0) {
			ErrorCtx_SetError("datetime components map contains an unknown key");	
			return SI_NullVal();
		}

		if(week_specified) {
			return DateTime_fromWeekDate(year, week, dayOfWeek, hour, minute,
					second, millisecond, microsecond, nanosecond);
		}

		if(quarter_specified) {
			return DateTime_fromQuarterDate(year, quarter, dayOfQuarter, hour,
					minute, second, millisecond, microsecond, nanosecond);
		
		}

		return DateTime_fromComponents(year, month, day, hour, minute, second,
				millisecond, microsecond, nanosecond);
	}
}

// create a new duration object
SIValue AR_DURATION
(
	SIValue *argv,      // arguments
	int argc,           // number of arguments
	void *private_data  // private data
) {
	SIValue arg = argv[0];
	
	if(SIValue_IsNull(arg)) {
		return SI_NullVal();
	}

	SIType t = SI_TYPE(arg);
	ASSERT(t & (T_STRING | T_MAP));

	// parse string as duration
	if(t == T_STRING) {
		return SI_DurationFromString(arg.stringval);
	}

	// create duration object from map
	// duration components
	float years   = 0;
	float months  = 0;
	float weeks   = 0;
	float days    = 0;
	float hours   = 0;
	float minutes = 0;
	float seconds = 0;

	uint n_keys = Map_KeyCount(arg);

	//----------------------------------------------------------------------
	// extract individual duration elements
	//----------------------------------------------------------------------

	bool years_specified   = _duration_get_component(&arg, "years",   &years);
	bool months_specified  = _duration_get_component(&arg, "months",  &months);
	bool weeks_specified   = _duration_get_component(&arg, "weeks",   &weeks);
	bool days_specified    = _duration_get_component(&arg, "days",    &days);
	bool hours_specified   = _duration_get_component(&arg, "hours",   &hours);
	bool minutes_specified = _duration_get_component(&arg, "minutes", &minutes);
	bool seconds_specified = _duration_get_component(&arg, "seconds", &seconds);

	n_keys -= years_specified + months_specified + weeks_specified +
			  days_specified + hours_specified + minutes_specified +
			  seconds_specified;

	// error incase components map contains an unknown key
	if(n_keys > 0) {
		ErrorCtx_SetError("datetime components map contains an unknown key");	
		return SI_NullVal();
	}

	return SI_DurationFromComponents(years, months, weeks, days, hours, minutes,
			seconds);
}

void Register_TimeFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 0);
	ret_type = T_INT64;
	func_desc = AR_FuncDescNew("timestamp", AR_TIMESTAMP, 0, 0, types, ret_type,
			false, false, false);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_STRING | T_MAP | T_NULL);
	ret_type = T_TIME | T_NULL;
	func_desc = AR_FuncDescNew("localtime", AR_LOCALTIME, 0, 1, types, ret_type,
			false, false, false);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_STRING | T_MAP | T_NULL);
	ret_type = T_TIME | T_NULL;
	func_desc = AR_FuncDescNew("localtime.transaction", AR_LOCALTIME, 0, 1,
			types, ret_type,
			false, true, false);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_STRING | T_MAP | T_NULL);
	ret_type = T_DATE | T_NULL;
	func_desc = AR_FuncDescNew("date", AR_DATE, 0, 1, types, ret_type, false,
			false, false);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_STRING | T_MAP | T_NULL);
	ret_type = T_DATE | T_NULL;
	func_desc = AR_FuncDescNew("date.transaction", AR_DATE, 0, 1, types,
			ret_type, false, true, false);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_STRING | T_MAP | T_NULL);
	ret_type = T_DATETIME | T_NULL;
	func_desc = AR_FuncDescNew("localdatetime", AR_LOCALDATETIME, 0, 1, types,
			ret_type, false, false, false);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_STRING | T_MAP | T_NULL);
	ret_type = T_DATETIME | T_NULL;
	func_desc = AR_FuncDescNew("localdatetime.transaction", AR_LOCALDATETIME, 0,
			1, types, ret_type, false, true, false);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_MAP | T_STRING | T_NULL);
	ret_type = T_DURATION | T_NULL;
	func_desc = AR_FuncDescNew("duration", AR_DURATION, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);
}

