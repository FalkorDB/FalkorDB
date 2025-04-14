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

	*v = -1;

	// extract component
	SIValue _v;
	bool exists = Map_Get(*map, SI_ConstStringVal(component), &_v);

	// validate component type
	if(exists) {
		if(SI_TYPE(_v) != T_INT64) {
			ErrorCtx_SetError("%s must be an integer value", component);
		} else {
			*v = _v.longval;
		}
	}

	if(*v < min_val || *v > max_val) {
		ErrorCtx_SetError("Invalid value for %s (valid values %d - %d)",
				component, min_val, max_val);
	}

	return exists;
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

	} else {
		// create datetime object from map

		// datetime components
		int year;             // mandatory
		int month       = 1;  // month defaults to 1
		int day         = 1;  // day defaults to 1
		int hour        = 0;  // hour defaults to 0
		int minute      = 0;  // minute defaults to 0
		int second      = 0;  // second defaults to 0
		int millisecond = 0;  // mili-second defaults to 0
		int nanosecond  = 0;  // nano-second defaults to 0
		int microsecond = 0;  // micro-second defaults to 0

		uint n_keys = Map_KeyCount(arg);

		//--------------------------------------------------------------------------
		// extract individual datetime elements
		//--------------------------------------------------------------------------

		// extract year
		bool year_specified = _get_component(&arg, "year", &year, -999999999, 999999999);

		if(!year_specified) {
			// year is mandatory
			ErrorCtx_SetError("year must be specified");
			return SI_NullVal();
		}

		bool month_specified       = _get_component(arg, "month",       &month,       1, 12);
		bool day_specified         = _get_component(arg, "day",         &day,         1, 31);
		bool hour_specified        = _get_component(arg, "hour",        &hour,        0, 23);
		bool minute_specified      = _get_component(arg, "minute",      &minute,      0, 59);
		bool second_specified      = _get_component(arg, "second",      &second,      0, 59);
		bool milisecond_specified  = _get_component(arg, "millisecond", &milisecond,  0, 999);
		bool microsecond_specified = _get_component(arg, "microsecond", &microsecond, 0, 999999);
		bool nanosecond_specified  = _get_component(arg, "nanosecond",  &nanosecond,  0, 999999999);

		n_keys -= month_specified + day_specified + hour_specified +
			minute_specified + second_specified + milisecond_specified +
			microsecond_specified + nanosecond_specified;

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

		return DateTime_fromComponents(year, month, day, hour, minute, second,
				millisecond, microsecond, nanosecond);
	}
}

void Register_TimeFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 0);
	ret_type = T_INT64;
	func_desc = AR_FuncDescNew("timestamp", AR_TIMESTAMP, 0, 0, types, ret_type,
			false, false);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_STRING | T_MAP | T_NULL);
	ret_type = T_DATETIME | T_NULL;
	func_desc = AR_FuncDescNew("localdatetime", AR_LOCALDATETIME, 0, 1, types,
			ret_type, false, false);
	AR_RegFunc(func_desc);
}

