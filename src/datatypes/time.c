/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../value.h"
#include "datetime.h"
#include "../util/rmalloc.h"

#define _XOPEN_SOURCE
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

// create a new time object representing the current time
SIValue Time_now(void) {
	return (SIValue) {
		.datetimeval = time(NULL), .type = T_TIME, .allocation = M_NONE
	};
}

// create a new time object from a ISO-8601 string time representation
SIValue Time_fromString
(
	char *time_str  // time ISO-8601 string representation
) {
    if(!time_str) {
        return SI_NullVal();
    }

    int hour        = 0;
	int minute      = 0;
	int second      = 0;
    int millisecond = 0;
	int microsecond = 0;
	int nanosecond  = 0;
    int parsed      = 0;

    // check if string contains colons (formatted time)
    if(strchr(time_str, ':')) {
        // parse formatted time: HH:MM:SS, HH:MM, etc
        char *colon1 = strchr(time_str, ':');
        char *colon2 = colon1 ? strchr(colon1 + 1, ':') : NULL;
        
        if(colon2) {
            // HH:MM:SS format
            parsed = sscanf(time_str, "%d:%d:%d", &hour, &minute, &second);
        } else {
            // HH:MM format  
            parsed = sscanf(time_str, "%d:%d", &hour, &minute);
            if(parsed == 2) {
                second = 0;
                parsed = 3; // treat as fully parsed
            }
        }
    } else {
        // parse compact format: HHMMSS, HHMM, HH
        int len = strlen(time_str);
        char *dot_pos = strchr(time_str, '.');
        int int_part_len = dot_pos ? (dot_pos - time_str) : len;
        
        if(int_part_len == 1 || int_part_len == 2) {
            // H or HH format
            parsed = sscanf(time_str, "%d", &hour);
            minute = second = 0;
            if(parsed == 1) parsed = 3; // treat as fully parsed
        } else if(int_part_len == 3 || int_part_len == 4) {
            // HMM or HHMM format
            int time_int;
            parsed = sscanf(time_str, "%d", &time_int);
            if(parsed == 1) {
                if(int_part_len == 3) {
                    // HMM
                    hour = time_int / 100;
                    minute = time_int % 100;
                } else {
                    // HHMM
                    hour = time_int / 100;
                    minute = time_int % 100;
                }
                second = 0;
                parsed = 3; // treat as fully parsed
            }
        } else if(int_part_len == 5 || int_part_len == 6) {
            // HMMSS or HHMMSS format
            int time_int;
            parsed = sscanf(time_str, "%d", &time_int);
            if(parsed == 1) {
                if(int_part_len == 5) {
                    // HMMSS
                    hour = time_int / 10000;
                    minute = (time_int / 100) % 100;
                    second = time_int % 100;
                } else {
                    // HHMMSS
                    hour = time_int / 10000;
                    minute = (time_int / 100) % 100;
                    second = time_int % 100;
                }
                parsed = 3; // treat as fully parsed
            }
        }
    }
    
    if(parsed < 1) {
        // invalid format
        return SI_NullVal();
    }
    
    // look for fractional seconds
    char *dot_pos = strchr(time_str, '.');
    if(dot_pos) {
        dot_pos++; // move past the dot
        
        // count digits and parse fractional part
        int frac_len = 0;
        char *end_pos = dot_pos;
        while(*end_pos >= '0' && *end_pos <= '9') {
            frac_len++;
            end_pos++;
        }
        
        if(frac_len > 0) {
            // parse the fractional part as an integer
            long long frac_value = 0;
            sscanf(dot_pos, "%lld", &frac_value);
            
            // convert to nanoseconds based on number of digits
            if(frac_len <= 3) {
                // milliseconds (pad to 3 digits)
                for(int i = frac_len; i < 3; i++) frac_value *= 10;
                millisecond = frac_value;
            } else if(frac_len <= 6) {
                // microseconds (pad to 6 digits)
                for(int i = frac_len; i < 6; i++) frac_value *= 10;
                millisecond = frac_value / 1000;
                microsecond = frac_value % 1000;
            } else {
                // nanoseconds (pad to 9 digits)
                for(int i = frac_len; i < 9; i++) frac_value *= 10;
                // truncate if more than 9 digits
                if(frac_len > 9) {
                    for(int i = 9; i < frac_len; i++) frac_value /= 10;
                }
                millisecond = frac_value / 1000000;
                microsecond = (frac_value / 1000) % 1000;
                nanosecond  = frac_value % 1000;
            }
        }
    }
    
    // Validate parsed values
    if(hour        < 0 || hour        > 23     ||
       minute      < 0 || minute      > 59     ||
       second      < 0 || second      > 59     ||
       millisecond < 0 || millisecond > 999    ||
       microsecond < 0 || microsecond > 999999 ||
       nanosecond  < 0 || nanosecond  > 999999999) {
        return SI_NullVal();
    }
    
	SIValue time = DateTime_fromComponents(1900, 1, 1, hour, minute, second,
			millisecond, microsecond, nanosecond);

	time.type = T_TIME;

	return time;
}

// extract component from time objects
// available components:
// second, minute, hour
bool Time_getComponent
(
	const SIValue *time,    // time object
	const char *component,  // time component to get
	int *value              // [output] component value
) {
	ASSERT(value              != NULL);
	ASSERT(time               != NULL);
	ASSERT(component          != NULL);
	ASSERT(SI_TYPE(*time) == T_TIME);

	// set output
	*value = -1;

	//--------------------------------------------------------------------------
	// convert from time_t to tm
	//--------------------------------------------------------------------------

	struct tm t;
	time_t rawtime = time->datetimeval;
	gmtime_r(&rawtime, &t);

	//--------------------------------------------------------------------------
	// extract component
	//--------------------------------------------------------------------------

	if(strcasecmp(component, "second") == 0) {
		// seconds after the minute — [0, 60]
		*value = t.tm_sec;
	} else if(strcasecmp(component, "minute") == 0) {
		// minutes after the hour — [0, 59]
		*value = t.tm_min;
	} else if(strcasecmp(component, "hour") == 0) {
		// hours since midnight — [0, 23]
		*value = t.tm_hour;
	}

	return (*value != -1);

}

// get a string representation of time
void Time_toString
(
	const SIValue *time,  // time object
	char **buf,           // print buffer
	size_t *bufferLen,    // print buffer length
	size_t *bytesWritten  // the actual number of bytes written to the buffer
) {
	ASSERT(buf            != NULL);
	ASSERT(time           != NULL);
	ASSERT(SI_TYPE(*time) == T_TIME);

	if(*bufferLen - *bytesWritten < 32) {
		*bufferLen += 32;
		*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);
	}

	// get a tm object from time_t
	struct tm t;
	time_t rawtime = time->datetimeval;
	gmtime_r(&rawtime, &t);

	// format the date and time up to seconds: 2025-04-14T06:08:21
	*bytesWritten += strftime(*buf + *bytesWritten, *bufferLen, "%H:%M:%S", &t);
	ASSERT(*bytesWritten > 0);
}

