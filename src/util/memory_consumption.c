/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#if defined(__APPLE__) && defined(__MACH__)
    #include <mach/mach.h>
#elif defined(__linux__)
    #include <sys/resource.h>
#endif

// returns the current Resident Set Size (RSS) in bytes
size_t get_current_rss(void) {
#if defined(__APPLE__) && defined(__MACH__)
    // macOS/Darwin logic
	struct mach_task_basic_info info ;
	mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT ;

	if (task_info (mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
				&count) == KERN_SUCCESS) {
		return (size_t)info.resident_size ;
    }

#elif defined(__linux__)
    // Linux logic: /proc/self/statm is the most reliable for current RSS
    FILE *fp = fopen ("/proc/self/statm", "r") ;
    if (fp) {
        long pages = 0 ;
        if (fscanf (fp, "%*s %ld", &pages) == 1) {
            long page_size = sysconf (_SC_PAGESIZE) ;
            fclose (fp) ;
            return (size_t)(pages * page_size) ;
        }
        fclose (fp) ;
    }
#endif

    return 0 ; // unknown or failure
}

