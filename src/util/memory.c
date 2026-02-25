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
    #include <sys/sysinfo.h>
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

// returns the total amount of available physical memory on the host in bytes
size_t get_host_available_memory(void) {
#if defined(__APPLE__) && defined(__MACH__)
    // macOS logic using Mach host statistics
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT ;
    vm_statistics64_data_t vm_stats ;
    host_t host = mach_host_self() ;

    if (host_statistics64 (host, HOST_VM_INFO64,
				(host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        long page_size = sysconf (_SC_PAGESIZE) ;
        // available memory is roughly free + inactive pages
        // inactive pages are kept by the OS but can be reclaimed immediately
        return (size_t)((vm_stats.free_count + vm_stats.inactive_count)
				* page_size) ;
    }

#elif defined(__linux__)
    // linux logic: /proc/meminfo is the source of truth for "Available" memory
    // note: MemAvailable (since Linux 3.14) is better than MemFree as it
    // accounts for reclaimable caches
    FILE *fp = fopen ("/proc/meminfo", "r") ;
    if (fp) {
        char buf[256] ;
        size_t available = 0 ;
        while (fgets (buf, sizeof(buf), fp)) {
            if (sscanf(buf, "MemAvailable: %zu kB", &available) == 1) {
                fclose (fp) ;
                return available * 1024 ; // Convert kB to bytes
            }
        }

        // fallback for very old kernels (< 3.14) using sysinfo
        rewind (fp) ;
        struct sysinfo si ;
        if (sysinfo (&si) == 0) {
            fclose (fp) ;
            return (size_t)si.freeram * si.mem_unit ;
        }
        fclose (fp) ;
    }
#endif
    return 0 ;
}

