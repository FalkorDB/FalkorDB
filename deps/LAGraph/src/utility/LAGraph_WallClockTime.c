//------------------------------------------------------------------------------
// LAGraph_WallClockTime: return the current wall clock time
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// Unlike all other LAGraph functions, this function does not return an error
// code as an int, nor does it have a char *msg parameter for error messages.
// It simply returns the current wall clock time, as a double value, indicating
// the amount of time passed in seconds since some fixed point in the past.

// Example usage:

/*
    double t1 = LAGraph_WallClockTime ( ) ;

    // ... do stuff
    double t2 = LAGraph_WallClockTime ( ) ;
    printf ("time to 'do stuff' : %g (seconds)\n', t2 - t1) ;
    // ... more stuff
    double t3 = LAGraph_WallClockTime ( ) ;
    printf ("time to 'do stuff' and 'more stuff': %g (seconds)\n', t3 - t1) ;
*/

#include "LG_internal.h"

#if !defined ( _OPENMP )
    #include <time.h>
    #if defined ( __linux__ ) || defined ( __GNU__ )
        #include <sys/time.h>
    #endif
    #if defined ( __MACH__ ) && defined ( __APPLE__ )
        #include <mach/clock.h>
        #include <mach/mach.h>
    #endif
#endif

double LAGraph_WallClockTime (void)
{
    double t_wallclock = 0 ;

    #if defined ( _OPENMP )

        // OpenMP is available; use the OpenMP timer function
        t_wallclock = omp_get_wtime ( ) ;

    #elif defined ( __linux__ )

        // Linux has a very low resolution clock() function, so use the high
        // resolution clock_gettime instead.  May require -lrt
        struct timespec t ;
        int e = clock_gettime (CLOCK_MONOTONIC, &t) ;
        if (e == 0)
        {
            t_wallclock = (double) t.tv_sec + 1e-9 * ((double) t.tv_nsec) ;
        }

    #elif defined ( __MACH__ )

        // Mac OSX
        clock_serv_t cclock ;
        mach_timespec_t t ;
        host_get_clock_service (mach_host_self ( ), SYSTEM_CLOCK, &cclock) ;
        clock_get_time (cclock, &t) ;
        mach_port_deallocate (mach_task_self ( ), cclock) ;
        t_wallclock = (double) t.tv_sec + 1e-9 * ((double) t.tv_nsec) ;

    #else

        // The ANSI C11 clock() function is used instead.  This gives the
        // processor time, not the wallclock time, and it might have low
        // resolution.  It returns the time since some unspecified fixed time
        // in the past, as a clock_t integer.  The clock ticks per second are
        // given by CLOCKS_PER_SEC.  In Mac OSX this is a very high resolution
        // clock, and clock ( ) is faster than clock_get_time (...) ;
        clock_t t = clock ( ) ;
        t_wallclock = ((double) t) / ((double) CLOCKS_PER_SEC) ;

    #endif

    return (t_wallclock) ;
}
