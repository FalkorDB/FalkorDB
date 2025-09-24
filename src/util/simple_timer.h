//------------------------------------------------------------------------------
// GraphBLAS/Demo/Include/simple_timer.h: a timer for performance measurements
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

/*
   There is no method that works on all operating systems for finding the
   current time with high resolution that is suitable for performance
   measurements.  The simple_timer.c and simple_timer.h functions provide a
   portable alternative.

   simple_tic (tic) ; gets the current time and saves it in tic [0..1].

   t = simple_toc (tic) ; returns the time in seconds since the last call to
   simple_toc, as a single double value.

   Usage:

        #include "simple_timer.h"
        double tic [2], r, s, t ;

        simple_tic (tic) ;          // start the timer
        // do some work A
        t = simple_toc (tic) ;      // t is time for work A, in seconds
        // do some work B
        s = simple_toc (tic) ;      // s is time for work A and B, in seconds

        simple_tic (tic) ;          // restart the timer
        // do some work C
        r = simple_toc (tic) ;      // r is time for work C, in seconds
*/

#ifndef SIMPLE_TIMER_H
#define SIMPLE_TIMER_H

#include <time.h>
#include <stdint.h>

//------------------------------------------------------------------------------
// decide which timer to use
//------------------------------------------------------------------------------

#if defined ( _OPENMP )

    // if OpenMP is available, use omp_get_wtime
    #include <omp.h>

#elif defined ( __linux__ ) || defined ( __GNU__ )

    // otherwise, on Linux/GNU, use clock_gettime. May require -lrt
    #include <sys/time.h>

#elif defined ( __MACH__ ) && defined ( __APPLE__ )

    // otherwise, on the Mac, use the MACH timer
    #include <mach/clock.h>
    #include <mach/mach.h>

#else

    // Finally, the ANSI C11 clock() function is used if no other timer
    // is available.

#endif // #if defined(_OPENMP)

typedef double simple_timer_t[2];

// Determines how many milliseconds are in a second.
#define MILLISECONDS_IN_SECOND 1000

// An alias to return the time in milliseconds passed since the previous call.
#define TIMER_GET_ELAPSED_MILLISECONDS(timer) \
    (simple_toc(timer) * MILLISECONDS_IN_SECOND)

//------------------------------------------------------------------------------

void simple_tic         // returns current time in seconds and nanoseconds
(
    double tic [2]      // tic [0]: seconds, tic [1]: nanoseconds
) ;

double simple_toc           // returns time since last simple_tic
(
    const double tic [2]    // tic from last call to simple_tic
) ;

void simple_timer_copy      // copies a simple time
(
    const double tic [2],    // tic from last call to simple_tic
    double cpy [2]           // pointer to recieve a copy of tic
) ;

uint64_t unix_timestamp    // returns UNIX timestamp
(
	void
) ;

#endif // SIMPLE_TIMER_H
