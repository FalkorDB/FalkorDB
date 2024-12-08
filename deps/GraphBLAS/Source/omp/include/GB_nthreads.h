//------------------------------------------------------------------------------
// GB_nthreads.h: determine # of threads to use for parallel region
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_NTHREADS_H
#define GB_NTHREADS_H

//------------------------------------------------------------------------------
// GB_nthreads: determine # of threads to use for a parallel loop or region
//------------------------------------------------------------------------------

// If work < 2*chunk, then only one thread is used.
// else if work < 3*chunk, then two threads are used, and so on.

static inline int GB_nthreads   // return # of threads to use
(
    double work,                // total work to do
    double chunk,               // give each thread at least this much work
    int nthreads_max            // max # of threads to use
)
{ 
    work  = GB_IMAX (work, 1) ;
    chunk = GB_IMAX (chunk, 1) ;
    int64_t nthreads = (int64_t) floor (work / chunk) ;
    nthreads = GB_IMIN (nthreads, nthreads_max) ;
    nthreads = GB_IMAX (nthreads, 1) ;
    return ((int) nthreads) ;
}

#endif

