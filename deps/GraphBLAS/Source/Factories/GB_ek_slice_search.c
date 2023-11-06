//------------------------------------------------------------------------------
// GB_ek_slice_search.c: find the first and last vectors in a slice
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_EK_SLICE_SEARCH_H
#define GB_EK_SLICE_SEARCH_H
#include "GB_search_for_vector_template.c"

static inline void GB_ek_slice_search
(
    // input:
    int taskid,
    int ntasks,
    const int64_t *restrict pstart_slice,    // size ntasks+1
    const int64_t *restrict Ap,              // size anvec
    int64_t anvec,                              // # of vectors in A
    int64_t avlen,                              // vector length of A
    // output:
    int64_t *restrict kfirst_slice,          // size ntasks
    int64_t *restrict klast_slice            // size ntasks
)
{
    int64_t pfirst = pstart_slice [taskid] ;
    int64_t plast  = pstart_slice [taskid+1] - 1 ;

    // find the first vector of the slice for task taskid: the
    // vector that owns the entry Ai [pfirst] and Ax [pfirst].
    int64_t kfirst ;
    if (taskid == 0)
    { 
        kfirst = 0 ;
    }
    else
    { 
        kfirst = GB_search_for_vector (pfirst, Ap, 0, anvec, avlen) ;
    }

    // find the last vector of the slice for task taskid: the
    // vector that owns the entry Ai [plast] and Ax [plast].
    int64_t klast ;
    if (taskid == ntasks-1)
    { 
        klast = anvec - 1 ;
    }
    else if (pfirst > plast)
    { 
        // this task does no work
        klast = kfirst ;
    }
    else
    { 
        klast = GB_search_for_vector (plast, Ap, kfirst, anvec, avlen) ;
    }
    kfirst_slice [taskid] = kfirst ;
    klast_slice  [taskid] = klast ;
}

#endif

