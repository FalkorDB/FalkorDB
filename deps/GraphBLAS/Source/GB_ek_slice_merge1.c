//------------------------------------------------------------------------------
// GB_ek_slice_merge1: merge column counts for a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input matrix A has been sliced via GB_ek_slice, and scanned to compute
// the counts of entries in each vector of C in Cp, Wfirst, and Wlast.  This
// phase finalizes the column counts, Cp, merging the results of each task.

// On input, Cp [k] has been partially computed.  Task tid operators on vector
// kfirst = kfirst_Aslice [tid] to klast = klast_Aslice [tid].  If kfirst < k <
// klast, then Cp [k] is the total count of entries in C(:,k).  Otherwise, the
// counts are held in Wfirst and Wlast, and Cp [k] is zero (or uninititalized).
// Wfirst [tid] is the number of entries in C(:,kfirst) constructed by task
// tid, and Wlast [tid] is the number of entries in C(:,klast) constructed by
// task tid.

// This function sums up the entries computed for C(:,k) by all tasks, so that
// on output, Cp [k] is the total count of entries in C(:,k).

#include "GB_ek_slice.h"

GB_CALLBACK_EK_SLICE_MERGE1_PROTO (GB_ek_slice_merge1)
{
    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;

    int64_t kprior = -1 ;

    for (int tid = 0 ; tid < A_ntasks ; tid++)
    {

        //----------------------------------------------------------------------
        // sum up the partial result that thread tid computed for kfirst
        //----------------------------------------------------------------------

        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;

        if (kfirst <= klast)
        {
            if (kprior < kfirst)
            { 
                // This thread is the first one that did work on
                // A(:,kfirst), so use it to start the reduction.
                Cp [kfirst] = Wfirst [tid] ;
            }
            else
            { 
                Cp [kfirst] += Wfirst [tid] ;
            }
            kprior = kfirst ;
        }

        //----------------------------------------------------------------------
        // sum up the partial result that thread tid computed for klast
        //----------------------------------------------------------------------

        if (kfirst < klast)
        { 
            ASSERT (kprior < klast) ;
            // This thread is the first one that did work on
            // A(:,klast), so use it to start the reduction.
            Cp [klast] = Wlast [tid] ;
            kprior = klast ;
        }
    }
}

