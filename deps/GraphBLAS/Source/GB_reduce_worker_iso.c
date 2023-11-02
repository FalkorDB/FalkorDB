//------------------------------------------------------------------------------
// GB_reduce_worker_iso: reduce n entries, all equal to a, to the scalar s
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Some built-in monoids could be done in O(1) time, but this takes at most
// O(log (n)) time which is fast enough, even if n = 2^60, and it works for all
// monoids including user-defined ones.

#include "GB_reduce.h"

void GB_reduce_worker_iso
(
    GB_void *restrict s,            // output scalar
    GxB_binary_function freduce,    // reduction function
    GB_void *restrict a,            // iso value of A
    uint64_t n,                     // number of entries in A to reduce
    size_t zsize                    // size of s and a
)
{

    if (n <= 1)
    { 
        memcpy (s, a, zsize) ;
    }
    else
    { 

        // reduce floor (n/2) entries to the scalar s
        GB_reduce_worker_iso (s, freduce, a, n/2, zsize) ;

        // s = freduce (s, s)
        freduce (s, s, s) ;

        // if n is even, s is now the reduction of 2*floor(n/2) == n entries.
        // if n is odd, s is now the reduction of 2*floor(n/2) == n-1 entries.

        if (n & 1)
        { 
            // n is odd, so add more more entry with s = freduce (s, a)
            freduce (s, s, a) ;
        }
    }
}

