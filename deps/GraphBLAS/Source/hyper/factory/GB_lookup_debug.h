//------------------------------------------------------------------------------
// GB_lookup_debug: find k where j == Ah [k], without using the A->Y hyper_hash
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_LOOKUP_DEBUG_H
#define GB_LOOKUP_DEBUG_H

// For a sparse, bitmap, or full matrix j == k.
// For a hypersparse matrix, find k so that j == Ah [k], if it
// appears in the list.

// k is not needed by the caller, just pstart, pend, pleft, and found.

// Once k is found, find pstart and pend, the start and end of the vector.
// pstart and pend are defined for all sparsity structures: hypersparse,
// sparse, bitmap, or full.

// With the introduction of the hyper_hash, this is used only for debugging.

#ifdef GB_DEBUG

#define GB_PTYPE uint32_t
#define GB_JTYPE uint32_t
#define GB_lookup_debug_T GB_lookup_debug_32_32
#define GB_binary_search_T GB_binary_search_32
#include "hyper/factory/GB_lookup_debug_template.h"

#define GB_PTYPE uint32_t
#define GB_JTYPE uint64_t
#define GB_lookup_debug_T GB_lookup_debug_32_64
#define GB_binary_search_T GB_binary_search_64
#include "hyper/factory/GB_lookup_debug_template.h"

#define GB_PTYPE uint64_t
#define GB_JTYPE uint32_t
#define GB_lookup_debug_T GB_lookup_debug_64_32
#define GB_binary_search_T GB_binary_search_32
#include "hyper/factory/GB_lookup_debug_template.h"

#define GB_PTYPE uint64_t
#define GB_JTYPE uint64_t
#define GB_lookup_debug_T GB_lookup_debug_64_64
#define GB_binary_search_T GB_binary_search_64
#include "hyper/factory/GB_lookup_debug_template.h"

static inline bool GB_lookup_debug  // find j = Ah [k]
(
    // input:
    const bool Ap_is_32,            // if true, Ap is 32-bit; else 64-bit
    const bool Aj_is_32,            // if true, Ah, Y->[pix] are 32-bit; else 64
    const bool A_is_hyper,          // true if A is hypersparse
    const void *Ah,                 // A->h [0..A->nvec-1]: list of vectors
    const void *Ap,                 // A->p [0..A->nvec  ]: pointers to vectors
    const int64_t avlen,            // A->vlen
    // input/output:
    int64_t *restrict pleft,        // on input: look in A->h [pleft..pright].
                                    // on output: pleft == k if found.
    // input:
    int64_t pright,                 // normally A->nvec-1, but can be trimmed
    const int64_t j,                // vector to find, as j = Ah [k]
    // output:
    int64_t *restrict pstart,       // start of vector: Ap [k]
    int64_t *restrict pend          // end of vector: Ap [k+1]
)
{
    if (Ap_is_32)
    {
        if (Aj_is_32)
        {
            // Ap is 32-bit; Ah is 32 bit
            return (GB_lookup_debug_32_32 (A_is_hyper, Ah, Ap, avlen,
                pleft, pright, j, pstart, pend)) ;
        }
        else
        {
            // Ap is 32-bit; Ah is 64-bit
            return (GB_lookup_debug_32_64 (A_is_hyper, Ah, Ap, avlen,
                pleft, pright, j, pstart, pend)) ;
        }
    }
    else
    {
        if (Aj_is_32)
        {
            // Ap is 64-bit; Ah is 32-bit
            return (GB_lookup_debug_64_32 (A_is_hyper, Ah, Ap, avlen,
                pleft, pright, j, pstart, pend)) ;
        }
        else
        {
            // Ap is 64-bit; Ah is 64-bit
            return (GB_lookup_debug_64_64 (A_is_hyper, Ah, Ap, avlen,
                pleft, pright, j, pstart, pend)) ;
        }
    }
}

#endif
#endif

