//------------------------------------------------------------------------------
// GB_hyper_hash_lookup: find k so that j == Ah [k], using the A->Y hyper_hash
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_HYPER_HASH_LOOKUP_H
#define GB_HYPER_HASH_LOOKUP_H

#define GB_PTYPE uint32_t
#define GB_JTYPE uint32_t
#define GB_hyper_hash_lookup_T GB_hyper_hash_lookup_32_32
#define GB_binary_search_T     GB_binary_search_32
#include "include/GB_hyper_hash_lookup_template.h"

#define GB_PTYPE uint32_t
#define GB_JTYPE uint64_t
#define GB_hyper_hash_lookup_T GB_hyper_hash_lookup_32_64
#define GB_binary_search_T     GB_binary_search_64
#include "include/GB_hyper_hash_lookup_template.h"

#define GB_PTYPE uint64_t
#define GB_JTYPE uint32_t
#define GB_hyper_hash_lookup_T GB_hyper_hash_lookup_64_32
#define GB_binary_search_T     GB_binary_search_32
#include "include/GB_hyper_hash_lookup_template.h"

#define GB_PTYPE uint64_t
#define GB_JTYPE uint64_t
#define GB_hyper_hash_lookup_T GB_hyper_hash_lookup_64_64
#define GB_binary_search_T     GB_binary_search_64
#include "include/GB_hyper_hash_lookup_template.h"

GB_STATIC_INLINE int64_t GB_hyper_hash_lookup // k if j==Ah[k]; -1 if not found
(
    // inputs, not modified:
    const bool Ap_is_32,            // if true, Ap is 32-bit; else 64-bit
    const bool Aj_is_32,            // if true, Ah, Y->[pix] are 32-bit; else 64
    const void *Ah,                 // A->h [0..A->nvec-1]: list of vectors
    const int64_t anvec,
    const void *Ap,                 // A->p [0..A->nvec]: pointers to vectors
    const void *A_Yp,               // A->Y->p
    const void *A_Yi,               // A->Y->i
    const void *A_Yx,               // A->Y->x
    const uint64_t hash_bits,       // A->Y->vdim-1, which is hash table size-1
    const int64_t j,                // find j in Ah [0..anvec-1], using A->Y
    // outputs:
    int64_t *restrict pstart,       // start of vector: Ap [k]
    int64_t *restrict pend          // end of vector: Ap [k+1]
)
{
    if (Ap_is_32)
    {
        if (Aj_is_32)
        { 
            // Ap is 32-bit; Ah, A_Y[pix] are 32-bit
            return (GB_hyper_hash_lookup_32_32 ((uint32_t *) Ah, anvec,
                (uint32_t *) Ap, (uint32_t *) A_Yp, (uint32_t *) A_Yi,
                (uint32_t *) A_Yx, hash_bits, j, pstart, pend)) ;
        }
        else
        { 
            // Ap is 32-bit; Ah, A_Y[pix] are 64-bit
            return (GB_hyper_hash_lookup_32_64 ((uint64_t *) Ah, anvec,
                (uint32_t *) Ap, (uint64_t *) A_Yp, (uint64_t *) A_Yi,
                (uint64_t *) A_Yx, hash_bits, j, pstart, pend)) ;
        }
    }
    else
    {
        if (Aj_is_32)
        { 
            // Ap is 64-bit; Ah, A_Y[pix] are 32-bit
            return (GB_hyper_hash_lookup_64_32 ((uint32_t *) Ah, anvec,
                (uint64_t *) Ap, (uint32_t *) A_Yp, (uint32_t *) A_Yi,
                (uint32_t *) A_Yx, hash_bits, j, pstart, pend)) ;
        }
        else
        { 
            // Ap is 64-bit; Ah, A_Y[pix] are 64-bit
            return (GB_hyper_hash_lookup_64_64 ((uint64_t *) Ah, anvec,
                (uint64_t *) Ap, (uint64_t *) A_Yp, (uint64_t *) A_Yi,
                (uint64_t *) A_Yx, hash_bits, j, pstart, pend)) ;
        }
    }
}

#endif

