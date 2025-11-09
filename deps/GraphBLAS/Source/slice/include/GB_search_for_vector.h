//------------------------------------------------------------------------------
// GB_search_for_vector: find the vector k that contains p
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Given an index p, find k so that Ap [k] <= p && p < Ap [k+1].  The search is
// limited to k in the range Ap [kleft ... anvec].

// A->p can come from any matrix: hypersparse, sparse, bitmap, or full.
// For the latter two cases, A->p is NULL.

#ifndef GB_SEARCH_FOR_VECTOR_H
#define GB_SEARCH_FOR_VECTOR_H

#define GB_SV_TYPE uint32_t
#define GB_search_for_vector_TYPE GB_search_for_vector_32
#define GB_split_binary_search_TYPE GB_split_binary_search_32
#include "include/GB_search_for_vector_template.h"

#define GB_SV_TYPE uint64_t
#define GB_search_for_vector_TYPE GB_search_for_vector_64
#define GB_split_binary_search_TYPE GB_split_binary_search_64
#include "include/GB_search_for_vector_template.h"

static inline int64_t GB_search_for_vector // return vector k
(
    const void *Ap,                 // vector pointers to search
    const bool Ap_is_32,            // if true, Ap is 32-bit, else 64-bit
    const int64_t p,                // search for vector k that contains p
    const int64_t kleft,            // left-most k to search
    const int64_t anvec,            // Ap is of size anvec+1
    const int64_t avlen             // A->vlen
)
{
    if (Ap_is_32)
    {
        return (GB_search_for_vector_32 (Ap, p, kleft, anvec, avlen)) ;
    }
    else
    {
        return (GB_search_for_vector_64 (Ap, p, kleft, anvec, avlen)) ;
    }
}

#endif

