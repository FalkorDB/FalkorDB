//------------------------------------------------------------------------------
// GB_rand.h: random number generator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_RAND_H
#define GB_RAND_H

#define GB_BASECASE (64 * 1024)

//------------------------------------------------------------------------------
// random number generator for quicksort
//------------------------------------------------------------------------------

// https://en.wikipedia.org/wiki/Xorshift

GB_STATIC_INLINE uint64_t GB_rand (uint64_t *state)
{
    uint64_t x = (*state) ;
    x ^= x << 7 ;
    x ^= x >> 9 ;
    (*state) = x ;
    return (x) ;
}

#endif

