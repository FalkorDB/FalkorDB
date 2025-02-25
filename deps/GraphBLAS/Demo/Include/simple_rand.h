//------------------------------------------------------------------------------
// GraphBLAS/Demo/Include/simple_rand.h: a very simple random number generator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef SIMPLE_RAND_H
#define SIMPLE_RAND_H

#include <stdint.h>

// return a random number between 0 and UINT64_MAX
static inline uint64_t simple_rand (uint64_t *state)
{
    uint64_t s = (*state) ;
    s ^= s << 13 ;
    s ^= s >> 7 ;
    s ^= s << 17 ;
    return ((*state) = s) ;
}

// return a random double between 0 and 1, inclusive
static inline double simple_rand_x (uint64_t *state)
{
    return (((double) simple_rand (state)) / ((double) UINT64_MAX)) ;
}

#endif
