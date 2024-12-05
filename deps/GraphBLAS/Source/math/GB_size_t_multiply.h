//------------------------------------------------------------------------------
// GB_size_t_multiply:  multiply two size_t and guard against overflow
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// c = a*b but check for overflow

#ifndef GB_SIZE_T_MULTIPLY_H
#define GB_SIZE_T_MULTIPLY_H

GB_STATIC_INLINE bool GB_size_t_multiply     // true if ok, false if overflow
(
    size_t *c,              // c = a*b, or zero if overflow occurs
    const size_t a,
    const size_t b
)
{
    uint64_t x ;
    bool ok = GB_uint64_multiply (&x, (uint64_t) a, (uint64_t) b) ;
    (*c) = (ok && x <= SIZE_MAX) ? ((size_t) x) : 0 ;
    return (ok) ;
}

#endif

