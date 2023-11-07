//------------------------------------------------------------------------------
// GB_int64_multiply:  multiply two integers and guard against overflow
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// c = a*b where c is GrB_Index (uint64_t), and a and b are int64_t.
// Check for overflow.  Requires a >= 0 and b >= 0.

#ifndef GB_INT64_MULTIPLY_H
#define GB_INT64_MULTIPLY_H

GB_STATIC_INLINE bool GB_int64_multiply      // true if ok, false if overflow
(
    uint64_t *restrict c,   // c = a*b, or zero if overflow occurs
    const int64_t a,
    const int64_t b
)
{
    ASSERT (c != NULL) ;
    (*c) = 0 ;
    if (a == 0 || b == 0) return (true) ;
    if (a < 0 || b < 0 || a > GB_NMAX || b > GB_NMAX) return (false) ;
    uint64_t x ;
    bool ok = GB_uint64_multiply (&x, (uint64_t) a, (uint64_t) b) ;
    (*c) = ok ? x : 0 ;
    return (ok) ;
}

#endif

