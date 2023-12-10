//------------------------------------------------------------------------------
// GB_uint64_multiply:  multiply two uint64_t and guard against overflow
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// c = a*b but check for overflow

// If c=a*b is large, c = UINT64_MAX is set, and the function returns false.
// "large" means that c=a*b might overflow; see details below.

// Otherwise c = a*b < INT64_MAX is guaranteed to be returned, and the function
// returns true.  In this case, c = a*b > GB_NMAX = (2^60)+1 may be returned,
// so if the caller requires c <= GB_NMAX, that must be checked after computing
// c=a*b with this function.

#ifndef GB_UINT64_MULTIPLY_H
#define GB_UINT64_MULTIPLY_H

#define GB_TWO_TO_THE_30 (0x40000000L)

GB_STATIC_INLINE bool GB_uint64_multiply    // c = a*b, return true if ok
(
    uint64_t *restrict c,
    const uint64_t a,
    const uint64_t b
)
{

    if (a <= 1 || b <= 1)
    { 
        (*c) = a*b ;
        return (true) ;
    }

    uint64_t a1 = a >> 30 ;     // a1 = a / 2^30
    uint64_t b1 = b >> 30 ;     // b1 = b / 2^30
    if (a1 > 0 && b1 > 0)
    { 
        // c = a*b will likely overflow, since both a and b are >= 2^30 and
        // thus c >= 2^60.  This is slightly pessimistic, but the function does
        // not need to compute the exact result in this case.  If a full matrix
        // has size a-by-b in this case, it will have 2^60 entries or more,
        // which is impossible.
        (*c) = UINT64_MAX ;
        return (false) ;
    }

    // a = a1 * 2^30 + a0
    uint64_t a0 = a & 0x3FFFFFFFL ;

    // b = b1 * 2^30 + b0
    uint64_t b0 = b & 0x3FFFFFFFL ;

    // a*b = (a1*b1) * 2^60 + (a1*b0 + a0*b1) * 2^30 + a0*b0

    // since either a1 or b1 are zero, a1*b1 is zero

    // a0, b0 are < 2^30
    // a1, b1 are < 2^34

    // a1*b0 < 2^64
    // a0*b1 < 2^64
    // a0*b0 < 2^60
    // thus

    // a*b = (a1*b0 + a0*b1) * 2^30 + a0*b0
    //     < (2^64  + 2^64 ) * 2^30 + 2^60

    // so it is safe to compute t0 and t1 without risk of overflow:

    uint64_t t0 = a1*b0 ;
    uint64_t t1 = a0*b1 ;

    // a*b = (t0 + t1) * 2^30 + a0*b0

    if (t0 >= GB_TWO_TO_THE_30 || t1 >= GB_TWO_TO_THE_30)
    { 
        // t >= 2^31, so t * 2^30 might overflow.  This is also slightly
        // pessimistic, but good enough for the usage of this function.
        (*c) = UINT64_MAX ;
        return (false) ;
    }

    // t = t0 + t1 < 2^30 + 2^30 < 2^31, so

    // c = a*b = t * 2^30 + a0*b0 < 2^61 + 2^60 < 2^62, no overflow possible
    uint64_t t = t0 + t1 ;
    (*c) = (t << 30) + a0*b0 ;
    return (true) ;
}

#endif

