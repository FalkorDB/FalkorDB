//------------------------------------------------------------------------------
// GB_int64_mult: multiplying two integer values safely; result z is int64_t
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This macro computes the same thing as GB_uint64_multiply, except that it
// does not return a boolean ok flag to indicate whether or not integer
// overflow is detected.  Instead, it just computes c as INT64_MAX if overflow
// occurs.  Both inputs x and y must be >= 0.

#ifndef GB_INT64_MULT
#define GB_INT64_MULT(z,x,y)                            \
{                                                       \
    uint64_t a = (uint64_t) (x) ;                       \
    uint64_t b = (uint64_t) (y) ;                       \
    if (a == 0 || b == 0)                               \
    {                                                   \
        (z) = 0 ;                                       \
    }                                                   \
    else                                                \
    {                                                   \
        uint64_t a1 = a >> 30 ;                         \
        uint64_t b1 = b >> 30 ;                         \
        if (a1 > 0 && b1 > 0)                           \
        {                                               \
            (z) = INT64_MAX ;                           \
        }                                               \
        else                                            \
        {                                               \
            uint64_t a0 = a & 0x3FFFFFFFL ;             \
            uint64_t b0 = b & 0x3FFFFFFFL ;             \
            uint64_t t0 = a1*b0 ;                       \
            uint64_t t1 = a0*b1 ;                       \
            if (t0 >= 0x40000000L || t1 >= 0x40000000L) \
            {                                           \
                (z) = INT64_MAX ;                       \
            }                                           \
            else                                        \
            {                                           \
                uint64_t t2 = t0 + t1 ;                 \
                uint64_t c = (t2 << 30) + a0*b0 ;       \
                (z) = (int64_t) c ;                     \
            }                                           \
        }                                               \
    }                                                   \
}
#endif

