//------------------------------------------------------------------------------
// GB_bitwise.h: definitions for bitwise operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GJ_bit* methods are only used in JIT kernels.

#ifndef GB_BITWISE_H
#define GB_BITWISE_H

//------------------------------------------------------------------------------
// bitget
//------------------------------------------------------------------------------

// bitget (x,k) returns a single bit from x, as 0 or 1, whose position is given
// by k.  k = 1 is the least significant bit, and k = bits (64 for uint64)
// is the most significant bit. If k is outside this range, the result is zero.

inline int8_t GB_bitget_int8 (int8_t x, int8_t k)
{
    if (k < 1 || k > 8) return (0) ;
    return ((x & (((int8_t) 1) << (k-1))) ? 1 : 0) ;
}

#define GJ_bitget_int8_DEFN                                      \
"int8_t GJ_bitget_int8 (int8_t x, int8_t k)                  \n" \
"{                                                           \n" \
"    if (k < 1 || k > 8) return (0) ;                        \n" \
"    return ((x & (((int8_t) 1) << (k-1))) ? 1 : 0) ;        \n" \
"}"

inline int16_t GB_bitget_int16 (int16_t x, int16_t k)
{
    if (k < 1 || k > 16) return (0) ;
    return ((x & (((int16_t) 1) << (k-1))) ? 1 : 0) ;
}

#define  GJ_bitget_int16_DEFN                                    \
"int16_t GJ_bitget_int16 (int16_t x, int16_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 16) return (0) ;                       \n" \
"    return ((x & (((int16_t) 1) << (k-1))) ? 1 : 0) ;       \n" \
"}"

inline int32_t GB_bitget_int32 (int32_t x, int32_t k)
{
    if (k < 1 || k > 32) return (0) ;
    return ((x & (((int32_t) 1) << (k-1))) ? 1 : 0) ;
}

#define  GJ_bitget_int32_DEFN                                    \
"int32_t GJ_bitget_int32 (int32_t x, int32_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 32) return (0) ;                       \n" \
"    return ((x & (((int32_t) 1) << (k-1))) ? 1 : 0) ;       \n" \
"}"

inline int64_t GB_bitget_int64 (int64_t x, int64_t k)
{
    if (k < 1 || k > 64) return (0) ;
    return ((x & (((int64_t) 1) << (k-1))) ? 1 : 0) ;
}

#define  GJ_bitget_int64_DEFN                                    \
"int64_t GJ_bitget_int64 (int64_t x, int64_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 64) return (0) ;                       \n" \
"    return ((x & (((int64_t) 1) << (k-1))) ? 1 : 0) ;       \n" \
"}"

inline uint8_t GB_bitget_uint8 (uint8_t x, uint8_t k)
{
    if (k < 1 || k > 8) return (0) ;
    return ((x & (((uint8_t) 1) << (k-1))) ? 1 : 0) ;
}

#define  GJ_bitget_uint8_DEFN                                    \
"uint8_t GJ_bitget_uint8 (uint8_t x, uint8_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 8) return (0) ;                        \n" \
"    return ((x & (((uint8_t) 1) << (k-1))) ? 1 : 0) ;       \n" \
"}"

inline uint16_t GB_bitget_uint16 (uint16_t x, uint16_t k)
{
    if (k < 1 || k > 16) return (0) ;
    return ((x & (((uint16_t) 1) << (k-1))) ? 1 : 0) ;
}

#define   GJ_bitget_uint16_DEFN                                  \
"uint16_t GJ_bitget_uint16 (uint16_t x, uint16_t k)          \n" \
"{                                                           \n" \
"    if (k < 1 || k > 16) return (0) ;                       \n" \
"    return ((x & (((uint16_t) 1) << (k-1))) ? 1 : 0) ;      \n" \
"}"

inline uint32_t GB_bitget_uint32 (uint32_t x, uint32_t k)
{
    if (k < 1 || k > 32) return (0) ;
    return ((x & (((uint32_t) 1) << (k-1))) ? 1 : 0) ;
}

#define   GJ_bitget_uint32_DEFN                                  \
"uint32_t GJ_bitget_uint32 (uint32_t x, uint32_t k)          \n" \
"{                                                           \n" \
"    if (k < 1 || k > 32) return (0) ;                       \n" \
"    return ((x & (((uint32_t) 1) << (k-1))) ? 1 : 0) ;      \n" \
"}"

inline uint64_t GB_bitget_uint64 (uint64_t x, uint64_t k)
{
    if (k < 1 || k > 64) return (0) ;
    return ((x & (((uint64_t) 1) << (k-1))) ? 1 : 0) ;
}

#define   GJ_bitget_uint64_DEFN                                  \
"uint64_t GJ_bitget_uint64 (uint64_t x, uint64_t k)          \n" \
"{                                                           \n" \
"    if (k < 1 || k > 64) return (0) ;                       \n" \
"    return ((x & (((uint64_t) 1) << (k-1))) ? 1 : 0) ;      \n" \
"}"

//------------------------------------------------------------------------------
// bitset
//------------------------------------------------------------------------------

// bitset (x,k) returns x modified by setting a bit from x to 1, whose position
// is given by k.  If k is in the range 1 to bits, then k gives the position
// of the bit to set.  If k is outside the range 1 to bits, then z = x is
// returned, unmodified.

inline int8_t GB_bitset_int8 (int8_t x, int8_t k)
{
    if (k < 1 || k > 8) return (x) ;
    return (x | (((int8_t) 1) << (k-1))) ;
}

#define GJ_bitset_int8_DEFN                                      \
"int8_t GJ_bitset_int8 (int8_t x, int8_t k)                  \n" \
"{                                                           \n" \
"    if (k < 1 || k > 8) return (x) ;                        \n" \
"    return (x | (((int8_t) 1) << (k-1))) ;                  \n" \
"}"

inline int16_t GB_bitset_int16 (int16_t x, int16_t k)
{
    if (k < 1 || k > 16) return (x) ;
    return (x | (((int16_t) 1) << (k-1))) ;
}

#define GJ_bitset_int16_DEFN                                     \
"int16_t GJ_bitset_int16 (int16_t x, int16_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 16) return (x) ;                       \n" \
"    return (x | (((int16_t) 1) << (k-1))) ;                 \n" \
"}"

inline int32_t GB_bitset_int32 (int32_t x, int32_t k)
{
    if (k < 1 || k > 32) return (x) ;
    return (x | (((int32_t) 1) << (k-1))) ;
}

#define  GJ_bitset_int32_DEFN                                    \
"int32_t GJ_bitset_int32 (int32_t x, int32_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 32) return (x) ;                       \n" \
"    return (x | (((int32_t) 1) << (k-1))) ;                 \n" \
"}"

inline int64_t GB_bitset_int64 (int64_t x, int64_t k)
{
    if (k < 1 || k > 64) return (x) ;
    int64_t z = (x | (((int64_t) 1) << (k-1))) ;
    return (z) ;
}

#define  GJ_bitset_int64_DEFN                                    \
"int64_t GJ_bitset_int64 (int64_t x, int64_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 64) return (x) ;                       \n" \
"    return (x | (((int64_t) 1) << (k-1))) ;                 \n" \
"}"

inline uint8_t GB_bitset_uint8 (uint8_t x, uint8_t k)
{
    if (k < 1 || k > 8) return (x) ;
    return (x | (((uint8_t) 1) << (k-1))) ;
}

#define  GJ_bitset_uint8_DEFN                                    \
"uint8_t GJ_bitset_uint8 (uint8_t x, uint8_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 8) return (x) ;                        \n" \
"    return (x | (((uint8_t) 1) << (k-1))) ;                 \n" \
"}"

inline uint16_t GB_bitset_uint16 (uint16_t x, uint16_t k)
{
    if (k < 1 || k > 16) return (x) ;
    return (x | (((uint16_t) 1) << (k-1))) ;
}

#define   GJ_bitset_uint16_DEFN                                  \
"uint16_t GJ_bitset_uint16 (uint16_t x, uint16_t k)          \n" \
"{                                                           \n" \
"    if (k < 1 || k > 16) return (x) ;                       \n" \
"    return (x | (((uint16_t) 1) << (k-1))) ;                \n" \
"}"

inline uint32_t GB_bitset_uint32 (uint32_t x, uint32_t k)
{
    if (k < 1 || k > 32) return (x) ;
    return (x | (((uint32_t) 1) << (k-1))) ;
}

#define   GJ_bitset_uint32_DEFN                                  \
"uint32_t GJ_bitset_uint32 (uint32_t x, uint32_t k)          \n" \
"{                                                           \n" \
"    if (k < 1 || k > 32) return (x) ;                       \n" \
"    return (x | (((uint32_t) 1) << (k-1))) ;                \n" \
"}"

inline uint64_t GB_bitset_uint64 (uint64_t x, uint64_t k)
{
    if (k < 1 || k > 64) return (x) ;
    return (x | (((uint64_t) 1) << (k-1))) ;
}

#define   GJ_bitset_uint64_DEFN                                  \
"uint64_t GJ_bitset_uint64 (uint64_t x, uint64_t k)          \n" \
"{                                                           \n" \
"    if (k < 1 || k > 64) return (x) ;                       \n" \
"    return (x | (((uint64_t) 1) << (k-1))) ;                \n" \
"}"

//------------------------------------------------------------------------------
// bitclr
//------------------------------------------------------------------------------

// bitclr (x,k) returns x modified by setting a bit from x to 0, whose position
// is given by k.  If k is in the range 1 to bits, then k gives the position of
// the bit to clear.  If k is outside the range 1 to GB_BITS, then z = x is
// returned, unmodified.

inline int8_t GB_bitclr_int8 (int8_t x, int8_t k)
{
    if (k < 1 || k > 8) return (x) ;
    return (x & ~(((int8_t) 1) << (k-1))) ;
}

#define GJ_bitclr_int8_DEFN                                      \
"int8_t GJ_bitclr_int8 (int8_t x, int8_t k)                  \n" \
"{                                                           \n" \
"    if (k < 1 || k > 8) return (x) ;                        \n" \
"    return (x & ~(((int8_t) 1) << (k-1))) ;                 \n" \
"}"

inline int16_t GB_bitclr_int16 (int16_t x, int16_t k)
{
    if (k < 1 || k > 16) return (x) ;
    return (x & ~(((int16_t) 1) << (k-1))) ;
}

#define  GJ_bitclr_int16_DEFN                                    \
"int16_t GJ_bitclr_int16 (int16_t x, int16_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 16) return (x) ;                       \n" \
"    return (x & ~(((int16_t) 1) << (k-1))) ;                \n" \
"}"

inline int32_t GB_bitclr_int32 (int32_t x, int32_t k)
{
    if (k < 1 || k > 32) return (x) ;
    return (x & ~(((int32_t) 1) << (k-1))) ;
}

#define  GJ_bitclr_int32_DEFN                                    \
"int32_t GJ_bitclr_int32 (int32_t x, int32_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 32) return (x) ;                       \n" \
"    return (x & ~(((int32_t) 1) << (k-1))) ;                \n" \
"}"

inline int64_t GB_bitclr_int64 (int64_t x, int64_t k)
{
    if (k < 1 || k > 64) return (x) ;
    return (x & ~(((int64_t) 1) << (k-1))) ;
}

#define  GJ_bitclr_int64_DEFN                                    \
"int64_t GJ_bitclr_int64 (int64_t x, int64_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 64) return (x) ;                       \n" \
"    return (x & ~(((int64_t) 1) << (k-1))) ;                \n" \
"}"

inline uint8_t GB_bitclr_uint8 (uint8_t x, uint8_t k)
{
    if (k < 1 || k > 8) return (x) ;
    return (x & ~(((uint8_t) 1) << (k-1))) ;
}

#define  GJ_bitclr_uint8_DEFN                                    \
"uint8_t GJ_bitclr_uint8 (uint8_t x, uint8_t k)              \n" \
"{                                                           \n" \
"    if (k < 1 || k > 8) return (x) ;                        \n" \
"    return (x & ~(((uint8_t) 1) << (k-1))) ;                \n" \
"}"

inline uint16_t GB_bitclr_uint16 (uint16_t x, uint16_t k)
{
    if (k < 1 || k > 16) return (x) ;
    return (x & ~(((uint16_t) 1) << (k-1))) ;
}

#define   GJ_bitclr_uint16_DEFN                                  \
"uint16_t GJ_bitclr_uint16 (uint16_t x, uint16_t k)          \n" \
"{                                                           \n" \
"    if (k < 1 || k > 16) return (x) ;                       \n" \
"    return (x & ~(((uint16_t) 1) << (k-1))) ;               \n" \
"}"

inline uint32_t GB_bitclr_uint32 (uint32_t x, uint32_t k)
{
    if (k < 1 || k > 32) return (x) ;
    return (x & ~(((uint32_t) 1) << (k-1))) ;
}

#define   GJ_bitclr_uint32_DEFN                                  \
"uint32_t GJ_bitclr_uint32 (uint32_t x, uint32_t k)          \n" \
"{                                                           \n" \
"    if (k < 1 || k > 32) return (x) ;                       \n" \
"    return (x & ~(((uint32_t) 1) << (k-1))) ;               \n" \
"}"

inline uint64_t GB_bitclr_uint64 (uint64_t x, uint64_t k)
{
    if (k < 1 || k > 64) return (x) ;
    return (x & ~(((uint64_t) 1) << (k-1))) ;
}

#define   GJ_bitclr_uint64_DEFN                                  \
"uint64_t GJ_bitclr_uint64 (uint64_t x, uint64_t k)          \n" \
"{                                                           \n" \
"    if (k < 1 || k > 64) return (x) ;                       \n" \
"    return (x & ~(((uint64_t) 1) << (k-1))) ;               \n" \
"}"

//------------------------------------------------------------------------------
// z = bitshift (x,y) when x and z are unsigned
//------------------------------------------------------------------------------

inline uint8_t GB_bitshift_uint8 (uint8_t x, int8_t k)
{
    if (k == 0)
    {
        // no shift to do at all
        return (x) ;
    }
    else if (k >= 8 || k <= -8)
    {
        // C11 states that the result of x << k is undefined if k is
        // negative or if k is greater than the # of bits in x.  Here, the
        // result is defined to be zero (the same as if shifting left
        // or right by 8).
        return (0) ;
    }
    else if (k > 0)
    {
        // left shift x by k bits.  z is defined by C11 as
        // (x * (2^k)) mod (uintmax + 1).
        return (x << k) ;
    }
    else
    {
        // right shift x by k bits.  z is defined by C11 as the
        // integral part of the quotient of x / (2^k).
        return (x >> (-k)) ;
    }
}

#define  GJ_bitshift_uint8_DEFN                                      \
"uint8_t GJ_bitshift_uint8 (uint8_t x, int8_t k)                 \n" \
"{                                                               \n" \
"    if (k == 0)                                                 \n" \
"    {                                                           \n" \
"        return (x) ;                                            \n" \
"    }                                                           \n" \
"    else if (k >= 8 || k <= -8)                                 \n" \
"    {                                                           \n" \
"        return (0) ;                                            \n" \
"    }                                                           \n" \
"    else if (k > 0)                                             \n" \
"    {                                                           \n" \
"        return (x << k) ;                                       \n" \
"    }                                                           \n" \
"    else                                                        \n" \
"    {                                                           \n" \
"        return (x >> (-k)) ;                                    \n" \
"    }                                                           \n" \
"}"

inline uint16_t GB_bitshift_uint16 (uint16_t x, int8_t k)
{
    if (k == 0)
    {
        return (x) ;
    }
    else if (k >= 16 || k <= -16)
    {
        return (0) ;
    }
    else if (k > 0)
    {
        return (x << k) ;
    }
    else
    {
        return (x >> (-k)) ;
    }
}

#define   GJ_bitshift_uint16_DEFN                                    \
"uint16_t GJ_bitshift_uint16 (uint16_t x, int8_t k)              \n" \
"{                                                               \n" \
"    if (k == 0)                                                 \n" \
"    {                                                           \n" \
"        return (x) ;                                            \n" \
"    }                                                           \n" \
"    else if (k >= 16 || k <= -16)                               \n" \
"    {                                                           \n" \
"        return (0) ;                                            \n" \
"    }                                                           \n" \
"    else if (k > 0)                                             \n" \
"    {                                                           \n" \
"        return (x << k) ;                                       \n" \
"    }                                                           \n" \
"    else                                                        \n" \
"    {                                                           \n" \
"        return (x >> (-k)) ;                                    \n" \
"    }                                                           \n" \
"}"

inline uint32_t GB_bitshift_uint32 (uint32_t x, int8_t k)
{
    if (k == 0)
    {
        return (x) ;
    }
    else if (k >= 32 || k <= -32)
    {
        return (0) ;
    }
    else if (k > 0)
    {
        return (x << k) ;
    }
    else
    {
        return (x >> (-k)) ;
    }
}

#define   GJ_bitshift_uint32_DEFN                                    \
"uint32_t GJ_bitshift_uint32 (uint32_t x, int8_t k)              \n" \
"{                                                               \n" \
"    if (k == 0)                                                 \n" \
"    {                                                           \n" \
"        return (x) ;                                            \n" \
"    }                                                           \n" \
"    else if (k >= 32 || k <= -32)                               \n" \
"    {                                                           \n" \
"        return (0) ;                                            \n" \
"    }                                                           \n" \
"    else if (k > 0)                                             \n" \
"    {                                                           \n" \
"        return (x << k) ;                                       \n" \
"    }                                                           \n" \
"    else                                                        \n" \
"    {                                                           \n" \
"        return (x >> (-k)) ;                                    \n" \
"    }                                                           \n" \
"}"

inline uint64_t GB_bitshift_uint64 (uint64_t x, int8_t k)
{
    if (k == 0)
    {
        return (x) ;
    }
    else if (k >= 64 || k <= -64)
    {
        return (0) ;
    }
    else if (k > 0)
    {
        return (x << k) ;
    }
    else
    {
        return (x >> (-k)) ;
    }
}

#define   GJ_bitshift_uint64_DEFN                                    \
"uint64_t GJ_bitshift_uint64 (uint64_t x, int8_t k)              \n" \
"{                                                               \n" \
"    if (k == 0)                                                 \n" \
"    {                                                           \n" \
"        return (x) ;                                            \n" \
"    }                                                           \n" \
"    else if (k >= 64 || k <= -64)                               \n" \
"    {                                                           \n" \
"        return (0) ;                                            \n" \
"    }                                                           \n" \
"    else if (k > 0)                                             \n" \
"    {                                                           \n" \
"        return (x << k) ;                                       \n" \
"    }                                                           \n" \
"    else                                                        \n" \
"    {                                                           \n" \
"        return (x >> (-k)) ;                                    \n" \
"    }                                                           \n" \
"}"

//------------------------------------------------------------------------------
// z = bitshift (x,y) when x and z are signed
//------------------------------------------------------------------------------

inline int8_t GB_bitshift_int8 (int8_t x, int8_t k)
{
    if (k == 0)
    {
        // no shift to do at all
        return (x) ;
    }
    else if (k >= 8)
    {
        // C11 states that z = x << k is undefined if k is greater
        // than the # of bits in x.  Here, the result is defined to be zero.
        return (0) ;
    }
    else if (k <= -8)
    {
        // C11 states that z = x >> (-k) is undefined if (-k) is
        // greater than the # of bits in x.  Here, the result is defined to
        // be the sign of x (z = 0 if x >= 0 and z = -1 if x is negative).
        return ((x >= 0) ? 0 : -1) ;
    }
    else if (k > 0)
    {
        // left shift x by k bits (where k is in range 1 to #bits - 1).
        // C11 states that z is defined only if x is non-negative and
        // x * (2^k) is representable.  This computation assumes x and z
        // are represented in 2's complement.  The result depends on the
        // underlying machine architecture and the compiler.
        return (x << k) ;
    }
    else
    {
        k = -k ;
        // right shift x by k bits (where k is in range 1 to 8)
        if (x >= 0)
        {
            // C11 defines z as the integral part of the quotient
            // of x / (2^k).
            return (x >> k) ;
        }
        else
        {
            // C11 states that the result is implementation-defined if
            // x is negative.  This computation assumes x and z are in 2's
            // complement, so 1-bits are shifted in on the left, and thus
            // the sign bit is always preserved.  The result depends on the
            // underlying machine architecture and the compiler.
            return ((x >> k) | (~(UINT8_MAX >> k))) ;
        }
    }
}

#define GJ_bitshift_int8_DEFN                                        \
"int8_t GJ_bitshift_int8 (int8_t x, int8_t k)                    \n" \
"{                                                               \n" \
"    if (k == 0)                                                 \n" \
"    {                                                           \n" \
"        return (x) ;                                            \n" \
"    }                                                           \n" \
"    else if (k >= 8)                                            \n" \
"    {                                                           \n" \
"        return (0) ;                                            \n" \
"    }                                                           \n" \
"    else if (k <= -8)                                           \n" \
"    {                                                           \n" \
"        return ((x >= 0) ? 0 : -1) ;                            \n" \
"    }                                                           \n" \
"    else if (k > 0)                                             \n" \
"    {                                                           \n" \
"        return (x << k) ;                                       \n" \
"    }                                                           \n" \
"    else                                                        \n" \
"    {                                                           \n" \
"        k = -k ;                                                \n" \
"        if (x >= 0)                                             \n" \
"        {                                                       \n" \
"            return (x >> k) ;                                   \n" \
"        }                                                       \n" \
"        else                                                    \n" \
"        {                                                       \n" \
"            return ((x >> k) | (~(UINT8_MAX >> k))) ;           \n" \
"        }                                                       \n" \
"    }                                                           \n" \
"}"

inline int16_t GB_bitshift_int16 (int16_t x, int8_t k)
{
    if (k == 0)
    {
        return (x) ;
    }
    else if (k >= 16)
    {
        return (0) ;
    }
    else if (k <= -16)
    {
        return ((x >= 0) ? 0 : -1) ;
    }
    else if (k > 0)
    {
        return (x << k) ;
    }
    else
    {
        k = -k ;
        if (x >= 0)
        {
            return (x >> k) ;
        }
        else
        {
            return ((x >> k) | (~(UINT16_MAX >> k))) ;
        }
    }
}

#define  GJ_bitshift_int16_DEFN                                      \
"int16_t GJ_bitshift_int16 (int16_t x, int8_t k)                 \n" \
"{                                                               \n" \
"    if (k == 0)                                                 \n" \
"    {                                                           \n" \
"        return (x) ;                                            \n" \
"    }                                                           \n" \
"    else if (k >= 16)                                           \n" \
"    {                                                           \n" \
"        return (0) ;                                            \n" \
"    }                                                           \n" \
"    else if (k <= -16)                                          \n" \
"    {                                                           \n" \
"        return ((x >= 0) ? 0 : -1) ;                            \n" \
"    }                                                           \n" \
"    else if (k > 0)                                             \n" \
"    {                                                           \n" \
"        return (x << k) ;                                       \n" \
"    }                                                           \n" \
"    else                                                        \n" \
"    {                                                           \n" \
"        k = -k ;                                                \n" \
"        if (x >= 0)                                             \n" \
"        {                                                       \n" \
"            return (x >> k) ;                                   \n" \
"        }                                                       \n" \
"        else                                                    \n" \
"        {                                                       \n" \
"            return ((x >> k) | (~(UINT16_MAX >> k))) ;          \n" \
"        }                                                       \n" \
"    }                                                           \n" \
"}"

inline int32_t GB_bitshift_int32 (int32_t x, int8_t k)
{
    if (k == 0)
    {
        return (x) ;
    }
    else if (k >= 32)
    {
        return (0) ;
    }
    else if (k <= -32)
    {
        return ((x >= 0) ? 0 : -1) ;
    }
    else if (k > 0)
    {
        return (x << k) ;
    }
    else
    {
        k = -k ;
        if (x >= 0)
        {
            return (x >> k) ;
        }
        else
        {
            return ((x >> k) | (~(UINT32_MAX >> k))) ;
        }
    }
}

#define  GJ_bitshift_int32_DEFN                                      \
"int32_t GJ_bitshift_int32 (int32_t x, int8_t k)                 \n" \
"{                                                               \n" \
"    if (k == 0)                                                 \n" \
"    {                                                           \n" \
"        return (x) ;                                            \n" \
"    }                                                           \n" \
"    else if (k >= 32)                                           \n" \
"    {                                                           \n" \
"        return (0) ;                                            \n" \
"    }                                                           \n" \
"    else if (k <= -32)                                          \n" \
"    {                                                           \n" \
"        return ((x >= 0) ? 0 : -1) ;                            \n" \
"    }                                                           \n" \
"    else if (k > 0)                                             \n" \
"    {                                                           \n" \
"        return (x << k) ;                                       \n" \
"    }                                                           \n" \
"    else                                                        \n" \
"    {                                                           \n" \
"        k = -k ;                                                \n" \
"        if (x >= 0)                                             \n" \
"        {                                                       \n" \
"            return (x >> k) ;                                   \n" \
"        }                                                       \n" \
"        else                                                    \n" \
"        {                                                       \n" \
"            return ((x >> k) | (~(UINT32_MAX >> k))) ;          \n" \
"        }                                                       \n" \
"    }                                                           \n" \
"}"

inline int64_t GB_bitshift_int64 (int64_t x, int8_t k)
{
    if (k == 0)
    {
        return (x) ;
    }
    else if (k >= 64)
    {
        return (0) ;
    }
    else if (k <= -64)
    {
        return ((x >= 0) ? 0 : -1) ;
    }
    else if (k > 0)
    {
        return (x << k) ;
    }
    else
    {
        k = -k ;
        if (x >= 0)
        {
            return (x >> k) ;
        }
        else
        {
            return ((x >> k) | (~(UINT64_MAX >> k))) ;
        }
    }
}

#define  GJ_bitshift_int64_DEFN                                      \
"int64_t GJ_bitshift_int64 (int64_t x, int8_t k)                 \n" \
"{                                                               \n" \
"    if (k == 0)                                                 \n" \
"    {                                                           \n" \
"        return (x) ;                                            \n" \
"    }                                                           \n" \
"    else if (k >= 64)                                           \n" \
"    {                                                           \n" \
"        return (0) ;                                            \n" \
"    }                                                           \n" \
"    else if (k <= -64)                                          \n" \
"    {                                                           \n" \
"        return ((x >= 0) ? 0 : -1) ;                            \n" \
"    }                                                           \n" \
"    else if (k > 0)                                             \n" \
"    {                                                           \n" \
"        return (x << k) ;                                       \n" \
"    }                                                           \n" \
"    else                                                        \n" \
"    {                                                           \n" \
"        k = -k ;                                                \n" \
"        if (x >= 0)                                             \n" \
"        {                                                       \n" \
"            return (x >> k) ;                                   \n" \
"        }                                                       \n" \
"        else                                                    \n" \
"        {                                                       \n" \
"            return ((x >> k) | (~(UINT64_MAX >> k))) ;          \n" \
"        }                                                       \n" \
"    }                                                           \n" \
"}"

#endif

