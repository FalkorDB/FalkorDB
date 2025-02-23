//------------------------------------------------------------------------------
// GB_math.h: definitions for complex types, and mathematical operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "cast/GB_casting.h"

#ifndef GB_MATH_H
#define GB_MATH_H

//------------------------------------------------------------------------------
// integer division
//------------------------------------------------------------------------------

// The GJ_idiv* definitions are used in JIT kernels only.

// Integer division is done carefully so that GraphBLAS does not terminate the
// user's application on divide-by-zero.  To compute x/0: if x is zero, the
// result is zero (like NaN).  if x is negative, the result is the negative
// integer with biggest magnitude (like -infinity).  if x is positive, the
// result is the biggest positive integer (like +infinity).

inline int8_t GB_idiv_int8 (int8_t x, int8_t y)
{
    // returns x/y when x and y are int8_t
    if (y == -1)
    {
        // INT32_MIN/(-1) causes floating point exception; avoid it
        return (-x) ;
    }
    else if (y == 0)
    {
        // zero divided by zero gives 'integer Nan'
        // x/0 where x is nonzero: result is integer -Inf or +Inf
        return ((x == 0) ? 0 : ((x < 0) ? INT8_MIN : INT8_MAX)) ;
    }
    else
    {
        // normal case for signed integer division
        return (x / y) ;
    }
}

#define GJ_idiv_int8_DEFN                                                \
"int8_t GJ_idiv_int8 (int8_t x, int8_t y)                            \n" \
"{                                                                   \n" \
"    if (y == -1)                                                    \n" \
"    {                                                               \n" \
"        return (-x) ;                                               \n" \
"    }                                                               \n" \
"    else if (y == 0)                                                \n" \
"    {                                                               \n" \
"        return ((x == 0) ? 0 : ((x < 0) ? INT8_MIN : INT8_MAX)) ;   \n" \
"    }                                                               \n" \
"    else                                                            \n" \
"    {                                                               \n" \
"        return (x / y) ;                                            \n" \
"    }                                                               \n" \
"}"

inline int16_t GB_idiv_int16 (int16_t x, int16_t y)
{
    // returns x/y when x and y are int16_t
    if (y == -1)
    {
        // INT32_MIN/(-1) causes floating point exception; avoid it
        return (-x) ;
    }
    else if (y == 0)
    {
        // zero divided by zero gives 'integer Nan'
        // x/0 where x is nonzero: result is integer -Inf or +Inf
        return ((x == 0) ? 0 : ((x < 0) ? INT16_MIN : INT16_MAX)) ;
    }
    else
    {
        // normal case for signed integer division
        return (x / y) ;
    }
}

#define  GJ_idiv_int16_DEFN                                              \
"int16_t GJ_idiv_int16 (int16_t x, int16_t y)                        \n" \
"{                                                                   \n" \
"    if (y == -1)                                                    \n" \
"    {                                                               \n" \
"        return (-x) ;                                               \n" \
"    }                                                               \n" \
"    else if (y == 0)                                                \n" \
"    {                                                               \n" \
"        return ((x == 0) ? 0 : ((x < 0) ? INT16_MIN : INT16_MAX)) ; \n" \
"    }                                                               \n" \
"    else                                                            \n" \
"    {                                                               \n" \
"        return (x / y) ;                                            \n" \
"    }                                                               \n" \
"}"

inline int32_t GB_idiv_int32 (int32_t x, int32_t y)
{
    // returns x/y when x and y are int32_t
    if (y == -1)
    {
        // INT32_MIN/(-1) causes floating point exception; avoid it
        return (-x) ;
    }
    else if (y == 0)
    {
        // zero divided by zero gives 'integer Nan'
        // x/0 where x is nonzero: result is integer -Inf or +Inf
        return ((x == 0) ? 0 : ((x < 0) ? INT32_MIN : INT32_MAX)) ;
    }
    else
    {
        // normal case for signed integer division
        return (x / y) ;
    }
}

#define  GJ_idiv_int32_DEFN                                              \
"int32_t GJ_idiv_int32 (int32_t x, int32_t y)                        \n" \
"{                                                                   \n" \
"    if (y == -1)                                                    \n" \
"    {                                                               \n" \
"        return (-x) ;                                               \n" \
"    }                                                               \n" \
"    else if (y == 0)                                                \n" \
"    {                                                               \n" \
"        return ((x == 0) ? 0 : ((x < 0) ? INT32_MIN : INT32_MAX)) ; \n" \
"    }                                                               \n" \
"    else                                                            \n" \
"    {                                                               \n" \
"        return (x / y) ;                                            \n" \
"    }                                                               \n" \
"}"

inline int64_t GB_idiv_int64 (int64_t x, int64_t y)
{
    // returns x/y when x and y are int64_t
    if (y == -1)
    {
        // INT32_MIN/(-1) causes floating point exception; avoid it
        return (-x) ;
    }
    else if (y == 0)
    {
        // zero divided by zero gives 'integer Nan'
        // x/0 where x is nonzero: result is integer -Inf or +Inf
        return ((x == 0) ? 0 : ((x < 0) ? INT64_MIN : INT64_MAX)) ;
    }
    else
    {
        // normal case for signed integer division
        return (x / y) ;
    }
}

#define  GJ_idiv_int64_DEFN                                              \
"int64_t GJ_idiv_int64 (int64_t x, int64_t y)                        \n" \
"{                                                                   \n" \
"    if (y == -1)                                                    \n" \
"    {                                                               \n" \
"        return (-x) ;                                               \n" \
"    }                                                               \n" \
"    else if (y == 0)                                                \n" \
"    {                                                               \n" \
"        return ((x == 0) ? 0 : ((x < 0) ? INT64_MIN : INT64_MAX)) ; \n" \
"    }                                                               \n" \
"    else                                                            \n" \
"    {                                                               \n" \
"        return (x / y) ;                                            \n" \
"    }                                                               \n" \
"}"

inline uint8_t GB_idiv_uint8 (uint8_t x, uint8_t y)
{
    if (y == 0)
    {
        // x/0:  0/0 is integer Nan, otherwise result is +Inf
        return ((x == 0) ? 0 : UINT8_MAX) ;
    }
    else
    {
        // normal case for unsigned integer division
        return (x / y) ;
    }
}

#define  GJ_idiv_uint8_DEFN                                              \
"uint8_t GJ_idiv_uint8 (uint8_t x, uint8_t y)                        \n" \
"{                                                                   \n" \
"    if (y == 0)                                                     \n" \
"    {                                                               \n" \
"        return ((x == 0) ? 0 : UINT8_MAX) ;                         \n" \
"    }                                                               \n" \
"    else                                                            \n" \
"    {                                                               \n" \
"        return (x / y) ;                                            \n" \
"    }                                                               \n" \
"}"

inline uint16_t GB_idiv_uint16 (uint16_t x, uint16_t y)
{
    if (y == 0)
    {
        // x/0:  0/0 is integer Nan, otherwise result is +Inf
        return ((x == 0) ? 0 : UINT16_MAX) ;
    }
    else
    {
        // normal case for unsigned integer division
        return (x / y) ;
    }
}

#define   GJ_idiv_uint16_DEFN                                            \
"uint16_t GJ_idiv_uint16 (uint16_t x, uint16_t y)                    \n" \
"{                                                                   \n" \
"    if (y == 0)                                                     \n" \
"    {                                                               \n" \
"        return ((x == 0) ? 0 : UINT16_MAX) ;                        \n" \
"    }                                                               \n" \
"    else                                                            \n" \
"    {                                                               \n" \
"        return (x / y) ;                                            \n" \
"    }                                                               \n" \
"}"

inline uint32_t GB_idiv_uint32 (uint32_t x, uint32_t y)
{
    if (y == 0)
    {
        // x/0:  0/0 is integer Nan, otherwise result is +Inf
        return ((x == 0) ? 0 : UINT32_MAX) ;
    }
    else
    {
        // normal case for unsigned integer division
        return (x / y) ;
    }
}

#define   GJ_idiv_uint32_DEFN                                            \
"uint32_t GJ_idiv_uint32 (uint32_t x, uint32_t y)                    \n" \
"{                                                                   \n" \
"    if (y == 0)                                                     \n" \
"    {                                                               \n" \
"        return ((x == 0) ? 0 : UINT32_MAX) ;                        \n" \
"    }                                                               \n" \
"    else                                                            \n" \
"    {                                                               \n" \
"        return (x / y) ;                                            \n" \
"    }                                                               \n" \
"}"

inline uint64_t GB_idiv_uint64 (uint64_t x, uint64_t y)
{
    if (y == 0)
    {
        // x/0:  0/0 is integer Nan, otherwise result is +Inf
        return ((x == 0) ? 0 : UINT64_MAX) ;
    }
    else
    {
        // normal case for unsigned integer division
        return (x / y) ;
    }
}

#define   GJ_idiv_uint64_DEFN                                            \
"uint64_t GJ_idiv_uint64 (uint64_t x, uint64_t y)                    \n" \
"{                                                                   \n" \
"    if (y == 0)                                                     \n" \
"    {                                                               \n" \
"        return ((x == 0) ? 0 : UINT64_MAX) ;                        \n" \
"    }                                                               \n" \
"    else                                                            \n" \
"    {                                                               \n" \
"        return (x / y) ;                                            \n" \
"    }                                                               \n" \
"}"

//------------------------------------------------------------------------------
// complex division
//------------------------------------------------------------------------------

// The GJ_FC*_div definitions are used in JIT kernels only.

// complex division is problematic.  It is not supported at all on MS Visual
// Studio.  With other compilers, complex division exists but it has different
// NaN and Inf behavior as compared with MATLAB, which causes the tests to
// fail.  As a result, the built-in complex division is not used, even if the
// compiler supports it.

// Three cases below are from ACM Algo 116, R. L. Smith, 1962.

inline GxB_FC64_t GB_FC64_div (GxB_FC64_t x, GxB_FC64_t y)
{
    double xr = GB_creal (x) ;
    double xi = GB_cimag (x) ;
    double yr = GB_creal (y) ;
    double yi = GB_cimag (y) ;
    int yr_class = fpclassify (yr) ;
    int yi_class = fpclassify (yi) ;
    if (yi_class == FP_ZERO)
    {
        // (zr,zi) = (xr,xi) / (yr,0)
        return (GB_CMPLX64 (xr / yr, xi / yr)) ;
    }
    else if (yr_class == FP_ZERO)
    {
        // (zr,zi) = (xr,xi) / (0,yi) = (xi,-xr) / (yi,0)
        return (GB_CMPLX64 (xi / yi, -xr / yi)) ;
    }
    else if (yi_class == FP_INFINITE && yr_class == FP_INFINITE)
    {
        // Using Smith's method for a very special case
        double r = (signbit (yr) == signbit (yi)) ? (1) : (-1) ;
        double d = yr + r * yi ;
        return (GB_CMPLX64 ((xr + xi * r) / d, (xi - xr * r) / d)) ;
    }
    else if (fabs (yr) >= fabs (yi))
    {
        // Smith's method (1st case)
        double r = yi / yr ;
        double d = yr + r * yi ;
        return (GB_CMPLX64 ((xr + xi * r) / d, (xi - xr * r) / d)) ;
    }
    else
    {
        // Smith's method (2nd case)
        double r = yr / yi ;
        double d = r * yr + yi ;
        return (GB_CMPLX64 ((xr * r + xi) / d, (xi * r - xr) / d)) ;
    }
}

#define     GJ_FC64_div_DEFN                                             \
"GxB_FC64_t GJ_FC64_div (GxB_FC64_t x, GxB_FC64_t y)                 \n" \
"{                                                                   \n" \
"    double xr = GB_creal (x) ;                                      \n" \
"    double xi = GB_cimag (x) ;                                      \n" \
"    double yr = GB_creal (y) ;                                      \n" \
"    double yi = GB_cimag (y) ;                                      \n" \
"    int yr_class = fpclassify (yr) ;                                \n" \
"    int yi_class = fpclassify (yi) ;                                \n" \
"    if (yi_class == FP_ZERO)                                        \n" \
"    {                                                               \n" \
"        return (GJ_CMPLX64 (xr / yr, xi / yr)) ;                    \n" \
"    }                                                               \n" \
"    else if (yr_class == FP_ZERO)                                   \n" \
"    {                                                               \n" \
"        return (GJ_CMPLX64 (xi / yi, -xr / yi)) ;                   \n" \
"    }                                                               \n" \
"    else if (yi_class == FP_INFINITE && yr_class == FP_INFINITE)    \n" \
"    {                                                               \n" \
"        double r = (signbit (yr) == signbit (yi)) ? (1) : (-1) ;    \n" \
"        double d = yr + r * yi ;                                    \n" \
"        return (GJ_CMPLX64 ((xr + xi * r) / d, (xi - xr * r) / d)) ;\n" \
"    }                                                               \n" \
"    else if (fabs (yr) >= fabs (yi))                                \n" \
"    {                                                               \n" \
"        double r = yi / yr ;                                        \n" \
"        double d = yr + r * yi ;                                    \n" \
"        return (GJ_CMPLX64 ((xr + xi * r) / d, (xi - xr * r) / d)) ;\n" \
"    }                                                               \n" \
"    else                                                            \n" \
"    {                                                               \n" \
"        double r = yr / yi ;                                        \n" \
"        double d = r * yr + yi ;                                    \n" \
"        return (GJ_CMPLX64 ((xr * r + xi) / d, (xi * r - xr) / d)) ;\n" \
"    }                                                               \n" \
"}"

inline GxB_FC32_t GB_FC32_div (GxB_FC32_t x, GxB_FC32_t y)
{
    // single complex division: cast double complex, do the division,
    // and then cast back to single complex.
    double xr = (double) GB_crealf (x) ;
    double xi = (double) GB_cimagf (x) ;
    double yr = (double) GB_crealf (y) ;
    double yi = (double) GB_cimagf (y) ;
    GxB_FC64_t zz = GB_FC64_div (GB_CMPLX64 (xr, xi), GB_CMPLX64 (yr, yi)) ;
    return (GB_CMPLX32 ((float) GB_creal (zz), (float) GB_cimag (zz))) ;
}

#define     GJ_FC32_div_DEFN                                                \
"GxB_FC32_t GJ_FC32_div (GxB_FC32_t x, GxB_FC32_t y)                    \n" \
"{                                                                      \n" \
"    double xr = (double) GB_crealf (x) ;                               \n" \
"    double xi = (double) GB_cimagf (x) ;                               \n" \
"    double yr = (double) GB_crealf (y) ;                               \n" \
"    double yi = (double) GB_cimagf (y) ;                               \n" \
"    GxB_FC64_t zz ;                                                    \n" \
"    zz = GJ_FC64_div (GJ_CMPLX64 (xr, xi), GJ_CMPLX64 (yr, yi)) ;      \n" \
"    return (GJ_CMPLX32 ((float) GB_creal(zz), (float) GB_cimag(zz))) ; \n" \
"}"

//------------------------------------------------------------------------------
// z = x^y: wrappers for pow, powf, cpow, and cpowf
//------------------------------------------------------------------------------

//      if x or y are NaN, then z is NaN
//      if y is zero, then z is 1
//      if (x and y are complex but with zero imaginary parts, and
//          (x >= 0 or if y is an integer, NaN, or Inf)), then z is real
//      else use the built-in C library function, z = pow (x,y)

inline float GB_powf (float x, float y)
{
    int xr_class = fpclassify (x) ;
    int yr_class = fpclassify (y) ;
    if (xr_class == FP_NAN || yr_class == FP_NAN)
    {
        // z is nan if either x or y are nan
        return (NAN) ;
    }
    if (yr_class == FP_ZERO)
    {
        // z is 1 if y is zero
        return (1) ;
    }
    // otherwise, z = powf (x,y)
    return (powf (x, y)) ;
}

#define GJ_powf_DEFN                                                     \
 "float GJ_powf (float x, float y)                                   \n" \
"{                                                                   \n" \
"    #ifndef GB_CUDA_KERNEL                                          \n" \
"    int xr_class = fpclassify (x) ;                                 \n" \
"    int yr_class = fpclassify (y) ;                                 \n" \
"    if (xr_class == FP_NAN || yr_class == FP_NAN)                   \n" \
"    {                                                               \n" \
"        return (NAN) ;                                              \n" \
"    }                                                               \n" \
"    if (yr_class == FP_ZERO)                                        \n" \
"    {                                                               \n" \
"        return (1) ;                                                \n" \
"    }                                                               \n" \
"    #endif                                                          \n" \
"    return (powf (x, y)) ;                                          \n" \
"}"

inline double GB_pow (double x, double y)
{
    int xr_class = fpclassify (x) ;
    int yr_class = fpclassify (y) ;
    if (xr_class == FP_NAN || yr_class == FP_NAN)
    {
        // z is nan if either x or y are nan
        return (NAN) ;
    }
    if (yr_class == FP_ZERO)
    {
        // z is 1 if y is zero
        return (1) ;
    }
    // otherwise, z = pow (x,y)
    return (pow (x, y)) ;
}

#define GJ_pow_DEFN                                                      \
"double GJ_pow (double x, double y)                                  \n" \
"{                                                                   \n" \
"    #ifndef GB_CUDA_KERNEL                                          \n" \
"    int xr_class = fpclassify (x) ;                                 \n" \
"    int yr_class = fpclassify (y) ;                                 \n" \
"    if (xr_class == FP_NAN || yr_class == FP_NAN)                   \n" \
"    {                                                               \n" \
"        // z is nan if either x or y are nan                        \n" \
"        return (NAN) ;                                              \n" \
"    }                                                               \n" \
"    if (yr_class == FP_ZERO)                                        \n" \
"    {                                                               \n" \
"        // z is 1 if y is zero                                      \n" \
"        return (1) ;                                                \n" \
"    }                                                               \n" \
"    // otherwise, z = pow (x,y)                                     \n" \
"    #endif                                                          \n" \
"    return (pow (x, y)) ;                                           \n" \
"}"

inline GxB_FC32_t GB_FC32_pow (GxB_FC32_t x, GxB_FC32_t y)
{
    float xr = GB_crealf (x) ;
    float yr = GB_crealf (y) ;
    int xr_class = fpclassify (xr) ;
    int yr_class = fpclassify (yr) ;
    int xi_class = fpclassify (GB_cimagf (x)) ;
    int yi_class = fpclassify (GB_cimagf (y)) ;
    if (xi_class == FP_ZERO && yi_class == FP_ZERO)
    {
        // both x and y are real; see if z should be real
        if (xr >= 0 || yr_class == FP_NAN ||
            yr_class == FP_INFINITE || yr == truncf (yr))
        {
            // z is real if x >= 0, or if y is an integer, NaN, or Inf
            return (GB_CMPLX32 (GB_powf (xr, yr), 0)) ;
        }
    }
    if (xr_class == FP_NAN || xi_class == FP_NAN ||
        yr_class == FP_NAN || yi_class == FP_NAN)
    {
        // z is (nan,nan) if any part of x or y are nan
        return (GB_CMPLX32 (NAN, NAN)) ;
    }
    if (yr_class == FP_ZERO && yi_class == FP_ZERO)
    {
        // z is (1,0) if y is (0,0)
        return (GxB_CMPLXF (1, 0)) ;
    }
    return (GB_cpowf (x, y)) ;
}

#define     GJ_FC32_pow_DEFN                                             \
"GxB_FC32_t GJ_FC32_pow (GxB_FC32_t x, GxB_FC32_t y)                 \n" \
"{                                                                   \n" \
"    float xr = GB_crealf (x) ;                                      \n" \
"    float yr = GB_crealf (y) ;                                      \n" \
"    int xr_class = fpclassify (xr) ;                                \n" \
"    int yr_class = fpclassify (yr) ;                                \n" \
"    int xi_class = fpclassify (GB_cimagf (x)) ;                     \n" \
"    int yi_class = fpclassify (GB_cimagf (y)) ;                     \n" \
"    if (xi_class == FP_ZERO && yi_class == FP_ZERO)                 \n" \
"    {                                                               \n" \
"        if (xr >= 0 || yr_class == FP_NAN ||                        \n" \
"            yr_class == FP_INFINITE || yr == truncf (yr))           \n" \
"        {                                                           \n" \
"            return (GJ_CMPLX32 (GJ_powf (xr, yr), 0)) ;             \n" \
"        }                                                           \n" \
"    }                                                               \n" \
"    if (xr_class == FP_NAN || xi_class == FP_NAN ||                 \n" \
"        yr_class == FP_NAN || yi_class == FP_NAN)                   \n" \
"    {                                                               \n" \
"        return (GJ_CMPLX32 (NAN, NAN)) ;                            \n" \
"    }                                                               \n" \
"    if (yr_class == FP_ZERO && yi_class == FP_ZERO)                 \n" \
"    {                                                               \n" \
"        return (GxB_CMPLXF (1, 0)) ;                                \n" \
"    }                                                               \n" \
"    return (GB_cpowf (x, y)) ;                                      \n" \
"}"

inline GxB_FC64_t GB_FC64_pow (GxB_FC64_t x, GxB_FC64_t y)
{
    double xr = GB_creal (x) ;
    double yr = GB_creal (y) ;
    int xr_class = fpclassify (xr) ;
    int yr_class = fpclassify (yr) ;
    int xi_class = fpclassify (GB_cimag (x)) ;
    int yi_class = fpclassify (GB_cimag (y)) ;
    if (xi_class == FP_ZERO && yi_class == FP_ZERO)
    {
        // both x and y are real; see if z should be real
        if (xr >= 0 || yr_class == FP_NAN ||
            yr_class == FP_INFINITE || yr == trunc (yr))
        {
            // z is real if x >= 0, or if y is an integer, NaN, or Inf
            return (GB_CMPLX64 (GB_pow (xr, yr), 0)) ;
        }
    }
    if (xr_class == FP_NAN || xi_class == FP_NAN ||
        yr_class == FP_NAN || yi_class == FP_NAN)
    {
        // z is (nan,nan) if any part of x or y are nan
        return (GB_CMPLX64 (NAN, NAN)) ;
    }
    if (yr_class == FP_ZERO && yi_class == FP_ZERO)
    {
        // z is (1,0) if y is (0,0)
        return (GxB_CMPLX (1, 0)) ;
    }
    return (GB_cpow (x, y)) ;
}

#define     GJ_FC64_pow_DEFN                                             \
"GxB_FC64_t GJ_FC64_pow (GxB_FC64_t x, GxB_FC64_t y)                 \n" \
"{                                                                   \n" \
"    double xr = GB_creal (x) ;                                      \n" \
"    double yr = GB_creal (y) ;                                      \n" \
"    int xr_class = fpclassify (xr) ;                                \n" \
"    int yr_class = fpclassify (yr) ;                                \n" \
"    int xi_class = fpclassify (GB_cimag (x)) ;                      \n" \
"    int yi_class = fpclassify (GB_cimag (y)) ;                      \n" \
"    if (xi_class == FP_ZERO && yi_class == FP_ZERO)                 \n" \
"    {                                                               \n" \
"        if (xr >= 0 || yr_class == FP_NAN ||                        \n" \
"            yr_class == FP_INFINITE || yr == trunc (yr))            \n" \
"        {                                                           \n" \
"            return (GJ_CMPLX64 (GJ_pow (xr, yr), 0)) ;              \n" \
"        }                                                           \n" \
"    }                                                               \n" \
"    if (xr_class == FP_NAN || xi_class == FP_NAN ||                 \n" \
"        yr_class == FP_NAN || yi_class == FP_NAN)                   \n" \
"    {                                                               \n" \
"        return (GJ_CMPLX64 (NAN, NAN)) ;                            \n" \
"    }                                                               \n" \
"    if (yr_class == FP_ZERO && yi_class == FP_ZERO)                 \n" \
"    {                                                               \n" \
"        return (GxB_CMPLX (1, 0)) ;                                 \n" \
"    }                                                               \n" \
"    return (GB_cpow (x, y)) ;                                       \n" \
"}"

inline int8_t GB_pow_int8 (int8_t x, int8_t y)
{
    return (GB_cast_to_int8_t (GB_pow ((double) x, (double) y))) ;
}

#define GJ_pow_int8_DEFN                                                \
"int8_t GJ_pow_int8 (int8_t x, int8_t y)                            \n" \
"{                                                                  \n" \
"    return (GJ_cast_to_int8 (GJ_pow ((double) x, (double) y))) ;   \n" \
"}"

inline int16_t GB_pow_int16 (int16_t x, int16_t y)
{
    return (GB_cast_to_int16_t (GB_pow ((double) x, (double) y))) ;
}

#define  GJ_pow_int16_DEFN                                              \
"int16_t GJ_pow_int16 (int16_t x, int16_t y)                        \n" \
"{                                                                  \n" \
"    return (GJ_cast_to_int16 (GJ_pow ((double) x, (double) y))) ;  \n" \
"}"

inline int32_t GB_pow_int32 (int32_t x, int32_t y)
{
    return (GB_cast_to_int32_t (GB_pow ((double) x, (double) y))) ;
}

#define  GJ_pow_int32_DEFN                                              \
"int32_t GJ_pow_int32 (int32_t x, int32_t y)                        \n" \
"{                                                                  \n" \
"    return (GJ_cast_to_int32 (GJ_pow ((double) x, (double) y))) ;  \n" \
"}"

inline int64_t GB_pow_int64 (int64_t x, int64_t y)
{
    return (GB_cast_to_int64_t (GB_pow ((double) x, (double) y))) ;
}

#define  GJ_pow_int64_DEFN                                              \
"int64_t GJ_pow_int64 (int64_t x, int64_t y)                        \n" \
"{                                                                  \n" \
"    return (GJ_cast_to_int64 (GJ_pow ((double) x, (double) y))) ;  \n" \
"}"

inline uint8_t GB_pow_uint8 (uint8_t x, uint8_t y)
{
    return (GB_cast_to_uint8_t (GB_pow ((double) x, (double) y))) ;
}

#define GJ_pow_uint8_DEFN                                               \
"int8_t GJ_pow_uint8 (int8_t x, int8_t y)                           \n" \
"{                                                                  \n" \
"    return (GJ_cast_to_uint8 (GJ_pow ((double) x, (double) y))) ;  \n" \
"}"

inline uint16_t GB_pow_uint16 (uint16_t x, uint16_t y)
{
    return (GB_cast_to_uint16_t (GB_pow ((double) x, (double) y))) ;
}

#define  GJ_pow_uint16_DEFN                                             \
"int16_t GJ_pow_uint16 (int16_t x, int16_t y)                       \n" \
"{                                                                  \n" \
"    return (GJ_cast_to_uint16 (GJ_pow ((double) x, (double) y))) ; \n" \
"}"

inline uint32_t GB_pow_uint32 (uint32_t x, uint32_t y)
{
    return (GB_cast_to_uint32_t (GB_pow ((double) x, (double) y))) ;
}

#define  GJ_pow_uint32_DEFN                                             \
"int32_t GJ_pow_uint32 (int32_t x, int32_t y)                       \n" \
"{                                                                  \n" \
"    return (GJ_cast_to_uint32 (GJ_pow ((double) x, (double) y))) ; \n" \
"}"

inline uint64_t GB_pow_uint64 (uint64_t x, uint64_t y)
{
    return (GB_cast_to_uint64_t (GB_pow ((double) x, (double) y))) ;
}

#define  GJ_pow_uint64_DEFN                                             \
"int64_t GJ_pow_uint64 (int64_t x, int64_t y)                       \n" \
"{                                                                  \n" \
"    return (GJ_cast_to_uint64 (GJ_pow ((double) x, (double) y))) ; \n" \
"}"

//------------------------------------------------------------------------------
// frexp for float and double
//------------------------------------------------------------------------------

inline float GB_frexpxf (float x)
{
    // ignore the exponent and just return the mantissa
    int exp_ignored ;
    return (frexpf (x, &exp_ignored)) ;
}

#define GJ_frexpxf_DEFN                                                 \
 "float GJ_frexpxf (float x)                                        \n" \
"{                                                                  \n" \
"    int exp_ignored ;                                              \n" \
"    return (frexpf (x, &exp_ignored)) ;                            \n" \
"}"

inline float GB_frexpef (float x)
{
    // ignore the mantissa and just return the exponent
    int exp ;
    (void) frexpf (x, &exp) ;
    return ((float) exp) ;
}

#define GJ_frexpef_DEFN                                                 \
 "float GJ_frexpef (float x)                                        \n" \
"{                                                                  \n" \
"    int exp ;                                                      \n" \
"    (void) frexpf (x, &exp) ;                                      \n" \
"    return ((float) exp) ;                                         \n" \
"}"

inline double GB_frexpx (double x)
{
    // ignore the exponent and just return the mantissa
    int exp_ignored ;
    return (frexp (x, &exp_ignored)) ;
}

#define GJ_frexpx_DEFN                                                  \
"double GJ_frexpx (double x)                                        \n" \
"{                                                                  \n" \
"    int exp_ignored ;                                              \n" \
"    return (frexp (x, &exp_ignored)) ;                             \n" \
"}"

inline double GB_frexpe (double x)
{
    // ignore the mantissa and just return the exponent
    int exp ;
    (void) frexp (x, &exp) ;
    return ((double) exp) ;
}

#define GJ_frexpe_DEFN                                                  \
"double GJ_frexpe (double x)                                        \n" \
"{                                                                  \n" \
"    int exp ;                                                      \n" \
"    (void) frexp (x, &exp) ;                                       \n" \
"    return ((double) exp) ;                                        \n" \
"}"

//------------------------------------------------------------------------------
// signum functions
//------------------------------------------------------------------------------

inline float GB_signumf (float x)
{
    if (isnan (x)) return (x) ;
    return ((float) ((x < 0) ? (-1) : ((x > 0) ? 1 : 0))) ;
}

#define GJ_signumf_DEFN                                                 \
 "float GJ_signumf (float x)                                        \n" \
"{                                                                  \n" \
"    if (isnan (x)) return (x) ;                                    \n" \
"    return ((float) ((x < 0) ? (-1) : ((x > 0) ? 1 : 0))) ;        \n" \
"}"

inline double GB_signum (double x)
{
    if (isnan (x)) return (x) ;
    return ((double) ((x < 0) ? (-1) : ((x > 0) ? 1 : 0))) ;
}

#define GJ_signum_DEFN                                                  \
"double GJ_signum (double x)                                        \n" \
"{                                                                  \n" \
"    if (isnan (x)) return (x) ;                                    \n" \
"    return ((double) ((x < 0) ? (-1) : ((x > 0) ? 1 : 0))) ;       \n" \
"}"

inline GxB_FC32_t GB_csignumf (GxB_FC32_t x)
{
    if (GB_crealf (x) == 0 && GB_cimagf (x) == 0)
    {
        return (GxB_CMPLXF (0,0)) ;
    }
    float y = GB_cabsf (x) ;
    return (GB_CMPLX32 (GB_crealf (x) / y, GB_cimagf (x) / y)) ;
}

#define     GJ_csignumf_DEFN                                            \
"GxB_FC32_t GJ_csignumf (GxB_FC32_t x)                              \n" \
"{                                                                  \n" \
"    if (GB_crealf (x) == 0 && GB_cimagf (x) == 0)                  \n" \
"    {                                                              \n" \
"        return (GxB_CMPLXF (0,0)) ;                                \n" \
"    }                                                              \n" \
"    float y = GB_cabsf (x) ;                                       \n" \
"    return (GJ_CMPLX32 (GB_crealf (x) / y, GB_cimagf (x) / y)) ;   \n" \
"}"

inline GxB_FC64_t GB_csignum (GxB_FC64_t x)
{
    if (GB_creal (x) == 0 && GB_cimag (x) == 0)
    {
        return (GxB_CMPLX (0,0)) ;
    }
    double y = GB_cabs (x) ;
    return (GB_CMPLX64 (GB_creal (x) / y, GB_cimag (x) / y)) ;
}

#define     GJ_csignum_DEFN                                             \
"GxB_FC64_t GJ_csignum (GxB_FC64_t x)                               \n" \
"{                                                                  \n" \
"    if (GB_creal (x) == 0 && GB_cimag (x) == 0)                    \n" \
"    {                                                              \n" \
"        return (GxB_CMPLX (0,0)) ;                                 \n" \
"    }                                                              \n" \
"    double y = GB_cabs (x) ;                                       \n" \
"    return (GJ_CMPLX64 (GB_creal (x) / y, GB_cimag (x) / y)) ;     \n" \
"}"

//------------------------------------------------------------------------------
// complex functions
//------------------------------------------------------------------------------

// The C11 math.h header defines the ceil, floor, round, trunc,
// exp2, expm1, log10, log1pm, or log2 functions for float and double,
// but the corresponding functions do not appear in the C11 complex.h.
// These functions are used instead, for float complex and double complex.

//------------------------------------------------------------------------------
// z = ceil (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_cceilf (GxB_FC32_t x)
{
    return (GB_CMPLX32 (ceilf (GB_crealf (x)), ceilf (GB_cimagf (x)))) ;
}

#define     GJ_cceilf_DEFN                                                    \
"GxB_FC32_t GJ_cceilf (GxB_FC32_t x)                                      \n" \
"{                                                                        \n" \
"    return (GJ_CMPLX32 (ceilf (GB_crealf (x)), ceilf (GB_cimagf (x)))) ; \n" \
"}"

//------------------------------------------------------------------------------
// z = ceil (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_cceil (GxB_FC64_t x)
{
    return (GB_CMPLX64 (ceil (GB_creal (x)), ceil (GB_cimag (x)))) ;
}

#define     GJ_cceil_DEFN                                                   \
"GxB_FC64_t GJ_cceil (GxB_FC64_t x)                                     \n" \
"{                                                                      \n" \
"    return (GJ_CMPLX64 (ceil (GB_creal (x)), ceil (GB_cimag (x)))) ;   \n" \
"}"

//------------------------------------------------------------------------------
// z = floor (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_cfloorf (GxB_FC32_t x)
{
    return (GB_CMPLX32 (floorf (GB_crealf (x)), floorf (GB_cimagf (x)))) ;
}

#define     GJ_cfloorf_DEFN                                                    \
"GxB_FC32_t GJ_cfloorf (GxB_FC32_t x)                                      \n" \
"{                                                                         \n" \
"    return (GJ_CMPLX32 (floorf (GB_crealf (x)), floorf (GB_cimagf (x)))) ;\n" \
"}"

//------------------------------------------------------------------------------
// z = floor (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_cfloor (GxB_FC64_t x)
{
    return (GB_CMPLX64 (floor (GB_creal (x)), floor (GB_cimag (x)))) ;
}

#define     GJ_cfloor_DEFN                                                  \
"GxB_FC64_t GJ_cfloor (GxB_FC64_t x)                                    \n" \
"{                                                                      \n" \
"    return (GJ_CMPLX64 (floor (GB_creal (x)), floor (GB_cimag (x)))) ; \n" \
"}"

//------------------------------------------------------------------------------
// z = round (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_croundf (GxB_FC32_t x)
{
    return (GB_CMPLX32 (roundf (GB_crealf (x)), roundf (GB_cimagf (x)))) ;
}

#define     GJ_croundf_DEFN                                                    \
"GxB_FC32_t GJ_croundf (GxB_FC32_t x)                                      \n" \
"{                                                                         \n" \
"    return (GJ_CMPLX32 (roundf (GB_crealf (x)), roundf (GB_cimagf (x)))) ;\n" \
"}"

//------------------------------------------------------------------------------
// z = round (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_cround (GxB_FC64_t x)
{
    return (GB_CMPLX64 (round (GB_creal (x)), round (GB_cimag (x)))) ;
}

#define     GJ_cround_DEFN                                                  \
"GxB_FC64_t GJ_cround (GxB_FC64_t x)                                    \n" \
"{                                                                      \n" \
"    return (GJ_CMPLX64 (round (GB_creal (x)), round (GB_cimag (x)))) ; \n" \
"}"

//------------------------------------------------------------------------------
// z = trunc (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_ctruncf (GxB_FC32_t x)
{
    return (GB_CMPLX32 (truncf (GB_crealf (x)), truncf (GB_cimagf (x)))) ;
}

#define     GJ_ctruncf_DEFN                                                    \
"GxB_FC32_t GJ_ctruncf (GxB_FC32_t x)                                      \n" \
"{                                                                         \n" \
"    return (GJ_CMPLX32 (truncf (GB_crealf (x)), truncf (GB_cimagf (x)))) ;\n" \
"}"

//------------------------------------------------------------------------------
// z = trunc (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_ctrunc (GxB_FC64_t x)
{
    return (GB_CMPLX64 (trunc (GB_creal (x)), trunc (GB_cimag (x)))) ;
}

#define     GJ_ctrunc_DEFN                                                  \
"GxB_FC64_t GJ_ctrunc (GxB_FC64_t x)                                    \n" \
"{                                                                      \n" \
"    return (GJ_CMPLX64 (trunc (GB_creal (x)), trunc (GB_cimag (x)))) ; \n" \
"}"

//------------------------------------------------------------------------------
// z = exp2 (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_cexp2f (GxB_FC32_t x)
{
    if (fpclassify (GB_cimagf (x)) == FP_ZERO)
    {
        // x is real, use exp2f
        return (GB_CMPLX32 (exp2f (GB_crealf (x)), 0)) ;
    }
    return (GB_FC32_pow (GxB_CMPLXF (2,0), x)) ;     // z = 2^x
}

#define     GJ_cexp2f_DEFN                                              \
"GxB_FC32_t GJ_cexp2f (GxB_FC32_t x)                                \n" \
"{                                                                  \n" \
"    if (fpclassify (GB_cimagf (x)) == FP_ZERO)                     \n" \
"    {                                                              \n" \
"        return (GJ_CMPLX32 (exp2f (GB_crealf (x)), 0)) ;           \n" \
"    }                                                              \n" \
"    return (GJ_FC32_pow (GxB_CMPLXF (2,0), x)) ;                   \n" \
"}"

//------------------------------------------------------------------------------
// z = exp2 (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_cexp2 (GxB_FC64_t x)
{
    if (fpclassify (GB_cimag (x)) == FP_ZERO)
    {
        // x is real, use exp2
        return (GB_CMPLX64 (exp2 (GB_creal (x)), 0)) ;
    }
    return (GB_FC64_pow (GxB_CMPLX (2,0), x)) ;      // z = 2^x
}

#define     GJ_cexp2_DEFN                                               \
"GxB_FC64_t GJ_cexp2 (GxB_FC64_t x)                                 \n" \
"{                                                                  \n" \
"    if (fpclassify (GB_cimag (x)) == FP_ZERO)                      \n" \
"    {                                                              \n" \
"        return (GJ_CMPLX64 (exp2 (GB_creal (x)), 0)) ;             \n" \
"    }                                                              \n" \
"    return (GJ_FC64_pow (GxB_CMPLX (2,0), x)) ;                    \n" \
"}"

//------------------------------------------------------------------------------
// z = expm1 (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_cexpm1 (GxB_FC64_t x)
{
    // FUTURE: GB_cexpm1 is not accurate
    // z = cexp (x) - 1
    GxB_FC64_t z = GB_cexp (x) ;
    return (GB_CMPLX64 (GB_creal (z) - 1, GB_cimag (z))) ;
}

#define     GJ_cexpm1_DEFN                                              \
"GxB_FC64_t GJ_cexpm1 (GxB_FC64_t x)                                \n" \
"{                                                                  \n" \
"    GxB_FC64_t z = GB_cexp (x) ;                                   \n" \
"    return (GJ_CMPLX64 (GB_creal (z) - 1, GB_cimag (z))) ;         \n" \
"}"

//------------------------------------------------------------------------------
// z = expm1 (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_cexpm1f (GxB_FC32_t x)
{
    // typecast to double and use GB_cexpm1
    GxB_FC64_t z = GB_CMPLX64 ((double) GB_crealf (x),
                               (double) GB_cimagf (x)) ;
    z = GB_cexpm1 (z) ;
    return (GB_CMPLX32 ((float) GB_creal (z),
                        (float) GB_cimag (z))) ;
}

#define     GJ_cexpm1f_DEFN                                             \
"GxB_FC32_t GJ_cexpm1f (GxB_FC32_t x)                               \n" \
"{                                                                  \n" \
"    GxB_FC64_t z = GJ_CMPLX64 ((double) GB_crealf (x),             \n" \
"                               (double) GB_cimagf (x)) ;           \n" \
"    z = GJ_cexpm1 (z) ;                                            \n" \
"    return (GJ_CMPLX32 ((float) GB_creal (z),                      \n" \
"                        (float) GB_cimag (z))) ;                   \n" \
"}"

//------------------------------------------------------------------------------
// z = log1p (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_clog1p (GxB_FC64_t x)
{
    // FUTURE: GB_clog1p is not accurate
    // z = clog (1+x)
    return (GB_clog (GB_CMPLX64 (GB_creal (x) + 1, GB_cimag (x)))) ;
}

#define     GJ_clog1p_DEFN                                                  \
"GxB_FC64_t GJ_clog1p (GxB_FC64_t x)                                    \n" \
"{                                                                      \n" \
"    return (GB_clog (GJ_CMPLX64 (GB_creal (x) + 1, GB_cimag (x)))) ;   \n" \
"}"

//------------------------------------------------------------------------------
// z = log1p (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_clog1pf (GxB_FC32_t x)
{
    // typecast to double and use GB_clog1p
    GxB_FC64_t z = GB_CMPLX64 ((double) GB_crealf (x),
                               (double) GB_cimagf (x)) ;
    z = GB_clog1p (z) ;
    return (GB_CMPLX32 ((float) GB_creal (z),
                        (float) GB_cimag (z))) ;
}

#define     GJ_clog1pf_DEFN                                             \
"GxB_FC32_t GJ_clog1pf (GxB_FC32_t x)                               \n" \
"{                                                                  \n" \
"    GxB_FC64_t z = GJ_CMPLX64 ((double) GB_crealf (x),             \n" \
"                               (double) GB_cimagf (x)) ;           \n" \
"    z = GJ_clog1p (z) ;                                            \n" \
"    return (GJ_CMPLX32 ((float) GB_creal (z),                      \n" \
"                        (float) GB_cimag (z))) ;                   \n" \
"}"

//------------------------------------------------------------------------------
// z = log10 (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_clog10f (GxB_FC32_t x)
{
    // z = log (x) / log (10)
    return (GB_FC32_div (GB_clogf (x), GxB_CMPLXF (2.3025851f, 0))) ;
}

#define     GJ_clog10f_DEFN                                                 \
"GxB_FC32_t GJ_clog10f (GxB_FC32_t x)                                   \n" \
"{                                                                      \n" \
"    return (GJ_FC32_div (GB_clogf (x), GxB_CMPLXF (2.3025851f, 0))) ;  \n" \
"}"

//------------------------------------------------------------------------------
// z = log10 (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_clog10 (GxB_FC64_t x)
{
    // z = log (x) / log (10)
    return (GB_FC64_div (GB_clog (x),
        GxB_CMPLX (2.302585092994045901, 0))) ;
}

#define     GJ_clog10_DEFN                                              \
"GxB_FC64_t GJ_clog10 (GxB_FC64_t x)                                \n" \
"{                                                                  \n" \
"    return (GJ_FC64_div (GB_clog (x),                              \n" \
"        GxB_CMPLX (2.302585092994045901, 0))) ;                    \n" \
"}"

//------------------------------------------------------------------------------
// z = log2 (x) for float complex
//------------------------------------------------------------------------------

inline GxB_FC32_t GB_clog2f (GxB_FC32_t x)
{
    // z = log (x) / log (2)
    return (GB_FC32_div (GB_clogf (x), GxB_CMPLXF (0.69314718f, 0))) ;
}

#define     GJ_clog2f_DEFN                                                  \
"GxB_FC32_t GJ_clog2f (GxB_FC32_t x)                                    \n" \
"{                                                                      \n" \
"    return (GJ_FC32_div (GB_clogf (x), GxB_CMPLXF (0.69314718f, 0))) ; \n" \
"}"

//------------------------------------------------------------------------------
// z = log2 (x) for double complex
//------------------------------------------------------------------------------

inline GxB_FC64_t GB_clog2 (GxB_FC64_t x)
{
    // z = log (x) / log (2)
    return (GB_FC64_div (GB_clog (x),
        GxB_CMPLX (0.693147180559945286, 0))) ;
}

#define     GJ_clog2_DEFN                                               \
"GxB_FC64_t GJ_clog2 (GxB_FC64_t x)                                 \n" \
"{                                                                  \n" \
"    return (GJ_FC64_div (GB_clog (x),                              \n" \
"        GxB_CMPLX (0.693147180559945286, 0))) ;                    \n" \
"}"

//------------------------------------------------------------------------------
// z = isinf (x) for float complex
//------------------------------------------------------------------------------

inline bool GB_cisinff (GxB_FC32_t x)
{
    return (isinf (GB_crealf (x)) || isinf (GB_cimagf (x))) ;
}

#define GJ_cisinff_DEFN                                                 \
  "bool GJ_cisinff (GxB_FC32_t x)                                   \n" \
"{                                                                  \n" \
"    return (isinf (GB_crealf (x)) || isinf (GB_cimagf (x))) ;      \n" \
"}"

//------------------------------------------------------------------------------
// z = isinf (x) for double complex
//------------------------------------------------------------------------------

inline bool GB_cisinf (GxB_FC64_t x)
{
    return (isinf (GB_creal (x)) || isinf (GB_cimag (x))) ;
}

#define GJ_cisinf_DEFN                                                  \
  "bool GJ_cisinf (GxB_FC64_t x)                                    \n" \
"{                                                                  \n" \
"    return (isinf (GB_creal (x)) || isinf (GB_cimag (x))) ;        \n" \
"}"

//------------------------------------------------------------------------------
// z = isnan (x) for float complex
//------------------------------------------------------------------------------

inline bool GB_cisnanf (GxB_FC32_t x)
{
    return (isnan (GB_crealf (x)) || isnan (GB_cimagf (x))) ;
}

#define GJ_cisnanf_DEFN                                                 \
  "bool GJ_cisnanf (GxB_FC32_t x)                                   \n" \
"{                                                                  \n" \
"    return (isnan (GB_crealf (x)) || isnan (GB_cimagf (x))) ;      \n" \
"}"

//------------------------------------------------------------------------------
// z = isnan (x) for double complex
//------------------------------------------------------------------------------

inline bool GB_cisnan (GxB_FC64_t x)
{
    return (isnan (GB_creal (x)) || isnan (GB_cimag (x))) ;
}

#define GJ_cisnan_DEFN                                                  \
  "bool GJ_cisnan (GxB_FC64_t x)                                    \n" \
"{                                                                  \n" \
"    return (isnan (GB_creal (x)) || isnan (GB_cimag (x))) ;        \n" \
"}"

//------------------------------------------------------------------------------
// z = isfinite (x) for float complex
//------------------------------------------------------------------------------

inline bool GB_cisfinitef (GxB_FC32_t x)
{
    return (isfinite (GB_crealf (x)) && isfinite (GB_cimagf (x))) ;
}

#define GJ_cisfinitef_DEFN                                              \
  "bool GJ_cisfinitef (GxB_FC32_t x)                                \n" \
"{                                                                  \n" \
"    return (isfinite (GB_crealf (x)) && isfinite (GB_cimagf (x))) ;\n" \
"}"

//------------------------------------------------------------------------------
// z = isfinite (x) for double complex
//------------------------------------------------------------------------------

inline bool GB_cisfinite (GxB_FC64_t x)
{
    return (isfinite (GB_creal (x)) && isfinite (GB_cimag (x))) ;
}

#define GJ_cisfinite_DEFN                                               \
  "bool GJ_cisfinite (GxB_FC64_t x)                                 \n" \
"{                                                                  \n" \
"    return (isfinite (GB_creal (x)) && isfinite (GB_cimag (x))) ;  \n" \
"}"

#endif

