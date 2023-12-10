//------------------------------------------------------------------------------
// GB_math.c: declaring functions from GB_math.h
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#if !GB_HAS_CMPLX_MACROS
// complex constructors when the C compiler does not provide CMPLX and CMPLXF
// macros.  See Source/Shared/GB_complex.h
extern GxB_FC32_t GB_complexf (float  xreal, float  ximag) ;
extern GxB_FC64_t GB_complex  (double xreal, double ximag) ;
#endif

// integer division
extern int8_t   GB_idiv_int8   (int8_t   x, int8_t   y) ;
extern int16_t  GB_idiv_int16  (int16_t  x, int16_t  y) ;
extern int32_t  GB_idiv_int32  (int32_t  x, int32_t  y) ;
extern int64_t  GB_idiv_int64  (int64_t  x, int64_t  y) ;
extern uint8_t  GB_idiv_uint8  (uint8_t  x, uint8_t  y) ;
extern uint16_t GB_idiv_uint16 (uint16_t x, uint16_t y) ;
extern uint32_t GB_idiv_uint32 (uint32_t x, uint32_t y) ;
extern uint64_t GB_idiv_uint64 (uint64_t x, uint64_t y) ;

// complex division
extern GxB_FC32_t GB_FC32_div (GxB_FC32_t x, GxB_FC32_t y) ;
extern GxB_FC64_t GB_FC64_div (GxB_FC64_t x, GxB_FC64_t y) ;

// complex unary functions that return complex
extern GxB_FC32_t GB_cceilf  (GxB_FC32_t x) ;
extern GxB_FC64_t GB_cceil   (GxB_FC64_t x) ;
extern GxB_FC32_t GB_cfloorf (GxB_FC32_t x) ;
extern GxB_FC64_t GB_cfloor  (GxB_FC64_t x) ;
extern GxB_FC32_t GB_croundf (GxB_FC32_t x) ;
extern GxB_FC64_t GB_cround  (GxB_FC64_t x) ;
extern GxB_FC32_t GB_ctruncf (GxB_FC32_t x) ;
extern GxB_FC64_t GB_ctrunc  (GxB_FC64_t x) ;
extern GxB_FC32_t GB_cexp2f  (GxB_FC32_t x) ;
extern GxB_FC64_t GB_cexp2   (GxB_FC64_t x) ;
extern GxB_FC32_t GB_cexpm1f (GxB_FC32_t x) ;
extern GxB_FC64_t GB_cexpm1  (GxB_FC64_t x) ;
extern GxB_FC32_t GB_clog1pf (GxB_FC32_t x) ;
extern GxB_FC64_t GB_clog1p  (GxB_FC64_t x) ;
extern GxB_FC32_t GB_clog10f (GxB_FC32_t x) ;
extern GxB_FC64_t GB_clog10  (GxB_FC64_t x) ;
extern GxB_FC32_t GB_clog2f  (GxB_FC32_t x) ;
extern GxB_FC64_t GB_clog2   (GxB_FC64_t x) ;

// complex unary functions that return bool
extern bool GB_cisinff    (GxB_FC32_t x) ;
extern bool GB_cisinf     (GxB_FC64_t x) ;
extern bool GB_cisnanf    (GxB_FC32_t x) ;
extern bool GB_cisnan     (GxB_FC64_t x) ;
extern bool GB_cisfinitef (GxB_FC32_t x) ;
extern bool GB_cisfinite  (GxB_FC64_t x) ;

// z = pow (x,y) for floating-point types
extern float      GB_powf     (float      x, float      y) ;
extern double     GB_pow      (double     x, double     y) ;
extern GxB_FC32_t GB_FC32_pow (GxB_FC32_t x, GxB_FC32_t y) ;
extern GxB_FC64_t GB_FC64_pow (GxB_FC64_t x, GxB_FC64_t y) ;

// z = pow (x,y) for integers
extern int8_t   GB_pow_int8   (int8_t   x, int8_t   y) ;
extern int16_t  GB_pow_int16  (int16_t  x, int16_t  y) ;
extern int32_t  GB_pow_int32  (int32_t  x, int32_t  y) ;
extern int64_t  GB_pow_int64  (int64_t  x, int64_t  y) ;
extern uint8_t  GB_pow_uint8  (uint8_t  x, uint8_t  y) ;
extern uint16_t GB_pow_uint16 (uint16_t x, uint16_t y) ;
extern uint32_t GB_pow_uint32 (uint32_t x, uint32_t y) ;
extern uint64_t GB_pow_uint64 (uint64_t x, uint64_t y) ;

// z = frexp* (x) for float and double
extern float  GB_frexpxf (float  x) ;
extern double GB_frexpx  (double x) ;
extern float  GB_frexpef (float  x) ;
extern double GB_frexpe  (double x) ;

// z = signum (x) for floating-point types
extern float      GB_signumf  (float      x) ;
extern double     GB_signum   (double     x) ;
extern GxB_FC32_t GB_csignumf (GxB_FC32_t x) ;
extern GxB_FC64_t GB_csignum  (GxB_FC64_t x) ;

