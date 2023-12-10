//------------------------------------------------------------------------------
// GraphBLAS/Demo/Include/usercomplex.h:  complex numbers as a user-defined type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef USERCOMPLEX_H
#define USERCOMPLEX_H

 typedef struct { double re ; double im ; } mycx ;
#define MYCX_DEFN \
"typedef struct { double re ; double im ; } mycx ;"

//------------------------------------------------------------------------------
// 10 binary functions, z=f(x,y), where CxC -> C
//------------------------------------------------------------------------------

extern
GrB_BinaryOp Complex_first , Complex_second , Complex_min ,
             Complex_max   , Complex_plus   , Complex_minus ,
             Complex_times , Complex_div    , Complex_rdiv  ,
             Complex_rminus, Complex_pair ;

//------------------------------------------------------------------------------
// 6 binary comparators, z=f(x,y), where CxC -> C
//------------------------------------------------------------------------------

extern
GrB_BinaryOp Complex_iseq , Complex_isne ,
             Complex_isgt , Complex_islt ,
             Complex_isge , Complex_isle ;

//------------------------------------------------------------------------------
// 3 binary boolean functions, z=f(x,y), where CxC -> C
//------------------------------------------------------------------------------

extern
GrB_BinaryOp Complex_or , Complex_and , Complex_xor ;

//------------------------------------------------------------------------------
// 6 binary comparators, z=f(x,y), where CxC -> bool
//------------------------------------------------------------------------------

extern
GrB_BinaryOp Complex_eq , Complex_ne ,
             Complex_gt , Complex_lt ,
             Complex_ge , Complex_le ;

//------------------------------------------------------------------------------
// 1 binary function, z=f(x,y), where double x double -> C
//------------------------------------------------------------------------------

extern GrB_BinaryOp Complex_complex ;

//------------------------------------------------------------------------------
// 5 unary functions, z=f(x) where C -> C
//------------------------------------------------------------------------------

extern
GrB_UnaryOp  Complex_identity , Complex_ainv , Complex_minv ,
             Complex_not ,      Complex_conj,
             Complex_one ,      Complex_abs  ;

//------------------------------------------------------------------------------
// 4 unary functions, z=f(x) where C -> double
//------------------------------------------------------------------------------

extern
GrB_UnaryOp Complex_real, Complex_imag,
            Complex_cabs, Complex_angle ;

//------------------------------------------------------------------------------
// 2 unary functions, z=f(x) where double -> C
//------------------------------------------------------------------------------

extern GrB_UnaryOp Complex_complex_real, Complex_complex_imag ;

//------------------------------------------------------------------------------
// Complex type, scalars, monoids, and semiring
//------------------------------------------------------------------------------

extern GrB_Type Complex ;
extern GrB_Monoid   Complex_plus_monoid, Complex_times_monoid ;
extern GrB_Semiring Complex_plus_times ;

GrB_Info Complex_init (bool builtin_complex) ;
GrB_Info Complex_finalize (void) ;

//------------------------------------------------------------------------------
// function prototypes
//------------------------------------------------------------------------------

void mycx_first (mycx *z, const mycx *x, const mycx *y) ;
void mycx_second (mycx *z, const mycx *x, const mycx *y) ;
void mycx_pair (mycx *z, const mycx *x, const mycx *y) ;
void mycx_plus (mycx *z, const mycx *x, const mycx *y) ;
void mycx_minus (mycx *z, const mycx *x, const mycx *y) ;
void mycx_rminus (mycx *z, const mycx *x, const mycx *y) ;
void mycx_times (mycx *z, const mycx *x, const mycx *y) ;
void mycx_div (mycx *z, const mycx *x, const mycx *y) ;
void mycx_rdiv (mycx *z, const mycx *x, const mycx *y) ;
void fx64_min (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_min (mycx *z, const mycx *x, const mycx *y) ;
void fx64_max (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_max (mycx *z, const mycx *x, const mycx *y) ;
void mycx_iseq (mycx *z, const mycx *x, const mycx *y) ;
void mycx_isne (mycx *z, const mycx *x, const mycx *y) ;
void fx64_isgt (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_isgt (mycx *z, const mycx *x, const mycx *y) ;
void fx64_islt (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_islt (mycx *z, const mycx *x, const mycx *y) ;
void fx64_isge (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_isge (mycx *z, const mycx *x, const mycx *y) ;
void fx64_isle (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_isle (mycx *z, const mycx *x, const mycx *y) ;
void fx64_or (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_or (mycx *z, const mycx *x, const mycx *y) ;
void fx64_and (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_and (mycx *z, const mycx *x, const mycx *y) ;
void fx64_xor (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_xor (mycx *z, const mycx *x, const mycx *y) ;
void mycx_eq (bool *z, const mycx *x, const mycx *y) ;
void mycx_ne (bool *z, const mycx *x, const mycx *y) ;
void fx64_gt (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_gt (bool *z, const mycx *x, const mycx *y) ;
void fx64_lt (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_lt (bool *z, const mycx *x, const mycx *y) ;
void fx64_ge (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_ge (bool *z, const mycx *x, const mycx *y) ;
void fx64_le (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y) ;
void mycx_le (bool *z, const mycx *x, const mycx *y) ;
void mycx_cmplx (mycx *z, const double *x, const double *y) ;
void mycx_one (mycx *z, const mycx *x) ;
void mycx_identity (mycx *z, const mycx *x) ;
void mycx_ainv (mycx *z, const mycx *x) ;
void mycx_minv (mycx *z, const mycx *x) ;
void mycx_conj (mycx *z, const mycx *x) ;
void fx64_abs (GxB_FC64_t *z, const GxB_FC64_t *x) ;
void mycx_abs (mycx *z, const mycx *x) ;
void fx64_not (GxB_FC64_t *z, const GxB_FC64_t *x) ;
void mycx_not (mycx *z, const mycx *x) ;
void mycx_real (double *z, const mycx *x) ;
void mycx_imag (double *z, const mycx *x) ;
void mycx_cabs (double *z, const mycx *x) ;
void mycx_angle (double *z, const mycx *x) ;
void fx64_cmplx_real (GxB_FC64_t *z, const double *x) ;
void mycx_cmplx_real (mycx *z, const double *x) ;
void fx64_cmplx_imag (GxB_FC64_t *z, const double *x) ;
void mycx_cmplx_imag (mycx *z, const double *x) ;

//------------------------------------------------------------------------------
// C++ compatibility
//------------------------------------------------------------------------------

#if defined ( __cplusplus )

    #define crealf(x)   std::real(x)
    #define creal(x)    std::real(x)
    #define cimagf(x)   std::imag(x)
    #define cimag(x)    std::imag(x)
    #define cpowf(x,y)  std::pow(x,y)
    #define cpow(x,y)   std::pow(x,y)
    #define powf(x,y)   std::pow(x,y)
    #define cexpf(x)    std::exp(x)
    #define cexp(x)     std::exp(x)
    #define expf(x)     std::exp(x)

    #define clogf(x)    std::log(x)
    #define clog(x)     std::log(x)
    #define logf(x)     std::log(x)

    #define cabsf(x)    std::abs(x)
    #define cabs(x)     std::abs(x)
    #define absf(x)     std::abs(x)

    #define csqrtf(x)   std::sqrt(x)
    #define csqrt(x)    std::sqrt(x)
    #define sqrtf(x)    std::sqrt(x)

    #define conjf(x)    std::conj(x)

    #define cargf(x)    std::arg(x)
    #define carg(x)     std::arg(x)

    #define csinf(x)    std::sin(x)
    #define csin(x)     std::sin(x)
    #define sinf(x)     std::sin(x)
    #define ccosf(x)    std::cos(x)
    #define ccos(x)     std::cos(x)
    #define cosf(x)     std::cos(x)
    #define ctanf(x)    std::tan(x)
    #define ctan(x)     std::tan(x)
    #define tanf(x)     std::tan(x)

    #define casinf(x)   std::asin(x)
    #define casin(x)    std::asin(x)
    #define asinf(x)    std::asin(x)
    #define cacosf(x)   std::acos(x)
    #define cacos(x)    std::acos(x)
    #define acosf(x)    std::acos(x)
    #define catanf(x)   std::atan(x)
    #define catan(x)    std::atan(x)
    #define atanf(x)    std::atan(x)

    #define csinhf(x)   std::sinh(x)
    #define csinh(x)    std::sinh(x)
    #define sinhf(x)    std::sinh(x)
    #define ccoshf(x)   std::cosh(x)
    #define ccosh(x)    std::cosh(x)
    #define coshf(x)    std::cosh(x)
    #define ctanhf(x)   std::tanh(x)
    #define ctanh(x)    std::tanh(x)
    #define tanhf(x)    std::tanh(x)

    #define casinhf(x)  std::asinh(x)
    #define casinh(x)   std::asinh(x)
    #define asinhf(x)   std::asinh(x)
    #define cacoshf(x)  std::acosh(x)
    #define cacosh(x)   std::acosh(x)
    #define acoshf(x)   std::acosh(x)
    #define catanhf(x)  std::atanh(x)
    #define catanh(x)   std::atanh(x)
    #define atanhf(x)   std::atanh(x)

#endif


#endif

