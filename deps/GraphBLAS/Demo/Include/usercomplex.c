//------------------------------------------------------------------------------
// GraphBLAS/Demo/Include/usercomplex.c:  complex numbers as a user-defined type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All functions must work if any input/outputs x, y, and/or z are aliased.
// Some methods (times, div, and rdiv) require temporary variables zre and zim
// as a result.  All other functions are OK as-is.

#ifdef MATLAB_MEX_FILE

    #include "GB_mex.h"

    #define OK(method)                                                      \
    {                                                                       \
        info = method ;                                                     \
        if (! (info == GrB_SUCCESS || info == GrB_NO_VALUE))                \
        {                                                                   \
            return (info) ;                                                 \
        }                                                                   \
    }

#else

    #include "GraphBLAS.h"
    #include "graphblas_demos.h"

    #if defined __INTEL_COMPILER
    #pragma warning (disable: 58 167 144 161 177 181 186 188 589 593 869 981 \
        1418 1419 1572 1599 2259 2282 2557 2547 3280 )
    #elif defined __GNUC__
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    #if !defined ( __cplusplus )
    #pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
    #endif
    #endif

    #undef OK
    #define OK(method)                                                      \
    {                                                                       \
        info = method ;                                                     \
        if (! (info == GrB_SUCCESS || info == GrB_NO_VALUE))                \
        {                                                                   \
            Complex_finalize ( ) ;                                          \
            return (info) ;                                                 \
        }                                                                   \
    }

#endif

GrB_BinaryOp Complex_first = NULL, Complex_second = NULL, Complex_min = NULL,
             Complex_max   = NULL, Complex_plus   = NULL, Complex_minus = NULL,
             Complex_times = NULL, Complex_div    = NULL, Complex_rminus = NULL,
             Complex_rdiv  = NULL, Complex_pair   = NULL ;

GrB_BinaryOp Complex_iseq = NULL, Complex_isne = NULL,
             Complex_isgt = NULL, Complex_islt = NULL,
             Complex_isge = NULL, Complex_isle = NULL ;

GrB_BinaryOp Complex_or = NULL, Complex_and = NULL, Complex_xor = NULL ;

GrB_BinaryOp Complex_eq = NULL, Complex_ne = NULL,
             Complex_gt = NULL, Complex_lt = NULL,
             Complex_ge = NULL, Complex_le = NULL ;

GrB_BinaryOp Complex_complex = NULL ;

GrB_UnaryOp  Complex_identity = NULL, Complex_ainv = NULL, Complex_minv = NULL,
             Complex_not = NULL,      Complex_conj = NULL,
             Complex_one = NULL,      Complex_abs  = NULL ;

GrB_UnaryOp Complex_real = NULL, Complex_imag = NULL,
            Complex_cabs = NULL, Complex_angle = NULL ;

GrB_UnaryOp Complex_complex_real = NULL, Complex_complex_imag = NULL ;

GrB_Type Complex = NULL ;
GrB_Monoid   Complex_plus_monoid = NULL, Complex_times_monoid = NULL ;
GrB_Semiring Complex_plus_times = NULL ;

//------------------------------------------------------------------------------
// binary functions, z=f(x,y), where CxC -> Complex
//------------------------------------------------------------------------------

void mycx_first (mycx *z, const mycx *x, const mycx *y)
{
    z->re = x->re ;
    z->im = x->im ;
}

#define MYCX_FIRST                                                          \
"void mycx_first (mycx *z, const mycx *x, const mycx *y)                \n" \
"{                                                                      \n" \
"    z->re = x->re ;                                                    \n" \
"    z->im = x->im ;                                                    \n" \
"}"

void mycx_second (mycx *z, const mycx *x, const mycx *y)
{
    z->re = y->re ;
    z->im = y->im ;
}

#define MYCX_SECOND                                                         \
"void mycx_second (mycx *z, const mycx *x, const mycx *y)               \n" \
"{                                                                      \n" \
"    z->re = y->re ;                                                    \n" \
"    z->im = y->im ;                                                    \n" \
"}"

void mycx_pair (mycx *z, const mycx *x, const mycx *y)
{
    z->re = 1 ;
    z->im = 0 ;
}

#define MYCX_PAIR                                                           \
"void mycx_pair (mycx *z, const mycx *x, const mycx *y)                 \n" \
"{                                                                      \n" \
"    z->re = 1 ;                                                        \n" \
"    z->im = 0 ;                                                        \n" \
"}"

void mycx_plus (mycx *z, const mycx *x, const mycx *y)
{
    z->re = x->re + y->re ;
    z->im = x->im + y->im ;
}

#define MYCX_PLUS                                                           \
"void mycx_plus (mycx *z, const mycx *x, const mycx *y)                 \n" \
"{                                                                      \n" \
"    z->re = x->re + y->re ;                                            \n" \
"    z->im = x->im + y->im ;                                            \n" \
"}"

void mycx_minus (mycx *z, const mycx *x, const mycx *y)
{
    z->re = x->re - y->re ;
    z->im = x->im - y->im ;
}

#define MYCX_MINUS                                                          \
"void mycx_minus (mycx *z, const mycx *x, const mycx *y)                \n" \
"{                                                                      \n" \
"    z->re = x->re - y->re ;                                            \n" \
"    z->im = x->im - y->im ;                                            \n" \
"}"

void mycx_rminus (mycx *z, const mycx *x, const mycx *y)
{
    z->re = y->re - x->re ;
    z->im = y->im - x->im ;
}

#define MYCX_RMINUS                                                         \
"void mycx_rminus (mycx *z, const mycx *x, const mycx *y)               \n" \
"{                                                                      \n" \
"    z->re = y->re - x->re ;                                            \n" \
"    z->im = y->im - x->im ;                                            \n" \
"}"

// temporary variables zre and zim are required for times, div, and rdiv:
void mycx_times (mycx *z, const mycx *x, const mycx *y)
{
    double zre = (x->re * y->re) - (x->im * y->im) ;
    double zim = (x->re * y->im) + (x->im * y->re) ;
    z->re = zre ;
    z->im = zim ;
}

#define MYCX_TIMES                                                          \
"void mycx_times (mycx *z, const mycx *x, const mycx *y)                \n" \
"{                                                                      \n" \
"    double zre = (x->re * y->re) - (x->im * y->im) ;                   \n" \
"    double zim = (x->re * y->im) + (x->im * y->re) ;                   \n" \
"    z->re = zre ;                                                      \n" \
"    z->im = zim ;                                                      \n" \
"}"

void mycx_div (mycx *z, const mycx *x, const mycx *y)
{
    double den = (y->re * y->re) + (y->im * y->im) ;
    double zre = ((x->re * y->re) + (x->im * y->im)) / den ;
    double zim = ((x->im * y->re) - (x->re * y->im)) / den ;
    z->re = zre ;
    z->im = zim ;
}

#define MYCX_DIV                                                            \
"void mycx_div (mycx *z, const mycx *x, const mycx *y)                  \n" \
"{                                                                      \n" \
"    double den = (y->re * y->re) + (y->im * y->im) ;                   \n" \
"    double zre = ((x->re * y->re) + (x->im * y->im)) / den ;           \n" \
"    double zim = ((x->im * y->re) - (x->re * y->im)) / den ;           \n" \
"    z->re = zre ;                                                      \n" \
"    z->im = zim ;                                                      \n" \
"}"

void mycx_rdiv (mycx *z, const mycx *x, const mycx *y)
{
    double den = (x->re * x->re) + (x->im * x->im) ;
    double zre = ((y->re * x->re) + (y->im * x->im)) / den ;
    double zim = ((y->im * x->re) - (y->re * x->im)) / den ;
    z->re = zre ;
    z->im = zim ;
}

#define MYCX_RDIV                                                           \
"void mycx_div (mycx *z, const mycx *x, const mycx *y)                  \n" \
"{                                                                      \n" \
"    double den = (x->re * x->re) + (x->im * x->im) ;                   \n" \
"    double zre = ((y->re * x->re) + (y->im * x->im)) / den ;           \n" \
"    double zim = ((y->im * x->re) - (y->re * x->im)) / den ;           \n" \
"    z->re = zre ;                                                      \n" \
"    z->im = zim ;                                                      \n" \
"}"

// min (x,y): complex number with smallest magnitude.  If tied, select the
// one with the smallest phase angle.  No special cases for NaNs.

void fx64_min (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    double absx = cabs (*x) ;
    double absy = cabs (*y) ;
    if (absx < absy)
    {
        (*z) = (*x) ;
    }
    else if (absx > absy)
    {
        (*z) = (*y) ;
    }
    else
    {
        (*z) = (carg (*x) < carg (*y)) ? (*x) : (*y) ;
    }
}

#define FX64_MIN                                                            \
"void fx64_min (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)\n" \
"{                                                                      \n" \
"    double absx = cabs (*x) ;                                          \n" \
"    double absy = cabs (*y) ;                                          \n" \
"    if (absx < absy)                                                   \n" \
"    {                                                                  \n" \
"        (*z) = (*x) ;                                                  \n" \
"    }                                                                  \n" \
"    else if (absx > absy)                                              \n" \
"    {                                                                  \n" \
"        (*z) = (*y) ;                                                  \n" \
"    }                                                                  \n" \
"    else                                                               \n" \
"    {                                                                  \n" \
"        (*z) = (carg (*x) < carg (*y)) ? (*x) : (*y) ;                 \n" \
"    }                                                                  \n" \
"}"

void mycx_min (mycx *z, const mycx *x, const mycx *y)
{
    GxB_FC64_t X = GxB_CMPLX (x->re, x->im) ;
    GxB_FC64_t Y = GxB_CMPLX (y->re, y->im) ;
    double absx = cabs (X) ;
    double absy = cabs (Y) ;
    if (absx < absy)
    {
        z->re = x->re ;
        z->im = x->im ;
    }
    else if (absx > absy)
    {
        z->re = y->re ;
        z->im = y->im ;
    }
    else
    {
        if (carg (X) < carg (Y))
        {
            z->re = x->re ;
            z->im = x->im ;
        }
        else
        {
            z->re = y->re ;
            z->im = y->im ;
        }
    }
}

#define MYCX_MIN                                                            \
"void mycx_min (mycx *z, const mycx *x, const mycx *y)                  \n" \
"{                                                                      \n" \
"    GxB_FC64_t X = GxB_CMPLX (x->re, x->im) ;                          \n" \
"    GxB_FC64_t Y = GxB_CMPLX (y->re, y->im) ;                          \n" \
"    double absx = cabs (X) ;                                           \n" \
"    double absy = cabs (Y) ;                                           \n" \
"    if (absx < absy)                                                   \n" \
"    {                                                                  \n" \
"        z->re = x->re ;                                                \n" \
"        z->im = x->im ;                                                \n" \
"    }                                                                  \n" \
"    else if (absx > absy)                                              \n" \
"    {                                                                  \n" \
"        z->re = y->re ;                                                \n" \
"        z->im = y->im ;                                                \n" \
"    }                                                                  \n" \
"    else                                                               \n" \
"    {                                                                  \n" \
"        if (carg (X) < carg (Y))                                       \n" \
"        {                                                              \n" \
"            z->re = x->re ;                                            \n" \
"            z->im = x->im ;                                            \n" \
"        }                                                              \n" \
"        else                                                           \n" \
"        {                                                              \n" \
"            z->re = y->re ;                                            \n" \
"            z->im = y->im ;                                            \n" \
"        }                                                              \n" \
"    }                                                                  \n" \
"}"

// max (x,y): complex number with largest magnitude.  If tied, select the one
// with the largest phase angle.  No special cases for NaNs.

void fx64_max (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    double absx = cabs (*x) ;
    double absy = cabs (*y) ;
    if (absx > absy)
    {
        (*z) = (*x) ;
    }
    else if (absx < absy)
    {
        (*z) = (*y) ;
    }
    else
    {
        (*z) = (carg (*x) > carg (*y)) ? (*x) : (*y) ;
    }
}

#define FX64_MAX                                                            \
"void fx64_max (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)\n" \
"{                                                                      \n" \
"    double absx = cabs (*x) ;                                          \n" \
"    double absy = cabs (*y) ;                                          \n" \
"    if (absx > absy)                                                   \n" \
"    {                                                                  \n" \
"        (*z) = (*x) ;                                                  \n" \
"    }                                                                  \n" \
"    else if (absx < absy)                                              \n" \
"    {                                                                  \n" \
"        (*z) = (*y) ;                                                  \n" \
"    }                                                                  \n" \
"    else                                                               \n" \
"    {                                                                  \n" \
"        (*z) = (carg (*x) > carg (*y)) ? (*x) : (*y) ;                 \n" \
"    }                                                                  \n" \
"}"

void mycx_max (mycx *z, const mycx *x, const mycx *y)
{
    GxB_FC64_t X = GxB_CMPLX (x->re, x->im) ;
    GxB_FC64_t Y = GxB_CMPLX (y->re, y->im) ;
    double absx = cabs (X) ;
    double absy = cabs (Y) ;
    if (absx > absy)
    {
        z->re = x->re ;
        z->im = x->im ;
    }
    else if (absx < absy)
    {
        z->re = y->re ;
        z->im = y->im ;
    }
    else
    {
        if (carg (X) > carg (Y))
        {
            z->re = x->re ;
            z->im = x->im ;
        }
        else
        {
            z->re = y->re ;
            z->im = y->im ;
        }
    }
}

#define MYCX_MAX                                                            \
"void mycx_max (mycx *z, const mycx *x, const mycx *y)                  \n" \
"{                                                                      \n" \
"    GxB_FC64_t X = GxB_CMPLX (x->re, x->im) ;                          \n" \
"    GxB_FC64_t Y = GxB_CMPLX (y->re, y->im) ;                          \n" \
"    double absx = cabs (X) ;                                           \n" \
"    double absy = cabs (Y) ;                                           \n" \
"    if (absx > absy)                                                   \n" \
"    {                                                                  \n" \
"        z->re = x->re ;                                                \n" \
"        z->im = x->im ;                                                \n" \
"    }                                                                  \n" \
"    else if (absx < absy)                                              \n" \
"    {                                                                  \n" \
"        z->re = y->re ;                                                \n" \
"        z->im = y->im ;                                                \n" \
"    }                                                                  \n" \
"    else                                                               \n" \
"    {                                                                  \n" \
"        if (carg (X) > carg (Y))                                       \n" \
"        {                                                              \n" \
"            z->re = x->re ;                                            \n" \
"            z->im = x->im ;                                            \n" \
"        }                                                              \n" \
"        else                                                           \n" \
"        {                                                              \n" \
"            z->re = y->re ;                                            \n" \
"            z->im = y->im ;                                            \n" \
"        }                                                              \n" \
"    }                                                                  \n" \
"}"

//------------------------------------------------------------------------------
// 6 binary functions, z=f(x,y); CxC -> Complex ; (1,0) = true, (0,0) = false
//------------------------------------------------------------------------------

void mycx_iseq (mycx *z, const mycx *x, const mycx *y)
{
    z->re = (x->re == y->re && x->im == y->im) ? 1 : 0 ;
    z->im = 0 ;
}

#define MYCX_ISEQ                                                           \
"void mycx_iseq (mycx *z, const mycx *x, const mycx *y)                 \n" \
"{                                                                      \n" \
"    z->re = (x->re == y->re && x->im == y->im) ? 1 : 0 ;               \n" \
"    z->im = 0 ;                                                        \n" \
"}"

void mycx_isne (mycx *z, const mycx *x, const mycx *y)
{
    z->re = (x->re != y->re || x->im != y->im) ? 1 : 0 ;
    z->im = 0 ;
}

#define MYCX_ISNE                                                           \
"void mycx_isne (mycx *z, const mycx *x, const mycx *y)                 \n" \
"{                                                                      \n" \
"    z->re = (x->re != y->re || x->im != y->im) ? 1 : 0 ;               \n" \
"    z->im = 0 ;                                                        \n" \
"}"

void fx64_isgt (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    bool gt = (creal (*x) > creal (*y)) ;
    (*z) = gt ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;
}

#define FX64_ISGT                                                           \
"void fx64_isgt (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)\n"\
"{                                                                      \n" \
"    bool gt = (creal (*x) > creal (*y)) ;                              \n" \
"    (*z) = gt ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;                    \n" \
"}"

void mycx_isgt (mycx *z, const mycx *x, const mycx *y)
{
    z->re = (x->re > y->re) ? 1 : 0 ;
    z->im = 0 ;
}

#define MYCX_ISGT                                                           \
"void mycx_isgt (mycx *z, const mycx *x, const mycx *y)                 \n" \
"{                                                                      \n" \
"    z->re = (x->re > y->re) ? 1 : 0 ;                                  \n" \
"    z->im = 0 ;                                                        \n" \
"}"

void fx64_islt (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    bool lt = (creal (*x) < creal (*y)) ;
    (*z) = lt ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;
}

#define FX64_ISLT                                                           \
"void fx64_islt (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)\n"\
"{                                                                      \n" \
"    bool lt = (creal (*x) < creal (*y)) ;                              \n" \
"    (*z) = lt ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;                    \n" \
"}"

void mycx_islt (mycx *z, const mycx *x, const mycx *y)
{
    z->re = (x->re < y->re) ? 1 : 0 ;
    z->im = 0 ;
}

#define MYCX_ISLT                                                           \
"void mycx_islt (mycx *z, const mycx *x, const mycx *y)                 \n" \
"{                                                                      \n" \
"    z->re = (x->re < y->re) ? 1 : 0 ;                                  \n" \
"    z->im = 0 ;                                                        \n" \
"}"

void fx64_isge (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    bool ge = (creal (*x) >= creal (*y)) ;
    (*z) = ge ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;
}

#define FX64_ISGE                                                           \
"void fx64_isge (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)\n"\
"{                                                                      \n" \
"    bool ge = (creal (*x) >= creal (*y)) ;                             \n" \
"    (*z) = ge ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;                    \n" \
"}"

void mycx_isge (mycx *z, const mycx *x, const mycx *y)
{
    z->re = (x->re >= y->re) ? 1 : 0 ;
    z->im = 0 ;
}

#define MYCX_ISGE                                                           \
"void mycx_isge (mycx *z, const mycx *x, const mycx *y)                 \n" \
"{                                                                      \n" \
"    z->re = (x->re >= y->re) ? 1 : 0 ;                                 \n" \
"    z->im = 0 ;                                                        \n" \
"}"

void fx64_isle (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    bool le = (creal (*x) <= creal (*y)) ;
    (*z) = le ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;
}

#define FX64_ISLE                                                           \
"void fx64_isle (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)\n"\
"{                                                                      \n" \
"    bool le = (creal (*x) <= creal (*y)) ;                             \n" \
"    (*z) = le ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;                    \n" \
"}"

void mycx_isle (mycx *z, const mycx *x, const mycx *y)
{
    z->re = (x->re <= y->re) ? 1 : 0 ;
    z->im = 0 ;
}

#define MYCX_ISLE                                                           \
"void mycx_isle (mycx *z, const mycx *x, const mycx *y)                 \n" \
"{                                                                      \n" \
"    z->re = (x->re <= y->re) ? 1 : 0 ;                                 \n" \
"    z->im = 0 ;                                                        \n" \
"}"

//------------------------------------------------------------------------------
// binary boolean functions, z=f(x,y), where CxC -> Complex
//------------------------------------------------------------------------------

void fx64_or (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    bool xbool = (creal (*x) != 0 || cimag (*x) != 0) ;
    bool ybool = (creal (*y) != 0 || cimag (*y) != 0) ;
    (*z) = (xbool || ybool) ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;
}

#define FX64_OR                                                             \
"void fx64_or (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y) \n" \
"{                                                                      \n" \
"    bool xbool = (creal (*x) != 0 || cimag (*x) != 0) ;                \n" \
"    bool ybool = (creal (*y) != 0 || cimag (*y) != 0) ;                \n" \
"    (*z) = (xbool || ybool) ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;      \n" \
"}"

void mycx_or (mycx *z, const mycx *x, const mycx *y)
{
    bool xbool = (x->re != 0 || x->im != 0) ;
    bool ybool = (y->re != 0 || y->im != 0) ;
    z->re = (xbool || ybool) ? 1 : 0 ;
    z->im = 0 ;
}

#define MYCX_OR                                                             \
"void mycx_or (mycx *z, const mycx *x, const mycx *y)                   \n" \
"{                                                                      \n" \
"    bool xbool = (x->re != 0 || x->im != 0) ;                          \n" \
"    bool ybool = (y->re != 0 || y->im != 0) ;                          \n" \
"    z->re = (xbool || ybool) ? 1 : 0 ;                                 \n" \
"    z->im = 0 ;                                                        \n" \
"}"

void fx64_and (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    bool xbool = (creal (*x) != 0 || cimag (*x) != 0) ;
    bool ybool = (creal (*y) != 0 || cimag (*y) != 0) ;
    (*z) = (xbool && ybool) ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;
}

#define FX64_AND                                                            \
"void fx64_and (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)\n" \
"{                                                                      \n" \
"    bool xbool = (creal (*x) != 0 || cimag (*x) != 0) ;                \n" \
"    bool ybool = (creal (*y) != 0 || cimag (*y) != 0) ;                \n" \
"    (*z) = (xbool && ybool) ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;      \n" \
"}"

void mycx_and (mycx *z, const mycx *x, const mycx *y)
{
    bool xbool = (x->re != 0 || x->im != 0) ;
    bool ybool = (y->re != 0 || y->im != 0) ;
    z->re = (xbool && ybool) ? 1 : 0 ;
    z->im = 0 ;
}

#define MYCX_AND                                                            \
"void mycx_and (mycx *z, const mycx *x, const mycx *y)                  \n" \
"{                                                                      \n" \
"    bool xbool = (x->re != 0 || x->im != 0) ;                          \n" \
"    bool ybool = (y->re != 0 || y->im != 0) ;                          \n" \
"    z->re = (xbool && ybool) ? 1 : 0 ;                                 \n" \
"    z->im = 0 ;                                                        \n" \
"}"

void fx64_xor (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    bool xbool = (creal (*x) != 0 || cimag (*x) != 0) ;
    bool ybool = (creal (*y) != 0 || cimag (*y) != 0) ;
    (*z) = (xbool != ybool) ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;
}

#define FX64_XOR                                                            \
"void fx64_xor (GxB_FC64_t *z, const GxB_FC64_t *x, const GxB_FC64_t *y)\n" \
"{                                                                      \n" \
"    bool xbool = (creal (*x) != 0 || cimag (*x) != 0) ;                \n" \
"    bool ybool = (creal (*y) != 0 || cimag (*y) != 0) ;                \n" \
"    (*z) = (xbool != ybool) ? GxB_CMPLX (1,0) : GxB_CMPLX (0,0) ;      \n" \
"}"

void mycx_xor (mycx *z, const mycx *x, const mycx *y)
{
    bool xbool = (x->re != 0 || x->im != 0) ;
    bool ybool = (y->re != 0 || y->im != 0) ;
    z->re = (xbool && ybool) ? 1 : 0 ;
    z->im = 0 ;
}

#define MYCX_XOR                                                            \
"void mycx_xor (mycx *z, const mycx *x, const mycx *y)                  \n" \
"{                                                                      \n" \
"    bool xbool = (x->re != 0 || x->im != 0) ;                          \n" \
"    bool ybool = (y->re != 0 || y->im != 0) ;                          \n" \
"    z->re = (xbool != ybool) ? 1 : 0 ;                                 \n" \
"    z->im = 0 ;                                                        \n" \
"}"

//------------------------------------------------------------------------------
// 6 binary functions, z=f(x,y), where CxC -> bool
//------------------------------------------------------------------------------

void mycx_eq (bool *z, const mycx *x, const mycx *y)
{
    (*z) = (x->re == y->re && x->im == y->im) ;
}

#define MYCX_EQ                                                             \
"void mycx_eq (bool *z, const mycx *x, const mycx *y)                   \n" \
"{                                                                      \n" \
"    (*z) = (x->re == y->re && x->im == y->im) ;                        \n" \
"}"

void mycx_ne (bool *z, const mycx *x, const mycx *y)
{
    (*z) = (x->re == y->re && x->im == y->im) ;
}

#define MYCX_NE                                                             \
"void mycx_ne (bool *z, const mycx *x, const mycx *y)                   \n" \
"{                                                                      \n" \
"    (*z) = (x->re == y->re && x->im == y->im) ;                        \n" \
"}"

void fx64_gt (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    (*z) = (creal (*x) > creal (*y)) ;
}

#define FX64_GT                                                             \
"void fx64_gt (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y)       \n" \
"{                                                                      \n" \
"    (*z) = (creal (*x) > creal (*y)) ;                                 \n" \
"}"

void mycx_gt (bool *z, const mycx *x, const mycx *y)
{
    (*z) = (x->re > y->re) ;
}

#define MYCX_GT                                                             \
"void mycx_gt (bool *z, const mycx *x, const mycx *y)                   \n" \
"{                                                                      \n" \
"    (*z) = (x->re > y->re) ;                                           \n" \
"}"

void fx64_lt (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    (*z) = (creal (*x) > creal (*y)) ;
}

#define FX64_LT                                                             \
"void fx64_lt (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y)       \n" \
"{                                                                      \n" \
"    (*z) = (creal (*x) < creal (*y)) ;                                 \n" \
"}"

void mycx_lt (bool *z, const mycx *x, const mycx *y)
{
    (*z) = (x->re < y->re) ;
}

#define MYCX_LT                                                             \
"void mycx_lt (bool *z, const mycx *x, const mycx *y)                   \n" \
"{                                                                      \n" \
"    (*z) = (x->re < y->re) ;                                           \n" \
"}"

void fx64_ge (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    (*z) = (creal (*x) >= creal (*y)) ;
}

#define FX64_GE                                                             \
"void fx64_ge (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y)       \n" \
"{                                                                      \n" \
"    (*z) = (creal (*x) >= creal (*y)) ;                                \n" \
"}"

void mycx_ge (bool *z, const mycx *x, const mycx *y)
{
    (*z) = (x->re >= y->re) ;
}

#define MYCX_GE                                                             \
"void mycx_ge (bool *z, const mycx *x, const mycx *y)                   \n" \
"{                                                                      \n" \
"    (*z) = (x->re >= y->re) ;                                          \n" \
"}"

void fx64_le (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y)
{
    (*z) = (creal (*x) >= creal (*y)) ;
}

#define FX64_LE                                                             \
"void fx64_lt (bool *z, const GxB_FC64_t *x, const GxB_FC64_t *y)       \n" \
"{                                                                      \n" \
"    (*z) = (creal (*x) <= creal (*y)) ;                                \n" \
"}"

void mycx_le (bool *z, const mycx *x, const mycx *y)
{
    (*z) = (x->re <= y->re) ;
}

#define MYCX_LE                                                             \
"void mycx_le (bool *z, const mycx *x, const mycx *y)                   \n" \
"{                                                                      \n" \
"    (*z) = (x->re <= y->re) ;                                          \n" \
"}"


//------------------------------------------------------------------------------
// binary functions, z=f(x,y), where double x double -> complex
//------------------------------------------------------------------------------

void mycx_cmplx (mycx *z, const double *x, const double *y)
{
    z->re = (*x) ;
    z->im = (*y) ;
}

#define MYCX_CMPLX                                                          \
"void mycx_cmplx (mycx *z, const double *x, const double *y)            \n" \
"{                                                                      \n" \
"    z->re = (*x) ;                                                     \n" \
"    z->im = (*y) ;                                                     \n" \
"}"

//------------------------------------------------------------------------------
// unary functions, z=f(x) where Complex -> Complex
//------------------------------------------------------------------------------

void mycx_one (mycx *z, const mycx *x)
{
    z->re = 1 ;
    z->im = 0 ;
}

#define MYCX_ONE                                                            \
"void mycx_one (mycx *z, const mycx *x)                                 \n" \
"{                                                                      \n" \
"    z->re = 1 ;                                                        \n" \
"    z->im = 0 ;                                                        \n" \
"}"

void mycx_identity (mycx *z, const mycx *x)
{
    z->re = x->re ;
    z->im = x->im ;
}

#define MYCX_IDENTITY                                                       \
"void mycx_identity (mycx *z, const mycx *x)                            \n" \
"{                                                                      \n" \
"    z->re = x->re ;                                                    \n" \
"    z->im = x->im ;                                                    \n" \
"}"

void mycx_ainv (mycx *z, const mycx *x)
{
    z->re = -(x->re) ;
    z->im = -(x->im) ;
}

#define MYCX_AINV                                                           \
"void mycx_ainv (mycx *z, const mycx *x)                                \n" \
"{                                                                      \n" \
"    z->re = -(x->re) ;                                                 \n" \
"    z->im = -(x->im) ;                                                 \n" \
"}"

void mycx_minv (mycx *z, const mycx *x)
{
    double den = (x->re * x->re) + (x->im * x->im) ;
    z->re =  (x->re) / den ;
    z->im = -(x->im) / den ;
}

#define MYCX_MINV                                                           \
"void mycx_minv (mycx *z, const mycx *x)                                \n" \
"{                                                                      \n" \
"    double den = (x->re * x->re) + (x->im * x->im) ;                   \n" \
"    z->re =  (x->re) / den ;                                           \n" \
"    z->im = -(x->im) / den ;                                           \n" \
"}"

void mycx_conj (mycx *z, const mycx *x)
{
    z->re =  (x->re) ;
    z->im = -(x->im) ;
}

#define MYCX_CONJ                                                           \
"void mycx_conj (mycx *z, const mycx *x)                                \n" \
"{                                                                      \n" \
"    z->re =  (x->re) ;                                                 \n" \
"    z->im = -(x->im) ;                                                 \n" \
"}"

void fx64_abs (GxB_FC64_t *z, const GxB_FC64_t *x)
{
    (*z) = GxB_CMPLX (cabs (*x), 0) ;
}

#define FX64_ABS                                                            \
"void fx64_abs (GxB_FC64_t *z, const GxB_FC64_t *x)                     \n" \
"{                                                                      \n" \
"    (*z) = GxB_CMPLX (cabs (*x), 0) ;                                  \n" \
"}"

void mycx_abs (mycx *z, const mycx *x)
{
    z->re = sqrt ((x->re * x->re) + (x->im * x->im)) ;
    z->im = 0 ;
}

#define MYCX_ABS                                                            \
"void mycx_abs (mycx *z, const mycx *x)                                 \n" \
"{                                                                      \n" \
"    z->re = sqrt ((x->re * x->re) + (x->im * x->im)) ;                 \n" \
"    z->im = 0 ;                                                        \n" \
"}"

void fx64_not (GxB_FC64_t *z, const GxB_FC64_t *x)
{
    bool xbool = (creal (*x) != 0 || cimag (*x) != 0) ;
    (*z) = xbool ? GxB_CMPLX (0,0) : GxB_CMPLX (1,0) ;
}

#define FX64_NOT                                                            \
"void fx64_not (GxB_FC64_t *z, const GxB_FC64_t *x)                     \n" \
"{                                                                      \n" \
"   bool xbool = (creal (*x) != 0 || cimag (*x) != 0) ;                 \n" \
"   (*z) = xbool ? GxB_CMPLX (0,0) : GxB_CMPLX (1,0) ;                  \n" \
"}"

void mycx_not (mycx *z, const mycx *x)
{
    z->re = (x->re != 0 || x->im != 0) ? 1 : 0 ;
    z->im = 0 ;
}

#define MYCX_NOT                                                            \
"void mycx_not (mycx *z, const mycx *x)                                 \n" \
"{                                                                      \n" \
"    z->re = (x->re != 0 || x->im != 0) ? 1 : 0 ;                       \n" \
"    z->im = 0 ;                                                        \n" \
"}"

//------------------------------------------------------------------------------
// unary functions, z=f(x) where Complex -> double
//------------------------------------------------------------------------------

void mycx_real (double *z, const mycx *x)
{
    (*z) = x->re ;
}

#define MYCX_REAL                                                           \
"void mycx_real (double *z, const mycx *x)                              \n" \
"{                                                                      \n" \
"    (*z) = x->re ;                                                     \n" \
"}"

void mycx_imag (double *z, const mycx *x)
{
    (*z) = x->im ;
}

#define MYCX_IMAG                                                           \
"void mycx_imag (double *z, const mycx *x)                              \n" \
"{                                                                      \n" \
"    (*z) = x->im ;                                                     \n" \
"}"

void mycx_cabs (double *z, const mycx *x)
{
    (*z) = sqrt ((x->re * x->re) + (x->im * x->im)) ;
}

#define MYCX_CABS                                                           \
"void mycx_cabs (double *z, const mycx *x)                              \n" \
"{                                                                      \n" \
"    (*z) = sqrt ((x->re * x->re) + (x->im * x->im)) ;                  \n" \
"}"

void mycx_angle (double *z, const mycx *x)
{
    GxB_FC64_t X = GxB_CMPLX (x->re, x->im) ;
    (*z) = carg (X) ;
}

#define MYCX_ANGLE                                                          \
"void mycx_angle (double *z, const mycx *x)                             \n" \
"{                                                                      \n" \
"    GxB_FC64_t X = GxB_CMPLX (x->re, x->im) ;                          \n" \
"    (*z) = carg (X) ;                                                  \n" \
"}"

//------------------------------------------------------------------------------
// unary functions, z=f(x) where double -> Complex
//------------------------------------------------------------------------------

void fx64_cmplx_real (GxB_FC64_t *z, const double *x)
{
    (*z) = GxB_CMPLX ((*x), 0) ;
}

#define FX64_CMPLX_REAL                                                     \
"void fx64_cmplx_real (GxB_FC64_t *z, const double *x)                  \n" \
"{                                                                      \n" \
"    (*z) = GxB_CMPLX ((*x), 0) ;                                       \n" \
"}"

void mycx_cmplx_real (mycx *z, const double *x)
{
    z->re = (*x) ;
    z->im = 0 ;
}

#define MYCX_CMPLX_REAL                                                     \
"void mycx_cmplx_real (mycx *z, const double *x)                        \n" \
"{                                                                      \n" \
"    z->re = (*x) ;                                                     \n" \
"    z->im = 0 ;                                                        \n" \
"}"

void fx64_cmplx_imag (GxB_FC64_t *z, const double *x)
{
    (*z) = GxB_CMPLX (0, (*x)) ;
}

#define FX64_CMPLX_IMAG                                                     \
"void fx64_cmplx_imag (GxB_FC64_t *z, const double *x)                  \n" \
"{                                                                      \n" \
"    (*z) = GxB_CMPLX (0, (*x)) ;                                       \n" \
"}"

void mycx_cmplx_imag (mycx *z, const double *x)
{
    z->re = 0 ;
    z->im = (*x) ;
}

#define MYCX_CMPLX_IMAG                                                     \
"void mycx_cmplx_imag (mycx *z, const double *x)                        \n" \
"{                                                                      \n" \
"    z->re = 0 ;                                                        \n" \
"    z->im = (*x) ;                                                     \n" \
"}"

//------------------------------------------------------------------------------
// Complex_init: create the complex type, operators, monoids, and semiring
//------------------------------------------------------------------------------

#define U (GxB_unary_function)
#define B (GxB_binary_function)

GrB_Info Complex_init (bool builtin_complex)
{

    GrB_Info info ;

    //--------------------------------------------------------------------------
    // create the Complex type, or set to GxB_FC64
    //--------------------------------------------------------------------------

    if (builtin_complex)
    {
        // use the built-in type
        Complex = GxB_FC64 ;
    }
    else
    {
        // create the user-defined type
        // Normally, the typename should be "GxB_FC64_t",
        // but the C type GxB_FC64_t is already defined.
        OK (GxB_Type_new (&Complex, sizeof (GxB_FC64_t), "mycx", MYCX_DEFN)) ;
    }

    //--------------------------------------------------------------------------
    // create the Complex binary operators, CxC->Complex
    //--------------------------------------------------------------------------

    if (builtin_complex)
    {
        // use the built-in versions
        Complex_first  = GxB_FIRST_FC64 ;
        Complex_second = GxB_SECOND_FC64 ;
        Complex_pair   = GxB_PAIR_FC64 ;
        Complex_plus   = GxB_PLUS_FC64 ;
        Complex_minus  = GxB_MINUS_FC64 ;
        Complex_rminus = GxB_RMINUS_FC64 ;
        Complex_times  = GxB_TIMES_FC64 ;
        Complex_div    = GxB_DIV_FC64 ;
        Complex_rdiv   = GxB_RDIV_FC64 ;
    }
    else
    {
        // create user-defined versions
        OK (GxB_BinaryOp_new (&Complex_first  , B mycx_first  , Complex, Complex, Complex, "mycx_first" , MYCX_FIRST)) ;
        OK (GxB_BinaryOp_new (&Complex_second , B mycx_second , Complex, Complex, Complex, "mycx_second", MYCX_SECOND)) ;
        OK (GxB_BinaryOp_new (&Complex_pair   , B mycx_pair   , Complex, Complex, Complex, "mycx_pair"  , MYCX_PAIR)) ;
        OK (GxB_BinaryOp_new (&Complex_plus   , B mycx_plus   , Complex, Complex, Complex, "mycx_plus"  , MYCX_PLUS)) ;
        OK (GxB_BinaryOp_new (&Complex_minus  , B mycx_minus  , Complex, Complex, Complex, "mycx_minus" , MYCX_MINUS)) ;
        OK (GxB_BinaryOp_new (&Complex_rminus , B mycx_rminus , Complex, Complex, Complex, "mycx_rminus", MYCX_RMINUS)) ;
        OK (GxB_BinaryOp_new (&Complex_times  , B mycx_times  , Complex, Complex, Complex, "mycx_times" , MYCX_TIMES)) ;
        OK (GxB_BinaryOp_new (&Complex_div    , B mycx_div    , Complex, Complex, Complex, "mycx_div"   , MYCX_DIV)) ;
        OK (GxB_BinaryOp_new (&Complex_rdiv   , B mycx_rdiv   , Complex, Complex, Complex, "mycx_rdiv"  , MYCX_RDIV)) ;
    }

    // these are not built-in
    if (builtin_complex)
    {
        OK (GxB_BinaryOp_new (&Complex_min    , B fx64_min    , Complex, Complex, Complex, "fx64_min"   , FX64_MIN)) ;
        OK (GxB_BinaryOp_new (&Complex_max    , B fx64_max    , Complex, Complex, Complex, "fx64_max"   , FX64_MAX)) ;
    }
    else
    {
        OK (GxB_BinaryOp_new (&Complex_min    , B mycx_min    , Complex, Complex, Complex, "mycx_min"   , MYCX_MIN)) ;
        OK (GxB_BinaryOp_new (&Complex_max    , B mycx_max    , Complex, Complex, Complex, "mycx_max"   , MYCX_MAX)) ;
    }

    //--------------------------------------------------------------------------
    // create the Complex binary comparators, CxC -> Complex
    //--------------------------------------------------------------------------

    if (builtin_complex)
    {
        // use the built-in versions
        Complex_iseq = GxB_ISEQ_FC64 ;
        Complex_isne = GxB_ISNE_FC64 ;
    }
    else
    {
        // create user-defined versions
        OK (GxB_BinaryOp_new (&Complex_iseq   , B mycx_iseq   , Complex, Complex, Complex, "mycx_iseq"  , MYCX_ISEQ)) ;
        OK (GxB_BinaryOp_new (&Complex_isne   , B mycx_isne   , Complex, Complex, Complex, "mycx_isne"  , MYCX_ISNE)) ;
    }

    // these are not built-in
    if (builtin_complex)
    {
        OK (GxB_BinaryOp_new (&Complex_isgt   , B fx64_isgt   , Complex, Complex, Complex, "fx64_isgt"  , FX64_ISGT)) ;
        OK (GxB_BinaryOp_new (&Complex_islt   , B fx64_islt   , Complex, Complex, Complex, "fx64_islt"  , FX64_ISLT)) ;
        OK (GxB_BinaryOp_new (&Complex_isge   , B fx64_isge   , Complex, Complex, Complex, "fx64_isge"  , FX64_ISGE)) ;
        OK (GxB_BinaryOp_new (&Complex_isle   , B fx64_isle   , Complex, Complex, Complex, "fx64_isle"  , FX64_ISLE)) ;
    }
    else
    {
        OK (GxB_BinaryOp_new (&Complex_isgt   , B mycx_isgt   , Complex, Complex, Complex, "mycx_isgt"  , MYCX_ISGT)) ;
        OK (GxB_BinaryOp_new (&Complex_islt   , B mycx_islt   , Complex, Complex, Complex, "mycx_islt"  , MYCX_ISLT)) ;
        OK (GxB_BinaryOp_new (&Complex_isge   , B mycx_isge   , Complex, Complex, Complex, "mycx_isge"  , MYCX_ISGE)) ;
        OK (GxB_BinaryOp_new (&Complex_isle   , B mycx_isle   , Complex, Complex, Complex, "mycx_isle"  , MYCX_ISLE)) ;
    }

    //--------------------------------------------------------------------------
    // create the Complex boolean operators, CxC -> Complex
    //--------------------------------------------------------------------------

    // these are not built-in
    if (builtin_complex)
    {
        OK (GxB_BinaryOp_new (&Complex_or     , B fx64_or     , Complex, Complex, Complex, "fx64_or"    , FX64_OR)) ;
        OK (GxB_BinaryOp_new (&Complex_and    , B fx64_and    , Complex, Complex, Complex, "fx64_and"   , FX64_AND)) ;
        OK (GxB_BinaryOp_new (&Complex_xor    , B fx64_xor    , Complex, Complex, Complex, "fx64_xor"   , FX64_XOR)) ;
    }
    else
    {
        OK (GxB_BinaryOp_new (&Complex_or     , B mycx_or     , Complex, Complex, Complex, "mycx_or"    , MYCX_OR)) ;
        OK (GxB_BinaryOp_new (&Complex_and    , B mycx_and    , Complex, Complex, Complex, "mycx_and"   , MYCX_AND)) ;
        OK (GxB_BinaryOp_new (&Complex_xor    , B mycx_xor    , Complex, Complex, Complex, "mycx_xor"   , MYCX_XOR)) ;
    }

    //--------------------------------------------------------------------------
    // create the Complex binary operators, CxC -> bool
    //--------------------------------------------------------------------------

    if (builtin_complex)
    {
        // use the built-in versions
        Complex_eq = GxB_EQ_FC64 ;
        Complex_ne = GxB_NE_FC64 ;
    }
    else
    {
        // create user-defined versions
        OK (GxB_BinaryOp_new (&Complex_eq     , B mycx_eq     ,GrB_BOOL, Complex, Complex, "mycx_eq"    , MYCX_EQ)) ;
        OK (GxB_BinaryOp_new (&Complex_ne     , B mycx_ne     ,GrB_BOOL, Complex, Complex, "mycx_ne"    , MYCX_NE)) ;
    }

    // these are not built-in
    if (builtin_complex)
    {
        OK (GxB_BinaryOp_new (&Complex_gt     , B fx64_gt     ,GrB_BOOL, Complex, Complex, "fx64_gt"    , FX64_GT)) ;
        OK (GxB_BinaryOp_new (&Complex_lt     , B fx64_lt     ,GrB_BOOL, Complex, Complex, "fx64_lt"    , FX64_LT)) ;
        OK (GxB_BinaryOp_new (&Complex_ge     , B fx64_ge     ,GrB_BOOL, Complex, Complex, "fx64_ge"    , FX64_GE)) ;
        OK (GxB_BinaryOp_new (&Complex_le     , B fx64_le     ,GrB_BOOL, Complex, Complex, "fx64_le"    , FX64_LE)) ;
    }
    else
    {
        OK (GxB_BinaryOp_new (&Complex_gt     , B mycx_gt     ,GrB_BOOL, Complex, Complex, "mycx_gt"    , MYCX_GT)) ;
        OK (GxB_BinaryOp_new (&Complex_lt     , B mycx_lt     ,GrB_BOOL, Complex, Complex, "mycx_lt"    , MYCX_LT)) ;
        OK (GxB_BinaryOp_new (&Complex_ge     , B mycx_ge     ,GrB_BOOL, Complex, Complex, "mycx_ge"    , MYCX_GE)) ;
        OK (GxB_BinaryOp_new (&Complex_le     , B mycx_le     ,GrB_BOOL, Complex, Complex, "mycx_le"    , MYCX_LE)) ;
    }

    //--------------------------------------------------------------------------
    // create the Complex binary operator, double x double -> Complex
    //--------------------------------------------------------------------------

    if (builtin_complex)
    {
        // use the built-in versions
        Complex_complex = GxB_CMPLX_FP64 ;
    }
    else
    {
        // create user-defined versions
        OK (GxB_BinaryOp_new (&Complex_complex, B mycx_cmplx  , Complex,GrB_FP64,GrB_FP64, "mycx_cmplx" ,MYCX_CMPLX)) ;
    }

    //--------------------------------------------------------------------------
    // create the Complex unary operators, Complex->Complex
    //--------------------------------------------------------------------------

    if (builtin_complex)
    {
        // use the built-in versions
        Complex_one      = GxB_ONE_FC64 ;
        Complex_identity = GxB_IDENTITY_FC64 ;
        Complex_ainv     = GxB_AINV_FC64 ;
        Complex_minv     = GxB_MINV_FC64 ;
        Complex_conj     = GxB_CONJ_FC64 ;
    }
    else
    {
        // create user-defined versions
        OK (GxB_UnaryOp_new (&Complex_one     , U mycx_one     , Complex, Complex, "mycx_one"     , MYCX_ONE)) ;
        OK (GxB_UnaryOp_new (&Complex_identity, U mycx_identity, Complex, Complex, "mycx_identity", MYCX_IDENTITY)) ;
        OK (GxB_UnaryOp_new (&Complex_ainv    , U mycx_ainv    , Complex, Complex, "mycx_ainv"    , MYCX_AINV)) ;
        OK (GxB_UnaryOp_new (&Complex_minv    , U mycx_minv    , Complex, Complex, "mycx_minv"    , MYCX_MINV)) ;
        OK (GxB_UnaryOp_new (&Complex_conj    , U mycx_conj    , Complex, Complex, "mycx_conj"    , MYCX_CONJ)) ;
    }

    // these are not built-in
    if (builtin_complex)
    {
        OK (GxB_UnaryOp_new (&Complex_abs     , U fx64_abs     , Complex, Complex, "fx64_abs"     , FX64_ABS)) ;
        OK (GxB_UnaryOp_new (&Complex_not     , U fx64_not     , Complex, Complex, "fx64_not"     , FX64_NOT)) ;
    }
    else
    {
        OK (GxB_UnaryOp_new (&Complex_abs     , U mycx_abs     , Complex, Complex, "mycx_abs"     , MYCX_ABS)) ;
        OK (GxB_UnaryOp_new (&Complex_not     , U mycx_not     , Complex, Complex, "mycx_not"     , MYCX_NOT)) ;
    }

    //--------------------------------------------------------------------------
    // create the unary functions, Complex -> double
    //--------------------------------------------------------------------------

    if (builtin_complex)
    {
        // use the built-in versions
        Complex_real  = GxB_CREAL_FC64 ;
        Complex_imag  = GxB_CIMAG_FC64 ;
        Complex_cabs  = GxB_ABS_FC64 ;
        Complex_angle = GxB_CARG_FC64 ;
    }
    else
    {
        // create user-defined versions
        OK (GxB_UnaryOp_new (&Complex_real    , U mycx_real    ,GrB_FP64, Complex, "mycx_real"    , MYCX_REAL)) ;
        OK (GxB_UnaryOp_new (&Complex_imag    , U mycx_imag    ,GrB_FP64, Complex, "mycx_imag"    , MYCX_IMAG)) ;
        OK (GxB_UnaryOp_new (&Complex_cabs    , U mycx_cabs    ,GrB_FP64, Complex, "mycx_cabs"    , MYCX_CABS)) ;
        OK (GxB_UnaryOp_new (&Complex_angle   , U mycx_angle   ,GrB_FP64, Complex, "mycx_angle"   , MYCX_ANGLE)) ;
    }

    //--------------------------------------------------------------------------
    // create the unary functions, double -> Complex
    //--------------------------------------------------------------------------

    // these are not built-in
    if (builtin_complex)
    {
        OK (GxB_UnaryOp_new (&Complex_complex_real, U fx64_cmplx_real, Complex, GrB_FP64, "fx64_cmplx_real", FX64_CMPLX_REAL)) ;
        OK (GxB_UnaryOp_new (&Complex_complex_imag, U fx64_cmplx_imag, Complex, GrB_FP64, "fx64_cmplx_imag", FX64_CMPLX_IMAG)) ;
    }
    else
    {
        OK (GxB_UnaryOp_new (&Complex_complex_real, U mycx_cmplx_real, Complex, GrB_FP64, "mycx_cmplx_real", MYCX_CMPLX_REAL)) ;
        OK (GxB_UnaryOp_new (&Complex_complex_imag, U mycx_cmplx_imag, Complex, GrB_FP64, "mycx_cmplx_imag", MYCX_CMPLX_IMAG)) ;
    }

    //--------------------------------------------------------------------------
    // create the Complex monoids
    //--------------------------------------------------------------------------

    if (builtin_complex)
    {
        // use the built-in versions
        Complex_plus_monoid  = GxB_PLUS_FC64_MONOID ;
        Complex_times_monoid = GxB_TIMES_FC64_MONOID ;
    }
    else
    {
        // create user-defined versions
        mycx one, zero ;
        one.re = 1 ;
        one.im = 0 ;
        zero.re = 0 ;
        zero.im = 0 ;
        OK (GrB_Monoid_new_UDT (&Complex_plus_monoid,  Complex_plus,  (void *) &zero)) ;
        OK (GrB_Monoid_new_UDT (&Complex_times_monoid, Complex_times, (void *) &one)) ;
    }

    //----------------------------------------------------------------------
    // create the Complex plus-times semiring
    //----------------------------------------------------------------------

    if (builtin_complex)
    {
        // use the built-in versions
        Complex_plus_times = GxB_PLUS_TIMES_FC64 ;
    }
    else
    {
        // more could be created, but this suffices for testing GraphBLAS
        OK (GrB_Semiring_new (&Complex_plus_times, Complex_plus_monoid, Complex_times)) ;
    }

    return (GrB_SUCCESS) ;
}


//------------------------------------------------------------------------------
// Complex_finalize: free all complex types, operators, monoids, and semiring
//------------------------------------------------------------------------------

// These may be built-in types and operators.  They are safe to free; the
// GrB_*_free functions silently do nothing if asked to free bulit-in objects.

GrB_Info Complex_finalize ( )
{

    //--------------------------------------------------------------------------
    // free the Complex plus-times semiring
    //--------------------------------------------------------------------------

    GrB_Semiring_free (&Complex_plus_times) ;

    //--------------------------------------------------------------------------
    // free the Complex monoids
    //--------------------------------------------------------------------------

    GrB_Monoid_free (&Complex_plus_monoid ) ;
    GrB_Monoid_free (&Complex_times_monoid) ;

    //--------------------------------------------------------------------------
    // free the Complex binary operators, CxC->complex
    //--------------------------------------------------------------------------

    GrB_BinaryOp_free (&Complex_first ) ;
    GrB_BinaryOp_free (&Complex_second) ;
    GrB_BinaryOp_free (&Complex_pair  ) ;
    GrB_BinaryOp_free (&Complex_min   ) ;
    GrB_BinaryOp_free (&Complex_max   ) ;
    GrB_BinaryOp_free (&Complex_plus  ) ;
    GrB_BinaryOp_free (&Complex_minus ) ;
    GrB_BinaryOp_free (&Complex_rminus) ;
    GrB_BinaryOp_free (&Complex_times ) ;
    GrB_BinaryOp_free (&Complex_div   ) ;
    GrB_BinaryOp_free (&Complex_rdiv  ) ;

    GrB_BinaryOp_free (&Complex_iseq) ;
    GrB_BinaryOp_free (&Complex_isne) ;
    GrB_BinaryOp_free (&Complex_isgt) ;
    GrB_BinaryOp_free (&Complex_islt) ;
    GrB_BinaryOp_free (&Complex_isge) ;
    GrB_BinaryOp_free (&Complex_isle) ;

    GrB_BinaryOp_free (&Complex_or) ;
    GrB_BinaryOp_free (&Complex_and) ;
    GrB_BinaryOp_free (&Complex_xor) ;

    //--------------------------------------------------------------------------
    // free the Complex binary operators, CxC -> bool
    //--------------------------------------------------------------------------

    GrB_BinaryOp_free (&Complex_eq) ;
    GrB_BinaryOp_free (&Complex_ne) ;
    GrB_BinaryOp_free (&Complex_gt) ;
    GrB_BinaryOp_free (&Complex_lt) ;
    GrB_BinaryOp_free (&Complex_ge) ;
    GrB_BinaryOp_free (&Complex_le) ;

    //--------------------------------------------------------------------------
    // free the Complex binary operator, double x double -> complex
    //--------------------------------------------------------------------------

    GrB_BinaryOp_free (&Complex_complex) ;

    //--------------------------------------------------------------------------
    // free the Complex unary operators, complex->complex
    //--------------------------------------------------------------------------

    GrB_UnaryOp_free (&Complex_one     ) ;
    GrB_UnaryOp_free (&Complex_identity) ;
    GrB_UnaryOp_free (&Complex_ainv    ) ;
    GrB_UnaryOp_free (&Complex_abs     ) ;
    GrB_UnaryOp_free (&Complex_minv    ) ;
    GrB_UnaryOp_free (&Complex_not     ) ;
    GrB_UnaryOp_free (&Complex_conj    ) ;

    //--------------------------------------------------------------------------
    // free the unary functions, complex -> double
    //--------------------------------------------------------------------------

    GrB_UnaryOp_free (&Complex_real ) ;
    GrB_UnaryOp_free (&Complex_imag ) ;
    GrB_UnaryOp_free (&Complex_cabs ) ;
    GrB_UnaryOp_free (&Complex_angle) ;

    //--------------------------------------------------------------------------
    // free the unary functions, double -> complex
    //--------------------------------------------------------------------------

    GrB_UnaryOp_free (&Complex_complex_real) ;
    GrB_UnaryOp_free (&Complex_complex_imag) ;

    //--------------------------------------------------------------------------
    // free the Complex type
    //--------------------------------------------------------------------------

    GrB_Type_free (&Complex) ;

    return (GrB_SUCCESS) ;
}

#undef U
#undef B

