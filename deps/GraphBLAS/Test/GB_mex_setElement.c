//------------------------------------------------------------------------------
// GB_mex_setElement: interface for A(i,j) = x
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// x = A (i,j), where i and j are zero-based.  If i and j arrays, then
// x (k) = A (i (k), j (k)) is done for all k.

// I and J and zero-based

#include "GB_mex.h"

#define USAGE "A = GB_mex_setElement (A, I, J, X, debug_wait,scalar)"

bool debug_wait = false ;
bool do_scalar = false ;
GrB_Type xtype = NULL ;

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GB_mx_put_global (true) ;           \
}

#if defined ( __GNUC__ )
#pragma GCC diagnostic ignored "-Wmissing-prototypes"
#endif

#define Return(info) { GrB_Scalar_free (&Scalar) ; return (info) ; }

//------------------------------------------------------------------------------
// set all elements of a matrix and return if an error is encountered
//------------------------------------------------------------------------------

#define setEl(prefix,name,type)                                             \
GrB_Info set_ ## name                                                       \
(GrB_Matrix A, type *X, GrB_Index *I, GrB_Index *J, GrB_Index ni)           \
{                                                                           \
    GrB_Info info ;                                                         \
    GrB_Scalar Scalar = NULL ;                                              \
    if (do_scalar)                                                          \
    {                                                                       \
        info = GrB_Scalar_new (&Scalar, xtype) ;                            \
        if (info != GrB_SUCCESS)                                            \
        {                                                                   \
            Return (info) ;                                                 \
        }                                                                   \
    }                                                                       \
    for (int64_t k = 0 ; k < ni ; k++)                                      \
    {                                                                       \
        if (do_scalar)                                                      \
        {                                                                   \
            info = prefix ## Scalar_setElement_ ## name                     \
                (Scalar, AMPERSAND (X [k])) ;                               \
            if (info != GrB_SUCCESS)                                        \
            {                                                               \
                Return (info) ;                                             \
            }                                                               \
            info = GrB_Matrix_setElement_Scalar                             \
                (A, Scalar, I [k], J [k]) ;                                 \
        }                                                                   \
        else                                                                \
        {                                                                   \
            info = prefix ## Matrix_setElement_ ## name                     \
                (A, AMPERSAND (X [k]), I [k], J [k]) ;                      \
        }                                                                   \
        if (info != GrB_SUCCESS)                                            \
        {                                                                   \
            Return (info) ;                                                 \
        }                                                                   \
    }                                                                       \
    if (debug_wait)                                                         \
    {                                                                       \
        Return (GB_wait (A, "A", NULL)) ;                                   \
    }                                                                       \
    Return (GrB_SUCCESS) ;                                                  \
}

//------------------------------------------------------------------------------
// create all the local set_TYPE functions
//------------------------------------------------------------------------------

#define AMPERSAND(x) x
setEl (GrB_, BOOL   , bool          ) ;
setEl (GrB_, INT8   , int8_t        ) ;
setEl (GrB_, UINT8  , uint8_t       ) ;
setEl (GrB_, INT16  , int16_t       ) ;
setEl (GrB_, UINT16 , uint16_t      ) ;
setEl (GrB_, INT32  , int32_t       ) ;
setEl (GrB_, UINT32 , uint32_t      ) ;
setEl (GrB_, INT64  , int64_t       ) ;
setEl (GrB_, UINT64 , uint64_t      ) ;
setEl (GrB_, FP32   , float         ) ;
setEl (GrB_, FP64   , double        ) ;
setEl (GxB_, FC32   , GxB_FC32_t    ) ;
setEl (GxB_, FC64   , GxB_FC64_t    ) ;
#undef  AMPERSAND
#define AMPERSAND(x) &x
setEl (GrB_, UDT    , GxB_FC64_t) ;
#undef  AMPERSAND

//------------------------------------------------------------------------------
// set all elements of a vector and return if an error is encountered
//------------------------------------------------------------------------------

#define vsetEl(prefix,name,type)                                            \
GrB_Info vset_ ## name                                                      \
(GrB_Matrix A, type *X, GrB_Index *I, GrB_Index ni)                         \
{                                                                           \
    GrB_Info info ;                                                         \
    GrB_Scalar Scalar = NULL ;                                              \
    if (do_scalar)                                                          \
    {                                                                       \
        info = GrB_Scalar_new (&Scalar, xtype) ;                            \
        if (info != GrB_SUCCESS) Return (info) ;                            \
    }                                                                       \
    GrB_Vector w = (GrB_Vector) A ;                                         \
    for (int64_t k = 0 ; k < ni ; k++)                                      \
    {                                                                       \
        if (do_scalar)                                                      \
        {                                                                   \
            info = prefix ## Scalar_setElement_ ## name                     \
                (Scalar, AMPERSAND (X [k])) ;                               \
            if (info != GrB_SUCCESS) Return (info) ;                        \
            info = GrB_Vector_setElement_Scalar                             \
                (w, Scalar, I [k]) ;                                        \
        }                                                                   \
        else                                                                \
        {                                                                   \
            info = prefix ## Vector_setElement_ ## name                     \
                (w, AMPERSAND (X [k]), I [k]) ;                             \
        }                                                                   \
        if (info != GrB_SUCCESS) Return (info) ;                            \
    }                                                                       \
    if (debug_wait)                                                         \
    {                                                                       \
        Return (GB_wait (A, "A", NULL)) ;                                   \
    }                                                                       \
    Return (GrB_SUCCESS) ;                                                  \
}

//------------------------------------------------------------------------------
// create all the local set_TYPE functions
//------------------------------------------------------------------------------

#define AMPERSAND(x) x
vsetEl (GrB_, BOOL   , bool          ) ;
vsetEl (GrB_, INT8   , int8_t        ) ;
vsetEl (GrB_, UINT8  , uint8_t       ) ;
vsetEl (GrB_, INT16  , int16_t       ) ;
vsetEl (GrB_, UINT16 , uint16_t      ) ;
vsetEl (GrB_, INT32  , int32_t       ) ;
vsetEl (GrB_, UINT32 , uint32_t      ) ;
vsetEl (GrB_, INT64  , int64_t       ) ;
vsetEl (GrB_, UINT64 , uint64_t      ) ;
vsetEl (GrB_, FP32   , float         ) ;
vsetEl (GrB_, FP64   , double        ) ;
vsetEl (GxB_, FC32   , GxB_FC32_t    ) ;
vsetEl (GxB_, FC64   , GxB_FC64_t    ) ;
#undef  AMPERSAND
#define AMPERSAND(x) &x
vsetEl (GrB_, UDT    , GxB_FC64_t) ;
#undef  AMPERSAND

//------------------------------------------------------------------------------
// GB_mex_setElement
//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;

    GrB_Matrix A = NULL ;
    GB_void *Y ;
    GrB_Index *I = NULL, ni = 0, I_range [3] ;
    GrB_Index *J = NULL, nj = 0, J_range [3] ;
    bool is_list ;

    // check inputs
    if (nargout > 1 || nargin < 4 || nargin > 6)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A (deep copy)
    #define GET_DEEP_COPY \
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", true, true) ;
    #define FREE_DEEP_COPY GrB_Matrix_free_(&A) ;
    GET_DEEP_COPY ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get I
    if (!GB_mx_mxArray_to_indices (&I, pargin [1], &ni, I_range, &is_list))
    {
        FREE_ALL ;
        mexErrMsgTxt ("I failed") ;
    }
    if (!is_list)
    {
        mexErrMsgTxt ("I is invalid; must be a list") ;
    }

    // get J
    if (!GB_mx_mxArray_to_indices (&J, pargin [2], &nj, J_range, &is_list))
    {
        FREE_ALL ;
        mexErrMsgTxt ("J failed") ;
    }
    if (!is_list)
    {
        mexErrMsgTxt ("J is invalid; must be a list") ;
    }

    if (ni != nj)
    {
        FREE_ALL ;
        mexErrMsgTxt ("I and J must be the same size") ;
    }

    // get X
    if (ni != mxGetNumberOfElements (pargin [3]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("I and X must be the same size") ;
    }
    if (!(mxIsNumeric (pargin [3]) || mxIsLogical (pargin [3])))
    {
        FREE_ALL ;
        mexErrMsgTxt ("X must be a numeric or logical array") ;
    }
    if (mxIsSparse (pargin [3]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("X cannot be sparse") ;
    }

    // get debug_wait (if true, to GB_wait after setElements)
    GET_SCALAR (4, bool, debug_wait, false) ;

    // get do_scalar (if true use GrB_*_setElement_Scalar)
    GET_SCALAR (5, bool, do_scalar, false) ;

    if (mxIsComplex (pargin [3]))
    {
        xtype = Complex ;
        Y = (GB_void *) mxGetComplexDoubles (pargin [3]) ;
    }
    else
    {
        Y = mxGetData (pargin [3]) ;
        xtype = GB_mx_Type (pargin [3]) ;
        if (xtype == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("X must be numeric") ;
        }
    }

    size_t s = 2 * sizeof (double) ;

    // A (i,j) = x, for a list of elements

    // the METHOD (...) macro is not used on each call to setElement, but
    // to all of them.  Thus, if any failure occurs, the computation is rolled
    // back to the very beginning, and another fresh, deep, copy of A is made,
    // and the sequence of setElements is tried again.  If a setElement fails
    // by running out of memory, it clears to whole matrix, so recovery cannot
    // be made.

    if (A->vdim == 1)
    {
        // test GrB_Vector_setElement
        switch (xtype->code)
        {
            case GB_BOOL_code   : METHOD (vset_BOOL   (A, (bool       *) Y, I, ni)) ; break ;
            case GB_INT8_code   : METHOD (vset_INT8   (A, (int8_t     *) Y, I, ni)) ; break ;
            case GB_INT16_code  : METHOD (vset_INT16  (A, (int16_t    *) Y, I, ni)) ; break ;
            case GB_INT32_code  : METHOD (vset_INT32  (A, (int32_t    *) Y, I, ni)) ; break ;
            case GB_INT64_code  : METHOD (vset_INT64  (A, (int64_t    *) Y, I, ni)) ; break ;
            case GB_UINT8_code  : METHOD (vset_UINT8  (A, (uint8_t    *) Y, I, ni)) ; break ;
            case GB_UINT16_code : METHOD (vset_UINT16 (A, (uint16_t   *) Y, I, ni)) ; break ;
            case GB_UINT32_code : METHOD (vset_UINT32 (A, (uint32_t   *) Y, I, ni)) ; break ;
            case GB_UINT64_code : METHOD (vset_UINT64 (A, (uint64_t   *) Y, I, ni)) ; break ;
            case GB_FP32_code   : METHOD (vset_FP32   (A, (float      *) Y, I, ni)) ; break ;
            case GB_FP64_code   : METHOD (vset_FP64   (A, (double     *) Y, I, ni)) ; break ;
            case GB_FC32_code   : METHOD (vset_FC32   (A, (GxB_FC32_t *) Y, I, ni)) ; break ;
            case GB_FC64_code   : METHOD (vset_FC64   (A, (GxB_FC64_t *) Y, I, ni)) ; break ;
            case GB_UDT_code    : METHOD (vset_UDT    (A, (void       *) Y, I, ni)) ; break ;
            default:
                FREE_ALL ;
                mexErrMsgTxt ("unsupported type") ;
        }
    }
    else
    {
        // test GrB_Matrix_setElement
        switch (xtype->code)
        {
            case GB_BOOL_code   : METHOD (set_BOOL   (A, (bool       *) Y, I, J, ni)) ; break ;
            case GB_INT8_code   : METHOD (set_INT8   (A, (int8_t     *) Y, I, J, ni)) ; break ;
            case GB_INT16_code  : METHOD (set_INT16  (A, (int16_t    *) Y, I, J, ni)) ; break ;
            case GB_INT32_code  : METHOD (set_INT32  (A, (int32_t    *) Y, I, J, ni)) ; break ;
            case GB_INT64_code  : METHOD (set_INT64  (A, (int64_t    *) Y, I, J, ni)) ; break ;
            case GB_UINT8_code  : METHOD (set_UINT8  (A, (uint8_t    *) Y, I, J, ni)) ; break ;
            case GB_UINT16_code : METHOD (set_UINT16 (A, (uint16_t   *) Y, I, J, ni)) ; break ;
            case GB_UINT32_code : METHOD (set_UINT32 (A, (uint32_t   *) Y, I, J, ni)) ; break ;
            case GB_UINT64_code : METHOD (set_UINT64 (A, (uint64_t   *) Y, I, J, ni)) ; break ;
            case GB_FP32_code   : METHOD (set_FP32   (A, (float      *) Y, I, J, ni)) ; break ;
            case GB_FP64_code   : METHOD (set_FP64   (A, (double     *) Y, I, J, ni)) ; break ;
            case GB_FC32_code   : METHOD (set_FC32   (A, (GxB_FC32_t *) Y, I, J, ni)) ; break ;
            case GB_FC64_code   : METHOD (set_FC64   (A, (GxB_FC64_t *) Y, I, J, ni)) ; break ;
            case GB_UDT_code    : METHOD (set_UDT    (A, (void       *) Y, I, J, ni)) ; break ;
            default:
                FREE_ALL ;
                mexErrMsgTxt ("unsupported type") ;
        }
    }

    // only do debug checks after adding lots of tuples
    if (ni > 1000) { ASSERT_MATRIX_OK (A, "A added pending tuples", GB0) ; }

    // return A as a struct and free the GraphBLAS A
    pargout [0] = GB_mx_Matrix_to_mxArray (&A, "A output", true) ;

    FREE_ALL ;
}

