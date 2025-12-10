//------------------------------------------------------------------------------
// GB_mex_edit: add/remove entries from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "C = GB_mex_edit (C, I, J, X, Action)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&C) ;              \
    GrB_Matrix_free_(&B) ;              \
    GB_mx_put_global (true) ;           \
}

#undef  OK
#define OK(method)                      \
{                                       \
    info = method ;                     \
    if (info != GrB_SUCCESS)            \
    {                                   \
        printf ("info %d line %d\n", info, __LINE__) ; \
        mexErrMsgTxt ("fail") ;         \
    }                                   \
}

#define GET_DEEP_COPY
#define FREE_DEEP_COPY ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Matrix C = NULL, B = NULL ;
    uint64_t *I = NULL, ni = 0, I_range [3] ;       // OK
    uint64_t *J = NULL, nj = 0, J_range [3] ;       // OK
    bool ignore ;
    bool malloc_debug = false ;
    GrB_Info info = GrB_SUCCESS ;
    int64_t nwork = 0 ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    malloc_debug = GB_mx_get_global (true) ;
    C = NULL ;

    // check inputs
    if (nargout > 1 || nargin != 5)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    //--------------------------------------------------------------------------
    // get C (make a deep copy)
    //--------------------------------------------------------------------------

    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }

    uint64_t ncols ;
    int fmt ;
    bool is_hyper ;
    OK (GrB_Matrix_ncols (&ncols, C)) ;
    OK (GxB_Matrix_Option_get (C, GxB_FORMAT, &fmt)) ;
    OK (GxB_Matrix_Option_get (C, GxB_IS_HYPER, &is_hyper)) ;   // historical
    bool is_vector = (fmt == GxB_BY_COL && !is_hyper && ncols == 1) ;

    // get I
    if (!GB_mx_mxArray_to_indices (pargin [1], &I, &ni, I_range, &ignore, NULL))
    {
        FREE_ALL ;
        mexErrMsgTxt ("I failed") ;
    }

    // get J
    if (!GB_mx_mxArray_to_indices (pargin [2], &J, &nj, J_range, &ignore, NULL))
    {
        FREE_ALL ;
        mexErrMsgTxt ("J failed") ;
    }

    // get X; must be double
    double *X = mxGetDoubles (pargin [3]) ;

    // get Action: must be double
    double *Action = mxGetDoubles (pargin [4]) ;

    nwork = ni ;
    if (nwork != nj ||
        nwork != mxGetNumberOfElements (pargin [3]) ||
        nwork != mxGetNumberOfElements (pargin [4]) ||
        mxGetClassID (pargin [3]) != mxDOUBLE_CLASS ||
        mxGetClassID (pargin [4]) != mxDOUBLE_CLASS)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    //--------------------------------------------------------------------------
    // turn off malloc debugging
    //--------------------------------------------------------------------------

    bool save = GB_Global_malloc_debug_get ( ) ;
    GB_Global_malloc_debug_set (false) ;
    GrB_Scalar Scalar = NULL ;
    OK (GrB_Scalar_new (&Scalar, GrB_FP64)) ;   // create an empty scalar

    //--------------------------------------------------------------------------
    // edit the matrix
    //--------------------------------------------------------------------------

    for (int64_t k = 0 ; k < nwork ; k++)
    {
        int64_t i = I [k] - 1 ;
        int64_t j = J [k] - 1 ;
        double x = X [k] ;
        double action = Action [k] ;
        if (action <= 0.2)
        {
            // remove the (i,j) entry
            if (is_vector)
            {
                OK (GrB_Vector_removeElement ((GrB_Vector) C, i)) ;
            }
            else
            {
                OK (GrB_Matrix_removeElement (C, i, j)) ;
            }
        }
        else if (action <= 0.4)
        {
            // remove the (i,j) entry using setElement_Scalar
            if (is_vector)
            {
                OK (GrB_Vector_setElement_Scalar ((GrB_Vector) C, Scalar, i)) ;
            }
            else
            {
                OK (GrB_Matrix_setElement_Scalar (C, Scalar, i, j)) ;
            }
        }
        else
        {
            // add the (i,j) entry
            if (is_vector)
            {
                OK (GrB_Vector_setElement_FP64_ ((GrB_Vector) C, x, i)) ;
            }
            else
            {
                OK (GrB_Matrix_setElement_FP64_ (C, x, i, j)) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // restore malloc debugging to test the method
    //--------------------------------------------------------------------------

    OK (GrB_Scalar_free (&Scalar)) ;
    GB_Global_malloc_debug_set (save) ;

    //--------------------------------------------------------------------------
    // duplicate the matrix and check if it stays the same
    //--------------------------------------------------------------------------

    METHOD (GrB_Matrix_dup (&B, C)) ;
    CHECK (GB_mx_isequal (B, C, 0)) ;
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (B, GrB_MATERIALIZE)) ;
    CHECK (GB_mx_isequal (B, C, 0)) ;

    //--------------------------------------------------------------------------
    // return C as a built-in sparse matrix
    //--------------------------------------------------------------------------

    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C mex_edit result", false) ;
    FREE_ALL ;
}

