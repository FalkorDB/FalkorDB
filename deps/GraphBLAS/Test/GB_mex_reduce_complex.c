//------------------------------------------------------------------------------
// GB_mex_mxm: C<Mask> = accum(C,A*B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// c = prod (A) with terminal times monoid (terminal value is 0)

#include "GB_mex.h"

#define USAGE "c = GB_mex_reduce_complex (A, mangle)"

#define FREE_ALL                            \
{                                           \
    GrB_Matrix_free_(&A) ;                  \
    GrB_Monoid_free_(&Times_terminal) ;     \
    GB_mx_put_global (true) ;               \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Info info ;
    GrB_Matrix A = NULL ;
    GrB_Monoid Times_terminal = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A (shallow copy; must be complex)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    if (A->type != Complex)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A must be complex") ;
    }

    if (A->iso)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A must not be iso") ;
    }

    GxB_FC64_t one  = GxB_CMPLX (1,0) ;
    GxB_FC64_t zero = GxB_CMPLX (0,0) ;

    // create the monoid
    if (Complex == GxB_FC64)
    {
        Times_terminal = GxB_TIMES_FC64_MONOID ;
    }
    else
    {
        info = GxB_Monoid_terminal_new_UDT (&Times_terminal,
            Complex_times, &one, &zero) ;
        if (info != GrB_SUCCESS)
        {
            FREE_ALL ;
            mexErrMsgTxt ("Times_terminal failed") ;
        }
    }

    int64_t GET_SCALAR (1, int64_t, mangle, -1) ;
    if (mangle >= 0)
    {
        GxB_FC64_t *Ax = A->x ;         // OK: A is non iso
        Ax [mangle] = GxB_CMPLX (0,0) ;
    }

    // allocate the output scalar
    pargout [0] = GB_mx_create_full (1, 1, GxB_FC64) ;
    GxB_FC64_t *c = (GxB_FC64_t *) mxGetComplexDoubles (pargout [0]) ;

    // reduce to a scalar
    if (Complex == GxB_FC64)
    {
        info = GxB_Matrix_reduce_FC64_(c, NULL, Times_terminal, A, NULL) ;
    }
    else
    {
        info = GrB_Matrix_reduce_UDT (c, NULL, Times_terminal, A, NULL) ;
    }
    if (info != GrB_SUCCESS)
    {
        FREE_ALL ;
        mexErrMsgTxt ("reduce failed") ;
    }

    FREE_ALL ;
}

