//------------------------------------------------------------------------------
// GB_mex_mxm_flops: compute flops to do C=A*B, C<M>=A*B or C<!M>=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "[bflops mwork] = GB_mex_mxm_flops (M, Mask_comp, A, B)"

#define FREE_ALL                            \
{                                           \
    GrB_Matrix_free_(&A) ;                  \
    GrB_Matrix_free_(&B) ;                  \
    GrB_Matrix_free_(&M) ;                  \
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
    GrB_Matrix A = NULL ;
    GrB_Matrix B = NULL ;
    GrB_Matrix M = NULL ;

    // check inputs
    GB_WERK (USAGE) ;
    if (nargout > 2 || nargin != 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get M (shallow copy)
    M = GB_mx_mxArray_to_Matrix (pargin [0], "M", false, false) ;
    if (M == NULL && !mxIsEmpty (pargin [0]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("M failed") ;
    }

    // get Mask_comp
    bool GET_SCALAR (1, bool, Mask_comp, 0) ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [2], "A", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get B (shallow copy)
    B = GB_mx_mxArray_to_Matrix (pargin [3], "B", false, true) ;
    if (B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("B failed") ;
    }

    // allocate Bflops (note the calloc)
    int64_t bnvec = B->nvec ;
    size_t bfsize = (bnvec+1) * sizeof (int64_t) ;
    int64_t *Bflops = mxMalloc (bfsize) ;
    memset (Bflops, 0, bfsize) ;

    // compute the flop count
    int64_t Mwork = 0 ;

    GB_AxB_saxpy3_flopcount (&Mwork, Bflops, M, Mask_comp, A, B, Werk) ;

    // return result
    pargout [0] = mxCreateDoubleMatrix (1, bnvec+1, mxREAL) ;
    double *Bflops_builtin = mxGetPr (pargout [0]) ; 
    for (int64_t kk = 0 ; kk <= bnvec ; kk++)
    {
        Bflops_builtin [kk] = (double) Bflops [kk] ;
    }

    pargout [1] = mxCreateDoubleScalar (Mwork) ;
    mxFree (Bflops) ;
    FREE_ALL ;
}

