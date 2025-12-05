//------------------------------------------------------------------------------
// GB_mex_kron_idx: C = kron(A,B) with a user-defined index binary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "C = GB_mex_kron_idx (A, B, atrans, btrans, C_is_csc)"

void gb_mykronidx (double *z,
    const void *x, uint64_t ix, uint64_t jx,
    const void *y, uint64_t iy, uint64_t jy,
    const int64_t *theta) ;

void gb_mykronidx (double *z,
    const void *x, uint64_t ix, uint64_t jx,
    const void *y, uint64_t iy, uint64_t jy,
    const int64_t *theta)
{
    (*z) = (double) (
        (ix+1) * 1000000 +
        (jx+1) *   10000 +
        (iy+1) *     100 +
        (jy+1)) ;
}

#define MYKRONIDX_DEFN                              \
"void gb_mykronidx (double *z,                     \n" \
"   const void *x, uint64_t ix, uint64_t jx,    \n" \
"   const void *y, uint64_t iy, uint64_t jy,    \n" \
"   const int64_t *theta)                       \n" \
"{                                              \n" \
"   (*z) = (double) (                           \n" \
"       (ix+1) * 1000000 +                      \n" \
"       (jx+1) *   10000 +                      \n" \
"       (iy+1) *     100 +                      \n" \
"       (jy+1)) ;                               \n" \
"}"

#define FREE_ALL                    \
{                                   \
    GrB_Matrix_free_(&A) ;          \
    GrB_Matrix_free_(&B) ;          \
    GrB_Matrix_free_(&C) ;          \
    GrB_Scalar_free_(&Theta) ;      \
    GrB_BinaryOp_free (&mult) ;     \
    GxB_IndexBinaryOp_free (&Iop) ; \
    GrB_Descriptor_free (&desc) ;   \
    GB_mx_put_global (true) ;       \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info = GrB_SUCCESS ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    GrB_Matrix B = NULL ;
    GrB_Matrix C = NULL ;
    GrB_Scalar Theta = NULL ;
    GrB_BinaryOp mult = NULL ;
    GxB_IndexBinaryOp Iop = NULL ;
    GrB_Descriptor desc = NULL ;
    uint64_t anrows = 0, ancols = 0, bnrows = 0, bncols = 0 ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 5)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get B (shallow copy)
    B = GB_mx_mxArray_to_Matrix (pargin [1], "B input", false, true) ;
    if (B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("B failed") ;
    }

    // get the atrans option
    bool GET_SCALAR (2, bool, atrans, false) ;

    // get the btrans option
    bool GET_SCALAR (3, bool, btrans, false) ;

    // get the C_is_csc option
    bool GET_SCALAR (4, bool, C_is_csc, true) ;

    // set the Descriptor
    OK (GrB_Descriptor_new (&desc)) ;
    OK (GrB_Descriptor_set (desc, GrB_INP0, atrans ? GrB_TRAN : GxB_DEFAULT)) ;
    OK (GrB_Descriptor_set (desc, GrB_INP1, btrans ? GrB_TRAN : GxB_DEFAULT)) ;

    // determine the dimensions
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nrows (&bnrows, B)) ;
    OK (GrB_Matrix_ncols (&bncols, B)) ;
    uint64_t cnrows = ((atrans) ? ancols : anrows)
                    * ((btrans) ? bncols : bnrows) ;
    uint64_t cncols = ((atrans) ? anrows : ancols)
                    * ((btrans) ? bnrows : bncols) ;

    // create the output matrix C
    OK (GrB_Matrix_new (&C, GrB_FP64, cnrows, cncols)) ;
    OK (GrB_Matrix_set_INT32 (C, C_is_csc, GrB_STORAGE_ORIENTATION_HINT)) ;

    // create the index binary op
    OK (GxB_IndexBinaryOp_new (&Iop,
        (GxB_index_binary_function) gb_mykronidx,
        GrB_FP64, GrB_FP64, GrB_FP64, GrB_INT64,
        "gb_mykronidx", MYKRONIDX_DEFN)) ;

    // create the mult binary op
    int64_t theta = 0 ;
    OK (GrB_Scalar_new (&Theta, GrB_INT64)) ;
    OK (GrB_Scalar_setElement_INT64 (Theta, theta)) ;
    OK (GxB_BinaryOp_new_IndexOp (&mult, Iop, Theta)) ;

    // C = kron(A,B)
    METHOD (GrB_Matrix_kronecker_BinaryOp_ (C, NULL, NULL, mult, A, B, desc)) ;

    // return C as a MATLAB sparse matrix and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", false) ;

    FREE_ALL ;
}

