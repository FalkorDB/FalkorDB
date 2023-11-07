//------------------------------------------------------------------------------
// GB_mex_assign_alias_mask: C<A> = A
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_assign_alias_mask (C, A, desc)"

#define FREE_ALL                            \
{                                           \
    GrB_Matrix_free_(&C) ;                  \
    GrB_Matrix_free_(&A) ;                  \
    GrB_Descriptor_free_(&desc) ;           \
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
    GrB_Matrix C = NULL, A = NULL ;
    GrB_Descriptor desc = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A C (make a deep copy)
    #define GET_DEEP_COPY       \
        C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;      \
        GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, C->sparsity_control) ; \
        A = GB_mx_mxArray_to_Matrix (pargin [1], "A input", true, true) ;      \
        GxB_Matrix_Option_set (A, GxB_SPARSITY_CONTROL, A->sparsity_control) ;
    #define FREE_DEEP_COPY      \
        GrB_Matrix_free_(&C) ;  \
        GrB_Matrix_free_(&A) ;
    GET_DEEP_COPY ;
    if (C == NULL || A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C or A failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (2), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    GrB_Index nrows, ncols ;
    GrB_Matrix_nrows (&nrows, C) ;
    GrB_Matrix_ncols (&ncols, C) ;

    // C<A> = A
    METHOD (GxB_Matrix_subassign_(C, A, NULL, A,
        GrB_ALL, nrows, GrB_ALL, ncols, desc)) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

