//------------------------------------------------------------------------------
// GB_mex_triple_mxm: C = A*B*E
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_triple_mxm (semiring, A, B, E, desc)"

#define FREE_ALL                                    \
{                                                   \
    GrB_Matrix_free_(&A) ;                          \
    GrB_Matrix_free_(&B) ;                          \
    GrB_Matrix_free_(&C) ;                          \
    GrB_Matrix_free_(&E) ;                          \
    GrB_Matrix_free_(&T) ;                          \
    if (semiring != Complex_plus_times)             \
    {                                               \
        if (semiring != NULL)                       \
        {                                           \
            GrB_Monoid_free_(&(semiring->add)) ;    \
        }                                           \
        GrB_Semiring_free_(&semiring) ;             \
    }                                               \
    GrB_Descriptor_free_(&desc) ;                   \
    GB_mx_put_global (true) ;                       \
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
    GrB_Matrix C = NULL ;
    GrB_Matrix E = NULL ;
    GrB_Matrix T = NULL ;
    GrB_Semiring semiring = NULL ;
    GrB_Descriptor desc = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 4 || nargin > 5)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [1], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get B (shallow copy)
    B = GB_mx_mxArray_to_Matrix (pargin [2], "B input", false, true) ;
    if (B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("B failed") ;
    }

    // get E (shallow copy)
    E = GB_mx_mxArray_to_Matrix (pargin [3], "E input", false, true) ;
    if (E == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("B failed") ;
    }

    bool user_complex = (Complex != GxB_FC64) && (A->type == Complex) ;

    // get semiring
    if (!GB_mx_mxArray_to_Semiring (&semiring, pargin [0], "semiring",  
        A->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("semiring failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (4), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // T = B*E
    GrB_Index nrows, ncols ;
    GrB_Matrix_nrows (&nrows, B) ;
    GrB_Matrix_ncols (&ncols, E) ;
    GrB_Matrix_new (&T, A->type, nrows, ncols) ;
    GxB_Matrix_Option_set_(T, GxB_SPARSITY_CONTROL, GxB_SPARSE) ;
    GrB_mxm (T, NULL, NULL, semiring, B, E, desc) ;

    // C = A*T
    GrB_Matrix_ncols (&nrows, A) ;
    GrB_Matrix_new (&C, A->type, nrows, ncols) ;
    GrB_mxm (C, NULL, NULL, semiring, A, T, desc) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C from GrB_mxm_triple", true) ;

    FREE_ALL ;
}

