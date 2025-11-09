//------------------------------------------------------------------------------
// GB_mex_Matrix_sort: [C,P] = sort (A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE \
    "[C,P] = GB_mex_Matrix_sort (op, A, desc, arg1, ptype)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&P) ;              \
    GrB_Matrix_free_(&C) ;              \
    GrB_free (&lt) ;                    \
    GrB_Descriptor_free_(&desc) ;       \
    GB_mx_put_global (true) ;           \
}

void my_lt_double (bool *z, double *x, double *y) ;
void my_lt_double (bool *z, double *x, double *y) { (*z) = (*x) < (*y) ; }

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL, C = NULL, P = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_BinaryOp lt = NULL ;

    // check inputs
    if (nargout > 2 || nargin < 2 || nargin > 5)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [1], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get operator
    bool user_complex = (Complex != GxB_FC64) && (A->type == Complex) ;
    GrB_BinaryOp op ;
    if (!GB_mx_mxArray_to_BinaryOp (&op, pargin [0], "op",
        A->type, user_complex) || op == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("add failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (2), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // get arg1
    int GET_SCALAR (3, int, arg1, false) ;
    if (arg1 < 0 && op == GrB_LT_FP64)
    { 
        // use a user-defined "<" op instead of GrB_LT_FP64
        GrB_BinaryOp_new (&lt,
            (GxB_binary_function)
            my_lt_double, GrB_BOOL, GrB_FP64, GrB_FP64) ;
        op = lt ;
    }

    // get ptype; defaults to GrB_INT64
    GrB_Type ptype = GB_mx_string_to_Type (PARGIN (4), GrB_INT64) ;

    // create C and P
    uint64_t nrows, ncols ;
    GrB_Matrix_nrows (&nrows, A) ;
    GrB_Matrix_ncols (&ncols, A) ;

    // P = GB_mex_Matrix_sort (op, A, desc, 1): do not compute C
    bool P_only = (arg1 >= 1 && nargout == 1) ;

    if (nargout == 1)
    {
        if (P_only)
        {
            // P = sort (op, A, desc), do not return C
            #undef  FREE_DEEP_COPY
            #define FREE_DEEP_COPY GrB_Matrix_free (&P) ;
            #undef  GET_DEEP_COPY
            #define GET_DEEP_COPY  GrB_Matrix_new (&P, ptype, nrows, ncols) ;
            GET_DEEP_COPY ;
            METHOD (GxB_Matrix_sort (NULL, P, op, A, desc)) ;
        }
        else
        {
            // C = sort (op, C, desc), in place
            #undef  FREE_DEEP_COPY
            #define FREE_DEEP_COPY GrB_Matrix_free (&C) ;
            #undef  GET_DEEP_COPY
            #define GET_DEEP_COPY  GrB_Matrix_dup (&C, A) ;
            GET_DEEP_COPY ;
            METHOD (GxB_Matrix_sort (C, NULL, op, C, desc)) ;
        }
    }
    else
    {
        // [C,P] = sort (op, A, desc)
        #undef  FREE_DEEP_COPY
        #define FREE_DEEP_COPY  \
                GrB_Matrix_free (&C) ;  \
                GrB_Matrix_free (&P) ;
        #undef  GET_DEEP_COPY
        #define GET_DEEP_COPY   \
            GrB_Matrix_new (&C, A->type, nrows, ncols) ;    \
            GrB_Matrix_new (&P, ptype, nrows, ncols) ;
        GET_DEEP_COPY ;
        METHOD (GxB_Matrix_sort (C, P, op, A, desc)) ;
    }

    if (P_only)
    {
        // return P as a struct and free the GraphBLAS P
        pargout [0] = GB_mx_Matrix_to_mxArray (&P, "P output", true) ;
    }
    else
    {
        // return C as a struct and free the GraphBLAS C
        pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;
        if (nargout > 1)
        {
            // return P as a struct and free the GraphBLAS P
            pargout [1] = GB_mx_Matrix_to_mxArray (&P, "P output", true) ;
        }
    }

    FREE_ALL ;
}

