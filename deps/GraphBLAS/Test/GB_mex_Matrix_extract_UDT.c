//------------------------------------------------------------------------------
// GB_mex_Matrix_extract_UDT: interface for C<Mask> = accum (C,A(I,J))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "C = GB_mex_Matrix_extract (C, Mask, accum, A, I, J, desc, use_mydouble)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&C) ;              \
    GrB_Matrix_free_(&C2) ;             \
    GrB_Matrix_free_(&Mask) ;           \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&A2) ;             \
    GrB_free (&mydouble) ;              \
    GrB_free (&castdouble) ;            \
    GrB_free (&castback) ;              \
    GrB_Descriptor_free_(&desc) ;       \
    GB_mx_put_global (true) ;           \
}

typedef double my_double ;
void cast_double (my_double *z, double *x) ;
void cast_double (my_double *z, double *x) { *z = (my_double) *x ; }
void cast_back   (double *z, my_double *x) ;
void cast_back   (double *z, my_double *x) { *z = (double) *x ; }

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
    GrB_Matrix C = NULL ;
    GrB_Matrix Mask = NULL ;
    GrB_Matrix A = NULL ;
    GrB_Descriptor desc = NULL ;
    uint64_t *I = NULL, ni = 0, I_range [3] ;
    uint64_t *J = NULL, nj = 0, J_range [3] ;
    bool ignore ;
    uint64_t m, n ;

    GrB_Type mydouble = NULL ;
    GrB_UnaryOp castdouble = NULL, castback = NULL ;
    GrB_Matrix A2 = NULL, C2 = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 6 || nargin > 8)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get C (make a deep copy)
    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }

    // get Mask (shallow copy)
    Mask = GB_mx_mxArray_to_Matrix (pargin [1], "Mask", false, false) ;
    if (Mask == NULL && !mxIsEmpty (pargin [1]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("Mask failed") ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [3], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get accum, if present
    bool user_complex = (Complex != GxB_FC64)
        && (C->type == Complex || A->type == Complex) ;
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        C->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get I
    if (!GB_mx_mxArray_to_indices (pargin [4], &I, &ni, I_range, &ignore, NULL))
    {
        FREE_ALL ;
        mexErrMsgTxt ("I failed") ;
    }

    // get J
    if (!GB_mx_mxArray_to_indices (pargin [5], &J, &nj, J_range, &ignore, NULL))
    {
        FREE_ALL ;
        mexErrMsgTxt ("J failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (6), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // get the use_mydouble option
    bool GET_SCALAR (7, bool, use_mydouble, false) ;

    if (use_mydouble)
    { 
        // A = (mydouble) A, if A is double
        if (A->type == GrB_FP64 && C->type == GrB_FP64)
        {
//          printf ("use_mydouble A\n") ;
            OK (GrB_Type_new (&mydouble, sizeof (double))) ;
            OK (GrB_UnaryOp_new (&castdouble, 
                (GxB_unary_function) cast_double, mydouble, GrB_FP64)) ;
            OK (GrB_UnaryOp_new (&castback,   
                (GxB_unary_function) cast_back,   GrB_FP64, mydouble)) ;
            OK (GrB_Matrix_nrows (&m, A)) ;
            OK (GrB_Matrix_ncols (&n, A)) ;
//          OK (GxB_print (A, 2)) ;
            OK (GrB_Matrix_new (&A2, mydouble, m, n)) ;
            OK (GrB_apply (A2, NULL, NULL, castdouble, A, NULL)) ;
            OK (GrB_free (&A)) ;
            A = A2 ;
            A2 = NULL ;

//          OK (GxB_print (C, 2)) ;
            OK (GrB_Matrix_nrows (&m, C)) ;
            OK (GrB_Matrix_ncols (&n, C)) ;
            OK (GrB_Matrix_new (&C2, mydouble, m, n)) ;
            OK (GrB_apply (C2, NULL, NULL, castdouble, C, NULL)) ;
            OK (GrB_free (&C)) ;
            C = C2 ;
            C2 = NULL ;
        }
    }

    // C<Mask> = accum (C,A(I,J))
//  printf ("do exttract:\n") ;
//  OK (GxB_print (Mask, 2)) ;
//  OK (GxB_print (C, 2)) ;
//  OK (GxB_print (A, 2)) ;
    OK (GrB_Matrix_extract_(C, Mask, accum, A, I, ni, J, nj, desc)) ;

    if (C->type == mydouble)
    {
        OK (GrB_Matrix_nrows (&m, C)) ;
        OK (GrB_Matrix_ncols (&n, C)) ;
        OK (GrB_Matrix_new (&C2, GrB_FP64, m, n)) ;
        OK (GrB_apply (C2, NULL, NULL, castback, C, NULL)) ;
        OK (GrB_free (&C)) ;
        C = C2 ;
        C2 = NULL ;
    }

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

