//------------------------------------------------------------------------------
// GB_mex_mxv: w<mask> = accum(w,A*u)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "w = GB_mex_mxv (w, mask, accum, semiring, A, u, desc)"

#define FREE_ALL                                    \
{                                                   \
    GrB_Vector_free_(&w) ;                          \
    GrB_Vector_free_(&u) ;                          \
    GrB_Matrix_free_(&A) ;                          \
    GrB_Vector_free_(&mask) ;                       \
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
    GrB_Vector w = NULL ;
    GrB_Vector u = NULL ;
    GrB_Matrix A = NULL ;
    GrB_Vector mask = NULL ;
    GrB_Semiring semiring = NULL ;
    GrB_Descriptor desc = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 6 || nargin > 7)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get w (make a deep copy)
    #define GET_DEEP_COPY \
    w = GB_mx_mxArray_to_Vector (pargin [0], "w input", true, true) ;
    #define FREE_DEEP_COPY GrB_Vector_free_(&w) ;
    GET_DEEP_COPY ;
    if (w == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("w failed") ;
    }

    // get mask (shallow copy)
    mask = GB_mx_mxArray_to_Vector (pargin [1], "mask", false, false) ;
    if (mask == NULL && !mxIsEmpty (pargin [1]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("mask failed") ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [4], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get u (shallow copy)
    u = GB_mx_mxArray_to_Vector (pargin [5], "A input", false, true) ;
    if (u == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("u failed") ;
    }

    bool user_complex = (Complex != GxB_FC64) && (w->type == Complex) ;

    // get semiring
    if (!GB_mx_mxArray_to_Semiring (&semiring, pargin [3], "semiring",
        w->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("semiring failed") ;
    }

    // get accum, if present
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        w->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (6), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // w<mask> = accum(w,A*u)
    METHOD (GrB_mxv (w, mask, accum, semiring, A, u, desc)) ;

    // return w as a struct and free the GraphBLAS w
    pargout [0] = GB_mx_Vector_to_mxArray (&w, "w output", true) ;

    FREE_ALL ;
}

