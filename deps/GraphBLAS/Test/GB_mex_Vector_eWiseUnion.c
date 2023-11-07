//------------------------------------------------------------------------------
// GB_mex_Vector_eWiseUnion: w<mask> = accum(w,u+v)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "w = GB_mex_Vector_eWiseUnion (w, mask, accum, add, u, alpha, v, beta, desc)"

#define FREE_ALL                    \
{                                   \
    GrB_Vector_free_(&w) ;          \
    GrB_Vector_free_(&u) ;          \
    GrB_Scalar_free_(&alpha) ;      \
    GrB_Vector_free_(&v) ;          \
    GrB_Scalar_free_(&beta) ;       \
    GrB_Descriptor_free_(&desc) ;   \
    GrB_Vector_free_(&mask) ;       \
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

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Vector w = NULL ;
    GrB_Vector u = NULL ;
    GrB_Vector v = NULL ;
    GrB_Vector mask = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_Scalar alpha = NULL, beta = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 8 || nargin > 9)
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

    // get u (shallow copy)
    u = GB_mx_mxArray_to_Vector (pargin [4], "u input", false, true) ;
    if (u == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("u failed") ;
    }

    // get alpha
    alpha = GB_mx_get_Scalar (pargin [5]) ; 
    if (alpha == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("alpha failed") ;
    }

    // get v (shallow copy)
    v = GB_mx_mxArray_to_Vector (pargin [6], "v input", false, true) ;
    if (v == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("v failed") ;
    }

    // get beta
    beta = GB_mx_get_Scalar (pargin [7]) ; 
    if (beta == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("beta failed") ;
    }

    // get add operator
    bool user_complex = (Complex != GxB_FC64)
        && (u->type == Complex || v->type == Complex) ;
    GrB_BinaryOp add ;
    if (!GB_mx_mxArray_to_BinaryOp (&add, pargin [3], "add",
        w->type, user_complex) || add == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("add failed") ;
    }

    // get accum, if present
    user_complex = (Complex != GxB_FC64)
        && (w->type == Complex || add->xtype == Complex) ;
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        w->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (8), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // w<mask> = accum(w,u+v)
    METHOD (GxB_Vector_eWiseUnion (w, mask, accum, add, u, alpha,
        v, beta, desc)) ;

    // return w as a struct and free the GraphBLAS w
    pargout [0] = GB_mx_Vector_to_mxArray (&w, "w output", true) ;

    FREE_ALL ;
}

