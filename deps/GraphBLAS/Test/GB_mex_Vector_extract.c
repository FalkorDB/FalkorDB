//------------------------------------------------------------------------------
// GB_mex_Vector_extract: interface for w<mask> = accum (w,u(I))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "w = GB_mex_Vector_extract (w, mask, accum, u, I, desc, method)"

#define FREE_ALL                        \
{                                       \
    GrB_Vector_free_(&w) ;              \
    GrB_Vector_free_(&mask) ;           \
    GrB_Vector_free_(&u) ;              \
    GrB_Vector_free_(&I_vector) ;       \
    GrB_Descriptor_free_(&desc) ;       \
    GB_mx_put_global (true) ;           \
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
    GrB_Vector w = NULL, I_vector = NULL, mask = NULL, u = NULL ;
    GrB_Descriptor desc = NULL ;
    uint64_t *I = NULL, ni = 0, I_range [3] ;
    bool ignore ;

    // check inputs
    if (nargout > 1 || nargin < 5 || nargin > 7)
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
    u = GB_mx_mxArray_to_Vector (pargin [3], "A input", false, true) ;
    if (u == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("u failed") ;
    }

    // get accum, if present
    bool user_complex = (Complex != GxB_FC64)
        && (w->type == Complex || u->type == Complex) ;
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        w->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get the method:  0: use (uint64_t *), 1: use GrB_Vector for I,J
    int GET_SCALAR (6, int, method, 0) ;

    if (method == 0)
    {
        // get I
        if (!GB_mx_mxArray_to_indices (pargin [4], &I, &ni, I_range, &ignore,
            NULL))
        {
            FREE_ALL ;
            mexErrMsgTxt ("I failed") ;
        }
    }
    else
    {
        // get I_vector
        I_vector = GB_mx_mxArray_to_Vector (pargin [4], "I", false, false) ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (5), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // w<mask> = accum (w,u(I))
    if (method == 0)
    {
        METHOD (GrB_Vector_extract_(w, mask, accum, u, I, ni, desc)) ;
    }
    else
    {
        METHOD (GxB_Vector_extract_Vector_(w, mask, accum, u, I_vector, desc)) ;
    }

    // return w as a struct and free the GraphBLAS w
    pargout [0] = GB_mx_Vector_to_mxArray (&w, "w output", true) ;

    FREE_ALL ;
}

