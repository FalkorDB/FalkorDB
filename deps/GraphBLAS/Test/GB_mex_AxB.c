//------------------------------------------------------------------------------
// GB_mex_AxB: compute C=A*B, A'*B, A*B', or A'*B'
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is for testing only.  See GrB_mxm instead.  Returns a plain built-in
// matrix, in double.

#include "GB_mex.h"

#define USAGE "C = GB_mex_AxB (A, B, atranspose, btranspose, axb_method)"

#define FREE_ALL                                \
{                                               \
    GrB_Matrix_free_(&A) ;                      \
    GrB_Matrix_free_(&Aconj) ;                  \
    GrB_Matrix_free_(&B) ;                      \
    GrB_Matrix_free_(&Bconj) ;                  \
    GrB_Matrix_free_(&C) ;                      \
    GrB_Matrix_free_(&Mask) ;                   \
    GrB_Monoid_free_(&add) ;                    \
    GrB_Semiring_free_(&semiring) ;             \
    GB_mx_put_global (true) ;                   \
}

//------------------------------------------------------------------------------

GrB_Info info ;
bool malloc_debug = false ;
bool ignore = false, ignore1 = false, ignore2 = false ;
bool atranspose = false ;
bool btranspose = false ;
GrB_Matrix A = NULL, B = NULL, C = NULL, Aconj = NULL, Bconj = NULL,
    Mask = NULL ;
GrB_Monoid add = NULL ;
GrB_Semiring semiring = NULL ;
int64_t anrows = 0 ;
int64_t ancols = 0 ;
int64_t bnrows = 0 ;
int64_t bncols = 0 ;
struct GB_Matrix_opaque C_header ;

GrB_Desc_Value AxB_method = GxB_DEFAULT ;

GrB_Info axb (GB_Werk Werk) ;
GrB_Info axb_complex (GB_Werk Werk) ;

//------------------------------------------------------------------------------

GrB_Info axb (GB_Werk Werk)
{

    // create the Semiring for regular z += x*y
    info = GrB_Monoid_new_FP64_(&add, GrB_PLUS_FP64, (double) 0) ;
    if (info != GrB_SUCCESS) return (info) ;

    info = GrB_Semiring_new (&semiring, add, GrB_TIMES_FP64) ;
    if (info != GrB_SUCCESS)
    {
        GrB_Monoid_free_(&add) ;
        return (info) ;
    }

    struct GB_Matrix_opaque MT_header ;
    GrB_Matrix MT = GB_clear_static_header (&MT_header) ;

    // C = A*B, A'*B, A*B', or A'*B'
    info = GB_AxB_meta (C, NULL,
        false,      // C_replace
        true,       // CSC
        MT,         // no MT returned
        &ignore1,   // M_transposed will be false
        NULL,       // no Mask
        false,      // mask not complemented
        false,      // mask not structural
        NULL,       // no accum
        A,
        B,
        semiring,   // GrB_PLUS_TIMES_FP64
        atranspose,
        btranspose,
        false,      // flipxy
        &ignore,    // mask_applied
        &ignore2,   // done_in_place
        AxB_method,
        true,       // do the sort
        Werk) ;

    GrB_Monoid_free_(&add) ;
    GrB_Semiring_free_(&semiring) ;

    return (info) ;
}

//------------------------------------------------------------------------------

GrB_Info axb_complex (GB_Werk Werk)
{

    // C = A*B, complex case

    Aconj = NULL ;
    Bconj = NULL ;

    if (atranspose)
    {
        // Aconj = A
        info = GrB_Matrix_new (&Aconj, Complex, A->vlen, A->vdim) ;
        if (info != GrB_SUCCESS) return (info) ;
        info = GrB_Matrix_apply_(Aconj, NULL, NULL, Complex_conj, A, NULL) ;
        if (info != GrB_SUCCESS)
        {
            GrB_Matrix_free_(&Aconj) ;
            return (info) ;
        }
    }

    if (btranspose)
    {
        // Bconj = B
        info = GrB_Matrix_new (&Bconj, Complex, B->vlen, B->vdim) ;
        if (info != GrB_SUCCESS)
        {
            GrB_Matrix_free_(&Aconj) ;
            return (info) ;
        }

        info = GrB_Matrix_apply_(Bconj, NULL, NULL, Complex_conj, B, NULL) ;
        if (info != GrB_SUCCESS)
        {
            GrB_Matrix_free_(&Bconj) ;
            GrB_Matrix_free_(&Aconj) ;
            return (info) ;
        }

    }

    // force completion
    if (Aconj != NULL)
    {
        info = GrB_Matrix_wait_(Aconj, GrB_MATERIALIZE) ;
        if (info != GrB_SUCCESS)
        {
            GrB_Matrix_free_(&Aconj) ;
            GrB_Matrix_free_(&Bconj) ;
            return (info) ;
        }
    }

    if (Bconj != NULL)
    {
        info = GrB_Matrix_wait_(Bconj, GrB_MATERIALIZE) ;
        if (info != GrB_SUCCESS)
        {
            GrB_Matrix_free_(&Aconj) ;
            GrB_Matrix_free_(&Bconj) ;
            return (info) ;
        }
    }

    struct GB_Matrix_opaque MT_header ;
    GrB_Matrix MT = GB_clear_static_header (&MT_header) ;

    info = GB_AxB_meta (C, NULL,
        false,      // C_replace
        true,       // CSC
        MT,         // no MT returned
        &ignore1,   // M_transposed will be false
        NULL,       // no Mask
        false,      // mask not complemented
        false,      // mask not structural
        NULL,       // no accum
        (atranspose) ? Aconj : A,
        (btranspose) ? Bconj : B,
        Complex_plus_times,
        atranspose,
        btranspose,
        false,      // flipxy
        &ignore,    // mask_applied
        &ignore2,   // done_in_place
        AxB_method,
        true,       // do the sort
        Werk) ;

    GrB_Matrix_free_(&Bconj) ;
    GrB_Matrix_free_(&Aconj) ;

    return (info) ;
}

//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    info = GrB_SUCCESS ;
    malloc_debug = GB_mx_get_global (true) ;
    ignore = false ;
    ignore1 = false ;
    ignore2 = false ;
    A = NULL ;
    B = NULL ;
    C = NULL ;
    Aconj = NULL ;
    Bconj = NULL ;
    Mask = NULL ;
    add = NULL ;
    semiring = NULL ;

    GB_WERK (USAGE) ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 5)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A and B
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A", false, true) ;
    B = GB_mx_mxArray_to_Matrix (pargin [1], "B", false, true) ;
    if (A == NULL || B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("failed") ;
    }

    if (!A->is_csc || !B->is_csc)
    {
        mexErrMsgTxt ("A and B must be in CSC format") ;
    }

    // get the atranspose option
    GET_SCALAR (2, bool, atranspose, false) ;

    // get the btranspose option
    GET_SCALAR (3, bool, btranspose, false) ;

    // get the axb_method
    // 0 or not present: default
    // 7081: Gustavson
    // 7083: dot
    // 7084: hash
    // 7085: saxpy
    GET_SCALAR (4, GrB_Desc_Value, AxB_method, GxB_DEFAULT) ;

    if (! ((AxB_method == GxB_DEFAULT) ||
        (AxB_method == GxB_AxB_GUSTAVSON) ||
        (AxB_method == GxB_AxB_HASH) ||
        (AxB_method == GxB_AxB_DOT)))
    {
        mexErrMsgTxt ("unknown method") ;
    }

    // determine the dimensions
    anrows = (atranspose) ? GB_NCOLS (A) : GB_NROWS (A) ;
    ancols = (atranspose) ? GB_NROWS (A) : GB_NCOLS (A) ;
    bnrows = (btranspose) ? GB_NCOLS (B) : GB_NROWS (B) ;
    bncols = (btranspose) ? GB_NROWS (B) : GB_NCOLS (B) ;
    if (ancols != bnrows)
    {
        FREE_ALL ;
        mexErrMsgTxt ("invalid dimensions") ;
    }

    C = GB_clear_static_header (&C_header) ;

    if (A->type == Complex)
    {
        METHOD (axb_complex (Werk)) ;
    }
    else
    {
        METHOD (axb (Werk)) ;
    }

    // return C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C AxB result", false) ;

    FREE_ALL ;
}

